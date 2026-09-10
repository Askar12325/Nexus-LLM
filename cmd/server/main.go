package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	"nexusllm/pkg/cache"
	"nexusllm/pkg/metrics"
	"nexusllm/pkg/proxy"
	"nexusllm/pkg/ratelimit"
	"nexusllm/pkg/router"
)

type Server struct {
	router    *router.Router
	cache     *cache.Cache
	limiter   *ratelimit.Limiter
	telemetry *metrics.Telemetry
	mu        sync.Mutex
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8082"
	}

	r := router.NewRouter()
	c := cache.NewCache(15 * time.Minute)
	l := ratelimit.NewLimiter()
	t := metrics.NewTelemetry()

	demoTenant := ratelimit.NewTenant(
		"tenant-default",
		"Production Workspace",
		"nx-key-demo-secret",
		ratelimit.TierEnterprise,
		ratelimit.TenantConfig{
			MaxRPS:           25.0,
			MaxTPM:           100000,
			MonthlyBudgetUSD: 1000.0,
		},
	)
	l.RegisterTenant(demoTenant)

	server := &Server{
		router:    r,
		cache:     c,
		limiter:   l,
		telemetry: t,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", server.handleDashboardUI)
	mux.HandleFunc("GET /health", server.handleHealth)
	mux.HandleFunc("POST /v1/chat/completions", server.handleOpenAICompletion)
	mux.HandleFunc("POST /api/v1/generate", server.handleGenerate)
	mux.HandleFunc("GET /api/v1/metrics", server.handleGetMetrics)
	mux.HandleFunc("GET /api/v1/circuit-breakers", server.handleGetCircuitBreakers)
	mux.HandleFunc("POST /api/v1/chaos", server.handleChaosToggle)
	mux.HandleFunc("POST /api/v1/cache/clear", server.handleClearCache)
	mux.HandleFunc("GET /api/v1/tenants", server.handleGetTenants)

	httpServer := &http.Server{
		Addr:    ":" + port,
		Handler: mux,
	}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down NexusLLM Gateway...")
		shutdownCtx, shutdownCancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer shutdownCancel()
		_ = httpServer.Shutdown(shutdownCtx)
	}()

	log.Printf("NexusLLM Gateway running on http://localhost:%s", port)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "UP",
		"service": "NexusLLM Enterprise AI Gateway",
		"time":    time.Now().UTC().Format(time.RFC3339),
	})
}

func (s *Server) handleOpenAICompletion(w http.ResponseWriter, r *http.Request) {
	apiKey := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if apiKey == "" {
		apiKey = "nx-key-demo-secret"
	}

	var req struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		Stream      bool    `json:"stream"`
		Temperature float64 `json:"temperature"`
		MaxTokens   int     `json:"max_tokens"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		req.Model = "gpt-4o"
	}

	var fullPrompt strings.Builder
	for _, m := range req.Messages {
		fullPrompt.WriteString(m.Content + " ")
	}
	prompt := strings.TrimSpace(fullPrompt.String())
	if prompt == "" {
		prompt = "Hello"
	}

	estTokens := int64(len(strings.Fields(prompt)) + 50)
	tenant, err := s.limiter.CheckAndConsume(apiKey, estTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("Rate limit violation: %v", err), http.StatusTooManyRequests)
		return
	}

	sanReport := proxy.SanitizePrompt(prompt)

	// Cache lookup
	if cachedItem, found := s.cache.Get(req.Model, sanReport.CleanedText); found {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID:            fmt.Sprintf("req-%d", time.Now().UnixNano()%1000000),
			Timestamp:     time.Now(),
			TenantID:      tenant.ID,
			Model:         req.Model,
			Provider:      "In-Memory Cache",
			PromptSummary: truncate(prompt, 60),
			TotalTokens:   cachedItem.PromptTokens + cachedItem.CompTokens,
			LatencyMs:     1,
			CostUSD:       0.0,
			CostSavedUSD:  cachedItem.CostSaved,
			FromCache:     true,
			Status:        "CACHED",
		})

		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Nexus-Cache", "HIT")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"id":      fmt.Sprintf("chatcmpl-cached-%d", time.Now().UnixNano()%100000),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   req.Model,
			"choices": []map[string]any{
				{
					"index": 0,
					"message": map[string]string{
						"role":    "assistant",
						"content": cachedItem.Response,
					},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     cachedItem.PromptTokens,
				"completion_tokens": cachedItem.CompTokens,
				"total_tokens":      cachedItem.PromptTokens + cachedItem.CompTokens,
			},
		})
		return
	}

	resp, err := s.router.RouteAndExecute(r.Context(), req.Model, sanReport.CleanedText)
	if err != nil {
		http.Error(w, fmt.Sprintf("Gateway routing error: %v", err), http.StatusBadGateway)
		return
	}

	s.cache.Set(req.Model, sanReport.CleanedText, resp.Text, resp.PromptTokens, resp.CompTokens, resp.EstimatedCost)
	s.limiter.RecordUsage(apiKey, resp.PromptTokens, resp.CompTokens, resp.EstimatedCost)

	status := "SUCCESS"
	if resp.WasFallback {
		status = "FALLBACK"
	}

	s.telemetry.RecordRequest(&metrics.RequestLog{
		ID:            resp.ID,
		Timestamp:     time.Now(),
		TenantID:      tenant.ID,
		Model:         resp.Model,
		Provider:      string(resp.Provider),
		PromptSummary: truncate(prompt, 60),
		TotalTokens:   resp.TotalTokens,
		LatencyMs:     resp.LatencyMs,
		CostUSD:       resp.EstimatedCost,
		CostSavedUSD:  0.0,
		FromCache:     false,
		WasFallback:   resp.WasFallback,
		Status:        status,
	})

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Nexus-Cache", "MISS")
	w.Header().Set("X-Nexus-Provider", string(resp.Provider))
	_ = json.NewEncoder(w).Encode(map[string]any{
		"id":      resp.ID,
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   resp.Model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": resp.Text,
				},
				"finish_reason": "stop",
			},
		},
		"usage": map[string]int{
			"prompt_tokens":     resp.PromptTokens,
			"completion_tokens": resp.CompTokens,
			"total_tokens":      resp.TotalTokens,
		},
	})
}

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		req.Model = "gpt-4o"
	}
	if req.Prompt == "" {
		req.Prompt = "Hello"
	}

	apiKey := "nx-key-demo-secret"
	estTokens := int64(len(strings.Fields(req.Prompt)) + 30)
	tenant, err := s.limiter.CheckAndConsume(apiKey, estTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("Rate limit violation: %v", err), http.StatusTooManyRequests)
		return
	}

	sanReport := proxy.SanitizePrompt(req.Prompt)

	// 1. Cache Check
	if cachedItem, found := s.cache.Get(req.Model, sanReport.CleanedText); found {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID:            fmt.Sprintf("req-cache-%d", time.Now().UnixNano()%100000),
			Timestamp:     time.Now(),
			TenantID:      tenant.ID,
			Model:         req.Model,
			Provider:      "Cache",
			PromptSummary: truncate(req.Prompt, 60),
			TotalTokens:   cachedItem.PromptTokens + cachedItem.CompTokens,
			LatencyMs:     1,
			CostUSD:       0.0,
			CostSavedUSD:  cachedItem.CostSaved,
			FromCache:     true,
			Status:        "CACHED",
		})

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"response": &router.CompletionResponse{
				ID:            fmt.Sprintf("chatcmpl-cached-%d", time.Now().UnixNano()%100000),
				Model:         req.Model,
				Provider:      "Cache",
				Text:          cachedItem.Response,
				PromptTokens:  cachedItem.PromptTokens,
				CompTokens:    cachedItem.CompTokens,
				TotalTokens:   cachedItem.PromptTokens + cachedItem.CompTokens,
				LatencyMs:     1,
				EstimatedCost: 0.0,
				FromCache:     true,
				WasFallback:   false,
				OriginalModel: req.Model,
			},
			"sanitization": sanReport,
			"cost_saved":   cachedItem.CostSaved,
		})
		return
	}

	// 2. Upstream Router Execution with Fallback Cascade
	resp, err := s.router.RouteAndExecute(r.Context(), req.Model, sanReport.CleanedText)
	if err != nil {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID:            fmt.Sprintf("req-err-%d", time.Now().UnixNano()%100000),
			Timestamp:     time.Now(),
			TenantID:      tenant.ID,
			Model:         req.Model,
			Provider:      "All Providers Failed",
			PromptSummary: truncate(req.Prompt, 60),
			LatencyMs:     120,
			Status:        "ERROR",
		})
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	s.cache.Set(req.Model, sanReport.CleanedText, resp.Text, resp.PromptTokens, resp.CompTokens, resp.EstimatedCost)
	s.limiter.RecordUsage(apiKey, resp.PromptTokens, resp.CompTokens, resp.EstimatedCost)

	status := "SUCCESS"
	if resp.WasFallback {
		status = "FALLBACK"
	}

	s.telemetry.RecordRequest(&metrics.RequestLog{
		ID:            resp.ID,
		Timestamp:     time.Now(),
		TenantID:      tenant.ID,
		Model:         resp.Model,
		Provider:      string(resp.Provider),
		PromptSummary: truncate(req.Prompt, 60),
		TotalTokens:   resp.TotalTokens,
		LatencyMs:     resp.LatencyMs,
		CostUSD:       resp.EstimatedCost,
		CostSavedUSD:  0.0,
		FromCache:     false,
		WasFallback:   resp.WasFallback,
		Status:        status,
	})

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"response":     resp,
		"sanitization": sanReport,
		"cost_saved":   0.0,
	})
}

func (s *Server) handleGetMetrics(w http.ResponseWriter, r *http.Request) {
	snap := s.telemetry.GetSnapshot()
	hits, misses, cacheSize, ratio := s.cache.Stats()

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"telemetry":  snap,
		"cache_size": cacheSize,
		"cache_hits": hits,
		"cache_miss": misses,
		"hit_ratio":  ratio,
	})
}

func (s *Server) handleGetCircuitBreakers(w http.ResponseWriter, r *http.Request) {
	providers := s.router.GetProviders()
	type BreakerStatus struct {
		Name             string `json:"name"`
		State            string `json:"state"`
		ConsecutiveFails int    `json:"consecutive_fails"`
		TotalTrips       int64  `json:"total_trips"`
		HasSimulatedErr  bool   `json:"has_simulated_error"`
	}

	var list []BreakerStatus
	for _, p := range providers {
		name, state, fails, trips := p.CircuitBreaker().Status()
		list = append(list, BreakerStatus{
			Name:             name,
			State:            string(state),
			ConsecutiveFails: fails,
			TotalTrips:       trips,
			HasSimulatedErr:  s.router.HasSimulatedError(p.Name()),
		})
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(list)
}

func (s *Server) handleChaosToggle(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Provider string `json:"provider"`
		Enabled  bool   `json:"enabled"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request", http.StatusBadRequest)
		return
	}

	s.router.SetSimulatedError(router.ProviderType(req.Provider), req.Enabled)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"provider": req.Provider,
		"enabled":  req.Enabled,
	})
}

func (s *Server) handleClearCache(w http.ResponseWriter, r *http.Request) {
	s.cache.Clear()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"message": "Cache successfully cleared"})
}

func (s *Server) handleGetTenants(w http.ResponseWriter, r *http.Request) {
	tenants := s.limiter.GetAllTenants()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(tenants)
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func (s *Server) handleDashboardUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	html := `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Nexus — Enterprise AI Gateway</title>
    <!-- Clean Minimalist Inter Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #0d0f12;
            --surface: #14171d;
            --surface-card: #1a1e26;
            --surface-hover: #222733;
            --border: rgba(255, 255, 255, 0.08);
            --border-hover: rgba(255, 255, 255, 0.16);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --green: #10b981;
            --red: #f43f5e;
            --amber: #f59e0b;
            --radius-sm: 6px;
            --radius-md: 10px;
            --radius-lg: 14px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
            -webkit-font-smoothing: antialiased;
        }

        /* Top Minimal Header */
        .navbar {
            height: 56px;
            background: var(--surface);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 10px;
            text-decoration: none;
            color: var(--text-primary);
        }
        .logo-symbol {
            width: 22px;
            height: 22px;
            background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .logo-symbol svg { width: 14px; height: 14px; fill: #0d0f12; }
        .logo-text { font-size: 15px; font-weight: 700; letter-spacing: -0.3px; }
        .logo-sub { font-size: 11px; color: var(--text-muted); font-weight: 500; margin-left: 4px; }

        .nav-center {
            display: flex;
            align-items: center;
            gap: 4px;
            background: var(--bg);
            padding: 3px;
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
        }
        .nav-tab-btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 12px;
            font-weight: 600;
            padding: 5px 14px;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.15s ease;
        }
        .nav-tab-btn.active {
            background: var(--surface-card);
            color: var(--text-primary);
        }

        .nav-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-secondary);
        }
        .status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green);
        }

        /* Main Workspace Container */
        .workspace-container {
            flex: 1;
            display: flex;
            max-width: 1280px;
            width: 100%;
            margin: 0 auto;
            padding: 24px;
            gap: 24px;
        }

        /* Left Side Metrics / Provider Status Panel */
        .sidebar {
            width: 320px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            flex-shrink: 0;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .metric-row {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
        }
        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 20px;
            font-weight: 600;
            color: var(--text-primary);
        }
        .metric-label { font-size: 12px; color: var(--text-secondary); }

        /* Downstream Providers List */
        .provider-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 10px;
            background: var(--surface-card);
            border-radius: var(--radius-sm);
            border: 1px solid var(--border);
            font-size: 13px;
        }
        .provider-name { font-weight: 600; color: var(--text-primary); }
        .provider-badge {
            font-size: 11px;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }
        .badge-healthy { background: rgba(16, 185, 129, 0.12); color: var(--green); }
        .badge-offline { background: rgba(244, 63, 94, 0.12); color: var(--red); }

        /* Chat / Copilot Workspace */
        .chat-workspace {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            overflow: hidden;
            min-height: 600px;
        }

        .chat-header {
            padding: 14px 20px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--surface);
        }
        .model-select-wrapper {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .model-select {
            background: var(--surface-card);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 6px 12px;
            border-radius: var(--radius-sm);
            font-size: 13px;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }
        .model-select:focus { border-color: var(--accent); }

        /* Message Thread */
        .messages-thread {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        .message-bubble {
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-width: 90%;
            animation: fadeIn 0.2s ease;
        }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

        .message-bubble.user {
            align-self: flex-end;
        }
        .message-bubble.assistant {
            align-self: flex-start;
            max-width: 95%;
        }

        .bubble-header {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .bubble-content {
            padding: 14px 18px;
            border-radius: var(--radius-md);
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-primary);
            white-space: pre-wrap;
        }
        .message-bubble.user .bubble-content {
            background: #2563eb;
            color: #ffffff;
            border-radius: 12px 12px 2px 12px;
        }
        .message-bubble.assistant .bubble-content {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px 12px 12px 2px;
        }

        .bubble-footer {
            display: flex;
            gap: 12px;
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 4px;
            font-family: 'JetBrains Mono', monospace;
        }
        .tag-cache { color: var(--amber); font-weight: 600; }
        .tag-fallback { color: var(--accent); font-weight: 600; }

        /* Quick Prompts Bar */
        .quick-prompts {
            display: flex;
            gap: 8px;
            padding: 8px 20px;
            overflow-x: auto;
            border-top: 1px solid var(--border);
            background: var(--surface);
        }
        .quick-btn {
            background: var(--surface-card);
            border: 1px solid var(--border);
            color: var(--text-secondary);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: var(--radius-sm);
            cursor: pointer;
            white-space: nowrap;
            transition: all 0.15s ease;
        }
        .quick-btn:hover {
            color: var(--text-primary);
            border-color: var(--border-hover);
            background: var(--surface-hover);
        }

        /* Input Dock */
        .input-dock {
            padding: 16px 20px;
            background: var(--surface);
            border-top: 1px solid var(--border);
            display: flex;
            gap: 12px;
            align-items: flex-end;
        }

        .prompt-textarea {
            flex: 1;
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 12px 16px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 14px;
            resize: none;
            min-height: 48px;
            max-height: 140px;
            outline: none;
            line-height: 1.4;
            transition: border-color 0.15s ease;
        }
        .prompt-textarea:focus { border-color: var(--accent); }

        .btn-send {
            height: 48px;
            padding: 0 20px;
            background: #ffffff;
            color: #0d0f12;
            border: none;
            border-radius: var(--radius-md);
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
            transition: opacity 0.15s ease;
            flex-shrink: 0;
        }
        .btn-send:hover { opacity: 0.9; }
        .btn-send:disabled { opacity: 0.4; cursor: not-allowed; }

        /* Tables and Secondary Views */
        .view-panel {
            flex: 1;
            display: flex;
            flex-direction: column;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: 24px;
            gap: 16px;
        }
        .table-clean {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }
        .table-clean th {
            text-align: left;
            padding: 10px 14px;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            font-size: 12px;
        }
        .table-clean td {
            padding: 12px 14px;
            border-bottom: 1px solid var(--border);
            color: var(--text-secondary);
        }

        .btn-sm {
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--surface-card);
            color: var(--text-primary);
            cursor: pointer;
        }
        .btn-sm:hover { background: var(--surface-hover); }

        @media (max-width: 900px) {
            .workspace-container { flex-direction: column; padding: 14px; }
            .sidebar { width: 100%; }
        }
    </style>
</head>
<body>
    <!-- Top Header -->
    <header class="navbar">
        <div class="nav-brand">
            <div class="logo-symbol">
                <svg viewBox="0 0 24 24"><path d="M12 2L2 19.5h20L12 2zm0 4l6.5 11.5h-13L12 6z"/></svg>
            </div>
            <div class="logo-text">Nexus<span class="logo-sub">Gateway</span></div>
        </div>

        <nav class="nav-center">
            <button class="nav-tab-btn active" id="tab-btn-chat" onclick="switchView('chat')">Playground</button>
            <button class="nav-tab-btn" id="tab-btn-resilience" onclick="switchView('resilience')">Resilience</button>
            <button class="nav-tab-btn" id="tab-btn-telemetry" onclick="switchView('telemetry')">Telemetry</button>
            <button class="nav-tab-btn" id="tab-btn-logs" onclick="switchView('logs')">Logs</button>
        </nav>

        <div class="nav-status">
            <div class="status-dot"></div>
            <span>Ready</span>
        </div>
    </header>

    <!-- Main Workspace -->
    <main class="workspace-container">
        <!-- Sidebar: Health & Telemetry Summary -->
        <aside class="sidebar">
            <div class="card">
                <div class="card-header">
                    <span>Performance</span>
                    <span style="color:var(--green);">Live</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">P95 Latency</span>
                    <span class="metric-value" id="side-p95">1ms</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Cache Hit Rate</span>
                    <span class="metric-value" id="side-cache">0.0%</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Cost Saved</span>
                    <span class="metric-value" style="color:var(--amber);" id="side-saved">$0.00</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <span>Provider Mesh</span>
                    <button class="btn-sm" onclick="refreshCircuitBreakers()">Refresh</button>
                </div>
                <div id="providers-list" style="display:flex; flex-direction:column; gap:8px;">
                    <!-- Rendered dynamically -->
                </div>
            </div>

            <div class="card" style="font-size:12px; color:var(--text-muted); line-height:1.5;">
                <span style="font-weight:600; color:var(--text-secondary); margin-bottom:4px; display:block;">About Fallbacks</span>
                If the primary model experiences a timeout or 429 rate limit, Nexus instantly cascades to backup providers with zero client errors.
            </div>
        </aside>

        <!-- VIEW 1: Clean Chat Playground -->
        <section class="chat-workspace" id="view-chat">
            <div class="chat-header">
                <div class="model-select-wrapper">
                    <span style="font-size:12px; font-weight:600; color:var(--text-muted);">Model:</span>
                    <select class="model-select" id="model-select">
                        <option value="gpt-4o">OpenAI • GPT-4o</option>
                        <option value="claude-3-5-sonnet">Anthropic • Claude 3.5 Sonnet</option>
                        <option value="gemini-1.5-pro">Google • Gemini 1.5 Pro</option>
                        <option value="deepseek-chat">DeepSeek • DeepSeek V3</option>
                    </select>
                </div>
                <div style="font-size:12px; color:var(--text-muted);">
                    PII Redaction: <strong style="color:var(--green);">Active</strong>
                </div>
            </div>

            <div class="messages-thread" id="messages-thread">
                <div class="message-bubble assistant">
                    <div class="bubble-header">
                        <span>Nexus Assistant</span>
                        <span>•</span>
                        <span>Ready</span>
                    </div>
                    <div class="bubble-content">Hello! Type any prompt below to test real-time AI responses, zero-downtime fallback routing, or instant prompt caching.</div>
                </div>
            </div>

            <div class="quick-prompts">
                <button class="quick-btn" onclick="setQuickPrompt('What is the capital of France?')">What is the capital of France?</button>
                <button class="quick-btn" onclick="setQuickPrompt('Draft a professional email requesting project updates')">Draft project email</button>
                <button class="quick-btn" onclick="setQuickPrompt('Explain how distributed consensus works in simple terms')">Explain consensus</button>
                <button class="quick-btn" onclick="setQuickPrompt('Calculate 450 * 1.15')">Math test</button>
            </div>

            <div class="input-dock">
                <textarea class="prompt-textarea" id="prompt-input" placeholder="Ask a question or enter a task..." rows="1"></textarea>
                <button class="btn-send" id="btn-send" onclick="sendPrompt()">Send</button>
            </div>
        </section>

        <!-- VIEW 2: Resilience & Chaos Switcher -->
        <section class="view-panel" id="view-resilience" style="display:none;">
            <div>
                <h2 style="font-size:16px; font-weight:700; margin-bottom:4px;">Resilience & Fault Injection</h2>
                <p style="font-size:13px; color:var(--text-muted);">Simulate provider outages to verify automatic fallback routing.</p>
            </div>
            <table class="table-clean">
                <thead>
                    <tr>
                        <th>Provider</th>
                        <th>Circuit State</th>
                        <th>Failures</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody id="resilience-table-body"></tbody>
            </table>
        </section>

        <!-- VIEW 3: Telemetry View -->
        <section class="view-panel" id="view-telemetry" style="display:none;">
            <div>
                <h2 style="font-size:16px; font-weight:700; margin-bottom:4px;">Telemetry & Latency Histograms</h2>
                <p style="font-size:13px; color:var(--text-muted);">Prometheus-compatible real-time performance telemetry.</p>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:16px; margin-top:12px;">
                <div class="card">
                    <span class="metric-label">P50 Latency</span>
                    <span class="metric-value" id="telemetry-p50">1 ms</span>
                </div>
                <div class="card">
                    <span class="metric-label">P99 Latency</span>
                    <span class="metric-value" id="telemetry-p99">2 ms</span>
                </div>
                <div class="card">
                    <span class="metric-label">Total Requests</span>
                    <span class="metric-value" id="telemetry-requests">0</span>
                </div>
                <div class="card">
                    <span class="metric-label">Total Tokens</span>
                    <span class="metric-value" id="telemetry-tokens">0</span>
                </div>
            </div>
        </section>

        <!-- VIEW 4: Logs View -->
        <section class="view-panel" id="view-logs" style="display:none;">
            <div>
                <h2 style="font-size:16px; font-weight:700; margin-bottom:4px;">Request Logs</h2>
                <p style="font-size:13px; color:var(--text-muted);">Real-time inspection of incoming requests.</p>
            </div>
            <table class="table-clean">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Model</th>
                        <th>Summary</th>
                        <th>Tokens</th>
                        <th>Latency</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody id="logs-table-body"></tbody>
            </table>
        </section>
    </main>

    <script>
        function switchView(view) {
            const views = ['chat', 'resilience', 'telemetry', 'logs'];
            views.forEach(v => {
                const el = document.getElementById('view-' + v);
                const btn = document.getElementById('tab-btn-' + v);
                if (el) el.style.display = (v === view) ? 'flex' : 'none';
                if (btn) btn.className = 'nav-tab-btn ' + (v === view ? 'active' : '');
            });
        }

        function setQuickPrompt(text) {
            const input = document.getElementById('prompt-input');
            input.value = text;
            input.focus();
        }

        async function sendPrompt() {
            const input = document.getElementById('prompt-input');
            const prompt = input.value.trim();
            if (!prompt) return;

            const model = document.getElementById('model-select').value;
            const thread = document.getElementById('messages-thread');
            const btn = document.getElementById('btn-send');

            // Append user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message-bubble user';
            userMsg.innerHTML = '<div class="bubble-content">' + escapeHtml(prompt) + '</div>';
            thread.appendChild(userMsg);

            input.value = '';
            btn.disabled = true;
            btn.innerText = 'Generating...';

            // Placeholder for assistant
            const assistantMsg = document.createElement('div');
            assistantMsg.className = 'message-bubble assistant';
            assistantMsg.innerHTML = 
                '<div class="bubble-header">' +
                    '<span>' + model + '</span>' +
                    '<span>•</span>' +
                    '<span>Thinking...</span>' +
                '</div>' +
                '<div class="bubble-content" style="color:var(--text-muted);">Generating response...</div>';
            thread.appendChild(assistantMsg);
            thread.scrollTop = thread.scrollHeight;

            try {
                const res = await fetch('/api/v1/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, prompt })
                });

                if (!res.ok) {
                    const errText = await res.text();
                    assistantMsg.querySelector('.bubble-content').innerText = 'Error: ' + errText;
                    return;
                }

                const data = await res.json();
                const resp = data.response;

                let footerTags = '<span>' + resp.latency_ms + 'ms</span><span>•</span><span>' + resp.total_tokens + ' tokens</span>';
                if (resp.from_cache) {
                    footerTags += '<span>•</span><span class="tag-cache">Cache Hit ($' + data.cost_saved.toFixed(4) + ' saved)</span>';
                }
                if (resp.was_fallback) {
                    footerTags += '<span>•</span><span class="tag-fallback">Fallback from ' + resp.original_model + '</span>';
                }

                assistantMsg.innerHTML = 
                    '<div class="bubble-header">' +
                        '<span>' + resp.provider + ' (' + resp.model + ')</span>' +
                    '</div>' +
                    '<div class="bubble-content">' + escapeHtml(resp.text) + '</div>' +
                    '<div class="bubble-footer">' + footerTags + '</div>';

                thread.scrollTop = thread.scrollHeight;
                refreshMetrics();
                refreshCircuitBreakers();
            } catch (err) {
                assistantMsg.querySelector('.bubble-content').innerText = 'Connection error: ' + err;
            } finally {
                btn.disabled = false;
                btn.innerText = 'Send';
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }

        async function refreshCircuitBreakers() {
            try {
                const res = await fetch('/api/v1/circuit-breakers');
                const list = await res.json();
                const container = document.getElementById('providers-list');
                const resTable = document.getElementById('resilience-table-body');

                let sideHtml = '';
                let tableHtml = '';

                list.forEach(b => {
                    const isClosed = b.state === 'CLOSED';
                    const badgeClass = isClosed ? 'badge-healthy' : 'badge-offline';

                    sideHtml += '<div class="provider-item">' +
                        '<span class="provider-name">' + b.name + '</span>' +
                        '<span class="provider-badge ' + badgeClass + '">' + b.state + '</span>' +
                    '</div>';

                    tableHtml += '<tr>' +
                        '<td style="font-weight:600; color:var(--text-primary);">' + b.name + '</td>' +
                        '<td><span class="provider-badge ' + badgeClass + '">' + b.state + '</span></td>' +
                        '<td>' + b.consecutive_fails + '</td>' +
                        '<td>' +
                            '<button class="btn-sm" onclick="toggleChaos(\'' + b.name + '\', ' + !b.has_simulated_error + ')">' +
                                (b.has_simulated_error ? 'Restore Provider' : 'Simulate 429 Outage') +
                            '</button>' +
                        '</td>' +
                    '</tr>';
                });

                if (container) container.innerHTML = sideHtml;
                if (resTable) resTable.innerHTML = tableHtml;
            } catch (err) {
                console.error("Breaker fetch error:", err);
            }
        }

        async function toggleChaos(provider, enabled) {
            await fetch('/api/v1/chaos', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, enabled })
            });
            refreshCircuitBreakers();
        }

        async function refreshMetrics() {
            try {
                const res = await fetch('/api/v1/metrics');
                const data = await res.json();
                const tel = data.telemetry;

                document.getElementById('side-p95').innerText = tel.p95_latency_ms + 'ms';
                document.getElementById('side-cache').innerText = data.hit_ratio.toFixed(1) + '%';
                document.getElementById('side-saved').innerText = '$' + tel.total_saved_usd.toFixed(4);

                const p50 = document.getElementById('telemetry-p50');
                const p99 = document.getElementById('telemetry-p99');
                const reqs = document.getElementById('telemetry-requests');
                const toks = document.getElementById('telemetry-tokens');

                if (p50) p50.innerText = tel.p50_latency_ms + ' ms';
                if (p99) p99.innerText = tel.p99_latency_ms + ' ms';
                if (reqs) reqs.innerText = tel.total_requests;
                if (toks) toks.innerText = tel.total_tokens.toLocaleString();

                const logsBody = document.getElementById('logs-table-body');
                if (logsBody && tel.recent_logs) {
                    let rows = '';
                    tel.recent_logs.forEach(l => {
                        rows += '<tr>' +
                            '<td style="color:var(--text-muted);">' + new Date(l.timestamp).toLocaleTimeString() + '</td>' +
                            '<td style="font-weight:600; color:var(--text-primary);">' + l.model + '</td>' +
                            '<td>' + escapeHtml(l.prompt_summary) + '</td>' +
                            '<td>' + l.total_tokens + '</td>' +
                            '<td>' + l.latency_ms + 'ms</td>' +
                            '<td>' + l.status + '</td>' +
                        '</tr>';
                    });
                    logsBody.innerHTML = rows || '<tr><td colspan="6">No logs yet.</td></tr>';
                }
            } catch (err) {
                console.error("Metrics error:", err);
            }
        }

        // Enter key to send (Shift+Enter for new line)
        document.addEventListener('DOMContentLoaded', () => {
            refreshCircuitBreakers();
            refreshMetrics();
            setInterval(refreshCircuitBreakers, 4000);
            setInterval(refreshMetrics, 4000);

            const textarea = document.getElementById('prompt-input');
            if (textarea) {
                textarea.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        sendPrompt();
                    }
                });
            }
        });
    </script>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(html))
}
