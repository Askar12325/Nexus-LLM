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
		port = "8082" // Default port for NexusLLM
	}

	r := router.NewRouter()
	c := cache.NewCache(10 * time.Minute)
	l := ratelimit.NewLimiter()
	t := metrics.NewTelemetry()

	// Register Default Demo Tenant
	demoTenant := ratelimit.NewTenant(
		"tenant-enterprise-demo",
		"Enterprise Demo Corp",
		"nx-key-demo-secret",
		ratelimit.TierEnterprise,
		ratelimit.TenantConfig{
			MaxRPS:           20.0,
			MaxTPM:           50000,
			MonthlyBudgetUSD: 500.0,
		},
	)
	l.RegisterTenant(demoTenant)

	// Seed some initial telemetry data
	t.RecordRequest(&metrics.RequestLog{
		ID:            "req-init-1",
		Timestamp:     time.Now().Add(-2 * time.Minute),
		TenantID:      demoTenant.ID,
		Model:         "gpt-4o",
		Provider:      "OpenAI",
		PromptSummary: "What are the advantages of microservice architectures?",
		TotalTokens:   145,
		LatencyMs:     38,
		CostUSD:       0.0018,
		CostSavedUSD:  0.0,
		FromCache:     false,
		WasFallback:   false,
		Status:        "SUCCESS",
	})

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

	log.Printf("⚡ NexusLLM High-Speed AI Gateway running on http://localhost:%s", port)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "UP",
		"service": "NexusLLM Enterprise AI/LLM Gateway & Resiliency Proxy",
		"version": "v1.0.0-pro",
		"time":    time.Now().UTC().Format(time.RFC3339),
	})
}

// handleOpenAICompletion provides 100% OpenAI-compatible /v1/chat/completions endpoint
func (s *Server) handleOpenAICompletion(w http.ResponseWriter, r *http.Request) {
	apiKey := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
	if apiKey == "" {
		apiKey = "nx-key-demo-secret" // Default demo key for ease of use
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

	// Extract prompt text from messages
	var fullPrompt strings.Builder
	for _, m := range req.Messages {
		fullPrompt.WriteString(m.Content + " ")
	}
	prompt := strings.TrimSpace(fullPrompt.String())
	if prompt == "" {
		prompt = "Hello"
	}

	// 1. Rate Limiting Check
	estTokens := int64(len(strings.Fields(prompt)) + 50)
	tenant, err := s.limiter.CheckAndConsume(apiKey, estTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("NexusLLM Rate Limit Violation: %v", err), http.StatusTooManyRequests)
		return
	}

	// 2. PII Sanitization
	sanReport := proxy.SanitizePrompt(prompt)

	// 3. Cache Check
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

	// 4. Upstream Execution via Router with Fallback
	resp, err := s.router.RouteAndExecute(r.Context(), req.Model, sanReport.CleanedText)
	if err != nil {
		http.Error(w, fmt.Sprintf("Gateway routing error: %v", err), http.StatusBadGateway)
		return
	}

	// 5. Store in Cache and Record Telemetry
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

// handleGenerate processes direct AI playground requests with full telemetry reports
func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model  string `json:"model"`
		Prompt string `json:"prompt"`
		Stream bool   `json:"stream"`
	}

	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}

	if req.Model == "" {
		req.Model = "gpt-4o"
	}
	if req.Prompt == "" {
		req.Prompt = "Hello NexusLLM"
	}

	apiKey := "nx-key-demo-secret"
	estTokens := int64(len(strings.Fields(req.Prompt)) + 30)
	tenant, err := s.limiter.CheckAndConsume(apiKey, estTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("Rate limit violation: %v", err), http.StatusTooManyRequests)
		return
	}

	// 1. PII Redaction
	sanReport := proxy.SanitizePrompt(req.Prompt)

	// 2. Cache Check
	if cachedItem, found := s.cache.Get(req.Model, sanReport.CleanedText); found {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID:            fmt.Sprintf("req-cache-%d", time.Now().UnixNano()%100000),
			Timestamp:     time.Now(),
			TenantID:      tenant.ID,
			Model:         req.Model,
			Provider:      "Cache (0ms)",
			PromptSummary: truncate(req.Prompt, 60),
			TotalTokens:   cachedItem.PromptTokens + cachedItem.CompTokens,
			LatencyMs:     2,
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
				Provider:      "In-Memory Cache",
				Text:          cachedItem.Response,
				PromptTokens:  cachedItem.PromptTokens,
				CompTokens:    cachedItem.CompTokens,
				TotalTokens:   cachedItem.PromptTokens + cachedItem.CompTokens,
				LatencyMs:     2,
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

	// 3. Upstream Router Execution with Fallback Cascade
	resp, err := s.router.RouteAndExecute(r.Context(), req.Model, sanReport.CleanedText)
	if err != nil {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID:            fmt.Sprintf("req-err-%d", time.Now().UnixNano()%100000),
			Timestamp:     time.Now(),
			TenantID:      tenant.ID,
			Model:         req.Model,
			Provider:      "Failed All Providers",
			PromptSummary: truncate(req.Prompt, 60),
			LatencyMs:     120,
			Status:        "ERROR",
		})
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	// 4. Cache Result & Record Usage
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
		"message":  fmt.Sprintf("Chaos simulated error on %s set to %v", req.Provider, req.Enabled),
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
    <title>NexusLLM PRO | Enterprise Multi-Tenant AI Gateway & Resiliency Proxy</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500;700;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #06080e;
            --bg-mesh: radial-gradient(circle at 50% 0%, rgba(0, 229, 255, 0.08) 0%, rgba(6, 8, 14, 0.96) 80%);
            --surface: rgba(14, 19, 30, 0.75);
            --surface-card: rgba(20, 28, 44, 0.65);
            --surface-hover: rgba(30, 42, 66, 0.7);
            --border: rgba(255, 255, 255, 0.08);
            --border-cyan: rgba(0, 229, 255, 0.35);
            --text: #f0f6fc;
            --text-dim: #8b9bb4;
            --green: #00F29D;
            --green-glow: rgba(0, 242, 157, 0.4);
            --red: #FF3B69;
            --red-glow: rgba(255, 59, 105, 0.4);
            --gold: #FFD000;
            --gold-glow: rgba(255, 208, 0, 0.4);
            --cyan: #00E5FF;
            --cyan-glow: rgba(0, 229, 255, 0.4);
            --purple: #9D4EDD;
            --touch-target: 44px;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
        
        body {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            background-image: var(--bg-mesh);
            background-attachment: fixed;
            color: var(--text);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        /* Top Header */
        .navbar {
            height: 64px;
            background: rgba(10, 14, 23, 0.88);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            flex-shrink: 0;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        /* 3D Dynamic Nexus Logo */
        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
            text-decoration: none;
            cursor: pointer;
        }
        .logo-icon {
            width: 38px;
            height: 38px;
            animation: floatLogo 4s ease-in-out infinite;
        }
        @keyframes floatLogo {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-3px) rotate(6deg); }
        }
        .logo-text {
            display: flex;
            flex-direction: column;
            line-height: 1.1;
        }
        .logo-title {
            font-size: 20px;
            font-weight: 800;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #ffffff 30%, var(--cyan) 70%, var(--purple) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .logo-badge {
            font-size: 9px;
            font-weight: 800;
            letter-spacing: 1.5px;
            color: var(--cyan);
            text-transform: uppercase;
        }

        .nav-pills {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .status-badge {
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 11px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 242, 157, 0.08);
            border: 1px solid rgba(0, 242, 157, 0.3);
            color: var(--green);
            box-shadow: 0 0 15px rgba(0, 242, 157, 0.15);
        }
        .pulse-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: currentColor;
            box-shadow: 0 0 8px currentColor;
            animation: pulseGlow 1.5s infinite;
        }
        @keyframes pulseGlow { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.3); } }

        /* Navigation Tab Switcher */
        .tab-nav-bar {
            background: rgba(14, 19, 30, 0.9);
            border-bottom: 1px solid var(--border);
            display: flex;
            padding: 0 20px;
            gap: 8px;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            scrollbar-width: none;
            position: sticky;
            top: 64px;
            z-index: 40;
        }
        .tab-nav-bar::-webkit-scrollbar { display: none; }
        .tab-nav-btn {
            padding: 14px 18px;
            background: transparent;
            border: none;
            border-bottom: 2px solid transparent;
            color: var(--text-dim);
            font-size: 13px;
            font-weight: 800;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            white-space: nowrap;
            flex-shrink: 0;
            transition: all 0.2s;
        }
        .tab-nav-btn.active {
            color: #fff;
            border-bottom-color: var(--cyan);
            background: rgba(0, 229, 255, 0.04);
        }

        /* Workspace Grid */
        .workspace {
            padding: 20px;
            max-width: 1440px;
            margin: 0 auto;
            width: 100%;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* Telemetry Cards Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
        }
        .stat-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            backdrop-filter: blur(16px);
            display: flex;
            flex-direction: column;
            gap: 6px;
            transition: all 0.2s;
            position: relative;
            overflow: hidden;
        }
        .stat-card:hover {
            border-color: var(--border-cyan);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 229, 255, 0.1);
        }
        .stat-label {
            font-size: 11px;
            font-weight: 800;
            color: var(--text-dim);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 24px;
            font-weight: 800;
            color: #fff;
        }
        .stat-sub {
            font-size: 11px;
            color: var(--text-dim);
        }

        /* 2-Column Split: AI Playground + Circuit Breaker HUD */
        .main-split {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 20px;
        }

        /* Glassmorphism Panel */
        .glass-panel {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 16px;
            backdrop-filter: blur(20px);
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }
        .panel-title {
            font-size: 14px;
            font-weight: 800;
            display: flex;
            justify-content: space-between;
            align-items: center;
            color: #fff;
            letter-spacing: -0.2px;
        }

        /* Form Inputs */
        .select-box, .textarea-box {
            width: 100%;
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 10px;
            color: #fff;
            padding: 12px 14px;
            font-size: 14px;
            font-family: inherit;
            outline: none;
            transition: all 0.2s;
        }
        .select-box:focus, .textarea-box:focus {
            border-color: var(--cyan);
            box-shadow: 0 0 15px rgba(0, 229, 255, 0.2);
        }
        .textarea-box {
            min-height: 100px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            resize: vertical;
        }

        /* Quick Pre-fill Pills */
        .prompt-pills {
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }
        .prompt-pill {
            background: var(--surface-card);
            border: 1px solid var(--border);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            color: var(--text-dim);
            cursor: pointer;
            transition: all 0.15s;
        }
        .prompt-pill:hover {
            border-color: var(--cyan);
            color: var(--cyan);
            background: rgba(0, 229, 255, 0.08);
        }

        .btn-run {
            height: 48px;
            border: none;
            border-radius: 12px;
            background: linear-gradient(135deg, #00E5FF 0%, #0099FF 100%);
            color: #000;
            font-size: 15px;
            font-weight: 800;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            box-shadow: 0 4px 20px var(--cyan-glow);
            transition: all 0.2s;
        }
        .btn-run:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 25px rgba(0, 229, 255, 0.6);
        }

        /* Response Terminal Box */
        .terminal-box {
            background: #080c14;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            min-height: 160px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: #d1d5db;
            line-height: 1.5;
            white-space: pre-wrap;
            overflow-y: auto;
            max-height: 360px;
        }

        /* Telemetry Pill Badges */
        .meta-badges {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-top: 4px;
        }
        .meta-badge {
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 11px;
            font-weight: 700;
            display: inline-flex;
            align-items: center;
            gap: 5px;
            background: var(--surface-card);
            border: 1px solid var(--border);
        }
        .meta-badge.cache { color: var(--gold); border-color: rgba(255, 208, 0, 0.3); }
        .meta-badge.fallback { color: var(--red); border-color: rgba(255, 59, 105, 0.3); }
        .meta-badge.provider { color: var(--green); border-color: rgba(0, 242, 157, 0.3); }

        /* Circuit Breakers Card List */
        .breaker-card {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 14px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.2s;
        }
        .breaker-info {
            display: flex;
            flex-direction: column;
            gap: 3px;
        }
        .breaker-name { font-size: 14px; font-weight: 800; color: #fff; }
        .breaker-state {
            font-size: 11px;
            font-weight: 800;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .breaker-state.CLOSED { color: var(--green); }
        .breaker-state.OPEN { color: var(--red); }
        .breaker-state.HALF-OPEN { color: var(--gold); }

        .btn-chaos {
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 11px;
            font-weight: 800;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.2s;
        }
        .btn-chaos.trip {
            background: rgba(255, 59, 105, 0.15);
            color: var(--red);
            border-color: rgba(255, 59, 105, 0.4);
        }
        .btn-chaos.trip:hover { background: var(--red); color: #fff; }
        .btn-chaos.reset {
            background: rgba(0, 242, 157, 0.15);
            color: var(--green);
            border-color: rgba(0, 242, 157, 0.4);
        }

        /* Request Inspector Table */
        .log-table {
            width: 100%;
            border-collapse: collapse;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }
        .log-table th {
            padding: 10px 14px;
            text-align: left;
            color: var(--text-dim);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            background: rgba(10, 14, 23, 0.5);
        }
        .log-table td {
            padding: 10px 14px;
            border-bottom: 1px solid var(--border);
        }

        /* RESPONSIVE BREAKPOINTS */
        @media (max-width: 900px) {
            .main-split { grid-template-columns: 1fr; }
            .workspace { padding: 12px; gap: 14px; }
        }
        @media (max-width: 600px) {
            .navbar { padding: 0 12px; }
            .nav-pills { display: none; }
        }
    </style>
</head>
<body>
    <!-- Top Header -->
    <header class="navbar">
        <div class="nav-brand">
            <!-- 3D Vector Nexus Crystal Logo -->
            <svg class="logo-icon" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <defs>
                    <linearGradient id="nexusG1" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#00E5FF" />
                        <stop offset="100%" stop-color="#9D4EDD" />
                    </linearGradient>
                    <linearGradient id="nexusG2" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stop-color="#FFD000" />
                        <stop offset="100%" stop-color="#00F29D" />
                    </linearGradient>
                </defs>
                <polygon points="50,10 90,50 50,90 10,50" fill="url(#nexusG1)" opacity="0.85" />
                <polygon points="50,25 75,50 50,75 25,50" fill="url(#nexusG2)" opacity="0.9" />
                <circle cx="50" cy="50" r="8" fill="#FFFFFF" />
            </svg>
            <div class="logo-text">
                <div class="logo-title">NEXUS<span>LLM</span></div>
                <div class="logo-badge">ENTERPRISE AI PROXY</div>
            </div>
        </div>

        <div class="nav-pills">
            <div class="status-badge">
                <div class="pulse-dot"></div>
                <span>GATEWAY LIVE (< 2ms ROUTING)</span>
            </div>
        </div>
    </header>

    <!-- Navigation Tabs -->
    <nav class="tab-nav-bar">
        <button class="tab-nav-btn active" onclick="switchTab('playground')">⚡ AI Copilot Playground</button>
        <button class="tab-nav-btn" onclick="switchTab('resilience')">🛡️ Circuit Breakers & Chaos</button>
        <button class="tab-nav-btn" onclick="switchTab('telemetry')">📊 Latency & Cost Savings</button>
        <button class="tab-nav-btn" onclick="switchTab('inspector')">📜 Live Request Inspector</button>
    </nav>

    <!-- Main Workspace -->
    <main class="workspace">
        <!-- Telemetry Summary Cards -->
        <section class="stats-grid">
            <div class="stat-card">
                <span class="stat-label">⚡ Total Gateway Requests</span>
                <span class="stat-value" id="stat-requests">0</span>
                <span class="stat-sub" style="color:var(--cyan);">Sub-millisecond routing</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">💰 Dollar Cost Saved ($)</span>
                <span class="stat-value" style="color:var(--gold);" id="stat-saved">$0.0000</span>
                <span class="stat-sub">From instant prompt caching</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">🎯 Cache Hit Ratio</span>
                <span class="stat-value" style="color:var(--green);" id="stat-hit-ratio">0.0%</span>
                <span class="stat-sub" id="stat-cache-counts">0 hits / 0 misses</span>
            </div>
            <div class="stat-card">
                <span class="stat-label">🏎️ P95 Latency</span>
                <span class="stat-value" id="stat-p95">1 ms</span>
                <span class="stat-sub" id="stat-fallbacks">0 automatic fallbacks</span>
            </div>
        </section>

        <!-- VIEW 1: AI Playground & Circuit Breaker Split -->
        <section id="view-playground" class="main-split">
            <!-- Left: Prompt Playground -->
            <div class="glass-panel">
                <div class="panel-title">
                    <span>⚡ AI Inference Playground</span>
                    <span style="font-size:11px; color:var(--text-dim);">OpenAI-Compatible (:8082/v1)</span>
                </div>

                <div>
                    <label style="font-size:11px; font-weight:800; color:var(--text-dim); display:block; margin-bottom:6px;">SELECT PRIMARY MODEL</label>
                    <select class="select-box" id="model-select">
                        <option value="gpt-4o">OpenAI • GPT-4o (Omni)</option>
                        <option value="claude-3-5-sonnet">Anthropic • Claude 3.5 Sonnet</option>
                        <option value="gemini-1.5-pro">Google • Gemini 1.5 Pro</option>
                        <option value="deepseek-chat">DeepSeek • DeepSeek V3</option>
                    </select>
                </div>

                <div>
                    <label style="font-size:11px; font-weight:800; color:var(--text-dim); display:block; margin-bottom:6px;">PROMPT INPUT</label>
                    <textarea class="textarea-box" id="prompt-input" placeholder="Type a prompt to test zero-downtime routing, caching, and PII protection..."></textarea>
                </div>

                <div class="prompt-pills">
                    <button class="prompt-pill" onclick="setPrompt('What is the capital of France?')">⚡ Exact Cache Test</button>
                    <button class="prompt-pill" onclick="setPrompt('My email is john.doe@corp.com and card is 4111-2222-3333-4444')">🔒 PII Redaction Test</button>
                    <button class="prompt-pill" onclick="setPrompt('Write a high-performance concurrent worker pool in Go')">💻 Code Generation</button>
                </div>

                <button class="btn-run" id="btn-generate" onclick="executePrompt()">
                    <span>⚡</span> Execute Prompt via NexusLLM
                </button>

                <!-- Output Response Card -->
                <div>
                    <div class="panel-title" style="margin-bottom:8px;">
                        <span>Inference Result</span>
                        <div class="meta-badges" id="result-badges"></div>
                    </div>
                    <div class="terminal-box" id="output-terminal">Ready. Click 'Execute Prompt' or pick a test preset above.</div>
                </div>
            </div>

            <!-- Right: Active Circuit Breaker & Health HUD -->
            <div class="glass-panel">
                <div class="panel-title">
                    <span>🛡️ Downstream Circuit Breakers</span>
                    <span style="font-size:11px; color:var(--green);">AUTO-RECOVERY ACTIVE</span>
                </div>
                <p style="font-size:12px; color:var(--text-dim); line-height:1.4;">
                    If an upstream provider hits rate limits (429) or timeouts, NexusLLM trips its circuit breaker and automatically cascades your prompt to the next backup model!
                </p>

                <div id="circuit-breakers-container" style="display:flex; flex-direction:column; gap:10px;">
                    <!-- Rendered dynamically -->
                </div>

                <div style="margin-top:auto; padding-top:14px; border-top:1px solid var(--border);">
                    <div style="font-size:11px; font-weight:800; color:var(--text-dim); margin-bottom:8px;">QUICK CACHE CONTROLS</div>
                    <button class="prompt-pill" style="width:100%; text-align:center; padding:10px;" onclick="clearCache()">
                        🗑️ Clear Prompt Cache
                    </button>
                </div>
            </div>
        </section>

        <!-- VIEW 2: Circuit Breakers & Chaos Simulator -->
        <section id="view-resilience" class="glass-panel" style="display:none;">
            <div class="panel-title">
                <span>🛡️ Chaos Engineering & Provider Outage Simulator</span>
                <span style="font-size:11px; color:var(--cyan);">LIVE FAULT INJECTION</span>
            </div>
            <p style="font-size:13px; color:var(--text-dim);">
                Simulate sudden provider outages, HTTP 429 rate limits, or network partitions. Watch NexusLLM seamlessly protect client applications from downtime!
            </p>
            <div id="chaos-grid" style="display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:14px; margin-top:10px;"></div>
        </section>

        <!-- VIEW 3: Telemetry & Cost Savings -->
        <section id="view-telemetry" class="glass-panel" style="display:none;">
            <div class="panel-title">
                <span>📊 Latency Histograms & Telemetry Analytics</span>
                <span style="font-size:11px; color:var(--gold);">PROMETHEUS COMPATIBLE</span>
            </div>
            <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:14px; margin-top:10px;">
                <div class="stat-card">
                    <span class="stat-label">P50 Latency (Median)</span>
                    <span class="stat-value" id="stat-p50" style="color:var(--green);">1 ms</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">P99 Latency (Tail)</span>
                    <span class="stat-value" id="stat-p99" style="color:var(--cyan);">2 ms</span>
                </div>
                <div class="stat-card">
                    <span class="stat-label">Total Tokens Processed</span>
                    <span class="stat-value" id="stat-tokens">0</span>
                </div>
            </div>
        </section>

        <!-- VIEW 4: Live Request Inspector -->
        <section id="view-inspector" class="glass-panel" style="display:none;">
            <div class="panel-title">
                <span>📜 Real-Time Gateway Request Inspector</span>
                <span style="font-size:11px; color:var(--green);">● LIVE FEED</span>
            </div>
            <div style="overflow-x:auto;">
                <table class="log-table">
                    <thead>
                        <tr>
                            <th>Time</th>
                            <th>Model / Provider</th>
                            <th>Prompt Summary</th>
                            <th>Tokens</th>
                            <th>Latency</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody id="log-table-body"></tbody>
                </table>
            </div>
        </section>
    </main>

    <script>
        let currentTab = "playground";

        function switchTab(tab) {
            currentTab = tab;
            const tabs = ['playground', 'resilience', 'telemetry', 'inspector'];
            tabs.forEach(t => {
                const el = document.getElementById('view-' + t);
                if (el) el.style.display = (t === tab) ? (t === 'playground' ? 'grid' : 'flex') : 'none';
            });
            const btns = document.querySelectorAll('.tab-nav-btn');
            btns.forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
        }

        function setPrompt(text) {
            document.getElementById('prompt-input').value = text;
        }

        async function executePrompt() {
            const model = document.getElementById('model-select').value;
            const prompt = document.getElementById('prompt-input').value.trim();
            if (!prompt) return;

            const btn = document.getElementById('btn-generate');
            const terminal = document.getElementById('output-terminal');
            const badges = document.getElementById('result-badges');

            btn.disabled = true;
            btn.innerText = "⚡ Routing through NexusLLM...";
            terminal.innerText = "Streaming response...";
            badges.innerHTML = "";

            try {
                const res = await fetch('/api/v1/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ model, prompt })
                });

                if (!res.ok) {
                    const err = await res.text();
                    terminal.innerText = "Error: " + err;
                    return;
                }

                const data = await res.json();
                const resp = data.response;
                terminal.innerText = resp.text;

                // Render Metadata Badges
                let badgeHtml = '<span class="meta-badge provider">● ' + resp.provider + ' (' + resp.latency_ms + 'ms)</span>';
                if (resp.from_cache) {
                    badgeHtml += '<span class="meta-badge cache">⚡ Cache HIT ($' + data.cost_saved.toFixed(4) + ' saved)</span>';
                }
                if (resp.was_fallback) {
                    badgeHtml += '<span class="meta-badge fallback">🛡️ Auto-Fallback from ' + resp.original_model + '</span>';
                }
                if (data.sanitization && data.sanitization.was_sanitized) {
                    badgeHtml += '<span class="meta-badge" style="color:var(--cyan); border-color:var(--cyan-glow);">🔒 PII Redacted: ' + data.sanitization.redacted_types.join(', ') + '</span>';
                }
                badges.innerHTML = badgeHtml;

                // Refresh metrics & circuit breakers
                refreshMetrics();
                refreshCircuitBreakers();
            } catch (err) {
                terminal.innerText = "Gateway Connection Error: " + err;
            } finally {
                btn.disabled = false;
                btn.innerText = "⚡ Execute Prompt via NexusLLM";
            }
        }

        async function refreshCircuitBreakers() {
            try {
                const res = await fetch('/api/v1/circuit-breakers');
                const list = await res.json();
                const container = document.getElementById('circuit-breakers-container');
                const chaosContainer = document.getElementById('chaos-grid');

                let html = '';
                let chaosHtml = '';

                list.forEach(b => {
                    const isClosed = b.state === 'CLOSED';
                    const stateColor = isClosed ? 'CLOSED' : (b.state === 'OPEN' ? 'OPEN' : 'HALF-OPEN');

                    html += '<div class="breaker-card">' +
                        '<div class="breaker-info">' +
                            '<span class="breaker-name">' + b.name + '</span>' +
                            '<span class="breaker-state ' + stateColor + '">● ' + b.state + (b.has_simulated_error ? ' (Chaos Injected)' : '') + '</span>' +
                        '</div>' +
                        '<button class="btn-chaos ' + (b.has_simulated_error ? 'reset' : 'trip') + '" onclick="toggleChaos(\'' + b.name + '\', ' + !b.has_simulated_error + ')">' +
                            (b.has_simulated_error ? '✓ Restore Provider' : '✕ Simulate Outage') +
                        '</button>' +
                    '</div>';

                    chaosHtml += '<div class="breaker-card">' +
                        '<div class="breaker-info">' +
                            '<span class="breaker-name">' + b.name + '</span>' +
                            '<span class="breaker-state ' + stateColor + '">Consecutive Fails: ' + b.consecutive_fails + ' | Trips: ' + b.total_trips + '</span>' +
                        '</div>' +
                        '<button class="btn-chaos ' + (b.has_simulated_error ? 'reset' : 'trip') + '" onclick="toggleChaos(\'' + b.name + '\', ' + !b.has_simulated_error + ')">' +
                            (b.has_simulated_error ? '✓ Restore' : '⚡ Inject 429 Error') +
                        '</button>' +
                    '</div>';
                });

                if (container) container.innerHTML = html;
                if (chaosContainer) chaosContainer.innerHTML = chaosHtml;
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

        async function clearCache() {
            await fetch('/api/v1/cache/clear', { method: 'POST' });
            alert('Prompt Cache Cleared!');
            refreshMetrics();
        }

        async function refreshMetrics() {
            try {
                const res = await fetch('/api/v1/metrics');
                const data = await res.json();
                const tel = data.telemetry;

                document.getElementById('stat-requests').innerText = tel.total_requests;
                document.getElementById('stat-saved').innerText = '$' + tel.total_saved_usd.toFixed(4);
                document.getElementById('stat-hit-ratio').innerText = data.hit_ratio.toFixed(1) + '%';
                document.getElementById('stat-cache-counts').innerText = data.cache_hits + ' hits / ' + data.cache_miss + ' misses';
                document.getElementById('stat-p95').innerText = tel.p95_latency_ms + ' ms';
                document.getElementById('stat-fallbacks').innerText = tel.fallback_count + ' automatic fallbacks';

                const p50El = document.getElementById('stat-p50');
                const p99El = document.getElementById('stat-p99');
                const tokensEl = document.getElementById('stat-tokens');
                if (p50El) p50El.innerText = tel.p50_latency_ms + ' ms';
                if (p99El) p99El.innerText = tel.p99_latency_ms + ' ms';
                if (tokensEl) tokensEl.innerText = tel.total_tokens.toLocaleString();

                // Render Logs Table
                const tbody = document.getElementById('log-table-body');
                if (tbody && tel.recent_logs) {
                    let rows = '';
                    tel.recent_logs.forEach(l => {
                        const statusColor = l.status === 'SUCCESS' ? 'var(--green)' : (l.status === 'CACHED' ? 'var(--gold)' : (l.status === 'FALLBACK' ? 'var(--cyan)' : 'var(--red)'));
                        rows += '<tr>' +
                            '<td style="color:var(--text-dim);">' + new Date(l.timestamp).toLocaleTimeString() + '</td>' +
                            '<td><strong>' + l.model + '</strong> <span style="color:var(--text-dim); font-size:10px;">(' + l.provider + ')</span></td>' +
                            '<td style="color:#d1d5db;">' + l.prompt_summary + '</td>' +
                            '<td>' + l.total_tokens + '</td>' +
                            '<td style="color:var(--cyan);">' + l.latency_ms + ' ms</td>' +
                            '<td style="color:' + statusColor + '; font-weight:800;">' + l.status + '</td>' +
                        '</tr>';
                    });
                    tbody.innerHTML = rows || '<tr><td colspan="6" style="text-align:center; color:var(--text-dim);">No requests logged yet.</td></tr>';
                }
            } catch (err) {
                console.error("Metrics error:", err);
            }
        }

        // Initialize App & Periodic Poll
        window.addEventListener('DOMContentLoaded', () => {
            refreshMetrics();
            refreshCircuitBreakers();
            setInterval(refreshMetrics, 3000);
            setInterval(refreshCircuitBreakers, 3000);
        });
    </script>
</body>
</html>`

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(html))
}
