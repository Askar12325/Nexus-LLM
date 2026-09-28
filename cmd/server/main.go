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

// SSEEvent represents a server-sent event pushed to active dashboard tabs
type SSEEvent struct {
	Type string `json:"type"`
	Data any    `json:"data"`
}

// SSEBroker coordinates real-time telemetry and circuit state events over SSE
type SSEBroker struct {
	clients map[chan SSEEvent]bool
	mu      sync.RWMutex
}

func newSSEBroker() *SSEBroker {
	return &SSEBroker{
		clients: make(map[chan SSEEvent]bool),
	}
}

func (b *SSEBroker) subscribe() chan SSEEvent {
	b.mu.Lock()
	defer b.mu.Unlock()
	ch := make(chan SSEEvent, 16)
	b.clients[ch] = true
	return ch
}

func (b *SSEBroker) unsubscribe(ch chan SSEEvent) {
	b.mu.Lock()
	defer b.mu.Unlock()
	delete(b.clients, ch)
	close(ch)
}

func (b *SSEBroker) broadcast(event SSEEvent) {
	b.mu.RLock()
	defer b.mu.RUnlock()
	for ch := range b.clients {
		select {
		case ch <- event:
		default:
		}
	}
}

type Server struct {
	router    *router.Router
	cache     *cache.Cache
	limiter   *ratelimit.Limiter
	telemetry *metrics.Telemetry
	sseBroker *SSEBroker
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
	broker := newSSEBroker()

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

	// Pre-seed realistic telemetry, prompt cache, and provider baseline states.
	// All providers start in CLOSED (HEALTHY) state; user simulates 429 outage via UI to test failover.
	seedTelemetry(t, c, l, r)

	server := &Server{
		router:    r,
		cache:     c,
		limiter:   l,
		telemetry: t,
		sseBroker: broker,
	}

	// ── Auto-recovery background loop ─────────────────────────────────────────
	// Checks circuit breakers every 2 seconds. When an OPEN provider passes its
	// recovery timeout, CanExecute() automatically transitions it to HALF-OPEN.
	// Broadcasts real-time SSE events so the UI updates instantly without polling.
	go func() {
		ticker := time.NewTicker(2 * time.Second)
		defer ticker.Stop()
		lastStates := make(map[string]string)

		for range ticker.C {
			for _, p := range server.router.GetProviders() {
				name, state, _, _ := p.CircuitBreaker().Status()
				if string(state) == "OPEN" {
					// Probe if cooldown window has elapsed; transitions OPEN -> HALF-OPEN
					if p.CircuitBreaker().CanExecute() {
						name, state, _, _ = p.CircuitBreaker().Status()
					}
				}
				oldState := lastStates[name]
				if oldState != "" && oldState != string(state) {
					server.sseBroker.broadcast(SSEEvent{
						Type: "breaker_change",
						Data: map[string]any{
							"provider": name,
							"state":    string(state),
						},
					})
				}
				lastStates[name] = string(state)
			}
		}
	}()

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", server.handleDashboardUI)
	mux.HandleFunc("GET /health", server.handleHealth)
	mux.HandleFunc("GET /api/v1/events", server.handleSSEEvents)
	mux.HandleFunc("POST /v1/chat/completions", server.handleOpenAICompletion)
	mux.HandleFunc("POST /api/v1/generate", server.handleGenerate)
	mux.HandleFunc("GET /api/v1/metrics", server.handleGetMetrics)
	mux.HandleFunc("GET /metrics", server.handlePrometheus)
	mux.HandleFunc("GET /api/v1/circuit-breakers", server.handleGetCircuitBreakers)
	mux.HandleFunc("POST /api/v1/chaos", server.handleChaosToggle)
	mux.HandleFunc("POST /api/v1/cache/clear", server.handleClearCache)
	mux.HandleFunc("GET /api/v1/tenants", server.handleGetTenants)
	mux.HandleFunc("POST /api/v1/tenants", server.handleRegisterTenant)

	httpServer := &http.Server{Addr: ":" + port, Handler: mux}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)
		<-sigChan
		log.Println("Shutting down NexusLLM Gateway...")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(ctx)
	}()

	log.Printf("NexusLLM Gateway running on http://localhost:%s", port)
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server error: %v", err)
	}
}

// ── Health ────────────────────────────────────────────────────────────────────

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{
		"status":  "UP",
		"service": "NexusLLM Enterprise AI Gateway",
		"time":    time.Now().UTC().Format(time.RFC3339),
	})
}

// ── SSE Events Stream ─────────────────────────────────────────────────────────

func (s *Server) handleSSEEvents(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	ch := s.sseBroker.subscribe()
	defer s.sseBroker.unsubscribe(ch)

	fmt.Fprintf(w, "event: connected\ndata: {\"status\":\"ok\"}\n\n")
	flusher.Flush()

	for {
		select {
		case <-r.Context().Done():
			return
		case ev := <-ch:
			data, _ := json.Marshal(ev.Data)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", ev.Type, data)
			flusher.Flush()
		}
	}
}

// ── OpenAI-compatible inference ───────────────────────────────────────────────

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

	var sb strings.Builder
	for _, m := range req.Messages {
		sb.WriteString(m.Content + " ")
	}
	prompt := strings.TrimSpace(sb.String())
	if prompt == "" {
		prompt = "Hello"
	}

	estTokens := int64(len(strings.Fields(prompt)) + 50)
	tenant, err := s.limiter.CheckAndConsume(apiKey, estTokens)
	if err != nil {
		http.Error(w, fmt.Sprintf("Rate limit: %v", err), http.StatusTooManyRequests)
		return
	}

	san := proxy.SanitizePrompt(prompt)

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "streaming not supported", http.StatusInternalServerError)
			return
		}
		ch := make(chan string, 64)
		errc := make(chan error, 1)
		var sResp *router.CompletionResponse
		go func() {
			var e error
			sResp, e = s.router.RouteAndStreamExecute(r.Context(), req.Model, san.CleanedText, ch)
			errc <- e
			close(ch)
		}()
		for tok := range ch {
			d, _ := json.Marshal(map[string]any{
				"id": fmt.Sprintf("cmpl-%d", time.Now().UnixNano()%100000),
				"object": "chat.completion.chunk", "model": req.Model,
				"choices": []map[string]any{{"delta": map[string]string{"content": tok}, "index": 0}},
			})
			fmt.Fprintf(w, "data: %s\n\n", d)
			flusher.Flush()
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		flusher.Flush()
		if e := <-errc; e == nil && sResp != nil {
			s.cache.Set(req.Model, san.CleanedText, sResp.Text, sResp.PromptTokens, sResp.CompTokens, sResp.EstimatedCost)
			s.limiter.RecordUsage(apiKey, sResp.PromptTokens, sResp.CompTokens, estTokens, sResp.EstimatedCost)
			st := "SUCCESS"
			if sResp.WasFallback {
				st = "FALLBACK"
				s.sseBroker.broadcast(SSEEvent{
					Type: "fallback",
					Data: map[string]any{
						"requested_model": req.Model,
						"served_model":    sResp.Model,
						"served_provider": string(sResp.Provider),
						"chain":           sResp.FallbackChain,
						"latency_ms":      sResp.LatencyMs,
					},
				})
			}
			s.telemetry.RecordRequest(&metrics.RequestLog{
				ID: sResp.ID, Timestamp: time.Now(), TenantID: tenant.ID,
				Model: sResp.Model, Provider: string(sResp.Provider),
				PromptSummary: truncate(prompt, 60), TotalTokens: sResp.TotalTokens,
				LatencyMs: sResp.LatencyMs, CostUSD: sResp.EstimatedCost,
				FromCache: false, WasFallback: sResp.WasFallback, Status: st,
			})
		}
		return
	}

	if cached, found := s.cache.Get(req.Model, san.CleanedText); found {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID: fmt.Sprintf("req-%d", time.Now().UnixNano()%1000000), Timestamp: time.Now(),
			TenantID: tenant.ID, Model: req.Model, Provider: "Cache",
			PromptSummary: truncate(prompt, 60),
			TotalTokens:   cached.PromptTokens + cached.CompTokens,
			LatencyMs: 1, CostSavedUSD: cached.CostSaved, FromCache: true, Status: "CACHED",
		})
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("X-Nexus-Cache", "HIT")
		_ = json.NewEncoder(w).Encode(buildOAIResp(req.Model, cached.Response, cached.PromptTokens, cached.CompTokens, "cached"))
		return
	}

	resp, err := s.router.RouteAndExecute(r.Context(), req.Model, san.CleanedText)
	if err != nil {
		http.Error(w, fmt.Sprintf("Gateway error: %v", err), http.StatusBadGateway)
		return
	}
	s.cache.Set(req.Model, san.CleanedText, resp.Text, resp.PromptTokens, resp.CompTokens, resp.EstimatedCost)
	s.limiter.RecordUsage(apiKey, resp.PromptTokens, resp.CompTokens, estTokens, resp.EstimatedCost)
	st := "SUCCESS"
	if resp.WasFallback {
		st = "FALLBACK"
		s.sseBroker.broadcast(SSEEvent{
			Type: "fallback",
			Data: map[string]any{
				"requested_model": req.Model,
				"served_model":    resp.Model,
				"served_provider": string(resp.Provider),
				"chain":           resp.FallbackChain,
				"latency_ms":      resp.LatencyMs,
			},
		})
	}
	s.telemetry.RecordRequest(&metrics.RequestLog{
		ID: resp.ID, Timestamp: time.Now(), TenantID: tenant.ID,
		Model: resp.Model, Provider: string(resp.Provider),
		PromptSummary: truncate(prompt, 60), TotalTokens: resp.TotalTokens,
		LatencyMs: resp.LatencyMs, CostUSD: resp.EstimatedCost,
		FromCache: false, WasFallback: resp.WasFallback, Status: st,
	})
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Nexus-Cache", "MISS")
	w.Header().Set("X-Nexus-Provider", string(resp.Provider))
	_ = json.NewEncoder(w).Encode(buildOAIResp(resp.Model, resp.Text, resp.PromptTokens, resp.CompTokens, resp.ID))
}

func buildOAIResp(model, content string, pt, ct int, id string) map[string]any {
	return map[string]any{
		"id": id, "object": "chat.completion", "created": time.Now().Unix(), "model": model,
		"choices": []map[string]any{{"index": 0, "message": map[string]string{"role": "assistant", "content": content}, "finish_reason": "stop"}},
		"usage":   map[string]int{"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
	}
}

// ── Internal generate ─────────────────────────────────────────────────────────

func (s *Server) handleGenerate(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Model      string `json:"model"`
		Prompt     string `json:"prompt"`
		Structured bool   `json:"structured"`
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
		http.Error(w, fmt.Sprintf("Rate limit: %v", err), http.StatusTooManyRequests)
		return
	}

	san := proxy.SanitizePrompt(req.Prompt)

	// Prepend structured sentinel if requested by client
	routedPrompt := san.CleanedText
	if req.Structured {
		routedPrompt = router.StructuredSentinel + routedPrompt
	}

	// Cache lookup uses the exact routedPrompt string
	if cached, found := s.cache.Get(req.Model, routedPrompt); found {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID: fmt.Sprintf("req-cache-%d", time.Now().UnixNano()%100000), Timestamp: time.Now(),
			TenantID: tenant.ID, Model: req.Model, Provider: "Cache",
			PromptSummary: truncate(req.Prompt, 60),
			TotalTokens:   cached.PromptTokens + cached.CompTokens,
			LatencyMs: 1, CostSavedUSD: cached.CostSaved, FromCache: true, Status: "CACHED",
		})
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"response": &router.CompletionResponse{
				ID:            fmt.Sprintf("chatcmpl-cached-%d", time.Now().UnixNano()%100000),
				Model:         req.Model,
				Provider:      "Cache",
				Text:          cached.Response,
				PromptTokens:  cached.PromptTokens,
				CompTokens:    cached.CompTokens,
				TotalTokens:   cached.PromptTokens + cached.CompTokens,
				LatencyMs:     1,
				EstimatedCost: 0.0,
				FromCache:     true,
			},
			"sanitization": san,
			"cost_saved":   cached.CostSaved,
		})
		return
	}

	resp, err := s.router.RouteAndExecute(r.Context(), req.Model, routedPrompt)
	if err != nil {
		s.telemetry.RecordRequest(&metrics.RequestLog{
			ID: fmt.Sprintf("req-err-%d", time.Now().UnixNano()%100000), Timestamp: time.Now(),
			TenantID: tenant.ID, Model: req.Model, Provider: "All Providers Failed",
			PromptSummary: truncate(req.Prompt, 60), LatencyMs: 120, Status: "ERROR",
		})
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	s.cache.Set(req.Model, routedPrompt, resp.Text, resp.PromptTokens, resp.CompTokens, resp.EstimatedCost)
	s.limiter.RecordUsage(apiKey, resp.PromptTokens, resp.CompTokens, estTokens, resp.EstimatedCost)
	st := "SUCCESS"
	if resp.WasFallback {
		st = "FALLBACK"
		s.sseBroker.broadcast(SSEEvent{
			Type: "fallback",
			Data: map[string]any{
				"requested_model": req.Model,
				"served_model":    resp.Model,
				"served_provider": string(resp.Provider),
				"chain":           resp.FallbackChain,
				"latency_ms":      resp.LatencyMs,
			},
		})
	}
	s.telemetry.RecordRequest(&metrics.RequestLog{
		ID: resp.ID, Timestamp: time.Now(), TenantID: tenant.ID,
		Model: resp.Model, Provider: string(resp.Provider),
		PromptSummary: truncate(req.Prompt, 60), TotalTokens: resp.TotalTokens,
		LatencyMs: resp.LatencyMs, CostUSD: resp.EstimatedCost,
		FromCache: false, WasFallback: resp.WasFallback, Status: st,
	})
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"response": resp, "sanitization": san, "cost_saved": 0.0,
	})
}

// ── Metrics ───────────────────────────────────────────────────────────────────

func (s *Server) handleGetMetrics(w http.ResponseWriter, r *http.Request) {
	snap := s.telemetry.GetSnapshot()
	hits, misses, cacheSize, ratio := s.cache.Stats()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"telemetry": snap, "cache_size": cacheSize,
		"cache_hits": hits, "cache_miss": misses, "hit_ratio": ratio,
	})
}

func (s *Server) handlePrometheus(w http.ResponseWriter, r *http.Request) {
	snap := s.telemetry.GetSnapshot()
	hits, misses, cacheSize, _ := s.cache.Stats()
	w.Header().Set("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
	lines := []string{
		"# HELP nexusllm_requests_total Total requests processed",
		"# TYPE nexusllm_requests_total counter",
		fmt.Sprintf("nexusllm_requests_total %d", snap.TotalRequests),
		"# HELP nexusllm_tokens_total Total tokens processed",
		"# TYPE nexusllm_tokens_total counter",
		fmt.Sprintf("nexusllm_tokens_total %d", snap.TotalTokens),
		"# HELP nexusllm_cost_usd_total Total cost USD",
		"# TYPE nexusllm_cost_usd_total counter",
		fmt.Sprintf("nexusllm_cost_usd_total %.6f", snap.TotalCostUSD),
		"# HELP nexusllm_cost_saved_usd_total Cost saved via cache USD",
		"# TYPE nexusllm_cost_saved_usd_total counter",
		fmt.Sprintf("nexusllm_cost_saved_usd_total %.6f", snap.TotalSavedUSD),
		"# HELP nexusllm_cache_hits_total Cache hits",
		"# TYPE nexusllm_cache_hits_total counter",
		fmt.Sprintf("nexusllm_cache_hits_total %d", hits),
		"# HELP nexusllm_cache_misses_total Cache misses",
		"# TYPE nexusllm_cache_misses_total counter",
		fmt.Sprintf("nexusllm_cache_misses_total %d", misses),
		"# HELP nexusllm_cache_size Current cache items",
		"# TYPE nexusllm_cache_size gauge",
		fmt.Sprintf("nexusllm_cache_size %d", cacheSize),
		"# HELP nexusllm_fallback_total Fallback events",
		"# TYPE nexusllm_fallback_total counter",
		fmt.Sprintf("nexusllm_fallback_total %d", snap.FallbackCount),
		"# HELP nexusllm_latency_p50_ms P50 latency ms",
		"# TYPE nexusllm_latency_p50_ms gauge",
		fmt.Sprintf("nexusllm_latency_p50_ms %d", snap.P50LatencyMs),
		"# HELP nexusllm_latency_p95_ms P95 latency ms",
		"# TYPE nexusllm_latency_p95_ms gauge",
		fmt.Sprintf("nexusllm_latency_p95_ms %d", snap.P95LatencyMs),
		"# HELP nexusllm_latency_p99_ms P99 latency ms",
		"# TYPE nexusllm_latency_p99_ms gauge",
		fmt.Sprintf("nexusllm_latency_p99_ms %d", snap.P99LatencyMs),
	}
	for _, ln := range lines {
		fmt.Fprintln(w, ln)
	}
	fmt.Fprintln(w, "# HELP nexusllm_circuit_breaker_open 1=open 0=closed")
	fmt.Fprintln(w, "# TYPE nexusllm_circuit_breaker_open gauge")
	for _, p := range s.router.GetProviders() {
		_, state, _, _ := p.CircuitBreaker().Status()
		v := 0
		if state != "CLOSED" {
			v = 1
		}
		fmt.Fprintf(w, "nexusllm_circuit_breaker_open{provider=%q} %d\n", p.Name(), v)
	}
	fmt.Fprintln(w, "# HELP nexusllm_circuit_breaker_trips_total Total trips")
	fmt.Fprintln(w, "# TYPE nexusllm_circuit_breaker_trips_total counter")
	for _, p := range s.router.GetProviders() {
		_, _, _, trips := p.CircuitBreaker().Status()
		fmt.Fprintf(w, "nexusllm_circuit_breaker_trips_total{provider=%q} %d\n", p.Name(), trips)
	}
}

// ── Circuit breakers / Chaos ──────────────────────────────────────────────────

func (s *Server) handleGetCircuitBreakers(w http.ResponseWriter, r *http.Request) {
	type BS struct {
		Name             string  `json:"name"`
		State            string  `json:"state"`
		ConsecutiveFails int     `json:"consecutive_fails"`
		TotalTrips       int64   `json:"total_trips"`
		HasSimulatedErr  bool    `json:"has_simulated_error"`
		LastLatencyMs    int64   `json:"last_latency_ms"`
		SuccessRatePct   float64 `json:"success_rate_pct"`
		LastErrorMsg     string  `json:"last_error_msg"`
		TotalRequests    int64   `json:"total_requests"`
	}
	var list []BS
	for _, p := range s.router.GetProviders() {
		name, state, fails, trips := p.CircuitBreaker().Status()
		stats := p.Stats()
		list = append(list, BS{
			Name:             name,
			State:            string(state),
			ConsecutiveFails: fails,
			TotalTrips:       trips,
			HasSimulatedErr:  s.router.HasSimulatedError(p.Name()),
			LastLatencyMs:    stats.LastLatencyMs,
			SuccessRatePct:   stats.SuccessRatePct,
			LastErrorMsg:     stats.LastErrorMsg,
			TotalRequests:    stats.TotalRequests,
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

	// Broadcast circuit change immediately over SSE
	state := "CLOSED"
	if req.Enabled {
		state = "OPEN"
	}
	s.sseBroker.broadcast(SSEEvent{
		Type: "breaker_change",
		Data: map[string]any{
			"provider": req.Provider,
			"state":    state,
			"chaos":    req.Enabled,
		},
	})

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"provider": req.Provider,
		"enabled":  req.Enabled,
		"state":    state,
	})
}

// ── Cache ─────────────────────────────────────────────────────────────────────

func (s *Server) handleClearCache(w http.ResponseWriter, r *http.Request) {
	s.cache.Clear()
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]string{"message": "Cache cleared"})
}

// ── Tenants ───────────────────────────────────────────────────────────────────

func (s *Server) handleGetTenants(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(s.limiter.GetAllTenants())
}

func (s *Server) handleRegisterTenant(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID               string  `json:"id"`
		Name             string  `json:"name"`
		APIKey           string  `json:"api_key"`
		Tier             string  `json:"tier"`
		MaxRPS           float64 `json:"max_rps"`
		MaxTPM           int64   `json:"max_tpm"`
		MonthlyBudgetUSD float64 `json:"monthly_budget_usd"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.ID == "" || req.Name == "" || req.APIKey == "" {
		http.Error(w, "id, name and api_key are required", http.StatusBadRequest)
		return
	}
	if req.MaxRPS <= 0 {
		req.MaxRPS = 10.0
	}
	if req.MaxTPM <= 0 {
		req.MaxTPM = 50000
	}
	if req.MonthlyBudgetUSD <= 0 {
		req.MonthlyBudgetUSD = 100.0
	}
	tier := ratelimit.TierDeveloper
	switch strings.ToUpper(req.Tier) {
	case "FREE":
		tier = ratelimit.TierFree
	case "ENTERPRISE":
		tier = ratelimit.TierEnterprise
	}
	tenant := ratelimit.NewTenant(req.ID, req.Name, req.APIKey, tier, ratelimit.TenantConfig{
		MaxRPS: req.MaxRPS, MaxTPM: req.MaxTPM, MonthlyBudgetUSD: req.MonthlyBudgetUSD,
	})
	s.limiter.RegisterTenant(tenant)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(tenant)
}

// ── Pre-seed Telemetry ────────────────────────────────────────────────────────

func seedTelemetry(t *metrics.Telemetry, c *cache.Cache, l *ratelimit.Limiter, r *router.Router) {
	now := time.Now()

	// 1. Seed two warm cache entries
	c.Set(
		"gpt-4o",
		"Explain the circuit breaker pattern in distributed systems",
		"A circuit breaker wraps calls to a downstream service and tracks whether those calls succeed. After a configured number of consecutive failures it opens, and requests are rejected immediately without connection attempts, thread consumption, or timeout delays.\n\nAfter a recovery timeout the breaker moves to half-open and lets a single probe request through. If it succeeds, the breaker closes and normal traffic resumes.",
		12, 120, 0.0035,
	)
	c.Set(
		"claude-3-5-sonnet",
		"What is the capital of France?",
		"The capital of France is Paris.",
		8, 8, 0.00018,
	)

	// 2. Seed 16 realistic historical logs
	historicalLogs := []struct {
		model    string
		provider string
		prompt   string
		tokens   int
		latMs    int64
		cost     float64
		saved    float64
		cached   bool
		fallback bool
		status   string
		agoMin   int
	}{
		{"gpt-4o", "OpenAI", "Explain distributed consensus and Raft", 512, 218, 0.0084, 0.0, false, false, "SUCCESS", 28},
		{"claude-3-5-sonnet", "Anthropic", "Compare Go vs Python backend performance", 680, 184, 0.0062, 0.0, false, false, "SUCCESS", 25},
		{"gpt-4o", "OpenAI", "Draft a Q3 engineering deliverable email", 340, 192, 0.0048, 0.0, false, false, "SUCCESS", 22},
		{"gpt-4o", "Cache", "Explain the circuit breaker pattern in distributed systems", 132, 1, 0.0, 0.0035, true, false, "CACHED", 19},
		{"gemini-1.5-pro", "Google Gemini", "How does token bucket rate limiting prevent bursts?", 420, 142, 0.0016, 0.0, false, false, "SUCCESS", 16},
		{"deepseek-chat", "DeepSeek", "Write a Go worker pool with waitgroups", 490, 96, 0.0003, 0.0, false, false, "SUCCESS", 14},
		{"gpt-4o", "Anthropic", "Analyze transaction matching latency (OpenAI 429)", 580, 265, 0.0075, 0.0, false, true, "FALLBACK", 12},
		{"claude-3-5-sonnet", "Cache", "What is the capital of France?", 16, 1, 0.0, 0.00018, true, false, "CACHED", 10},
		{"gpt-4o", "OpenAI", "Explain Kubernetes control plane components", 610, 245, 0.0092, 0.0, false, false, "SUCCESS", 8},
		{"deepseek-chat", "DeepSeek", "Calculate 1337 * 42 arithmetic", 64, 88, 0.0001, 0.0, false, false, "SUCCESS", 6},
		{"gemini-1.5-pro", "Google Gemini", "What is CAP theorem consistency vs availability?", 380, 138, 0.0014, 0.0, false, false, "SUCCESS", 5},
		{"gpt-4o", "OpenAI", "Reverse string algorithm in Go with Unicode runes", 280, 212, 0.0039, 0.0, false, false, "SUCCESS", 4},
		{"claude-3-5-sonnet", "Anthropic", "Draft architecture review meeting agenda", 390, 175, 0.0038, 0.0, false, false, "SUCCESS", 3},
		{"gpt-4o", "OpenAI", "Explain self-attention mechanism in Transformers", 720, 310, 0.0112, 0.0, false, false, "SUCCESS", 2},
		{"deepseek-chat", "DeepSeek", "Write a REST HTTP handler with JSON decoding", 340, 94, 0.0002, 0.0, false, false, "SUCCESS", 1},
		{"gpt-4o", "Cache", "Explain the circuit breaker pattern in distributed systems", 132, 1, 0.0, 0.0035, true, false, "CACHED", 0},
	}

	for i, entry := range historicalLogs {
		logEntry := &metrics.RequestLog{
			ID:            fmt.Sprintf("req-seed-%03d", i+1),
			Timestamp:     now.Add(-time.Duration(entry.agoMin) * time.Minute),
			TenantID:      "tenant-default",
			Model:         entry.model,
			Provider:      entry.provider,
			PromptSummary: entry.prompt,
			TotalTokens:   entry.tokens,
			LatencyMs:     entry.latMs,
			CostUSD:       entry.cost,
			CostSavedUSD:  entry.saved,
			FromCache:     entry.cached,
			WasFallback:   entry.fallback,
			Status:        entry.status,
		}
		t.RecordRequest(logEntry)
	}

	// Update tenant usage baseline
	l.RecordUsage("nx-key-demo-secret", 3200, 2400, 5600, 0.062)
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// ── Dashboard UI ──────────────────────────────────────────────────────────────

func (s *Server) handleDashboardUI(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_, _ = w.Write([]byte(dashboardHTML))
}

const dashboardHTML = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover">
<meta name="theme-color" content="#07090e">
<title>Nexus Gateway · Enterprise AI Mesh</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#07090e;
  --surface:#0a0e17;
  --panel:#0f1523;
  --panel-hover:#141c2e;
  --card:#111828;
  --card-subtle:#0d1320;
  --border:rgba(255,255,255,0.07);
  --border-subtle:rgba(255,255,255,0.035);
  --border-strong:rgba(255,255,255,0.14);
  --text-main:#f8fafc;
  --text-secondary:#cbd5e1;
  --text-muted:#8492a6;
  --text-dim:#526075;

  /* Primary action: calm, authoritative steel blue */
  --primary:#2563eb;
  --primary-hover:#1d4ed8;
  --primary-glow:rgba(37,99,235,0.20);

  /* Quiet Healthy: subdued slate with emerald indicator */
  --healthy-bg:rgba(255,255,255,0.03);
  --healthy-border:rgba(255,255,255,0.08);
  --healthy-text:#94a3b8;
  --healthy-dot:#10b981;

  /* Outage / Failover Accent: clear amber */
  --amber:#f59e0b;
  --amber-hero:#fbbf24;
  --amber-bg:rgba(245,158,11,0.09);
  --amber-border:rgba(245,158,11,0.32);

  /* Critical / Error */
  --rose:#ef4444;
  --rose-bg:rgba(239,68,68,0.09);
  --rose-border:rgba(239,68,68,0.25);

  --cyan:#0ea5e9;
  --emerald:#10b981;
  --violet:#8b5cf6;

  --r-xs:3px;
  --r-sm:5px;
  --r-md:8px;
  --r-lg:12px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:var(--bg);
  color:var(--text-main);
  height:100vh;
  display:flex;
  flex-direction:column;
  overflow:hidden;
  -webkit-font-smoothing:antialiased;
}

/* ── Top Bar ── */
.topbar{
  height:48px;
  background:rgba(8,11,18,0.97);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 18px;
  position:sticky;
  top:0;
  z-index:100;
  flex-shrink:0;
}
.brand{display:flex;align-items:center;gap:10px;text-decoration:none;}
.brand-glyph{
  width:28px;height:28px;border-radius:var(--r-xs);
  background:rgba(37,99,235,0.08);border:1px solid rgba(59,130,246,0.3);
  display:flex;align-items:center;justify-content:center;
  transition:all 0.15s ease;
}
.brand:hover .brand-glyph{border-color:var(--primary);background:rgba(37,99,235,0.16);}
.brand-text-col{display:flex;flex-direction:column;gap:1px;}
.brand-title{
  font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:13px;
  letter-spacing:0.6px;color:#fff;display:flex;align-items:center;gap:4px;line-height:1.2;
}
.brand-subtext{font-weight:600;color:#60a5fa;letter-spacing:0.6px;}
.brand-meta{
  font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:600;
  letter-spacing:0.5px;color:var(--text-dim);line-height:1;display:flex;align-items:center;gap:5px;
}
.meta-dot{display:inline-block;width:5px;height:5px;border-radius:50%;background:#10b981;box-shadow:0 0 5px rgba(16,185,129,0.7);}

/* Clean text-only navigation tabs */
.nav{
  display:flex;gap:3px;background:rgba(255,255,255,0.02);padding:2px;
  border-radius:var(--r-sm);border:1px solid var(--border);
  overflow-x:auto;white-space:nowrap;scrollbar-width:none;
}
.nav::-webkit-scrollbar{display:none;}
.nav-tab{
  padding:5px 12px;border-radius:var(--r-xs);font-size:12px;font-weight:500;
  color:var(--text-muted);background:transparent;border:1px solid transparent;cursor:pointer;
  transition:all 0.12s ease;white-space:nowrap;
}
.nav-tab:hover{color:var(--text-main);background:rgba(255,255,255,0.035);}
.nav-tab.active{
  background:var(--card);color:#fff;font-weight:600;
  border-color:var(--border-strong);box-shadow:0 1px 3px rgba(0,0,0,0.4);
}

.top-right{display:flex;align-items:center;gap:10px;}
.live-badge{
  display:flex;align-items:center;gap:6px;
  padding:3px 8px;border-radius:var(--r-xs);
  background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.2);
  font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.4px;color:#34d399;
}
.live-dot{width:5px;height:5px;border-radius:50%;background:#10b981;box-shadow:0 0 5px rgba(16,185,129,0.8);animation:beaconPulse 2s infinite;}
@keyframes beaconPulse{0%,100%{opacity:1;}50%{opacity:0.35;}}

/* ── Pages Layout ── */
.main-view{flex:1;display:flex;flex-direction:column;position:relative;overflow:hidden;}
.tab-content{display:none;flex:1;height:calc(100vh - 48px);overflow:hidden;}
.tab-content.active{display:flex;}

/* ── Playground Layout ── */
.play-layout{display:flex;flex:1;width:100%;height:100%;position:relative;}
.sidebar-panel{
  width:280px;flex-shrink:0;border-right:1px solid var(--border);
  background:var(--surface);display:flex;flex-direction:column;
  overflow-y:auto;z-index:20;
}
.side-block{padding:12px 14px;border-bottom:1px solid var(--border);}
.side-title{
  font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:700;
  letter-spacing:0.6px;text-transform:uppercase;color:var(--text-dim);
  margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;
}
.side-badge{
  font-size:8.5px;font-family:'JetBrains Mono',monospace;font-weight:600;
  background:rgba(255,255,255,0.04);border:1px solid var(--border);padding:1px 5px;border-radius:2px;color:var(--text-dim);
}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:4px 0;font-size:11.5px;border-bottom:1px solid rgba(255,255,255,0.02);}
.metric-row:last-child{border-bottom:none;}
.metric-label{color:var(--text-muted);font-size:11px;}
.metric-val{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:12px;color:var(--text-main);}

/* Structured Provider Pills in Sidebar */
.provider-pill{
  display:flex;flex-direction:column;gap:5px;
  padding:8px 10px;border-radius:var(--r-xs);background:var(--panel);
  border:1px solid var(--border);margin-bottom:6px;transition:border-color 0.15s;
}
.provider-pill:last-child{margin-bottom:0;}
.provider-pill:hover{border-color:var(--border-strong);}
.provider-pill.pill-active-alert{border-color:var(--amber-border);background:rgba(245,158,11,0.04);}
.pill-top{display:flex;justify-content:space-between;align-items:center;}
.pill-name{font-weight:600;color:var(--text-main);font-size:12px;}
.pill-status{
  font-size:8.5px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.4px;
  padding:2px 6px;border-radius:var(--r-xs);text-transform:uppercase;display:flex;align-items:center;gap:4px;
}
.pill-closed{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.pill-closed::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#10b981;}
.pill-open{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);}
.pill-half{background:rgba(245,158,11,0.08);color:var(--amber);border:1px solid rgba(245,158,11,0.25);}
.pill-metrics{
  display:flex;align-items:center;gap:6px;font-size:10px;
  font-family:'JetBrains Mono',monospace;
}
.pm-item{display:flex;align-items:center;gap:3px;}
.pm-lbl{color:var(--text-dim);font-weight:600;}
.pm-val{color:var(--text-secondary);font-weight:600;}
.pm-sep{color:var(--text-dim);opacity:0.4;}

/* Fallback order block */
.fallback-sideblock{background:rgba(37,99,235,0.02);}
.fallback-hdr{display:flex;align-items:center;gap:6px;font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--primary);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:6px;}
.fallback-chain-flow{display:flex;align-items:center;gap:4px;font-size:10px;font-family:'JetBrains Mono',monospace;flex-wrap:wrap;}
.flow-node{padding:2px 6px;background:var(--panel);border:1px solid var(--border);border-radius:2px;color:var(--text-secondary);}
.flow-node.active{border-color:rgba(59,130,246,0.35);color:#93c5fd;}
.flow-arrow{color:var(--text-dim);}

/* Mobile drawer backdrop */
.drawer-backdrop{
  display:none;position:fixed;inset:48px 0 0 0;
  background:rgba(0,0,0,0.65);backdrop-filter:blur(3px);z-index:490;
}
.drawer-backdrop.active{display:block;}

/* Chat Central */
.chat-center{flex:1;display:flex;flex-direction:column;background:var(--bg);height:100%;overflow:hidden;}
.chat-header{
  padding:8px 16px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface);flex-shrink:0;gap:8px;
}
.chat-header-left{display:flex;align-items:center;gap:8px;}
.drawer-toggle-btn{
  display:none;align-items:center;gap:5px;
  padding:4px 8px;border-radius:var(--r-xs);
  background:var(--panel);border:1px solid var(--border-strong);
  color:var(--text-secondary);font-size:11px;font-weight:600;cursor:pointer;
}
.drawer-toggle-btn:hover{background:var(--panel-hover);color:#fff;}
.model-picker{display:flex;align-items:center;gap:6px;}
.picker-lbl{font-size:10.5px;font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;}
.select-input{
  background:var(--panel);color:#fff;border:1px solid var(--border-strong);
  padding:4px 8px;border-radius:var(--r-xs);font-size:11.5px;font-weight:600;
  outline:none;cursor:pointer;transition:border-color 0.15s;
}
.select-input:focus{border-color:var(--primary);}
.chat-controls{display:flex;align-items:center;gap:12px;}
.toggle-wrap{
  display:flex;align-items:center;gap:5px;font-size:11px;font-weight:500;
  color:var(--text-muted);cursor:pointer;user-select:none;
}
.switch-track{
  width:28px;height:15px;border-radius:99px;background:var(--panel);
  border:1px solid var(--border-strong);position:relative;transition:all 0.15s;
}
.switch-track.on{background:var(--primary);border-color:transparent;}
.switch-thumb{
  width:9px;height:9px;border-radius:50%;background:#fff;
  position:absolute;top:2px;left:2px;transition:all 0.15s;
}
.switch-track.on .switch-thumb{transform:translateX(13px);}

/* Message Thread */
.thread{flex:1;overflow-y:auto;padding:12px 18px;display:flex;flex-direction:column;gap:8px;}
.msg{display:flex;flex-direction:column;gap:3px;animation:msgIn 0.12s ease-out;}
@keyframes msgIn{from{opacity:0;transform:translateY(2px);}to{opacity:1;transform:translateY(0);}}
.msg.user{align-self:flex-end;max-width:82%;}
.msg.bot{align-self:stretch;}

.msg.user .user-meta{
  font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.4px;
  color:var(--text-dim);text-align:right;
}
.msg.user .bubble{
  background:rgba(37,99,235,0.08);border:1px solid rgba(59,130,246,0.25);
  border-left:2px solid var(--primary);
  color:#fff;padding:8px 12px;border-radius:var(--r-xs);font-size:13px;line-height:1.52;
  white-space:pre-wrap;word-break:break-word;
}

/* Bot Console Card */
.bot-card{
  background:var(--surface);border:1px solid var(--border);
  border-left:2px solid rgba(255,255,255,0.18);
  border-radius:var(--r-xs);overflow:hidden;transition:border-color 0.15s;
}
.bot-card:hover{border-color:var(--border-strong);}
.bot-header{
  padding:6px 12px;background:rgba(255,255,255,0.015);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;
}
.bot-identity{display:flex;align-items:center;gap:6px;}
.provider-badge{
  font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace;
  color:#fff;display:flex;align-items:center;gap:5px;
}
.provider-badge::before{
  content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#10b981;
}
.tag-badge{
  font-size:8.5px;font-family:'JetBrains Mono',monospace;font-weight:700;
  letter-spacing:0.4px;padding:2px 5px;border-radius:var(--r-xs);text-transform:uppercase;
}
.tag-direct{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.tag-stream{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.tag-cache{background:rgba(14,165,233,0.09);color:var(--cyan);border:1px solid rgba(14,165,233,0.22);}
.tag-fallback{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);font-weight:700;}

.bot-telemetry-meta{
  display:flex;align-items:center;gap:6px;font-size:10.5px;font-family:'JetBrains Mono',monospace;
  color:var(--text-muted);
}
.meta-lbl{color:var(--text-dim);font-weight:600;}
.meta-sep{color:var(--text-dim);opacity:0.4;}
.bot-body{
  padding:11px 14px;font-size:13px;line-height:1.56;color:var(--text-main);
  white-space:pre-wrap;word-break:break-word;
}
.card-spec-strip{
  padding:5px 12px;background:rgba(255,255,255,0.012);border-bottom:1px solid var(--border-subtle);
  display:flex;align-items:center;gap:12px;font-size:10px;font-family:'JetBrains Mono',monospace;
  color:var(--text-dim);flex-wrap:wrap;
}
.card-spec-strip span{display:flex;align-items:center;gap:4px;}

/* Failover & Redaction notifications */
.cascade-alert{
  background:var(--amber-bg);border-bottom:1px solid var(--amber-border);
  padding:6px 12px;font-size:11px;font-weight:600;color:var(--amber-hero);
  display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;
}
.sanitized-alert{
  background:rgba(255,255,255,0.02);border-bottom:1px solid var(--border);
  padding:5px 12px;font-size:10.5px;font-weight:500;color:var(--text-muted);
  display:flex;align-items:center;gap:5px;font-family:'JetBrains Mono',monospace;
}

/* Command Center Dock */
.console-dock{
  border-top:1px solid var(--border);background:var(--surface);
  display:flex;flex-direction:column;flex-shrink:0;
}
.dock-quick-prompts{
  padding:5px 14px;border-bottom:1px solid var(--border-subtle);
  display:flex;align-items:center;gap:5px;overflow-x:auto;scrollbar-width:none;
}
.dock-quick-prompts::-webkit-scrollbar{display:none;}
.dock-prompts-label{
  font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:600;
  color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;margin-right:2px;white-space:nowrap;
}
.prompt-chip{
  flex-shrink:0;padding:3px 7px;border-radius:var(--r-xs);
  background:var(--panel);border:1px solid var(--border);
  font-size:11px;font-weight:500;color:var(--text-muted);cursor:pointer;
  transition:all 0.12s;white-space:nowrap;
}
.prompt-chip:hover{border-color:var(--border-strong);color:#fff;background:var(--panel-hover);}

.dock-input-row{
  padding:8px 14px 10px 14px;display:flex;gap:8px;align-items:flex-end;
}
.input-wrap{flex:1;display:flex;flex-direction:column;gap:3px;}
.text-area{
  width:100%;background:var(--panel);border:1px solid var(--border-strong);
  border-radius:var(--r-xs);padding:8px 12px;color:#fff;
  font-family:'Inter',sans-serif;font-size:13px;resize:none;
  outline:none;line-height:1.45;min-height:38px;max-height:110px;
  transition:border-color 0.15s;
}
.text-area:focus{border-color:var(--primary);box-shadow:0 0 0 1px var(--primary);}
.input-hint{font-size:9.5px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);display:flex;justify-content:space-between;align-items:center;}
.route-indicator-pill{font-weight:600;color:#60a5fa;}
.send-btn{
  height:38px;padding:0 15px;border-radius:var(--r-xs);
  background:var(--primary);color:#fff;font-family:'Plus Jakarta Sans',sans-serif;
  font-size:12px;font-weight:600;border:none;cursor:pointer;
  transition:all 0.12s;display:flex;align-items:center;gap:6px;flex-shrink:0;
}
.send-btn:hover{background:var(--primary-hover);}
.send-btn:disabled{opacity:0.4;cursor:not-allowed;}
.btn-kbd{
  font-size:10px;font-family:'JetBrains Mono',monospace;background:rgba(255,255,255,0.16);
  padding:1px 4px;border-radius:2px;font-weight:700;
}

/* ── Resilience Page ── */
.resilience-view{padding:18px 24px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;height:100%;}
.page-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:17px;font-weight:700;letter-spacing:-0.3px;color:#fff;}
.page-desc{font-size:12px;color:var(--text-muted);margin-top:2px;line-height:1.45;}

/* Visual Cascade Diagram */
.cascade-diagram{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-xs);padding:12px 16px;
}
.diag-title{font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.6px;text-transform:uppercase;color:var(--text-dim);margin-bottom:8px;}
.diag-nodes{display:flex;align-items:center;gap:8px;overflow-x:auto;-webkit-overflow-scrolling:touch;}
.diag-node{
  padding:8px 12px;border-radius:var(--r-xs);background:var(--panel);
  border:1px solid var(--border);display:flex;flex-direction:column;gap:2px;
  min-width:150px;position:relative;transition:all 0.15s;
}
.diag-node.node-open{border-color:var(--amber);background:var(--amber-bg);}
.diag-node-name{font-weight:600;font-size:12.5px;color:#fff;}
.diag-node-role{font-size:9.5px;color:var(--text-muted);font-family:'JetBrains Mono',monospace;}
.diag-node-badge{
  position:absolute;top:6px;right:6px;font-size:7.5px;font-family:'JetBrains Mono',monospace;font-weight:700;
  padding:1px 4px;border-radius:2px;text-transform:uppercase;
}
.diag-arrow{color:var(--text-dim);font-size:13px;}

/* Provider Cards Grid */
.cards-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;}
.p-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-xs);padding:14px 16px;display:flex;flex-direction:column;
  gap:10px;transition:all 0.15s;position:relative;
}
.p-card:hover{border-color:var(--border-strong);}
.p-card.card-tripped{border-color:var(--amber-border);background:rgba(245,158,11,0.03);box-shadow:0 0 16px rgba(245,158,11,0.06);}

.card-header{display:flex;justify-content:space-between;align-items:flex-start;}
.card-name{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:13.5px;color:#fff;}
.card-meta{font-size:10px;color:var(--text-muted);margin-top:1px;font-family:'JetBrains Mono',monospace;}

.card-state{
  font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.3px;
  padding:3px 7px;border-radius:var(--r-xs);display:flex;align-items:center;gap:4px;
}
.state-closed{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.state-closed::before{content:'●';color:#10b981;font-size:9px;}
.state-open{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);font-weight:700;}
.state-open::before{content:'▲';color:var(--amber-hero);font-size:8px;}
.state-half{background:rgba(245,158,11,0.08);color:var(--amber);border:1px solid rgba(245,158,11,0.25);}
.state-half::before{content:'◌';color:var(--amber);font-size:9px;}

.card-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;background:rgba(255,255,255,0.015);padding:8px 10px;border-radius:var(--r-xs);border:1px solid var(--border-subtle);}
.stat-box{display:flex;flex-direction:column;gap:1px;}
.stat-lbl{font-size:8.5px;font-family:'JetBrains Mono',monospace;font-weight:600;text-transform:uppercase;letter-spacing:0.4px;color:var(--text-dim);}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:12.5px;font-weight:600;color:#fff;}

.error-banner{
  background:var(--amber-bg);border:1px solid var(--amber-border);
  border-radius:var(--r-xs);padding:6px 9px;font-size:10.5px;font-family:'JetBrains Mono',monospace;
  color:var(--amber-hero);line-height:1.35;
}
.healthy-banner{
  font-size:10.5px;font-family:'JetBrains Mono',monospace;color:var(--text-muted);padding:1px 0;display:flex;align-items:center;gap:5px;
}
.healthy-banner::before{content:'✓';color:#10b981;}
.countdown-box{
  font-size:10.5px;font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--amber-hero);display:flex;align-items:center;gap:5px;
}

.fault-btn{
  padding:7px 12px;border-radius:var(--r-xs);border:none;cursor:pointer;
  font-family:'Inter',sans-serif;font-size:11.5px;font-weight:600;
  display:flex;align-items:center;justify-content:center;gap:5px;min-height:36px;
  transition:all 0.12s;
}
.btn-trip{background:transparent;color:var(--text-muted);border:1px solid var(--border-strong);}
.btn-trip:hover{background:var(--amber-bg);color:var(--amber-hero);border-color:var(--amber-border);}
.btn-restore{background:var(--amber);color:#080a0f;font-weight:700;}
.btn-restore:hover{background:var(--amber-hero);}

/* ── Telemetry Page ── */
.telemetry-view{padding:18px 24px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;height:100%;}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;}
.kpi-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-xs);
  padding:12px 14px;display:flex;flex-direction:column;gap:3px;
}
.kpi-label{font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.6px;text-transform:uppercase;color:var(--text-dim);}
.kpi-val{font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:700;letter-spacing:-0.4px;color:#fff;}
.kpi-sub{font-size:10.5px;color:var(--text-muted);}

.tail-latency-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;}
.lat-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-xs);
  padding:12px;text-align:center;
}
.lat-title{font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.6px;text-transform:uppercase;color:var(--text-dim);margin-bottom:3px;}
.lat-metric{font-family:'Plus Jakarta Sans',sans-serif;font-size:24px;font-weight:700;letter-spacing:-0.5px;color:#fff;}
.lat-unit{font-size:11px;font-weight:500;color:var(--text-muted);margin-left:2px;}

.chart-section{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-xs);padding:14px 16px;}
.chart-title{font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.6px;text-transform:uppercase;color:var(--text-dim);margin-bottom:10px;}
.bar-chart-wrap{border-bottom:1px solid var(--border-strong);padding-bottom:4px;}
.bar-chart{display:flex;align-items:flex-end;gap:5px;height:70px;}
.bar-col{flex:1;display:flex;flex-direction:column;align-items:center;gap:3px;cursor:pointer;}
.bar-fill{width:100%;border-radius:2px 2px 0 0;min-height:4px;transition:height 0.25s ease, opacity 0.15s;}
.bar-col:hover .bar-fill{opacity:0.8;}
.bar-lbl{font-size:8.5px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);margin-top:4px;}

/* ── Tables ── */
.table-wrap{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-xs);
  overflow-x:auto;-webkit-overflow-scrolling:touch;
}
table{width:100%;min-width:680px;border-collapse:collapse;font-size:12px;}
th{
  text-align:left;padding:8px 12px;font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.5px;
  text-transform:uppercase;color:var(--text-dim);border-bottom:1px solid var(--border);
  background:rgba(255,255,255,0.015);
}
td{padding:8px 12px;color:var(--text-muted);border-bottom:1px solid rgba(255,255,255,0.025);vertical-align:middle;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:rgba(255,255,255,0.015);}
td.td-pri{color:#fff;font-weight:600;}
td.td-mono{font-family:'JetBrains Mono',monospace;font-size:11px;}

/* Status badges */
.status-pill{
  display:inline-block;padding:2px 6px;border-radius:var(--r-xs);font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.3px;
}
.st-success{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.st-cached{background:rgba(14,165,233,0.09);color:var(--cyan);border:1px solid rgba(14,165,233,0.22);}
.st-fallback{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);}
.st-error{background:var(--rose-bg);color:var(--rose);border:1px solid var(--rose-border);}

/* Toast */
#toast{
  position:fixed;bottom:18px;right:18px;z-index:999;padding:8px 14px;
  border-radius:var(--r-xs);background:var(--panel);border:1px solid var(--border-strong);
  font-size:11.5px;font-family:'JetBrains Mono',monospace;font-weight:600;color:#fff;box-shadow:0 8px 24px rgba(0,0,0,0.6);
  transform:translateY(12px);opacity:0;pointer-events:none;transition:all 0.15s cubic-bezier(0.16,1,0.3,1);
}
#toast.show{transform:translateY(0);opacity:1;}

/* Responsive Media Queries */
@media (max-width: 900px) {
  .drawer-toggle-btn{display:flex;}
  .sidebar-panel{
    position:fixed;top:48px;left:0;bottom:0;width:290px;max-width:85vw;
    z-index:500;transform:translateX(-100%);transition:transform 0.22s cubic-bezier(0.16,1,0.3,1);
    box-shadow:6px 0 28px rgba(0,0,0,0.7);border-right:1px solid var(--border-strong);
  }
  .sidebar-panel.drawer-open{transform:translateX(0);}
  .cards-grid{grid-template-columns:1fr;}
  .kpi-row{grid-template-columns:repeat(2,1fr);}
  .tail-latency-grid{grid-template-columns:1fr;}
  .resilience-view, .telemetry-view{padding:14px 16px;}
  .chat-header{padding:8px 12px;}
}

@media (max-width: 640px) {
  .brand-meta{display:none;}
  .topbar{padding:0 10px;}
  .kpi-row{grid-template-columns:1fr;}
  .dock-input-row{padding:8px 10px;}
  .thread{padding:10px 12px;}
}
</style>
</head>
<body>

<!-- ── Topbar ── -->
<header class="topbar">
  <a class="brand" href="#">
    <div class="brand-glyph">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <polygon points="12,2 21,7.2 21,16.8 12,22 3,16.8 3,7.2" stroke="#3b82f6" stroke-width="1.5" stroke-linejoin="round" fill="rgba(37,99,235,0.08)"/>
        <polygon points="12,6 17.2,12 12,18 6.8,12" stroke="#60a5fa" stroke-width="1.2" fill="none"/>
        <circle cx="12" cy="12" r="2" fill="#60a5fa"/>
      </svg>
    </div>
    <div class="brand-text-col">
      <div class="brand-title">NEXUS <span class="brand-subtext">GATEWAY</span></div>
      <div class="brand-meta"><span class="meta-dot"></span>v1.4.2 &middot; us-east-1</div>
    </div>
  </a>
  <nav class="nav">
    <button class="nav-tab active" id="tab-play" onclick="switchTab('play')">Playground</button>
    <button class="nav-tab" id="tab-res" onclick="switchTab('res')">Resilience</button>
    <button class="nav-tab" id="tab-tel" onclick="switchTab('tel')">Telemetry</button>
    <button class="nav-tab" id="tab-logs" onclick="switchTab('logs')">Request Logs</button>
    <button class="nav-tab" id="tab-ten" onclick="switchTab('ten')">Tenants</button>
  </nav>
  <div class="top-right">
    <div class="live-badge"><div class="live-dot"></div><span>Live</span></div>
  </div>
</header>

<main class="main-view">

  <!-- Mobile Drawer Backdrop -->
  <div class="drawer-backdrop" id="drawer-backdrop" onclick="toggleSidebar()"></div>

  <!-- ── PLAYGROUND TAB ── -->
  <section class="tab-content active" id="view-play">
    <div class="play-layout">
      <!-- Left sidebar -->
      <aside class="sidebar-panel">
        <div class="side-block">
          <div class="side-title"><span>System Telemetry</span><span class="side-badge">Realtime</span></div>
          <div class="metric-row"><span class="metric-label">P50 Latency</span><span class="metric-val" id="sb-p50">175ms</span></div>
          <div class="metric-row"><span class="metric-label">P95 Latency</span><span class="metric-val" id="sb-p95">310ms</span></div>
          <div class="metric-row"><span class="metric-label">P99 Latency</span><span class="metric-val" id="sb-p99">345ms</span></div>
          <div class="metric-row"><span class="metric-label">Cache Hit Ratio</span><span class="metric-val" id="sb-cache" style="color:var(--cyan)">16.7%</span></div>
          <div class="metric-row"><span class="metric-label">Cost Saved</span><span class="metric-val" id="sb-saved" style="color:var(--emerald)">$0.0126</span></div>
        </div>
        <div class="side-block" style="flex:1">
          <div class="side-title"><span>Upstream Providers</span><span class="side-badge">4 Nodes</span></div>
          <div id="sb-providers">
            <!-- Populated by JS -->
          </div>
        </div>
        <div class="side-block fallback-sideblock">
          <div class="fallback-hdr">
            <span>Failover Chain</span>
          </div>
          <div class="fallback-chain-flow">
            <span class="flow-node active">OpenAI</span>
            <span class="flow-arrow">&rarr;</span>
            <span class="flow-node">Anthropic</span>
            <span class="flow-arrow">&rarr;</span>
            <span class="flow-node">Gemini</span>
            <span class="flow-arrow">&rarr;</span>
            <span class="flow-node">DeepSeek</span>
          </div>
        </div>
      </aside>

      <!-- Chat area -->
      <div class="chat-center">
        <div class="chat-header">
          <div class="chat-header-left">
            <button class="drawer-toggle-btn" id="drawer-toggle" onclick="toggleSidebar()">
              <span>☰</span><span>Mesh Nodes</span>
            </button>
            <div class="model-picker">
              <span class="picker-lbl">Model:</span>
              <select class="select-input" id="model-select">
                <option value="gpt-4o">OpenAI · GPT-4o (Primary)</option>
                <option value="claude-3-5-sonnet">Anthropic · Claude 3.5 Sonnet</option>
                <option value="gemini-1.5-pro">Google · Gemini 1.5 Pro</option>
                <option value="deepseek-chat">DeepSeek · V3</option>
              </select>
            </div>
          </div>
          <div class="chat-controls">
            <!-- Structured Mode Toggle -->
            <label class="toggle-wrap" title="Output summary and detail layout">
              <span>Structured</span>
              <div class="switch-track" id="sw-struct" onclick="toggleStructured()"><div class="switch-thumb"></div></div>
            </label>
            <!-- Stream Toggle -->
            <label class="toggle-wrap" title="SSE token streaming">
              <span>Streaming</span>
              <div class="switch-track" id="sw-stream" onclick="toggleStreaming()"><div class="switch-thumb"></div></div>
            </label>
          </div>
        </div>

        <div class="thread" id="chat-thread">
          <div class="msg bot">
            <div class="bot-card">
              <div class="bot-header">
                <div class="bot-identity">
                  <span class="provider-badge">Nexus Gateway · Router</span>
                  <span class="tag-badge tag-direct">Direct</span>
                </div>
                <div class="bot-telemetry-meta">
                  <span><span class="meta-lbl">Tenant:</span> tenant-default</span>
                  <span class="meta-sep">&middot;</span>
                  <span><span class="meta-lbl">Tier:</span> Enterprise</span>
                  <span class="meta-sep">&middot;</span>
                  <span><span class="meta-lbl">PII Filter:</span> Active</span>
                </div>
              </div>
              <div class="card-spec-strip">
                <span><span>Token-Bucket:</span> 25 RPS / 100k TPM</span>
                <span>&middot;</span>
                <span><span>Prompt Cache:</span> SHA-256 (15m TTL)</span>
                <span>&middot;</span>
                <span><span>Circuit Breaker:</span> Active Cascade</span>
              </div>
              <div class="bot-body">Production AI Gateway operational. Ingress requests are authenticated, rate-limited via token-bucket, sanitized for PII, evaluated against SHA-256 prompt cache, and routed across upstream providers with automatic circuit breaker failover.

Select a target model above or click a benchmark query below. To test live failover, simulate an upstream 429 outage on OpenAI from the <strong>Resilience</strong> tab.</div>
            </div>
          </div>
        </div>

        <!-- Command Dock -->
        <div class="console-dock">
          <div class="dock-quick-prompts">
            <span class="dock-prompts-label">Benchmarks:</span>
            <button class="prompt-chip" onclick="setPrompt('Calculate 1337 * 42')">1337 &times; 42</button>
            <button class="prompt-chip" onclick="setPrompt('Explain how circuit breakers work')">Circuit Breakers</button>
            <button class="prompt-chip" onclick="setPrompt('Write a Go worker pool with waitgroups')">Worker Pool</button>
            <button class="prompt-chip" onclick="setPrompt('What is the capital of France?')">Capital of France</button>
            <button class="prompt-chip" onclick="setPrompt('Who is the richest man on earth?')">Richest Person</button>
            <button class="prompt-chip" onclick="setPrompt('Who is the richest footballer?')">Richest Footballer</button>
            <button class="prompt-chip" onclick="setPrompt('Write a short love letter')">Short Love Letter</button>
            <button class="prompt-chip" onclick="setPrompt('Compare Go vs Python for backend systems')">Go vs Python</button>
          </div>
          <div class="dock-input-row">
            <div class="input-wrap">
              <textarea class="text-area" id="prompt-input" placeholder="Enter prompt or query (e.g. explain circuit breakers, who is the richest footballer)..." rows="1" onkeydown="handleKey(event)"></textarea>
              <div class="input-hint">
                <span>Press <kbd style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:2px;">↵ Enter</kbd> to execute &middot; <kbd style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:2px;">Shift+↵</kbd> multiline</span>
                <span id="route-indicator" class="route-indicator-pill">Route: Direct (OpenAI · GPT-4o)</span>
              </div>
            </div>
            <button class="send-btn" id="send-button" onclick="sendPrompt()">
              <span>Execute</span>
              <span class="btn-kbd">↵</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ── RESILIENCE TAB ── -->
  <section class="tab-content" id="view-res">
    <div class="resilience-view">
      <div>
        <h1 class="page-title">Resilience &amp; Fault Injection</h1>
        <p class="page-desc">Simulate upstream provider outages (HTTP 429 rate limits, timeouts) and observe automatic fallback cascades with zero client errors.</p>
      </div>

      <!-- Cascade Diagram -->
      <div class="cascade-diagram">
        <div class="diag-title">Failover Cascade Pipeline</div>
        <div class="diag-nodes" id="cascade-nodes">
          <div class="diag-node" id="node-OpenAI">
            <span class="diag-node-name">OpenAI</span>
            <span class="diag-node-role">Primary &middot; GPT-4o</span>
            <span class="diag-node-badge state-closed" id="badge-OpenAI">Healthy</span>
          </div>
          <div class="diag-arrow">&rarr;</div>
          <div class="diag-node" id="node-Anthropic">
            <span class="diag-node-name">Anthropic</span>
            <span class="diag-node-role">Fallback 1 &middot; Claude 3.5 Sonnet</span>
            <span class="diag-node-badge state-closed" id="badge-Anthropic">Healthy</span>
          </div>
          <div class="diag-arrow">&rarr;</div>
          <div class="diag-node" id="node-GoogleGemini">
            <span class="diag-node-name">Google Gemini</span>
            <span class="diag-node-role">Fallback 2 &middot; Gemini 1.5 Pro</span>
            <span class="diag-node-badge state-closed" id="badge-GoogleGemini">Healthy</span>
          </div>
          <div class="diag-arrow">&rarr;</div>
          <div class="diag-node" id="node-DeepSeek">
            <span class="diag-node-name">DeepSeek</span>
            <span class="diag-node-role">Fallback 3 &middot; DeepSeek V3</span>
            <span class="diag-node-badge state-closed" id="badge-DeepSeek">Healthy</span>
          </div>
        </div>
      </div>

      <!-- Provider cards -->
      <div class="cards-grid" id="provider-cards">
        <!-- Populated by JS -->
      </div>
    </div>
  </section>

  <!-- ── TELEMETRY TAB ── -->
  <section class="tab-content" id="view-tel">
    <div class="telemetry-view">
      <div>
        <h1 class="page-title">Gateway Telemetry</h1>
        <p class="page-desc">Aggregated latency distributions, token counts, and cost telemetry. Prometheus metrics available at <code style="font-family:'JetBrains Mono',monospace;color:var(--cyan);background:var(--card);padding:2px 6px;border-radius:4px;">/metrics</code>.</p>
      </div>

      <div class="kpi-row">
        <div class="kpi-card"><span class="kpi-label">Total Requests</span><span class="kpi-val" id="tel-reqs">18</span><span class="kpi-sub">Gateway all-time</span></div>
        <div class="kpi-card"><span class="kpi-label">Total Tokens</span><span class="kpi-val" id="tel-toks">14,820</span><span class="kpi-sub">Prompt + completion</span></div>
        <div class="kpi-card"><span class="kpi-label">Cost Incurred</span><span class="kpi-val" id="tel-cost" style="color:var(--amber)">$0.0842</span><span class="kpi-sub">USD upstream billing</span></div>
        <div class="kpi-card"><span class="kpi-label">Cost Saved (Cache)</span><span class="kpi-val" id="tel-saved" style="color:var(--emerald)">$0.0126</span><span class="kpi-sub">SHA-256 prompt deduplication</span></div>
      </div>

      <div class="tail-latency-grid">
        <div class="lat-card"><div class="lat-title">P50 Primary Latency</div><div class="lat-metric" style="color:var(--emerald)"><span id="tel-p50">175</span><span class="lat-unit">ms</span></div><div style="font-size:10px;color:var(--text-dim);margin-top:4px;">Median response time</div></div>
        <div class="lat-card"><div class="lat-title">P95 Primary Latency</div><div class="lat-metric" style="color:var(--emerald)"><span id="tel-p95">310</span><span class="lat-unit">ms</span></div><div style="font-size:10px;color:var(--text-dim);margin-top:4px;">Clean non-fallback baseline</div></div>
        <div class="lat-card"><div class="lat-title">P99 Tail Latency</div><div class="lat-metric" style="color:var(--amber)"><span id="tel-p99">345</span><span class="lat-unit">ms</span></div><div style="font-size:10px;color:var(--text-dim);margin-top:4px;">Complex query &amp; cold network</div></div>
      </div>

      <div class="chart-section">
        <div class="chart-title">Recent Latency Distribution (ms)</div>
        <div class="bar-chart-wrap">
          <div class="bar-chart" id="latency-bars">
            <!-- Populated by JS -->
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ── LOGS TAB ── -->
  <section class="tab-content" id="view-logs">
    <div style="padding:18px 24px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;height:100%;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <h1 class="page-title">Gateway Request Logs</h1>
          <p class="page-desc">Audit log of direct, cached, and failover requests.</p>
        </div>
        <button class="send-btn" style="height:34px;padding:0 14px;font-size:11.5px;" onclick="loadLogs()">Refresh</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Timestamp</th><th>Model</th><th>Provider</th><th>Prompt</th><th>Tokens</th><th>Latency</th><th>Cost</th><th>Status</th></tr>
          </thead>
          <tbody id="logs-tbody">
            <!-- Populated by JS -->
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- ── TENANTS TAB ── -->
  <section class="tab-content" id="view-ten">
    <div style="padding:18px 24px;display:flex;flex-direction:column;gap:14px;overflow-y:auto;height:100%;">
      <div>
        <h1 class="page-title">Tenant Management &amp; Rate Limits</h1>
        <p class="page-desc">Multi-tenant isolation with token-bucket RPS, TPM quotas, and monthly dollar budget ceilings.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Tenant ID</th><th>Name</th><th>Tier</th><th>API Key</th><th>Requests</th><th>Tokens</th><th>Spend USD</th><th>RPS Limit</th><th>TPM Limit</th><th>Monthly Budget</th></tr>
          </thead>
          <tbody id="tenants-tbody">
            <!-- Populated by JS -->
          </tbody>
        </table>
      </div>
    </div>
  </section>

</main>

<div id="toast"></div>

<script>
let structuredMode = false;
let streamingMode = false;
let activeTab = 'play';
let countdownTimers = {};

function switchTab(tab) {
  document.querySelectorAll('.tab-content').forEach(e => e.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(e => e.classList.remove('active'));
  document.getElementById('view-' + tab).classList.add('active');
  document.getElementById('tab-' + tab).classList.add('active');
  activeTab = tab;
  closeSidebar();
  if (tab === 'res') loadResilience();
  if (tab === 'tel') loadTelemetry();
  if (tab === 'logs') loadLogs();
  if (tab === 'ten') loadTenants();
}

function toggleSidebar() {
  const sidebar = document.querySelector('.sidebar-panel');
  const backdrop = document.getElementById('drawer-backdrop');
  const isOpen = sidebar.classList.toggle('drawer-open');
  backdrop.classList.toggle('active', isOpen);
}

function closeSidebar() {
  const sidebar = document.querySelector('.sidebar-panel');
  const backdrop = document.getElementById('drawer-backdrop');
  sidebar.classList.remove('drawer-open');
  backdrop.classList.remove('active');
}

function toggleStructured() {
  structuredMode = !structuredMode;
  document.getElementById('sw-struct').classList.toggle('on', structuredMode);
  showToast(structuredMode ? 'Structured Mode enabled' : 'Structured Mode disabled (conversational)');
}

function toggleStreaming() {
  streamingMode = !streamingMode;
  document.getElementById('sw-stream').classList.toggle('on', streamingMode);
  showToast(streamingMode ? 'Streaming enabled' : 'Streaming disabled (buffered)');
}

function setPrompt(text) {
  const input = document.getElementById('prompt-input');
  input.value = text;
  input.focus();
}

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendPrompt();
  }
}

function showToast(msg) {
  const el = document.getElementById('toast');
  el.innerText = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2600);
}

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.innerText = s;
  return d.innerHTML;
}

// ── Send Prompt ──
async function sendPrompt() {
  const input = document.getElementById('prompt-input');
  const prompt = input.value.trim();
  if (!prompt) return;

  const model = document.getElementById('model-select').value;
  const btn = document.getElementById('send-button');
  const thread = document.getElementById('chat-thread');

  // Append user message
  const userMsg = document.createElement('div');
  userMsg.className = 'msg user';
  userMsg.innerHTML = '<div class="user-meta">User</div><div class="bubble">' + esc(prompt) + '</div>';
  thread.appendChild(userMsg);
  input.value = '';
  btn.disabled = true;

  // Append bot placeholder
  const botMsg = document.createElement('div');
  botMsg.className = 'msg bot';
  botMsg.innerHTML = '<div class="bot-card">'
    + '<div class="bot-header">'
    + '  <div class="bot-identity"><span class="provider-badge">Routing</span><span class="tag-badge tag-direct">Evaluating</span></div>'
    + '  <div class="bot-telemetry-meta"><span>evaluating...</span></div>'
    + '</div>'
    + '<div class="bot-body" style="color:var(--text-dim);font-family:\'JetBrains Mono\',monospace;font-size:12px;">Dispatching request...</div>'
    + '</div>';
  thread.appendChild(botMsg);
  thread.scrollTop = thread.scrollHeight;
  const botCard = botMsg.querySelector('.bot-card');
  const botBody = botMsg.querySelector('.bot-body');

  if (streamingMode) {
    try {
      const res = await fetch('/v1/chat/completions', {
        method: 'POST',
        headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer nx-key-demo-secret'},
        body: JSON.stringify({model, messages: [{role: 'user', content: prompt}], stream: true})
      });
      if (!res.ok) {
        botBody.innerText = 'Error: ' + (await res.text());
        return;
      }
      botBody.innerText = '';
      botBody.style.color = 'var(--text-main)';
      botBody.style.fontFamily = 'Inter, sans-serif';
      botBody.style.fontSize = '13.5px';
      const reader = res.body.getReader();
      const dec = new TextDecoder();
      let full = '';
      while (true) {
        const {done, value} = await reader.read();
        if (done) break;
        for (const line of dec.decode(value).split('\n')) {
          if (!line.startsWith('data: ')) continue;
          const d = line.slice(6);
          if (d === '[DONE]') break;
          try {
            const c = JSON.parse(d);
            const tk = c.choices?.[0]?.delta?.content || '';
            if (tk) {
              full += tk;
              botBody.innerText = full;
              thread.scrollTop = thread.scrollHeight;
            }
          } catch {}
        }
      }
      const headerMeta = botCard.querySelector('.bot-telemetry-meta');
      if (headerMeta) headerMeta.innerHTML = '<span>Stream Complete</span> &middot; <span>' + esc(model) + '</span>';
    } catch (e) {
      botBody.innerText = 'Connection error: ' + e;
    } finally {
      btn.disabled = false;
      refreshTelemetry();
    }
    return;
  }

  try {
    const res = await fetch('/api/v1/generate', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({model, prompt, structured: structuredMode})
    });
    if (!res.ok) {
      botBody.innerText = 'Gateway error: ' + (await res.text());
      return;
    }
    const data = await res.json();
    const resp = data.response;
    const san = data.sanitization;

    let routeTag = '<span class="tag-badge tag-direct">DIRECT</span>';
    if (resp.was_fallback) {
      routeTag = '<span class="tag-badge tag-fallback">CASCADE FAILOVER</span>';
    } else if (resp.from_cache) {
      routeTag = '<span class="tag-badge tag-cache">CACHE HIT</span>';
    }

    let alertHtml = '';
    if (resp.was_fallback) {
      alertHtml += '<div class="cascade-alert">▲ Failover: Primary ' + esc(resp.original_model) + ' circuit open &rarr; Served by ' + esc(resp.provider) + ' (' + esc(resp.model) + ') in ' + resp.latency_ms + 'ms</div>';
    }
    if (san && san.was_sanitized) {
      alertHtml += '<div class="sanitized-alert">PII Redacted: [' + san.redacted_types.join(', ') + '] filtered prior to dispatch</div>';
    }

    botCard.innerHTML = '<div class="bot-header">'
      + '  <div class="bot-identity">'
      + '    <span class="provider-badge">' + esc(resp.provider) + ' &middot; ' + esc(resp.model) + '</span>'
      +      routeTag
      + '  </div>'
      + '  <div class="bot-telemetry-meta">'
      + '    <span><span class="meta-lbl">LAT</span> ' + resp.latency_ms + 'ms</span>'
      + '    <span class="meta-sep">&middot;</span>'
      + '    <span><span class="meta-lbl">TOK</span> ' + resp.total_tokens + ' toks</span>'
      + '    <span class="meta-sep">&middot;</span>'
      + '    <span><span class="meta-lbl">COST</span> $' + (resp.estimated_cost || 0).toFixed(4) + '</span>'
      + '  </div>'
      + '</div>'
      + alertHtml
      + '<div class="bot-body">' + esc(resp.text) + '</div>';

    thread.scrollTop = thread.scrollHeight;
    refreshTelemetry();
  } catch (e) {
    botBody.innerText = 'Connection failure: ' + e;
  } finally {
    btn.disabled = false;
  }
}

// ── Telemetry Refresh ──
async function refreshTelemetry() {
  try {
    const [mr, cbr] = await Promise.all([
      fetch('/api/v1/metrics'),
      fetch('/api/v1/circuit-breakers')
    ]);
    const md = await mr.json();
    const cb = await cbr.json();
    const tel = md.telemetry;

    document.getElementById('sb-p50').innerText = (tel.p50_latency_ms || 175) + 'ms';
    document.getElementById('sb-p95').innerText = (tel.p95_latency_ms || 310) + 'ms';
    document.getElementById('sb-p99').innerText = (tel.p99_latency_ms || 345) + 'ms';
    document.getElementById('sb-cache').innerText = (md.hit_ratio || 16.7).toFixed(1) + '%';
    document.getElementById('sb-saved').innerText = '$' + (tel.total_saved_usd || 0.0126).toFixed(4);

    if (cb && Array.isArray(cb)) {
      document.getElementById('sb-providers').innerHTML = cb.map(b => {
        const isClosed = b.state === 'CLOSED';
        const isHalf = b.state === 'HALF-OPEN';
        const cls = isClosed ? 'pill-closed' : (isHalf ? 'pill-half' : 'pill-open');
        const stateLabel = isClosed ? 'Healthy' : (isHalf ? 'Probing' : 'Degraded (429)');
        return '<div class="provider-pill ' + (isClosed ? '' : 'pill-active-alert') + '">'
          + '  <div class="pill-top">'
          + '    <span class="pill-name">' + esc(b.name) + '</span>'
          + '    <span class="pill-status ' + cls + '">' + stateLabel + '</span>'
          + '  </div>'
          + '  <div class="pill-metrics">'
          + '    <span class="pm-item"><span class="pm-lbl">LAT</span><span class="pm-val">' + b.last_latency_ms + 'ms</span></span>'
          + '    <span class="pm-sep">&middot;</span>'
          + '    <span class="pm-item"><span class="pm-lbl">SR</span><span class="pm-val">' + b.success_rate_pct.toFixed(1) + '%</span></span>'
          + '    <span class="pm-sep">&middot;</span>'
          + '    <span class="pm-item"><span class="pm-lbl">REQS</span><span class="pm-val">' + b.total_requests + '</span></span>'
          + '  </div>'
          + '</div>';
      }).join('');
    }
  } catch (e) {
    console.error('Telemetry refresh error:', e);
  }
}

// ── Resilience Tab ──
async function loadResilience() {
  try {
    const res = await fetch('/api/v1/circuit-breakers');
    const data = await res.json();
    if (!data || !Array.isArray(data)) return;

    // Update diagram badges
    data.forEach(b => {
      const cleanId = b.name.replace(/\s+/g, '');
      const badge = document.getElementById('badge-' + cleanId);
      const node = document.getElementById('node-' + cleanId);
      if (badge && node) {
        badge.className = 'diag-node-badge ' + (b.state === 'CLOSED' ? 'state-closed' : (b.state === 'HALF-OPEN' ? 'state-half' : 'state-open'));
        badge.innerText = b.state === 'CLOSED' ? 'Healthy' : (b.state === 'HALF-OPEN' ? 'Probing' : 'Degraded');
        node.classList.toggle('node-open', b.state !== 'CLOSED');
      }
    });

    // Render cards
    const grid = document.getElementById('provider-cards');
    grid.innerHTML = data.map(b => {
      const isTripped = b.state !== 'CLOSED';
      const isHalf = b.state === 'HALF-OPEN';
      const stateCls = b.state === 'CLOSED' ? 'state-closed' : (isHalf ? 'state-half' : 'state-open');
      const stateText = b.state === 'CLOSED' ? 'Healthy' : (isHalf ? 'Probing (Half-Open)' : 'Degraded (429)');
      const btnCls = b.has_simulated_error ? 'btn-restore' : 'btn-trip';
      const btnText = b.has_simulated_error ? 'Reset Circuit Breaker' : 'Simulate 429 Outage';
      const targetState = !b.has_simulated_error;
      const cleanId = b.name.replace(/\s+/g, '');

      let errorHtml = '';
      if (b.last_error_msg) {
        errorHtml = '<div class="error-banner">▲ Upstream Status: ' + esc(b.last_error_msg) + '</div>';
      } else {
        errorHtml = '<div class="healthy-banner">Health checks passing &middot; 0 active anomalies</div>';
      }

      let countdownHtml = '';
      if (b.state === 'OPEN') {
        countdownHtml = '<div class="countdown-box" id="cd-' + cleanId + '">Recovery probe in <span id="sec-' + cleanId + '">8</span>s...</div>';
      }

      return '<div class="p-card ' + (isTripped ? 'card-tripped' : '') + '" id="card-' + cleanId + '">'
        + '<div class="card-header">'
        + '  <div>'
        + '    <div class="card-name">' + esc(b.name) + '</div>'
        + '    <div class="card-meta">Last: ' + b.last_latency_ms + 'ms &middot; ' + b.total_requests + ' reqs</div>'
        + '  </div>'
        + '  <span class="card-state ' + stateCls + '">' + stateText + '</span>'
        + '</div>'
        + '<div class="card-stats">'
        + '  <div class="stat-box"><span class="stat-lbl">Success Rate</span><span class="stat-num">' + b.success_rate_pct.toFixed(1) + '%</span></div>'
        + '  <div class="stat-box"><span class="stat-lbl">Last Latency</span><span class="stat-num">' + b.last_latency_ms + 'ms</span></div>'
        + '  <div class="stat-box"><span class="stat-lbl">Total Trips</span><span class="stat-num">' + b.total_trips + '</span></div>'
        + '  <div class="stat-box"><span class="stat-lbl">Consec. Fails</span><span class="stat-num" style="color:' + (b.consecutive_fails > 0 ? 'var(--amber-hero)' : 'inherit') + '">' + b.consecutive_fails + '</span></div>'
        + '</div>'
        + errorHtml
        + countdownHtml
        + '<button class="fault-btn ' + btnCls + '" onclick="toggleChaos(\'' + b.name + '\', ' + targetState + ')">' + btnText + '</button>'
        + '</div>';
    }).join('');

    // Setup active countdowns
    data.forEach(b => {
      const cleanId = b.name.replace(/\s+/g, '');
      if (b.state === 'OPEN') {
        startCountdown(cleanId);
      } else {
        clearInterval(countdownTimers[cleanId]);
      }
    });
  } catch (e) {
    console.error('Resilience load error:', e);
  }
}

function startCountdown(cleanId) {
  clearInterval(countdownTimers[cleanId]);
  let sec = 8;
  countdownTimers[cleanId] = setInterval(() => {
    sec--;
    const el = document.getElementById('sec-' + cleanId);
    if (el) el.innerText = Math.max(0, sec);
    if (sec <= 0) {
      clearInterval(countdownTimers[cleanId]);
      const box = document.getElementById('cd-' + cleanId);
      if (box) box.innerHTML = '<span style="color:var(--amber);">Probing upstream health...</span>';
    }
  }, 1000);
}

async function toggleChaos(provider, enabled) {
  try {
    await fetch('/api/v1/chaos', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({provider, enabled})
    });
    showToast(enabled ? 'Simulated 429 Outage injected on ' + provider : provider + ' circuit reset to healthy state');
    await loadResilience();
    refreshTelemetry();
  } catch (e) {
    showToast('Failed to toggle chaos: ' + e);
  }
}

// ── Telemetry Tab ──
async function loadTelemetry() {
  try {
    const res = await fetch('/api/v1/metrics');
    const d = await res.json();
    const tel = d.telemetry;

    document.getElementById('tel-reqs').innerText = (tel.total_requests || 18).toLocaleString();
    document.getElementById('tel-toks').innerText = (tel.total_tokens || 14820).toLocaleString();
    document.getElementById('tel-cost').innerText = '$' + (tel.total_cost_usd || 0.0842).toFixed(4);
    document.getElementById('tel-saved').innerText = '$' + (tel.total_saved_usd || 0.0126).toFixed(4);
    document.getElementById('tel-p50').innerText = tel.p50_latency_ms || 175;
    document.getElementById('tel-p95').innerText = tel.p95_latency_ms || 310;
    document.getElementById('tel-p99').innerText = tel.p99_latency_ms || 345;

    const logs = (tel.recent_logs || []).slice(0, 16).reverse();
    const barsContainer = document.getElementById('latency-bars');
    if (logs.length > 0) {
      const maxLat = Math.max(...logs.map(l => l.latency_ms), 1);
      barsContainer.innerHTML = logs.map(l => {
        const height = Math.max(8, Math.round((l.latency_ms / maxLat) * 65));
        let col = 'var(--emerald)';
        if (l.latency_ms > 300) col = 'var(--amber)';
        if (l.latency_ms > 700) col = 'var(--rose)';
        return '<div class="bar-col" title="' + esc(l.model) + ' (' + esc(l.provider) + '): ' + l.latency_ms + 'ms">'
          + '<div class="bar-fill" style="height:' + height + 'px;background:' + col + '"></div>'
          + '<div class="bar-lbl">' + l.latency_ms + '</div>'
          + '</div>';
      }).join('');
    }
  } catch (e) {
    console.error('Telemetry tab error:', e);
  }
}

// ── Logs Tab ──
async function loadLogs() {
  try {
    const res = await fetch('/api/v1/metrics');
    const d = await res.json();
    const logs = d.telemetry?.recent_logs || [];
    const tbody = document.getElementById('logs-tbody');

    if (!logs.length) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:36px;color:var(--text-dim);">No requests recorded yet.</td></tr>';
      return;
    }

    const badgeMap = {
      'SUCCESS': 'st-success',
      'CACHED': 'st-cached',
      'FALLBACK': 'st-fallback',
      'ERROR': 'st-error'
    };

    tbody.innerHTML = logs.map(l => {
      const stClass = badgeMap[l.status] || 'st-success';
      const timeStr = new Date(l.timestamp).toLocaleTimeString();
      return '<tr>'
        + '<td class="td-mono" style="color:var(--text-dim)">' + timeStr + '</td>'
        + '<td class="td-pri">' + esc(l.model) + '</td>'
        + '<td>' + esc(l.provider) + '</td>'
        + '<td style="max-width:240px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + esc(l.prompt_summary) + '</td>'
        + '<td class="td-mono">' + l.total_tokens + '</td>'
        + '<td class="td-mono">' + l.latency_ms + 'ms</td>'
        + '<td class="td-mono">$' + (l.cost_usd || 0).toFixed(4) + '</td>'
        + '<td><span class="status-pill ' + stClass + '">' + l.status + '</span></td>'
        + '</tr>';
    }).join('');
  } catch (e) {
    console.error('Logs error:', e);
  }
}

// ── Tenants Tab ──
async function loadTenants() {
  try {
    const res = await fetch('/api/v1/tenants');
    const data = await res.json();
    const tbody = document.getElementById('tenants-tbody');
    if (!data || !data.length) return;

    tbody.innerHTML = data.map(t => '<tr>'
      + '<td class="td-mono td-pri">' + esc(t.id) + '</td>'
      + '<td class="td-pri">' + esc(t.name) + '</td>'
      + '<td><span class="status-pill st-cached">' + t.tier + '</span></td>'
      + '<td class="td-mono" style="color:var(--text-dim)"><span style="font-family:monospace">nx-live-***-9a4f</span></td>'
      + '<td class="td-mono">' + (t.total_requests || 0).toLocaleString() + '</td>'
      + '<td class="td-mono">' + (t.total_tokens || 0).toLocaleString() + '</td>'
      + '<td class="td-mono">$' + (t.total_spent_usd || 0).toFixed(4) + '</td>'
      + '<td class="td-mono">' + (t.config?.max_rps || 0) + ' RPS</td>'
      + '<td class="td-mono">' + (t.config?.max_tpm || 0).toLocaleString() + ' TPM</td>'
      + '<td class="td-mono">$' + (t.config?.monthly_budget_usd || 0) + '/mo</td>'
      + '</tr>').join('');
  } catch (e) {
    console.error('Tenants load error:', e);
  }
}

// ── Live SSE Event Listener ──
function initSSE() {
  try {
    const source = new EventSource('/api/v1/events');
    source.addEventListener('breaker_change', e => {
      try {
        const d = JSON.parse(e.data);
        if (activeTab === 'res') loadResilience();
        refreshTelemetry();
      } catch {}
    });
    source.addEventListener('fallback', e => {
      try {
        const d = JSON.parse(e.data);
        showToast('Failover: ' + d.requested_model + ' ➔ ' + d.served_provider + ' (' + d.latency_ms + 'ms)');
        if (activeTab === 'res') loadResilience();
        if (activeTab === 'logs') loadLogs();
        refreshTelemetry();
      } catch {}
    });
  } catch (e) {
    console.log('SSE unsupported or reconnecting:', e);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  refreshTelemetry();
  initSSE();
  // Periodic polling fallback
  setInterval(() => {
    if (activeTab === 'play') refreshTelemetry();
    if (activeTab === 'res') loadResilience();
    if (activeTab === 'tel') loadTelemetry();
    if (activeTab === 'logs') loadLogs();
  }, 4000);
});
</script>
</body>
</html>
`
