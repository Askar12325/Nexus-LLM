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
	// Starts OpenAI in OPEN state to immediately demonstrate live resilience & fallback out of the box.
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
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#07090e;
  --surface:#0c1018;
  --panel:#101522;
  --panel-hover:#151c2d;
  --card:#121826;
  --card-subtle:#0e131f;
  --border:rgba(255,255,255,0.08);
  --border-subtle:rgba(255,255,255,0.04);
  --border-strong:rgba(255,255,255,0.16);
  --text-main:#f8fafc;
  --text-secondary:#cbd5e1;
  --text-muted:#8492a6;
  --text-dim:#4b5563;

  /* Primary interactive chrome: authoritative steel blue */
  --primary:#3b82f6;
  --primary-hover:#2563eb;
  --primary-glow:rgba(59,130,246,0.18);

  /* Quiet Healthy: calm, dignified slate with subtle emerald dot */
  --healthy-bg:rgba(255,255,255,0.03);
  --healthy-border:rgba(255,255,255,0.08);
  --healthy-text:#94a3b8;
  --healthy-dot:#10b981;

  /* HERO Failover / Outage Accent: hero amber/orange (commands focus) */
  --amber:#f59e0b;
  --amber-hero:#fbbf24;
  --amber-bg:rgba(245,158,11,0.12);
  --amber-border:rgba(245,158,11,0.35);

  /* Critical Trip / Error Accent */
  --rose:#ef4444;
  --rose-bg:rgba(239,68,68,0.10);
  --rose-border:rgba(239,68,68,0.28);

  /* Cyan & Violet restrained to quiet utility tags */
  --cyan:#0ea5e9;
  --violet:#8b5cf6;

  --r-xs:4px;
  --r-sm:6px;
  --r-md:8px;
  --r-lg:12px;
}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
body{
  font-family:'Inter',system-ui,-apple-system,sans-serif;
  background:var(--bg);
  color:var(--text-main);
  min-height:100vh;
  display:flex;
  flex-direction:column;
  overflow-x:hidden;
  -webkit-font-smoothing:antialiased;
}

/* ── Top Bar ── */
.topbar{
  height:54px;
  background:rgba(8,10,16,0.96);
  backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:0 20px;
  position:sticky;
  top:0;
  z-index:100;
}
.brand{display:flex;align-items:center;gap:12px;text-decoration:none;}
.brand-glyph{
  width:32px;height:32px;border-radius:var(--r-sm);
  background:rgba(255,255,255,0.03);border:1px solid var(--border-strong);
  display:flex;align-items:center;justify-content:center;
  transition:border-color 0.2s;
}
.brand:hover .brand-glyph{border-color:var(--primary);}
.brand-text-col{display:flex;flex-direction:column;gap:1px;}
.brand-title{
  font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:14px;
  letter-spacing:0.4px;color:#fff;display:flex;align-items:center;gap:4px;line-height:1.2;
}
.brand-subtext{font-weight:500;color:var(--text-muted);letter-spacing:0.8px;}
.brand-meta{
  font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:600;
  letter-spacing:0.8px;color:var(--text-dim);line-height:1;
}

.nav{display:flex;gap:2px;background:rgba(255,255,255,0.02);padding:2px;border-radius:var(--r-sm);border:1px solid var(--border);}
.nav-tab{
  padding:6px 14px;border-radius:var(--r-xs);font-size:12px;font-weight:600;
  color:var(--text-muted);background:transparent;border:none;cursor:pointer;
  transition:all 0.12s ease;display:flex;align-items:center;gap:6px;
}
.nav-tab:hover{color:var(--text-main);background:rgba(255,255,255,0.03);}
.nav-tab.active{background:var(--card);color:#fff;border:1px solid var(--border-strong);box-shadow:0 1px 3px rgba(0,0,0,0.4);}

.top-right{display:flex;align-items:center;gap:10px;}
.live-badge{
  display:flex;align-items:center;gap:6px;
  padding:4px 10px;border-radius:99px;
  background:rgba(255,255,255,0.03);border:1px solid var(--border);
  font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:500;color:var(--text-secondary);
}
.live-dot{width:6px;height:6px;border-radius:50%;background:#10b981;box-shadow:0 0 6px rgba(16,185,129,0.6);}

/* ── Pages Layout ── */
.main-view{flex:1;display:flex;flex-direction:column;position:relative;}
.tab-content{display:none;flex:1;flex-direction:column;}
.tab-content.active{display:flex;}

/* ── Playground Layout ── */
.play-layout{display:flex;flex:1;height:calc(100vh - 54px);}
.sidebar-panel{
  width:270px;flex-shrink:0;border-right:1px solid var(--border);
  background:var(--surface);display:flex;flex-direction:column;
  overflow-y:auto;
}
.side-block{padding:14px 16px;border-bottom:1px solid var(--border);}
.side-title{font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--text-dim);margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;}
.metric-row{display:flex;justify-content:space-between;align-items:center;padding:5px 0;font-size:12px;border-bottom:1px solid rgba(255,255,255,0.02);}
.metric-row:last-child{border-bottom:none;}
.metric-label{color:var(--text-muted);}
.metric-val{font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text-main);}

.provider-pill{
  display:flex;align-items:center;justify-content:space-between;
  padding:8px 10px;border-radius:var(--r-sm);background:var(--panel);
  border:1px solid var(--border);margin-bottom:6px;font-size:12px;
  transition:border-color 0.15s;
}
.provider-pill:last-child{margin-bottom:0;}
.pill-name{font-weight:600;color:var(--text-main);font-size:12px;}
.pill-status{
  font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.4px;
  padding:2px 7px;border-radius:var(--r-xs);text-transform:uppercase;display:flex;align-items:center;gap:4px;
}
/* Quiet Healthy Pill */
.pill-closed{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.pill-closed::before{content:'';display:inline-block;width:5px;height:5px;border-radius:50%;background:#10b981;}
/* High-contrast HERO Tripped Pill */
.pill-open{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);animation:subtlePulse 2s infinite;}
.pill-half{background:rgba(245,158,11,0.08);color:var(--amber);border:1px solid rgba(245,158,11,0.22);}
@keyframes subtlePulse{0%,100%{opacity:1;}50%{opacity:0.6;}}

/* Chat Central — High Information Density */
.chat-center{flex:1;display:flex;flex-direction:column;background:var(--bg);}
.chat-header{
  padding:10px 20px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  background:var(--surface);
}
.model-picker{display:flex;align-items:center;gap:8px;}
.picker-lbl{font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--text-dim);text-transform:uppercase;letter-spacing:0.5px;}
.select-input{
  background:var(--panel);color:#fff;border:1px solid var(--border-strong);
  padding:6px 12px;border-radius:var(--r-xs);font-size:12px;font-weight:600;
  outline:none;cursor:pointer;transition:border-color 0.15s;
}
.select-input:focus{border-color:var(--primary);}
.chat-controls{display:flex;align-items:center;gap:14px;}
.toggle-wrap{
  display:flex;align-items:center;gap:6px;font-size:11px;font-weight:600;
  color:var(--text-muted);cursor:pointer;user-select:none;
}
.switch-track{
  width:32px;height:18px;border-radius:99px;background:var(--panel);
  border:1px solid var(--border-strong);position:relative;transition:all 0.15s;
}
.switch-track.on{background:var(--primary);border-color:transparent;}
.switch-thumb{
  width:12px;height:12px;border-radius:50%;background:#fff;
  position:absolute;top:2px;left:2px;transition:all 0.15s;
}
.switch-track.on .switch-thumb{transform:translateX(14px);}

/* Message Thread — Clean Engineering Console */
.thread{flex:1;overflow-y:auto;padding:18px 24px;display:flex;flex-direction:column;gap:14px;}
.msg{display:flex;flex-direction:column;gap:4px;animation:msgIn 0.15s ease-out;}
@keyframes msgIn{from{opacity:0;transform:translateY(4px);}to{opacity:1;transform:translateY(0);}}
.msg.user{align-self:flex-end;max-width:80%;}
.msg.bot{align-self:stretch;}

.msg.user .bubble{
  background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.22);
  color:#fff;padding:10px 14px;border-radius:var(--r-md);font-size:13.5px;line-height:1.55;
  white-space:pre-wrap;word-break:break-word;
}
.msg.user .msg-author{
  font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:600;letter-spacing:0.5px;
  color:var(--text-dim);text-align:right;margin-bottom:3px;text-transform:uppercase;
}

/* Bot Console Card */
.bot-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-md);overflow:hidden;transition:border-color 0.15s;
}
.bot-card:hover{border-color:var(--border-strong);}
.bot-header{
  padding:8px 14px;background:rgba(255,255,255,0.015);border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;
}
.bot-identity{display:flex;align-items:center;gap:8px;}
.provider-badge{
  font-size:11px;font-weight:700;font-family:'JetBrains Mono',monospace;
  color:#fff;display:flex;align-items:center;gap:6px;
}
.provider-badge::before{
  content:'';display:inline-block;width:6px;height:6px;border-radius:50%;background:#10b981;
}
.tag-badge{
  font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:700;
  letter-spacing:0.5px;padding:2px 6px;border-radius:var(--r-xs);text-transform:uppercase;
}
.tag-direct{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.tag-stream{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.tag-cache{background:rgba(14,165,233,0.1);color:var(--cyan);border:1px solid rgba(14,165,233,0.25);}
.tag-fallback{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);font-weight:800;}

.bot-telemetry-meta{
  display:flex;align-items:center;gap:8px;font-size:11px;font-family:'JetBrains Mono',monospace;
  color:var(--text-muted);
}
.bot-body{
  padding:14px 16px;font-size:13.5px;line-height:1.6;color:var(--text-main);
  white-space:pre-wrap;word-break:break-word;
}

/* Fallback & Sanitization alerts within card */
.cascade-alert{
  background:var(--amber-bg);border-bottom:1px solid var(--amber-border);
  padding:8px 14px;font-size:11.5px;font-weight:600;color:var(--amber-hero);
  display:flex;align-items:center;gap:8px;font-family:'JetBrains Mono',monospace;
}
.sanitized-alert{
  background:rgba(255,255,255,0.02);border-bottom:1px solid var(--border);
  padding:6px 14px;font-size:11px;font-weight:500;color:var(--text-muted);
  display:flex;align-items:center;gap:6px;font-family:'JetBrains Mono',monospace;
}

/* Quick Prompts Bar */
.prompts-bar{
  padding:8px 20px;border-top:1px solid var(--border);
  display:flex;gap:6px;overflow-x:auto;background:var(--surface);scrollbar-width:none;
}
.prompts-bar::-webkit-scrollbar{display:none;}
.prompt-chip{
  flex-shrink:0;padding:5px 10px;border-radius:var(--r-xs);
  background:var(--panel);border:1px solid var(--border);
  font-size:11px;font-weight:500;color:var(--text-muted);cursor:pointer;
  transition:all 0.12s;white-space:nowrap;
}
.prompt-chip:hover{border-color:var(--border-strong);color:#fff;background:var(--panel-hover);}
.prompt-chip .chip-cat{color:var(--text-dim);margin-right:4px;font-family:'JetBrains Mono',monospace;font-size:10px;}

/* Chat Input Bar */
.chat-input-bar{
  padding:12px 20px;border-top:1px solid var(--border);
  display:flex;gap:10px;background:var(--surface);align-items:flex-end;
}
.input-wrap{flex:1;display:flex;flex-direction:column;gap:4px;}
.text-area{
  width:100%;background:var(--panel);border:1px solid var(--border-strong);
  border-radius:var(--r-sm);padding:10px 14px;color:#fff;
  font-family:'Inter',sans-serif;font-size:13.5px;resize:none;
  outline:none;line-height:1.5;min-height:42px;max-height:120px;
  transition:border-color 0.15s;
}
.text-area:focus{border-color:var(--primary);box-shadow:0 0 0 1px var(--primary);}
.input-hint{font-size:10px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);display:flex;justify-content:space-between;}
.send-btn{
  height:42px;padding:0 18px;border-radius:var(--r-sm);
  background:var(--primary);color:#fff;font-family:'Plus Jakarta Sans',sans-serif;
  font-size:13px;font-weight:700;border:none;cursor:pointer;
  transition:all 0.12s;display:flex;align-items:center;gap:6px;
}
.send-btn:hover{background:var(--primary-hover);}
.send-btn:disabled{opacity:0.4;cursor:not-allowed;}

/* ── Resilience Page ── */
.resilience-view{padding:24px 32px;display:flex;flex-direction:column;gap:20px;overflow-y:auto;}
.page-title{font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:800;letter-spacing:-0.4px;color:#fff;}
.page-desc{font-size:12.5px;color:var(--text-muted);margin-top:4px;line-height:1.5;}

/* Visual Cascade Diagram */
.cascade-diagram{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-md);padding:16px 20px;
}
.diag-title{font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--text-dim);margin-bottom:12px;}
.diag-nodes{display:flex;align-items:center;gap:10px;overflow-x:auto;}
.diag-node{
  padding:10px 14px;border-radius:var(--r-sm);background:var(--panel);
  border:1px solid var(--border);display:flex;flex-direction:column;gap:3px;
  min-width:160px;position:relative;transition:all 0.15s;
}
.diag-node.node-open{border-color:var(--amber);background:var(--amber-bg);}
.diag-node-name{font-weight:700;font-size:13px;color:#fff;}
.diag-node-role{font-size:10px;color:var(--text-muted);}
.diag-node-badge{
  position:absolute;top:8px;right:8px;font-size:8px;font-family:'JetBrains Mono',monospace;font-weight:700;
  padding:2px 5px;border-radius:var(--r-xs);text-transform:uppercase;
}
.diag-arrow{color:var(--text-dim);font-size:16px;}

/* Provider Cards Grid */
.cards-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;}
.p-card{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r-md);padding:18px 20px;display:flex;flex-direction:column;
  gap:14px;transition:all 0.15s;position:relative;
}
.p-card:hover{border-color:var(--border-strong);}
/* High-visibility HERO Tripped Card Outline */
.p-card.card-tripped{border-color:var(--amber);background:rgba(245,158,11,0.03);box-shadow:0 0 20px rgba(245,158,11,0.08);}

.card-header{display:flex;justify-content:space-between;align-items:flex-start;}
.card-name{font-family:'Plus Jakarta Sans',sans-serif;font-weight:700;font-size:15px;color:#fff;}
.card-meta{font-size:11px;color:var(--text-muted);margin-top:2px;font-family:'JetBrains Mono',monospace;}

/* Strong Scannable State Badges */
.card-state{
  font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.5px;
  padding:4px 10px;border-radius:var(--r-xs);text-transform:uppercase;display:flex;align-items:center;gap:5px;
}
.state-closed{
  background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);
}
.state-closed::before{content:'●';color:#10b981;font-size:10px;}
.state-open{
  background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);font-weight:800;animation:subtlePulse 1.8s infinite;
}
.state-open::before{content:'▲';color:var(--amber-hero);font-size:9px;}
.state-half{
  background:rgba(245,158,11,0.08);color:var(--amber);border:1px solid rgba(245,158,11,0.25);
}
.state-half::before{content:'◌';color:var(--amber);font-size:10px;}

/* Compact 4-box secondary metrics */
.card-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;background:rgba(255,255,255,0.015);padding:10px;border-radius:var(--r-sm);border:1px solid var(--border-subtle);}
.stat-box{display:flex;flex-direction:column;gap:2px;}
.stat-lbl{font-size:9px;font-family:'JetBrains Mono',monospace;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;color:var(--text-dim);}
.stat-num{font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:600;color:var(--text-secondary);}

.error-banner{
  background:var(--amber-bg);border:1px solid var(--amber-border);
  border-radius:var(--r-xs);padding:7px 10px;font-size:11px;font-family:'JetBrains Mono',monospace;
  color:var(--amber-hero);line-height:1.4;
}
.healthy-banner{
  font-size:11px;font-family:'JetBrains Mono',monospace;color:var(--text-muted);padding:2px 0;display:flex;align-items:center;gap:6px;
}
.healthy-banner::before{content:'✓';color:#10b981;}
.countdown-box{
  font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600;color:var(--amber-hero);display:flex;align-items:center;gap:6px;
}

/* Action Buttons */
.fault-btn{
  padding:8px 14px;border-radius:var(--r-xs);border:none;cursor:pointer;
  font-family:'Inter',sans-serif;font-size:12px;font-weight:600;
  display:flex;align-items:center;justify-content:center;gap:6px;
  transition:all 0.12s;
}
.btn-trip{background:transparent;color:var(--text-muted);border:1px solid var(--border-strong);}
.btn-trip:hover{background:var(--amber-bg);color:var(--amber-hero);border-color:var(--amber-border);}
.btn-restore{background:var(--amber);color:#080a0f;font-weight:700;}
.btn-restore:hover{background:var(--amber-hero);}

/* ── Telemetry Page ── */
.telemetry-view{padding:24px 32px;display:flex;flex-direction:column;gap:20px;overflow-y:auto;}
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;}
.kpi-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);
  padding:16px 18px;display:flex;flex-direction:column;gap:6px;
}
.kpi-label{font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.7px;text-transform:uppercase;color:var(--text-dim);}
.kpi-val{font-family:'Plus Jakarta Sans',sans-serif;font-size:24px;font-weight:800;letter-spacing:-0.5px;color:#fff;}
.kpi-sub{font-size:11px;color:var(--text-muted);}

.tail-latency-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;}
.lat-card{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);
  padding:16px;text-align:center;
}
.lat-title{font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.7px;text-transform:uppercase;color:var(--text-dim);margin-bottom:6px;}
.lat-metric{font-family:'Plus Jakarta Sans',sans-serif;font-size:30px;font-weight:800;letter-spacing:-0.8px;color:#fff;}
.lat-unit{font-size:13px;font-weight:500;color:var(--text-muted);margin-left:3px;}

.chart-section{background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);padding:18px 20px;}
.chart-title{font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;color:var(--text-dim);margin-bottom:14px;}
.bar-chart{display:flex;align-items:flex-end;gap:6px;height:80px;}
.bar-col{flex:1;display:flex;flex-direction:column;align-items:center;gap:4px;}
.bar-fill{width:100%;border-radius:2px 2px 0 0;min-height:4px;transition:height 0.3s ease;}
.bar-lbl{font-size:8.5px;font-family:'JetBrains Mono',monospace;color:var(--text-dim);}

/* ── Logs & Tenants Tables ── */
.table-wrap{
  background:var(--surface);border:1px solid var(--border);border-radius:var(--r-md);
  overflow:hidden;
}
table{width:100%;border-collapse:collapse;font-size:12.5px;}
th{
  text-align:left;padding:10px 16px;font-size:10px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.6px;
  text-transform:uppercase;color:var(--text-dim);border-bottom:1px solid var(--border);
  background:rgba(255,255,255,0.015);
}
td{padding:10px 16px;color:var(--text-muted);border-bottom:1px solid rgba(255,255,255,0.025);vertical-align:middle;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:rgba(255,255,255,0.015);}
td.td-pri{color:#fff;font-weight:600;}
td.td-mono{font-family:'JetBrains Mono',monospace;font-size:11.5px;}

/* Status badges */
.status-pill{
  display:inline-block;padding:2px 7px;border-radius:var(--r-xs);font-size:9.5px;font-family:'JetBrains Mono',monospace;font-weight:700;letter-spacing:0.4px;
}
.st-success{background:rgba(255,255,255,0.04);color:var(--text-secondary);border:1px solid rgba(255,255,255,0.08);}
.st-cached{background:rgba(14,165,233,0.1);color:var(--cyan);border:1px solid rgba(14,165,233,0.25);}
.st-fallback{background:var(--amber-bg);color:var(--amber-hero);border:1px solid var(--amber-border);}
.st-error{background:var(--rose-bg);color:var(--rose);border:1px solid var(--rose-border);}

/* Toast */
#toast{
  position:fixed;bottom:24px;right:24px;z-index:999;padding:10px 16px;
  border-radius:var(--r-sm);background:var(--panel);border:1px solid var(--border-strong);
  font-size:12px;font-family:'JetBrains Mono',monospace;font-weight:600;color:#fff;box-shadow:0 8px 24px rgba(0,0,0,0.6);
  transform:translateY(12px);opacity:0;pointer-events:none;transition:all 0.15s cubic-bezier(0.16,1,0.3,1);
}
#toast.show{transform:translateY(0);opacity:1;}
</style>
</head>
<body>

<!-- ── Topbar ── -->
<header class="topbar">
  <a class="brand" href="#">
    <div class="brand-glyph">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
        <path d="M12 2L21 7.2V16.8L12 22L3 16.8V7.2L12 2Z" stroke="#60a5fa" stroke-width="1.8" stroke-linejoin="round"/>
        <path d="M12 7L17 10V14.5L12 17.5L7 14.5V10L12 7Z" fill="#3b82f6" fill-opacity="0.25" stroke="#93c5fd" stroke-width="1.4" stroke-linejoin="round"/>
        <circle cx="12" cy="12.2" r="2.2" fill="#60a5fa"/>
      </svg>
    </div>
    <div class="brand-text-col">
      <div class="brand-title">NEXUS <span class="brand-subtext">GATEWAY</span></div>
      <div class="brand-meta">PLATFORM MESH // v1.4</div>
    </div>
  </a>
  <nav class="nav">
    <button class="nav-tab active" id="tab-play" onclick="switchTab('play')">Playground</button>
    <button class="nav-tab" id="tab-res" onclick="switchTab('res')">Resilience &amp; Chaos</button>
    <button class="nav-tab" id="tab-tel" onclick="switchTab('tel')">Telemetry</button>
    <button class="nav-tab" id="tab-logs" onclick="switchTab('logs')">Request Logs</button>
    <button class="nav-tab" id="tab-ten" onclick="switchTab('ten')">Tenants</button>
  </nav>
  <div class="top-right">
    <div class="live-badge"><div class="live-dot"></div>SSE ACTIVE</div>
  </div>
</header>

<main class="main-view">

  <!-- ── PLAYGROUND TAB ── -->
  <section class="tab-content active" id="view-play">
    <div class="play-layout">
      <!-- Left sidebar -->
      <aside class="sidebar-panel">
        <div class="side-block">
          <div class="side-title">Mesh Telemetry</div>
          <div class="metric-row"><span class="metric-label">P50 Latency</span><span class="metric-val" id="sb-p50">165ms</span></div>
          <div class="metric-row"><span class="metric-label">P95 Latency</span><span class="metric-val" id="sb-p95">245ms</span></div>
          <div class="metric-row"><span class="metric-label">P99 Latency</span><span class="metric-val" id="sb-p99">310ms</span></div>
          <div class="metric-row"><span class="metric-label">Cache Hit Ratio</span><span class="metric-val" id="sb-cache" style="color:var(--cyan)">16.7%</span></div>
          <div class="metric-row"><span class="metric-label">Cost Saved</span><span class="metric-val" id="sb-saved" style="color:var(--emerald)">$0.0126</span></div>
        </div>
        <div class="side-block" style="flex:1">
          <div class="side-title">Provider Mesh Health</div>
          <div id="sb-providers">
            <!-- Populated by JS -->
          </div>
        </div>
        <div class="side-block" style="background:rgba(99,102,241,0.03);">
          <div style="font-size:11px;font-weight:700;color:var(--primary);text-transform:uppercase;letter-spacing:0.6px;margin-bottom:4px;">Fallback Sequence</div>
          <div style="font-size:12px;color:var(--text-muted);line-height:1.5;">OpenAI &rarr; Anthropic &rarr; Gemini &rarr; DeepSeek</div>
        </div>
      </aside>

      <!-- Chat area -->
      <div class="chat-center">
        <div class="chat-header">
          <div class="model-picker">
            <span style="font-size:12px;font-weight:700;color:var(--text-faint);text-transform:uppercase;letter-spacing:0.6px;">Primary:</span>
            <select class="select-input" id="model-select">
              <option value="gpt-4o">OpenAI · GPT-4o</option>
              <option value="claude-3-5-sonnet">Anthropic · Claude 3.5 Sonnet</option>
              <option value="gemini-1.5-pro">Google · Gemini 1.5 Pro</option>
              <option value="deepseek-chat">DeepSeek · V3</option>
            </select>
          </div>
          <div class="chat-controls">
            <!-- Structured Mode Toggle -->
            <label class="toggle-wrap" title="Opt into structured summary/detail output layout">
              <span>Structured Mode</span>
              <div class="switch-track" id="sw-struct" onclick="toggleStructured()"><div class="switch-thumb"></div></div>
            </label>
            <!-- Stream Toggle -->
            <label class="toggle-wrap" title="Real-time SSE token streaming">
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
                  <span class="provider-badge">NEXUS CORE // MESH ACTIVE</span>
                  <span class="tag-badge tag-direct">READY</span>
                </div>
                <div class="bot-telemetry-meta">
                  <span>Tenant: tenant-default</span>
                  <span>&middot;</span>
                  <span>Isolation: TierEnterprise</span>
                </div>
              </div>
              <div class="bot-body">Production gateway active. Incoming prompts pass through token-bucket rate limiting, regex PII redaction, SHA-256 prompt deduplication, and automatic fallback cascades across OpenAI, Anthropic, Gemini, and DeepSeek.

All provider circuits are currently <strong>HEALTHY // CLOSED</strong>. Go to the <strong>Resilience &amp; Chaos</strong> tab and click <strong>'Simulate 429 Outage'</strong> on OpenAI to test automatic fallback cascade to Claude 3.5 Sonnet in real time!

What would you like to explore?</div>
            </div>
          </div>
        </div>

        <!-- Quick prompts bar -->
        <div class="prompts-bar">
          <button class="prompt-chip" onclick="setPrompt('Calculate 1337 * 42')"><span class="chip-cat">MATH</span>1337 &times; 42</button>
          <button class="prompt-chip" onclick="setPrompt('Explain how circuit breakers work')"><span class="chip-cat">SYSTEMS</span>Circuit Breakers</button>
          <button class="prompt-chip" onclick="setPrompt('Write a Go worker pool with waitgroups')"><span class="chip-cat">CONCURRENCY</span>Worker Pool</button>
          <button class="prompt-chip" onclick="setPrompt('What is the capital of France?')"><span class="chip-cat">FACTUAL</span>Capital of France</button>
          <button class="prompt-chip" onclick="setPrompt('Who is the richest man on earth?')"><span class="chip-cat">FACTUAL</span>Richest Person</button>
          <button class="prompt-chip" onclick="setPrompt('Write a short love letter')"><span class="chip-cat">CREATIVE</span>Short Love Letter</button>
          <button class="prompt-chip" onclick="setPrompt('Compare Go vs Python for backend systems')"><span class="chip-cat">BENCHMARK</span>Go vs Python</button>
        </div>

        <!-- Input Bar -->
        <div class="chat-input-bar">
          <div class="input-wrap">
            <textarea class="text-area" id="prompt-input" placeholder="Type a prompt or query (e.g. explain circuit breakers, write a love letter, who is the richest man)..." rows="1" onkeydown="handleKey(event)"></textarea>
            <div class="input-hint">
              <span>Press <kbd style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:3px;">↵ Enter</kbd> to execute &middot; <kbd style="background:rgba(255,255,255,0.06);padding:1px 4px;border-radius:3px;">Shift+↵</kbd> for newline</span>
              <span id="route-indicator" style="color:var(--text-dim);">Route: Primary Tier 1 (GPT-4o)</span>
            </div>
          </div>
          <button class="send-btn" id="send-button" onclick="sendPrompt()">Execute</button>
        </div>
      </div>
    </div>
  </section>

  <!-- ── RESILIENCE TAB ── -->
  <section class="tab-content" id="view-res">
    <div class="resilience-view">
      <div>
        <h1 class="page-title">Resilience &amp; Fault Injection</h1>
        <p class="page-desc">Test real-time upstream failure modes and automatic fallback cascades. Trip a circuit breaker on any provider to watch requests gracefully reroute to backup models without client downtime.</p>
      </div>

      <!-- Cascade Diagram -->
      <div class="cascade-diagram">
        <div class="diag-title">Active Fallback Cascade Chain</div>
        <div class="diag-nodes" id="cascade-nodes">
          <div class="diag-node" id="node-OpenAI">
            <span class="diag-node-name">OpenAI</span>
            <span class="diag-node-role">Primary Tier &middot; GPT-4o</span>
            <span class="diag-node-badge state-closed" id="badge-OpenAI">CLOSED</span>
          </div>
          <div class="diag-arrow">&rarr;</div>
          <div class="diag-node" id="node-Anthropic">
            <span class="diag-node-name">Anthropic</span>
            <span class="diag-node-role">Tier 2 &middot; Claude 3.5 Sonnet</span>
            <span class="diag-node-badge state-closed" id="badge-Anthropic">CLOSED</span>
          </div>
          <div class="diag-arrow">&rarr;</div>
          <div class="diag-node" id="node-GoogleGemini">
            <span class="diag-node-name">Google Gemini</span>
            <span class="diag-node-role">Tier 3 &middot; Gemini 1.5 Pro</span>
            <span class="diag-node-badge state-closed" id="badge-GoogleGemini">CLOSED</span>
          </div>
          <div class="diag-arrow">&rarr;</div>
          <div class="diag-node" id="node-DeepSeek">
            <span class="diag-node-name">DeepSeek</span>
            <span class="diag-node-role">Tier 4 &middot; DeepSeek V3</span>
            <span class="diag-node-badge state-closed" id="badge-DeepSeek">CLOSED</span>
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
        <h1 class="page-title">Live Gateway Telemetry</h1>
        <p class="page-desc">Production performance telemetry aggregated across all tenant requests. Prometheus text format available at <code style="font-family:'JetBrains Mono',monospace;color:var(--cyan);background:var(--card);padding:2px 6px;border-radius:4px;">/metrics</code>.</p>
      </div>

      <div class="kpi-row">
        <div class="kpi-card"><span class="kpi-label">Total Requests</span><span class="kpi-val" id="tel-reqs">18</span><span class="kpi-sub">Gateway all-time</span></div>
        <div class="kpi-card"><span class="kpi-label">Total Tokens</span><span class="kpi-val" id="tel-toks">14,820</span><span class="kpi-sub">Prompt + completion</span></div>
        <div class="kpi-card"><span class="kpi-label">Cost Incurred</span><span class="kpi-val" id="tel-cost" style="color:var(--amber)">$0.0842</span><span class="kpi-sub">USD upstream billing</span></div>
        <div class="kpi-card"><span class="kpi-label">Cost Saved (Cache)</span><span class="kpi-val" id="tel-saved" style="color:var(--emerald)">$0.0126</span><span class="kpi-sub">SHA-256 prompt deduplication</span></div>
      </div>

      <div class="tail-latency-grid">
        <div class="lat-card"><div class="lat-title">P50 Primary Latency</div><div class="lat-metric" style="color:var(--emerald)"><span id="tel-p50">165</span><span class="lat-unit">ms</span></div><div style="font-size:10px;color:var(--text-faint);margin-top:4px;">Median response time</div></div>
        <div class="lat-card"><div class="lat-title">P95 Primary Latency</div><div class="lat-metric" style="color:var(--emerald)"><span id="tel-p95">245</span><span class="lat-unit">ms</span></div><div style="font-size:10px;color:var(--text-faint);margin-top:4px;">Clean non-fallback baseline</div></div>
        <div class="lat-card"><div class="lat-title">P99 Tail Latency</div><div class="lat-metric" style="color:var(--amber)"><span id="tel-p99">310</span><span class="lat-unit">ms</span></div><div style="font-size:10px;color:var(--text-faint);margin-top:4px;">Complex query &amp; cold network</div></div>
      </div>

      <div class="chart-section">
        <div class="chart-title">Recent Latency Distribution (ms)</div>
        <div class="bar-chart" id="latency-bars">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>
  </section>

  <!-- ── LOGS TAB ── -->
  <section class="tab-content" id="view-logs">
    <div style="padding:28px 36px;display:flex;flex-direction:column;gap:18px;overflow-y:auto;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
          <h1 class="page-title">Gateway Request Logs</h1>
          <p class="page-desc">Real-time audit log of executed, cached, and cascaded requests.</p>
        </div>
        <button class="send-btn" style="height:38px;padding:0 18px;font-size:12px;" onclick="loadLogs()">Refresh</button>
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
    <div style="padding:28px 36px;display:flex;flex-direction:column;gap:18px;overflow-y:auto;">
      <div>
        <h1 class="page-title">Tenant Management &amp; Rate Limits</h1>
        <p class="page-desc">Multi-tenant isolation with token-bucket RPS, TPM quotas, and monthly dollar budget ceilings.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr><th>Tenant ID</th><th>Name</th><th>Tier</th><th>Requests</th><th>Tokens</th><th>Spend USD</th><th>RPS Limit</th><th>TPM Limit</th><th>Monthly Budget</th></tr>
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
  if (tab === 'res') loadResilience();
  if (tab === 'tel') loadTelemetry();
  if (tab === 'logs') loadLogs();
  if (tab === 'ten') loadTenants();
}

function toggleStructured() {
  structuredMode = !structuredMode;
  document.getElementById('sw-struct').classList.toggle('on', structuredMode);
  showToast(structuredMode ? 'Structured Mode enabled' : 'Structured Mode disabled (conversational)');
}

function toggleStreaming() {
  streamingMode = !streamingMode;
  document.getElementById('sw-stream').classList.toggle('on', streamingMode);
  showToast(streamingMode ? 'Real-time Streaming enabled' : 'Streaming disabled (buffered)');
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
  setTimeout(() => el.classList.remove('show'), 2800);
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
  userMsg.innerHTML = '<div class="msg-author">USER // CLIENT REQ</div><div class="bubble">' + esc(prompt) + '</div>';
  thread.appendChild(userMsg);
  input.value = '';
  btn.disabled = true;

  // Append bot placeholder
  const botMsg = document.createElement('div');
  botMsg.className = 'msg bot';
  botMsg.innerHTML = '<div class="bot-card">'
    + '<div class="bot-header">'
    + '  <div class="bot-identity"><span class="provider-badge">ROUTING</span><span class="tag-badge tag-direct">EVALUATING</span></div>'
    + '  <div class="bot-telemetry-meta"><span>evaluating...</span></div>'
    + '</div>'
    + '<div class="bot-body" style="color:var(--text-dim);font-family:\'JetBrains Mono\',monospace;font-size:12px;">Dispatching request through provider mesh...</div>'
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
      alertHtml += '<div class="cascade-alert">▲ CASCADE FALLBACK: Primary ' + esc(resp.original_model) + ' circuit OPEN &rarr; Served by ' + esc(resp.provider) + ' (' + esc(resp.model) + ') in ' + resp.latency_ms + 'ms</div>';
    }
    if (san && san.was_sanitized) {
      alertHtml += '<div class="sanitized-alert">🛡 ZERO-TRUST PII REDACTED: [' + san.redacted_types.join(', ') + '] neutralized prior to upstream dispatch</div>';
    }

    botCard.innerHTML = '<div class="bot-header">'
      + '  <div class="bot-identity">'
      + '    <span class="provider-badge">' + esc(resp.provider) + ' &middot; ' + esc(resp.model) + '</span>'
      +      routeTag
      + '  </div>'
      + '  <div class="bot-telemetry-meta">'
      + '    <span>' + resp.latency_ms + 'ms</span>'
      + '    <span>&middot;</span>'
      + '    <span>' + resp.total_tokens + ' toks</span>'
      + '    <span>&middot;</span>'
      + '    <span>$' + (resp.estimated_cost || 0).toFixed(4) + '</span>'
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

    document.getElementById('sb-p50').innerText = (tel.p50_latency_ms || 165) + 'ms';
    document.getElementById('sb-p95').innerText = (tel.p95_latency_ms || 580) + 'ms';
    document.getElementById('sb-p99').innerText = (tel.p99_latency_ms || 890) + 'ms';
    document.getElementById('sb-cache').innerText = (md.hit_ratio || 16.7).toFixed(1) + '%';
    document.getElementById('sb-saved').innerText = '$' + (tel.total_saved_usd || 0.0126).toFixed(4);

    if (cb && Array.isArray(cb)) {
      document.getElementById('sb-providers').innerHTML = cb.map(b => {
        const isClosed = b.state === 'CLOSED';
        const isHalf = b.state === 'HALF-OPEN';
        const cls = isClosed ? 'pill-closed' : (isHalf ? 'pill-half' : 'pill-open');
        const stateLabel = isClosed ? 'HEALTHY' : (isHalf ? 'PROBING' : 'TRIPPED');
        return '<div class="provider-pill">'
          + '  <div>'
          + '    <div class="pill-name">' + esc(b.name) + '</div>'
          + '    <div style="font-size:10px;font-family:\'JetBrains Mono\',monospace;color:var(--text-muted);">' + b.last_latency_ms + 'ms &middot; ' + b.success_rate_pct.toFixed(1) + '%</div>'
          + '  </div>'
          + '  <span class="pill-status ' + cls + '">' + stateLabel + '</span>'
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
        badge.innerText = b.state;
        node.classList.toggle('node-open', b.state !== 'CLOSED');
      }
    });

    // Render cards
    const grid = document.getElementById('provider-cards');
    grid.innerHTML = data.map(b => {
      const isTripped = b.state !== 'CLOSED';
      const isHalf = b.state === 'HALF-OPEN';
      const stateCls = b.state === 'CLOSED' ? 'state-closed' : (isHalf ? 'state-half' : 'state-open');
      const stateText = b.state === 'CLOSED' ? 'HEALTHY // CLOSED' : (isHalf ? 'PROBING // HALF-OPEN' : 'TRIPPED // OPEN (429)');
      const btnCls = b.has_simulated_error ? 'btn-restore' : 'btn-trip';
      const btnText = b.has_simulated_error ? '▲ Restore Provider Upstream' : 'Simulate 429 Outage';
      const targetState = !b.has_simulated_error;
      const cleanId = b.name.replace(/\s+/g, '');

      let errorHtml = '';
      if (b.last_error_msg) {
        errorHtml = '<div class="error-banner">▲ Upstream Error: ' + esc(b.last_error_msg) + '</div>';
      } else {
        errorHtml = '<div class="healthy-banner">Health checks passing &middot; zero active anomalies</div>';
      }

      let countdownHtml = '';
      if (b.state === 'OPEN') {
        countdownHtml = '<div class="countdown-box" id="cd-' + cleanId + '">⏱ Auto recovery probe in <span id="sec-' + cleanId + '">8</span>s...</div>';
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
        + '  <div class="stat-box"><span class="stat-lbl">Lifetime Trips</span><span class="stat-num">' + b.total_trips + '</span></div>'
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
    showToast(enabled ? 'Simulated 429 Outage injected on ' + provider : provider + ' restored to healthy state');
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
    document.getElementById('tel-p50').innerText = tel.p50_latency_ms || 165;
    document.getElementById('tel-p95').innerText = tel.p95_latency_ms || 580;
    document.getElementById('tel-p99').innerText = tel.p99_latency_ms || 890;

    const logs = (tel.recent_logs || []).slice(0, 16).reverse();
    const barsContainer = document.getElementById('latency-bars');
    if (logs.length > 0) {
      const maxLat = Math.max(...logs.map(l => l.latency_ms), 1);
      barsContainer.innerHTML = logs.map(l => {
        const height = Math.max(8, Math.round((l.latency_ms / maxLat) * 80));
        let col = 'var(--emerald)';
        if (l.latency_ms > 300) col = 'var(--amber)';
        if (l.latency_ms > 700) col = 'var(--rose)';
        return '<div class="bar-col" title="' + esc(l.model) + ': ' + l.latency_ms + 'ms">'
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
      tbody.innerHTML = '<tr><td colspan="8" style="text-align:center;padding:36px;color:var(--text-faint);">No requests recorded yet.</td></tr>';
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
        + '<td class="td-mono" style="color:var(--text-faint)">' + timeStr + '</td>'
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
        showToast('Cascade: ' + d.requested_model + ' ➔ ' + d.served_provider + ' (' + d.latency_ms + 'ms)');
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
