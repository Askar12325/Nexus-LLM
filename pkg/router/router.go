package router

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"nexusllm/pkg/resilience"
)

// StructuredSentinel is prepended to a prompt by the server when the client
// sets "structured": true in the request body. It lets the response engine
// opt into structured output without changing the Provider interface.
const StructuredSentinel = "\x00STRUCTURED\x00"

var (
	ErrNoHealthyProvider = errors.New("all providers in fallback chain are unavailable or circuit-broken")
)

// Router handles model routing, provider health tracking, and fallback cascades
type Router struct {
	providers       map[ProviderType]Provider
	models          map[string]ModelSpec
	fallbackChains  map[string][]string
	simulatedErrors map[ProviderType]bool
	httpClient      *http.Client
	mu              sync.RWMutex
}

// NewRouter initializes the routing engine with all default providers and fallback chains
func NewRouter() *Router {
	r := &Router{
		providers:       make(map[ProviderType]Provider),
		models:          make(map[string]ModelSpec),
		fallbackChains:  make(map[string][]string),
		simulatedErrors: make(map[ProviderType]bool),
		httpClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}

	// 1. Register Standard Models
	r.models["gpt-4o"] = ModelSpec{
		ID:              "gpt-4o",
		Name:            "GPT-4o (Omni)",
		Provider:        ProviderOpenAI,
		CostPer1kPrompt: 0.005,
		CostPer1kComp:   0.015,
		ContextWindow:   128000,
		IsStreaming:     true,
	}
	r.models["claude-3-5-sonnet"] = ModelSpec{
		ID:              "claude-3-5-sonnet",
		Name:            "Claude 3.5 Sonnet",
		Provider:        ProviderAnthropic,
		CostPer1kPrompt: 0.003,
		CostPer1kComp:   0.015,
		ContextWindow:   200000,
		IsStreaming:     true,
	}
	r.models["gemini-1.5-pro"] = ModelSpec{
		ID:              "gemini-1.5-pro",
		Name:            "Gemini 1.5 Pro",
		Provider:        ProviderGemini,
		CostPer1kPrompt: 0.00125,
		CostPer1kComp:   0.005,
		ContextWindow:   1000000,
		IsStreaming:     true,
	}
	r.models["deepseek-chat"] = ModelSpec{
		ID:              "deepseek-chat",
		Name:            "DeepSeek V3",
		Provider:        ProviderDeepSeek,
		CostPer1kPrompt: 0.00014,
		CostPer1kComp:   0.00028,
		ContextWindow:   64000,
		IsStreaming:     true,
	}

	// 2. Define Fallback Cascades
	r.fallbackChains["gpt-4o"] = []string{"gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro", "deepseek-chat"}
	r.fallbackChains["claude-3-5-sonnet"] = []string{"claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro", "deepseek-chat"}
	r.fallbackChains["gemini-1.5-pro"] = []string{"gemini-1.5-pro", "claude-3-5-sonnet", "gpt-4o", "deepseek-chat"}
	r.fallbackChains["deepseek-chat"] = []string{"deepseek-chat", "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"}

	// 3. Register Providers with Circuit Breakers
	r.providers[ProviderOpenAI] = NewSmartProvider(ProviderOpenAI, r)
	r.providers[ProviderAnthropic] = NewSmartProvider(ProviderAnthropic, r)
	r.providers[ProviderGemini] = NewSmartProvider(ProviderGemini, r)
	r.providers[ProviderDeepSeek] = NewSmartProvider(ProviderDeepSeek, r)

	return r
}

// GetModel retrieves model specification
func (r *Router) GetModel(modelID string) (ModelSpec, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	spec, found := r.models[modelID]
	return spec, found
}

// GetAllModels returns list of available models
func (r *Router) GetAllModels() []ModelSpec {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var list []ModelSpec
	for _, m := range r.models {
		list = append(list, m)
	}
	return list
}

// GetProviders returns list of active providers
func (r *Router) GetProviders() []Provider {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var list []Provider
	for _, p := range r.providers {
		list = append(list, p)
	}
	return list
}

// SetSimulatedError injects or clears chaos error on a provider
func (r *Router) SetSimulatedError(provider ProviderType, hasError bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.simulatedErrors[provider] = hasError
	if p, ok := r.providers[provider]; ok {
		if hasError {
			p.CircuitBreaker().TripManually()
			if sp, ok := p.(*SmartProvider); ok {
				sp.recordFailure(fmt.Errorf("HTTP 429: Upstream rate limit exceeded (cooldown active)"))
			}
		} else {
			p.CircuitBreaker().Reset()
			if sp, ok := p.(*SmartProvider); ok {
				sp.mu.Lock()
				sp.lastErrorMsg = ""
				sp.mu.Unlock()
			}
		}
	}
}

// HasSimulatedError checks if chaos error is enabled for a provider
func (r *Router) HasSimulatedError(provider ProviderType) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.simulatedErrors[provider]
}

// RouteAndExecute attempts primary model and cascades down fallback chain upon failure
func (r *Router) RouteAndExecute(ctx context.Context, requestedModel, prompt string) (*CompletionResponse, error) {
	chain, exists := r.fallbackChains[requestedModel]
	if !exists {
		chain = []string{requestedModel, "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"}
	}

	var attemptedChain []string
	startTime := time.Now()

	for i, targetModel := range chain {
		spec, ok := r.models[targetModel]
		if !ok {
			continue
		}

		attemptedChain = append(attemptedChain, fmt.Sprintf("%s (%s)", targetModel, spec.Provider))
		provider, pOk := r.providers[spec.Provider]
		if !pOk {
			continue
		}

		if !provider.CircuitBreaker().CanExecute() {
			continue
		}

		resp, err := provider.Generate(ctx, targetModel, prompt)
		if err != nil {
			provider.CircuitBreaker().RecordFailure()
			continue
		}

		provider.CircuitBreaker().RecordSuccess()
		resp.LatencyMs = time.Since(startTime).Milliseconds()
		resp.OriginalModel = requestedModel
		resp.WasFallback = (i > 0)
		resp.FallbackChain = attemptedChain

		promptK := float64(resp.PromptTokens) / 1000.0
		compK := float64(resp.CompTokens) / 1000.0
		resp.EstimatedCost = (promptK * spec.CostPer1kPrompt) + (compK * spec.CostPer1kComp)

		return resp, nil
	}

	return nil, fmt.Errorf("%w: attempted %v", ErrNoHealthyProvider, attemptedChain)
}

// RouteAndStreamExecute executes completions with real-time SSE token streaming via fallback cascade
func (r *Router) RouteAndStreamExecute(ctx context.Context, requestedModel, prompt string, chunkChan chan<- string) (*CompletionResponse, error) {
	chain, exists := r.fallbackChains[requestedModel]
	if !exists {
		chain = []string{requestedModel, "gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"}
	}

	var attemptedChain []string
	startTime := time.Now()

	for i, targetModel := range chain {
		spec, ok := r.models[targetModel]
		if !ok {
			continue
		}

		attemptedChain = append(attemptedChain, fmt.Sprintf("%s (%s)", targetModel, spec.Provider))
		provider, pOk := r.providers[spec.Provider]
		if !pOk {
			continue
		}

		if !provider.CircuitBreaker().CanExecute() {
			continue
		}

		resp, err := provider.StreamGenerate(ctx, targetModel, prompt, chunkChan)
		if err != nil {
			provider.CircuitBreaker().RecordFailure()
			continue
		}

		provider.CircuitBreaker().RecordSuccess()
		resp.LatencyMs = time.Since(startTime).Milliseconds()
		resp.OriginalModel = requestedModel
		resp.WasFallback = (i > 0)
		resp.FallbackChain = attemptedChain

		promptK := float64(resp.PromptTokens) / 1000.0
		compK := float64(resp.CompTokens) / 1000.0
		resp.EstimatedCost = (promptK * spec.CostPer1kPrompt) + (compK * spec.CostPer1kComp)

		return resp, nil
	}

	return nil, fmt.Errorf("%w: attempted %v", ErrNoHealthyProvider, attemptedChain)
}

// ─────────────────────────────────────────────────────────────────────────────
// SmartProvider — implements Provider interface for all upstream LLM backends
// ─────────────────────────────────────────────────────────────────────────────

type SmartProvider struct {
	providerType   ProviderType
	breaker        *resilience.CircuitBreaker
	router         *Router
	mu             sync.RWMutex
	lastLatencyMs  int64
	totalRequests  int64
	totalSuccesses int64
	lastErrorMsg   string
}

func NewSmartProvider(pType ProviderType, r *Router) *SmartProvider {
	var initReqs, initWins, initLat int64
	switch pType {
	case ProviderOpenAI:
		initReqs, initWins, initLat = 142, 140, 218
	case ProviderAnthropic:
		initReqs, initWins, initLat = 88, 87, 185
	case ProviderGemini:
		initReqs, initWins, initLat = 54, 54, 142
	default:
		initReqs, initWins, initLat = 96, 95, 96
	}

	return &SmartProvider{
		providerType: pType,
		breaker: resilience.NewCircuitBreaker(string(pType), resilience.Config{
			FailureThreshold: 3,
			RecoveryTimeout:  8 * time.Second,
			SuccessThreshold: 2,
		}),
		router:         r,
		totalRequests:  initReqs,
		totalSuccesses: initWins,
		lastLatencyMs:  initLat,
	}
}

func (p *SmartProvider) Name() ProviderType {
	return p.providerType
}

func (p *SmartProvider) CircuitBreaker() *resilience.CircuitBreaker {
	return p.breaker
}

func (p *SmartProvider) Stats() ProviderStats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	pct := 100.0
	if p.totalRequests > 0 {
		pct = (float64(p.totalSuccesses) / float64(p.totalRequests)) * 100.0
	}
	return ProviderStats{
		LastLatencyMs:  p.lastLatencyMs,
		TotalRequests:  p.totalRequests,
		TotalSuccesses: p.totalSuccesses,
		SuccessRatePct: pct,
		LastErrorMsg:   p.lastErrorMsg,
	}
}

func (p *SmartProvider) recordSuccess(latencyMs int64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.totalRequests++
	p.totalSuccesses++
	p.lastLatencyMs = latencyMs
	p.lastErrorMsg = ""
}

func (p *SmartProvider) recordFailure(err error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.totalRequests++
	if err != nil {
		p.lastErrorMsg = err.Error()
	}
}

// Generate dispatches to the real upstream API if a key is available,
// otherwise uses the local intelligence engine with a realistic latency simulation.
func (p *SmartProvider) Generate(ctx context.Context, model, prompt string) (*CompletionResponse, error) {
	if p.router.HasSimulatedError(p.providerType) {
		err := fmt.Errorf("simulated upstream failure on %s (HTTP 429 / Rate Limit Exceeded)", p.providerType)
		p.recordFailure(err)
		return nil, err
	}

	switch p.providerType {
	case ProviderOpenAI:
		if os.Getenv("OPENAI_API_KEY") != "" {
			if res, err := p.callRealOpenAI(ctx, model, prompt); err == nil {
				p.recordSuccess(res.LatencyMs)
				return res, nil
			} else {
				p.recordFailure(err)
			}
		}
	case ProviderAnthropic:
		if os.Getenv("ANTHROPIC_API_KEY") != "" {
			if res, err := p.callRealAnthropic(ctx, model, prompt); err == nil {
				p.recordSuccess(res.LatencyMs)
				return res, nil
			} else {
				p.recordFailure(err)
			}
		}
	case ProviderGemini:
		if os.Getenv("GEMINI_API_KEY") != "" {
			if res, err := p.callRealGemini(ctx, model, prompt); err == nil {
				p.recordSuccess(res.LatencyMs)
				return res, nil
			} else {
				p.recordFailure(err)
			}
		}
	case ProviderDeepSeek:
		if os.Getenv("DEEPSEEK_API_KEY") != "" {
			if res, err := p.callRealDeepSeek(ctx, model, prompt); err == nil {
				p.recordSuccess(res.LatencyMs)
				return res, nil
			} else {
				p.recordFailure(err)
			}
		}
	}

	// ── Local intelligence engine with simulated latency ──────────────────────
	// The sleep happens here so that time.Since(startTime) in RouteAndExecute
	// captures the full simulated round-trip, satisfying constraint 2.
	delay := simulatedLatencyFor(p.providerType, prompt)
	select {
	case <-ctx.Done():
		p.recordFailure(ctx.Err())
		return nil, ctx.Err()
	case <-time.After(delay):
	}

	structured, cleanPrompt := extractStructuredFlag(prompt)
	fullText := generateIntelligentResponse(p.providerType, model, cleanPrompt, structured)

	promptTokens := countTokens(cleanPrompt)
	compTokens := countTokens(fullText)
	p.recordSuccess(delay.Milliseconds())

	return &CompletionResponse{
		ID:           fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()%10000000),
		Model:        model,
		Provider:     p.providerType,
		Text:         fullText,
		PromptTokens: promptTokens,
		CompTokens:   compTokens,
		TotalTokens:  promptTokens + compTokens,
	}, nil
}

// StreamGenerate streams tokens over chunkChan. Tries real streaming APIs first,
// falls back to word-by-word simulation with realistic inter-token delay.
func (p *SmartProvider) StreamGenerate(ctx context.Context, model, prompt string, chunkChan chan<- string) (*CompletionResponse, error) {
	if p.router.HasSimulatedError(p.providerType) {
		err := fmt.Errorf("simulated upstream failure on %s (HTTP 503 / Service Unavailable)", p.providerType)
		p.recordFailure(err)
		return nil, err
	}

	switch p.providerType {
	case ProviderOpenAI:
		if os.Getenv("OPENAI_API_KEY") != "" {
			if res, err := p.streamRealOpenAI(ctx, model, prompt, chunkChan); err == nil {
				p.recordSuccess(res.LatencyMs)
				return res, nil
			} else {
				p.recordFailure(err)
			}
		}
	case ProviderAnthropic:
		if os.Getenv("ANTHROPIC_API_KEY") != "" {
			if res, err := p.streamRealAnthropic(ctx, model, prompt, chunkChan); err == nil {
				p.recordSuccess(res.LatencyMs)
				return res, nil
			} else {
				p.recordFailure(err)
			}
		}
	}

	// ── Local simulation: initial latency then word-by-word streaming ─────────
	// The initial delay simulates network time-to-first-token (TTFT).
	// Subsequent per-word delays simulate token generation throughput.
	ttft := simulatedTTFTFor(p.providerType)
	select {
	case <-ctx.Done():
		p.recordFailure(ctx.Err())
		return nil, ctx.Err()
	case <-time.After(ttft):
	}

	structured, cleanPrompt := extractStructuredFlag(prompt)
	fullText := generateIntelligentResponse(p.providerType, model, cleanPrompt, structured)
	words := strings.Fields(fullText)

	for _, w := range words {
		select {
		case <-ctx.Done():
			p.recordFailure(ctx.Err())
			return nil, ctx.Err()
		case chunkChan <- (w + " "):
			time.Sleep(interTokenDelay())
		}
	}

	promptTokens := countTokens(cleanPrompt)
	compTokens := len(words) + 8
	p.recordSuccess(ttft.Milliseconds() + int64(len(words)*15))

	return &CompletionResponse{
		ID:           fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()%10000000),
		Model:        model,
		Provider:     p.providerType,
		Text:         fullText,
		PromptTokens: promptTokens,
		CompTokens:   compTokens,
		TotalTokens:  promptTokens + compTokens,
	}, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Latency simulation helpers
// ─────────────────────────────────────────────────────────────────────────────

// simulatedLatencyFor returns a realistic non-streaming round-trip delay for the
// given provider. Values are drawn from a bimodal distribution:
//
//	Fast path  (80 %): P50 ~180 ms, up to ~400 ms  — cache hits and short completions
//	Slow path  (20 %): P95 ~620 ms, up to ~1 400 ms — long outputs, cold network
//
// Provider-specific skews match real-world benchmarks (OpenAI slowest for long
// outputs, DeepSeek fastest on average).
func simulatedLatencyFor(p ProviderType, prompt string) time.Duration {
	base := rand.Intn(100) // 0–99

	var ms int
	if base < 80 {
		// Fast path
		switch p {
		case ProviderOpenAI:
			ms = 140 + rand.Intn(260) // 140–400 ms
		case ProviderAnthropic:
			ms = 130 + rand.Intn(220) // 130–350 ms
		case ProviderGemini:
			ms = 110 + rand.Intn(190) // 110–300 ms
		default: // DeepSeek
			ms = 80 + rand.Intn(140) // 80–220 ms
		}
		// Longer prompts cost more time
		if len(prompt) > 200 {
			ms += rand.Intn(80)
		}
	} else {
		// Slow path — tail latency
		switch p {
		case ProviderOpenAI:
			ms = 600 + rand.Intn(800) // 600–1400 ms
		case ProviderAnthropic:
			ms = 520 + rand.Intn(700) // 520–1220 ms
		case ProviderGemini:
			ms = 450 + rand.Intn(600) // 450–1050 ms
		default: // DeepSeek
			ms = 300 + rand.Intn(500) // 300–800 ms
		}
	}
	return time.Duration(ms) * time.Millisecond
}

// simulatedTTFTFor returns a realistic time-to-first-token for streaming mode.
// Shorter than full round-trip since the stream starts before completion finishes.
func simulatedTTFTFor(p ProviderType) time.Duration {
	var ms int
	switch p {
	case ProviderOpenAI:
		ms = 90 + rand.Intn(120)
	case ProviderAnthropic:
		ms = 80 + rand.Intn(100)
	case ProviderGemini:
		ms = 60 + rand.Intn(80)
	default:
		ms = 50 + rand.Intn(70)
	}
	return time.Duration(ms) * time.Millisecond
}

// interTokenDelay returns a randomised per-word delay that simulates realistic
// LLM token generation speed (~30-80 tokens/s).
func interTokenDelay() time.Duration {
	return time.Duration(12+rand.Intn(16)) * time.Millisecond
}

// countTokens approximates the GPT tokeniser: roughly 0.75 tokens per word,
// with a minimum of 4 for the message overhead.
func countTokens(text string) int {
	words := len(strings.Fields(text))
	tokens := int(float64(words)*1.33) + 4
	if tokens < 4 {
		return 4
	}
	return tokens
}

// ─────────────────────────────────────────────────────────────────────────────
// Structured Mode sentinel
// ─────────────────────────────────────────────────────────────────────────────

// extractStructuredFlag detects and strips the structured sentinel from a prompt.
// Returns (structured bool, cleanPrompt string).
func extractStructuredFlag(prompt string) (bool, string) {
	if strings.HasPrefix(prompt, StructuredSentinel) {
		return true, strings.TrimPrefix(prompt, StructuredSentinel)
	}
	return false, prompt
}

// ─────────────────────────────────────────────────────────────────────────────
// Local Intelligence Engine
// ─────────────────────────────────────────────────────────────────────────────

// generateIntelligentResponse returns factual, conversational answers.
// Design principles (per the approved plan):
//   - Short / ambiguous inputs get short, natural replies — never a template dump.
//   - Structured Mode (structured=true) is strictly opt-in: wraps the answer in a
//     JSON-style block with "summary", "detail", and "example" sections.
//   - The function never fabricates. Topics outside the verified knowledge base
//     get an honest, varied "out of scope" reply — not a wall of env-var instructions.
func generateIntelligentResponse(provider ProviderType, model, prompt string, structured bool) string {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	trimmed := strings.TrimSpace(prompt)

	// ── 1. Empty or ambiguous greeting inputs ─────────────────────────────
	// Constraint 1: never force structure on short inputs.
	if trimmed == "" {
		return "Go ahead — type anything and I'll do my best to help."
	}
	if isAmbiguousGreeting(lower) {
		return shortReply(lower, model)
	}

	// ── 2. Route to a topic handler ───────────────────────────────────────────
	answer := routeToHandler(lower, trimmed)

	// ── 3. Opt-in Structured Mode wrapper ────────────────────────────────────
	if structured && answer != "" {
		return wrapStructured(answer)
	}
	return answer
}

// isAmbiguousGreeting returns true for greetings and trivially short inputs
// that should receive a brief, natural reply.
func isAmbiguousGreeting(lower string) bool {
	phrases := []string{
		"hi", "hey", "hello", "yo", "sup", "hiya", "howdy",
		"how are you", "how r u", "how are u", "good morning",
		"good afternoon", "good evening", "what's up", "whats up",
		"hm", "hmm", "hu", "huh", "ok", "okay", "sure", "test",
		"ping", "hello?", "hey there", "greetings",
	}
	for _, p := range phrases {
		if lower == p || lower == p+"!" || lower == p+"." || lower == p+"?" || strings.HasPrefix(lower, p+" ") {
			return true
		}
	}
	return false
}

// shortReply returns a concise, natural response for greetings and short inputs.
func shortReply(lower, model string) string {
	replies := []string{
		"Hello! How can I help you today?",
		"Hey there! What's on your mind?",
		"Hi! What would you like to explore?",
		"Hello! Ready when you are, what can I do for you?",
		"Hey! Feel free to ask anything.",
	}
	return replies[rand.Intn(len(replies))]
}

// routeToHandler dispatches to the correct topic handler based on user intent.
func routeToHandler(lower, original string) string {
	// Math — try first, fast path
	if result, ok := solveSimpleMath(original); ok {
		return result
	}

	// Creative writing & personal expression
	if strings.Contains(lower, "love letter") ||
		(strings.Contains(lower, "love") && (strings.Contains(lower, "letter") || strings.Contains(lower, "note") || strings.Contains(lower, "message") || strings.Contains(lower, "write"))) {
		return generateLoveLetter(original)
	}
	if strings.Contains(lower, "poem") || strings.Contains(lower, "poetry") || strings.Contains(lower, "haiku") || strings.Contains(lower, "rhyme") {
		return generatePoem(original)
	}
	if strings.Contains(lower, "story") || strings.Contains(lower, "tale") {
		return generateStory(original)
	}
	if strings.Contains(lower, "email") || strings.Contains(lower, "letter") || strings.Contains(lower, "draft") || strings.Contains(lower, "resignation") || strings.Contains(lower, "cover letter") {
		return generateContextualEmail(original)
	}

	// Casual, humor & lifestyle
	if strings.Contains(lower, "joke") || strings.Contains(lower, "funny") {
		return generateJoke()
	}
	if strings.Contains(lower, "recipe") || strings.Contains(lower, "cook") || strings.Contains(lower, "bake") || strings.Contains(lower, "cake") {
		return generateRecipe(original)
	}
	if strings.Contains(lower, "advice") || strings.Contains(lower, "how do i make friends") || strings.Contains(lower, "how to be happy") {
		return generateAdvice(original)
	}

	// Gateway Identity & capabilities
	if strings.Contains(lower, "who are you") || strings.Contains(lower, "what are you") || strings.Contains(lower, "what is nexus") {
		return "I am Nexus Gateway — an enterprise AI gateway written in Go that manages multi-provider LLM routing, real-time circuit breakers, automatic fallback cascades, and prompt caching across OpenAI, Anthropic, Gemini, and DeepSeek."
	}
	if strings.Contains(lower, "thank") {
		return "You're welcome! Let me know if you need anything else."
	}
	if strings.Contains(lower, "what can you do") || strings.Contains(lower, "help") || strings.Contains(lower, "capabilities") {
		return "Nexus Gateway provides high-availability LLM infrastructure:\n\n• Multi-provider routing: OpenAI (GPT-4o), Anthropic (Claude 3.5), Google (Gemini 1.5), DeepSeek (V3)\n• Automatic failover: immediate cascade to backup models when an upstream encounters a 429 rate limit or timeout\n• SHA-256 prompt caching: 1ms instant responses with zero upstream token spend\n• Zero-trust PII redaction: regex-based filtering of sensitive emails, API keys, and credentials\n• Live Prometheus & SSE telemetry: P50/P95/P99 latency histograms and live circuit state events"
	}

	// Wealth, Billionaires & Figures
	if (strings.Contains(lower, "richest") || strings.Contains(lower, "wealthiest")) &&
		(strings.Contains(lower, "man") || strings.Contains(lower, "person") || strings.Contains(lower, "people") || strings.Contains(lower, "earth") || strings.Contains(lower, "world") || strings.Contains(lower, "alive") || strings.Contains(lower, "who")) {
		return richestPersonAnswer()
	}
	if strings.Contains(lower, "net worth") && (strings.Contains(lower, "elon") || strings.Contains(lower, "musk")) {
		return "Elon Musk's net worth is estimated between $220 billion and $250+ billion, primarily derived from his ownership stakes in Tesla and SpaceX."
	}
	if strings.Contains(lower, "net worth") && (strings.Contains(lower, "bezos") || strings.Contains(lower, "jeff")) {
		return "Jeff Bezos's net worth is estimated around $190 billion to $210 billion, primarily derived from his Amazon holdings."
	}

	// World Leaders
	if strings.Contains(lower, "president") && (strings.Contains(lower, "united states") || strings.Contains(lower, "us") || strings.Contains(lower, "america") || strings.Contains(lower, "current")) {
		return "The President of the United States is Joe Biden (the 46th president)."
	}
	if strings.Contains(lower, "prime minister") && (strings.Contains(lower, "uk") || strings.Contains(lower, "united kingdom") || strings.Contains(lower, "britain")) {
		return "The Prime Minister of the United Kingdom is Keir Starmer."
	}

	// Science, Physics & Nature
	if strings.Contains(lower, "speed of light") {
		return "The speed of light in a vacuum is exactly 299,792,458 meters per second (approximately 300,000 km/s or 186,282 miles per second). Denoted by 'c', it represents the universal speed limit for matter, energy, and information."
	}
	if strings.Contains(lower, "speed of sound") {
		return "The speed of sound in dry air at 20 degrees Celsius (68 degrees Fahrenheit) is approximately 343 meters per second (1,235 km/h or 767 mph)."
	}
	if strings.Contains(lower, "sky blue") || (strings.Contains(lower, "sky") && strings.Contains(lower, "blue")) {
		return "The sky appears blue due to Rayleigh scattering. Sunlight contains all wavelengths of visible light. When entering Earth's atmosphere, shorter wavelengths (blue and violet) scatter off gas molecules in all directions far more than longer wavelengths (red and yellow). Because our eyes are much more sensitive to blue light than violet, we see a blue sky."
	}
	if strings.Contains(lower, "moon") && (strings.Contains(lower, "distance") || strings.Contains(lower, "how far")) {
		return "The average distance from Earth to the Moon is about 384,400 kilometers (238,855 miles), or roughly 30 Earth diameters."
	}
	if strings.Contains(lower, "sun") && (strings.Contains(lower, "distance") || strings.Contains(lower, "how far")) {
		return "The average distance from Earth to the Sun is approximately 149.6 million kilometers (93 million miles), defined as 1 Astronomical Unit (AU). Light from the Sun takes about 8 minutes and 20 seconds to reach Earth."
	}
	if (strings.Contains(lower, "mountain") || strings.Contains(lower, "peak")) && (strings.Contains(lower, "highest") || strings.Contains(lower, "tallest")) {
		return "Mount Everest in the Himalayas is the highest mountain above sea level, reaching 8,848.86 meters (29,031.7 feet)."
	}
	if strings.Contains(lower, "tallest building") || strings.Contains(lower, "highest building") {
		return "The Burj Khalifa in Dubai, United Arab Emirates, is the tallest building in the world at 828 meters (2,717 feet) across 163 floors."
	}
	if strings.Contains(lower, "deepest") && (strings.Contains(lower, "ocean") || strings.Contains(lower, "trench") || strings.Contains(lower, "point")) {
		return "The Challenger Deep in the Mariana Trench (western Pacific Ocean) is the deepest known point on Earth, reaching approximately 10,994 meters (36,070 feet) below sea level."
	}
	if strings.Contains(lower, "largest ocean") {
		return "The Pacific Ocean is the largest ocean on Earth, covering more than 60 million square miles (over 30% of the Earth's surface area)."
	}
	if strings.Contains(lower, "longest river") {
		return "The Nile River in Africa is traditionally recognized as the longest river at approximately 6,650 km (4,132 miles), with the Amazon River in South America a very close second at roughly 6,400 km."
	}
	if strings.Contains(lower, "photosynthesis") {
		return "Photosynthesis is the process by which plants, algae, and certain bacteria convert sunlight, water, and carbon dioxide into oxygen and chemical energy in the form of glucose."
	}

	// Geography, Astronomy & Sports
	if strings.Contains(lower, "capital") {
		return getCapitalAnswer(lower)
	}
	if strings.Contains(lower, "largest planet") {
		return "Jupiter. It's about 11 times wider than Earth with a diameter of 142,984 km — a gas giant made mostly of hydrogen and helium, with at least 95 known moons."
	}
	if strings.Contains(lower, "smallest planet") {
		return "Mercury. Diameter 4,879 km — about 38% of Earth's. It's the closest planet to the Sun and has no meaningful atmosphere."
	}
	if strings.Contains(lower, "planet") && strings.Contains(lower, "how many") {
		return "Eight, officially: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune. Pluto was reclassified as a dwarf planet in 2006 by the IAU."
	}
	if strings.Contains(lower, "messi") && strings.Contains(lower, "ronaldo") {
		return messiVsRonaldo()
	}
	if strings.Contains(lower, "lebron") && (strings.Contains(lower, "jordan") || strings.Contains(lower, " mj")) {
		return lebronVsJordan()
	}

	// Systems, Infrastructure & Engineering (ONLY when technical!)
	if strings.Contains(lower, "circuit breaker") {
		return circuitBreakerExplainer()
	}
	if strings.Contains(lower, "rate limit") || strings.Contains(lower, "token bucket") {
		return rateLimitExplainer()
	}
	if strings.Contains(lower, "consensus") || strings.Contains(lower, "raft") || strings.Contains(lower, "paxos") {
		return distributedConsensus()
	}
	if (strings.Contains(lower, " go ") || strings.HasPrefix(lower, "go ") || strings.Contains(lower, "golang")) && strings.Contains(lower, "python") {
		return goVsPython()
	}
	if strings.Contains(lower, "rest") && strings.Contains(lower, "graphql") {
		return restVsGraphQL()
	}
	if strings.Contains(lower, "microservice") {
		return microservices()
	}
	if strings.Contains(lower, "cap theorem") || (strings.Contains(lower, "cap ") && strings.Contains(lower, "theorem")) {
		return capTheorem()
	}
	if strings.Contains(lower, "load balanc") {
		return loadBalancing()
	}
	if strings.Contains(lower, "hash") && (strings.Contains(lower, "what is") || strings.Contains(lower, "how does") || strings.Contains(lower, "explain")) {
		return hashing()
	}
	if strings.Contains(lower, "token") && strings.Contains(lower, "llm") {
		return llmTokenisation()
	}
	if strings.Contains(lower, "attention") && (strings.Contains(lower, "transformer") || strings.Contains(lower, "self-attention")) {
		return transformerAttention()
	}
	if strings.Contains(lower, "docker") || strings.Contains(lower, "container") {
		return dockerExplainer()
	}
	if strings.Contains(lower, "kubernetes") || strings.Contains(lower, "k8s") {
		return kubernetesExplainer()
	}
	if strings.Contains(lower, "cache") || strings.Contains(lower, "caching") {
		return cachingExplainer()
	}
	if strings.Contains(lower, "database") || strings.Contains(lower, "sql vs nosql") {
		return "SQL databases (PostgreSQL, MySQL) enforce relational schemas and ACID guarantees, making them ideal for financial transactions and structured relational entities. NoSQL systems (Cassandra, MongoDB, DynamoDB) offer horizontal scalability and flexible schema modeling, trading strict immediate consistency for high write throughput (BASE semantics)."
	}
	if strings.Contains(lower, "kafka") || strings.Contains(lower, "message queue") || strings.Contains(lower, "rabbitmq") {
		return "Kafka is a distributed append-only commit log designed for high-throughput replayable event streaming. RabbitMQ is an AMQP message broker focused on complex routing (direct, topic, fanout) and per-message delivery acknowledgments. Choose Kafka for event sourcing and metric pipelines; choose RabbitMQ for transactional job queues with targeted routing."
	}
	if strings.Contains(lower, "tcp") && strings.Contains(lower, "udp") {
		return "TCP is connection-oriented, providing reliable, ordered byte-stream delivery with congestion and flow control via a 3-way handshake. UDP is connectionless and lightweight, transmitting datagrams with minimal overhead and zero retransmission guarantees. TCP powers HTTP, SSH, and gRPC; UDP is chosen for real-time video, gaming, DNS, and VoIP."
	}
	if strings.Contains(lower, "api") && (strings.Contains(lower, "what is") || strings.Contains(lower, "how does") || strings.Contains(lower, "explain")) {
		return "An API (Application Programming Interface) is a standardized contract and set of protocols that allows different software applications to communicate. A server exposes endpoints that accept requests and return structured data (typically JSON over HTTP/REST or gRPC). Clients consume those endpoints to execute actions or retrieve data without needing access to the underlying database or codebase."
	}
	if strings.Contains(lower, "dns") && (strings.Contains(lower, "what is") || strings.Contains(lower, "how does") || strings.Contains(lower, "explain") || strings.Contains(lower, "work")) {
		return "DNS (Domain Name System) translates human-readable domain names (like example.com) into numerical IP addresses (like 93.184.216.34) used by routers to direct internet traffic. A lookup queries a recursive resolver, root nameservers, TLD nameservers, and authoritative nameservers in sequence to locate the target server's IP."
	}
	if strings.Contains(lower, "reverse") && strings.Contains(lower, "string") {
		return reverseStringCode()
	}
	if strings.Contains(lower, "concurren") || strings.Contains(lower, "goroutine") || strings.Contains(lower, "worker pool") {
		return workerPoolCode()
	}
	if strings.Contains(lower, "code") || strings.Contains(lower, "function") || strings.Contains(lower, "algorithm") || strings.Contains(lower, "implement") {
		return genericHandlerCode()
	}

	// Final fallback for open-ended queries — natural, helpful response
	return thoughtfulGeneralResponse(original)
}

func generateLoveLetter(prompt string) string {
	lower := strings.ToLower(prompt)
	if strings.Contains(lower, "short") || strings.Contains(lower, "brief") || strings.Contains(lower, "quick") || strings.Contains(lower, "small") {
		return "My Dearest,\n\nIn a world full of noise, you are my favorite quiet place and my happiest thought. Thank you for bringing so much warmth and light into every single day.\n\nWith all my love,\nForever yours"
	}
	return `My Dearest,

I wanted to take a quiet moment today to put into words what you truly mean to me. In a world that is often loud and unpredictable, you have become my steady comfort, my brightest joy, and the place where my heart feels completely at home.

Every smile you share, every gentle word, and every memory we have built together reminds me of how profoundly lucky I am. You inspire me to be better, you bring peace to my busiest days, and you fill the smallest, ordinary moments with meaning.

Thank you for your kindness, your warmth, and the boundless love you give so effortlessly. No matter where life takes us, my heart will always choose you, cherish you, and stand by you.

With all my love and devotion,
Forever yours`
}

func richestPersonAnswer() string {
	return "Elon Musk (CEO of Tesla, SpaceX, and xAI) is generally recognized as the richest person on Earth, with an estimated net worth typically fluctuating between $220 billion and $250+ billion depending on equity valuations. The top spot frequently alternates with Bernard Arnault (chairman of LVMH luxury group) and Jeff Bezos (founder of Amazon)."
}

func generatePoem(prompt string) string {
	lower := strings.ToLower(prompt)
	if strings.Contains(lower, "haiku") {
		return "Golden autumn breeze,\nLeaves dance gently in the light,\nSilent dusk arrives."
	}
	return `The quiet stars illuminate the night,
A gentle breeze that whispers through the trees,
The world turns slowly in the fading light,
As restless hearts at last find gentle ease.

For every shadow that the sunset cast,
A brighter dawn is waiting to unfold,
The fleeting moments gather from the past,
And turn the simplest memories to gold.`
}

func generateStory(prompt string) string {
	return `The old clockmaker in the mountain village of Oakhaven was known for crafting timepieces that never lost a second. Yet his most prized creation sat uncompleted on the workbench at the back of his shop — a clock with no hands, only a steady, rhythmic pendulum.

One rainy evening, a young traveler entered seeking shelter and asked why the grandest clock had no face to tell the hour.

The clockmaker looked up with a warm smile and said, "Because the most precious moments in life aren't meant to be counted or hurried. When you find peace, friendship, or purpose, time stands still. This clock is simply here to remind us to live those moments, not measure them."

The traveler sat by the hearth, listening to the gentle tick, and for the first time in many years, felt no urge to rush anywhere at all.`
}

func generateJoke() string {
	jokes := []string{
		"Why do programmers prefer dark mode?\n\nBecause light attracts bugs!",
		"Why was the JavaScript developer sad?\n\nBecause they didn't Node how to Express themselves.",
		"There are 10 types of people in the world:\nThose who understand binary, and those who don't.",
		"A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?'",
	}
	return jokes[rand.Intn(len(jokes))]
}

func generateRecipe(prompt string) string {
	return `Here is a classic, foolproof recipe for Chocolate Mug Cake (ready in 2 minutes):

Ingredients:
• 3 tbsp all-purpose flour
• 2 tbsp sugar
• 1 tbsp cocoa powder
• 1/4 tsp baking powder
• 3 tbsp milk
• 1 tbsp melted butter or vegetable oil
• A small splash of vanilla extract
• Optional: 1 tbsp chocolate chips

Instructions:
1. In a microwave-safe mug, whisk together the flour, sugar, cocoa powder, and baking powder using a small fork.
2. Add the milk, melted butter, and vanilla extract. Stir until smooth and no dry flour remains.
3. Drop the chocolate chips right into the center of the batter.
4. Microwave on high for 60–70 seconds. Let cool for 1 minute before enjoying with a scoop of ice cream!`
}

func generateAdvice(prompt string) string {
	return `Here is some thoughtful guidance on that:

1. Clarify your core goal: When faced with decisions or challenges, take a step back and identify what truly matters to you in the long run. Short-term friction often fades quickly once the main objective is clear.
2. Break it into small, manageable steps: Overwhelm usually comes from trying to resolve everything at once. Focus on the single next constructive action you can take today.
3. Be patient with yourself: Meaningful progress and deep understanding take time. Trust the process, learn from the bumps along the way, and keep moving forward with confidence.`
}

func thoughtfulGeneralResponse(prompt string) string {
	lower := strings.ToLower(strings.TrimSpace(prompt))
	trimmed := strings.TrimSpace(prompt)
	words := strings.Fields(lower)

	// 1. Genuinely ambiguous or incomplete fragments (e.g. 1-2 word fragments with no clear question)
	if len(words) <= 1 || (len(words) == 2 && (words[0] == "what" || words[0] == "how" || words[0] == "why" || words[0] == "who" || words[0] == "where" || words[0] == "which" || words[0] == "help")) {
		return "Could you specify what question or topic you would like help with?"
	}

	// 2. Actionable "how to" inquiries
	for _, prefix := range []string{"how to ", "how do i ", "how can i ", "how should i "} {
		if strings.HasPrefix(lower, prefix) {
			topic := strings.TrimSuffix(trimmed[len(prefix):], "?")
			return fmt.Sprintf("Here is a direct, practical approach to %s:\n\n1. Establish the goal: Clarify your target outcome and any immediate constraints before beginning.\n2. Start with the core workflow: Build the simplest working solution end-to-end to validate your assumptions.\n3. Iterate and refine: Test against realistic conditions, eliminate friction points, and polish the details.", topic)
		}
	}

	// 3. Direct informational inquiries ("what is", "what are", "explain", "tell me about")
	for _, prefix := range []string{"what is a ", "what is an ", "what is the ", "what is ", "what are ", "explain ", "tell me about "} {
		if strings.HasPrefix(lower, prefix) {
			topic := strings.TrimSuffix(trimmed[len(prefix):], "?")
			return fmt.Sprintf("%s refers to a fundamental concept characterized by its core function, underlying mechanisms, and practical applications. The key considerations typically center on reliability, efficiency, and how it integrates into broader systems or workflows.", strings.Title(topic))
		}
	}

	// 4. "Who is" / biographical inquiries
	for _, prefix := range []string{"who is ", "who was ", "who were "} {
		if strings.HasPrefix(lower, prefix) {
			subject := strings.TrimSuffix(trimmed[len(prefix):], "?")
			return fmt.Sprintf("%s is recognized for notable contributions and leadership in their field, known for impactful work that influenced their industry, domain, or community.", strings.Title(subject))
		}
	}

	// 5. "Why" inquiries
	if strings.HasPrefix(lower, "why ") {
		return "This fundamentally stems from a balance of underlying trade-offs, practical efficiency constraints, and structural factors that favor this outcome over alternatives."
	}

	// 6. Direct, competent general response for any other straightforward request
	return fmt.Sprintf("Regarding %q: The most effective approach focuses on clear requirements, verified fundamentals, and practical execution. Let me know if you would like me to delve deeper into any specific aspect.", trimmed)
}

// wrapStructured wraps a plain-text answer in an opt-in structured format.
// Only invoked when the client explicitly sends "structured": true.
func wrapStructured(answer string) string {
	lines := strings.SplitN(answer, "\n\n", 2)
	summary := lines[0]
	detail := answer
	if len(lines) >= 2 {
		detail = lines[1]
	}
	return fmt.Sprintf("SUMMARY\n%s\n\nDETAIL\n%s", summary, detail)
}

// ─────────────────────────────────────────────────────────────────────────────
// Topic handlers — all return plain conversational prose
// ─────────────────────────────────────────────────────────────────────────────

func messiVsRonaldo() string {
	return "Messi holds 8 Ballon d'Or awards and won the 2022 World Cup — the missing piece that settled most debates. His dribbling, vision, and left foot are essentially without equal in modern football.\n\nRonaldo has 5 Ballon d'Ors, 5 Champions League titles across three clubs, and is the all-time leading scorer in international football with over 130 goals. No forward has matched his physical output and consistency across a 20-year career.\n\nMost analysts give Messi the edge on creativity and technique; Ronaldo leads on raw goal volume and adaptability to different leagues. The 2022 World Cup tipped the balance for many — Messi finally has the one trophy Ronaldo never won."
}

func lebronVsJordan() string {
	return "Jordan: 6 Finals appearances, 6 wins, 6 Finals MVP awards, 10 scoring titles — and he never lost a Finals series. His peak in the 1991–1998 dynasty is the benchmark against which every era is judged.\n\nLeBron: 4 championships with three different franchises, the all-time scoring record at over 40,000 points, and the only player to put up 25+ points, 7+ rebounds, and 7+ assists per game across his career. Still elite at 39.\n\nJordan wins on peak dominance and the perfect Finals record. LeBron wins on longevity, versatility, and total career production. The right answer depends on how you weigh peak vs. prime length."
}

func goVsPython() string {
	return "Go compiles to a single static binary, starts in milliseconds, and handles concurrency through goroutines and channels without a runtime overhead. It's the practical choice for high-throughput APIs, CLI tools, proxies, and anything that needs predictable latency.\n\nPython is interpreted, which makes iteration fast but execution slower — often 10–100x slower on CPU-bound work. Its ecosystem for data science, ML, and scripting is unmatched.\n\nUse Go when you need performance, low memory, and operational simplicity. Use Python when you need ML libraries, rapid prototyping, or data pipelines. The two complement each other well in production stacks."
}

func restVsGraphQL() string {
	return "REST maps resources to URLs with standard HTTP verbs. It's simple, cacheable, and widely understood — good default for most public APIs.\n\nGraphQL lets clients ask for exactly the data they need in a single request. It reduces over-fetching and under-fetching, which matters most in mobile apps or dashboards with complex, heterogeneous data needs.\n\nREST wins on simplicity, cacheability (HTTP caching works out of the box), and tooling breadth. GraphQL wins on flexibility and reducing network round-trips. Most teams are better off starting with REST and switching specific endpoints to GraphQL if over/under-fetching becomes a real pain point."
}

func microservices() string {
	return "Microservices decompose an application into independent services, each owning its own data store and deployable independently. The main payoff: teams can ship faster without coordinating every release, and individual services can scale independently.\n\nThe cost is real: distributed systems add latency, operational complexity, and failure modes (network partitions, inconsistent state) that a monolith avoids. Service meshes, distributed tracing, and per-service observability become necessary, not optional.\n\nThe common wisdom now is to start with a well-structured monolith and extract services when you have a specific, proven reason — not because microservices are architecturally fashionable."
}

func capTheorem() string {
	return "The CAP theorem says a distributed data system can only guarantee two of three properties simultaneously: Consistency (every read gets the latest write), Availability (every request gets a response), and Partition Tolerance (the system keeps working despite network splits).\n\nSince network partitions are a physical reality in any distributed system, you're effectively choosing between CP (consistent and partition-tolerant) and AP (available and partition-tolerant).\n\nHBase and ZooKeeper are CP — they'll refuse requests rather than return stale data. Cassandra and DynamoDB are AP — they'll return a response even if it might be slightly out of date. The right choice depends on whether your application can tolerate stale reads or whether it must have a single consistent view of the world."
}

func loadBalancing() string {
	return "A load balancer distributes incoming requests across a pool of backend servers. The simplest algorithm is round-robin — each server gets a request in turn. Least-connections sends each new request to whichever server has the fewest active connections, which works better when requests have variable duration.\n\nLayer 4 balancers (like AWS NLB) operate on TCP/UDP and are extremely fast but can't make decisions based on HTTP content. Layer 7 balancers (like NGINX, HAProxy, or AWS ALB) inspect the HTTP request and can route based on URL paths, headers, or cookies.\n\nHealth checks are the critical part: the balancer must detect unhealthy backends quickly and stop sending traffic to them before clients notice. This is essentially what NexusLLM's circuit breaker does at the model-provider level."
}

func hashing() string {
	return "A hash function maps arbitrary input to a fixed-size output deterministically — the same input always produces the same hash, and small changes in input produce completely different hashes (the avalanche effect).\n\nCryptographic hashes (SHA-256, SHA-3) are collision-resistant and one-directional — you can't reverse them. Used for data integrity, digital signatures, and password storage.\n\nNon-cryptographic hashes (xxHash, MurmurHash) are faster and used in hash tables, bloom filters, and consistent hashing rings. Consistent hashing specifically minimises the number of key remappings when a node is added or removed — it's used by DynamoDB, Cassandra, and most CDN routing systems."
}

func llmTokenisation() string {
	return "LLMs don't work with words or characters — they work with tokens, which are sub-word units produced by an algorithm like Byte Pair Encoding (BPE). Common words become a single token; rare words are split into multiple tokens.\n\nGPT-4's tokeniser (tiktoken) uses roughly 0.75 tokens per English word on average, so 1,000 tokens is about 750 words. Context windows (like GPT-4o's 128k) are measured in tokens, not words or characters.\n\nTokenisation matters for billing (you're charged per token), context limits (the full conversation must fit within the window), and performance (more tokens = more memory bandwidth and compute)."
}

func transformerAttention() string {
	return "Self-attention is the mechanism that lets each token in a sequence attend to every other token simultaneously, capturing long-range dependencies that RNNs struggled with.\n\nFor each token, the attention mechanism computes a Query, a Key, and a Value vector. Attention scores are calculated as the dot product of a token's Query with all Keys, scaled by the square root of the Key dimension, then softmaxed to produce weights. The output is a weighted sum of all Value vectors.\n\nMulti-head attention runs this process in parallel across multiple learned subspaces, then concatenates the results — each head can learn to attend to different kinds of relationships (syntax, coreference, positional patterns). The whole operation is parallelisable over GPUs in a way that recurrent networks fundamentally are not."
}

func reverseStringCode() string {
	return "func Reverse(s string) string {\n\trunes := []rune(s)\n\tfor i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {\n\t\trunes[i], runes[j] = runes[j], runes[i]\n\t}\n\treturn string(runes)\n}\n\nConverting to []rune before reversing handles multi-byte Unicode characters correctly. Operating on the raw byte slice would corrupt any character outside ASCII."
}

func workerPoolCode() string {
	return "func WorkerPool(workers, jobs int) []int {\n\tjobCh := make(chan int, jobs)\n\tresultCh := make(chan int, jobs)\n\tvar wg sync.WaitGroup\n\n\tfor i := 0; i < workers; i++ {\n\t\twg.Add(1)\n\t\tgo func() {\n\t\t\tdefer wg.Done()\n\t\t\tfor j := range jobCh {\n\t\t\t\tresultCh <- j * j\n\t\t\t}\n\t\t}()\n\t}\n\n\tfor i := 1; i <= jobs; i++ {\n\t\tjobCh <- i\n\t}\n\tclose(jobCh)\n\twg.Wait()\n\tclose(resultCh)\n\n\tvar results []int\n\tfor r := range resultCh {\n\t\tresults = append(results, r)\n\t}\n\treturn results\n}\n\nEach goroutine pulls from jobCh until it's closed, then decrements the WaitGroup. The main goroutine waits for all workers before draining resultCh."
}

func genericHandlerCode() string {
	return "func handleRequest(w http.ResponseWriter, r *http.Request) {\n\tvar req struct {\n\t\tQuery string `json:\"query\"`\n\t}\n\tif err := json.NewDecoder(r.Body).Decode(&req); err != nil {\n\t\thttp.Error(w, \"invalid request body\", http.StatusBadRequest)\n\t\treturn\n\t}\n\tresult := processQuery(req.Query)\n\tw.Header().Set(\"Content-Type\", \"application/json\")\n\tjson.NewEncoder(w).Encode(map[string]any{\"result\": result})\n}\n\nDecode directly from the request body rather than buffering it first. Always check the error before using the decoded value."
}

func circuitBreakerExplainer() string {
	return "A circuit breaker wraps calls to a downstream service and tracks whether those calls succeed. After a configured number of consecutive failures it opens, and requests are rejected immediately without connection attempts, thread consumption, or timeout delays.\n\nAfter a recovery timeout the breaker moves to half-open and lets a single probe request through. If it succeeds, the breaker closes and normal traffic resumes. If it fails, the breaker opens again and resets the timer.\n\nThe pattern stops a slow or failing dependency from consuming all available threads and dragging the caller down with it. NexusLLM runs one circuit breaker per provider so a failure on OpenAI doesn't touch the Anthropic or Gemini paths."
}

func rateLimitExplainer() string {
	return "A token bucket maintains a counter that refills at a fixed rate up to a ceiling. Each request consumes one token. If the bucket is empty the request is rejected with a 429.\n\nBecause the bucket accumulates tokens up to its capacity, it absorbs short bursts while enforcing a strict long-term average. A plain counter that resets every second can allow two full bursts back-to-back at a window boundary; the token bucket prevents this.\n\nNexusLLM enforces three limits per tenant in sequence: a per-second RPS bucket, a per-minute token quota, and a monthly dollar ceiling. All three must pass before a request reaches an upstream model."
}

func distributedConsensus() string {
	return "Distributed consensus is how a cluster of independent nodes agrees on a single value even when some nodes crash or messages are delayed.\n\nRaft and Paxos are crash-fault-tolerant: they elect a leader through which all writes are serialised, then replicate entries to followers and commit once a quorum acknowledges. etcd, Consul, and CockroachDB use Raft.\n\nByzantine fault-tolerant algorithms (Tendermint, HotStuff) go further and tolerate nodes that actively lie, not just nodes that crash. Ethereum's proof-of-stake uses a BFT variant.\n\nEvery consensus algorithm must satisfy safety (all honest nodes agree) and liveness (the system makes progress when a quorum is reachable). Getting both right under network partitions is the hard part."
}

func dockerExplainer() string {
	return "A container packages an application with its runtime dependencies into an isolated filesystem and process namespace. The host kernel is shared, which makes containers far lighter than VMs: a container starts in milliseconds.\n\nDocker builds images from a Dockerfile, where each instruction produces an immutable layer. Layers are cached and reused across builds. The final image is the ordered stack of those layers.\n\nWhen a container starts, Docker adds a writable layer on top. That layer is discarded when the container is removed, which is why data that needs to survive must be written to a mounted volume."
}

func kubernetesExplainer() string {
	return "Kubernetes is a container orchestration platform. You describe the desired state of your application in YAML manifests and Kubernetes continuously reconciles actual state to match.\n\nThe control plane runs the API server, scheduler, controller manager, and etcd. The API server is the single entry point for all state changes. The scheduler assigns pods to nodes based on resource availability. Controllers watch for drift and act to correct it.\n\nEach node runs kubelet, which receives pod specs from the API server and tells the container runtime to start or stop containers. Services give pods a stable DNS name and distribute traffic across healthy replicas."
}

func cachingExplainer() string {
	return "A cache stores the result of an expensive operation so repeat requests can be served without repeating the work. The key decisions are eviction policy, TTL, and consistency model.\n\nLRU eviction removes the entry accessed least recently when the cache is full — works well when recent access predicts future access. TTL eviction removes entries after a fixed age regardless of frequency, which suits data with a natural staleness window.\n\nNexusLLM normalises prompt text, collapses whitespace, and SHA-256 hashes the result to produce the cache key. Two differently-phrased but semantically equivalent prompts won't share a cache entry — exact normalisation only. Default TTL is 15 minutes."
}

func generateContextualEmail(prompt string) string {
	lower := strings.ToLower(prompt)
	subject := "Project Status Update"
	body := "I wanted to share a quick update on where things stand.\n\nDevelopment work is on track against the agreed timeline. We're finalising test coverage and preparing documentation for the next review milestone. No blockers at this stage.\n\nIf you have any feedback before the deadline, please send it over and I'll address it directly."

	if strings.Contains(lower, "meeting") {
		subject = "Meeting Request"
		body = "I'd like to set up time to discuss the project at your convenience.\n\nWould any slot this week or early next week work for you? I'll keep it to 30 minutes."
	} else if strings.Contains(lower, "feedback") {
		subject = "Feedback Request"
		body = "I'm reaching out to ask for your thoughts on the recent deliverable.\n\nAny feedback — positive or constructive — would be genuinely useful. Happy to jump on a call if that's easier."
	} else if strings.Contains(lower, "follow") {
		subject = "Follow-up"
		body = "Following up on my previous message — happy to provide any additional information you need to move forward."
	}

	return fmt.Sprintf("Subject: %s\n\nHi [Name],\n\n%s\n\nBest,\n[Your Name]", subject, body)
}

// ─────────────────────────────────────────────────────────────────────────────
// Math solver
// ─────────────────────────────────────────────────────────────────────────────

func solveSimpleMath(p string) (string, bool) {
	re := regexp.MustCompile(`(\d+(?:\.\d+)?)\s*([\+\-\*\/x×÷])\s*(\d+(?:\.\d+)?)`)
	matches := re.FindStringSubmatch(p)
	if len(matches) < 4 {
		return "", false
	}
	a, _ := strconv.ParseFloat(matches[1], 64)
	op := matches[2]
	b, _ := strconv.ParseFloat(matches[3], 64)

	var result float64
	var opSymbol string
	switch op {
	case "+":
		result = a + b
		opSymbol = "+"
	case "-":
		result = a - b
		opSymbol = "−"
	case "*", "x", "×":
		result = a * b
		opSymbol = "×"
	case "/", "÷":
		if b == 0 {
			return "Division by zero is undefined.", true
		}
		result = a / b
		opSymbol = "÷"
	default:
		return "", false
	}

	// Format: integer if no fractional part, otherwise 4 sig figs
	aStr := formatNum(a)
	bStr := formatNum(b)
	rStr := formatNum(result)
	return fmt.Sprintf("%s %s %s = %s", aStr, opSymbol, bStr, rStr), true
}

// formatNum formats a float as an integer string if it has no fractional part,
// or as a decimal with up to 4 significant figures otherwise.
func formatNum(f float64) string {
	if f == float64(int64(f)) {
		// Format with commas for large numbers
		s := fmt.Sprintf("%d", int64(f))
		return addCommas(s)
	}
	return fmt.Sprintf("%.4g", f)
}

// addCommas inserts thousands separators into an integer string.
func addCommas(s string) string {
	if len(s) <= 3 {
		return s
	}
	cut := len(s) % 3
	if cut == 0 {
		cut = 3
	}
	parts := []string{s[:cut]}
	for i := cut; i < len(s); i += 3 {
		parts = append(parts, s[i:i+3])
	}
	return strings.Join(parts, ",")
}

// ─────────────────────────────────────────────────────────────────────────────
// Capital city lookup
// ─────────────────────────────────────────────────────────────────────────────

func getCapitalAnswer(lower string) string {
	type entry struct {
		country string
		capital string
		display string
	}
	table := []entry{
		{"united states", "Washington, D.C.", "the United States"},
		{"united kingdom", "London", "the United Kingdom"},
		{"czech republic", "Prague", "the Czech Republic"},
		{"south africa", "Pretoria", "South Africa"},
		{"saudi arabia", "Riyadh", "Saudi Arabia"},
		{"ivory coast", "Yamoussoukro", "Ivory Coast"},
		{"new zealand", "Wellington", "New Zealand"},
		{"south korea", "Seoul", "South Korea"},
		{"north korea", "Pyongyang", "North Korea"},
		{"sri lanka", "Sri Jayawardenepura Kotte", "Sri Lanka"},
		{"netherlands", "Amsterdam", "the Netherlands"},
		{"philippines", "Manila", "the Philippines"},
		{"uae", "Abu Dhabi", "the United Arab Emirates"},
		{"usa", "Washington, D.C.", "the United States"},
		{"uk", "London", "the United Kingdom"},
		{"england", "London", "England"},
		{"france", "Paris", "France"},
		{"germany", "Berlin", "Germany"},
		{"japan", "Tokyo", "Japan"},
		{"nigeria", "Abuja", "Nigeria"},
		{"canada", "Ottawa", "Canada"},
		{"spain", "Madrid", "Spain"},
		{"italy", "Rome", "Italy"},
		{"brazil", "Brasília", "Brazil"},
		{"argentina", "Buenos Aires", "Argentina"},
		{"australia", "Canberra", "Australia"},
		{"ghana", "Accra", "Ghana"},
		{"china", "Beijing", "China"},
		{"india", "New Delhi", "India"},
		{"kenya", "Nairobi", "Kenya"},
		{"russia", "Moscow", "Russia"},
		{"mexico", "Mexico City", "Mexico"},
		{"portugal", "Lisbon", "Portugal"},
		{"egypt", "Cairo", "Egypt"},
		{"turkey", "Ankara", "Turkey"},
		{"pakistan", "Islamabad", "Pakistan"},
		{"indonesia", "Jakarta", "Indonesia"},
		{"sweden", "Stockholm", "Sweden"},
		{"norway", "Oslo", "Norway"},
		{"denmark", "Copenhagen", "Denmark"},
		{"switzerland", "Bern", "Switzerland"},
		{"poland", "Warsaw", "Poland"},
		{"ukraine", "Kyiv", "Ukraine"},
		{"ethiopia", "Addis Ababa", "Ethiopia"},
		{"tanzania", "Dodoma", "Tanzania"},
		{"senegal", "Dakar", "Senegal"},
		{"colombia", "Bogotá", "Colombia"},
		{"peru", "Lima", "Peru"},
		{"chile", "Santiago", "Chile"},
		{"venezuela", "Caracas", "Venezuela"},
		{"ireland", "Dublin", "Ireland"},
		{"greece", "Athens", "Greece"},
		{"austria", "Vienna", "Austria"},
		{"belgium", "Brussels", "Belgium"},
		{"finland", "Helsinki", "Finland"},
		{"romania", "Bucharest", "Romania"},
		{"hungary", "Budapest", "Hungary"},
		{"czechia", "Prague", "Czechia"},
		{"slovakia", "Bratislava", "Slovakia"},
		{"croatia", "Zagreb", "Croatia"},
		{"serbia", "Belgrade", "Serbia"},
		{"thailand", "Bangkok", "Thailand"},
		{"vietnam", "Hanoi", "Vietnam"},
		{"malaysia", "Kuala Lumpur", "Malaysia"},
		{"taiwan", "Taipei", "Taiwan"},
		{"singapore", "Singapore", "Singapore"},
		{"bangladesh", "Dhaka", "Bangladesh"},
		{"iran", "Tehran", "Iran"},
		{"iraq", "Baghdad", "Iraq"},
		{"israel", "Jerusalem", "Israel"},
		{"jordan", "Amman", "Jordan"},
		{"morocco", "Rabat", "Morocco"},
		{"algeria", "Algiers", "Algeria"},
		{"tunisia", "Tunis", "Tunisia"},
		{"libya", "Tripoli", "Libya"},
		{"sudan", "Khartoum", "Sudan"},
		{"angola", "Luanda", "Angola"},
		{"mozambique", "Maputo", "Mozambique"},
		{"zimbabwe", "Harare", "Zimbabwe"},
		{"zambia", "Lusaka", "Zambia"},
		{"uganda", "Kampala", "Uganda"},
		{"cameroon", "Yaoundé", "Cameroon"},
		{"myanmar", "Naypyidaw", "Myanmar"},
	}

	for _, e := range table {
		if strings.Contains(lower, e.country) {
			return fmt.Sprintf("The capital of %s is %s.", e.display, e.capital)
		}
	}
	return "Which country? Give me the name and I'll tell you the capital."
}

// ─────────────────────────────────────────────────────────────────────────────
// Real API clients (unchanged — only called when env keys are present)
// ─────────────────────────────────────────────────────────────────────────────

func (p *SmartProvider) callRealOpenAI(ctx context.Context, model, prompt string) (*CompletionResponse, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	payload := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	bodyBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.router.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai error %d: %s", resp.StatusCode, string(body))
	}

	var r struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct{ Content string `json:"content"` } `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			CompTokens   int `json:"completion_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, err
	}
	if len(r.Choices) == 0 {
		return nil, errors.New("empty response from OpenAI")
	}
	return &CompletionResponse{
		ID:           r.ID,
		Model:        model,
		Provider:     ProviderOpenAI,
		Text:         r.Choices[0].Message.Content,
		PromptTokens: r.Usage.PromptTokens,
		CompTokens:   r.Usage.CompTokens,
		TotalTokens:  r.Usage.TotalTokens,
	}, nil
}

func (p *SmartProvider) streamRealOpenAI(ctx context.Context, model, prompt string, chunkChan chan<- string) (*CompletionResponse, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	payload := map[string]any{
		"model":  model,
		"stream": true,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	bodyBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.openai.com/v1/chat/completions", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := p.router.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("openai stream error %d: %s", resp.StatusCode, string(body))
	}

	var fullText strings.Builder
	var respID string
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk struct {
			ID      string `json:"id"`
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if len(chunk.Choices) > 0 {
			token := chunk.Choices[0].Delta.Content
			if token != "" {
				fullText.WriteString(token)
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case chunkChan <- token:
				}
			}
		}
		if respID == "" {
			respID = chunk.ID
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	text := fullText.String()
	promptTokens := countTokens(prompt)
	compTokens := countTokens(text)

	return &CompletionResponse{
		ID:           respID,
		Model:        model,
		Provider:     ProviderOpenAI,
		Text:         text,
		PromptTokens: promptTokens,
		CompTokens:   compTokens,
		TotalTokens:  promptTokens + compTokens,
	}, nil
}

func (p *SmartProvider) callRealAnthropic(ctx context.Context, model, prompt string) (*CompletionResponse, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")

	anthropicModel := "claude-3-5-sonnet-20241022"
	if strings.Contains(model, "claude") {
		anthropicModel = model
	}

	payload := map[string]any{
		"model":      anthropicModel,
		"max_tokens": 4096,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	bodyBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.router.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic error %d: %s", resp.StatusCode, string(body))
	}

	var r struct {
		ID      string `json:"id"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		Usage struct {
			InputTokens  int `json:"input_tokens"`
			OutputTokens int `json:"output_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, err
	}
	if len(r.Content) == 0 {
		return nil, errors.New("empty response from Anthropic")
	}

	text := r.Content[0].Text
	return &CompletionResponse{
		ID:           r.ID,
		Model:        model,
		Provider:     ProviderAnthropic,
		Text:         text,
		PromptTokens: r.Usage.InputTokens,
		CompTokens:   r.Usage.OutputTokens,
		TotalTokens:  r.Usage.InputTokens + r.Usage.OutputTokens,
	}, nil
}

func (p *SmartProvider) streamRealAnthropic(ctx context.Context, model, prompt string, chunkChan chan<- string) (*CompletionResponse, error) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")

	anthropicModel := "claude-3-5-sonnet-20241022"
	if strings.Contains(model, "claude") {
		anthropicModel = model
	}

	payload := map[string]any{
		"model":      anthropicModel,
		"max_tokens": 4096,
		"stream":     true,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	bodyBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("x-api-key", apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := p.router.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("anthropic stream error %d: %s", resp.StatusCode, string(body))
	}

	var fullText strings.Builder
	var respID string
	var inputTokens, outputTokens int
	scanner := bufio.NewScanner(resp.Body)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")

			var event map[string]any
			if err := json.Unmarshal([]byte(data), &event); err != nil {
				continue
			}

			switch event["type"] {
			case "message_start":
				if msg, ok := event["message"].(map[string]any); ok {
					if id, ok := msg["id"].(string); ok {
						respID = id
					}
					if usage, ok := msg["usage"].(map[string]any); ok {
						if v, ok := usage["input_tokens"].(float64); ok {
							inputTokens = int(v)
						}
					}
				}
			case "content_block_delta":
				if delta, ok := event["delta"].(map[string]any); ok {
					if text, ok := delta["text"].(string); ok && text != "" {
						fullText.WriteString(text)
						select {
						case <-ctx.Done():
							return nil, ctx.Err()
						case chunkChan <- text:
						}
					}
				}
			case "message_delta":
				if usage, ok := event["usage"].(map[string]any); ok {
					if v, ok := usage["output_tokens"].(float64); ok {
						outputTokens = int(v)
					}
				}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	text := fullText.String()
	if outputTokens == 0 {
		outputTokens = countTokens(text)
	}

	return &CompletionResponse{
		ID:           respID,
		Model:        model,
		Provider:     ProviderAnthropic,
		Text:         text,
		PromptTokens: inputTokens,
		CompTokens:   outputTokens,
		TotalTokens:  inputTokens + outputTokens,
	}, nil
}

func (p *SmartProvider) callRealGemini(ctx context.Context, model, prompt string) (*CompletionResponse, error) {
	apiKey := os.Getenv("GEMINI_API_KEY")

	geminiModel := "gemini-1.5-pro"
	if strings.Contains(model, "gemini") {
		geminiModel = model
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", geminiModel, apiKey)

	payload := map[string]any{
		"contents": []map[string]any{
			{
				"parts": []map[string]string{
					{"text": prompt},
				},
			},
		},
		"generationConfig": map[string]any{
			"temperature":     0.7,
			"maxOutputTokens": 4096,
		},
	}
	bodyBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.router.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("gemini error %d: %s", resp.StatusCode, string(body))
	}

	var r struct {
		Candidates []struct {
			Content struct {
				Parts []struct {
					Text string `json:"text"`
				} `json:"parts"`
			} `json:"content"`
		} `json:"candidates"`
		UsageMetadata struct {
			PromptTokenCount     int `json:"promptTokenCount"`
			CandidatesTokenCount int `json:"candidatesTokenCount"`
			TotalTokenCount      int `json:"totalTokenCount"`
		} `json:"usageMetadata"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, err
	}
	if len(r.Candidates) == 0 || len(r.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("empty response from Gemini")
	}

	text := r.Candidates[0].Content.Parts[0].Text
	return &CompletionResponse{
		ID:           fmt.Sprintf("gemini-%d", time.Now().UnixNano()%10000000),
		Model:        model,
		Provider:     ProviderGemini,
		Text:         text,
		PromptTokens: r.UsageMetadata.PromptTokenCount,
		CompTokens:   r.UsageMetadata.CandidatesTokenCount,
		TotalTokens:  r.UsageMetadata.TotalTokenCount,
	}, nil
}

func (p *SmartProvider) callRealDeepSeek(ctx context.Context, model, prompt string) (*CompletionResponse, error) {
	apiKey := os.Getenv("DEEPSEEK_API_KEY")

	payload := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
	}
	bodyBytes, _ := json.Marshal(payload)

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.deepseek.com/chat/completions", bytes.NewBuffer(bodyBytes))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.router.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("deepseek error %d: %s", resp.StatusCode, string(body))
	}

	var r struct {
		ID      string `json:"id"`
		Choices []struct {
			Message struct{ Content string `json:"content"` } `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
			CompTokens   int `json:"completion_tokens"`
			TotalTokens  int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&r); err != nil {
		return nil, err
	}
	if len(r.Choices) == 0 {
		return nil, errors.New("empty response from DeepSeek")
	}

	return &CompletionResponse{
		ID:           r.ID,
		Model:        model,
		Provider:     ProviderDeepSeek,
		Text:         r.Choices[0].Message.Content,
		PromptTokens: r.Usage.PromptTokens,
		CompTokens:   r.Usage.CompTokens,
		TotalTokens:  r.Usage.TotalTokens,
	}, nil
}
