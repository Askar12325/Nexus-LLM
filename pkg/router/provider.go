package router

import (
	"context"
	"nexusllm/pkg/resilience"
)

// ProviderType identifies the upstream model provider
type ProviderType string

const (
	ProviderOpenAI    ProviderType = "OpenAI"
	ProviderAnthropic ProviderType = "Anthropic"
	ProviderGemini    ProviderType = "Google Gemini"
	ProviderDeepSeek  ProviderType = "DeepSeek"
)

// ModelSpec defines metadata, capabilities, and pricing for a model
type ModelSpec struct {
	ID              string       `json:"id"`
	Name            string       `json:"name"`
	Provider        ProviderType `json:"provider"`
	CostPer1kPrompt float64      `json:"cost_per_1k_prompt"`
	CostPer1kComp   float64      `json:"cost_per_1k_comp"`
	ContextWindow   int          `json:"context_window"`
	IsStreaming     bool         `json:"is_streaming"`
}

// CompletionRequest represents standard incoming prompt request
type CompletionRequest struct {
	Model       string  `json:"model"`
	Prompt      string  `json:"prompt"`
	Stream      bool    `json:"stream"`
	Temperature float64 `json:"temperature"`
	MaxTokens   int     `json:"max_tokens"`
}

// CompletionResponse represents standard outgoing model response
type CompletionResponse struct {
	ID            string       `json:"id"`
	Model         string       `json:"model"`
	Provider      ProviderType `json:"provider"`
	Text          string       `json:"text"`
	PromptTokens  int          `json:"prompt_tokens"`
	CompTokens    int          `json:"completion_tokens"`
	TotalTokens   int          `json:"total_tokens"`
	LatencyMs     int64        `json:"latency_ms"`
	EstimatedCost float64      `json:"estimated_cost"`
	FromCache     bool         `json:"from_cache"`
	WasFallback   bool         `json:"was_fallback"`
	OriginalModel string       `json:"original_model"`
	FallbackChain []string     `json:"fallback_chain"`
}

// ProviderStats captures live telemetry and reliability numbers per provider
type ProviderStats struct {
	LastLatencyMs  int64   `json:"last_latency_ms"`
	TotalRequests  int64   `json:"total_requests"`
	TotalSuccesses int64   `json:"total_successes"`
	SuccessRatePct float64 `json:"success_rate_pct"`
	LastErrorMsg   string  `json:"last_error_msg"`
}

// Provider is the interface for upstream LLM backends
type Provider interface {
	Name() ProviderType
	CircuitBreaker() *resilience.CircuitBreaker
	Stats() ProviderStats
	Generate(ctx context.Context, model string, prompt string) (*CompletionResponse, error)
	StreamGenerate(ctx context.Context, model string, prompt string, chunkChan chan<- string) (*CompletionResponse, error)
}
