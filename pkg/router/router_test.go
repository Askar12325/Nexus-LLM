package router

import (
	"context"
	"testing"
)

func TestRouter_DirectExecution(t *testing.T) {
	r := NewRouter()
	ctx := context.Background()

	resp, err := r.RouteAndExecute(ctx, "gpt-4o", "Hello world")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if resp.Model != "gpt-4o" {
		t.Fatalf("expected model gpt-4o, got %s", resp.Model)
	}
	if resp.Provider != ProviderOpenAI {
		t.Fatalf("expected provider OpenAI, got %s", resp.Provider)
	}
	if resp.WasFallback {
		t.Fatalf("expected WasFallback to be false for direct execution")
	}
}

func TestRouter_FallbackCascadeOnFailure(t *testing.T) {
	r := NewRouter()
	ctx := context.Background()

	// Simulate outage on OpenAI
	r.SetSimulatedError(ProviderOpenAI, true)

	// User requests gpt-4o -> router should automatically fallback to claude-3-5-sonnet (Anthropic)
	resp, err := r.RouteAndExecute(ctx, "gpt-4o", "Analyze this system")
	if err != nil {
		t.Fatalf("expected fallback to succeed, got error: %v", err)
	}

	if !resp.WasFallback {
		t.Fatalf("expected WasFallback to be true")
	}
	if resp.Provider != ProviderAnthropic {
		t.Fatalf("expected provider to fallback to Anthropic, got %s", resp.Provider)
	}
	if resp.Model != "claude-3-5-sonnet" {
		t.Fatalf("expected model to fallback to claude-3-5-sonnet, got %s", resp.Model)
	}
	if len(resp.FallbackChain) < 2 {
		t.Fatalf("expected fallback chain to show attempted providers, got %v", resp.FallbackChain)
	}
}
