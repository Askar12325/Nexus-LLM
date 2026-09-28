package router

import (
	"context"
	"strings"
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

func TestRouter_ConversationAccuracy(t *testing.T) {
	r := NewRouter()
	ctx := context.Background()

	// Test 1: "who is the richest footballer" must NOT return Elon Musk
	resp, err := r.RouteAndExecute(ctx, "gpt-4o", "who is the richest footballer")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if strings.Contains(strings.ToLower(resp.Text), "elon musk") || strings.Contains(strings.ToLower(resp.Text), "tesla") {
		t.Fatalf("CRITICAL REGRESSION: 'who is the richest footballer' returned Elon Musk/Tesla: %s", resp.Text)
	}
	if !strings.Contains(resp.Text, "Ronaldo") && !strings.Contains(resp.Text, "Bolkiah") {
		t.Fatalf("expected answer about Ronaldo or Bolkiah, got: %s", resp.Text)
	}

	// Test 2: "who is the richest man on earth"
	resp, err = r.RouteAndExecute(ctx, "gpt-4o", "who is the richest man on earth")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(resp.Text, "Elon Musk") {
		t.Fatalf("expected answer mentioning Elon Musk, got: %s", resp.Text)
	}

	// Test 3: "what is the capital of France"
	resp, err = r.RouteAndExecute(ctx, "gpt-4o", "what is the capital of France")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(resp.Text, "Paris") {
		t.Fatalf("expected answer mentioning Paris, got: %s", resp.Text)
	}

	// Test 4: "write a short love letter"
	resp, err = r.RouteAndExecute(ctx, "gpt-4o", "write a short love letter")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !strings.Contains(resp.Text, "My Dearest") || !strings.Contains(resp.Text, "love") {
		t.Fatalf("expected love letter, got: %s", resp.Text)
	}
}
