package cache

import (
	"testing"
	"time"
)

func TestCache_SetAndGet(t *testing.T) {
	c := NewCache(1 * time.Minute)

	model := "gpt-4o"
	prompt := "What is the capital of France?"
	response := "The capital of France is Paris."

	c.Set(model, prompt, response, 10, 8, 0.0003)

	item, found := c.Get(model, prompt)
	if !found {
		t.Fatalf("expected cache item to be found")
	}
	if item.Response != response {
		t.Fatalf("expected response %q, got %q", response, item.Response)
	}

	hits, misses, size, ratio := c.Stats()
	if hits != 1 || misses != 0 || size != 1 || ratio != 100.0 {
		t.Fatalf("unexpected stats: hits=%d misses=%d size=%d ratio=%.2f", hits, misses, size, ratio)
	}
}

func TestCache_WhitespaceNormalization(t *testing.T) {
	c := NewCache(1 * time.Minute)

	model := "claude-3-5-sonnet"
	prompt1 := "  Translate   hello world   to Spanish   "
	prompt2 := "Translate hello world to Spanish"
	response := "Hola Mundo"

	c.Set(model, prompt1, response, 12, 4, 0.0002)

	item, found := c.Get(model, prompt2)
	if !found {
		t.Fatalf("expected normalized prompt to hit cache")
	}
	if item.Response != response {
		t.Fatalf("expected %q, got %q", response, item.Response)
	}
}

func TestCache_Expiration(t *testing.T) {
	c := NewCache(50 * time.Millisecond)

	model := "gemini-1.5-pro"
	prompt := "Ping"
	response := "Pong"

	c.Set(model, prompt, response, 2, 2, 0.0001, 50*time.Millisecond)

	time.Sleep(70 * time.Millisecond)

	_, found := c.Get(model, prompt)
	if found {
		t.Fatalf("expected cache item to have expired")
	}
}
