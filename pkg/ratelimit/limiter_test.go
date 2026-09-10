package ratelimit

import (
	"testing"
	"time"
)

func TestLimiter_RPSAndTPM(t *testing.T) {
	l := NewLimiter()
	tenant := NewTenant("t-1", "Acme AI Corp", "nx-key-123", TierDeveloper, TenantConfig{
		MaxRPS:          2.0,
		MaxTPM:          1000,
		MonthlyBudgetUSD: 50.0,
	})
	l.RegisterTenant(tenant)

	// First two requests should pass
	_, err := l.CheckAndConsume("nx-key-123", 100)
	if err != nil {
		t.Fatalf("unexpected error on req 1: %v", err)
	}

	_, err = l.CheckAndConsume("nx-key-123", 100)
	if err != nil {
		t.Fatalf("unexpected error on req 2: %v", err)
	}

	// 3rd immediate request exceeds 2.0 RPS
	_, err = l.CheckAndConsume("nx-key-123", 100)
	if err != ErrRateLimitExceeded {
		t.Fatalf("expected ErrRateLimitExceeded, got %v", err)
	}

	// Wait for bucket leak and consume within TPM limit (200 + 500 = 700 <= 1000 TPM)
	time.Sleep(600 * time.Millisecond)
	_, err = l.CheckAndConsume("nx-key-123", 500)
	if err != nil {
		t.Fatalf("expected request after leak to succeed, got %v", err)
	}
}

func TestLimiter_BudgetExceeded(t *testing.T) {
	l := NewLimiter()
	tenant := NewTenant("t-2", "Startup Inc", "nx-key-budget", TierFree, TenantConfig{
		MaxRPS:          5.0,
		MaxTPM:          10000,
		MonthlyBudgetUSD: 5.0,
	})
	tenant.TotalSpentUSD = 5.50
	l.RegisterTenant(tenant)

	_, err := l.CheckAndConsume("nx-key-budget", 50)
	if err == nil {
		t.Fatalf("expected budget exceeded error, got nil")
	}
}
