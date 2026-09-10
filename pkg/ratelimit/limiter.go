package ratelimit

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

var (
	ErrTenantNotFound = errors.New("invalid or missing API key")
	ErrTenantBlocked  = errors.New("tenant is blocked due to policy violations")
	ErrRateLimitExceeded = errors.New("rate limit exceeded (RPS cap reached)")
	ErrTPMExceeded    = errors.New("token throughput cap exceeded (TPM cap reached)")
	ErrBudgetExceeded = errors.New("monthly spend budget exceeded")
)

// Limiter manages multi-tenant rate limits, token quotas, and spending budgets
type Limiter struct {
	tenantsByAPIKey map[string]*Tenant
	tenantsByID     map[string]*Tenant
	mu              sync.RWMutex
}

// NewLimiter creates an instance of the rate limiting & quota manager
func NewLimiter() *Limiter {
	return &Limiter{
		tenantsByAPIKey: make(map[string]*Tenant),
		tenantsByID:     make(map[string]*Tenant),
	}
}

// RegisterTenant adds or updates a tenant
func (l *Limiter) RegisterTenant(tenant *Tenant) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.tenantsByAPIKey[tenant.APIKey] = tenant
	l.tenantsByID[tenant.ID] = tenant
}

// GetTenantByAPIKey retrieves a tenant by API key
func (l *Limiter) GetTenantByAPIKey(apiKey string) (*Tenant, error) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	t, exists := l.tenantsByAPIKey[apiKey]
	if !exists {
		return nil, ErrTenantNotFound
	}
	return t, nil
}

// GetAllTenants returns a snapshot of all tenants
func (l *Limiter) GetAllTenants() []*Tenant {
	l.mu.RLock()
	defer l.mu.RUnlock()
	var list []*Tenant
	for _, t := range l.tenantsByID {
		list = append(list, t)
	}
	return list
}

// CheckAndConsume verifies whether the tenant is permitted to make a request and consumes 1 token from RPS bucket
func (l *Limiter) CheckAndConsume(apiKey string, estimatedTokens int64) (*Tenant, error) {
	t, err := l.GetTenantByAPIKey(apiKey)
	if err != nil {
		return nil, err
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	if t.IsBlocked {
		return nil, ErrTenantBlocked
	}

	// 1. Budget Cap Check
	if t.Config.MonthlyBudgetUSD > 0 && t.TotalSpentUSD >= t.Config.MonthlyBudgetUSD {
		return nil, fmt.Errorf("%w: $%.2f spent of $%.2f limit", ErrBudgetExceeded, t.TotalSpentUSD, t.Config.MonthlyBudgetUSD)
	}

	now := time.Now()

	// 2. Token Throughput per Minute (TPM) Window Reset
	if now.Sub(t.LastTokenReset) >= time.Minute {
		t.CurrentTokensMin = 0
		t.LastTokenReset = now
	}

	if t.Config.MaxTPM > 0 && (t.CurrentTokensMin+estimatedTokens) > t.Config.MaxTPM {
		return nil, fmt.Errorf("%w: %d tokens requested, max %d TPM", ErrTPMExceeded, t.CurrentTokensMin+estimatedTokens, t.Config.MaxTPM)
	}

	// 3. Token-Bucket Leaky Rate Limiter (RPS)
	elapsed := now.Sub(t.LastBucketLeak).Seconds()
	t.RequestBucket += elapsed * t.Config.MaxRPS
	if t.RequestBucket > t.Config.MaxRPS {
		t.RequestBucket = t.Config.MaxRPS
	}
	t.LastBucketLeak = now

	if t.RequestBucket < 1.0 {
		return nil, ErrRateLimitExceeded
	}

	// Consume 1 request capacity
	t.RequestBucket -= 1.0
	t.TotalRequests++
	t.CurrentTokensMin += estimatedTokens
	t.TotalTokens += estimatedTokens

	return t, nil
}

// RecordUsage settles the exact tokens consumed and cost incurred
func (l *Limiter) RecordUsage(apiKey string, promptTokens, compTokens int, costUSD float64) {
	t, err := l.GetTenantByAPIKey(apiKey)
	if err != nil {
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()
	actualTokens := int64(promptTokens + compTokens)
	t.TotalSpentUSD += costUSD
	t.TotalTokens += actualTokens
}
