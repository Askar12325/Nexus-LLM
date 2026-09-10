package ratelimit

import (
	"sync"
	"time"
)

// Tier represents the service tier of a tenant
type Tier string

const (
	TierFree       Tier = "FREE"
	TierDeveloper  Tier = "DEVELOPER"
	TierEnterprise Tier = "ENTERPRISE"
)

// TenantConfig holds limits for a tenant
type TenantConfig struct {
	MaxRPS          float64 // Maximum Requests Per Second
	MaxTPM          int64   // Maximum Tokens Per Minute
	MonthlyBudgetUSD float64 // Maximum monthly dollar spend
}

// Tenant represents an authorized API consumer with budget and quota tracking
type Tenant struct {
	ID              string       `json:"id"`
	Name            string       `json:"name"`
	APIKey          string       `json:"api_key"`
	Tier            Tier         `json:"tier"`
	Config          TenantConfig `json:"config"`
	TotalRequests   int64        `json:"total_requests"`
	TotalTokens     int64        `json:"total_tokens"`
	TotalSpentUSD   float64      `json:"total_spent_usd"`
	CurrentTokensMin int64       `json:"current_tokens_min"`
	LastTokenReset  time.Time    `json:"last_token_reset"`
	RequestBucket   float64      `json:"request_bucket"`
	LastBucketLeak  time.Time    `json:"last_bucket_leak"`
	IsBlocked       bool         `json:"is_blocked"`
	mu              sync.Mutex
}

// NewTenant initializes a new tenant with specific limits
func NewTenant(id, name, apiKey string, tier Tier, cfg TenantConfig) *Tenant {
	return &Tenant{
		ID:             id,
		Name:           name,
		APIKey:         apiKey,
		Tier:           tier,
		Config:         cfg,
		RequestBucket:  cfg.MaxRPS,
		LastBucketLeak: time.Now(),
		LastTokenReset: time.Now(),
	}
}
