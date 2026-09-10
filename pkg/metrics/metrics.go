package metrics

import (
	"sort"
	"sync"
	"time"
)

// RequestLog represents an individual gateway execution event
type RequestLog struct {
	ID            string    `json:"id"`
	Timestamp     time.Time `json:"timestamp"`
	TenantID      string    `json:"tenant_id"`
	Model         string    `json:"model"`
	Provider      string    `json:"provider"`
	PromptSummary string    `json:"prompt_summary"`
	TotalTokens   int       `json:"total_tokens"`
	LatencyMs     int64     `json:"latency_ms"`
	CostUSD       float64   `json:"cost_usd"`
	CostSavedUSD  float64   `json:"cost_saved_usd"`
	FromCache     bool      `json:"from_cache"`
	WasFallback   bool      `json:"was_fallback"`
	Status        string    `json:"status"` // "SUCCESS", "FALLBACK", "CACHED", "RATE_LIMITED", "ERROR"
}

// Telemetry provides metrics aggregation for dashboard and Prometheus
type Telemetry struct {
	TotalRequests   int64         `json:"total_requests"`
	TotalTokens     int64         `json:"total_tokens"`
	TotalCostUSD    float64       `json:"total_cost_usd"`
	TotalSavedUSD   float64       `json:"total_saved_usd"`
	CacheHits       int64         `json:"cache_hits"`
	CacheMisses     int64         `json:"cache_misses"`
	FallbackCount   int64         `json:"fallback_count"`
	RecentLatencies []int64       `json:"-"`
	RecentLogs      []*RequestLog `json:"recent_logs"`
	mu              sync.RWMutex
}

// NewTelemetry creates a telemetry tracker instance
func NewTelemetry() *Telemetry {
	return &Telemetry{
		RecentLatencies: make([]int64, 0, 1000),
		RecentLogs:      make([]*RequestLog, 0, 50),
	}
}

// RecordRequest records an executed or cached request into telemetry
func (t *Telemetry) RecordRequest(log *RequestLog) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.TotalRequests++
	t.TotalTokens += int64(log.TotalTokens)
	t.TotalCostUSD += log.CostUSD
	t.TotalSavedUSD += log.CostSavedUSD

	if log.FromCache {
		t.CacheHits++
	} else {
		t.CacheMisses++
	}

	if log.WasFallback {
		t.FallbackCount++
	}

	t.RecentLatencies = append(t.RecentLatencies, log.LatencyMs)
	if len(t.RecentLatencies) > 500 {
		t.RecentLatencies = t.RecentLatencies[1:]
	}

	// Prepend to recent logs
	t.RecentLogs = append([]*RequestLog{log}, t.RecentLogs...)
	if len(t.RecentLogs) > 30 {
		t.RecentLogs = t.RecentLogs[:30]
	}
}

// Snapshot returns calculated P50, P95, P99 latencies, cache hit ratio, and stats
type Snapshot struct {
	TotalRequests int64         `json:"total_requests"`
	TotalTokens   int64         `json:"total_tokens"`
	TotalCostUSD  float64       `json:"total_cost_usd"`
	TotalSavedUSD float64       `json:"total_saved_usd"`
	CacheHits     int64         `json:"cache_hits"`
	CacheMisses   int64         `json:"cache_misses"`
	HitRatioPct   float64       `json:"hit_ratio_pct"`
	FallbackCount int64         `json:"fallback_count"`
	P50LatencyMs  int64         `json:"p50_latency_ms"`
	P95LatencyMs  int64         `json:"p95_latency_ms"`
	P99LatencyMs  int64         `json:"p99_latency_ms"`
	RecentLogs    []*RequestLog `json:"recent_logs"`
}

// GetSnapshot calculates latency percentiles and returns a telemetry snapshot
func (t *Telemetry) GetSnapshot() Snapshot {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var p50, p95, p99 int64
	if len(t.RecentLatencies) > 0 {
		sorted := make([]int64, len(t.RecentLatencies))
		copy(sorted, t.RecentLatencies)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

		n := len(sorted)
		p50 = sorted[int(float64(n)*0.50)]
		p95 = sorted[int(float64(n)*0.95)]
		p99 = sorted[int(float64(n)*0.99)]
	}

	totalCache := t.CacheHits + t.CacheMisses
	hitRatio := 0.0
	if totalCache > 0 {
		hitRatio = (float64(t.CacheHits) / float64(totalCache)) * 100.0
	}

	logsCopy := make([]*RequestLog, len(t.RecentLogs))
	copy(logsCopy, t.RecentLogs)

	return Snapshot{
		TotalRequests: t.TotalRequests,
		TotalTokens:   t.TotalTokens,
		TotalCostUSD:  t.TotalCostUSD,
		TotalSavedUSD: t.TotalSavedUSD,
		CacheHits:     t.CacheHits,
		CacheMisses:   t.CacheMisses,
		HitRatioPct:   hitRatio,
		FallbackCount: t.FallbackCount,
		P50LatencyMs:  p50,
		P95LatencyMs:  p95,
		P99LatencyMs:  p99,
		RecentLogs:    logsCopy,
	}
}
