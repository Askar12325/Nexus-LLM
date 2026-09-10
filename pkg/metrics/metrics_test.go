package metrics

import (
	"testing"
	"time"
)

func TestTelemetry_RecordAndSnapshot(t *testing.T) {
	tel := NewTelemetry()

	tel.RecordRequest(&RequestLog{
		ID:           "req-1",
		Timestamp:    time.Now(),
		TenantID:     "demo-corp",
		Model:        "gpt-4o",
		TotalTokens:  120,
		LatencyMs:    45,
		CostUSD:      0.001,
		CostSavedUSD: 0.0,
		FromCache:    false,
		Status:       "SUCCESS",
	})

	tel.RecordRequest(&RequestLog{
		ID:           "req-2",
		Timestamp:    time.Now(),
		TenantID:     "demo-corp",
		Model:        "gpt-4o",
		TotalTokens:  120,
		LatencyMs:    2,
		CostUSD:      0.0,
		CostSavedUSD: 0.001,
		FromCache:    true,
		Status:       "CACHED",
	})

	snap := tel.GetSnapshot()
	if snap.TotalRequests != 2 {
		t.Fatalf("expected 2 total requests, got %d", snap.TotalRequests)
	}
	if snap.CacheHits != 1 || snap.CacheMisses != 1 {
		t.Fatalf("expected 1 hit and 1 miss, got hits=%d misses=%d", snap.CacheHits, snap.CacheMisses)
	}
	if snap.HitRatioPct != 50.0 {
		t.Fatalf("expected 50%% hit ratio, got %.2f%%", snap.HitRatioPct)
	}
	if snap.TotalSavedUSD != 0.001 {
		t.Fatalf("expected $0.001 saved, got $%.4f", snap.TotalSavedUSD)
	}
}
