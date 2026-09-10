package resilience

import (
	"testing"
	"time"
)

func TestCircuitBreaker_TrippingAndRecovery(t *testing.T) {
	cb := NewCircuitBreaker("openai-provider", Config{
		FailureThreshold: 3,
		RecoveryTimeout:  50 * time.Millisecond,
		SuccessThreshold: 2,
	})

	if !cb.CanExecute() {
		t.Fatalf("expected initial state to allow execution")
	}

	// 2 failures should remain CLOSED
	cb.RecordFailure()
	cb.RecordFailure()
	_, state, fails, _ := cb.Status()
	if state != StateClosed || fails != 2 {
		t.Fatalf("expected CLOSED state with 2 fails, got %s (fails=%d)", state, fails)
	}

	// 3rd failure trips the breaker to OPEN
	cb.RecordFailure()
	_, state, _, trips := cb.Status()
	if state != StateOpen || trips != 1 {
		t.Fatalf("expected OPEN state after 3 fails, got %s (trips=%d)", state, trips)
	}

	// Should not allow execution during cooldown
	if cb.CanExecute() {
		t.Fatalf("expected CanExecute to return false while OPEN")
	}

	// Wait for recovery timeout -> should transition to HALF-OPEN on next check
	time.Sleep(60 * time.Millisecond)
	if !cb.CanExecute() {
		t.Fatalf("expected CanExecute to return true after timeout (entering HALF-OPEN)")
	}

	_, state, _, _ = cb.Status()
	if state != StateHalfOpen {
		t.Fatalf("expected HALF-OPEN state, got %s", state)
	}

	// 1st successful probe
	cb.RecordSuccess()
	_, state, _, _ = cb.Status()
	if state != StateHalfOpen {
		t.Fatalf("expected still HALF-OPEN after 1 success, got %s", state)
	}

	// 2nd successful probe closes the breaker
	cb.RecordSuccess()
	_, state, _, _ = cb.Status()
	if state != StateClosed {
		t.Fatalf("expected CLOSED state after reaching success threshold, got %s", state)
	}
}
