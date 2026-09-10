package resilience

import (
	"errors"
	"sync"
	"time"
)

// State represents the state of a Circuit Breaker
type State string

const (
	StateClosed   State = "CLOSED"    // Healthy: requests pass through normally
	StateOpen     State = "OPEN"      // Tripped: requests fail fast or reroute to fallbacks
	StateHalfOpen State = "HALF-OPEN" // Probing: allowing trial request to test recovery
)

var (
	ErrCircuitOpen = errors.New("circuit breaker is open (provider unavailable)")
)

// Config defines thresholds for circuit breaker tripping and recovery
type Config struct {
	FailureThreshold int           // Consecutive errors to trip breaker (e.g. 3)
	RecoveryTimeout  time.Duration // Time to wait before entering Half-Open (e.g. 5s)
	SuccessThreshold int           // Consecutive successes to fully close breaker (e.g. 2)
}

// CircuitBreaker guards a downstream LLM provider against cascading failures
type CircuitBreaker struct {
	name             string
	cfg              Config
	state            State
	consecutiveFails int
	consecutiveWins  int
	lastStateChange  time.Time
	totalTrips       int64
	mu               sync.RWMutex
}

// NewCircuitBreaker creates a circuit breaker instance for a provider
func NewCircuitBreaker(name string, cfg Config) *CircuitBreaker {
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 3
	}
	if cfg.RecoveryTimeout <= 0 {
		cfg.RecoveryTimeout = 5 * time.Second
	}
	if cfg.SuccessThreshold <= 0 {
		cfg.SuccessThreshold = 2
	}

	return &CircuitBreaker{
		name:            name,
		cfg:             cfg,
		state:           StateClosed,
		lastStateChange: time.Now(),
	}
}

// CanExecute checks if a request is allowed to proceed to this provider
func (cb *CircuitBreaker) CanExecute() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	if cb.state == StateClosed {
		return true
	}

	if cb.state == StateOpen {
		// Check if recovery cooldown period has elapsed
		if time.Since(cb.lastStateChange) >= cb.cfg.RecoveryTimeout {
			cb.state = StateHalfOpen
			cb.lastStateChange = time.Now()
			cb.consecutiveWins = 0
			return true
		}
		return false
	}

	// In Half-Open, allow single probe request
	return true
}

// RecordSuccess records a successful provider response
func (cb *CircuitBreaker) RecordSuccess() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.consecutiveFails = 0

	if cb.state == StateHalfOpen {
		cb.consecutiveWins++
		if cb.consecutiveWins >= cb.cfg.SuccessThreshold {
			cb.state = StateClosed
			cb.lastStateChange = time.Now()
			cb.consecutiveWins = 0
		}
	}
}

// RecordFailure records an error (timeout, HTTP 429, 5xx) from the provider
func (cb *CircuitBreaker) RecordFailure() {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	cb.consecutiveFails++
	cb.consecutiveWins = 0

	if cb.state == StateHalfOpen || cb.consecutiveFails >= cb.cfg.FailureThreshold {
		cb.state = StateOpen
		cb.lastStateChange = time.Now()
		cb.totalTrips++
	}
}

// TripManually forces the breaker into OPEN state (e.g. for chaos testing or kill switch)
func (cb *CircuitBreaker) TripManually() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateOpen
	cb.lastStateChange = time.Now()
	cb.totalTrips++
}

// Reset manually restores the breaker to CLOSED state
func (cb *CircuitBreaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateClosed
	cb.consecutiveFails = 0
	cb.consecutiveWins = 0
	cb.lastStateChange = time.Now()
}

// Status returns current state snapshot
func (cb *CircuitBreaker) Status() (name string, state State, consecutiveFails int, totalTrips int64) {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.name, cb.state, cb.consecutiveFails, cb.totalTrips
}
