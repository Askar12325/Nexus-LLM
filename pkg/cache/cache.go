package cache

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"sync"
	"time"
)

// Item represents a cached LLM response
type Item struct {
	Response     string    `json:"response"`
	Model        string    `json:"model"`
	PromptTokens int       `json:"prompt_tokens"`
	CompTokens   int       `json:"completion_tokens"`
	CostSaved    float64   `json:"cost_saved"`
	CachedAt     time.Time `json:"cached_at"`
	ExpiresAt    time.Time `json:"expires_at"`
}

// Cache provides thread-safe in-memory caching for prompt-response pairs
type Cache struct {
	items map[string]*Item
	mu    sync.RWMutex
	hits  int64
	miss  int64
	ttl   time.Duration
}

// NewCache creates a new in-memory cache instance with a default TTL
func NewCache(defaultTTL time.Duration) *Cache {
	c := &Cache{
		items: make(map[string]*Item),
		ttl:   defaultTTL,
	}
	go c.startCleanupWorker()
	return c
}

// HashPrompt creates a normalized SHA-256 hash key for a given model + prompt
func HashPrompt(model, prompt string) string {
	// Normalize: trim, lowercase whitespace, collapse internal spaces
	cleanPrompt := strings.Join(strings.Fields(strings.TrimSpace(prompt)), " ")
	hasher := sha256.New()
	hasher.Write([]byte(model + ":" + cleanPrompt))
	return hex.EncodeToString(hasher.Sum(nil))
}

// Get retrieves a cached response if present and not expired
func (c *Cache) Get(model, prompt string) (*Item, bool) {
	key := HashPrompt(model, prompt)
	c.mu.Lock()
	defer c.mu.Unlock()

	item, exists := c.items[key]
	if !exists {
		c.miss++
		return nil, false
	}

	if time.Now().After(item.ExpiresAt) {
		delete(c.items, key)
		c.miss++
		return nil, false
	}

	c.hits++
	return item, true
}

// Set stores a prompt response in the cache
func (c *Cache) Set(model, prompt, response string, promptTokens, compTokens int, costSaved float64, customTTL ...time.Duration) {
	key := HashPrompt(model, prompt)
	ttl := c.ttl
	if len(customTTL) > 0 && customTTL[0] > 0 {
		ttl = customTTL[0]
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	c.items[key] = &Item{
		Response:     response,
		Model:        model,
		PromptTokens: promptTokens,
		CompTokens:   compTokens,
		CostSaved:    costSaved,
		CachedAt:     time.Now().UTC(),
		ExpiresAt:    time.Now().UTC().Add(ttl),
	}
}

// Invalidate removes a cached item
func (c *Cache) Invalidate(model, prompt string) {
	key := HashPrompt(model, prompt)
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, key)
}

// Clear clears all cache entries
func (c *Cache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*Item)
}

// Stats returns cache telemetry
func (c *Cache) Stats() (hits int64, misses int64, size int, hitRatio float64) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	total := c.hits + c.miss
	ratio := 0.0
	if total > 0 {
		ratio = float64(c.hits) / float64(total) * 100.0
	}
	return c.hits, c.miss, len(c.items), ratio
}

func (c *Cache) startCleanupWorker() {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		c.mu.Lock()
		now := time.Now()
		for k, v := range c.items {
			if now.After(v.ExpiresAt) {
				delete(c.items, k)
			}
		}
		c.mu.Unlock()
	}
}
