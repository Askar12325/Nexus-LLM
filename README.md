# Nexus Gateway

**Production-style multi-provider LLM gateway with circuit breakers, automatic fallback, and real-time observability.**

Nexus Gateway sits in front of OpenAI, Anthropic, Google Gemini, and DeepSeek. It provides:

- Automatic failover on 429 / timeout
- Circuit breakers with CLOSED → OPEN → HALF-OPEN states
- Live SSE updates
- Prompt caching (SHA-256)
- Basic cost tracking & telemetry
- Multi-tenant rate limiting foundation

Built as a single static Go binary with an embedded UI. Zero external runtime dependencies.

---

## Quick Start

```bash
# With Docker
docker compose up --build

# Or locally
go run ./cmd/server
```
