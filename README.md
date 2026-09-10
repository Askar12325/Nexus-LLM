# ⚡ NexusLLM: Enterprise Multi-Tenant AI / LLM Gateway & Resiliency Proxy

[![Go Report Card](https://goreportcard.com/badge/github.com/Askar12325/nexusllm)](https://github.com/Askar12325/nexusllm)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-100%25%20PASS-brightgreen.svg)]()

> A high-throughput, sub-millisecond AI Gateway and Reverse Proxy built in Go for enterprise multi-tenancy. Features automatic model fallback cascades, adaptive circuit breakers, token-aware rate limiting, instant SHA-256 prompt caching, PII data sanitization, and a real-time 3D Cyber-Fintech observability dashboard.

---

## 🌟 Architecture & Key Highlights

* 🛡️ **Zero-Downtime Fallback Cascades:** Seamlessly reroutes prompts from `gpt-4o` $\to$ `claude-3-5-sonnet` $\to$ `gemini-1.5-pro` $\to$ `deepseek-chat` upon upstream 429 rate limits, 5xx server errors, or timeouts.
* ⚡ **Sub-Millisecond Prompt Caching (< 2ms):** Exact and whitespace-normalized SHA-256 caching saves up to 90% in inference costs and serves cached prompts in 1ms at \$0.00 cost.
* 🔒 **Automatic PII & Privacy Redactor:** Automatically strips sensitive emails, credit cards, SSNs, phone numbers, and secret API keys before sending prompts upstream.
* 🏎️ **Token-Aware Hierarchical Rate Limiter:** Enforces dual Token Bucket limits (Requests Per Second + Tokens Per Minute) and monthly budget dollar caps per tenant API key.
* 🔌 **100% OpenAI-Compatible (`/v1/chat/completions`):** Drop-in proxy replacement for Cursor AI, LibreChat, VS Code extensions, Python scripts, and microservices.
* 📊 **3D Glassmorphism Observability Hub:** Real-time P50/P95/P99 latency histograms, live token tape, Chaos Engineering fault injector, and AI Copilot Playground.

---

## 🚀 Quick Start

### 1. Run the Gateway Locally:
```bash
git clone https://github.com/Askar12325/nexusllm.git
cd nexusllm
go run ./cmd/server/main.go
```

### 2. Open the Live Web Dashboard:
👉 **`http://localhost:8082`**

### 3. Send an Inference Request via cURL:
```bash
curl -X POST http://localhost:8082/v1/chat/completions \
  -H "Authorization: Bearer nx-key-demo-secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

---

## 🧪 Automated Unit & Chaos Tests

```bash
go test -v ./...
```
* `TestCircuitBreaker_TrippingAndRecovery`: 100% PASS
* `TestRouter_FallbackCascadeOnFailure`: 100% PASS
* `TestLimiter_RPSAndTPM`: 100% PASS
* `TestCache_WhitespaceNormalization`: 100% PASS
* `TestSanitizer_RedactsSensitiveData`: 100% PASS
* `TestTelemetry_RecordAndSnapshot`: 100% PASS

---

## 📜 License
MIT License. Built for enterprise AI reliability.
