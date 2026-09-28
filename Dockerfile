# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /nexusllm ./cmd/server

# Final stage
FROM alpine:3.20

RUN apk --no-cache add ca-certificates

WORKDIR /app
COPY --from=builder /nexusllm .

EXPOSE 8082

CMD ["./nexusllm"]
