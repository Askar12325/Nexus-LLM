FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy go module files
COPY go.mod ./
RUN go mod download

# Copy source tree
COPY . .

# Build statically linked binary
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -ldflags="-w -s" -o nexusllm ./cmd/server

# Final scratch/alpine runner
FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app
COPY --from=builder /app/nexusllm .

EXPOSE 8082

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:8082/health || exit 1

ENTRYPOINT ["./nexusllm"]
