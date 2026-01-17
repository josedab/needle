# Needle Vector Database - Multi-stage Docker Build
# Produces a minimal release image for production deployment

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM rust:1.92-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Create dummy main.rs to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs && echo "" > src/lib.rs

# Build dependencies (this layer is cached if Cargo.toml/Cargo.lock don't change)
RUN cargo build --release --features full && rm -rf src

# Copy actual source code
COPY src ./src
COPY benches ./benches 2>/dev/null || true
COPY tests ./tests 2>/dev/null || true

# Touch main.rs to force rebuild with actual source
RUN touch src/main.rs

# Build release binary with server features
RUN cargo build --release --features full

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -r -s /bin/false needle

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/needle /app/needle

# Create data directory
RUN mkdir -p /data && chown needle:needle /data

# Switch to non-root user
USER needle

# Default port for HTTP server
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default data directory
VOLUME ["/data"]

# Environment variables
ENV NEEDLE_DATA_DIR=/data
ENV RUST_LOG=info

# Default command: run server
ENTRYPOINT ["/app/needle"]
CMD ["serve", "--address", "0.0.0.0:8080", "--database", "/data/needle.db"]
