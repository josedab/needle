# ann-benchmarks Docker image for Needle
# Build: docker build -t needle-ann-benchmark .
# Run: docker run needle-ann-benchmark --dataset sift-128-euclidean

FROM rust:1.85-bookworm AS builder

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src/ src/
COPY crates/ crates/

RUN cargo build --release --features simd -p needle

FROM python:3.11-slim-bookworm

# Install ann-benchmarks dependencies
RUN pip install --no-cache-dir numpy h5py

# Copy Needle binary
COPY --from=builder /app/target/release/needle /usr/local/bin/needle

# Copy benchmark scripts
COPY scripts/ann-benchmark.sh /usr/local/bin/
COPY scripts/ann-benchmark-docker/run_benchmark.py /app/

WORKDIR /app

ENTRYPOINT ["python", "run_benchmark.py"]
