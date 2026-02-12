---
sidebar_position: 5
---

# Production Deployment

This guide covers best practices for deploying Needle in production environments.

## Deployment Options

### Embedded (Recommended for Most Cases)

Needle runs inside your application with no external dependencies:

```rust
use needle::Database;

// Opens or creates database file
let db = Database::open("/data/vectors.needle")?;
```

**Pros:**
- Zero latency for database calls
- No network overhead
- Simple deployment
- Easy backup (single file)

**Cons:**
- Single machine only
- Application must manage concurrent access

### HTTP Server Mode

For multi-language access or service-oriented architectures:

```bash
# Start the server
needle serve -a 0.0.0.0:8080 -d /data/vectors.needle
```

**Pros:**
- Language-agnostic REST API
- Can run on dedicated hardware
- Built-in connection handling

**Cons:**
- Network latency
- Additional service to manage

### Docker Deployment

```dockerfile
FROM rust:1.75-slim as builder

WORKDIR /app
COPY . .
RUN cargo build --release --features server

FROM debian:bookworm-slim

COPY --from=builder /app/target/release/needle /usr/local/bin/
EXPOSE 8080

CMD ["needle", "serve", "-a", "0.0.0.0:8080", "-d", "/data/vectors.needle"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  needle:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - needle_data:/data
    environment:
      - RUST_LOG=info
    restart: unless-stopped

volumes:
  needle_data:
```

## Configuration

### Memory Management

```rust
use needle::{Database, DatabaseConfig};

let config = DatabaseConfig::new()
    .with_mmap_threshold(10 * 1024 * 1024)  // Use mmap for files > 10MB
    .with_cache_size(1024 * 1024 * 1024);   // 1GB cache

let db = Database::open_with_config("/data/vectors.needle", config)?;
```

### Collection Sizing

```rust
use needle::{CollectionConfig, QuantizationType};

// Calculate memory requirements
let num_vectors = 10_000_000;
let dimensions = 384;
let hnsw_m = 16;

// Approximate memory per vector:
// - Vector data: dimensions * 4 bytes (or less with quantization)
// - HNSW graph: ~hnsw_m * 2 * 8 bytes
// - Metadata: varies

let bytes_per_vector = dimensions * 4 + hnsw_m * 2 * 8;
let total_memory = num_vectors * bytes_per_vector;

println!("Estimated memory: {} GB", total_memory as f64 / 1e9);

// For 10M vectors at 384 dims: ~17 GB without quantization
// With scalar quantization: ~6 GB
```

### Feature Flags

```toml
# Cargo.toml - Include only needed features
[dependencies]
needle = { version = "0.1", features = ["server", "metrics"] }
```

| Feature | Description | Production Use |
|---------|-------------|----------------|
| `simd` | Hardware-accelerated distance | Recommended |
| `server` | HTTP REST API | If needed |
| `metrics` | Prometheus metrics | Recommended |
| `hybrid` | BM25 hybrid search | If needed |

## Monitoring

### Prometheus Metrics

Enable metrics with the `metrics` feature:

```rust
use needle::metrics;

// Start metrics server
metrics::start_server("0.0.0.0:9090")?;
```

Available metrics:

| Metric | Type | Description |
|--------|------|-------------|
| `needle_search_latency_seconds` | Histogram | Search latency |
| `needle_insert_latency_seconds` | Histogram | Insert latency |
| `needle_vectors_total` | Gauge | Total vectors stored |
| `needle_collections_total` | Gauge | Number of collections |
| `needle_memory_bytes` | Gauge | Memory usage |
| `needle_search_requests_total` | Counter | Total search requests |

### Health Checks

```rust
// HTTP endpoint for health checks
async fn health_check(db: &Database) -> impl IntoResponse {
    match db.list_collections() {
        Ok(_) => (StatusCode::OK, "healthy"),
        Err(_) => (StatusCode::SERVICE_UNAVAILABLE, "unhealthy"),
    }
}
```

### Logging

```rust
use tracing::{info, warn, error};
use tracing_subscriber;

// Initialize logging
tracing_subscriber::fmt()
    .with_env_filter("needle=info")
    .json()  // JSON format for log aggregation
    .init();

// Example usage in your application
info!(collection = "documents", count = 100, "Indexed documents");
warn!(query_time_ms = 50, "Slow query detected");
```

## Performance Optimization

### 1. Index Tuning

```rust
use needle::{auto_tune, TuningConstraints, PerformanceProfile};

let constraints = TuningConstraints::new(num_vectors, dimensions)
    .with_profile(PerformanceProfile::Balanced)
    .with_memory_budget(available_memory)
    .with_latency_target_ms(10);

let tuned = auto_tune(&constraints);

let config = CollectionConfig::new("collection", dimensions)
    .with_distance(DistanceFunction::Cosine)
    .with_hnsw_m(tuned.config.hnsw_m)
    .with_hnsw_ef_construction(tuned.config.ef_construction);
```

### 2. Batch Operations

```rust
// Batch inserts for better throughput
let batch_size = 1000;
for chunk in vectors.chunks(batch_size) {
    for (id, vec, meta) in chunk {
        collection.insert(id, vec, Some(meta.clone()))?;
    }
}
db.save()?;

// Batch search for multiple queries
let results = collection.batch_search(&queries, k, None)?;
```

### 3. Connection Pooling (HTTP Server)

```rust
use axum::extract::State;
use std::sync::Arc;

struct AppState {
    db: Arc<Database>,
}

// Share database across all requests
let state = AppState {
    db: Arc::new(Database::open("/data/vectors.needle")?),
};

let app = Router::new()
    .route("/search", post(search_handler))
    .with_state(state);
```

### 4. Warm-Up on Startup

```rust
// Pre-load database into memory on startup
fn warm_up(db: &Database) -> needle::Result<()> {
    for collection_name in db.list_collections()? {
        let collection = db.collection(&collection_name)?;

        // Touch all vectors to load into cache
        let _ = collection.count()?;

        // Run a sample query to warm up HNSW
        let dummy_query = vec![0.0; collection.dimensions()];
        let _ = collection.search(&dummy_query, 1)?;
    }
    Ok(())
}
```

## High Availability

### Backup Strategy

```bash
# Simple backup - copy the single file
cp /data/vectors.needle /backups/vectors-$(date +%Y%m%d).needle

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DB_FILE="/data/vectors.needle"
RETENTION_DAYS=7

# Create backup
cp "$DB_FILE" "$BACKUP_DIR/vectors-$(date +%Y%m%d-%H%M%S).needle"

# Clean old backups
find "$BACKUP_DIR" -name "vectors-*.needle" -mtime +$RETENTION_DAYS -delete
```

### Replication (Optional)

For read scaling, replicate the database file:

```rust
use notify::{Watcher, RecursiveMode};
use std::sync::mpsc;

fn start_replication(primary_path: &str, replica_paths: &[&str]) {
    let (tx, rx) = mpsc::channel();

    let mut watcher = notify::recommended_watcher(move |res| {
        tx.send(res).unwrap();
    }).unwrap();

    watcher.watch(primary_path, RecursiveMode::NonRecursive).unwrap();

    for event in rx {
        if let Ok(event) = event {
            // Copy to replicas on change
            for replica in replica_paths {
                std::fs::copy(primary_path, replica).unwrap();
            }
        }
    }
}
```

### Read Replicas Pattern

```rust
use rand::seq::SliceRandom;

struct ReplicatedDatabase {
    primary: Database,
    replicas: Vec<Database>,
}

impl ReplicatedDatabase {
    fn search(&self, query: &[f32], k: usize) -> needle::Result<Vec<SearchResult>> {
        // Round-robin to replicas for reads
        let db = self.replicas.choose(&mut rand::thread_rng())
            .unwrap_or(&self.primary);

        let collection = db.collection("documents")?;
        collection.search(query, k)
    }

    fn insert(&self, id: &str, vector: &[f32], metadata: Value) -> needle::Result<()> {
        // Writes go to primary
        let collection = self.primary.collection("documents")?;
        collection.insert(id, vector, Some(metadata))?;
        self.primary.save()?;

        // Replicas sync via file copy (background)
        Ok(())
    }
}
```

## Security

### File Permissions

```bash
# Restrict database file access
chmod 600 /data/vectors.needle
chown app:app /data/vectors.needle
```

### Encryption at Rest

```rust
use needle::{Database, EncryptionConfig};

// Enable AES-256 encryption
let config = DatabaseConfig::new()
    .with_encryption(EncryptionConfig::aes256(&encryption_key));

let db = Database::open_with_config("/data/vectors.needle", config)?;
```

### Network Security (HTTP Server)

```rust
// Enable TLS
needle serve -a 0.0.0.0:8443 \
    --tls-cert /path/to/cert.pem \
    --tls-key /path/to/key.pem \
    -d /data/vectors.needle

// Add authentication middleware
let app = Router::new()
    .route("/search", post(search_handler))
    .layer(middleware::from_fn(auth_middleware));
```

## Troubleshooting

### High Memory Usage

1. Enable quantization
2. Reduce HNSW M parameter
3. Use memory mapping for large files
4. Compact after deletions

### Slow Queries

1. Check ef_search parameter
2. Verify index is built (not too many new vectors)
3. Consider more restrictive filters
4. Profile with `search_explain`

### Disk Space Issues

1. Run `compact()` to reclaim deleted space
2. Archive old collections
3. Use quantization

### Startup Time

1. Use memory mapping (automatic for large files)
2. Implement lazy loading
3. Use warm-up only for critical collections

## Checklist

Before going to production:

- [ ] Choose appropriate quantization
- [ ] Tune HNSW parameters for your workload
- [ ] Enable metrics and monitoring
- [ ] Set up automated backups
- [ ] Configure health checks
- [ ] Test disaster recovery
- [ ] Load test with realistic traffic
- [ ] Set up alerting on key metrics
- [ ] Document runbooks for common issues

## Next Steps

- [API Reference](/docs/api-reference) - Complete API documentation
- [HNSW Index](/docs/concepts/hnsw-index) - Tune search parameters
- [Quantization](/docs/guides/quantization) - Optimize memory usage
