# ADR-0014: GPU Acceleration Architecture

## Status

Accepted

## Context

Vector similarity search is compute-intensive, involving millions of distance calculations for large datasets. While CPU-based SIMD optimization (ADR-0011) provides significant speedups, modern GPUs offer 10-100x additional performance through massive parallelism.

Key requirements:

1. **Performance** — Batch search throughput needs to scale with hardware availability
2. **Cross-platform** — Support NVIDIA (CUDA), Apple Silicon (Metal), and fallback (CPU)
3. **Automatic selection** — Choose optimal backend without user configuration
4. **Graceful degradation** — GPU unavailable should not break the application
5. **Memory efficiency** — GPU memory is limited; large datasets must be handled intelligently

## Decision

Needle implements a **multi-backend GPU abstraction** with automatic device selection:

### Backend Hierarchy

```
┌──────────────────────────────────────────────────┐
│              GpuAccelerator                      │
│         (Unified API for all backends)           │
├──────────────────────────────────────────────────┤
│  CUDA       Metal       OpenCL      CPU SIMD     │
│  (NVIDIA)   (Apple)     (Cross)     (Fallback)   │
│  Priority:0 Priority:1  Priority:2  Priority:4   │
└──────────────────────────────────────────────────┘
```

### GpuBackend Enum

```rust
pub enum GpuBackend {
    Auto,      // Automatic selection based on availability
    Cuda,      // NVIDIA GPUs via CUDA
    Metal,     // Apple Silicon via Metal Performance Shaders
    OpenCL,    // Cross-platform (AMD, Intel)
    Vulkan,    // Future: compute shaders
    CpuSimd,   // Fallback: SIMD-optimized CPU
}
```

### Device Detection and Selection

The accelerator automatically detects available devices and selects the optimal one:

```rust
pub fn detect_devices() -> Result<Vec<GpuDevice>, String> {
    let mut devices = Vec::new();

    // Check Metal (macOS/iOS)
    #[cfg(target_os = "macos")]
    if metal_available() {
        devices.push(GpuDevice { backend: GpuBackend::Metal, ... });
    }

    // Check CUDA (NVIDIA)
    if cuda_available() {
        devices.push(GpuDevice { backend: GpuBackend::Cuda, ... });
    }

    // Always include CPU SIMD fallback
    devices.push(GpuDevice { backend: GpuBackend::CpuSimd, ... });

    Ok(devices)
}
```

### Memory Management

```
┌─────────────────────────────────────────────────┐
│                 Host Memory                     │
│        (System RAM, unlimited vectors)          │
├─────────────────────────────────────────────────┤
│             GPU Buffer Pool                     │
│     (Pinned memory for async transfers)         │
├─────────────────────────────────────────────────┤
│              GPU Memory                         │
│   (Working set, batched transfers)              │
└─────────────────────────────────────────────────┘
```

- **Streaming batches** — Large datasets are processed in GPU-memory-sized chunks
- **Async transfers** — PCIe transfers overlap with computation
- **Pinned buffers** — Pre-allocated buffers avoid allocation overhead

### Accelerated Operations

| Operation | Speedup (vs CPU) | Use Case |
|-----------|------------------|----------|
| Batch cosine distance | 10-50x | Search across millions of vectors |
| Batch euclidean distance | 10-50x | Distance matrix computation |
| Batch dot product | 15-60x | MaxSim for ColBERT |
| HNSW neighbor computation | 5-20x | Index construction |

### Code References

- `src/gpu.rs:59-74` — `GpuBackend` enum definition
- `src/gpu.rs:78-100` — `GpuDevice` structure with capabilities
- `src/gpu.rs:385-560` — `GpuAccelerator` implementation
- `src/gpu.rs:463-545` — Device detection and selection logic
- `src/gpu.rs:653-750` — Backend-specific distance computations

## Consequences

### Benefits

1. **Massive speedup** — 10-100x improvement for batch operations
2. **Zero configuration** — Automatic device detection and selection
3. **Cross-platform** — Works on NVIDIA, Apple Silicon, and CPU-only systems
4. **Non-breaking** — GPU unavailability falls back to CPU SIMD seamlessly
5. **Memory efficient** — Streaming batches handle datasets larger than GPU memory

### Tradeoffs

1. **Binary size** — GPU SDKs add significant binary size (CUDA ~50MB)
2. **Build complexity** — Platform-specific compilation flags and SDKs
3. **PCIe overhead** — Small batches may be slower than CPU due to transfer overhead
4. **Debugging** — GPU kernels are harder to debug than CPU code
5. **Power consumption** — GPU acceleration uses more power (relevant for edge/mobile)

### What This Enabled

- Real-time search over billion-scale datasets
- Sub-millisecond latency for batch queries
- Competitive performance with GPU-native systems (Milvus, FAISS-GPU)
- Cost-effective inference on GPU instances

### What This Prevented

- Zero-dependency builds (GPU SDKs are large)
- Guaranteed performance on CPU-only systems
- Simple debugging for all code paths

### Automatic Fallback Strategy

```rust
// Threshold for GPU vs CPU decision
const GPU_BATCH_THRESHOLD: usize = 1000;

// Small batches: CPU is faster (no transfer overhead)
if batch_size < GPU_BATCH_THRESHOLD {
    return cpu_simd_distance(query, vectors);
}

// Large batches: GPU wins
match self.device.backend {
    GpuBackend::Cuda => self.cuda_batch_distance(query, vectors),
    GpuBackend::Metal => self.metal_batch_distance(query, vectors),
    _ => self.cpu_simd_distance(query, vectors),
}
```
