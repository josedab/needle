//! Shared types, constants, and helper functions for GPU acceleration.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Minimum vector count for parallel processing (below this, sequential is faster)
pub(super) const PARALLEL_THRESHOLD: usize = 100;

/// Chunk size for parallel batch processing (reserved for future streaming operations)
#[allow(dead_code)]
pub(super) const PARALLEL_CHUNK_SIZE: usize = 256;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum GpuBackend {
    /// Automatically select best available backend
    #[default]
    Auto,
    /// NVIDIA CUDA
    Cuda,
    /// Apple Metal
    Metal,
    /// OpenCL (cross-platform)
    OpenCL,
    /// Vulkan compute shaders
    Vulkan,
    /// CPU fallback with SIMD
    CpuSimd,
}

/// GPU device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device ID
    pub id: usize,
    /// Device name
    pub name: String,
    /// Backend type
    pub backend: GpuBackend,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(u32, u32)>,
    /// Number of compute units
    pub compute_units: u32,
    /// Maximum work group size
    pub max_work_group_size: u32,
    /// Whether device supports FP16
    pub supports_fp16: bool,
    /// Whether device supports FP64
    pub supports_fp64: bool,
}

/// GPU memory buffer handle
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    /// Buffer ID
    pub id: u64,
    /// Size in bytes
    pub size: usize,
    /// Element count
    pub element_count: usize,
    /// Data type
    pub dtype: DataType,
    /// Whether buffer is on device
    pub on_device: bool,
}

/// Data types for GPU operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    Float16,
    Float32,
    Float64,
    Int8,
    Int32,
    Uint8,
    Uint32,
}

impl DataType {
    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            DataType::Float16 => 2,
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int8 => 1,
            DataType::Int32 => 4,
            DataType::Uint8 => 1,
            DataType::Uint32 => 4,
        }
    }
}

/// GPU kernel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelType {
    /// Euclidean distance calculation
    EuclideanDistance,
    /// Cosine similarity calculation
    CosineSimilarity,
    /// Dot product
    DotProduct,
    /// Vector normalization
    Normalize,
    /// Matrix multiplication
    MatMul,
    /// Element-wise addition
    Add,
    /// Element-wise multiplication
    Multiply,
    /// Top-K selection
    TopK,
    /// Quantization (FP32 to INT8)
    Quantize,
    /// Dequantization (INT8 to FP32)
    Dequantize,
    /// PCA projection
    PcaProject,
    /// K-means clustering step
    KMeansAssign,
}

/// GPU operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuResult {
    /// Operation succeeded
    pub success: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Memory transferred (bytes)
    pub memory_transferred: u64,
    /// GFLOPS achieved
    pub gflops: Option<f64>,
}

/// Configuration for GPU accelerator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Preferred backend
    pub preferred_backend: GpuBackend,
    /// Device ID to use (None for auto-select)
    pub device_id: Option<usize>,
    /// Memory limit in MB
    pub memory_limit_mb: u64,
    /// Enable async operations
    pub enable_async: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Enable FP16 computation
    pub use_fp16: bool,
    /// Enable kernel fusion
    pub enable_fusion: bool,
    /// Stream count for parallelism
    pub stream_count: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            preferred_backend: GpuBackend::Auto,
            device_id: None,
            memory_limit_mb: 4096,
            enable_async: true,
            batch_size: 1024,
            use_fp16: false,
            enable_fusion: true,
            stream_count: 4,
        }
    }
}

impl GpuConfig {
    /// Create a new builder
    pub fn builder() -> GpuConfigBuilder {
        GpuConfigBuilder::default()
    }
}

/// Builder for GPU configuration
#[derive(Debug, Default)]
pub struct GpuConfigBuilder {
    config: GpuConfig,
}

impl GpuConfigBuilder {
    #[must_use]
    pub fn preferred_backend(mut self, backend: GpuBackend) -> Self {
        self.config.preferred_backend = backend;
        self
    }

    #[must_use]
    pub fn device_id(mut self, id: usize) -> Self {
        self.config.device_id = Some(id);
        self
    }

    #[must_use]
    pub fn memory_limit_mb(mut self, limit: u64) -> Self {
        self.config.memory_limit_mb = limit;
        self
    }

    #[must_use]
    pub fn enable_async(mut self, enable: bool) -> Self {
        self.config.enable_async = enable;
        self
    }

    #[must_use]
    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    #[must_use]
    pub fn use_fp16(mut self, enable: bool) -> Self {
        self.config.use_fp16 = enable;
        self
    }

    #[must_use]
    pub fn enable_fusion(mut self, enable: bool) -> Self {
        self.config.enable_fusion = enable;
        self
    }

    #[must_use]
    pub fn stream_count(mut self, count: usize) -> Self {
        self.config.stream_count = count;
        self
    }

    pub fn build(self) -> GpuConfig {
        self.config
    }
}

/// Performance metrics for GPU operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuMetrics {
    /// Total operations executed
    pub total_operations: u64,
    /// Total execution time
    pub total_execution_time: Duration,
    /// Total memory transferred
    pub total_memory_transferred: u64,
    /// Peak memory usage
    pub peak_memory_usage: u64,
    /// Kernel execution counts
    pub kernel_counts: HashMap<KernelType, u64>,
    /// Average GFLOPS per kernel type
    pub avg_gflops: HashMap<KernelType, f64>,
}

/// Result from adaptive batch search
#[derive(Debug, Clone)]
pub struct AdaptiveBatchResult {
    /// Search results (index, distance)
    pub results: Vec<(usize, f32)>,
    /// Chunk size used for processing
    pub chunk_size_used: usize,
    /// Number of chunks processed
    pub chunks_processed: usize,
    /// Total processing time
    pub total_time: Duration,
    /// Throughput in vectors per second
    pub throughput_vps: f64,
}

impl AdaptiveBatchResult {
    /// Get just the indices of results
    pub fn indices(&self) -> Vec<usize> {
        self.results.iter().map(|(idx, _)| *idx).collect()
    }

    /// Get just the distances
    pub fn distances(&self) -> Vec<f32> {
        self.results.iter().map(|(_, d)| *d).collect()
    }
}

/// Quantization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
    pub min_val: f32,
    pub max_val: f32,
}

/// Distance type for searches
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceType {
    Cosine,
    Euclidean,
    DotProduct,
}

/// GPU memory pool for efficient allocation
#[derive(Debug)]
#[allow(dead_code)]
pub(super) struct MemoryPool {
    /// Total pool size
    total_size: u64,
    /// Allocated blocks
    allocated: HashMap<u64, (usize, usize)>, // buffer_id -> (offset, size)
    /// Free blocks (offset, size)
    free_blocks: Vec<(usize, usize)>,
    /// Next buffer ID
    next_buffer_id: u64,
}

impl MemoryPool {
    pub(super) fn new(size: u64) -> Self {
        Self {
            total_size: size,
            allocated: HashMap::new(),
            free_blocks: vec![(0, size as usize)],
            next_buffer_id: 0,
        }
    }

    pub(super) fn allocate(&mut self, size: usize) -> Option<u64> {
        // Find first fit
        for i in 0..self.free_blocks.len() {
            let (offset, block_size) = self.free_blocks[i];
            if block_size >= size {
                let buffer_id = self.next_buffer_id;
                self.next_buffer_id += 1;

                self.allocated.insert(buffer_id, (offset, size));

                if block_size > size {
                    self.free_blocks[i] = (offset + size, block_size - size);
                } else {
                    self.free_blocks.remove(i);
                }

                return Some(buffer_id);
            }
        }
        None
    }

    pub(super) fn deallocate(&mut self, buffer_id: u64) -> bool {
        if let Some((offset, size)) = self.allocated.remove(&buffer_id) {
            // Add back to free list (simple approach - could coalesce)
            self.free_blocks.push((offset, size));
            // Sort and coalesce adjacent blocks
            self.free_blocks.sort_by_key(|b| b.0);
            self.coalesce_free_blocks();
            true
        } else {
            false
        }
    }

    fn coalesce_free_blocks(&mut self) {
        if self.free_blocks.len() < 2 {
            return;
        }

        let mut coalesced = Vec::new();
        let mut current = self.free_blocks[0];

        for i in 1..self.free_blocks.len() {
            let next = self.free_blocks[i];
            if current.0 + current.1 == next.0 {
                // Adjacent blocks - merge
                current.1 += next.1;
            } else {
                coalesced.push(current);
                current = next;
            }
        }
        coalesced.push(current);
        self.free_blocks = coalesced;
    }

    pub(super) fn available(&self) -> u64 {
        self.free_blocks.iter().map(|(_, size)| *size as u64).sum()
    }
}

// ---------------------------------------------------------------------------
// Hardware Capability Detection & Kernel Dispatch
// ---------------------------------------------------------------------------

/// Runtime-detected hardware capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// CPU supports AVX2 SIMD
    pub has_avx2: bool,
    /// CPU supports AVX-512
    pub has_avx512: bool,
    /// CPU supports ARM NEON
    pub has_neon: bool,
    /// CUDA toolkit detected and version
    pub cuda_version: Option<String>,
    /// Metal API available (macOS/iOS)
    pub metal_available: bool,
    /// Number of CPU cores
    pub cpu_cores: u32,
    /// L2 cache size estimate in bytes
    pub l2_cache_bytes: usize,
    /// Recommended compute backend
    pub recommended_backend: GpuBackend,
}

impl HardwareCapabilities {
    /// Detect capabilities of the current host.
    pub fn detect() -> Self {
        let has_avx2 = cfg!(target_feature = "avx2") || cfg!(target_arch = "x86_64");
        let has_avx512 = cfg!(target_feature = "avx512f");
        let has_neon = cfg!(target_arch = "aarch64");
        let metal_available = cfg!(target_os = "macos") || cfg!(target_os = "ios");
        let cpu_cores = num_cpus();

        let recommended_backend = if metal_available {
            GpuBackend::Metal
        } else if has_avx512 || has_avx2 {
            GpuBackend::CpuSimd
        } else {
            GpuBackend::CpuSimd
        };

        Self {
            has_avx2,
            has_avx512,
            has_neon,
            cuda_version: None,
            metal_available,
            cpu_cores,
            l2_cache_bytes: 256 * 1024, // conservative default
            recommended_backend,
        }
    }

    /// Estimated peak distance computations per second for cosine distance.
    pub fn estimated_throughput(&self, dimensions: usize) -> f64 {
        let base = self.cpu_cores as f64 * 100_000.0; // ~100K/sec per core
        let simd_factor = if self.has_avx512 {
            8.0
        } else if self.has_avx2 {
            4.0
        } else if self.has_neon {
            2.0
        } else {
            1.0
        };
        base * simd_factor / (dimensions as f64 / 384.0) // normalize to 384-dim
    }
}

/// Selects and dispatches the appropriate kernel at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelDispatch {
    /// Scalar CPU loop
    Scalar,
    /// SIMD via wide crate
    Simd,
    /// Rayon parallel + SIMD
    ParallelSimd,
    /// GPU kernel (CUDA or Metal)
    Gpu,
}

/// Select the optimal kernel based on problem size and hardware.
pub fn select_kernel(
    num_queries: usize,
    num_vectors: usize,
    _dimensions: usize,
    caps: &HardwareCapabilities,
) -> KernelDispatch {
    let total_work = num_queries * num_vectors;

    if total_work < 1_000 {
        // Very small: overhead of parallelism not worth it
        if caps.has_avx2 || caps.has_neon {
            KernelDispatch::Simd
        } else {
            KernelDispatch::Scalar
        }
    } else if total_work < 100_000 {
        // Medium: parallel SIMD
        KernelDispatch::ParallelSimd
    } else if caps.metal_available || caps.cuda_version.is_some() {
        // Large: use GPU if available
        KernelDispatch::Gpu
    } else {
        KernelDispatch::ParallelSimd
    }
}

/// Summary of a kernel execution for profiling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelProfile {
    pub kernel: String,
    pub dispatch: String,
    pub num_queries: usize,
    pub num_vectors: usize,
    pub dimensions: usize,
    pub wall_time_ms: f64,
    pub throughput_ops_per_sec: f64,
}

// SIMD-optimized helper functions

pub(super) fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    // Process in chunks of 4 for SIMD
    let chunks = a.len() / 4;
    let mut sum = 0.0;

    for i in 0..chunks {
        let offset = i * 4;
        sum += a[offset] * b[offset]
            + a[offset + 1] * b[offset + 1]
            + a[offset + 2] * b[offset + 2]
            + a[offset + 3] * b[offset + 3];
    }

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        sum += a[i] * b[i];
    }

    sum
}

pub(super) fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let chunks = a.len() / 4;
    let mut sum = 0.0;

    for i in 0..chunks {
        let offset = i * 4;
        let d0 = a[offset] - b[offset];
        let d1 = a[offset + 1] - b[offset + 1];
        let d2 = a[offset + 2] - b[offset + 2];
        let d3 = a[offset + 3] - b[offset + 3];
        sum += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }

    for i in (chunks * 4)..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }

    sum.sqrt()
}

/// Standard cosine distance (used when query norm is not pre-computed)
#[allow(dead_code)]
pub(super) fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_a = dot_product_simd(a, a).sqrt();
    let norm_b = dot_product_simd(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

/// Optimized cosine distance with precomputed query norm
pub(super) fn cosine_distance_simd_precomputed(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_b = dot_product_simd(b, b).sqrt();

    if a_norm == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (a_norm * norm_b))
    }
}

/// Efficient partial sort for top-k selection
pub(super) fn partial_sort_top_k(distances: Vec<f32>, k: usize) -> Vec<(usize, f32)> {
    use std::cmp::Ordering;
    use std::collections::BinaryHeap;

    #[derive(PartialEq)]
    struct MaxHeapItem(usize, f32);

    impl Eq for MaxHeapItem {}

    impl PartialOrd for MaxHeapItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for MaxHeapItem {
        fn cmp(&self, other: &Self) -> Ordering {
            // Reverse ordering for max-heap (we want smallest k)
            other
                .1
                .partial_cmp(&self.1)
                .unwrap_or(Ordering::Equal)
                .reverse()
        }
    }

    let mut heap: BinaryHeap<MaxHeapItem> = BinaryHeap::with_capacity(k + 1);

    for (idx, dist) in distances.into_iter().enumerate() {
        if heap.len() < k {
            heap.push(MaxHeapItem(idx, dist));
        } else if let Some(top) = heap.peek() {
            if dist < top.1 {
                heap.pop();
                heap.push(MaxHeapItem(idx, dist));
            }
        }
    }

    let mut results: Vec<(usize, f32)> = heap.into_iter().map(|item| (item.0, item.1)).collect();
    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
    results
}

pub(super) fn normalize_vector_simd(v: &mut [f32]) {
    let norm = dot_product_simd(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Safely compute a product of usize values and convert to u64, returning an error on overflow.
pub(super) fn checked_gpu_bytes(factors: &[usize]) -> Result<u64, String> {
    let mut result: usize = 1;
    for &f in factors {
        result = result.checked_mul(f).ok_or("GPU buffer size overflow")?;
    }
    Ok(result as u64)
}

pub(super) fn calculate_gflops(flops: usize, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs > 0.0 {
        (flops as f64) / (secs * 1e9)
    } else {
        0.0
    }
}

pub(super) fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4)
}
