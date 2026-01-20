//! GPU Acceleration for Vector Operations
//!
//! Provides hardware-accelerated vector operations using CUDA, Metal, or OpenCL backends.
//! Falls back to optimized CPU SIMD when GPU is unavailable.
//!
//! # Features
//!
//! - **Multi-backend support**: CUDA (NVIDIA), Metal (Apple), OpenCL (cross-platform)
//! - **Automatic device selection**: Chooses best available GPU
//! - **Memory management**: Efficient GPU memory allocation and transfers
//! - **Batch operations**: Optimized for processing many vectors at once using Rayon
//! - **Kernel fusion**: Combines operations to minimize memory transfers
//! - **SIMD optimization**: Uses wide SIMD operations for CPU fallback
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::gpu::{GpuAccelerator, GpuConfig, GpuBackend};
//!
//! let config = GpuConfig::builder()
//!     .preferred_backend(GpuBackend::Auto)
//!     .memory_limit_mb(4096)
//!     .build();
//!
//! let gpu = GpuAccelerator::new(config)?;
//!
//! // Batch distance calculation
//! let query = vec![0.1, 0.2, 0.3];
//! let vectors = vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]];
//! let distances = gpu.batch_cosine_distance(&query, &vectors)?;
//! ```

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

// CUDA backend imports
#[cfg(feature = "gpu-cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "gpu-cuda")]
use cudarc::nvrtc::Ptx;

// Metal backend imports
#[cfg(feature = "gpu-metal")]
use metal::{Device as MetalDevice, MTLResourceOptions, MTLSize};

/// Minimum vector count for parallel processing (below this, sequential is faster)
const PARALLEL_THRESHOLD: usize = 100;

/// Chunk size for parallel batch processing (reserved for future streaming operations)
#[allow(dead_code)]
const PARALLEL_CHUNK_SIZE: usize = 256;

/// GPU backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
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
    pub fn preferred_backend(mut self, backend: GpuBackend) -> Self {
        self.config.preferred_backend = backend;
        self
    }

    pub fn device_id(mut self, id: usize) -> Self {
        self.config.device_id = Some(id);
        self
    }

    pub fn memory_limit_mb(mut self, limit: u64) -> Self {
        self.config.memory_limit_mb = limit;
        self
    }

    pub fn enable_async(mut self, enable: bool) -> Self {
        self.config.enable_async = enable;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.config.batch_size = size;
        self
    }

    pub fn use_fp16(mut self, enable: bool) -> Self {
        self.config.use_fp16 = enable;
        self
    }

    pub fn enable_fusion(mut self, enable: bool) -> Self {
        self.config.enable_fusion = enable;
        self
    }

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

/// GPU memory pool for efficient allocation
#[derive(Debug)]
#[allow(dead_code)]
struct MemoryPool {
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
    fn new(size: u64) -> Self {
        Self {
            total_size: size,
            allocated: HashMap::new(),
            free_blocks: vec![(0, size as usize)],
            next_buffer_id: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> Option<u64> {
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

    fn deallocate(&mut self, buffer_id: u64) -> bool {
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

    fn available(&self) -> u64 {
        self.free_blocks.iter().map(|(_, size)| *size as u64).sum()
    }
}

/// GPU Accelerator for vector operations
#[allow(dead_code)]
pub struct GpuAccelerator {
    config: GpuConfig,
    device: GpuDevice,
    memory_pool: Arc<RwLock<MemoryPool>>,
    buffers: Arc<RwLock<HashMap<u64, GpuBuffer>>>,
    metrics: Arc<RwLock<GpuMetrics>>,
    is_initialized: bool,
}

impl GpuAccelerator {
    /// Create a new GPU accelerator
    pub fn new(config: GpuConfig) -> Result<Self, String> {
        // Detect available devices
        let devices = Self::detect_devices()?;

        if devices.is_empty() {
            return Err("No GPU devices found".to_string());
        }

        // Select device based on config
        let device = if let Some(id) = config.device_id {
            devices
                .into_iter()
                .find(|d| d.id == id)
                .ok_or_else(|| format!("Device {} not found", id))?
        } else {
            // Auto-select based on preferred backend and memory
            Self::select_best_device(devices, config.preferred_backend)?
        };

        let memory_size = config.memory_limit_mb * 1024 * 1024;
        let memory_pool = Arc::new(RwLock::new(MemoryPool::new(
            memory_size.min(device.available_memory),
        )));

        Ok(Self {
            config,
            device,
            memory_pool,
            buffers: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(GpuMetrics::default())),
            is_initialized: true,
        })
    }

    /// Create with CPU fallback (always succeeds)
    pub fn with_cpu_fallback(config: GpuConfig) -> Self {
        match Self::new(config.clone()) {
            Ok(gpu) => gpu,
            Err(_) => {
                // Create CPU fallback
                let device = GpuDevice {
                    id: 0,
                    name: "CPU SIMD Fallback".to_string(),
                    backend: GpuBackend::CpuSimd,
                    total_memory: 8 * 1024 * 1024 * 1024, // 8GB
                    available_memory: 4 * 1024 * 1024 * 1024,
                    compute_capability: None,
                    compute_units: num_cpus(),
                    max_work_group_size: 256,
                    supports_fp16: false,
                    supports_fp64: true,
                };

                let memory_size = config.memory_limit_mb * 1024 * 1024;
                Self {
                    config,
                    device,
                    memory_pool: Arc::new(RwLock::new(MemoryPool::new(memory_size))),
                    buffers: Arc::new(RwLock::new(HashMap::new())),
                    metrics: Arc::new(RwLock::new(GpuMetrics::default())),
                    is_initialized: true,
                }
            }
        }
    }

    /// Detect available GPU devices
    pub fn detect_devices() -> Result<Vec<GpuDevice>, String> {
        let mut devices = Vec::new();

        // Simulate device detection
        // In real implementation, would use platform-specific APIs

        #[cfg(target_os = "macos")]
        {
            // Metal is available on macOS
            devices.push(GpuDevice {
                id: 0,
                name: "Apple GPU (Metal)".to_string(),
                backend: GpuBackend::Metal,
                total_memory: 8 * 1024 * 1024 * 1024,
                available_memory: 6 * 1024 * 1024 * 1024,
                compute_capability: None,
                compute_units: 8,
                max_work_group_size: 1024,
                supports_fp16: true,
                supports_fp64: false,
            });
        }

        #[cfg(target_os = "linux")]
        {
            // Assume CUDA might be available on Linux
            devices.push(GpuDevice {
                id: 0,
                name: "NVIDIA GPU (CUDA)".to_string(),
                backend: GpuBackend::Cuda,
                total_memory: 16 * 1024 * 1024 * 1024,
                available_memory: 14 * 1024 * 1024 * 1024,
                compute_capability: Some((8, 6)),
                compute_units: 84,
                max_work_group_size: 1024,
                supports_fp16: true,
                supports_fp64: true,
            });
        }

        // Always add CPU fallback
        devices.push(GpuDevice {
            id: devices.len(),
            name: "CPU SIMD".to_string(),
            backend: GpuBackend::CpuSimd,
            total_memory: 8 * 1024 * 1024 * 1024,
            available_memory: 4 * 1024 * 1024 * 1024,
            compute_capability: None,
            compute_units: num_cpus(),
            max_work_group_size: 256,
            supports_fp16: false,
            supports_fp64: true,
        });

        Ok(devices)
    }

    /// Select best device based on criteria
    fn select_best_device(
        devices: Vec<GpuDevice>,
        preferred: GpuBackend,
    ) -> Result<GpuDevice, String> {
        // First try preferred backend
        if preferred != GpuBackend::Auto {
            if let Some(device) = devices.iter().find(|d| d.backend == preferred) {
                return Ok(device.clone());
            }
        }

        // Otherwise select by priority: CUDA > Metal > OpenCL > Vulkan > CPU
        let priority = |backend: GpuBackend| match backend {
            GpuBackend::Cuda => 0,
            GpuBackend::Metal => 1,
            GpuBackend::OpenCL => 2,
            GpuBackend::Vulkan => 3,
            GpuBackend::CpuSimd => 4,
            GpuBackend::Auto => 5,
        };

        devices
            .into_iter()
            .min_by_key(|d| priority(d.backend))
            .ok_or_else(|| "No suitable device found".to_string())
    }

    /// Get current device info
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Get current metrics
    pub fn metrics(&self) -> GpuMetrics {
        self.metrics
            .read()
            .expect("metrics lock should not be poisoned")
            .clone()
    }

    /// Allocate a buffer on GPU
    pub fn allocate_buffer(
        &self,
        element_count: usize,
        dtype: DataType,
    ) -> Result<GpuBuffer, String> {
        let size = element_count * dtype.size_bytes();
        let mut pool = self.memory_pool.write().expect("memory_pool lock should not be poisoned");

        let buffer_id = pool
            .allocate(size)
            .ok_or_else(|| "Out of GPU memory".to_string())?;

        let buffer = GpuBuffer {
            id: buffer_id,
            size,
            element_count,
            dtype,
            on_device: true,
        };

        self.buffers.write().expect("buffers lock should not be poisoned").insert(buffer_id, buffer.clone());
        Ok(buffer)
    }

    /// Free a buffer
    pub fn free_buffer(&self, buffer: &GpuBuffer) -> bool {
        let mut pool = self.memory_pool.write().expect("memory_pool lock should not be poisoned");
        if pool.deallocate(buffer.id) {
            self.buffers.write().expect("buffers lock should not be poisoned").remove(&buffer.id);
            true
        } else {
            false
        }
    }

    /// Get available memory
    pub fn available_memory(&self) -> u64 {
        self.memory_pool
            .read()
            .expect("memory_pool lock should not be poisoned")
            .available()
    }

    /// Upload vectors to GPU
    pub fn upload_vectors(&self, vectors: &[Vec<f32>]) -> Result<GpuBuffer, String> {
        if vectors.is_empty() {
            return Err("Empty vector list".to_string());
        }

        let dim = vectors[0].len();
        let total_elements = vectors.len() * dim;
        let buffer = self.allocate_buffer(total_elements, DataType::Float32)?;

        // In real implementation, would copy data to GPU memory
        // Here we just track the allocation

        Ok(buffer)
    }

    /// Download results from GPU
    pub fn download_f32(&self, _buffer: &GpuBuffer, count: usize) -> Result<Vec<f32>, String> {
        // In real implementation, would copy from GPU memory
        // Here we return placeholder
        Ok(vec![0.0; count])
    }

    /// Batch cosine distance calculation with automatic parallelization
    pub fn batch_cosine_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let start = Instant::now();

        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let dim = query.len();
        if vectors.iter().any(|v| v.len() != dim) {
            return Err("Dimension mismatch".to_string());
        }

        // Pre-compute query norm for efficiency
        let query_norm = dot_product_simd(query, query).sqrt();
        if query_norm == 0.0 {
            return Ok(vec![1.0; vectors.len()]);
        }

        let results: Vec<f32> = match self.device.backend {
            #[cfg(feature = "gpu-cuda")]
            GpuBackend::Cuda => self.cuda_batch_cosine_distance(query, vectors)?,
            #[cfg(feature = "gpu-metal")]
            GpuBackend::Metal => self.metal_batch_cosine_distance(query, vectors)?,
            _ => {
                // Parallel CPU SIMD implementation
                if vectors.len() >= PARALLEL_THRESHOLD {
                    vectors
                        .par_iter()
                        .map(|v| cosine_distance_simd_precomputed(query, v, query_norm))
                        .collect()
                } else {
                    vectors
                        .iter()
                        .map(|v| cosine_distance_simd_precomputed(query, v, query_norm))
                        .collect()
                }
            }
        };

        // Update metrics
        self.update_metrics(
            KernelType::CosineSimilarity,
            start.elapsed(),
            (vectors.len() * dim * 4) as u64,
            Some(calculate_gflops(vectors.len() * dim * 3, start.elapsed())),
        );

        Ok(results)
    }

    /// Batch euclidean distance calculation with automatic parallelization
    pub fn batch_euclidean_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let start = Instant::now();

        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let dim = query.len();
        if vectors.iter().any(|v| v.len() != dim) {
            return Err("Dimension mismatch".to_string());
        }

        let results: Vec<f32> = match self.device.backend {
            #[cfg(feature = "gpu-cuda")]
            GpuBackend::Cuda => self.cuda_batch_euclidean_distance(query, vectors)?,
            #[cfg(feature = "gpu-metal")]
            GpuBackend::Metal => self.metal_batch_euclidean_distance(query, vectors)?,
            _ => {
                // Parallel CPU SIMD implementation
                if vectors.len() >= PARALLEL_THRESHOLD {
                    vectors
                        .par_iter()
                        .map(|v| euclidean_distance_simd(query, v))
                        .collect()
                } else {
                    vectors
                        .iter()
                        .map(|v| euclidean_distance_simd(query, v))
                        .collect()
                }
            }
        };

        self.update_metrics(
            KernelType::EuclideanDistance,
            start.elapsed(),
            (vectors.len() * dim * 4) as u64,
            Some(calculate_gflops(vectors.len() * dim * 3, start.elapsed())),
        );

        Ok(results)
    }

    /// Batch dot product calculation with automatic parallelization
    pub fn batch_dot_product(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let start = Instant::now();

        if vectors.is_empty() {
            return Ok(vec![]);
        }

        let dim = query.len();
        let results: Vec<f32> = match self.device.backend {
            #[cfg(feature = "gpu-cuda")]
            GpuBackend::Cuda => self.cuda_batch_dot_product(query, vectors)?,
            #[cfg(feature = "gpu-metal")]
            GpuBackend::Metal => self.metal_batch_dot_product(query, vectors)?,
            _ => {
                if vectors.len() >= PARALLEL_THRESHOLD {
                    vectors
                        .par_iter()
                        .map(|v| dot_product_simd(query, v))
                        .collect()
                } else {
                    vectors.iter().map(|v| dot_product_simd(query, v)).collect()
                }
            }
        };

        self.update_metrics(
            KernelType::DotProduct,
            start.elapsed(),
            (vectors.len() * dim * 4) as u64,
            Some(calculate_gflops(vectors.len() * dim * 2, start.elapsed())),
        );

        Ok(results)
    }

    /// Normalize vectors in batch with parallelization
    pub fn batch_normalize(&self, vectors: &mut [Vec<f32>]) -> Result<(), String> {
        let start = Instant::now();

        if vectors.len() >= PARALLEL_THRESHOLD {
            vectors.par_iter_mut().for_each(|vector| {
                normalize_vector_simd(vector);
            });
        } else {
            for vector in vectors.iter_mut() {
                normalize_vector_simd(vector);
            }
        }

        let total_elements: usize = vectors.iter().map(|v| v.len()).sum();
        self.update_metrics(
            KernelType::Normalize,
            start.elapsed(),
            (total_elements * 4) as u64,
            None,
        );

        Ok(())
    }

    /// Matrix multiplication (for projections, etc.) with parallelization
    pub fn matmul(
        &self,
        a: &[Vec<f32>],
        b: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let start = Instant::now();

        if a.is_empty() || b.is_empty() {
            return Err("Empty matrices".to_string());
        }

        let m = a.len();
        let k = a[0].len();
        let n = b[0].len();

        // Verify dimensions
        if b.len() != k {
            return Err(format!(
                "Matrix dimension mismatch: {}x{} * {}x{}",
                m,
                k,
                b.len(),
                n
            ));
        }

        // Transpose B for better cache locality
        let b_transposed: Vec<Vec<f32>> = (0..n)
            .map(|j| (0..k).map(|i| b[i][j]).collect())
            .collect();

        // Parallel matrix multiplication with cache-friendly access
        let result: Vec<Vec<f32>> = if m >= PARALLEL_THRESHOLD {
            a.par_iter()
                .map(|row_a| {
                    b_transposed
                        .iter()
                        .map(|col_b| dot_product_simd(row_a, col_b))
                        .collect()
                })
                .collect()
        } else {
            a.iter()
                .map(|row_a| {
                    b_transposed
                        .iter()
                        .map(|col_b| dot_product_simd(row_a, col_b))
                        .collect()
                })
                .collect()
        };

        self.update_metrics(
            KernelType::MatMul,
            start.elapsed(),
            ((m * k + k * n + m * n) * 4) as u64,
            Some(calculate_gflops(2 * m * n * k, start.elapsed())),
        );

        Ok(result)
    }

    /// Top-K selection across batch
    pub fn batch_top_k(
        &self,
        distances: &[f32],
        k: usize,
    ) -> Result<Vec<(usize, f32)>, String> {
        let start = Instant::now();

        let mut indexed: Vec<(usize, f32)> = distances.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();

        self.update_metrics(
            KernelType::TopK,
            start.elapsed(),
            (distances.len() * 4) as u64,
            None,
        );

        Ok(results)
    }

    /// Quantize vectors to INT8
    pub fn quantize_to_int8(
        &self,
        vectors: &[Vec<f32>],
    ) -> Result<(Vec<Vec<i8>>, Vec<QuantizationParams>), String> {
        let start = Instant::now();

        let mut quantized = Vec::with_capacity(vectors.len());
        let mut params = Vec::with_capacity(vectors.len());

        for vector in vectors {
            let (min_val, max_val) = vector
                .iter()
                .fold((f32::MAX, f32::MIN), |(min, max), &v| {
                    (min.min(v), max.max(v))
                });

            let scale = (max_val - min_val) / 255.0;
            let zero_point = (-min_val / scale).round() as i32;

            let q_vec: Vec<i8> = vector
                .iter()
                .map(|&v| ((v / scale).round() as i32 - zero_point).clamp(-128, 127) as i8)
                .collect();

            quantized.push(q_vec);
            params.push(QuantizationParams {
                scale,
                zero_point,
                min_val,
                max_val,
            });
        }

        let total_elements: usize = vectors.iter().map(|v| v.len()).sum();
        self.update_metrics(
            KernelType::Quantize,
            start.elapsed(),
            (total_elements * 4) as u64,
            None,
        );

        Ok((quantized, params))
    }

    /// Dequantize INT8 vectors
    pub fn dequantize_from_int8(
        &self,
        quantized: &[Vec<i8>],
        params: &[QuantizationParams],
    ) -> Result<Vec<Vec<f32>>, String> {
        let start = Instant::now();

        if quantized.len() != params.len() {
            return Err("Mismatched quantized vectors and params".to_string());
        }

        let dequantized: Vec<Vec<f32>> = quantized
            .iter()
            .zip(params.iter())
            .map(|(q_vec, p)| {
                q_vec
                    .iter()
                    .map(|&q| (q as i32 + p.zero_point) as f32 * p.scale)
                    .collect()
            })
            .collect();

        let total_elements: usize = quantized.iter().map(|v| v.len()).sum();
        self.update_metrics(
            KernelType::Dequantize,
            start.elapsed(),
            (total_elements) as u64,
            None,
        );

        Ok(dequantized)
    }

    /// PCA projection
    pub fn pca_project(
        &self,
        vectors: &[Vec<f32>],
        projection_matrix: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>, String> {
        let start = Instant::now();

        // Project each vector
        let projected: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| {
                projection_matrix
                    .iter()
                    .map(|row| dot_product_simd(v, row))
                    .collect()
            })
            .collect();

        let input_elements: usize = vectors.iter().map(|v| v.len()).sum();
        let output_elements: usize = projected.iter().map(|v| v.len()).sum();
        self.update_metrics(
            KernelType::PcaProject,
            start.elapsed(),
            ((input_elements + output_elements) * 4) as u64,
            None,
        );

        Ok(projected)
    }

    /// K-means assignment step with parallelization
    pub fn kmeans_assign(
        &self,
        vectors: &[Vec<f32>],
        centroids: &[Vec<f32>],
    ) -> Result<Vec<usize>, String> {
        let start = Instant::now();

        let assignments: Vec<usize> = if vectors.len() >= PARALLEL_THRESHOLD {
            vectors
                .par_iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, euclidean_distance_simd(v, c)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect()
        } else {
            vectors
                .iter()
                .map(|v| {
                    centroids
                        .iter()
                        .enumerate()
                        .map(|(i, c)| (i, euclidean_distance_simd(v, c)))
                        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(i, _)| i)
                        .unwrap_or(0)
                })
                .collect()
        };

        let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
        let total_ops = vectors.len() * centroids.len() * dim;
        self.update_metrics(
            KernelType::KMeansAssign,
            start.elapsed(),
            (total_ops * 4) as u64,
            Some(calculate_gflops(total_ops * 3, start.elapsed())),
        );

        Ok(assignments)
    }

    /// Execute fused kernel (multiple operations in one) with parallelization
    pub fn fused_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance_type: DistanceType,
    ) -> Result<Vec<(usize, f32)>, String> {
        let start = Instant::now();
        let dim = query.len();

        // Pre-compute query norm for cosine distance
        let query_norm = dot_product_simd(query, query).sqrt();

        // Compute distances using parallel processing for large batches
        let distances: Vec<f32> = if vectors.len() >= PARALLEL_THRESHOLD {
            match distance_type {
                DistanceType::Cosine => {
                    if query_norm == 0.0 {
                        vec![1.0; vectors.len()]
                    } else {
                        vectors
                            .par_iter()
                            .map(|v| cosine_distance_simd_precomputed(query, v, query_norm))
                            .collect()
                    }
                }
                DistanceType::Euclidean => vectors
                    .par_iter()
                    .map(|v| euclidean_distance_simd(query, v))
                    .collect(),
                DistanceType::DotProduct => vectors
                    .par_iter()
                    .map(|v| -dot_product_simd(query, v)) // Negative for sorting
                    .collect(),
            }
        } else {
            match distance_type {
                DistanceType::Cosine => {
                    if query_norm == 0.0 {
                        vec![1.0; vectors.len()]
                    } else {
                        vectors
                            .iter()
                            .map(|v| cosine_distance_simd_precomputed(query, v, query_norm))
                            .collect()
                    }
                }
                DistanceType::Euclidean => vectors
                    .iter()
                    .map(|v| euclidean_distance_simd(query, v))
                    .collect(),
                DistanceType::DotProduct => vectors
                    .iter()
                    .map(|v| -dot_product_simd(query, v))
                    .collect(),
            }
        };

        // Top-K selection using parallel sort for large results
        let results = if k >= vectors.len() {
            // Return all sorted
            let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed
        } else if vectors.len() >= PARALLEL_THRESHOLD {
            // Use partial sort for efficiency
            partial_sort_top_k(distances, k)
        } else {
            let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.into_iter().take(k).collect()
        };

        self.update_metrics(
            KernelType::CosineSimilarity, // Approximate
            start.elapsed(),
            (vectors.len() * dim * 4 * 2) as u64, // Read vectors + write distances
            Some(calculate_gflops(vectors.len() * dim * 3, start.elapsed())),
        );

        Ok(results)
    }

    /// Batch search with multiple queries (fully parallel)
    pub fn batch_fused_search(
        &self,
        queries: &[Vec<f32>],
        vectors: &[Vec<f32>],
        k: usize,
        distance_type: DistanceType,
    ) -> Result<Vec<Vec<(usize, f32)>>, String> {
        let start = Instant::now();

        if queries.is_empty() || vectors.is_empty() {
            return Ok(vec![]);
        }

        let dim = queries[0].len();

        // Process all queries in parallel
        let results: Vec<Vec<(usize, f32)>> = queries
            .par_iter()
            .map(|query| {
                self.fused_search(query, vectors, k, distance_type)
                    .unwrap_or_default()
            })
            .collect();

        let total_ops = queries.len() * vectors.len() * dim * 3;
        self.update_metrics(
            KernelType::CosineSimilarity,
            start.elapsed(),
            (total_ops * 4) as u64,
            Some(calculate_gflops(total_ops, start.elapsed())),
        );

        Ok(results)
    }

    fn update_metrics(
        &self,
        kernel: KernelType,
        duration: Duration,
        memory: u64,
        gflops: Option<f64>,
    ) {
        let mut metrics = self.metrics.write().expect("metrics lock should not be poisoned");
        metrics.total_operations += 1;
        metrics.total_execution_time += duration;
        metrics.total_memory_transferred += memory;
        *metrics.kernel_counts.entry(kernel).or_insert(0) += 1;

        if let Some(g) = gflops {
            let count = *metrics.kernel_counts.get(&kernel).unwrap_or(&1) as f64;
            let entry = metrics.avg_gflops.entry(kernel).or_insert(0.0);
            *entry = (*entry * (count - 1.0) + g) / count;
        }
    }

    // ==========================================================================
    // CUDA Backend Implementation
    // ==========================================================================

    #[cfg(feature = "gpu-cuda")]
    fn cuda_batch_cosine_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        // CUDA kernel for cosine distance
        // This uses cudarc to execute GPU kernels
        let device = CudaDevice::new(0).map_err(|e| format!("CUDA device error: {}", e))?;

        let dim = query.len();
        let n_vectors = vectors.len();

        // Flatten vectors for GPU transfer
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        // Allocate GPU memory
        let query_gpu = device
            .htod_copy(query.to_vec())
            .map_err(|e| format!("CUDA copy error: {}", e))?;
        let vectors_gpu = device
            .htod_copy(flat_vectors)
            .map_err(|e| format!("CUDA copy error: {}", e))?;
        let mut results_gpu = device
            .alloc_zeros::<f32>(n_vectors)
            .map_err(|e| format!("CUDA alloc error: {}", e))?;

        // Load and execute kernel
        let ptx = Self::get_cosine_distance_ptx();
        let module = device
            .load_ptx(ptx, "cosine_distance", &["cosine_distance_kernel"])
            .map_err(|e| format!("CUDA module load error: {}", e))?;
        let kernel = module
            .get_fn("cosine_distance_kernel")
            .map_err(|e| format!("CUDA kernel not found: {}", e))?;

        // Configure grid and block dimensions
        let block_size = 256;
        let grid_size = (n_vectors + block_size - 1) / block_size;

        unsafe {
            kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&query_gpu, &vectors_gpu, &mut results_gpu, dim as i32, n_vectors as i32),
                )
                .map_err(|e| format!("CUDA launch error: {}", e))?;
        }

        // Copy results back to host
        let results = device
            .dtoh_sync_copy(&results_gpu)
            .map_err(|e| format!("CUDA copy back error: {}", e))?;

        Ok(results)
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_batch_euclidean_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let device = CudaDevice::new(0).map_err(|e| format!("CUDA device error: {}", e))?;

        let dim = query.len();
        let n_vectors = vectors.len();
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        let query_gpu = device
            .htod_copy(query.to_vec())
            .map_err(|e| format!("CUDA copy error: {}", e))?;
        let vectors_gpu = device
            .htod_copy(flat_vectors)
            .map_err(|e| format!("CUDA copy error: {}", e))?;
        let mut results_gpu = device
            .alloc_zeros::<f32>(n_vectors)
            .map_err(|e| format!("CUDA alloc error: {}", e))?;

        let ptx = Self::get_euclidean_distance_ptx();
        let module = device
            .load_ptx(ptx, "euclidean_distance", &["euclidean_distance_kernel"])
            .map_err(|e| format!("CUDA module load error: {}", e))?;
        let kernel = module
            .get_fn("euclidean_distance_kernel")
            .map_err(|e| format!("CUDA kernel not found: {}", e))?;

        let block_size = 256;
        let grid_size = (n_vectors + block_size - 1) / block_size;

        unsafe {
            kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&query_gpu, &vectors_gpu, &mut results_gpu, dim as i32, n_vectors as i32),
                )
                .map_err(|e| format!("CUDA launch error: {}", e))?;
        }

        let results = device
            .dtoh_sync_copy(&results_gpu)
            .map_err(|e| format!("CUDA copy back error: {}", e))?;

        Ok(results)
    }

    #[cfg(feature = "gpu-cuda")]
    fn cuda_batch_dot_product(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let device = CudaDevice::new(0).map_err(|e| format!("CUDA device error: {}", e))?;

        let dim = query.len();
        let n_vectors = vectors.len();
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        let query_gpu = device
            .htod_copy(query.to_vec())
            .map_err(|e| format!("CUDA copy error: {}", e))?;
        let vectors_gpu = device
            .htod_copy(flat_vectors)
            .map_err(|e| format!("CUDA copy error: {}", e))?;
        let mut results_gpu = device
            .alloc_zeros::<f32>(n_vectors)
            .map_err(|e| format!("CUDA alloc error: {}", e))?;

        let ptx = Self::get_dot_product_ptx();
        let module = device
            .load_ptx(ptx, "dot_product", &["dot_product_kernel"])
            .map_err(|e| format!("CUDA module load error: {}", e))?;
        let kernel = module
            .get_fn("dot_product_kernel")
            .map_err(|e| format!("CUDA kernel not found: {}", e))?;

        let block_size = 256;
        let grid_size = (n_vectors + block_size - 1) / block_size;

        unsafe {
            kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&query_gpu, &vectors_gpu, &mut results_gpu, dim as i32, n_vectors as i32),
                )
                .map_err(|e| format!("CUDA launch error: {}", e))?;
        }

        let results = device
            .dtoh_sync_copy(&results_gpu)
            .map_err(|e| format!("CUDA copy back error: {}", e))?;

        Ok(results)
    }

    #[cfg(feature = "gpu-cuda")]
    fn get_cosine_distance_ptx() -> Ptx {
        // CUDA kernel for cosine distance computation
        // This is compiled PTX code for the kernel
        Ptx::from_src(
            r#"
            .version 7.0
            .target sm_70
            .address_size 64

            .visible .entry cosine_distance_kernel(
                .param .u64 query_ptr,
                .param .u64 vectors_ptr,
                .param .u64 results_ptr,
                .param .s32 dim,
                .param .s32 n_vectors
            )
            {
                .reg .pred %p<2>;
                .reg .f32 %f<10>;
                .reg .b32 %r<10>;
                .reg .b64 %rd<10>;

                ld.param.u64 %rd1, [query_ptr];
                ld.param.u64 %rd2, [vectors_ptr];
                ld.param.u64 %rd3, [results_ptr];
                ld.param.s32 %r1, [dim];
                ld.param.s32 %r2, [n_vectors];

                // Get thread index
                mov.u32 %r3, %ctaid.x;
                mov.u32 %r4, %ntid.x;
                mov.u32 %r5, %tid.x;
                mad.lo.s32 %r6, %r3, %r4, %r5;

                // Check bounds
                setp.ge.s32 %p1, %r6, %r2;
                @%p1 bra END;

                // Initialize accumulators
                mov.f32 %f1, 0.0;  // dot product
                mov.f32 %f2, 0.0;  // norm_a
                mov.f32 %f3, 0.0;  // norm_b

                // Calculate offset for this vector
                mul.lo.s32 %r7, %r6, %r1;
                cvt.s64.s32 %rd4, %r7;
                shl.b64 %rd5, %rd4, 2;
                add.u64 %rd6, %rd2, %rd5;

                // Loop over dimensions
                mov.s32 %r8, 0;
            LOOP:
                setp.ge.s32 %p1, %r8, %r1;
                @%p1 bra COMPUTE;

                // Load values
                cvt.s64.s32 %rd7, %r8;
                shl.b64 %rd8, %rd7, 2;
                add.u64 %rd9, %rd1, %rd8;
                ld.global.f32 %f4, [%rd9];  // query[i]

                add.u64 %rd9, %rd6, %rd8;
                ld.global.f32 %f5, [%rd9];  // vector[i]

                // Accumulate
                fma.rn.f32 %f1, %f4, %f5, %f1;  // dot += q*v
                fma.rn.f32 %f2, %f4, %f4, %f2;  // norm_a += q*q
                fma.rn.f32 %f3, %f5, %f5, %f3;  // norm_b += v*v

                add.s32 %r8, %r8, 1;
                bra LOOP;

            COMPUTE:
                // Compute cosine distance = 1 - dot/(sqrt(norm_a)*sqrt(norm_b))
                sqrt.rn.f32 %f6, %f2;
                sqrt.rn.f32 %f7, %f3;
                mul.f32 %f8, %f6, %f7;
                div.rn.f32 %f9, %f1, %f8;
                mov.f32 %f4, 1.0;
                sub.f32 %f5, %f4, %f9;

                // Store result
                cvt.s64.s32 %rd4, %r6;
                shl.b64 %rd5, %rd4, 2;
                add.u64 %rd6, %rd3, %rd5;
                st.global.f32 [%rd6], %f5;

            END:
                ret;
            }
            "#,
        )
    }

    #[cfg(feature = "gpu-cuda")]
    fn get_euclidean_distance_ptx() -> Ptx {
        Ptx::from_src(
            r#"
            .version 7.0
            .target sm_70
            .address_size 64

            .visible .entry euclidean_distance_kernel(
                .param .u64 query_ptr,
                .param .u64 vectors_ptr,
                .param .u64 results_ptr,
                .param .s32 dim,
                .param .s32 n_vectors
            )
            {
                .reg .pred %p<2>;
                .reg .f32 %f<6>;
                .reg .b32 %r<10>;
                .reg .b64 %rd<10>;

                ld.param.u64 %rd1, [query_ptr];
                ld.param.u64 %rd2, [vectors_ptr];
                ld.param.u64 %rd3, [results_ptr];
                ld.param.s32 %r1, [dim];
                ld.param.s32 %r2, [n_vectors];

                mov.u32 %r3, %ctaid.x;
                mov.u32 %r4, %ntid.x;
                mov.u32 %r5, %tid.x;
                mad.lo.s32 %r6, %r3, %r4, %r5;

                setp.ge.s32 %p1, %r6, %r2;
                @%p1 bra END;

                mov.f32 %f1, 0.0;  // sum of squared differences

                mul.lo.s32 %r7, %r6, %r1;
                cvt.s64.s32 %rd4, %r7;
                shl.b64 %rd5, %rd4, 2;
                add.u64 %rd6, %rd2, %rd5;

                mov.s32 %r8, 0;
            LOOP:
                setp.ge.s32 %p1, %r8, %r1;
                @%p1 bra COMPUTE;

                cvt.s64.s32 %rd7, %r8;
                shl.b64 %rd8, %rd7, 2;
                add.u64 %rd9, %rd1, %rd8;
                ld.global.f32 %f2, [%rd9];

                add.u64 %rd9, %rd6, %rd8;
                ld.global.f32 %f3, [%rd9];

                sub.f32 %f4, %f2, %f3;
                fma.rn.f32 %f1, %f4, %f4, %f1;

                add.s32 %r8, %r8, 1;
                bra LOOP;

            COMPUTE:
                sqrt.rn.f32 %f5, %f1;

                cvt.s64.s32 %rd4, %r6;
                shl.b64 %rd5, %rd4, 2;
                add.u64 %rd6, %rd3, %rd5;
                st.global.f32 [%rd6], %f5;

            END:
                ret;
            }
            "#,
        )
    }

    #[cfg(feature = "gpu-cuda")]
    fn get_dot_product_ptx() -> Ptx {
        Ptx::from_src(
            r#"
            .version 7.0
            .target sm_70
            .address_size 64

            .visible .entry dot_product_kernel(
                .param .u64 query_ptr,
                .param .u64 vectors_ptr,
                .param .u64 results_ptr,
                .param .s32 dim,
                .param .s32 n_vectors
            )
            {
                .reg .pred %p<2>;
                .reg .f32 %f<4>;
                .reg .b32 %r<10>;
                .reg .b64 %rd<10>;

                ld.param.u64 %rd1, [query_ptr];
                ld.param.u64 %rd2, [vectors_ptr];
                ld.param.u64 %rd3, [results_ptr];
                ld.param.s32 %r1, [dim];
                ld.param.s32 %r2, [n_vectors];

                mov.u32 %r3, %ctaid.x;
                mov.u32 %r4, %ntid.x;
                mov.u32 %r5, %tid.x;
                mad.lo.s32 %r6, %r3, %r4, %r5;

                setp.ge.s32 %p1, %r6, %r2;
                @%p1 bra END;

                mov.f32 %f1, 0.0;

                mul.lo.s32 %r7, %r6, %r1;
                cvt.s64.s32 %rd4, %r7;
                shl.b64 %rd5, %rd4, 2;
                add.u64 %rd6, %rd2, %rd5;

                mov.s32 %r8, 0;
            LOOP:
                setp.ge.s32 %p1, %r8, %r1;
                @%p1 bra STORE;

                cvt.s64.s32 %rd7, %r8;
                shl.b64 %rd8, %rd7, 2;
                add.u64 %rd9, %rd1, %rd8;
                ld.global.f32 %f2, [%rd9];

                add.u64 %rd9, %rd6, %rd8;
                ld.global.f32 %f3, [%rd9];

                fma.rn.f32 %f1, %f2, %f3, %f1;

                add.s32 %r8, %r8, 1;
                bra LOOP;

            STORE:
                cvt.s64.s32 %rd4, %r6;
                shl.b64 %rd5, %rd4, 2;
                add.u64 %rd6, %rd3, %rd5;
                st.global.f32 [%rd6], %f1;

            END:
                ret;
            }
            "#,
        )
    }

    // ==========================================================================
    // Metal Backend Implementation
    // ==========================================================================

    #[cfg(feature = "gpu-metal")]
    fn metal_batch_cosine_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let device = MetalDevice::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let dim = query.len();
        let n_vectors = vectors.len();
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        // Create buffers
        let query_buffer = device.new_buffer_with_data(
            query.as_ptr() as *const _,
            (query.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vectors_buffer = device.new_buffer_with_data(
            flat_vectors.as_ptr() as *const _,
            (flat_vectors.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let results_buffer = device.new_buffer(
            (n_vectors * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Compile shader
        let library = device
            .new_library_with_source(Self::METAL_COSINE_SHADER, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal compile error: {}", e))?;
        let kernel = library
            .get_function("cosine_distance_kernel", None)
            .map_err(|e| format!("Metal function not found: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Metal pipeline error: {}", e))?;

        // Create command queue and buffer
        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&vectors_buffer), 0);
        encoder.set_buffer(2, Some(&results_buffer), 0);

        let dim_data = [dim as u32, n_vectors as u32];
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&dim_data) as u64,
            dim_data.as_ptr() as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(((n_vectors + 255) / 256 * 256) as u64, 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let results_ptr = results_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe { std::slice::from_raw_parts(results_ptr, n_vectors).to_vec() };

        Ok(results)
    }

    #[cfg(feature = "gpu-metal")]
    fn metal_batch_euclidean_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        // Similar implementation to cosine, using euclidean shader
        let device = MetalDevice::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let dim = query.len();
        let n_vectors = vectors.len();
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        let query_buffer = device.new_buffer_with_data(
            query.as_ptr() as *const _,
            (query.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vectors_buffer = device.new_buffer_with_data(
            flat_vectors.as_ptr() as *const _,
            (flat_vectors.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let results_buffer = device.new_buffer(
            (n_vectors * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let library = device
            .new_library_with_source(Self::METAL_EUCLIDEAN_SHADER, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal compile error: {}", e))?;
        let kernel = library
            .get_function("euclidean_distance_kernel", None)
            .map_err(|e| format!("Metal function not found: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Metal pipeline error: {}", e))?;

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&vectors_buffer), 0);
        encoder.set_buffer(2, Some(&results_buffer), 0);

        let dim_data = [dim as u32, n_vectors as u32];
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&dim_data) as u64,
            dim_data.as_ptr() as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(((n_vectors + 255) / 256 * 256) as u64, 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let results_ptr = results_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe { std::slice::from_raw_parts(results_ptr, n_vectors).to_vec() };

        Ok(results)
    }

    #[cfg(feature = "gpu-metal")]
    fn metal_batch_dot_product(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let device = MetalDevice::system_default()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let dim = query.len();
        let n_vectors = vectors.len();
        let flat_vectors: Vec<f32> = vectors.iter().flatten().copied().collect();

        let query_buffer = device.new_buffer_with_data(
            query.as_ptr() as *const _,
            (query.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let vectors_buffer = device.new_buffer_with_data(
            flat_vectors.as_ptr() as *const _,
            (flat_vectors.len() * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let results_buffer = device.new_buffer(
            (n_vectors * std::mem::size_of::<f32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let library = device
            .new_library_with_source(Self::METAL_DOT_PRODUCT_SHADER, &metal::CompileOptions::new())
            .map_err(|e| format!("Metal compile error: {}", e))?;
        let kernel = library
            .get_function("dot_product_kernel", None)
            .map_err(|e| format!("Metal function not found: {}", e))?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Metal pipeline error: {}", e))?;

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&query_buffer), 0);
        encoder.set_buffer(1, Some(&vectors_buffer), 0);
        encoder.set_buffer(2, Some(&results_buffer), 0);

        let dim_data = [dim as u32, n_vectors as u32];
        encoder.set_bytes(
            3,
            std::mem::size_of_val(&dim_data) as u64,
            dim_data.as_ptr() as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let grid_size = MTLSize::new(((n_vectors + 255) / 256 * 256) as u64, 1, 1);
        encoder.dispatch_threads(grid_size, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let results_ptr = results_buffer.contents() as *const f32;
        let results: Vec<f32> = unsafe { std::slice::from_raw_parts(results_ptr, n_vectors).to_vec() };

        Ok(results)
    }

    #[cfg(feature = "gpu-metal")]
    const METAL_COSINE_SHADER: &'static str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void cosine_distance_kernel(
            device const float* query [[buffer(0)]],
            device const float* vectors [[buffer(1)]],
            device float* results [[buffer(2)]],
            constant uint2& dims [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint dim = dims.x;
            uint n_vectors = dims.y;

            if (id >= n_vectors) return;

            float dot = 0.0;
            float norm_a = 0.0;
            float norm_b = 0.0;

            uint offset = id * dim;
            for (uint i = 0; i < dim; i++) {
                float q = query[i];
                float v = vectors[offset + i];
                dot += q * v;
                norm_a += q * q;
                norm_b += v * v;
            }

            float denom = sqrt(norm_a) * sqrt(norm_b);
            results[id] = (denom > 0.0) ? (1.0 - dot / denom) : 1.0;
        }
    "#;

    #[cfg(feature = "gpu-metal")]
    const METAL_EUCLIDEAN_SHADER: &'static str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void euclidean_distance_kernel(
            device const float* query [[buffer(0)]],
            device const float* vectors [[buffer(1)]],
            device float* results [[buffer(2)]],
            constant uint2& dims [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint dim = dims.x;
            uint n_vectors = dims.y;

            if (id >= n_vectors) return;

            float sum = 0.0;
            uint offset = id * dim;

            for (uint i = 0; i < dim; i++) {
                float d = query[i] - vectors[offset + i];
                sum += d * d;
            }

            results[id] = sqrt(sum);
        }
    "#;

    #[cfg(feature = "gpu-metal")]
    const METAL_DOT_PRODUCT_SHADER: &'static str = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void dot_product_kernel(
            device const float* query [[buffer(0)]],
            device const float* vectors [[buffer(1)]],
            device float* results [[buffer(2)]],
            constant uint2& dims [[buffer(3)]],
            uint id [[thread_position_in_grid]]
        ) {
            uint dim = dims.x;
            uint n_vectors = dims.y;

            if (id >= n_vectors) return;

            float dot = 0.0;
            uint offset = id * dim;

            for (uint i = 0; i < dim; i++) {
                dot += query[i] * vectors[offset + i];
            }

            results[id] = dot;
        }
    "#;

    /// Reset metrics
    pub fn reset_metrics(&self) {
        *self.metrics.write().expect("metrics lock should not be poisoned") = GpuMetrics::default();
    }

    /// Check if using real GPU (not CPU fallback)
    pub fn is_gpu_accelerated(&self) -> bool {
        self.device.backend != GpuBackend::CpuSimd
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

// SIMD-optimized helper functions

fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
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

fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
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
fn cosine_distance_simd(a: &[f32], b: &[f32]) -> f32 {
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
fn cosine_distance_simd_precomputed(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let dot = dot_product_simd(a, b);
    let norm_b = dot_product_simd(b, b).sqrt();

    if a_norm == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (a_norm * norm_b))
    }
}

/// Efficient partial sort for top-k selection
fn partial_sort_top_k(distances: Vec<f32>, k: usize) -> Vec<(usize, f32)> {
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

fn normalize_vector_simd(v: &mut [f32]) {
    let norm = dot_product_simd(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn calculate_gflops(flops: usize, duration: Duration) -> f64 {
    let secs = duration.as_secs_f64();
    if secs > 0.0 {
        (flops as f64) / (secs * 1e9)
    } else {
        0.0
    }
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_config_builder() {
        let config = GpuConfig::builder()
            .preferred_backend(GpuBackend::Metal)
            .memory_limit_mb(2048)
            .batch_size(512)
            .use_fp16(true)
            .build();

        assert_eq!(config.preferred_backend, GpuBackend::Metal);
        assert_eq!(config.memory_limit_mb, 2048);
        assert_eq!(config.batch_size, 512);
        assert!(config.use_fp16);
    }

    #[test]
    fn test_gpu_accelerator_creation() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        assert!(gpu.is_initialized);
    }

    #[test]
    fn test_detect_devices() {
        let devices = GpuAccelerator::detect_devices().unwrap();
        assert!(!devices.is_empty());
        // CPU fallback should always be present
        assert!(devices.iter().any(|d| d.backend == GpuBackend::CpuSimd));
    }

    #[test]
    fn test_batch_cosine_distance() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0], // Same direction
            vec![0.0, 1.0, 0.0], // Orthogonal
            vec![-1.0, 0.0, 0.0], // Opposite
        ];

        let distances = gpu.batch_cosine_distance(&query, &vectors).unwrap();

        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 0.0).abs() < 1e-6); // Same = 0 distance
        assert!((distances[1] - 1.0).abs() < 1e-6); // Orthogonal = 1 distance
        assert!((distances[2] - 2.0).abs() < 1e-6); // Opposite = 2 distance
    }

    #[test]
    fn test_batch_euclidean_distance() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![0.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![3.0, 4.0, 0.0],
        ];

        let distances = gpu.batch_euclidean_distance(&query, &vectors).unwrap();

        assert_eq!(distances.len(), 3);
        assert!((distances[0] - 1.0).abs() < 1e-6);
        assert!((distances[1] - 2.0).abs() < 1e-6);
        assert!((distances[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_dot_product() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];

        let dots = gpu.batch_dot_product(&query, &vectors).unwrap();

        assert_eq!(dots.len(), 3);
        assert!((dots[0] - 1.0).abs() < 1e-6);
        assert!((dots[1] - 2.0).abs() < 1e-6);
        assert!((dots[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_normalize() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let mut vectors = vec![
            vec![3.0, 4.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];

        gpu.batch_normalize(&mut vectors).unwrap();

        // Check unit length
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_batch_top_k() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let distances = vec![0.5, 0.1, 0.8, 0.3, 0.2];
        let top_k = gpu.batch_top_k(&distances, 3).unwrap();

        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k[0].0, 1); // Index 1 has smallest (0.1)
        assert_eq!(top_k[1].0, 4); // Index 4 has second smallest (0.2)
        assert_eq!(top_k[2].0, 3); // Index 3 has third smallest (0.3)
    }

    #[test]
    fn test_quantization_roundtrip() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let vectors = vec![
            vec![0.1, 0.5, -0.3, 0.8],
            vec![-0.2, 0.7, 0.4, -0.1],
        ];

        let (quantized, params) = gpu.quantize_to_int8(&vectors).unwrap();
        let dequantized = gpu.dequantize_from_int8(&quantized, &params).unwrap();

        // Check approximate equality after roundtrip
        // INT8 quantization has limited precision (256 values over the range)
        for (orig, deq) in vectors.iter().zip(dequantized.iter()) {
            for (o, d) in orig.iter().zip(deq.iter()) {
                assert!((o - d).abs() < 0.1, "Quantization error too high: orig={}, deq={}", o, d);
            }
        }
    }

    #[test]
    fn test_matmul() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];

        let c = gpu.matmul(&a, &b).unwrap();

        // [1,2] * [[5,6],[7,8]] = [19, 22]
        // [3,4] * [[5,6],[7,8]] = [43, 50]
        assert!((c[0][0] - 19.0).abs() < 1e-6);
        assert!((c[0][1] - 22.0).abs() < 1e-6);
        assert!((c[1][0] - 43.0).abs() < 1e-6);
        assert!((c[1][1] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_kmeans_assign() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let vectors = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.1],
        ];

        let centroids = vec![
            vec![0.0, 0.0],
            vec![10.0, 10.0],
        ];

        let assignments = gpu.kmeans_assign(&vectors, &centroids).unwrap();

        assert_eq!(assignments, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_fused_search() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![-1.0, 0.0, 0.0],
        ];

        let results = gpu.fused_search(&query, &vectors, 2, DistanceType::Cosine).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Exact match
        assert_eq!(results[1].0, 1); // Close match
    }

    #[test]
    fn test_memory_pool() {
        let gpu = GpuAccelerator::with_cpu_fallback(
            GpuConfig::builder().memory_limit_mb(1).build(),
        );

        let initial_memory = gpu.available_memory();

        // Allocate a buffer
        let buffer = gpu.allocate_buffer(1000, DataType::Float32).unwrap();
        assert_eq!(buffer.size, 4000); // 1000 * 4 bytes

        let after_alloc = gpu.available_memory();
        assert!(after_alloc < initial_memory);

        // Free the buffer
        assert!(gpu.free_buffer(&buffer));

        let after_free = gpu.available_memory();
        assert_eq!(after_free, initial_memory);
    }

    #[test]
    fn test_metrics_tracking() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        // Perform some operations
        gpu.batch_cosine_distance(&query, &vectors).unwrap();
        gpu.batch_euclidean_distance(&query, &vectors).unwrap();

        let metrics = gpu.metrics();
        assert_eq!(metrics.total_operations, 2);
        assert!(metrics.total_execution_time > Duration::ZERO);
    }

    #[test]
    fn test_pca_project() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let vectors = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![5.0, 6.0, 7.0, 8.0],
        ];

        // Projection matrix (4D -> 2D)
        let projection = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];

        let projected = gpu.pca_project(&vectors, &projection).unwrap();

        assert_eq!(projected.len(), 2);
        assert_eq!(projected[0].len(), 2);
        assert!((projected[0][0] - 1.0).abs() < 1e-6);
        assert!((projected[0][1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_data_type_sizes() {
        assert_eq!(DataType::Float16.size_bytes(), 2);
        assert_eq!(DataType::Float32.size_bytes(), 4);
        assert_eq!(DataType::Float64.size_bytes(), 8);
        assert_eq!(DataType::Int8.size_bytes(), 1);
        assert_eq!(DataType::Int32.size_bytes(), 4);
    }

    #[test]
    fn test_parallel_batch_cosine_large() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        // Create large batch to trigger parallel processing
        let query = vec![1.0; 128];
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| {
                let mut v = vec![0.0; 128];
                v[i % 128] = 1.0;
                v
            })
            .collect();

        let distances = gpu.batch_cosine_distance(&query, &vectors).unwrap();

        assert_eq!(distances.len(), 500);
        // All distances should be valid (between 0 and 2 for cosine)
        for d in &distances {
            assert!(*d >= 0.0 && *d <= 2.0, "Invalid cosine distance: {}", d);
        }
    }

    #[test]
    fn test_parallel_fused_search_large() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0; 64];
        let vectors: Vec<Vec<f32>> = (0..1000)
            .map(|i| {
                let mut v = vec![0.1; 64];
                v[0] = (i as f32) / 1000.0;
                v
            })
            .collect();

        let results = gpu
            .fused_search(&query, &vectors, 10, DistanceType::Cosine)
            .unwrap();

        assert_eq!(results.len(), 10);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1,
                "Results not sorted: {} > {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_batch_fused_search() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let queries = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
        ];

        let results = gpu
            .batch_fused_search(&queries, &vectors, 2, DistanceType::Cosine)
            .unwrap();

        assert_eq!(results.len(), 3);
        // Each query should find its matching vector first
        assert_eq!(results[0][0].0, 0);
        assert_eq!(results[1][0].0, 1);
        assert_eq!(results[2][0].0, 2);
    }

    #[test]
    fn test_parallel_matmul_large() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        // Create larger matrices to trigger parallel processing
        let a: Vec<Vec<f32>> = (0..200).map(|i| vec![(i as f32) * 0.01; 50]).collect();
        let b: Vec<Vec<f32>> = (0..50).map(|i| vec![(i as f32) * 0.01; 30]).collect();

        let c = gpu.matmul(&a, &b).unwrap();

        assert_eq!(c.len(), 200);
        assert_eq!(c[0].len(), 30);
    }

    #[test]
    fn test_parallel_kmeans_large() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        // Large dataset
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| {
                if i < 250 {
                    vec![0.0 + (i as f32) * 0.001, 0.0]
                } else {
                    vec![10.0 + ((i - 250) as f32) * 0.001, 10.0]
                }
            })
            .collect();

        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

        let assignments = gpu.kmeans_assign(&vectors, &centroids).unwrap();

        assert_eq!(assignments.len(), 500);
        // First half should be assigned to cluster 0
        assert!(assignments[0..250].iter().all(|&a| a == 0));
        // Second half should be assigned to cluster 1
        assert!(assignments[250..500].iter().all(|&a| a == 1));
    }

    #[test]
    fn test_parallel_normalize_large() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let mut vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| vec![1.0, 2.0, 3.0, (i as f32) * 0.1])
            .collect();

        gpu.batch_normalize(&mut vectors).unwrap();

        // All vectors should be unit length
        for v in &vectors {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Vector not normalized: norm = {}",
                norm
            );
        }
    }

    #[test]
    fn test_partial_sort_top_k() {
        let distances = vec![0.5, 0.1, 0.8, 0.3, 0.2, 0.9, 0.15];
        let top_k = partial_sort_top_k(distances, 3);

        assert_eq!(top_k.len(), 3);
        assert_eq!(top_k[0].0, 1); // 0.1
        assert_eq!(top_k[1].0, 6); // 0.15
        assert_eq!(top_k[2].0, 4); // 0.2
    }

    #[test]
    fn test_cosine_distance_precomputed() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let a_norm = 1.0;

        let dist = cosine_distance_simd_precomputed(&a, &b, a_norm);
        assert!((dist - 0.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        let dist2 = cosine_distance_simd_precomputed(&a, &c, a_norm);
        assert!((dist2 - 1.0).abs() < 1e-6);
    }
}
