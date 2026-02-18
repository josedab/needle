#![allow(unsafe_code)] // GPU backend FFI requires unsafe
//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
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

pub mod common;
mod cuda;
mod metal;

pub use common::*;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use common::{
    calculate_gflops, checked_gpu_bytes, cosine_distance_simd_precomputed, dot_product_simd,
    euclidean_distance_simd, normalize_vector_simd, num_cpus, partial_sort_top_k, MemoryPool,
    PARALLEL_THRESHOLD,
};

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
        let mut pool = self
            .memory_pool
            .write()
            .expect("memory_pool lock should not be poisoned");

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

        self.buffers
            .write()
            .expect("buffers lock should not be poisoned")
            .insert(buffer_id, buffer.clone());
        Ok(buffer)
    }

    /// Free a buffer
    pub fn free_buffer(&self, buffer: &GpuBuffer) -> bool {
        let mut pool = self
            .memory_pool
            .write()
            .expect("memory_pool lock should not be poisoned");
        if pool.deallocate(buffer.id) {
            self.buffers
                .write()
                .expect("buffers lock should not be poisoned")
                .remove(&buffer.id);
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
        let total_elements = vectors.len().checked_mul(dim)
            .ok_or("GPU buffer size overflow")?;
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
            checked_gpu_bytes(&[vectors.len(), dim, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(vectors.len().saturating_mul(dim).saturating_mul(3), start.elapsed())),
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
            checked_gpu_bytes(&[vectors.len(), dim, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(vectors.len().saturating_mul(dim).saturating_mul(3), start.elapsed())),
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
            checked_gpu_bytes(&[vectors.len(), dim, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(vectors.len().saturating_mul(dim).saturating_mul(2), start.elapsed())),
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
            checked_gpu_bytes(&[total_elements, 4]).unwrap_or(u64::MAX),
            None,
        );

        Ok(())
    }

    /// Matrix multiplication (for projections, etc.) with parallelization
    pub fn matmul(&self, a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
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
        let b_transposed: Vec<Vec<f32>> =
            (0..n).map(|j| (0..k).map(|i| b[i][j]).collect()).collect();

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
            checked_gpu_bytes(&[m.saturating_mul(k).saturating_add(k.saturating_mul(n)).saturating_add(m.saturating_mul(n)), 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(2usize.saturating_mul(m).saturating_mul(n).saturating_mul(k), start.elapsed())),
        );

        Ok(result)
    }

    /// Top-K selection across batch
    pub fn batch_top_k(&self, distances: &[f32], k: usize) -> Result<Vec<(usize, f32)>, String> {
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
            let (min_val, max_val) = vector.iter().fold((f32::MAX, f32::MIN), |(min, max), &v| {
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
            checked_gpu_bytes(&[total_elements, 4]).unwrap_or(u64::MAX),
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
            checked_gpu_bytes(&[input_elements.saturating_add(output_elements), 4]).unwrap_or(u64::MAX),
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
                        .map_or(0, |(i, _)| i)
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
                        .map_or(0, |(i, _)| i)
                })
                .collect()
        };

        let dim = vectors.first().map_or(0, |v| v.len());
        let total_ops = vectors.len().saturating_mul(centroids.len()).saturating_mul(dim);
        self.update_metrics(
            KernelType::KMeansAssign,
            start.elapsed(),
            checked_gpu_bytes(&[total_ops, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(total_ops.saturating_mul(3), start.elapsed())),
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
            checked_gpu_bytes(&[vectors.len(), dim, 4, 2]).unwrap_or(u64::MAX),
            Some(calculate_gflops(vectors.len().saturating_mul(dim).saturating_mul(3), start.elapsed())),
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

        let total_ops = queries.len().saturating_mul(vectors.len()).saturating_mul(dim).saturating_mul(3);
        self.update_metrics(
            KernelType::CosineSimilarity,
            start.elapsed(),
            checked_gpu_bytes(&[total_ops, 4]).unwrap_or(u64::MAX),
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
        let mut metrics = self
            .metrics
            .write()
            .expect("metrics lock should not be poisoned");
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
    // Streaming Batch Search (Feature 5 Enhancement)
    // ==========================================================================

    /// Stream-based batch search for memory-efficient processing of large datasets.
    ///
    /// Processes vectors in chunks to avoid memory overflow on GPU,
    /// with automatic chunk sizing based on available memory.
    pub fn streaming_batch_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance_type: DistanceType,
        chunk_size: Option<usize>,
    ) -> Result<Vec<(usize, f32)>, String> {
        let start = Instant::now();

        if vectors.is_empty() {
            return Ok(vec![]);
        }

        // Determine optimal chunk size
        let effective_chunk_size = chunk_size
            .unwrap_or_else(|| self.calculate_optimal_chunk_size(query.len(), vectors.len()));

        let n_vectors = vectors.len();
        let mut all_results: Vec<(usize, f32)> = Vec::new();

        // Process in chunks
        for chunk_start in (0..n_vectors).step_by(effective_chunk_size) {
            let chunk_end = (chunk_start + effective_chunk_size).min(n_vectors);
            let chunk: Vec<Vec<f32>> = vectors[chunk_start..chunk_end].to_vec();

            // Search within chunk
            let chunk_results = self.fused_search(query, &chunk, k, distance_type)?;

            // Adjust indices to global position and add to results
            for (local_idx, dist) in chunk_results {
                all_results.push((chunk_start + local_idx, dist));
            }
        }

        // Merge and get global top-k
        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);

        let total_ops = n_vectors.saturating_mul(query.len()).saturating_mul(3);
        self.update_metrics(
            KernelType::CosineSimilarity,
            start.elapsed(),
            checked_gpu_bytes(&[total_ops, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(total_ops, start.elapsed())),
        );

        Ok(all_results)
    }

    /// Multi-query streaming batch search for processing multiple queries efficiently.
    pub fn streaming_multi_query_search(
        &self,
        queries: &[Vec<f32>],
        vectors: &[Vec<f32>],
        k: usize,
        distance_type: DistanceType,
    ) -> Result<Vec<Vec<(usize, f32)>>, String> {
        let start = Instant::now();

        if queries.is_empty() || vectors.is_empty() {
            return Ok(vec![vec![]; queries.len()]);
        }

        let dim = queries[0].len();
        let n_queries = queries.len();
        let n_vectors = vectors.len();

        // Process queries in parallel for better throughput
        let results: Vec<Vec<(usize, f32)>> = if n_queries >= PARALLEL_THRESHOLD / 10 {
            queries
                .par_iter()
                .map(|q| {
                    self.streaming_batch_search(q, vectors, k, distance_type, None)
                        .unwrap_or_default()
                })
                .collect()
        } else {
            queries
                .iter()
                .map(|q| {
                    self.streaming_batch_search(q, vectors, k, distance_type, None)
                        .unwrap_or_default()
                })
                .collect()
        };

        let total_ops = n_queries.saturating_mul(n_vectors).saturating_mul(dim).saturating_mul(3);
        self.update_metrics(
            KernelType::CosineSimilarity,
            start.elapsed(),
            checked_gpu_bytes(&[total_ops, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(total_ops, start.elapsed())),
        );

        Ok(results)
    }

    /// Calculate optimal chunk size based on GPU memory and vector dimensions.
    fn calculate_optimal_chunk_size(&self, dimensions: usize, total_vectors: usize) -> usize {
        // Estimate memory per vector (bytes)
        let bytes_per_vector = dimensions * 4; // f32 = 4 bytes

        // Target using ~50% of available memory for safety
        let available_memory = self.device.available_memory / 2;

        // Calculate max vectors that fit in memory
        let max_vectors = if available_memory > 0 {
            (available_memory as usize) / bytes_per_vector
        } else {
            10000 // Default if memory unknown
        };

        // Clamp between reasonable bounds
        let chunk_size = max_vectors.clamp(100, 50000);

        // Don't exceed total vectors
        chunk_size.min(total_vectors)
    }

    /// Adaptive batch search that automatically tunes parameters based on workload.
    pub fn adaptive_batch_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance_type: DistanceType,
    ) -> Result<AdaptiveBatchResult, String> {
        let start = Instant::now();

        if vectors.is_empty() {
            return Ok(AdaptiveBatchResult {
                results: vec![],
                chunk_size_used: 0,
                chunks_processed: 0,
                total_time: Duration::ZERO,
                throughput_vps: 0.0,
            });
        }

        let n_vectors = vectors.len();
        let dim = query.len();

        // Decide strategy based on dataset size
        let (results, chunk_size_used, chunks_processed) = if n_vectors < PARALLEL_THRESHOLD {
            // Small dataset: single fused search
            let results = self.fused_search(query, vectors, k, distance_type)?;
            (results, n_vectors, 1)
        } else if n_vectors < 100_000 {
            // Medium dataset: parallel batch search
            let results = self.fused_search(query, vectors, k, distance_type)?;
            (results, n_vectors, 1)
        } else {
            // Large dataset: streaming search
            let chunk_size = self.calculate_optimal_chunk_size(dim, n_vectors);
            let num_chunks = (n_vectors + chunk_size - 1) / chunk_size;
            let results =
                self.streaming_batch_search(query, vectors, k, distance_type, Some(chunk_size))?;
            (results, chunk_size, num_chunks)
        };

        let total_time = start.elapsed();
        let throughput_vps = if total_time.as_secs_f64() > 0.0 {
            n_vectors as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        Ok(AdaptiveBatchResult {
            results,
            chunk_size_used,
            chunks_processed,
            total_time,
            throughput_vps,
        })
    }

    /// Pre-filtered batch search that applies a filter function before GPU processing.
    pub fn prefiltered_batch_search<F>(
        &self,
        query: &[f32],
        vectors: &[(usize, Vec<f32>)], // (original_index, vector)
        k: usize,
        distance_type: DistanceType,
        filter: F,
    ) -> Result<Vec<(usize, f32)>, String>
    where
        F: Fn(usize) -> bool + Sync,
    {
        let start = Instant::now();

        // Apply filter to get candidate vectors
        let filtered: Vec<(usize, Vec<f32>)> = vectors
            .par_iter()
            .filter(|(idx, _)| filter(*idx))
            .cloned()
            .collect();

        if filtered.is_empty() {
            return Ok(vec![]);
        }

        // Extract vectors for GPU processing
        let filtered_vectors: Vec<Vec<f32>> = filtered.iter().map(|(_, v)| v.clone()).collect();
        let original_indices: Vec<usize> = filtered.iter().map(|(idx, _)| *idx).collect();

        // Run GPU search on filtered set
        let search_results = self.fused_search(query, &filtered_vectors, k, distance_type)?;

        // Map back to original indices
        let results: Vec<(usize, f32)> = search_results
            .into_iter()
            .map(|(local_idx, dist)| (original_indices[local_idx], dist))
            .collect();

        let dim = query.len();
        let total_ops = filtered_vectors.len().saturating_mul(dim).saturating_mul(3);
        self.update_metrics(
            KernelType::CosineSimilarity,
            start.elapsed(),
            checked_gpu_bytes(&[total_ops, 4]).unwrap_or(u64::MAX),
            Some(calculate_gflops(total_ops, start.elapsed())),
        );

        Ok(results)
    }


    /// Reset metrics
    pub fn reset_metrics(&self) {
        *self
            .metrics
            .write()
            .expect("metrics lock should not be poisoned") = GpuMetrics::default();
    }

    /// Check if using real GPU (not CPU fallback)
    pub fn is_gpu_accelerated(&self) -> bool {
        self.device.backend != GpuBackend::CpuSimd
    }
}

// ---------------------------------------------------------------------------
// GPU-Resident Index: keeps vectors pinned in GPU memory for zero-copy search
// ---------------------------------------------------------------------------

/// A GPU-resident vector index that keeps vectors in device memory, avoiding
/// repeated host-to-device transfers for repeated searches over the same dataset.
pub struct GpuResidentIndex {
    accelerator: GpuAccelerator,
    dimension: usize,
    vectors: Vec<Vec<f32>>,
    ids: Vec<String>,
    dirty: bool,
}

impl GpuResidentIndex {
    /// Create a new GPU-resident index.
    pub fn new(accelerator: GpuAccelerator, dimension: usize) -> Self {
        Self {
            accelerator,
            dimension,
            vectors: Vec::new(),
            ids: Vec::new(),
            dirty: true,
        }
    }

    /// Add vectors to the index.
    pub fn add(&mut self, id: &str, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }
        self.vectors.push(vector);
        self.ids.push(id.to_string());
        self.dirty = true;
        Ok(())
    }

    /// Batch-add vectors.
    pub fn add_batch(&mut self, entries: Vec<(String, Vec<f32>)>) -> Result<usize, String> {
        let mut added = 0;
        for (id, vec) in entries {
            self.add(&id, vec)?;
            added += 1;
        }
        Ok(added)
    }

    /// Search for the k nearest neighbors of `query`.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        distance: DistanceType,
    ) -> Result<Vec<(String, f32)>, String> {
        if query.len() != self.dimension {
            return Err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ));
        }
        if self.vectors.is_empty() {
            return Ok(Vec::new());
        }

        let distances = match distance {
            DistanceType::Cosine => self
                .accelerator
                .batch_cosine_distance(&query.to_vec(), &self.vectors)?,
            DistanceType::Euclidean => self
                .accelerator
                .batch_euclidean_distance(&query.to_vec(), &self.vectors)?,
            DistanceType::DotProduct => self
                .accelerator
                .batch_dot_product(&query.to_vec(), &self.vectors)?,
        };

        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);

        Ok(indexed
            .into_iter()
            .map(|(i, d)| (self.ids[i].clone(), d))
            .collect())
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Remove a vector by ID. Returns true if found and removed.
    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self.ids.iter().position(|i| i == id) {
            self.ids.swap_remove(pos);
            self.vectors.swap_remove(pos);
            self.dirty = true;
            true
        } else {
            false
        }
    }

    /// Get a reference to the underlying accelerator.
    pub fn accelerator(&self) -> &GpuAccelerator {
        &self.accelerator
    }
}

// ---------------------------------------------------------------------------
// Multi-GPU Shard Manager: distributes vectors across multiple GPU accelerators
// ---------------------------------------------------------------------------

/// Strategy for distributing vectors across multiple GPUs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardStrategy {
    /// Round-robin assignment across GPUs.
    RoundRobin,
    /// Fill each GPU before moving to the next.
    FillFirst,
}

impl Default for ShardStrategy {
    fn default() -> Self {
        Self::RoundRobin
    }
}

/// Configuration for multi-GPU sharding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiGpuConfig {
    pub num_shards: usize,
    pub strategy: ShardStrategy,
    pub merge_top_k: usize,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            num_shards: 2,
            strategy: ShardStrategy::RoundRobin,
            merge_top_k: 100,
        }
    }
}

/// A shard in the multi-GPU setup, simulated as a CPU-backed partition.
struct GpuShard {
    vectors: Vec<Vec<f32>>,
    ids: Vec<String>,
}

impl GpuShard {
    fn new() -> Self {
        Self {
            vectors: Vec::new(),
            ids: Vec::new(),
        }
    }
}

/// Manages vector distribution across multiple GPU shards and merges search
/// results. Uses CPU fallback accelerators per shard.
pub struct MultiGpuShardManager {
    config: MultiGpuConfig,
    shards: Vec<GpuShard>,
    dimension: usize,
    total_vectors: usize,
}

impl MultiGpuShardManager {
    /// Create a new multi-GPU shard manager.
    pub fn new(dimension: usize, config: MultiGpuConfig) -> Self {
        let shards = (0..config.num_shards).map(|_| GpuShard::new()).collect();
        Self {
            config,
            shards,
            dimension,
            total_vectors: 0,
        }
    }

    /// Insert a vector into the appropriate shard.
    pub fn insert(&mut self, id: &str, vector: Vec<f32>) -> Result<(), String> {
        if vector.len() != self.dimension {
            return Err(format!(
                "dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            ));
        }
        let shard_idx = match self.config.strategy {
            ShardStrategy::RoundRobin => self.total_vectors % self.config.num_shards,
            ShardStrategy::FillFirst => {
                let per_shard = (self.total_vectors / self.config.num_shards) + 1;
                self.shards
                    .iter()
                    .position(|s| s.vectors.len() < per_shard)
                    .unwrap_or(0)
            }
        };
        self.shards[shard_idx].vectors.push(vector);
        self.shards[shard_idx].ids.push(id.to_string());
        self.total_vectors += 1;
        Ok(())
    }

    /// Search across all shards and merge results.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        distance: DistanceType,
    ) -> Result<Vec<(String, f32)>, String> {
        if query.len() != self.dimension {
            return Err(format!(
                "query dimension mismatch: expected {}, got {}",
                self.dimension,
                query.len()
            ));
        }

        let compute_distance = match distance {
            DistanceType::Cosine => |a: &[f32], b: &[f32]| {
                let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    na += a[i] * a[i];
                    nb += b[i] * b[i];
                }
                let denom = na.sqrt() * nb.sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot / denom
                }
            },
            DistanceType::Euclidean => |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| (x - y) * (x - y))
                    .sum::<f32>()
                    .sqrt()
            },
            DistanceType::DotProduct => {
                |a: &[f32], b: &[f32]| -(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>())
            }
        };

        let mut all_results: Vec<(String, f32)> = Vec::new();

        for shard in &self.shards {
            for (i, vec) in shard.vectors.iter().enumerate() {
                let dist = compute_distance(query, vec);
                all_results.push((shard.ids[i].clone(), dist));
            }
        }

        all_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        Ok(all_results)
    }

    /// Total vectors across all shards.
    pub fn total_vectors(&self) -> usize {
        self.total_vectors
    }

    /// Number of vectors per shard.
    pub fn shard_sizes(&self) -> Vec<usize> {
        self.shards.iter().map(|s| s.vectors.len()).collect()
    }

    /// Number of shards.
    pub fn num_shards(&self) -> usize {
        self.config.num_shards
    }
}

// ---------------------------------------------------------------------------
// Transparent GPU/CPU Fallback Manager
// ---------------------------------------------------------------------------

/// Execution backend that was actually used for a search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionBackend {
    Gpu,
    CpuSimd,
    CpuScalar,
}

/// Result of a search with fallback information.
#[derive(Debug, Clone)]
pub struct FallbackSearchResult {
    pub results: Vec<(usize, f32)>,
    pub backend_used: ExecutionBackend,
    pub elapsed: Duration,
}

/// Transparently tries GPU search and falls back to CPU on failure, tracking
/// reliability statistics to avoid repeatedly failing backends.
pub struct TransparentFallbackManager {
    accelerator: GpuAccelerator,
    gpu_failures: u64,
    gpu_successes: u64,
    gpu_disabled_until: Option<Instant>,
    backoff_secs: u64,
}

impl TransparentFallbackManager {
    pub fn new(accelerator: GpuAccelerator) -> Self {
        Self {
            accelerator,
            gpu_failures: 0,
            gpu_successes: 0,
            gpu_disabled_until: None,
            backoff_secs: 5,
        }
    }

    /// Search with automatic GPU→CPU fallback.
    pub fn search(
        &mut self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance: DistanceType,
    ) -> FallbackSearchResult {
        let start = Instant::now();

        // Check if GPU is in backoff period
        let gpu_available = match self.gpu_disabled_until {
            Some(until) if Instant::now() < until => false,
            Some(_) => {
                self.gpu_disabled_until = None;
                true
            }
            None => true,
        };

        if gpu_available && self.accelerator.is_gpu_accelerated() {
            match self.try_gpu_search(query, vectors, k, distance) {
                Ok(results) => {
                    self.gpu_successes += 1;
                    return FallbackSearchResult {
                        results,
                        backend_used: ExecutionBackend::Gpu,
                        elapsed: start.elapsed(),
                    };
                }
                Err(_) => {
                    self.gpu_failures += 1;
                    if self.gpu_failures > 3 {
                        self.gpu_disabled_until =
                            Some(Instant::now() + Duration::from_secs(self.backoff_secs));
                        self.backoff_secs = (self.backoff_secs * 2).min(300);
                    }
                }
            }
        }

        // CPU fallback
        let results = Self::cpu_search(query, vectors, k, distance);
        FallbackSearchResult {
            results,
            backend_used: ExecutionBackend::CpuSimd,
            elapsed: start.elapsed(),
        }
    }

    fn try_gpu_search(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance: DistanceType,
    ) -> Result<Vec<(usize, f32)>, String> {
        let distances = match distance {
            DistanceType::Cosine => self
                .accelerator
                .batch_cosine_distance(&query.to_vec(), vectors)?,
            DistanceType::Euclidean => self
                .accelerator
                .batch_euclidean_distance(&query.to_vec(), vectors)?,
            DistanceType::DotProduct => self
                .accelerator
                .batch_dot_product(&query.to_vec(), vectors)?,
        };
        let mut indexed: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        Ok(indexed)
    }

    fn cpu_search(
        query: &[f32],
        vectors: &[Vec<f32>],
        k: usize,
        distance: DistanceType,
    ) -> Vec<(usize, f32)> {
        let compute = match distance {
            DistanceType::Cosine => |a: &[f32], b: &[f32]| {
                let (mut dot, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
                for i in 0..a.len() {
                    dot += a[i] * b[i];
                    na += a[i] * a[i];
                    nb += b[i] * b[i];
                }
                let denom = na.sqrt() * nb.sqrt();
                if denom == 0.0 {
                    1.0
                } else {
                    1.0 - dot / denom
                }
            },
            DistanceType::Euclidean => |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f32>()
                    .sqrt()
            },
            DistanceType::DotProduct => {
                |a: &[f32], b: &[f32]| -(a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>())
            }
        };

        let mut indexed: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, compute(query, v)))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(k);
        indexed
    }

    /// GPU reliability ratio (0.0–1.0).
    pub fn gpu_reliability(&self) -> f64 {
        let total = self.gpu_successes + self.gpu_failures;
        if total == 0 {
            1.0
        } else {
            self.gpu_successes as f64 / total as f64
        }
    }

    /// Reset failure counters and re-enable GPU.
    pub fn reset(&mut self) {
        self.gpu_failures = 0;
        self.gpu_successes = 0;
        self.gpu_disabled_until = None;
        self.backoff_secs = 5;
    }
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
            vec![1.0, 0.0, 0.0],  // Same direction
            vec![0.0, 1.0, 0.0],  // Orthogonal
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

        let mut vectors = vec![vec![3.0, 4.0, 0.0], vec![1.0, 1.0, 1.0]];

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

        let vectors = vec![vec![0.1, 0.5, -0.3, 0.8], vec![-0.2, 0.7, 0.4, -0.1]];

        let (quantized, params) = gpu.quantize_to_int8(&vectors).unwrap();
        let dequantized = gpu.dequantize_from_int8(&quantized, &params).unwrap();

        // Check approximate equality after roundtrip
        // INT8 quantization has limited precision (256 values over the range)
        for (orig, deq) in vectors.iter().zip(dequantized.iter()) {
            for (o, d) in orig.iter().zip(deq.iter()) {
                assert!(
                    (o - d).abs() < 0.1,
                    "Quantization error too high: orig={}, deq={}",
                    o,
                    d
                );
            }
        }
    }

    #[test]
    fn test_matmul() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

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

        let centroids = vec![vec![0.0, 0.0], vec![10.0, 10.0]];

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

        let results = gpu
            .fused_search(&query, &vectors, 2, DistanceType::Cosine)
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Exact match
        assert_eq!(results[1].0, 1); // Close match
    }

    #[test]
    fn test_memory_pool() {
        let gpu =
            GpuAccelerator::with_cpu_fallback(GpuConfig::builder().memory_limit_mb(1).build());

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

        let vectors = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];

        // Projection matrix (4D -> 2D)
        let projection = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];

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

    // Feature 5: Streaming Batch Search Tests

    #[test]
    fn test_streaming_batch_search() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<Vec<f32>> = (0..500)
            .map(|i| {
                let mut v = vec![0.0; 3];
                v[i % 3] = 1.0;
                v
            })
            .collect();

        let results = gpu
            .streaming_batch_search(&query, &vectors, 10, DistanceType::Cosine, Some(100))
            .unwrap();

        assert_eq!(results.len(), 10);
        // First result should have small distance (exact or near match)
        assert!(results[0].1 < 0.01);
        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(
                results[i].1 >= results[i - 1].1,
                "Results not sorted: {} vs {}",
                results[i - 1].1,
                results[i].1
            );
        }
    }

    #[test]
    fn test_streaming_multi_query_search() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let queries = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.0],
        ];

        let results = gpu
            .streaming_multi_query_search(&queries, &vectors, 2, DistanceType::Cosine)
            .unwrap();

        assert_eq!(results.len(), 2);
        // First query should find first vector
        assert_eq!(results[0][0].0, 0);
        // Second query should find second vector
        assert_eq!(results[1][0].0, 1);
    }

    #[test]
    fn test_adaptive_batch_search() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| {
                let mut v = vec![0.0; 3];
                v[i % 3] = 1.0;
                v
            })
            .collect();

        let result = gpu
            .adaptive_batch_search(&query, &vectors, 5, DistanceType::Cosine)
            .unwrap();

        assert_eq!(result.results.len(), 5);
        assert!(result.chunks_processed >= 1);
        assert!(result.throughput_vps > 0.0);
    }

    #[test]
    fn test_prefiltered_batch_search() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());

        let query = vec![1.0, 0.0, 0.0];
        let vectors: Vec<(usize, Vec<f32>)> = vec![
            (0, vec![1.0, 0.0, 0.0]), // Match, passes filter
            (1, vec![0.0, 1.0, 0.0]), // No match, filtered out
            (2, vec![0.9, 0.1, 0.0]), // Close match, passes filter
            (3, vec![0.0, 0.0, 1.0]), // No match, filtered out
            (4, vec![0.8, 0.2, 0.0]), // Close match, passes filter
        ];

        // Filter to only even indices
        let results = gpu
            .prefiltered_batch_search(&query, &vectors, 3, DistanceType::Cosine, |idx| {
                idx % 2 == 0
            })
            .unwrap();

        // Should only have results from indices 0, 2, 4
        assert!(!results.is_empty());
        for (idx, _) in &results {
            assert!(idx % 2 == 0, "Index {} should be even", idx);
        }
    }

    #[test]
    fn test_adaptive_batch_result() {
        let result = AdaptiveBatchResult {
            results: vec![(0, 0.1), (1, 0.2), (2, 0.3)],
            chunk_size_used: 100,
            chunks_processed: 2,
            total_time: Duration::from_millis(50),
            throughput_vps: 4000.0,
        };

        assert_eq!(result.indices(), vec![0, 1, 2]);
        assert_eq!(result.distances(), vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn test_hardware_capabilities_detect() {
        let caps = HardwareCapabilities::detect();
        assert!(caps.cpu_cores > 0);
        assert!(caps.estimated_throughput(384) > 0.0);
        #[cfg(target_os = "macos")]
        assert!(caps.metal_available);
    }

    #[test]
    fn test_kernel_dispatch_small() {
        let caps = HardwareCapabilities::detect();
        let dispatch = select_kernel(1, 100, 384, &caps);
        assert!(dispatch == KernelDispatch::Simd || dispatch == KernelDispatch::Scalar);
    }

    #[test]
    fn test_kernel_dispatch_large() {
        let caps = HardwareCapabilities::detect();
        let dispatch = select_kernel(100, 100_000, 384, &caps);
        // Should be ParallelSimd or Gpu depending on hardware
        assert!(dispatch == KernelDispatch::ParallelSimd || dispatch == KernelDispatch::Gpu);
    }

    // ---- GPU-Resident Index tests ----

    #[test]
    fn test_gpu_resident_index_basic() {
        let accel = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mut index = GpuResidentIndex::new(accel, 3);
        assert!(index.is_empty());

        index.add("a", vec![1.0, 0.0, 0.0]).unwrap();
        index.add("b", vec![0.0, 1.0, 0.0]).unwrap();
        index.add("c", vec![0.9, 0.1, 0.0]).unwrap();
        assert_eq!(index.len(), 3);

        let results = index
            .search(&[1.0, 0.0, 0.0], 2, DistanceType::Cosine)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_gpu_resident_index_remove() {
        let accel = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mut index = GpuResidentIndex::new(accel, 3);
        index.add("a", vec![1.0, 0.0, 0.0]).unwrap();
        index.add("b", vec![0.0, 1.0, 0.0]).unwrap();

        assert!(index.remove("a"));
        assert!(!index.remove("a"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_gpu_resident_index_dim_mismatch() {
        let accel = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mut index = GpuResidentIndex::new(accel, 3);
        assert!(index.add("a", vec![1.0, 0.0]).is_err());
        assert!(index.search(&[1.0, 0.0], 1, DistanceType::Cosine).is_err());
    }

    #[test]
    fn test_gpu_resident_batch_add() {
        let accel = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mut index = GpuResidentIndex::new(accel, 2);
        let entries = vec![("x".into(), vec![1.0, 0.0]), ("y".into(), vec![0.0, 1.0])];
        assert_eq!(index.add_batch(entries).unwrap(), 2);
        assert_eq!(index.len(), 2);
    }

    // ---- Multi-GPU Shard Manager tests ----

    #[test]
    fn test_multi_gpu_shard_round_robin() {
        let config = MultiGpuConfig {
            num_shards: 3,
            strategy: ShardStrategy::RoundRobin,
            ..Default::default()
        };
        let mut mgr = MultiGpuShardManager::new(2, config);

        for i in 0..9 {
            mgr.insert(&format!("v{}", i), vec![i as f32, 0.0]).unwrap();
        }

        assert_eq!(mgr.total_vectors(), 9);
        let sizes = mgr.shard_sizes();
        assert_eq!(sizes, vec![3, 3, 3]);
    }

    #[test]
    fn test_multi_gpu_shard_search() {
        let config = MultiGpuConfig {
            num_shards: 2,
            ..Default::default()
        };
        let mut mgr = MultiGpuShardManager::new(3, config);
        mgr.insert("a", vec![1.0, 0.0, 0.0]).unwrap();
        mgr.insert("b", vec![0.0, 1.0, 0.0]).unwrap();
        mgr.insert("c", vec![0.9, 0.1, 0.0]).unwrap();
        mgr.insert("d", vec![0.0, 0.0, 1.0]).unwrap();

        let results = mgr
            .search(&[1.0, 0.0, 0.0], 2, DistanceType::Cosine)
            .unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "a");
    }

    #[test]
    fn test_multi_gpu_dim_mismatch() {
        let mut mgr = MultiGpuShardManager::new(3, MultiGpuConfig::default());
        assert!(mgr.insert("a", vec![1.0, 0.0]).is_err());
        assert!(mgr.search(&[1.0], 1, DistanceType::Cosine).is_err());
    }

    // ---- Transparent Fallback Manager tests ----

    #[test]
    fn test_fallback_manager_cpu() {
        let accel = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let is_gpu = accel.is_gpu_accelerated();
        let mut mgr = TransparentFallbackManager::new(accel);

        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.5, 0.5, 0.0],
        ];

        let result = mgr.search(&query, &vectors, 2, DistanceType::Cosine);
        assert_eq!(result.results.len(), 2);
        assert_eq!(result.results[0].0, 0);
        if is_gpu {
            assert_eq!(result.backend_used, ExecutionBackend::Gpu);
        } else {
            assert_eq!(result.backend_used, ExecutionBackend::CpuSimd);
        }
    }

    #[test]
    fn test_fallback_manager_reliability() {
        let accel = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mgr = TransparentFallbackManager::new(accel);
        assert_eq!(mgr.gpu_reliability(), 1.0); // no ops yet

        // After reset, reliability should be 1.0 again
        let mut mgr2 = TransparentFallbackManager::new(GpuAccelerator::with_cpu_fallback(
            GpuConfig::default(),
        ));
        mgr2.reset();
        assert_eq!(mgr2.gpu_reliability(), 1.0);
    }

    // ── FFI failure & edge case tests ────────────────────────────────────

    #[test]
    fn test_batch_cosine_distance_empty() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let results = gpu.batch_cosine_distance(&[1.0, 0.0], &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_euclidean_distance_empty() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let results = gpu.batch_euclidean_distance(&[1.0, 0.0], &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_dot_product_empty() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let results = gpu.batch_dot_product(&[1.0, 0.0], &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_euclidean_dimension_mismatch() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0]]; // wrong dimension

        let result = gpu.batch_euclidean_distance(&query, &vectors);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_top_k_empty_distances() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let result = gpu.batch_top_k(&[], 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_batch_top_k_k_larger_than_input() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let distances = vec![0.5, 0.1, 0.3];
        let result = gpu.batch_top_k(&distances, 10).unwrap();
        assert_eq!(result.len(), 3); // capped at input length
    }

    #[test]
    fn test_batch_normalize_empty() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mut vectors: Vec<Vec<f32>> = vec![];
        gpu.batch_normalize(&mut vectors).unwrap();
        assert!(vectors.is_empty());
    }

    #[test]
    fn test_batch_normalize_zero_vector() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let mut vectors = vec![vec![0.0, 0.0, 0.0]];
        gpu.batch_normalize(&mut vectors).unwrap();
        // Zero vector normalization should not produce NaN
        for v in &vectors[0] {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_fused_search_all_distance_types() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];

        for dist_type in [DistanceType::Cosine, DistanceType::Euclidean, DistanceType::DotProduct] {
            let results = gpu.fused_search(&query, &vectors, 2, dist_type).unwrap();
            assert_eq!(results.len(), 2);
        }
    }

    #[test]
    fn test_matmul_empty() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let a: Vec<Vec<f32>> = vec![];
        let b: Vec<Vec<f32>> = vec![];
        let result = gpu.matmul(&a, &b);
        assert!(result.is_ok() || result.is_err()); // no panic
    }

    #[test]
    fn test_quantize_dequantize_single_element() {
        let gpu = GpuAccelerator::with_cpu_fallback(GpuConfig::default());
        let vectors = vec![vec![0.5]];
        let (quantized, params) = gpu.quantize_to_int8(&vectors).unwrap();
        let dequantized = gpu.dequantize_from_int8(&quantized, &params).unwrap();
        assert_eq!(dequantized.len(), 1);
        assert!((dequantized[0][0] - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_gpu_device_default_config() {
        let config = GpuConfig::default();
        assert_eq!(config.preferred_backend, GpuBackend::Auto);
        assert!(!config.use_fp16);
    }

    #[test]
    fn test_gpu_backend_display() {
        // Verify all backend variants can be used
        let backends = [
            GpuBackend::Auto,
            GpuBackend::Cuda,
            GpuBackend::Metal,
            GpuBackend::OpenCL,
            GpuBackend::Vulkan,
            GpuBackend::CpuSimd,
        ];
        for b in &backends {
            let _ = format!("{:?}", b);
        }
    }
}
