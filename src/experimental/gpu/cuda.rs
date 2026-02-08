//! CUDA backend implementation for GPU-accelerated vector operations.

#[cfg(feature = "gpu-cuda")]
use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};
#[cfg(feature = "gpu-cuda")]
use cudarc::nvrtc::Ptx;

use super::GpuAccelerator;

impl GpuAccelerator {
    #[cfg(feature = "gpu-cuda")]
    pub(super) fn cuda_batch_cosine_distance(
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

        // SAFETY: All GPU buffers (query_gpu, vectors_gpu, results_gpu) are allocated above
        // with correct sizes. Grid/block dimensions are computed to cover n_vectors elements.
        // The kernel reads dim×n_vectors floats from vectors_gpu and writes n_vectors floats
        // to results_gpu, both matching their allocation sizes.
        unsafe {
            kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &query_gpu,
                        &vectors_gpu,
                        &mut results_gpu,
                        dim as i32,
                        n_vectors as i32,
                    ),
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
    pub(super) fn cuda_batch_euclidean_distance(
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

        // SAFETY: All GPU buffers (query_gpu, vectors_gpu, results_gpu) are allocated above
        // with correct sizes. Grid/block dimensions are computed to cover n_vectors elements.
        // The kernel reads dim×n_vectors floats from vectors_gpu and writes n_vectors floats
        // to results_gpu, both matching their allocation sizes.
        unsafe {
            kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &query_gpu,
                        &vectors_gpu,
                        &mut results_gpu,
                        dim as i32,
                        n_vectors as i32,
                    ),
                )
                .map_err(|e| format!("CUDA launch error: {}", e))?;
        }

        let results = device
            .dtoh_sync_copy(&results_gpu)
            .map_err(|e| format!("CUDA copy back error: {}", e))?;

        Ok(results)
    }

    #[cfg(feature = "gpu-cuda")]
    pub(super) fn cuda_batch_dot_product(
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

        // SAFETY: All GPU buffers (query_gpu, vectors_gpu, results_gpu) are allocated above
        // with correct sizes. Grid/block dimensions are computed to cover n_vectors elements.
        // The kernel reads dim×n_vectors floats from vectors_gpu and writes n_vectors floats
        // to results_gpu, both matching their allocation sizes.
        unsafe {
            kernel
                .clone()
                .launch(
                    LaunchConfig {
                        grid_dim: (grid_size as u32, 1, 1),
                        block_dim: (block_size as u32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &query_gpu,
                        &vectors_gpu,
                        &mut results_gpu,
                        dim as i32,
                        n_vectors as i32,
                    ),
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
}
