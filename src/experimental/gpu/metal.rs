//! Metal backend implementation for GPU-accelerated vector operations.

#[cfg(feature = "gpu-metal")]
use metal::{Device as MetalDevice, MTLResourceOptions, MTLSize};

use super::GpuAccelerator;

impl GpuAccelerator {
    #[cfg(feature = "gpu-metal")]
    pub(super) fn metal_batch_cosine_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let device =
            MetalDevice::system_default().ok_or_else(|| "No Metal device found".to_string())?;

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
        // SAFETY: results_buffer was allocated with n_vectors × sizeof(f32) bytes above.
        // The Metal command buffer has completed (wait_until_completed), so the data is
        // fully written. The pointer is valid for the lifetime of results_buffer.
        let results: Vec<f32> =
            unsafe { std::slice::from_raw_parts(results_ptr, n_vectors).to_vec() };

        Ok(results)
    }

    #[cfg(feature = "gpu-metal")]
    pub(super) fn metal_batch_euclidean_distance(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        // Similar implementation to cosine, using euclidean shader
        let device =
            MetalDevice::system_default().ok_or_else(|| "No Metal device found".to_string())?;

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
        // SAFETY: results_buffer was allocated with n_vectors × sizeof(f32) bytes above.
        // The Metal command buffer has completed (wait_until_completed), so the data is
        // fully written. The pointer is valid for the lifetime of results_buffer.
        let results: Vec<f32> =
            unsafe { std::slice::from_raw_parts(results_ptr, n_vectors).to_vec() };

        Ok(results)
    }

    #[cfg(feature = "gpu-metal")]
    pub(super) fn metal_batch_dot_product(
        &self,
        query: &[f32],
        vectors: &[Vec<f32>],
    ) -> Result<Vec<f32>, String> {
        let device =
            MetalDevice::system_default().ok_or_else(|| "No Metal device found".to_string())?;

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
            .new_library_with_source(
                Self::METAL_DOT_PRODUCT_SHADER,
                &metal::CompileOptions::new(),
            )
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
        // SAFETY: results_buffer was allocated with n_vectors × sizeof(f32) bytes above.
        // The Metal command buffer has completed (wait_until_completed), so the data is
        // fully written. The pointer is valid for the lifetime of results_buffer.
        let results: Vec<f32> =
            unsafe { std::slice::from_raw_parts(results_ptr, n_vectors).to_vec() };

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
}
