//! Distance Functions for Vector Similarity
//!
//! This module provides various distance and similarity metrics for comparing vectors,
//! with SIMD-optimized implementations for high performance.
//!
//! # Supported Distance Functions
//!
//! - **Cosine**: Measures the angle between vectors (1 - cosine similarity).
//!   Best for text embeddings and normalized vectors.
//! - **Euclidean (L2)**: Standard geometric distance. Good for image embeddings.
//! - **Dot Product**: Inner product (negated). Best for maximum inner product search.
//! - **Manhattan (L1)**: Sum of absolute differences. Robust to outliers.
//!
//! # SIMD Optimization
//!
//! All distance functions have SIMD-optimized implementations:
//! - **x86_64**: AVX2 instructions for 8-wide float operations
//! - **aarch64**: NEON instructions for 4-wide float operations
//!
//! The appropriate implementation is selected automatically at runtime.
//!
//! # Example
//!
//! ```
//! use needle::{DistanceFunction, distance::cosine_distance};
//!
//! let a = vec![1.0, 0.0, 0.0];
//! let b = vec![0.0, 1.0, 0.0];
//!
//! // Orthogonal vectors have cosine distance of 1.0
//! let dist = cosine_distance(&a, &b);
//! assert!((dist - 1.0).abs() < 1e-6);
//! ```

use serde::{Deserialize, Serialize};

/// Distance function types for vector similarity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// Cosine distance (1 - cosine similarity)
    #[default]
    Cosine,
    /// Cosine distance for pre-normalized vectors (faster, skips normalization)
    ///
    /// Use this when you know your vectors are already unit-normalized.
    /// This is ~3x faster than regular cosine distance as it only requires
    /// a single dot product instead of computing both vector norms.
    ///
    /// # Warning
    /// Using this with non-normalized vectors will produce incorrect results.
    CosineNormalized,
    /// Euclidean (L2) distance
    Euclidean,
    /// Dot product (negative, so smaller = more similar)
    DotProduct,
    /// Manhattan (L1) distance
    Manhattan,
}

impl DistanceFunction {
    /// Compute distance between two vectors
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            Self::Cosine => cosine_distance(a, b),
            Self::CosineNormalized => cosine_distance_normalized(a, b),
            Self::Euclidean => euclidean_distance(a, b),
            Self::DotProduct => dot_product_distance(a, b),
            Self::Manhattan => manhattan_distance(a, b),
        }
    }
}

/// Compute cosine distance (1 - cosine similarity)
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length for cosine distance");
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0;
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute cosine distance for pre-normalized vectors (faster)
///
/// This function assumes both vectors are already unit-normalized (magnitude = 1).
/// It computes `1 - dot_product(a, b)` which is equivalent to cosine distance
/// for normalized vectors, but ~3x faster since it skips norm computation.
///
/// # Performance
/// - Regular cosine: 3 dot products (a·b, a·a, b·b) + 1 sqrt each
/// - Normalized cosine: 1 dot product only
///
/// # Warning
/// Using this with non-normalized vectors will produce incorrect results.
/// Use [`normalize`] or [`normalized`] to normalize vectors first.
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
///
/// # Example
/// ```
/// use needle::distance::{cosine_distance_normalized, normalized};
///
/// let a = normalized(&[1.0, 2.0, 3.0]);
/// let b = normalized(&[4.0, 5.0, 6.0]);
/// let dist = cosine_distance_normalized(&a, &b);
/// assert!(dist >= 0.0 && dist <= 2.0);
/// ```
#[inline]
pub fn cosine_distance_normalized(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length for cosine distance");
    1.0 - dot_product(a, b)
}

/// Compute Euclidean (L2) distance
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared(a, b).sqrt()
}

/// Compute squared Euclidean distance (faster, for comparisons)
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length for euclidean distance");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::euclidean_squared_avx2(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { simd_arm::euclidean_squared_neon(a, b) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        euclidean_squared_scalar(a, b)
    }
}

/// Scalar fallback for squared Euclidean distance
#[inline]
#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
fn euclidean_squared_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute dot product distance (negative dot product)
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}

/// Compute dot product of two vectors
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length for dot product");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::dot_product_avx2(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { simd_arm::dot_product_neon(a, b) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        dot_product_scalar(a, b)
    }
}

/// Scalar fallback for dot product
#[inline]
#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute Manhattan (L1) distance
///
/// # Panics
/// Panics if `a` and `b` have different lengths.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must have equal length for manhattan distance");
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { simd_x86::manhattan_avx2(a, b) };
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { simd_arm::manhattan_neon(a, b) }
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        manhattan_scalar(a, b)
    }
}

/// Scalar fallback for Manhattan distance
#[inline]
#[cfg_attr(all(target_arch = "aarch64", target_feature = "neon"), allow(dead_code))]
fn manhattan_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

/// Normalize a vector in-place
pub fn normalize(vector: &mut [f32]) {
    // Use SIMD dot product for computing squared norm
    let norm_squared = dot_product(vector, vector);
    let norm = norm_squared.sqrt();
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        normalize_scale(vector, inv_norm);
    }
}

/// Scale a vector by a constant (SIMD-optimized)
#[inline]
fn normalize_scale(vector: &mut [f32], scale: f32) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { simd_x86::scale_avx2(vector, scale) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe { simd_arm::scale_neon(vector, scale) };
    }

    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    {
        for x in vector.iter_mut() {
            *x *= scale;
        }
    }
}

/// Normalize a vector, returning a new vector
pub fn normalized(vector: &[f32]) -> Vec<f32> {
    let mut result = vector.to_vec();
    normalize(&mut result);
    result
}

// x86_64 SIMD implementations
#[cfg(target_arch = "x86_64")]
mod simd_x86 {
    #[cfg(target_feature = "avx2")]
    use std::arch::x86_64::*;

    /// Horizontal sum of all 8 floats in a __m256 register.
    #[inline(always)]
    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    unsafe fn hsum_avx(v: __m256) -> f32 {
        let sum128 = _mm_add_ps(_mm256_extractf128_ps(v, 0), _mm256_extractf128_ps(v, 1));
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut result: f32 = 0.0;
        _mm_store_ss(&mut result, sum32);
        result
    }

    /// Compute dot product using AVX2 SIMD instructions.
    ///
    /// # Safety
    /// - Caller must ensure `a` and `b` have the same length.
    /// - This function uses unaligned loads, so no alignment requirements.
    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            sum = _mm256_fmadd_ps(va, vb, sum);
        }

        let mut result = hsum_avx(sum);

        // Handle remainder
        for i in (chunks * 8)..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    /// Compute squared Euclidean distance using AVX2 SIMD instructions.
    ///
    /// # Safety
    /// - Caller must ensure `a` and `b` have the same length.
    /// - This function uses unaligned loads, so no alignment requirements.
    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn euclidean_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            sum = _mm256_fmadd_ps(diff, diff, sum);
        }

        let mut result = hsum_avx(sum);

        // Handle remainder
        for i in (chunks * 8)..a.len() {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result
    }

    /// Compute Manhattan (L1) distance using AVX2 SIMD instructions.
    ///
    /// # Safety
    /// - Caller must ensure `a` and `b` have the same length.
    /// - This function uses unaligned loads, so no alignment requirements.
    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn manhattan_avx2(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let sign_mask = _mm256_set1_ps(-0.0);
        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff);
            sum = _mm256_add_ps(sum, abs_diff);
        }

        let mut result = hsum_avx(sum);

        // Handle remainder
        for i in (chunks * 8)..a.len() {
            result += (a[i] - b[i]).abs();
        }

        result
    }

    #[target_feature(enable = "avx2")]
    #[cfg(target_feature = "avx2")]
    pub unsafe fn scale_avx2(vector: &mut [f32], scale: f32) {
        let scale_vec = _mm256_set1_ps(scale);
        let chunks = vector.len() / 8;

        for i in 0..chunks {
            let ptr = vector.as_mut_ptr().add(i * 8);
            let v = _mm256_loadu_ps(ptr);
            let scaled = _mm256_mul_ps(v, scale_vec);
            _mm256_storeu_ps(ptr, scaled);
        }

        // Handle remainder
        for i in (chunks * 8)..vector.len() {
            vector[i] *= scale;
        }
    }
}

// ARM NEON SIMD implementations
#[cfg(target_arch = "aarch64")]
mod simd_arm {
    use std::arch::aarch64::*;

    /// Compute dot product using ARM NEON SIMD instructions.
    ///
    /// # Safety
    /// - Caller must ensure `a` and `b` have the same length.
    /// - This function uses unaligned loads, so no alignment requirements.
    #[target_feature(enable = "neon")]
    pub unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            sum = vfmaq_f32(sum, va, vb);
        }

        let mut result = vaddvq_f32(sum);

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    /// Compute squared Euclidean distance using ARM NEON SIMD instructions.
    ///
    /// # Safety
    /// - Caller must ensure `a` and `b` have the same length.
    /// - This function uses unaligned loads, so no alignment requirements.
    #[target_feature(enable = "neon")]
    pub unsafe fn euclidean_squared_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            sum = vfmaq_f32(sum, diff, diff);
        }

        let mut result = vaddvq_f32(sum);

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            let diff = a[i] - b[i];
            result += diff * diff;
        }

        result
    }

    /// Compute Manhattan (L1) distance using ARM NEON SIMD instructions.
    ///
    /// # Safety
    /// - Caller must ensure `a` and `b` have the same length.
    /// - This function uses unaligned loads, so no alignment requirements.
    #[target_feature(enable = "neon")]
    pub unsafe fn manhattan_neon(a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "vectors must have equal length");
        let mut sum = vdupq_n_f32(0.0);
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let va = vld1q_f32(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            let abs_diff = vabsq_f32(diff);
            sum = vaddq_f32(sum, abs_diff);
        }

        let mut result = vaddvq_f32(sum);

        // Handle remainder
        for i in (chunks * 4)..a.len() {
            result += (a[i] - b[i]).abs();
        }

        result
    }

    #[target_feature(enable = "neon")]
    #[allow(clippy::needless_range_loop)]
    pub unsafe fn scale_neon(vector: &mut [f32], scale: f32) {
        let scale_vec = vdupq_n_f32(scale);
        let chunks = vector.len() / 4;

        for i in 0..chunks {
            let ptr = vector.as_mut_ptr().add(i * 4);
            let v = vld1q_f32(ptr);
            let scaled = vmulq_f32(v, scale_vec);
            vst1q_f32(ptr, scaled);
        }

        // Handle remainder
        for i in (chunks * 4)..vector.len() {
            vector[i] *= scale;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 2.0];
        let result = euclidean_distance(&a, &b);
        assert!((result - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let result = cosine_distance(&a, &b);
        assert!((result - 1.0).abs() < 1e-6); // Orthogonal vectors

        let c = vec![1.0, 0.0];
        let d = vec![1.0, 0.0];
        let result2 = cosine_distance(&c, &d);
        assert!(result2.abs() < 1e-6); // Same direction
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = manhattan_distance(&a, &b);
        assert!((result - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_distance_normalized() {
        // Test with pre-normalized vectors
        let a = normalized(&[1.0, 0.0]);
        let b = normalized(&[0.0, 1.0]);
        let result = cosine_distance_normalized(&a, &b);
        assert!((result - 1.0).abs() < 1e-6); // Orthogonal vectors

        let c = normalized(&[1.0, 0.0]);
        let d = normalized(&[1.0, 0.0]);
        let result2 = cosine_distance_normalized(&c, &d);
        assert!(result2.abs() < 1e-6); // Same direction

        // Verify it matches regular cosine for normalized vectors
        let v1 = normalized(&[1.0, 2.0, 3.0]);
        let v2 = normalized(&[4.0, 5.0, 6.0]);
        let regular = cosine_distance(&v1, &v2);
        let fast = cosine_distance_normalized(&v1, &v2);
        assert!((regular - fast).abs() < 1e-5, "Normalized cosine should match regular for unit vectors");
    }

    #[test]
    fn test_distance_function_cosine_normalized() {
        let a = normalized(&[1.0, 2.0, 3.0, 4.0]);
        let b = normalized(&[5.0, 6.0, 7.0, 8.0]);

        let regular = DistanceFunction::Cosine.compute(&a, &b);
        let fast = DistanceFunction::CosineNormalized.compute(&a, &b);

        assert!((regular - fast).abs() < 1e-5);
    }
}
