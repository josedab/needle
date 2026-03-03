//! # Vector Quantization
//!
//! This module provides quantization techniques for compressing vectors to reduce
//! memory usage while maintaining search quality.
//!
//! ## Available Quantizers
//!
//! - [`ScalarQuantizer`]: 8-bit scalar quantization (4x compression)
//! - [`ProductQuantizer`]: Product quantization for higher compression ratios
//! - [`BinaryQuantizer`]: 1-bit per dimension (32x compression)
//!
//! ## Example: Scalar Quantization
//!
//! ```
//! use needle::ScalarQuantizer;
//!
//! // Training vectors
//! let vectors: Vec<Vec<f32>> = vec![
//!     vec![0.1, 0.2, 0.3, 0.4],
//!     vec![0.5, 0.6, 0.7, 0.8],
//!     vec![0.2, 0.3, 0.4, 0.5],
//! ];
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! // Train the quantizer
//! let sq = ScalarQuantizer::train(&refs);
//!
//! // Quantize a vector
//! let original = vec![0.3, 0.4, 0.5, 0.6];
//! let quantized = sq.quantize(&original);
//! assert_eq!(quantized.len(), 4);
//!
//! // Dequantize back to f32
//! let restored = sq.dequantize(&quantized);
//! assert_eq!(restored.len(), 4);
//! ```
//!
//! ## Example: Product Quantization
//!
//! ```
//! use needle::ProductQuantizer;
//!
//! // Training vectors (dimension must be divisible by num_subvectors)
//! let vectors: Vec<Vec<f32>> = (0..100)
//!     .map(|i| vec![i as f32 * 0.01; 8])
//!     .collect();
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! // Train with 2 subvectors (each subvector is 4 dimensions)
//! let pq = ProductQuantizer::train(&refs, 2);
//!
//! // Encode a vector to PQ codes
//! let vector = vec![0.5; 8];
//! let codes = pq.encode(&vector);
//! assert_eq!(codes.len(), 2); // One code per subvector
//!
//! // Decode back to approximate vector
//! let decoded = pq.decode(&codes);
//! assert_eq!(decoded.len(), 8);
//! ```
//!
//! ## Example: Binary Quantization
//!
//! ```
//! use needle::BinaryQuantizer;
//!
//! // Training vectors
//! let vectors: Vec<Vec<f32>> = vec![
//!     vec![0.1, 0.5, 0.3, 0.8, 0.2, 0.6, 0.4, 0.7],
//!     vec![0.9, 0.4, 0.7, 0.2, 0.8, 0.3, 0.6, 0.5],
//! ];
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! // Train the binary quantizer
//! let bq = BinaryQuantizer::train(&refs);
//!
//! // Quantize to binary codes (1 byte for 8 dimensions)
//! let binary = bq.quantize(&vectors[0]);
//! assert_eq!(binary.len(), 1); // 8 dimensions fit in 1 byte
//!
//! // Compute Hamming distance between binary codes
//! let binary2 = bq.quantize(&vectors[1]);
//! let distance = BinaryQuantizer::hamming_distance(&binary, &binary2);
//! assert!(distance <= 8); // Max 8 bits can differ
//! ```
//!
//! ## Theoretical Background
//!
//! ### Product Quantization — k-means Subvector Partitioning
//!
//! Product quantization (PQ) partitions each vector into `m` equal-length subvectors and
//! independently clusters each subspace using k-means. This decomposition exploits the
//! observation that high-dimensional distances can be well-approximated by summing
//! per-subvector distances, reducing the codebook size from `k^d` (full-space quantization)
//! to `m * k` (sum of subspace codebooks). Each subvector is replaced by the index of its
//! nearest centroid, yielding compact `m`-byte codes.
//!
//! ### Asymmetric Distance Computation (ADC)
//!
//! During search, PQ uses asymmetric distance computation: the query vector is left
//! unquantized while database vectors are represented by their PQ codes. For each query,
//! a lookup table of distances from the query's subvectors to all centroids is precomputed
//! (`m * k` distance calculations). The approximate distance to any database vector is then
//! a sum of `m` table lookups — O(m) per candidate instead of O(d). This asymmetry yields
//! better accuracy than symmetric (quantized-vs-quantized) comparison at negligible extra cost.

use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Scalar Quantization (SQ8) - quantizes f32 to u8
///
/// Scalar quantization maps each floating-point value to an 8-bit integer,
/// achieving 4x compression with minimal accuracy loss.
///
/// # Example
///
/// ```
/// use needle::ScalarQuantizer;
///
/// let training_data: Vec<Vec<f32>> = vec![
///     vec![0.0, 1.0, 2.0],
///     vec![3.0, 4.0, 5.0],
/// ];
/// let refs: Vec<&[f32]> = training_data.iter().map(|v| v.as_slice()).collect();
///
/// let sq = ScalarQuantizer::train(&refs);
/// let quantized = sq.quantize(&[1.5, 2.5, 3.5]);
/// let restored = sq.dequantize(&quantized);
///
/// // Restored values are close to originals
/// assert!((restored[0] - 1.5).abs() < 0.1);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalarQuantizer {
    /// Minimum value per dimension
    min_vals: Vec<f32>,
    /// Maximum value per dimension
    max_vals: Vec<f32>,
    /// Scale factor per dimension
    scale: Vec<f32>,
    /// Dimensions
    dimensions: usize,
    /// Cached average scale for distance computation (precomputed for performance)
    #[serde(default)]
    avg_scale_cached: f32,
}

impl ScalarQuantizer {
    /// Train the quantizer on a set of vectors
    ///
    /// # Panics
    /// Panics if any vector contains NaN or infinite values.
    pub fn train(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            return Self {
                min_vals: Vec::new(),
                max_vals: Vec::new(),
                scale: Vec::new(),
                dimensions: 0,
                avg_scale_cached: 1.0,
            };
        }

        let dims = vectors[0].len();
        let mut min_vals = vec![f32::MAX; dims];
        let mut max_vals = vec![f32::MIN; dims];

        // Find min/max per dimension
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "Training vector contains non-finite value at index {}: {}",
                    i,
                    v
                );
                min_vals[i] = min_vals[i].min(v);
                max_vals[i] = max_vals[i].max(v);
            }
        }

        // Compute scale factors
        let scale: Vec<f32> = min_vals
            .iter()
            .zip(max_vals.iter())
            .map(|(min, max)| {
                let range = max - min;
                if range > 1e-10 {
                    255.0 / range
                } else {
                    1.0
                }
            })
            .collect();

        // Precompute average scale for distance computation
        let avg_scale_cached = if scale.is_empty() {
            1.0
        } else {
            scale.iter().sum::<f32>() / scale.len() as f32
        };

        Self {
            min_vals,
            max_vals,
            scale,
            dimensions: dims,
            avg_scale_cached,
        }
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Quantize a f32 vector to u8
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        let mut out = vec![0u8; vector.len()];
        self.quantize_into(vector, &mut out);
        out
    }

    /// Quantize a f32 vector into a pre-allocated buffer.
    ///
    /// This avoids allocation in hot paths. The output buffer must be at least
    /// as long as the input vector.
    ///
    /// # Panics
    /// Panics if `out.len() < vector.len()`.
    pub fn quantize_into(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= vector.len(), "output buffer too small");
        for (i, &v) in vector.iter().enumerate() {
            let normalized = (v - self.min_vals[i]) * self.scale[i];
            out[i] = normalized.clamp(0.0, 255.0) as u8;
        }
    }

    /// Dequantize a u8 vector back to f32
    pub fn dequantize(&self, codes: &[u8]) -> Vec<f32> {
        let mut out = vec![0.0f32; codes.len()];
        self.dequantize_into(codes, &mut out);
        out
    }

    /// Dequantize a u8 vector into a pre-allocated buffer.
    ///
    /// This avoids allocation in hot paths. The output buffer must be at least
    /// as long as the codes slice.
    ///
    /// # Panics
    /// Panics if `out.len() < codes.len()`.
    pub fn dequantize_into(&self, codes: &[u8], out: &mut [f32]) {
        assert!(out.len() >= codes.len(), "output buffer too small");
        for (i, &c) in codes.iter().enumerate() {
            let normalized = c as f32 / self.scale[i];
            out[i] = normalized + self.min_vals[i];
        }
    }

    /// Compute squared Euclidean distance between quantized vectors
    ///
    /// Uses precomputed average scale for faster distance computation.
    pub fn distance_squared(&self, a: &[u8], b: &[u8]) -> f32 {
        let mut sum: u32 = 0;
        for (va, vb) in a.iter().zip(b.iter()) {
            let diff = (*va as i32) - (*vb as i32);
            sum += (diff * diff) as u32;
        }

        // Scale back to original space using cached average scale
        (sum as f32) / (self.avg_scale_cached * self.avg_scale_cached)
    }

    /// Compute asymmetric distance (query is f32, db vector is u8)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (i, (&q, &c)) in query.iter().zip(codes.iter()).enumerate() {
            let decoded = (c as f32 / self.scale[i]) + self.min_vals[i];
            let diff = q - decoded;
            sum += diff * diff;
        }
        sum.sqrt()
    }
}

/// Product Quantization (PQ) - divides vector into subvectors and quantizes each
///
/// Product quantization achieves high compression ratios by dividing vectors into
/// subvectors and learning a codebook for each subspace. This is particularly
/// effective for high-dimensional vectors.
///
/// # Example
///
/// ```
/// use needle::ProductQuantizer;
///
/// // Create training data (8-dimensional vectors)
/// let vectors: Vec<Vec<f32>> = (0..50)
///     .map(|i| (0..8).map(|j| (i * j) as f32 * 0.01).collect())
///     .collect();
/// let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
///
/// // Train with 2 subvectors
/// let pq = ProductQuantizer::train(&refs, 2);
/// assert_eq!(pq.num_subvectors(), 2);
/// assert_eq!(pq.subvector_dim(), 4);
///
/// // Encode and decode
/// let original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
/// let codes = pq.encode(&original);
/// let decoded = pq.decode(&codes);
/// assert_eq!(decoded.len(), 8);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProductQuantizer {
    /// Number of subvectors
    num_subvectors: usize,
    /// Dimension of each subvector
    subvector_dim: usize,
    /// Codebooks: [num_subvectors][256][subvector_dim]
    codebooks: Vec<Vec<Vec<f32>>>,
    /// Number of centroids per subspace (usually 256 for 8-bit codes)
    num_centroids: usize,
}

impl ProductQuantizer {
    /// Train the product quantizer using k-means
    pub fn train(vectors: &[&[f32]], num_subvectors: usize) -> Self {
        if vectors.is_empty() {
            return Self {
                num_subvectors,
                subvector_dim: 0,
                codebooks: Vec::new(),
                num_centroids: 256,
            };
        }

        let dim = vectors[0].len();
        let subvector_dim = dim / num_subvectors;
        let num_centroids = 256; // 8-bit codes

        let mut codebooks = Vec::with_capacity(num_subvectors);

        for i in 0..num_subvectors {
            // Extract subvectors for this partition
            let subvectors: Vec<Vec<f32>> = vectors
                .iter()
                .map(|v| v[i * subvector_dim..(i + 1) * subvector_dim].to_vec())
                .collect();

            // Run k-means to find centroids
            let centroids = kmeans(&subvectors, num_centroids, 20);
            codebooks.push(centroids);
        }

        Self {
            num_subvectors,
            subvector_dim,
            codebooks,
            num_centroids,
        }
    }

    /// Get number of subvectors
    pub fn num_subvectors(&self) -> usize {
        self.num_subvectors
    }

    /// Get subvector dimension
    pub fn subvector_dim(&self) -> usize {
        self.subvector_dim
    }

    /// Encode a vector to PQ codes
    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = vec![0u8; self.num_subvectors];
        self.encode_into(vector, &mut codes);
        codes
    }

    /// Encode a vector into a pre-allocated buffer.
    ///
    /// This avoids allocation in hot paths. The output buffer must be at least
    /// `num_subvectors()` bytes long.
    ///
    /// # Panics
    /// Panics if `out.len() < self.num_subvectors()`.
    pub fn encode_into(&self, vector: &[f32], out: &mut [u8]) {
        assert!(out.len() >= self.num_subvectors, "output buffer too small");
        for (i, out_byte) in out.iter_mut().take(self.num_subvectors).enumerate() {
            let start = i * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvec = &vector[start..end];
            *out_byte = self.find_nearest_centroid(i, subvec);
        }
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.num_subvectors * self.subvector_dim];
        self.decode_into(codes, &mut vector);
        vector
    }

    /// Decode PQ codes into a pre-allocated buffer.
    ///
    /// This avoids allocation in hot paths. The output buffer must be at least
    /// `num_subvectors() * subvector_dim()` floats long.
    ///
    /// # Panics
    /// Panics if `out.len() < self.num_subvectors() * self.subvector_dim()`.
    pub fn decode_into(&self, codes: &[u8], out: &mut [f32]) {
        let required_len = self.num_subvectors * self.subvector_dim;
        assert!(out.len() >= required_len, "output buffer too small");
        for (i, &code) in codes.iter().enumerate().take(self.num_subvectors) {
            let start = i * self.subvector_dim;
            let centroid = &self.codebooks[i][code as usize];
            out[start..start + self.subvector_dim].copy_from_slice(centroid);
        }
    }

    /// Find the nearest centroid for a subvector
    fn find_nearest_centroid(&self, partition: usize, subvec: &[f32]) -> u8 {
        let mut best_code = 0u8;
        let mut best_dist = f32::MAX;

        for (code, centroid) in self.codebooks[partition].iter().enumerate() {
            let dist: f32 = subvec
                .iter()
                .zip(centroid.iter())
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();

            if dist < best_dist {
                best_dist = dist;
                best_code = code as u8;
            }
        }

        best_code
    }

    /// Compute asymmetric distance (query is not quantized)
    pub fn asymmetric_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;

        for (i, &code) in codes.iter().enumerate().take(self.num_subvectors) {
            let start = i * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subquery = &query[start..end];
            let centroid = &self.codebooks[i][code as usize];

            for (q, c) in subquery.iter().zip(centroid.iter()) {
                let diff = q - c;
                dist += diff * diff;
            }
        }

        dist.sqrt()
    }

    /// Precompute distance tables for faster search
    pub fn compute_distance_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let mut table = Vec::with_capacity(self.num_subvectors);

        for i in 0..self.num_subvectors {
            let start = i * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subquery = &query[start..end];

            let distances: Vec<f32> = self.codebooks[i]
                .iter()
                .map(|centroid| {
                    subquery
                        .iter()
                        .zip(centroid.iter())
                        .map(|(q, c)| {
                            let d = q - c;
                            d * d
                        })
                        .sum()
                })
                .collect();

            table.push(distances);
        }

        table
    }

    /// Fast distance computation using precomputed table
    pub fn distance_with_table(&self, table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let mut dist = 0.0f32;
        for (i, &code) in codes.iter().enumerate() {
            dist += table[i][code as usize];
        }
        dist.sqrt()
    }
}

/// Binary Quantization - extreme compression using 1 bit per dimension
///
/// Binary quantization offers the highest compression ratio (32x for f32 vectors)
/// by representing each dimension as a single bit. Values above the learned threshold
/// are encoded as 1, values below as 0.
///
/// # Example
///
/// ```
/// use needle::BinaryQuantizer;
///
/// // Training data
/// let vectors: Vec<Vec<f32>> = vec![
///     vec![0.1, 0.9, 0.2, 0.8],
///     vec![0.3, 0.7, 0.4, 0.6],
/// ];
/// let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
///
/// // Train the quantizer
/// let bq = BinaryQuantizer::train(&refs);
///
/// // Quantize vectors
/// let code1 = bq.quantize(&vectors[0]);
/// let code2 = bq.quantize(&vectors[1]);
///
/// // Compute Hamming distance (number of differing bits)
/// let dist = BinaryQuantizer::hamming_distance(&code1, &code2);
/// println!("Hamming distance: {}", dist);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryQuantizer {
    /// Threshold per dimension (usually mean)
    thresholds: Vec<f32>,
    /// Dimensions
    dimensions: usize,
}

impl BinaryQuantizer {
    /// Train the binary quantizer
    pub fn train(vectors: &[&[f32]]) -> Self {
        if vectors.is_empty() {
            return Self {
                thresholds: Vec::new(),
                dimensions: 0,
            };
        }

        let dims = vectors[0].len();
        let n = vectors.len() as f32;

        // Compute mean per dimension as threshold
        let mut thresholds = vec![0.0f32; dims];
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                thresholds[i] += v;
            }
        }
        for t in &mut thresholds {
            *t /= n;
        }

        Self {
            thresholds,
            dimensions: dims,
        }
    }

    /// Quantize to binary (packed into bytes)
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        let num_bytes = self.dimensions.div_ceil(8);
        let mut codes = vec![0u8; num_bytes];

        for (i, &v) in vector.iter().enumerate() {
            if v > self.thresholds[i] {
                codes[i / 8] |= 1 << (i % 8);
            }
        }

        codes
    }

    /// Compute Hamming distance between binary codes
    pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x ^ y).count_ones())
            .sum()
    }
}

/// Simple k-means clustering
fn kmeans(vectors: &[Vec<f32>], k: usize, max_iterations: usize) -> Vec<Vec<f32>> {
    if vectors.is_empty() || k == 0 {
        return Vec::new();
    }

    let dim = vectors[0].len();
    let n = vectors.len();
    let k = k.min(n); // Can't have more centroids than points

    let mut rng = rand::thread_rng();

    // Initialize centroids using k-means++ style initialization
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);

    // Pick first centroid randomly
    centroids.push(vectors[rng.gen_range(0..n)].clone());

    // Pick remaining centroids with probability proportional to distance squared
    while centroids.len() < k {
        let mut distances: Vec<f32> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| euclidean_distance_squared(v, c))
                    .fold(f32::MAX, f32::min)
            })
            .collect();

        let total: f32 = distances.iter().sum();
        if total <= 0.0 {
            // All points are at centroids, just pick randomly
            let remaining: Vec<usize> = (0..n)
                .filter(|i| !centroids.iter().any(|c| c == &vectors[*i]))
                .collect();
            if let Some(&idx) = remaining.choose(&mut rng) {
                centroids.push(vectors[idx].clone());
            } else {
                break;
            }
            continue;
        }

        // Normalize to probabilities
        for d in &mut distances {
            *d /= total;
        }

        // Sample based on probability
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (i, &d) in distances.iter().enumerate() {
            cumsum += d;
            if cumsum >= r {
                centroids.push(vectors[i].clone());
                break;
            }
        }
    }

    // Run k-means iterations
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iterations {
        // Assign each point to nearest centroid
        let mut changed = false;
        for (i, vec) in vectors.iter().enumerate() {
            let mut best_centroid = 0;
            let mut best_dist = f32::MAX;

            for (j, centroid) in centroids.iter().enumerate() {
                let dist = euclidean_distance_squared(vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_centroid = j;
                }
            }

            if assignments[i] != best_centroid {
                assignments[i] = best_centroid;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        // Update centroids
        let mut new_centroids = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];

        for (i, &cluster) in assignments.iter().enumerate() {
            counts[cluster] += 1;
            for (j, &v) in vectors[i].iter().enumerate() {
                new_centroids[cluster][j] += v;
            }
        }

        for (i, centroid) in new_centroids.iter_mut().enumerate() {
            if counts[i] > 0 {
                for v in centroid.iter_mut() {
                    *v /= counts[i] as f32;
                }
            } else {
                // Empty cluster, reinitialize randomly
                *centroid = vectors[rng.gen_range(0..n)].clone();
            }
        }

        centroids = new_centroids;
    }

    centroids
}

/// Squared Euclidean distance
fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

// ── Matryoshka Dimension Truncation ─────────────────────────────────────────

/// Adaptive dimension truncation for Matryoshka-style embeddings.
///
/// Matryoshka embeddings are trained so that prefix subsets of dimensions
/// retain meaningful representations. This enables searching at reduced
/// dimensionality for speed, then re-ranking at full dimensionality for
/// accuracy.
///
/// # Supported Truncation Levels
///
/// Common truncation: 384→128→64, 768→256→128, 1536→512→256.
///
/// # Example
///
/// ```
/// use needle::MatryoshkaTruncation;
///
/// let truncation = MatryoshkaTruncation::new(384, vec![128, 64]);
///
/// let full_vector = vec![0.1f32; 384];
/// let truncated = truncation.truncate(&full_vector, 128);
/// assert_eq!(truncated.len(), 128);
///
/// // Distance correction compensates for lost dimensions
/// let raw_dist = 0.5f32;
/// let corrected = truncation.correct_distance(raw_dist, 128);
/// assert!(corrected > 0.0);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaTruncation {
    /// Original full dimensionality.
    full_dimensions: usize,
    /// Supported truncation targets (sorted descending).
    truncation_levels: Vec<usize>,
    /// Per-level variance ratios learned during calibration.
    /// If empty, uses dimension-ratio heuristic.
    variance_ratios: Vec<f32>,
}

impl MatryoshkaTruncation {
    /// Create a new truncation engine with the given full dimensionality
    /// and supported truncation levels.
    pub fn new(full_dimensions: usize, mut truncation_levels: Vec<usize>) -> Self {
        truncation_levels.sort_unstable_by(|a, b| b.cmp(a));
        truncation_levels.retain(|&d| d > 0 && d < full_dimensions);
        Self {
            full_dimensions,
            truncation_levels,
            variance_ratios: Vec::new(),
        }
    }

    /// Calibrate variance ratios from a sample of vectors for more accurate
    /// distance correction.
    pub fn calibrate(&mut self, sample_vectors: &[&[f32]]) {
        if sample_vectors.is_empty() || self.truncation_levels.is_empty() {
            return;
        }
        let full_var = Self::compute_variance(sample_vectors, self.full_dimensions);
        if full_var < f32::EPSILON {
            return;
        }
        self.variance_ratios = self
            .truncation_levels
            .iter()
            .map(|&dims| {
                let trunc_var = Self::compute_variance(sample_vectors, dims);
                (trunc_var / full_var).clamp(0.01, 1.0)
            })
            .collect();
    }

    /// Truncate a vector to the specified number of dimensions.
    ///
    /// Returns the first `target_dims` elements. If `target_dims` exceeds
    /// the vector length, returns the full vector.
    pub fn truncate<'a>(&self, vector: &'a [f32], target_dims: usize) -> &'a [f32] {
        let dims = target_dims.min(vector.len());
        &vector[..dims]
    }

    /// Apply distance correction factor to compensate for truncated dimensions.
    ///
    /// Uses calibrated variance ratios when available, otherwise falls back to
    /// a dimension-ratio heuristic: `distance * (full_dims / truncated_dims)`.
    pub fn correct_distance(&self, raw_distance: f32, truncated_dims: usize) -> f32 {
        if truncated_dims >= self.full_dimensions || truncated_dims == 0 {
            return raw_distance;
        }
        if let Some(idx) = self.truncation_levels.iter().position(|&d| d == truncated_dims) {
            if let Some(&ratio) = self.variance_ratios.get(idx) {
                return raw_distance / ratio;
            }
        }
        // Heuristic: scale by dimension ratio
        raw_distance * (self.full_dimensions as f32 / truncated_dims as f32)
    }

    /// Return the best truncation level ≤ `max_dims`.
    /// Returns `full_dimensions` if no truncation level fits.
    pub fn nearest_level(&self, max_dims: usize) -> usize {
        self.truncation_levels
            .iter()
            .find(|&&d| d <= max_dims)
            .copied()
            .unwrap_or(self.full_dimensions)
    }

    /// Full dimensionality this engine was configured for.
    pub fn full_dimensions(&self) -> usize {
        self.full_dimensions
    }

    /// Supported truncation levels (sorted descending).
    pub fn truncation_levels(&self) -> &[usize] {
        &self.truncation_levels
    }

    /// Compute the memory savings ratio for a given truncation level.
    /// Returns a value like 3.0 meaning "3× memory savings".
    pub fn memory_savings(&self, truncated_dims: usize) -> f32 {
        if truncated_dims == 0 {
            return 0.0;
        }
        self.full_dimensions as f32 / truncated_dims as f32
    }

    fn compute_variance(vectors: &[&[f32]], dims: usize) -> f32 {
        if vectors.is_empty() || dims == 0 {
            return 0.0;
        }
        let n = vectors.len() as f32;
        let mut sum = vec![0.0f32; dims];
        let mut sum_sq = vec![0.0f32; dims];
        for v in vectors {
            let d = dims.min(v.len());
            for i in 0..d {
                sum[i] += v[i];
                sum_sq[i] += v[i] * v[i];
            }
        }
        let mut var_total = 0.0f32;
        for i in 0..dims {
            let mean = sum[i] / n;
            var_total += sum_sq[i] / n - mean * mean;
        }
        var_total
    }
}

/// Result of a two-phase adaptive search using Matryoshka truncation.
#[derive(Debug, Clone)]
pub struct AdaptiveSearchResult {
    /// Vector ID.
    pub id: usize,
    /// Distance at full dimensionality after re-ranking.
    pub distance: f32,
    /// Distance at reduced dimensionality (fast scan phase).
    pub coarse_distance: f32,
}

/// Perform a two-phase adaptive search using Matryoshka truncation.
///
/// Phase 1: Fast scan at `coarse_dims` with over-fetch factor.
/// Phase 2: Re-rank top candidates at full dimensionality.
pub fn adaptive_search(
    query: &[f32],
    vectors: &[Vec<f32>],
    truncation: &MatryoshkaTruncation,
    coarse_dims: usize,
    k: usize,
    overfetch_factor: usize,
) -> Vec<AdaptiveSearchResult> {
    use ordered_float::OrderedFloat;
    use std::collections::BinaryHeap;
    use std::cmp::Reverse;

    let effective_dims = truncation.nearest_level(coarse_dims);
    let coarse_k = k * overfetch_factor.max(1);
    let truncated_query = truncation.truncate(query, effective_dims);

    // Phase 1: coarse search at reduced dimensions
    let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();
    for (idx, vec) in vectors.iter().enumerate() {
        let truncated_vec = truncation.truncate(vec, effective_dims);
        let dist = euclidean_distance_squared(truncated_query, truncated_vec).sqrt();
        if heap.len() < coarse_k {
            heap.push(Reverse((OrderedFloat(dist), idx)));
        } else if let Some(&Reverse((top_dist, _))) = heap.peek() {
            if OrderedFloat(dist) < top_dist {
                heap.pop();
                heap.push(Reverse((OrderedFloat(dist), idx)));
            }
        }
    }

    let candidates: Vec<(usize, f32)> = heap
        .into_sorted_vec()
        .into_iter()
        .map(|Reverse((d, idx))| (idx, d.into_inner()))
        .collect();

    // Phase 2: re-rank at full dimensionality
    let mut results: Vec<AdaptiveSearchResult> = candidates
        .into_iter()
        .map(|(idx, coarse_dist)| {
            let full_dist = euclidean_distance_squared(query, &vectors[idx]).sqrt();
            AdaptiveSearchResult {
                id: idx,
                distance: full_dist,
                coarse_distance: coarse_dist,
            }
        })
        .collect();

    results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

// ── Quantized Index Persistence ─────────────────────────────────────────────

/// Envelope for persisting any quantized index alongside its type tag.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizedIndex {
    /// Scalar quantization (4× compression)
    Scalar(ScalarQuantizer),
    /// Product quantization (8-32× compression)
    Product(ProductQuantizer),
    /// Binary quantization (32× compression)
    Binary(BinaryQuantizer),
    /// Matryoshka dimension truncation (3-6× compression)
    Matryoshka(MatryoshkaTruncation),
}

impl QuantizedIndex {
    /// Serialize the quantized index to bytes (JSON format).
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Deserialize a quantized index from bytes.
    pub fn from_bytes(data: &[u8]) -> std::result::Result<Self, String> {
        serde_json::from_slice(data).map_err(|e| format!("Failed to deserialize quantized index: {}", e))
    }

    /// Returns the compression ratio description.
    pub fn compression_label(&self) -> &'static str {
        match self {
            Self::Scalar(_) => "4x (scalar u8)",
            Self::Product(_) => "8-32x (product)",
            Self::Binary(_) => "32x (binary)",
            Self::Matryoshka(_) => "3-6x (matryoshka truncation)",
        }
    }

    /// Returns the number of dimensions this quantizer was trained for.
    pub fn dimensions(&self) -> usize {
        match self {
            Self::Scalar(q) => q.dimensions(),
            Self::Product(q) => q.num_subvectors() * q.subvector_dim(),
            Self::Binary(q) => q.thresholds.len(),
            Self::Matryoshka(q) => q.full_dimensions(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_quantizer() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.5, 1.0],
            vec![0.2, 0.7, 0.8],
            vec![0.1, 0.3, 0.9],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let sq = ScalarQuantizer::train(&refs);

        let original = vec![0.15, 0.4, 0.85];
        let quantized = sq.quantize(&original);
        let dequantized = sq.dequantize(&quantized);

        // Check that dequantized is close to original
        for (o, d) in original.iter().zip(dequantized.iter()) {
            assert!(
                (o - d).abs() < 0.1,
                "Dequantized value too far from original"
            );
        }
    }

    #[test]
    fn test_product_quantizer() {
        let dim = 16;
        let num_subvectors = 4;
        let mut rng = rand::thread_rng();

        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, num_subvectors);

        // Test encode/decode
        let original = &vectors[0];
        let codes = pq.encode(original);
        let decoded = pq.decode(&codes);

        assert_eq!(codes.len(), num_subvectors);
        assert_eq!(decoded.len(), dim);

        // Decoded should be somewhat close to original
        let error: f32 = original
            .iter()
            .zip(decoded.iter())
            .map(|(o, d)| (o - d).powi(2))
            .sum::<f32>()
            .sqrt();

        assert!(error < 2.0, "PQ reconstruction error too high: {}", error);
    }

    #[test]
    fn test_product_quantizer_distance_table() {
        let dim = 8;
        let num_subvectors = 2;
        let mut rng = rand::thread_rng();

        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pq = ProductQuantizer::train(&refs, num_subvectors);

        let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        let table = pq.compute_distance_table(&query);

        // Test that table-based distance matches asymmetric distance
        for vec in &vectors {
            let codes = pq.encode(vec);
            let d1 = pq.asymmetric_distance(&query, &codes);
            let d2 = pq.distance_with_table(&table, &codes);

            assert!(
                (d1 - d2).abs() < 1e-5,
                "Distance mismatch: {} vs {}",
                d1,
                d2
            );
        }
    }

    #[test]
    fn test_binary_quantizer() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 1.0, 0.0, 1.0],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![1.0, 0.0, 1.0, 0.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let bq = BinaryQuantizer::train(&refs);

        let v1 = vec![0.0, 1.0, 0.0, 1.0];
        let v2 = vec![1.0, 0.0, 1.0, 0.0];

        let c1 = bq.quantize(&v1);
        let c2 = bq.quantize(&v2);

        // These should be maximally different
        let dist = BinaryQuantizer::hamming_distance(&c1, &c2);
        assert!(dist > 0, "Binary codes should be different");
    }

    // ── Edge case tests ──────────────────────────────────────────────────

    #[test]
    fn test_scalar_quantizer_empty_training() {
        let sq = ScalarQuantizer::train(&[]);
        assert_eq!(sq.dimensions(), 0);
    }

    #[test]
    fn test_scalar_quantizer_single_vector() {
        let vectors: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        let quantized = sq.quantize(&[1.0, 2.0, 3.0]);
        let dequantized = sq.dequantize(&quantized);
        assert_eq!(dequantized.len(), 3);
    }

    #[test]
    fn test_scalar_quantizer_identical_values() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![5.0, 5.0, 5.0],
            vec![5.0, 5.0, 5.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        // Range is 0, scale should be 1.0 (fallback)
        let quantized = sq.quantize(&[5.0, 5.0, 5.0]);
        assert_eq!(quantized.len(), 3);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn test_scalar_quantizer_nan_training_panics() {
        let vectors: Vec<Vec<f32>> = vec![vec![f32::NAN, 1.0, 2.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        ScalarQuantizer::train(&refs);
    }

    #[test]
    #[should_panic(expected = "non-finite")]
    fn test_scalar_quantizer_infinity_training_panics() {
        let vectors: Vec<Vec<f32>> = vec![vec![f32::INFINITY, 1.0, 2.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        ScalarQuantizer::train(&refs);
    }

    #[test]
    fn test_scalar_quantizer_out_of_range_clamped() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        // Values outside training range should be clamped
        let quantized = sq.quantize(&[-100.0, 100.0]);
        assert_eq!(quantized[0], 0);   // clamped to min
        assert_eq!(quantized[1], 255); // clamped to max
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_scalar_quantize_into_buffer_too_small() {
        let vectors: Vec<Vec<f32>> = vec![vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        let mut out = [0u8; 1]; // too small for 2 dims
        sq.quantize_into(&[0.5, 0.5], &mut out);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_scalar_dequantize_into_buffer_too_small() {
        let vectors: Vec<Vec<f32>> = vec![vec![0.0, 1.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        let mut out = [0.0f32; 1]; // too small
        sq.dequantize_into(&[128, 128], &mut out);
    }

    #[test]
    fn test_scalar_distance_squared() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
        ];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        let a = sq.quantize(&[0.0, 0.0]);
        let b = sq.quantize(&[1.0, 1.0]);
        let dist = sq.distance_squared(&a, &b);
        assert!(dist > 0.0);

        // Same vector should have zero distance
        let self_dist = sq.distance_squared(&a, &a);
        assert!((self_dist - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_product_quantizer_empty_training() {
        let pq = ProductQuantizer::train(&[], 4);
        assert_eq!(pq.subvector_dim(), 0);
    }

    #[test]
    #[should_panic(expected = "output buffer too small")]
    fn test_product_encode_into_buffer_too_small() {
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32 * 0.01; 8])
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let pq = ProductQuantizer::train(&refs, 2);

        let mut out = [0u8; 1]; // too small for 2 subvectors
        pq.encode_into(&[0.5; 8], &mut out);
    }

    #[test]
    fn test_binary_quantizer_empty_training() {
        let bq = BinaryQuantizer::train(&[]);
        assert_eq!(bq.dimensions, 0);
    }

    #[test]
    fn test_binary_quantizer_hamming_distance_same() {
        let dist = BinaryQuantizer::hamming_distance(&[0xFF], &[0xFF]);
        assert_eq!(dist, 0);
    }

    #[test]
    fn test_binary_quantizer_hamming_distance_max() {
        let dist = BinaryQuantizer::hamming_distance(&[0x00], &[0xFF]);
        assert_eq!(dist, 8);
    }

    #[test]
    fn test_binary_quantizer_hamming_distance_empty() {
        let dist = BinaryQuantizer::hamming_distance(&[], &[]);
        assert_eq!(dist, 0);
    }

    // ── QuantizedIndex persistence ───────────────────────────────────────

    #[test]
    fn test_quantized_index_serde_roundtrip_scalar() {
        let vectors: Vec<Vec<f32>> = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        let sq = ScalarQuantizer::train(&refs);

        let qi = QuantizedIndex::Scalar(sq);
        let bytes = qi.to_bytes();
        let restored = QuantizedIndex::from_bytes(&bytes).unwrap();
        assert_eq!(restored.dimensions(), 2);
        assert_eq!(restored.compression_label(), "4x (scalar u8)");
    }

    #[test]
    fn test_quantized_index_from_bytes_corrupted() {
        let result = QuantizedIndex::from_bytes(b"not valid json");
        assert!(result.is_err());
    }

    #[test]
    fn test_quantized_index_from_bytes_empty() {
        let result = QuantizedIndex::from_bytes(b"");
        assert!(result.is_err());
    }

    // ── Matryoshka truncation tests ──────────────────────────────────────

    #[test]
    fn test_matryoshka_truncation_basic() {
        let trunc = MatryoshkaTruncation::new(384, vec![128, 64]);
        assert_eq!(trunc.full_dimensions(), 384);
        assert_eq!(trunc.truncation_levels(), &[128, 64]);
    }

    #[test]
    fn test_matryoshka_truncate_vector() {
        let trunc = MatryoshkaTruncation::new(384, vec![128, 64]);
        let vec = vec![0.1f32; 384];
        let truncated = trunc.truncate(&vec, 128);
        assert_eq!(truncated.len(), 128);
        let truncated_64 = trunc.truncate(&vec, 64);
        assert_eq!(truncated_64.len(), 64);
    }

    #[test]
    fn test_matryoshka_distance_correction() {
        let trunc = MatryoshkaTruncation::new(384, vec![128, 64]);
        let corrected = trunc.correct_distance(1.0, 128);
        assert!((corrected - 3.0).abs() < 0.01); // 384/128 = 3.0
        // Full dims should return raw distance
        assert!((trunc.correct_distance(1.0, 384) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_matryoshka_nearest_level() {
        let trunc = MatryoshkaTruncation::new(384, vec![128, 64]);
        assert_eq!(trunc.nearest_level(200), 128);
        assert_eq!(trunc.nearest_level(64), 64);
        assert_eq!(trunc.nearest_level(32), 384); // no level fits
    }

    #[test]
    fn test_matryoshka_memory_savings() {
        let trunc = MatryoshkaTruncation::new(384, vec![128, 64]);
        assert!((trunc.memory_savings(128) - 3.0).abs() < 0.01);
        assert!((trunc.memory_savings(64) - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_matryoshka_calibrate() {
        let mut trunc = MatryoshkaTruncation::new(8, vec![4, 2]);
        let vectors: Vec<Vec<f32>> = (0..50)
            .map(|i| (0..8).map(|j| (i * 8 + j) as f32 * 0.01).collect())
            .collect();
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
        trunc.calibrate(&refs);
        assert_eq!(trunc.variance_ratios.len(), 2);
        // Calibrated correction should differ from heuristic
        let corrected = trunc.correct_distance(1.0, 4);
        assert!(corrected > 0.0);
    }

    #[test]
    fn test_adaptive_search() {
        let vectors: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let trunc = MatryoshkaTruncation::new(8, vec![4, 2]);
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let results = adaptive_search(&query, &vectors, &trunc, 4, 2, 3);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, 0); // exact match should be first
    }

    #[test]
    fn test_matryoshka_quantized_index_variant() {
        let trunc = MatryoshkaTruncation::new(384, vec![128, 64]);
        let qi = QuantizedIndex::Matryoshka(trunc);
        assert_eq!(qi.dimensions(), 384);
        assert_eq!(qi.compression_label(), "3-6x (matryoshka truncation)");
    }
}
