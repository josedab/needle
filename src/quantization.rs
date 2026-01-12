use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Scalar Quantization (SQ8) - quantizes f32 to u8
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
            };
        }

        let dims = vectors[0].len();
        let mut min_vals = vec![f32::MAX; dims];
        let mut max_vals = vec![f32::MIN; dims];

        // Find min/max per dimension
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                assert!(v.is_finite(), "Training vector contains non-finite value at index {}: {}", i, v);
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

        Self {
            min_vals,
            max_vals,
            scale,
            dimensions: dims,
        }
    }

    /// Get dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Quantize a f32 vector to u8
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        vector
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let normalized = (v - self.min_vals[i]) * self.scale[i];
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    /// Dequantize a u8 vector back to f32
    pub fn dequantize(&self, codes: &[u8]) -> Vec<f32> {
        codes
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let normalized = c as f32 / self.scale[i];
                normalized + self.min_vals[i]
            })
            .collect()
    }

    /// Compute squared Euclidean distance between quantized vectors
    pub fn distance_squared(&self, a: &[u8], b: &[u8]) -> f32 {
        let mut sum: u32 = 0;
        for (va, vb) in a.iter().zip(b.iter()) {
            let diff = (*va as i32) - (*vb as i32);
            sum += (diff * diff) as u32;
        }

        // Scale back to original space (approximate)
        let avg_scale: f32 = self.scale.iter().sum::<f32>() / self.scale.len() as f32;
        (sum as f32) / (avg_scale * avg_scale)
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
        let mut codes = Vec::with_capacity(self.num_subvectors);

        for i in 0..self.num_subvectors {
            let start = i * self.subvector_dim;
            let end = start + self.subvector_dim;
            let subvec = &vector[start..end];

            let code = self.find_nearest_centroid(i, subvec);
            codes.push(code);
        }

        codes
    }

    /// Decode PQ codes back to approximate vector
    pub fn decode(&self, codes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.num_subvectors * self.subvector_dim);

        for (i, &code) in codes.iter().enumerate() {
            vector.extend_from_slice(&self.codebooks[i][code as usize]);
        }

        vector
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
}
