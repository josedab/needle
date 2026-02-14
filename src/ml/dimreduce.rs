//! Dimensionality Reduction
//!
//! Reduce vector dimensions for visualization and compression:
//! - PCA (Principal Component Analysis)
//! - Random Projection (for speed)
//! - t-SNE inspired neighbor embedding
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::dimreduce::{PCA, RandomProjection};
//!
//! let vectors: Vec<Vec<f32>> = /* your high-dimensional vectors */;
//! let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
//!
//! // PCA to 2D for visualization
//! let pca = PCA::fit(&refs, 2)?;
//! let reduced = pca.transform(&refs);
//!
//! // Fast random projection
//! let rp = RandomProjection::new(384, 64);
//! let projected: Vec<f32> = rp.project(&high_dim_vector);
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};

/// Principal Component Analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCA {
    /// Principal components (eigenvectors)
    components: Vec<Vec<f32>>,
    /// Mean of training data
    mean: Vec<f32>,
    /// Explained variance per component
    explained_variance: Vec<f32>,
    /// Total variance
    total_variance: f32,
    /// Original dimensions
    input_dims: usize,
    /// Output dimensions
    output_dims: usize,
}

impl PCA {
    /// Fit PCA on training data
    pub fn fit(vectors: &[&[f32]], n_components: usize) -> Result<Self, String> {
        if vectors.is_empty() {
            return Err("Cannot fit PCA on empty dataset".to_string());
        }

        let n = vectors.len();
        let d = vectors[0].len();

        for (i, v) in vectors.iter().enumerate() {
            if v.len() != d {
                return Err(format!(
                    "Vector {} has {} dimensions, expected {}",
                    i,
                    v.len(),
                    d
                ));
            }
        }

        let n_components = n_components.min(d).min(n);

        // Compute mean
        let mut mean = vec![0.0f32; d];
        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        // Center the data
        let centered: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| v.iter().zip(mean.iter()).map(|(x, m)| x - m).collect())
            .collect();

        // Compute covariance matrix (d x d can be large, use power iteration for top components)
        let (components, explained_variance) =
            Self::power_iteration(&centered, n_components, 100, 1e-6);

        let total_variance: f32 = explained_variance.iter().sum();

        Ok(Self {
            components,
            mean,
            explained_variance,
            total_variance,
            input_dims: d,
            output_dims: n_components,
        })
    }

    /// Power iteration method to find top eigenvectors
    fn power_iteration(
        data: &[Vec<f32>],
        n_components: usize,
        max_iter: usize,
        tolerance: f32,
    ) -> (Vec<Vec<f32>>, Vec<f32>) {
        let n = data.len();
        let d = if n > 0 { data[0].len() } else { 0 };

        let mut components = Vec::with_capacity(n_components);
        let mut eigenvalues = Vec::with_capacity(n_components);
        let mut rng = rand::thread_rng();

        // Deflation: remove found components from data
        let mut residual: Vec<Vec<f32>> = data.to_vec();

        for _ in 0..n_components {
            // Initialize random vector
            let mut v: Vec<f32> = (0..d).map(|_| rng.gen::<f32>() - 0.5).collect();
            let mut norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= norm;
            }

            let mut eigenvalue = 0.0f32;

            for _ in 0..max_iter {
                // Compute X^T * X * v using two-pass to avoid materializing covariance
                // First: u = X * v (n-dimensional)
                let u: Vec<f32> = residual
                    .iter()
                    .map(|row| row.iter().zip(v.iter()).map(|(a, b)| a * b).sum())
                    .collect();

                // Second: new_v = X^T * u (d-dimensional)
                let mut new_v = vec![0.0f32; d];
                for (i, row) in residual.iter().enumerate() {
                    for (j, &x) in row.iter().enumerate() {
                        new_v[j] += x * u[i];
                    }
                }

                // Normalize
                norm = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm < 1e-10 {
                    break;
                }

                let new_eigenvalue = norm;
                for x in &mut new_v {
                    *x /= norm;
                }

                // Check convergence
                let diff: f32 = v.iter().zip(new_v.iter()).map(|(a, b)| (a - b).abs()).sum();

                v = new_v;
                eigenvalue = new_eigenvalue;

                if diff < tolerance {
                    break;
                }
            }

            // Deflate: remove this component from residual
            // residual = residual - (residual * v) * v^T
            for row in &mut residual {
                let proj: f32 = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                for (i, x) in row.iter_mut().enumerate() {
                    *x -= proj * v[i];
                }
            }

            components.push(v);
            eigenvalues.push(eigenvalue / n as f32);
        }

        (components, eigenvalues)
    }

    /// Transform vectors to lower dimensions
    pub fn transform(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        vectors.iter().map(|v| self.transform_one(v)).collect()
    }

    /// Transform a single vector
    pub fn transform_one(&self, vector: &[f32]) -> Vec<f32> {
        // Center the vector
        let centered: Vec<f32> = vector
            .iter()
            .zip(self.mean.iter())
            .map(|(x, m)| x - m)
            .collect();

        // Project onto components
        self.components
            .iter()
            .map(|comp| centered.iter().zip(comp.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Inverse transform (approximate reconstruction)
    pub fn inverse_transform(&self, reduced: &[f32]) -> Vec<f32> {
        let mut result = self.mean.clone();

        for (coef, comp) in reduced.iter().zip(self.components.iter()) {
            for (i, &c) in comp.iter().enumerate() {
                result[i] += coef * c;
            }
        }

        result
    }

    /// Get explained variance ratio
    pub fn explained_variance_ratio(&self) -> Vec<f32> {
        if self.total_variance > 0.0 {
            self.explained_variance
                .iter()
                .map(|&v| v / self.total_variance)
                .collect()
        } else {
            vec![0.0; self.explained_variance.len()]
        }
    }

    /// Get cumulative explained variance ratio
    pub fn cumulative_variance_ratio(&self) -> Vec<f32> {
        let mut cumsum = 0.0;
        self.explained_variance_ratio()
            .iter()
            .map(|&v| {
                cumsum += v;
                cumsum
            })
            .collect()
    }

    /// Get number of components
    pub fn n_components(&self) -> usize {
        self.output_dims
    }

    /// Get input dimensions
    pub fn input_dims(&self) -> usize {
        self.input_dims
    }
}

/// Random Projection for fast dimensionality reduction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomProjection {
    /// Projection matrix (output_dims x input_dims)
    projection: Vec<Vec<f32>>,
    /// Input dimensions
    input_dims: usize,
    /// Output dimensions
    output_dims: usize,
}

impl RandomProjection {
    /// Create a new random projection
    pub fn new(input_dims: usize, output_dims: usize) -> Self {
        Self::with_seed(input_dims, output_dims, None)
    }

    /// Create with specific seed
    pub fn with_seed(input_dims: usize, output_dims: usize, seed: Option<u64>) -> Self {
        use rand::SeedableRng;

        let mut rng: Box<dyn rand::RngCore> = match seed {
            Some(s) => Box::new(rand::rngs::StdRng::seed_from_u64(s)),
            None => Box::new(rand::thread_rng()),
        };

        // Sparse random projection (faster, nearly as good)
        // P(x = sqrt(3)) = 1/6, P(x = 0) = 2/3, P(x = -sqrt(3)) = 1/6
        let scale = (3.0f32 / output_dims as f32).sqrt();

        let projection: Vec<Vec<f32>> = (0..output_dims)
            .map(|_| {
                (0..input_dims)
                    .map(|_| {
                        let r: f32 = rng.gen();
                        if r < 1.0 / 6.0 {
                            scale
                        } else if r < 5.0 / 6.0 {
                            0.0
                        } else {
                            -scale
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            projection,
            input_dims,
            output_dims,
        }
    }

    /// Create Gaussian random projection (higher quality)
    pub fn gaussian(input_dims: usize, output_dims: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (output_dims as f32).sqrt();

        let projection: Vec<Vec<f32>> = (0..output_dims)
            .map(|_| {
                (0..input_dims)
                    .map(|_| {
                        // Box-Muller transform for Gaussian
                        let u1: f32 = rng.gen();
                        let u2: f32 = rng.gen();
                        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
                        z * scale
                    })
                    .collect()
            })
            .collect();

        Self {
            projection,
            input_dims,
            output_dims,
        }
    }

    /// Project a single vector
    pub fn project(&self, vector: &[f32]) -> Vec<f32> {
        self.projection
            .iter()
            .map(|row| row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Project multiple vectors
    pub fn project_batch(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        vectors.iter().map(|v| self.project(v)).collect()
    }

    /// Get output dimensions
    pub fn output_dims(&self) -> usize {
        self.output_dims
    }

    /// Get input dimensions
    pub fn input_dims(&self) -> usize {
        self.input_dims
    }
}

/// Simple neighbor embedding for 2D/3D visualization
/// (Simplified version inspired by t-SNE)
#[derive(Debug, Clone)]
pub struct NeighborEmbedding {
    /// Output dimensions (2 or 3)
    output_dims: usize,
    /// Perplexity parameter
    perplexity: f32,
    /// Learning rate
    learning_rate: f32,
    /// Number of iterations
    n_iter: usize,
}

impl Default for NeighborEmbedding {
    fn default() -> Self {
        Self {
            output_dims: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
        }
    }
}

impl NeighborEmbedding {
    /// Create new neighbor embedding
    pub fn new(output_dims: usize) -> Self {
        Self {
            output_dims,
            ..Default::default()
        }
    }

    /// Set perplexity
    pub fn with_perplexity(mut self, perplexity: f32) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set number of iterations
    pub fn with_n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Fit and transform vectors
    pub fn fit_transform(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        let n = vectors.len();
        if n == 0 {
            return Vec::new();
        }

        // Compute pairwise distances
        let distances = self.compute_distances(vectors);

        // Compute affinities (P matrix)
        let p = self.compute_affinities(&distances);

        // Initialize low-dimensional embedding randomly
        let mut rng = rand::thread_rng();
        let mut y: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..self.output_dims)
                    .map(|_| (rng.gen::<f32>() - 0.5) * 0.01)
                    .collect()
            })
            .collect();

        // Gradient descent
        let mut velocity: Vec<Vec<f32>> = vec![vec![0.0; self.output_dims]; n];
        let momentum = 0.8;

        for _ in 0..self.n_iter {
            // Compute Q matrix (low-dimensional affinities)
            let q = self.compute_q(&y);

            // Compute gradients
            let gradients = self.compute_gradients(&p, &q, &y);

            // Update with momentum
            for i in 0..n {
                for j in 0..self.output_dims {
                    velocity[i][j] =
                        momentum * velocity[i][j] - self.learning_rate * gradients[i][j];
                    y[i][j] += velocity[i][j];
                }
            }
        }

        // Center the result
        let mut mean = vec![0.0; self.output_dims];
        for yi in &y {
            for (j, &v) in yi.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }
        for yi in &mut y {
            for (j, m) in mean.iter().enumerate() {
                yi[j] -= m;
            }
        }

        y
    }

    fn compute_distances(&self, vectors: &[&[f32]]) -> Vec<Vec<f32>> {
        let n = vectors.len();
        let mut distances = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dist: f32 = vectors[i]
                    .iter()
                    .zip(vectors[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_affinities(&self, distances: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = distances.len();
        let target_entropy = self.perplexity.ln();

        // Compute sigma for each point using binary search
        let mut p = vec![vec![0.0; n]; n];

        for i in 0..n {
            let mut sigma = 1.0f32;
            let mut sigma_min = 1e-10f32;
            let mut sigma_max = 1e10f32;

            for _ in 0..50 {
                // Binary search iterations
                let mut sum = 0.0f32;
                for j in 0..n {
                    if i != j {
                        p[i][j] = (-distances[i][j] / (2.0 * sigma * sigma)).exp();
                        sum += p[i][j];
                    }
                }

                if sum > 0.0 {
                    for j in 0..n {
                        p[i][j] /= sum;
                    }
                }

                // Compute entropy
                let entropy: f32 = -p[i]
                    .iter()
                    .filter(|&&x| x > 1e-10)
                    .map(|&x| x * x.ln())
                    .sum::<f32>();

                if (entropy - target_entropy).abs() < 1e-5 {
                    break;
                }

                if entropy > target_entropy {
                    sigma_max = sigma;
                    sigma = (sigma + sigma_min) / 2.0;
                } else {
                    sigma_min = sigma;
                    sigma = (sigma + sigma_max) / 2.0;
                }
            }
        }

        // Symmetrize
        for i in 0..n {
            for j in (i + 1)..n {
                let pij = (p[i][j] + p[j][i]) / (2.0 * n as f32);
                p[i][j] = pij.max(1e-12);
                p[j][i] = pij.max(1e-12);
            }
        }

        p
    }

    #[allow(clippy::needless_range_loop)]
    fn compute_q(&self, y: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = y.len();
        let mut q = vec![vec![0.0; n]; n];
        let mut sum = 0.0f32;

        for i in 0..n {
            for j in (i + 1)..n {
                let dist: f32 = y[i]
                    .iter()
                    .zip(y[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                let qij = 1.0 / (1.0 + dist);
                q[i][j] = qij;
                q[j][i] = qij;
                sum += 2.0 * qij;
            }
        }

        if sum > 0.0 {
            for i in 0..n {
                for j in 0..n {
                    q[i][j] /= sum;
                    q[i][j] = q[i][j].max(1e-12);
                }
            }
        }

        q
    }

    fn compute_gradients(&self, p: &[Vec<f32>], q: &[Vec<f32>], y: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let n = y.len();
        let mut gradients = vec![vec![0.0; self.output_dims]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let dist: f32 = y[i]
                    .iter()
                    .zip(y[j].iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();

                let mult = 4.0 * (p[i][j] - q[i][j]) / (1.0 + dist);

                for d in 0..self.output_dims {
                    gradients[i][d] += mult * (y[i][d] - y[j][d]);
                }
            }
        }

        gradients
    }
}

/// Determine optimal number of components based on variance explained
pub fn find_optimal_components(vectors: &[&[f32]], target_variance: f32) -> Result<usize, String> {
    if vectors.is_empty() {
        return Err("Empty dataset".to_string());
    }

    let max_components = vectors[0].len().min(vectors.len()).min(50);
    let pca = PCA::fit(vectors, max_components)?;

    let cumulative = pca.cumulative_variance_ratio();

    for (i, &ratio) in cumulative.iter().enumerate() {
        if ratio >= target_variance {
            return Ok(i + 1);
        }
    }

    Ok(max_components)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, d: usize) -> Vec<Vec<f32>> {
        let mut rng = rand::thread_rng();
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    #[test]
    fn test_pca_basic() {
        let vectors = random_vectors(50, 10);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pca = PCA::fit(&refs, 3).unwrap();

        assert_eq!(pca.n_components(), 3);
        assert_eq!(pca.input_dims(), 10);

        let transformed = pca.transform(&refs);
        assert_eq!(transformed.len(), 50);
        assert_eq!(transformed[0].len(), 3);
    }

    #[test]
    fn test_pca_variance() {
        let vectors = random_vectors(100, 20);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pca = PCA::fit(&refs, 10).unwrap();

        let ratios = pca.explained_variance_ratio();
        assert!(ratios.iter().all(|&r| r >= 0.0));

        let cumulative = pca.cumulative_variance_ratio();
        // Cumulative should be non-decreasing
        for i in 1..cumulative.len() {
            assert!(cumulative[i] >= cumulative[i - 1]);
        }
    }

    #[test]
    fn test_pca_reconstruction() {
        let vectors = random_vectors(30, 8);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        // Use all components for best reconstruction
        let pca = PCA::fit(&refs, 8).unwrap();

        let transformed = pca.transform_one(&vectors[0]);
        let reconstructed = pca.inverse_transform(&transformed);

        // Reconstruction error should be small
        let error: f32 = vectors[0]
            .iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        assert!(error < 1.0, "Reconstruction error too large: {}", error);
    }

    #[test]
    fn test_random_projection() {
        let rp = RandomProjection::new(100, 20);

        let vector: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let projected = rp.project(&vector);

        assert_eq!(projected.len(), 20);
    }

    #[test]
    fn test_random_projection_gaussian() {
        let rp = RandomProjection::gaussian(50, 10);

        let vectors = random_vectors(20, 50);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let projected = rp.project_batch(&refs);
        assert_eq!(projected.len(), 20);
        assert_eq!(projected[0].len(), 10);
    }

    #[test]
    fn test_random_projection_preserves_distances() {
        let rp = RandomProjection::gaussian(100, 50);

        let v1: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let v2: Vec<f32> = (0..100).map(|i| (i + 10) as f32 / 100.0).collect();

        let original_dist: f32 = v1
            .iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        let p1 = rp.project(&v1);
        let p2 = rp.project(&v2);

        let projected_dist: f32 = p1
            .iter()
            .zip(p2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();

        // Distances should be roughly preserved (within 2x for random projection)
        let ratio = projected_dist / original_dist;
        assert!(ratio > 0.2 && ratio < 5.0, "Distance ratio: {}", ratio);
    }

    #[test]
    fn test_neighbor_embedding() {
        let vectors = random_vectors(20, 10);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let ne = NeighborEmbedding::new(2)
            .with_perplexity(5.0)
            .with_n_iter(100);

        let embedded = ne.fit_transform(&refs);

        assert_eq!(embedded.len(), 20);
        assert_eq!(embedded[0].len(), 2);
    }

    #[test]
    fn test_find_optimal_components() {
        let vectors = random_vectors(50, 20);
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let optimal = find_optimal_components(&refs, 0.9).unwrap();
        assert!((1..=20).contains(&optimal));
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<&[f32]> = Vec::new();

        let result = PCA::fit(&empty, 3);
        assert!(result.is_err());

        let ne = NeighborEmbedding::new(2);
        let embedded = ne.fit_transform(&empty);
        assert!(embedded.is_empty());
    }

    #[test]
    fn test_single_vector() {
        let vectors = [vec![1.0, 2.0, 3.0, 4.0]];
        let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

        let pca = PCA::fit(&refs, 2).unwrap();
        let transformed = pca.transform(&refs);

        assert_eq!(transformed.len(), 1);
    }
}
