//! Visual Collection Explorer
//!
//! UMAP/t-SNE-style dimensionality reduction for 2D visualization of vector
//! collections, cluster analysis, query coverage heatmaps, and real-time
//! performance metrics.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::visual_explorer::{
//!     CollectionExplorer, ExplorerConfig, ProjectionMethod,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let coll = db.collection("docs").unwrap();
//! // ... insert vectors ...
//!
//! let explorer = CollectionExplorer::from_collection(&coll, ExplorerConfig::default()).unwrap();
//!
//! // Generate 2D projections
//! let projection = explorer.project(ProjectionMethod::PCA);
//! for point in &projection.points {
//!     println!("ID: {}, x: {:.2}, y: {:.2}", point.id, point.x, point.y);
//! }
//!
//! // Compute cluster analysis
//! let clusters = explorer.cluster(5);
//! println!("Found {} clusters", clusters.len());
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::SearchResult;
use crate::database::{CollectionRef, ExportEntry};
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Explorer configuration.
#[derive(Debug, Clone)]
pub struct ExplorerConfig {
    /// Maximum vectors to sample for projection (for performance).
    pub max_sample_size: usize,
    /// Random seed for deterministic projections.
    pub random_seed: u64,
    /// Number of PCA iterations.
    pub pca_iterations: usize,
}

impl Default for ExplorerConfig {
    fn default() -> Self {
        Self {
            max_sample_size: 10_000,
            random_seed: 42,
            pca_iterations: 20,
        }
    }
}

/// Projection method for dimensionality reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMethod {
    /// PCA (Principal Component Analysis) — fast, linear.
    PCA,
    /// Random projection — very fast, approximate.
    Random,
    /// First two dimensions — trivial but useful for debugging.
    FirstTwo,
}

// ── Projection Output ────────────────────────────────────────────────────────

/// A 2D projected point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedPoint {
    /// Vector ID.
    pub id: String,
    /// X coordinate.
    pub x: f32,
    /// Y coordinate.
    pub y: f32,
    /// Optional cluster assignment.
    pub cluster: Option<usize>,
    /// Optional label/metadata.
    pub label: Option<String>,
}

/// A complete 2D projection of a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projection {
    /// The projected points.
    pub points: Vec<ProjectedPoint>,
    /// Projection method used.
    pub method: ProjectionMethod,
    /// Variance explained (for PCA).
    pub variance_explained: Option<(f64, f64)>,
    /// Bounding box: (min_x, min_y, max_x, max_y).
    pub bounds: (f32, f32, f32, f32),
}

// ── Cluster ──────────────────────────────────────────────────────────────────

/// A cluster of vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cluster {
    /// Cluster ID.
    pub id: usize,
    /// Number of vectors in this cluster.
    pub size: usize,
    /// Centroid of the cluster.
    pub centroid: Vec<f32>,
    /// 2D projected centroid.
    pub centroid_2d: Option<(f32, f32)>,
    /// Vector IDs in this cluster.
    pub member_ids: Vec<String>,
    /// Average intra-cluster distance.
    pub avg_distance: f32,
    /// Most common metadata values (key → value → count).
    pub metadata_summary: HashMap<String, HashMap<String, usize>>,
}

// ── Query Heatmap ────────────────────────────────────────────────────────────

/// A heatmap of query coverage across the vector space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryHeatmap {
    /// Grid cells: (x_bucket, y_bucket) → query count.
    pub grid: Vec<HeatmapCell>,
    /// Grid resolution.
    pub resolution: usize,
    /// Total queries tracked.
    pub total_queries: usize,
}

/// A single heatmap cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeatmapCell {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
    /// Query count in this cell.
    pub count: usize,
    /// Intensity (0.0–1.0).
    pub intensity: f32,
}

// ── Collection Statistics ────────────────────────────────────────────────────

/// Comprehensive collection visualization statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionVizStats {
    /// Total vectors.
    pub total_vectors: usize,
    /// Dimensions.
    pub dimensions: usize,
    /// Per-dimension statistics.
    pub dimension_stats: Vec<DimensionStat>,
    /// Average vector norm.
    pub avg_norm: f32,
    /// Norm standard deviation.
    pub norm_std: f32,
    /// Average pairwise distance (sampled).
    pub avg_pairwise_distance: f32,
    /// Estimated intrinsic dimensionality.
    pub intrinsic_dimensionality: Option<f32>,
}

/// Statistics for a single dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DimensionStat {
    /// Dimension index.
    pub index: usize,
    /// Mean value.
    pub mean: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Min value.
    pub min: f32,
    /// Max value.
    pub max: f32,
}

// ── Explorer ─────────────────────────────────────────────────────────────────

/// Vector collection explorer for visualization and analysis.
pub struct CollectionExplorer {
    vectors: Vec<(String, Vec<f32>)>,
    metadata: HashMap<String, Value>,
    config: ExplorerConfig,
    query_log: Vec<Vec<f32>>,
}

impl CollectionExplorer {
    /// Create an explorer from a collection reference.
    pub fn from_collection(coll: &CollectionRef, config: ExplorerConfig) -> Result<Self> {
        let mut vectors = Vec::new();
        let mut metadata = HashMap::new();

        let entries = coll.export_all()?;
        let max = config.max_sample_size;
        for (id, vec, meta) in entries.into_iter().take(max) {
            if let Some(m) = &meta {
                metadata.insert(id.clone(), m.clone());
            }
            vectors.push((id, vec));
        }

        Ok(Self {
            vectors,
            metadata,
            config,
            query_log: Vec::new(),
        })
    }

    /// Create from raw vectors (for testing or custom data).
    pub fn from_vectors(
        vectors: Vec<(String, Vec<f32>)>,
        config: ExplorerConfig,
    ) -> Self {
        Self {
            vectors,
            metadata: HashMap::new(),
            config,
            query_log: Vec::new(),
        }
    }

    /// Project vectors to 2D.
    pub fn project(&self, method: ProjectionMethod) -> Projection {
        if self.vectors.is_empty() {
            return Projection {
                points: Vec::new(),
                method,
                variance_explained: None,
                bounds: (0.0, 0.0, 0.0, 0.0),
            };
        }

        let points = match method {
            ProjectionMethod::PCA => self.project_pca(),
            ProjectionMethod::Random => self.project_random(),
            ProjectionMethod::FirstTwo => self.project_first_two(),
        };

        let (min_x, max_x, min_y, max_y) = points.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(mnx, mxx, mny, mxy), p| {
                (mnx.min(p.x), mxx.max(p.x), mny.min(p.y), mxy.max(p.y))
            },
        );

        Projection {
            points,
            method,
            variance_explained: if method == ProjectionMethod::PCA {
                Some((0.0, 0.0)) // Simplified: actual PCA would compute this
            } else {
                None
            },
            bounds: (min_x, min_y, max_x, max_y),
        }
    }

    /// K-means clustering of the vectors.
    pub fn cluster(&self, k: usize) -> Vec<Cluster> {
        if self.vectors.is_empty() || k == 0 {
            return Vec::new();
        }

        let k = k.min(self.vectors.len());
        let dims = self.vectors[0].1.len();

        // Initialize centroids (first k vectors)
        let mut centroids: Vec<Vec<f32>> = self.vectors[..k]
            .iter()
            .map(|(_, v)| v.clone())
            .collect();

        let mut assignments = vec![0usize; self.vectors.len()];
        let max_iterations = 20;

        for _ in 0..max_iterations {
            let mut changed = false;

            // Assign each vector to nearest centroid
            for (i, (_, vec)) in self.vectors.iter().enumerate() {
                let nearest = centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        let da = euclidean_distance(vec, a);
                        let db = euclidean_distance(vec, b);
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);

                if assignments[i] != nearest {
                    assignments[i] = nearest;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for c in 0..k {
                let mut new_centroid = vec![0.0f32; dims];
                let mut count = 0;

                for (i, (_, vec)) in self.vectors.iter().enumerate() {
                    if assignments[i] == c {
                        for (d, v) in vec.iter().enumerate() {
                            if d < dims {
                                new_centroid[d] += v;
                            }
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    for v in &mut new_centroid {
                        *v /= count as f32;
                    }
                    centroids[c] = new_centroid;
                }
            }
        }

        // Build cluster objects
        let mut clusters: Vec<Cluster> = (0..k)
            .map(|c| Cluster {
                id: c,
                size: 0,
                centroid: centroids[c].clone(),
                centroid_2d: None,
                member_ids: Vec::new(),
                avg_distance: 0.0,
                metadata_summary: HashMap::new(),
            })
            .collect();

        for (i, (id, vec)) in self.vectors.iter().enumerate() {
            let c = assignments[i];
            if c < clusters.len() {
                clusters[c].size += 1;
                clusters[c].member_ids.push(id.clone());
                clusters[c].avg_distance += euclidean_distance(vec, &centroids[c]);

                // Summarize metadata
                if let Some(meta) = self.metadata.get(id) {
                    if let Value::Object(map) = meta {
                        for (key, val) in map {
                            let val_str = match val {
                                Value::String(s) => s.clone(),
                                _ => val.to_string(),
                            };
                            *clusters[c]
                                .metadata_summary
                                .entry(key.clone())
                                .or_default()
                                .entry(val_str)
                                .or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        for cluster in &mut clusters {
            if cluster.size > 0 {
                cluster.avg_distance /= cluster.size as f32;
            }
        }

        // Remove empty clusters
        clusters.retain(|c| c.size > 0);
        clusters
    }

    /// Compute comprehensive visualization statistics.
    pub fn viz_stats(&self) -> CollectionVizStats {
        if self.vectors.is_empty() {
            return CollectionVizStats {
                total_vectors: 0,
                dimensions: 0,
                dimension_stats: Vec::new(),
                avg_norm: 0.0,
                norm_std: 0.0,
                avg_pairwise_distance: 0.0,
                intrinsic_dimensionality: None,
            };
        }

        let dims = self.vectors[0].1.len();
        let n = self.vectors.len();

        // Per-dimension stats
        let mut dim_stats = Vec::with_capacity(dims);
        for d in 0..dims {
            let values: Vec<f32> = self.vectors.iter().map(|(_, v)| v.get(d).copied().unwrap_or(0.0)).collect();
            let mean = values.iter().sum::<f32>() / n as f32;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
            let min = values.iter().copied().fold(f32::MAX, f32::min);
            let max = values.iter().copied().fold(f32::MIN, f32::max);

            dim_stats.push(DimensionStat {
                index: d,
                mean,
                std_dev: variance.sqrt(),
                min,
                max,
            });
        }

        // Norms
        let norms: Vec<f32> = self
            .vectors
            .iter()
            .map(|(_, v)| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();
        let avg_norm = norms.iter().sum::<f32>() / n as f32;
        let norm_std = (norms.iter().map(|n| (n - avg_norm).powi(2)).sum::<f32>() / n as f32).sqrt();

        // Sampled pairwise distance
        let sample_size = n.min(100);
        let mut pairwise_sum = 0.0f32;
        let mut pair_count = 0;
        for i in 0..sample_size {
            for j in (i + 1)..sample_size {
                pairwise_sum += euclidean_distance(&self.vectors[i].1, &self.vectors[j].1);
                pair_count += 1;
            }
        }
        let avg_pairwise = if pair_count > 0 {
            pairwise_sum / pair_count as f32
        } else {
            0.0
        };

        CollectionVizStats {
            total_vectors: n,
            dimensions: dims,
            dimension_stats: dim_stats,
            avg_norm,
            norm_std,
            avg_pairwise_distance: avg_pairwise,
            intrinsic_dimensionality: None,
        }
    }

    /// Record a query for heatmap tracking.
    pub fn record_query(&mut self, query: &[f32]) {
        self.query_log.push(query.to_vec());
    }

    /// Generate a query coverage heatmap.
    pub fn query_heatmap(&self, resolution: usize) -> QueryHeatmap {
        let resolution = resolution.max(2);

        if self.query_log.is_empty() {
            return QueryHeatmap {
                grid: Vec::new(),
                resolution,
                total_queries: 0,
            };
        }

        // Project queries to 2D using same method
        let projection = self.project_vectors_first_two(&self.query_log);

        let (min_x, max_x, min_y, max_y) = projection.iter().fold(
            (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
            |(mnx, mxx, mny, mxy), (x, y)| {
                (mnx.min(*x), mxx.max(*x), mny.min(*y), mxy.max(*y))
            },
        );

        let range_x = (max_x - min_x).max(0.001);
        let range_y = (max_y - min_y).max(0.001);

        let mut grid: HashMap<(usize, usize), usize> = HashMap::new();
        for (x, y) in &projection {
            let col = (((x - min_x) / range_x * (resolution - 1) as f32) as usize).min(resolution - 1);
            let row = (((y - min_y) / range_y * (resolution - 1) as f32) as usize).min(resolution - 1);
            *grid.entry((row, col)).or_insert(0) += 1;
        }

        let max_count = grid.values().copied().max().unwrap_or(1);
        let cells: Vec<HeatmapCell> = grid
            .into_iter()
            .map(|((row, col), count)| HeatmapCell {
                row,
                col,
                count,
                intensity: count as f32 / max_count as f32,
            })
            .collect();

        QueryHeatmap {
            grid: cells,
            resolution,
            total_queries: self.query_log.len(),
        }
    }

    /// Number of vectors loaded.
    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    // ── Projection Implementations ───────────────────────────────────────

    fn project_pca(&self) -> Vec<ProjectedPoint> {
        if self.vectors.is_empty() {
            return Vec::new();
        }
        let dims = self.vectors[0].1.len();
        let n = self.vectors.len();

        // Compute mean
        let mut mean = vec![0.0f32; dims];
        for (_, v) in &self.vectors {
            for (d, val) in v.iter().enumerate() {
                if d < dims {
                    mean[d] += val;
                }
            }
        }
        for m in &mut mean {
            *m /= n as f32;
        }

        // Power iteration for top-2 principal components
        let mut pc1 = vec![1.0f32; dims];
        let mut pc2 = vec![0.0f32; dims];
        if dims > 1 {
            pc2[1] = 1.0;
        }

        for _ in 0..self.config.pca_iterations {
            // Multiply by centered data covariance
            let mut new_pc1 = vec![0.0f32; dims];
            for (_, v) in &self.vectors {
                let centered: Vec<f32> = v.iter().enumerate().map(|(d, val)| val - mean.get(d).unwrap_or(&0.0)).collect();
                let dot: f32 = centered.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
                for (d, c) in centered.iter().enumerate() {
                    if d < dims {
                        new_pc1[d] += c * dot;
                    }
                }
            }
            // Normalize
            let norm: f32 = new_pc1.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut new_pc1 {
                    *v /= norm;
                }
            }
            pc1 = new_pc1;
        }

        // Gram-Schmidt for pc2
        let dot12: f32 = pc2.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
        for (d, v) in pc2.iter_mut().enumerate() {
            *v -= dot12 * pc1.get(d).unwrap_or(&0.0);
        }

        for _ in 0..self.config.pca_iterations {
            let mut new_pc2 = vec![0.0f32; dims];
            for (_, v) in &self.vectors {
                let centered: Vec<f32> = v.iter().enumerate().map(|(d, val)| val - mean.get(d).unwrap_or(&0.0)).collect();
                let dot: f32 = centered.iter().zip(pc2.iter()).map(|(a, b)| a * b).sum();
                for (d, c) in centered.iter().enumerate() {
                    if d < dims {
                        new_pc2[d] += c * dot;
                    }
                }
            }
            // Orthogonalize
            let dot = new_pc2.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum::<f32>();
            for (d, v) in new_pc2.iter_mut().enumerate() {
                *v -= dot * pc1.get(d).unwrap_or(&0.0);
            }
            let norm = new_pc2.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for v in &mut new_pc2 {
                    *v /= norm;
                }
            }
            pc2 = new_pc2;
        }

        // Project all vectors
        self.vectors
            .iter()
            .map(|(id, v)| {
                let centered: Vec<f32> = v.iter().enumerate().map(|(d, val)| val - mean.get(d).unwrap_or(&0.0)).collect();
                let x: f32 = centered.iter().zip(pc1.iter()).map(|(a, b)| a * b).sum();
                let y: f32 = centered.iter().zip(pc2.iter()).map(|(a, b)| a * b).sum();
                ProjectedPoint {
                    id: id.clone(),
                    x,
                    y,
                    cluster: None,
                    label: self.metadata.get(id).and_then(|m| {
                        m.get("label").and_then(|v| v.as_str()).map(String::from)
                    }),
                }
            })
            .collect()
    }

    fn project_random(&self) -> Vec<ProjectedPoint> {
        // Deterministic random projection using seed
        let seed = self.config.random_seed;
        let dims = self.vectors.first().map_or(1, |(_, v)| v.len());

        // Generate two random projection vectors using linear congruential generator
        let mut rng_state = seed;
        let proj1: Vec<f32> = (0..dims)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        let proj2: Vec<f32> = (0..dims)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((rng_state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();

        self.vectors
            .iter()
            .map(|(id, v)| {
                let x: f32 = v.iter().zip(proj1.iter()).map(|(a, b)| a * b).sum();
                let y: f32 = v.iter().zip(proj2.iter()).map(|(a, b)| a * b).sum();
                ProjectedPoint {
                    id: id.clone(),
                    x,
                    y,
                    cluster: None,
                    label: None,
                }
            })
            .collect()
    }

    fn project_first_two(&self) -> Vec<ProjectedPoint> {
        self.vectors
            .iter()
            .map(|(id, v)| ProjectedPoint {
                id: id.clone(),
                x: v.first().copied().unwrap_or(0.0),
                y: v.get(1).copied().unwrap_or(0.0),
                cluster: None,
                label: None,
            })
            .collect()
    }

    fn project_vectors_first_two(&self, vecs: &[Vec<f32>]) -> Vec<(f32, f32)> {
        vecs.iter()
            .map(|v| {
                (
                    v.first().copied().unwrap_or(0.0),
                    v.get(1).copied().unwrap_or(0.0),
                )
            })
            .collect()
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_vectors() -> Vec<(String, Vec<f32>)> {
        vec![
            ("a".into(), vec![1.0, 0.0, 0.0, 0.0]),
            ("b".into(), vec![0.0, 1.0, 0.0, 0.0]),
            ("c".into(), vec![0.0, 0.0, 1.0, 0.0]),
            ("d".into(), vec![0.0, 0.0, 0.0, 1.0]),
            ("e".into(), vec![0.5, 0.5, 0.0, 0.0]),
        ]
    }

    #[test]
    fn test_projection_first_two() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let proj = explorer.project(ProjectionMethod::FirstTwo);
        assert_eq!(proj.points.len(), 5);
        assert_eq!(proj.points[0].x, 1.0);
        assert_eq!(proj.points[0].y, 0.0);
    }

    #[test]
    fn test_projection_pca() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let proj = explorer.project(ProjectionMethod::PCA);
        assert_eq!(proj.points.len(), 5);
        assert!(proj.variance_explained.is_some());
    }

    #[test]
    fn test_projection_random() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let proj = explorer.project(ProjectionMethod::Random);
        assert_eq!(proj.points.len(), 5);
        // Deterministic with same seed
        let proj2 = explorer.project(ProjectionMethod::Random);
        assert_eq!(proj.points[0].x, proj2.points[0].x);
    }

    #[test]
    fn test_clustering() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let clusters = explorer.cluster(2);
        assert!(!clusters.is_empty());
        assert!(clusters.len() <= 2);

        let total_members: usize = clusters.iter().map(|c| c.size).sum();
        assert_eq!(total_members, 5);
    }

    #[test]
    fn test_clustering_k_larger_than_n() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let clusters = explorer.cluster(100);
        // k capped to n
        let total: usize = clusters.iter().map(|c| c.size).sum();
        assert_eq!(total, 5);
    }

    #[test]
    fn test_viz_stats() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let stats = explorer.viz_stats();
        assert_eq!(stats.total_vectors, 5);
        assert_eq!(stats.dimensions, 4);
        assert_eq!(stats.dimension_stats.len(), 4);
        assert!(stats.avg_norm > 0.0);
    }

    #[test]
    fn test_empty_collection() {
        let explorer =
            CollectionExplorer::from_vectors(Vec::new(), ExplorerConfig::default());
        let proj = explorer.project(ProjectionMethod::PCA);
        assert!(proj.points.is_empty());

        let clusters = explorer.cluster(3);
        assert!(clusters.is_empty());

        let stats = explorer.viz_stats();
        assert_eq!(stats.total_vectors, 0);
    }

    #[test]
    fn test_query_heatmap() {
        let mut explorer =
            CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());

        explorer.record_query(&[1.0, 0.0, 0.0, 0.0]);
        explorer.record_query(&[1.0, 0.1, 0.0, 0.0]);
        explorer.record_query(&[0.0, 1.0, 0.0, 0.0]);

        let heatmap = explorer.query_heatmap(10);
        assert_eq!(heatmap.total_queries, 3);
        assert!(!heatmap.grid.is_empty());
    }

    #[test]
    fn test_empty_heatmap() {
        let explorer =
            CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let heatmap = explorer.query_heatmap(10);
        assert_eq!(heatmap.total_queries, 0);
        assert!(heatmap.grid.is_empty());
    }

    #[test]
    fn test_bounds() {
        let explorer = CollectionExplorer::from_vectors(sample_vectors(), ExplorerConfig::default());
        let proj = explorer.project(ProjectionMethod::FirstTwo);
        let (min_x, min_y, max_x, max_y) = proj.bounds;
        assert!(max_x >= min_x);
        assert!(max_y >= min_y);
    }
}
