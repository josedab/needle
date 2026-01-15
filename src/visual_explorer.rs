//! Feature 8: Interactive Visual Explorer
//!
//! Provides data structures and logic for collection browsing,
//! dimensionality reduction (t-SNE, UMAP, PCA), and query debugging.

use crate::collection::SearchResult;
use crate::error::{NeedleError, Result};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Projection types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f32,
    pub y: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Point3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ProjectionMethod {
    TSNE {
        perplexity: f32,
        learning_rate: f32,
        iterations: usize,
    },
    UMAP {
        n_neighbors: usize,
        min_dist: f32,
    },
    PCA,
    Random,
}

impl Default for ProjectionMethod {
    fn default() -> Self {
        ProjectionMethod::PCA
    }
}

impl ProjectionMethod {
    pub fn default_tsne() -> Self {
        ProjectionMethod::TSNE {
            perplexity: 30.0,
            learning_rate: 200.0,
            iterations: 1000,
        }
    }

    pub fn default_umap() -> Self {
        ProjectionMethod::UMAP {
            n_neighbors: 15,
            min_dist: 0.1,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectedPoint {
    pub id: String,
    pub position: Point2D,
    pub metadata: Option<serde_json::Value>,
    pub cluster_id: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectionResult {
    pub points: Vec<ProjectedPoint>,
    pub method: ProjectionMethod,
    pub input_dimensions: usize,
    pub point_count: usize,
    pub elapsed_ms: u64,
}

// ---------------------------------------------------------------------------
// VectorProjector
// ---------------------------------------------------------------------------

pub struct VectorProjector {
    method: ProjectionMethod,
}

impl VectorProjector {
    pub fn new(method: ProjectionMethod) -> Self {
        Self { method }
    }

    pub fn project(
        &self,
        vectors: &[(String, Vec<f32>, Option<serde_json::Value>)],
    ) -> Result<ProjectionResult> {
        if vectors.is_empty() {
            return Ok(ProjectionResult {
                points: Vec::new(),
                method: self.method.clone(),
                input_dimensions: 0,
                point_count: 0,
                elapsed_ms: 0,
            });
        }

        let start = std::time::Instant::now();
        let dim = vectors[0].1.len();

        let positions = match &self.method {
            ProjectionMethod::TSNE {
                perplexity,
                learning_rate,
                iterations,
            } => self.project_tsne(vectors, *perplexity, *learning_rate, *iterations)?,
            ProjectionMethod::UMAP {
                n_neighbors,
                min_dist,
            } => self.project_umap(vectors, *n_neighbors, *min_dist)?,
            ProjectionMethod::PCA => self.project_pca(vectors)?,
            ProjectionMethod::Random => self.project_random(vectors),
        };

        let points = vectors
            .iter()
            .zip(positions.into_iter())
            .map(|((id, _, meta), pos)| ProjectedPoint {
                id: id.clone(),
                position: pos,
                metadata: meta.clone(),
                cluster_id: None,
            })
            .collect::<Vec<_>>();

        let elapsed_ms = start.elapsed().as_millis() as u64;

        Ok(ProjectionResult {
            point_count: points.len(),
            points,
            method: self.method.clone(),
            input_dimensions: dim,
            elapsed_ms,
        })
    }

    // -- t-SNE (simplified Barnes-Hut-style) --------------------------------

    fn project_tsne(
        &self,
        vectors: &[(String, Vec<f32>, Option<serde_json::Value>)],
        perplexity: f32,
        learning_rate: f32,
        iterations: usize,
    ) -> Result<Vec<Point2D>> {
        let n = vectors.len();
        if n < 2 {
            return Ok(vectors
                .iter()
                .map(|_| Point2D { x: 0.0, y: 0.0 })
                .collect());
        }

        // Compute pairwise squared distances in high-dimensional space
        let mut dist2 = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                let d: f64 = vectors[i]
                    .1
                    .iter()
                    .zip(vectors[j].1.iter())
                    .map(|(a, b)| {
                        let diff = (*a as f64) - (*b as f64);
                        diff * diff
                    })
                    .sum();
                dist2[i][j] = d;
                dist2[j][i] = d;
            }
        }

        // Compute pairwise affinities using Gaussian kernel with fixed bandwidth
        let sigma = perplexity as f64;
        let sigma2 = 2.0 * sigma * sigma;
        let mut p = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            let mut row_sum = 0.0f64;
            for j in 0..n {
                if i != j {
                    p[i][j] = (-dist2[i][j] / sigma2).exp();
                    row_sum += p[i][j];
                }
            }
            if row_sum > 0.0 {
                for j in 0..n {
                    p[i][j] /= row_sum;
                }
            }
        }

        // Symmetrise: p_ij = (p_i|j + p_j|i) / (2n)
        let mut p_sym = vec![vec![0.0f64; n]; n];
        let two_n = (2 * n) as f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = (p[i][j] + p[j][i]) / two_n;
                p_sym[i][j] = v.max(1e-12);
                p_sym[j][i] = v.max(1e-12);
            }
        }

        // Initialise random 2D positions using simple deterministic seed
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let seed = i as f64;
            y.push([
                (seed * 0.7 + 0.3).sin() as f64,
                (seed * 1.1 + 0.7).cos() as f64,
            ]);
        }

        let iters = iterations.min(100); // cap for performance
        let lr = learning_rate as f64;

        for _t in 0..iters {
            // Compute low-dimensional affinities (Student-t, df=1)
            let mut q = vec![vec![0.0f64; n]; n];
            let mut q_sum = 0.0f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = y[i][0] - y[j][0];
                    let dy = y[i][1] - y[j][1];
                    let v = 1.0 / (1.0 + dx * dx + dy * dy);
                    q[i][j] = v;
                    q[j][i] = v;
                    q_sum += 2.0 * v;
                }
            }
            if q_sum > 0.0 {
                for i in 0..n {
                    for j in 0..n {
                        q[i][j] /= q_sum;
                        q[i][j] = q[i][j].max(1e-12);
                    }
                }
            }

            // Gradient
            let mut grad = vec![[0.0f64; 2]; n];
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let dx = y[i][0] - y[j][0];
                        let dy = y[i][1] - y[j][1];
                        let mult = 4.0 * (p_sym[i][j] - q[i][j])
                            / (1.0 + dx * dx + dy * dy);
                        grad[i][0] += mult * dx;
                        grad[i][1] += mult * dy;
                    }
                }
            }

            // Update positions
            for i in 0..n {
                y[i][0] -= lr * grad[i][0];
                y[i][1] -= lr * grad[i][1];
            }
        }

        Ok(y.iter()
            .map(|p| Point2D {
                x: p[0] as f32,
                y: p[1] as f32,
            })
            .collect())
    }

    // -- PCA ----------------------------------------------------------------

    fn project_pca(
        &self,
        vectors: &[(String, Vec<f32>, Option<serde_json::Value>)],
    ) -> Result<Vec<Point2D>> {
        let n = vectors.len();
        let dim = vectors[0].1.len();
        if n == 0 || dim == 0 {
            return Ok(Vec::new());
        }

        // Compute mean
        let mut mean = vec![0.0f64; dim];
        for (_, v, _) in vectors {
            for (i, val) in v.iter().enumerate() {
                mean[i] += *val as f64;
            }
        }
        let n_f = n as f64;
        for m in mean.iter_mut() {
            *m /= n_f;
        }

        // Center data
        let centered: Vec<Vec<f64>> = vectors
            .iter()
            .map(|(_, v, _)| {
                v.iter()
                    .enumerate()
                    .map(|(i, val)| *val as f64 - mean[i])
                    .collect()
            })
            .collect();

        // Power iteration for top 2 eigenvectors of the covariance matrix.
        // We work with X^T X implicitly via X (n x dim).
        let mut components = Vec::new();
        let mut deflated = centered.clone();

        for _ in 0..2 {
            let mut eigvec = vec![1.0f64; dim];
            // Normalise initial vector
            let norm: f64 = eigvec.iter().map(|v| v * v).sum::<f64>().sqrt();
            for v in eigvec.iter_mut() {
                *v /= norm;
            }

            for _ in 0..100 {
                // new = X^T (X * eigvec)
                let proj: Vec<f64> = deflated
                    .iter()
                    .map(|row| row.iter().zip(eigvec.iter()).map(|(a, b)| a * b).sum::<f64>())
                    .collect();

                let mut new_vec = vec![0.0f64; dim];
                for (row, &p) in deflated.iter().zip(proj.iter()) {
                    for (j, val) in row.iter().enumerate() {
                        new_vec[j] += val * p;
                    }
                }

                let norm: f64 = new_vec.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm < 1e-15 {
                    break;
                }
                for v in new_vec.iter_mut() {
                    *v /= norm;
                }
                eigvec = new_vec;
            }

            // Deflate: remove component from data
            for row in deflated.iter_mut() {
                let dot: f64 = row.iter().zip(eigvec.iter()).map(|(a, b)| a * b).sum();
                for (j, v) in eigvec.iter().enumerate() {
                    row[j] -= dot * v;
                }
            }

            components.push(eigvec);
        }

        // Project centered data onto the 2 components
        Ok(centered
            .iter()
            .map(|row| {
                let x: f64 = row
                    .iter()
                    .zip(components[0].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                let y: f64 = if components.len() > 1 {
                    row.iter()
                        .zip(components[1].iter())
                        .map(|(a, b)| a * b)
                        .sum()
                } else {
                    0.0
                };
                Point2D {
                    x: x as f32,
                    y: y as f32,
                }
            })
            .collect())
    }

    // -- Random -------------------------------------------------------------

    fn project_random(
        &self,
        vectors: &[(String, Vec<f32>, Option<serde_json::Value>)],
    ) -> Vec<Point2D> {
        vectors
            .iter()
            .map(|(id, v, _)| {
                // Simple hash-based deterministic random position
                let hash = Self::simple_hash(id, v);
                let x = ((hash & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
                let y = (((hash >> 16) & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0;
                Point2D { x, y }
            })
            .collect()
    }

    fn simple_hash(id: &str, v: &[f32]) -> u64 {
        let mut h: u64 = 14695981039346656037; // FNV offset basis
        for b in id.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        for val in v.iter() {
            let bits = val.to_bits() as u64;
            h ^= bits;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }

    // -- UMAP (simplified: kNN + force-directed layout) ---------------------

    fn project_umap(
        &self,
        vectors: &[(String, Vec<f32>, Option<serde_json::Value>)],
        n_neighbors: usize,
        min_dist: f32,
    ) -> Result<Vec<Point2D>> {
        let n = vectors.len();
        if n < 2 {
            return Ok(vectors
                .iter()
                .map(|_| Point2D { x: 0.0, y: 0.0 })
                .collect());
        }

        let k = n_neighbors.min(n - 1).max(1);

        // Build kNN graph
        let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
        for i in 0..n {
            let mut dists: Vec<(OrderedFloat<f32>, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d: f32 = vectors[i]
                        .1
                        .iter()
                        .zip(vectors[j].1.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum::<f32>()
                        .sqrt();
                    (OrderedFloat(d), j)
                })
                .collect();
            dists.sort();
            neighbors.push(dists.iter().take(k).map(|(_, j)| *j).collect());
        }

        // Initialise positions deterministically
        let mut y: Vec<[f64; 2]> = (0..n)
            .map(|i| {
                let seed = i as f64;
                [
                    (seed * 0.7 + 0.3).sin(),
                    (seed * 1.1 + 0.7).cos(),
                ]
            })
            .collect();

        // Force-directed iterations
        let min_d = min_dist as f64;
        for _t in 0..100 {
            let mut forces = vec![[0.0f64; 2]; n];

            // Attractive forces between neighbors
            for i in 0..n {
                for &j in &neighbors[i] {
                    let dx = y[j][0] - y[i][0];
                    let dy = y[j][1] - y[i][1];
                    let dist = (dx * dx + dy * dy).sqrt().max(1e-6);
                    let attraction = (dist - min_d).max(0.0) * 0.01;
                    forces[i][0] += attraction * dx / dist;
                    forces[i][1] += attraction * dy / dist;
                }
            }

            // Repulsive forces (sample-based for efficiency)
            for i in 0..n {
                for j in (i + 1)..n {
                    let dx = y[i][0] - y[j][0];
                    let dy = y[i][1] - y[j][1];
                    let dist2 = dx * dx + dy * dy + 1e-6;
                    let repulsion = 0.001 / dist2;
                    forces[i][0] += repulsion * dx;
                    forces[i][1] += repulsion * dy;
                    forces[j][0] -= repulsion * dx;
                    forces[j][1] -= repulsion * dy;
                }
            }

            for i in 0..n {
                y[i][0] += forces[i][0];
                y[i][1] += forces[i][1];
            }
        }

        Ok(y.iter()
            .map(|p| Point2D {
                x: p[0] as f32,
                y: p[1] as f32,
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Query debugging types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ScoreBreakdown {
    pub vector_score: f32,
    pub filter_score: Option<f32>,
    pub boost_score: Option<f32>,
    pub final_score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExplainedResult {
    pub id: String,
    pub distance: f32,
    pub rank: usize,
    pub score_breakdown: ScoreBreakdown,
    pub metadata_matched: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QueryExplanation {
    pub query_id: String,
    pub query_text: Option<String>,
    pub results: Vec<ExplainedResult>,
    pub total_candidates_evaluated: usize,
    pub index_layers_traversed: usize,
    pub filter_applied: bool,
    pub elapsed_us: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SearchComparison {
    pub common_results: usize,
    pub unique_to_a: usize,
    pub unique_to_b: usize,
    pub rank_correlation: f32,
    pub avg_distance_diff: f32,
}

// ---------------------------------------------------------------------------
// QueryDebugger
// ---------------------------------------------------------------------------

pub struct QueryDebugger;

impl QueryDebugger {
    pub fn new() -> Self {
        Self
    }

    pub fn explain_search(
        &self,
        query: &[f32],
        results: &[SearchResult],
        candidates_evaluated: usize,
    ) -> QueryExplanation {
        let start = std::time::Instant::now();

        let explained: Vec<ExplainedResult> = results
            .iter()
            .enumerate()
            .map(|(rank, sr)| {
                let score_breakdown = ScoreBreakdown {
                    vector_score: sr.distance,
                    filter_score: None,
                    boost_score: None,
                    final_score: sr.distance,
                };
                ExplainedResult {
                    id: sr.id.clone(),
                    distance: sr.distance,
                    rank: rank + 1,
                    score_breakdown,
                    metadata_matched: sr.metadata.is_some(),
                }
            })
            .collect();

        let elapsed_us = start.elapsed().as_micros() as u64;

        QueryExplanation {
            query_id: format!("q-{}", Self::hash_query(query)),
            query_text: None,
            results: explained,
            total_candidates_evaluated: candidates_evaluated,
            index_layers_traversed: 0,
            filter_applied: false,
            elapsed_us,
        }
    }

    pub fn compare_searches(
        &self,
        explanation_a: &QueryExplanation,
        explanation_b: &QueryExplanation,
    ) -> SearchComparison {
        let ids_a: HashMap<&str, (usize, f32)> = explanation_a
            .results
            .iter()
            .map(|r| (r.id.as_str(), (r.rank, r.distance)))
            .collect();
        let ids_b: HashMap<&str, (usize, f32)> = explanation_b
            .results
            .iter()
            .map(|r| (r.id.as_str(), (r.rank, r.distance)))
            .collect();

        let mut common = 0usize;
        let mut rank_pairs: Vec<(f64, f64)> = Vec::new();
        let mut dist_diffs: Vec<f32> = Vec::new();

        for (id, &(rank_a, dist_a)) in &ids_a {
            if let Some(&(rank_b, dist_b)) = ids_b.get(id) {
                common += 1;
                rank_pairs.push((rank_a as f64, rank_b as f64));
                dist_diffs.push((dist_a - dist_b).abs());
            }
        }

        let unique_to_a = ids_a.len() - common;
        let unique_to_b = ids_b.len() - common;

        let avg_distance_diff = if dist_diffs.is_empty() {
            0.0
        } else {
            dist_diffs.iter().sum::<f32>() / dist_diffs.len() as f32
        };

        // Spearman rank correlation (simplified)
        let rank_correlation = if rank_pairs.len() < 2 {
            0.0
        } else {
            let n = rank_pairs.len() as f64;
            let d_sq_sum: f64 = rank_pairs
                .iter()
                .map(|(a, b)| {
                    let d = a - b;
                    d * d
                })
                .sum();
            let rho = 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1.0));
            rho as f32
        };

        SearchComparison {
            common_results: common,
            unique_to_a,
            unique_to_b,
            rank_correlation,
            avg_distance_diff,
        }
    }

    fn hash_query(query: &[f32]) -> u64 {
        let mut h: u64 = 14695981039346656037;
        for val in query {
            let bits = val.to_bits() as u64;
            h ^= bits;
            h = h.wrapping_mul(1099511628211);
        }
        h
    }
}

// ---------------------------------------------------------------------------
// Collection browser
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrowseItem {
    pub id: String,
    pub metadata: Option<serde_json::Value>,
    pub preview: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BrowsePage {
    pub items: Vec<BrowseItem>,
    pub page: usize,
    pub total_pages: usize,
    pub total_items: usize,
}

pub struct CollectionBrowser {
    pub page_size: usize,
}

impl CollectionBrowser {
    pub fn new(page_size: usize) -> Self {
        Self { page_size }
    }

    pub fn browse(
        &self,
        vectors: &[(String, Option<serde_json::Value>)],
        page: usize,
    ) -> BrowsePage {
        let total_items = vectors.len();
        let page_size = self.page_size.max(1);
        let total_pages = if total_items == 0 {
            0
        } else {
            (total_items + page_size - 1) / page_size
        };

        let start = page * page_size;
        let items: Vec<BrowseItem> = if start >= total_items {
            Vec::new()
        } else {
            let end = (start + page_size).min(total_items);
            vectors[start..end]
                .iter()
                .map(|(id, meta)| {
                    let preview = match meta {
                        Some(v) => {
                            let s = v.to_string();
                            if s.len() > 80 {
                                format!("{}…", &s[..80])
                            } else {
                                s
                            }
                        }
                        None => String::from("(no metadata)"),
                    };
                    BrowseItem {
                        id: id.clone(),
                        metadata: meta.clone(),
                        preview,
                    }
                })
                .collect()
        };

        BrowsePage {
            items,
            page,
            total_pages,
            total_items,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_vectors(n: usize, dim: usize) -> Vec<(String, Vec<f32>, Option<serde_json::Value>)> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32 * 0.1).collect();
                let meta = json!({"index": i});
                (format!("vec-{i}"), v, Some(meta))
            })
            .collect()
    }

    fn sample_search_results(n: usize) -> Vec<SearchResult> {
        (0..n)
            .map(|i| SearchResult {
                id: format!("res-{i}"),
                distance: i as f32 * 0.5,
                metadata: Some(json!({"rank": i})),
            })
            .collect()
    }

    #[test]
    fn test_random_projection() {
        let vecs = sample_vectors(10, 8);
        let proj = VectorProjector::new(ProjectionMethod::Random);
        let result = proj.project(&vecs).unwrap();

        assert_eq!(result.point_count, 10);
        assert_eq!(result.input_dimensions, 8);
        // Deterministic: same input → same output
        let result2 = proj.project(&vecs).unwrap();
        for (a, b) in result.points.iter().zip(result2.points.iter()) {
            assert_eq!(a.position.x, b.position.x);
            assert_eq!(a.position.y, b.position.y);
        }
    }

    #[test]
    fn test_pca_projection() {
        let vecs = sample_vectors(20, 16);
        let proj = VectorProjector::new(ProjectionMethod::PCA);
        let result = proj.project(&vecs).unwrap();

        assert_eq!(result.point_count, 20);
        assert_eq!(result.input_dimensions, 16);
        // Points should not all be at the origin
        let non_zero = result
            .points
            .iter()
            .any(|p| p.position.x.abs() > 1e-6 || p.position.y.abs() > 1e-6);
        assert!(non_zero, "PCA should produce non-trivial positions");
    }

    #[test]
    fn test_tsne_projection() {
        let vecs = sample_vectors(10, 8);
        let proj = VectorProjector::new(ProjectionMethod::TSNE {
            perplexity: 5.0,
            learning_rate: 50.0,
            iterations: 50,
        });
        let result = proj.project(&vecs).unwrap();

        assert_eq!(result.point_count, 10);
        assert_eq!(result.input_dimensions, 8);
        // Should produce spread-out points
        let xs: Vec<f32> = result.points.iter().map(|p| p.position.x).collect();
        let range = xs.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
            - xs.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(range > 0.0, "t-SNE should spread points apart");
    }

    #[test]
    fn test_projection_preserves_ids() {
        let vecs = sample_vectors(5, 4);
        for method in [
            ProjectionMethod::PCA,
            ProjectionMethod::Random,
            ProjectionMethod::default_tsne(),
        ] {
            let proj = VectorProjector::new(method);
            let result = proj.project(&vecs).unwrap();
            let ids: Vec<&str> = result.points.iter().map(|p| p.id.as_str()).collect();
            assert_eq!(ids, vec!["vec-0", "vec-1", "vec-2", "vec-3", "vec-4"]);
        }
    }

    #[test]
    fn test_query_explanation() {
        let debugger = QueryDebugger::new();
        let query = vec![1.0f32, 2.0, 3.0];
        let results = sample_search_results(5);
        let explanation = debugger.explain_search(&query, &results, 100);

        assert_eq!(explanation.results.len(), 5);
        assert_eq!(explanation.total_candidates_evaluated, 100);
        assert!(!explanation.filter_applied);
        assert_eq!(explanation.results[0].rank, 1);
        assert_eq!(explanation.results[4].rank, 5);
        assert!(explanation.query_id.starts_with("q-"));
    }

    #[test]
    fn test_search_comparison() {
        let debugger = QueryDebugger::new();
        let query = vec![1.0f32, 2.0, 3.0];

        let results_a = sample_search_results(5);
        let mut results_b = sample_search_results(3);
        // Add a unique result to b
        results_b.push(SearchResult {
            id: "unique-b".to_string(),
            distance: 1.5,
            metadata: None,
        });

        let exp_a = debugger.explain_search(&query, &results_a, 50);
        let exp_b = debugger.explain_search(&query, &results_b, 30);
        let cmp = debugger.compare_searches(&exp_a, &exp_b);

        // res-0, res-1, res-2 are common
        assert_eq!(cmp.common_results, 3);
        assert_eq!(cmp.unique_to_a, 2); // res-3, res-4
        assert_eq!(cmp.unique_to_b, 1); // unique-b
    }

    #[test]
    fn test_collection_browser_pagination() {
        let data: Vec<(String, Option<serde_json::Value>)> = (0..25)
            .map(|i| (format!("item-{i}"), Some(json!({"i": i}))))
            .collect();

        let browser = CollectionBrowser::new(10);
        let page0 = browser.browse(&data, 0);
        assert_eq!(page0.items.len(), 10);
        assert_eq!(page0.total_pages, 3);
        assert_eq!(page0.total_items, 25);
        assert_eq!(page0.page, 0);

        let page2 = browser.browse(&data, 2);
        assert_eq!(page2.items.len(), 5);

        // Out of range page
        let page_oob = browser.browse(&data, 10);
        assert!(page_oob.items.is_empty());
    }

    #[test]
    fn test_browse_empty() {
        let browser = CollectionBrowser::new(10);
        let page = browser.browse(&[], 0);
        assert!(page.items.is_empty());
        assert_eq!(page.total_pages, 0);
        assert_eq!(page.total_items, 0);
    }

    #[test]
    fn test_score_breakdown() {
        let sb = ScoreBreakdown {
            vector_score: 0.85,
            filter_score: Some(1.0),
            boost_score: Some(0.1),
            final_score: 0.95,
        };
        assert_eq!(sb.vector_score, 0.85);
        assert_eq!(sb.filter_score, Some(1.0));
        assert_eq!(sb.boost_score, Some(0.1));
        assert_eq!(sb.final_score, 0.95);

        // Serialisation round-trip
        let json = serde_json::to_string(&sb).unwrap();
        let deser: ScoreBreakdown = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.final_score, 0.95);
    }

    #[test]
    fn test_config_defaults() {
        match ProjectionMethod::default_tsne() {
            ProjectionMethod::TSNE {
                perplexity,
                learning_rate,
                iterations,
            } => {
                assert_eq!(perplexity, 30.0);
                assert_eq!(learning_rate, 200.0);
                assert_eq!(iterations, 1000);
            }
            _ => panic!("Expected TSNE"),
        }

        match ProjectionMethod::default_umap() {
            ProjectionMethod::UMAP {
                n_neighbors,
                min_dist,
            } => {
                assert_eq!(n_neighbors, 15);
                assert!((min_dist - 0.1).abs() < f32::EPSILON);
            }
            _ => panic!("Expected UMAP"),
        }

        assert!(matches!(ProjectionMethod::default(), ProjectionMethod::PCA));
    }

    #[test]
    fn test_umap_projection() {
        let vecs = sample_vectors(10, 8);
        let proj = VectorProjector::new(ProjectionMethod::UMAP {
            n_neighbors: 3,
            min_dist: 0.1,
        });
        let result = proj.project(&vecs).unwrap();
        assert_eq!(result.point_count, 10);
    }
}
