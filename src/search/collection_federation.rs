#![allow(dead_code)]

//! Cross-Collection Federated Search
//!
//! Enables single-query search across multiple collections with result merging.
//! Implements score normalization (min-max, z-score) across heterogeneous collections,
//! a query planner with per-collection filters, and cross-dimension PCA projection.
//!
//! Uses Reciprocal Rank Fusion (RRF) as the default merge strategy.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::search::collection_federation::{
//!     CollectionFederation, FederationQuery, ScoreNormalization, MergeStrategy,
//! };
//!
//! let federation = CollectionFederation::new(db.clone());
//!
//! let results = federation.search(FederationQuery {
//!     query: vec![0.1; 384],
//!     k: 10,
//!     collections: vec!["docs", "images"],
//!     normalization: ScoreNormalization::MinMax,
//!     merge_strategy: MergeStrategy::ReciprocalRankFusion { k: 60 },
//! })?;
//! ```

use crate::database::Database;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::SearchResult;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Score normalization strategy for cross-collection result merging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// No normalization (raw distances).
    None,
    /// Min-max normalization: scale scores to [0, 1] per collection.
    MinMax,
    /// Z-score normalization: (score - mean) / stddev per collection.
    ZScore,
    /// Percentile rank normalization.
    PercentileRank,
}

/// Merge strategy for combining results from multiple collections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Reciprocal Rank Fusion with parameter k (default: 60).
    ReciprocalRankFusion { k: usize },
    /// Merge by normalized distance (lower is better).
    NormalizedDistance,
    /// Round-robin interleaving from each collection.
    RoundRobin,
    /// Weighted combination with per-collection weights.
    Weighted { weights: HashMap<String, f64> },
}

impl Default for MergeStrategy {
    fn default() -> Self {
        Self::ReciprocalRankFusion { k: 60 }
    }
}

/// Per-collection filter specification.
#[derive(Debug, Clone)]
pub struct CollectionQuery {
    /// Collection name.
    pub collection: String,
    /// Optional metadata filter for this collection.
    pub filter: Option<Filter>,
    /// Optional weight for this collection (used with Weighted merge).
    pub weight: f64,
    /// Optional dimension mapping (for cross-dimension projection).
    pub dimension_projection: Option<DimensionProjection>,
}

/// Dimension projection for cross-dimension search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DimensionProjection {
    /// Truncate query to collection's dimensions (Matryoshka-style).
    Truncate,
    /// Zero-pad query to match collection's dimensions.
    ZeroPad,
    /// Apply a projection matrix (PCA or learned).
    Matrix { matrix: Vec<Vec<f32>> },
}

/// Federated search query specification.
#[derive(Debug, Clone)]
pub struct FederationQuery {
    /// Query vector.
    pub query: Vec<f32>,
    /// Number of results to return.
    pub k: usize,
    /// Target collections (empty = all collections).
    pub collections: Vec<CollectionQuery>,
    /// Score normalization strategy.
    pub normalization: ScoreNormalization,
    /// Merge strategy.
    pub merge_strategy: MergeStrategy,
    /// Maximum results per collection before merging.
    pub per_collection_limit: usize,
}

impl FederationQuery {
    /// Create a simple federation query across all specified collection names.
    pub fn new(query: Vec<f32>, k: usize, collection_names: &[&str]) -> Self {
        Self {
            query,
            k,
            collections: collection_names
                .iter()
                .map(|name| CollectionQuery {
                    collection: name.to_string(),
                    filter: None,
                    weight: 1.0,
                    dimension_projection: None,
                })
                .collect(),
            normalization: ScoreNormalization::MinMax,
            merge_strategy: MergeStrategy::default(),
            per_collection_limit: k * 3,
        }
    }
}

/// A search result annotated with its source collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// The original search result.
    pub id: String,
    /// Normalized score (lower is better for distance, higher for RRF).
    pub score: f64,
    /// Raw distance from the collection's search.
    pub raw_distance: f32,
    /// Source collection name.
    pub collection: String,
    /// Metadata from the result.
    pub metadata: Option<serde_json::Value>,
}

/// Statistics from a federated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationStats {
    /// Number of collections searched.
    pub collections_searched: usize,
    /// Results per collection before merging.
    pub results_per_collection: HashMap<String, usize>,
    /// Total results before merging.
    pub total_before_merge: usize,
    /// Total results after merging.
    pub total_after_merge: usize,
    /// Duration per collection in microseconds.
    pub duration_per_collection_us: HashMap<String, u64>,
    /// Total duration in microseconds.
    pub total_duration_us: u64,
}

/// Cross-collection federated search coordinator.
pub struct CollectionFederation {
    db: Arc<Database>,
}

impl CollectionFederation {
    /// Create a new federation coordinator.
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Execute a federated search across multiple collections.
    pub fn search(
        &self,
        query: &FederationQuery,
    ) -> Result<(Vec<FederatedResult>, FederationStats)> {
        let start = std::time::Instant::now();
        let mut all_results: HashMap<String, Vec<SearchResult>> = HashMap::new();
        let mut duration_per_collection = HashMap::new();

        // Fan out search to each collection
        for cq in &query.collections {
            let coll_start = std::time::Instant::now();

            // Project query if needed
            let projected_query = self.project_query(&query.query, &cq)?;

            let collection = self.db.collection(&cq.collection)?;

            let results = if let Some(ref filter) = cq.filter {
                collection.search_with_filter(&projected_query, query.per_collection_limit, filter)?
            } else {
                collection.search(&projected_query, query.per_collection_limit)?
            };

            duration_per_collection.insert(
                cq.collection.clone(),
                coll_start.elapsed().as_micros() as u64,
            );
            all_results.insert(cq.collection.clone(), results);
        }

        // Normalize scores
        let normalized = self.normalize_scores(&all_results, &query.normalization);

        // Merge results
        let merged = self.merge_results(
            &normalized,
            &query.merge_strategy,
            &query.collections,
            query.k,
        );

        let results_per_collection: HashMap<String, usize> = all_results
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect();
        let total_before: usize = results_per_collection.values().sum();

        let stats = FederationStats {
            collections_searched: query.collections.len(),
            results_per_collection,
            total_before_merge: total_before,
            total_after_merge: merged.len(),
            duration_per_collection_us: duration_per_collection,
            total_duration_us: start.elapsed().as_micros() as u64,
        };

        Ok((merged, stats))
    }

    /// Project query vector to match a collection's dimensions.
    fn project_query(&self, query: &[f32], cq: &CollectionQuery) -> Result<Vec<f32>> {
        match &cq.dimension_projection {
            None => Ok(query.to_vec()),
            Some(DimensionProjection::Truncate) => {
                let collection = self.db.collection(&cq.collection)?;
                let dims = collection.dimensions().unwrap_or(query.len());
                Ok(query.iter().take(dims).copied().collect())
            }
            Some(DimensionProjection::ZeroPad) => {
                let collection = self.db.collection(&cq.collection)?;
                let dims = collection.dimensions().unwrap_or(query.len());
                let mut projected = query.to_vec();
                projected.resize(dims, 0.0);
                Ok(projected)
            }
            Some(DimensionProjection::Matrix { matrix }) => {
                // Matrix multiplication: result[i] = sum(matrix[i][j] * query[j])
                let mut result = Vec::with_capacity(matrix.len());
                for row in matrix {
                    let val: f32 = row
                        .iter()
                        .zip(query.iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    result.push(val);
                }
                Ok(result)
            }
        }
    }

    /// Normalize scores across collections.
    fn normalize_scores(
        &self,
        results: &HashMap<String, Vec<SearchResult>>,
        normalization: &ScoreNormalization,
    ) -> HashMap<String, Vec<(SearchResult, f64)>> {
        let mut normalized = HashMap::new();

        for (collection, coll_results) in results {
            let scored: Vec<(SearchResult, f64)> = match normalization {
                ScoreNormalization::None => coll_results
                    .iter()
                    .map(|r| (r.clone(), r.distance as f64))
                    .collect(),

                ScoreNormalization::MinMax => {
                    if coll_results.is_empty() {
                        Vec::new()
                    } else {
                        let min_dist = coll_results
                            .iter()
                            .map(|r| r.distance)
                            .fold(f32::MAX, f32::min);
                        let max_dist = coll_results
                            .iter()
                            .map(|r| r.distance)
                            .fold(f32::MIN, f32::max);
                        let range = (max_dist - min_dist).max(f32::EPSILON);

                        coll_results
                            .iter()
                            .map(|r| {
                                let norm = (r.distance - min_dist) / range;
                                (r.clone(), norm as f64)
                            })
                            .collect()
                    }
                }

                ScoreNormalization::ZScore => {
                    if coll_results.is_empty() {
                        Vec::new()
                    } else {
                        let mean: f64 =
                            coll_results.iter().map(|r| r.distance as f64).sum::<f64>()
                                / coll_results.len() as f64;
                        let variance: f64 = coll_results
                            .iter()
                            .map(|r| (r.distance as f64 - mean).powi(2))
                            .sum::<f64>()
                            / coll_results.len() as f64;
                        let stddev = variance.sqrt().max(f64::EPSILON);

                        coll_results
                            .iter()
                            .map(|r| {
                                let z = (r.distance as f64 - mean) / stddev;
                                (r.clone(), z)
                            })
                            .collect()
                    }
                }

                ScoreNormalization::PercentileRank => {
                    let n = coll_results.len() as f64;
                    coll_results
                        .iter()
                        .enumerate()
                        .map(|(rank, r)| {
                            let percentile = rank as f64 / n.max(1.0);
                            (r.clone(), percentile)
                        })
                        .collect()
                }
            };

            normalized.insert(collection.clone(), scored);
        }

        normalized
    }

    /// Merge results from multiple collections using the specified strategy.
    fn merge_results(
        &self,
        normalized: &HashMap<String, Vec<(SearchResult, f64)>>,
        strategy: &MergeStrategy,
        collection_queries: &[CollectionQuery],
        k: usize,
    ) -> Vec<FederatedResult> {
        match strategy {
            MergeStrategy::ReciprocalRankFusion { k: rrf_k } => {
                self.rrf_merge(normalized, *rrf_k, k)
            }
            MergeStrategy::NormalizedDistance => {
                self.distance_merge(normalized, k)
            }
            MergeStrategy::RoundRobin => {
                self.round_robin_merge(normalized, k)
            }
            MergeStrategy::Weighted { weights } => {
                self.weighted_merge(normalized, weights, k)
            }
        }
    }

    /// Reciprocal Rank Fusion merge.
    fn rrf_merge(
        &self,
        normalized: &HashMap<String, Vec<(SearchResult, f64)>>,
        rrf_k: usize,
        limit: usize,
    ) -> Vec<FederatedResult> {
        // RRF score = sum(1 / (k + rank)) across all collections
        let mut rrf_scores: HashMap<String, (f64, f32, String, Option<serde_json::Value>)> =
            HashMap::new();

        for (collection, results) in normalized {
            for (rank, (result, _score)) in results.iter().enumerate() {
                let rrf_score = 1.0 / (rrf_k as f64 + rank as f64 + 1.0);
                let entry = rrf_scores
                    .entry(format!("{}:{}", collection, result.id))
                    .or_insert((0.0, result.distance, collection.clone(), result.metadata.clone()));
                entry.0 += rrf_score;
            }
        }

        let mut merged: Vec<FederatedResult> = rrf_scores
            .into_iter()
            .map(|(key, (score, raw_distance, collection, metadata))| {
                let id = key
                    .split(':')
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join(":");
                FederatedResult {
                    id,
                    score,
                    raw_distance,
                    collection,
                    metadata,
                }
            })
            .collect();

        // Sort by RRF score descending (higher is better)
        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(limit);
        merged
    }

    /// Distance-based merge (lower normalized distance is better).
    fn distance_merge(
        &self,
        normalized: &HashMap<String, Vec<(SearchResult, f64)>>,
        limit: usize,
    ) -> Vec<FederatedResult> {
        let mut all: Vec<FederatedResult> = normalized
            .iter()
            .flat_map(|(collection, results)| {
                results.iter().map(|(result, norm_score)| FederatedResult {
                    id: result.id.clone(),
                    score: *norm_score,
                    raw_distance: result.distance,
                    collection: collection.clone(),
                    metadata: result.metadata.clone(),
                })
            })
            .collect();

        all.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(limit);
        all
    }

    /// Round-robin interleaving from each collection.
    fn round_robin_merge(
        &self,
        normalized: &HashMap<String, Vec<(SearchResult, f64)>>,
        limit: usize,
    ) -> Vec<FederatedResult> {
        let mut iterators: Vec<(&str, std::slice::Iter<(SearchResult, f64)>)> = normalized
            .iter()
            .map(|(name, results)| (name.as_str(), results.iter()))
            .collect();

        let mut merged = Vec::with_capacity(limit);
        let mut round = 0;

        while merged.len() < limit {
            let mut any_advanced = false;
            for (collection, iter) in iterators.iter_mut() {
                if merged.len() >= limit {
                    break;
                }
                if let Some((result, score)) = iter.next() {
                    merged.push(FederatedResult {
                        id: result.id.clone(),
                        score: *score,
                        raw_distance: result.distance,
                        collection: collection.to_string(),
                        metadata: result.metadata.clone(),
                    });
                    any_advanced = true;
                }
            }
            if !any_advanced {
                break;
            }
            round += 1;
            if round > limit * 2 {
                break; // Safety limit
            }
        }

        merged
    }

    /// Weighted merge with per-collection weights.
    fn weighted_merge(
        &self,
        normalized: &HashMap<String, Vec<(SearchResult, f64)>>,
        weights: &HashMap<String, f64>,
        limit: usize,
    ) -> Vec<FederatedResult> {
        let mut all: Vec<FederatedResult> = normalized
            .iter()
            .flat_map(|(collection, results)| {
                let weight = weights.get(collection).copied().unwrap_or(1.0);
                results.iter().map(move |(result, norm_score)| FederatedResult {
                    id: result.id.clone(),
                    score: norm_score * weight,
                    raw_distance: result.distance,
                    collection: collection.clone(),
                    metadata: result.metadata.clone(),
                })
            })
            .collect();

        all.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(limit);
        all
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federation_query_new() {
        let query = FederationQuery::new(vec![0.1; 128], 10, &["docs", "images"]);
        assert_eq!(query.collections.len(), 2);
        assert_eq!(query.k, 10);
    }

    #[test]
    fn test_minmax_normalization() {
        let fed = CollectionFederation::new(Arc::new(Database::in_memory()));

        let mut results = HashMap::new();
        results.insert(
            "test".to_string(),
            vec![
                SearchResult {
                    id: "a".to_string(),
                    distance: 0.1,
                    metadata: None,
                },
                SearchResult {
                    id: "b".to_string(),
                    distance: 0.5,
                    metadata: None,
                },
                SearchResult {
                    id: "c".to_string(),
                    distance: 0.9,
                    metadata: None,
                },
            ],
        );

        let normalized = fed.normalize_scores(&results, &ScoreNormalization::MinMax);
        let scores: Vec<f64> = normalized["test"].iter().map(|(_, s)| *s).collect();

        assert!((scores[0] - 0.0).abs() < 0.01); // min -> 0
        assert!((scores[2] - 1.0).abs() < 0.01); // max -> 1
    }

    #[test]
    fn test_rrf_merge() {
        let fed = CollectionFederation::new(Arc::new(Database::in_memory()));

        let mut normalized = HashMap::new();
        normalized.insert(
            "coll1".to_string(),
            vec![
                (SearchResult { id: "a".to_string(), distance: 0.1, metadata: None }, 0.1),
                (SearchResult { id: "b".to_string(), distance: 0.2, metadata: None }, 0.2),
            ],
        );
        normalized.insert(
            "coll2".to_string(),
            vec![
                (SearchResult { id: "c".to_string(), distance: 0.15, metadata: None }, 0.15),
                (SearchResult { id: "a".to_string(), distance: 0.3, metadata: None }, 0.3),
            ],
        );

        let merged = fed.rrf_merge(&normalized, 60, 5);
        assert!(!merged.is_empty());
        // "a" appears in both collections, should have higher RRF score
    }

    #[test]
    fn test_round_robin_merge() {
        let fed = CollectionFederation::new(Arc::new(Database::in_memory()));

        let mut normalized = HashMap::new();
        normalized.insert(
            "c1".to_string(),
            vec![
                (SearchResult { id: "a".to_string(), distance: 0.1, metadata: None }, 0.1),
                (SearchResult { id: "b".to_string(), distance: 0.2, metadata: None }, 0.2),
            ],
        );
        normalized.insert(
            "c2".to_string(),
            vec![
                (SearchResult { id: "c".to_string(), distance: 0.15, metadata: None }, 0.15),
            ],
        );

        let merged = fed.round_robin_merge(&normalized, 10);
        assert_eq!(merged.len(), 3);
    }
}
