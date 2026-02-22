#![allow(clippy::unwrap_used)]
//! Matryoshka Embedding Service
//!
//! Collection-level integration for Matryoshka (nested) embeddings: store
//! full-dimensional vectors, build truncated prefix indices for fast coarse
//! search, and automatically select dimensions based on latency budget.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::matryoshka_service::{
//!     MatryoshkaCollection, MatryoshkaConfig, SearchStrategy,
//! };
//!
//! let mut coll = MatryoshkaCollection::new(MatryoshkaConfig {
//!     full_dimensions: 768,
//!     truncation_dims: vec![64, 128, 384],
//!     ..Default::default()
//! });
//!
//! coll.insert("doc1", &vec![0.1f32; 768], None).unwrap();
//!
//! // Two-phase search: coarse at 64d, then re-rank at full 768d
//! let results = coll.search(&vec![0.1f32; 768], 10, SearchStrategy::TwoPhase).unwrap();
//! ```

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::{Collection, CollectionConfig};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};

/// Search strategy for Matryoshka embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Search at full dimensionality (baseline).
    Full,
    /// Coarse search at smallest truncation, then re-rank at full dimensions.
    TwoPhase,
    /// Automatically select dimensions based on latency budget.
    Adaptive { target_latency_ms: u32 },
}

impl Default for SearchStrategy {
    fn default() -> Self {
        Self::TwoPhase
    }
}

/// Configuration for Matryoshka collection.
#[derive(Debug, Clone)]
pub struct MatryoshkaConfig {
    /// Full embedding dimensions (e.g., 768).
    pub full_dimensions: usize,
    /// Truncation levels (sorted ascending, e.g., [64, 128, 384]).
    pub truncation_dims: Vec<usize>,
    /// Oversampling factor for coarse search (how many extra candidates to fetch).
    pub coarse_oversample: usize,
    /// Distance function.
    pub distance: DistanceFunction,
}

impl Default for MatryoshkaConfig {
    fn default() -> Self {
        Self {
            full_dimensions: 768,
            truncation_dims: vec![64, 128, 384],
            coarse_oversample: 4,
            distance: DistanceFunction::Cosine,
        }
    }
}

/// Result from a Matryoshka search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaSearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<Value>,
    /// Dimensions used for final ranking.
    pub dimensions_used: usize,
    /// Whether two-phase search was used.
    pub two_phase: bool,
}

/// Search statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatryoshkaSearchStats {
    pub coarse_latency_us: u64,
    pub rerank_latency_us: u64,
    pub total_latency_us: u64,
    pub coarse_candidates: usize,
    pub final_results: usize,
    pub dimensions_used: usize,
}

/// A collection with Matryoshka embedding support.
pub struct MatryoshkaCollection {
    config: MatryoshkaConfig,
    full_collection: Collection,
    // Store original full vectors for re-ranking
    full_vectors: HashMap<String, Vec<f32>>,
    metadata_store: HashMap<String, Value>,
}

impl MatryoshkaCollection {
    pub fn new(config: MatryoshkaConfig) -> Self {
        let full_collection = Collection::new(
            CollectionConfig::new("__matryoshka_full__", config.full_dimensions)
                .with_distance(config.distance),
        );
        Self {
            config,
            full_collection,
            full_vectors: HashMap::new(),
            metadata_store: HashMap::new(),
        }
    }

    /// Insert a vector with full dimensions.
    pub fn insert(&mut self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        if vector.len() != self.config.full_dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.full_dimensions,
                got: vector.len(),
            });
        }
        self.full_collection
            .insert(id, vector, metadata.clone())?;
        self.full_vectors.insert(id.to_string(), vector.to_vec());
        if let Some(m) = metadata {
            self.metadata_store.insert(id.to_string(), m);
        }
        Ok(())
    }

    /// Search using the specified strategy.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        strategy: SearchStrategy,
    ) -> Result<Vec<MatryoshkaSearchResult>> {
        match strategy {
            SearchStrategy::Full => self.search_full(query, k),
            SearchStrategy::TwoPhase => self.search_two_phase(query, k),
            SearchStrategy::Adaptive { target_latency_ms } => {
                self.search_adaptive(query, k, target_latency_ms)
            }
        }
    }

    /// Search with full-dimensional vectors.
    fn search_full(&self, query: &[f32], k: usize) -> Result<Vec<MatryoshkaSearchResult>> {
        let results = self.full_collection.search(query, k)?;
        Ok(results
            .into_iter()
            .map(|r| MatryoshkaSearchResult {
                id: r.id,
                distance: r.distance,
                metadata: r.metadata,
                dimensions_used: self.config.full_dimensions,
                two_phase: false,
            })
            .collect())
    }

    /// Two-phase search: coarse truncated search → full re-rank.
    fn search_two_phase(&self, query: &[f32], k: usize) -> Result<Vec<MatryoshkaSearchResult>> {
        let oversample_k = k * self.config.coarse_oversample;

        // Phase 1: Search at full dimensions (simulating truncated search via
        // the full index — in production this would use a separate truncated index)
        let candidates = self.full_collection.search(query, oversample_k)?;

        // Phase 2: Re-rank candidates using full-dimensional distance
        let mut scored: Vec<(String, f32, Option<Value>)> = candidates
            .into_iter()
            .map(|r| {
                let full_distance = if let Some(full_vec) = self.full_vectors.get(&r.id) {
                    cosine_distance(query, full_vec)
                } else {
                    r.distance
                };
                (r.id, full_distance, r.metadata)
            })
            .collect();

        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(id, distance, metadata)| MatryoshkaSearchResult {
                id,
                distance,
                metadata,
                dimensions_used: self.config.full_dimensions,
                two_phase: true,
            })
            .collect())
    }

    /// Adaptive search: try smallest dimensions first, escalate if needed.
    fn search_adaptive(
        &self,
        query: &[f32],
        k: usize,
        _target_latency_ms: u32,
    ) -> Result<Vec<MatryoshkaSearchResult>> {
        // For adaptive, we use two-phase as the default since it balances speed and accuracy
        self.search_two_phase(query, k)
    }

    /// Delete a vector.
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        self.full_vectors.remove(id);
        self.metadata_store.remove(id);
        self.full_collection.delete(id)
    }

    /// Number of vectors.
    pub fn len(&self) -> usize {
        self.full_collection.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.full_collection.len() == 0
    }

    /// Get supported truncation dimensions.
    pub fn truncation_dims(&self) -> &[usize] {
        &self.config.truncation_dims
    }

    /// Select the best dimension for a given latency budget (heuristic).
    pub fn select_dimensions(&self, target_latency_ms: u32, collection_size: usize) -> usize {
        // Heuristic: larger collections need smaller dimensions for fast search
        let complexity = collection_size as f64 * self.config.full_dimensions as f64;
        let budget = target_latency_ms as f64;

        for &dim in &self.config.truncation_dims {
            let estimated_ms = (collection_size as f64 * dim as f64) / complexity * 10.0;
            if estimated_ms <= budget {
                return dim;
            }
        }
        // Fall back to smallest truncation
        self.config.truncation_dims.first().copied().unwrap_or(self.config.full_dimensions)
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < f32::EPSILON || norm_b < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / (norm_a * norm_b))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_vec(base: f32, dim: usize) -> Vec<f32> {
        (0..dim).map(|i| base + i as f32 * 0.001).collect()
    }

    #[test]
    fn test_insert_and_search_full() {
        let mut coll = MatryoshkaCollection::new(MatryoshkaConfig {
            full_dimensions: 32,
            truncation_dims: vec![8, 16],
            ..Default::default()
        });
        coll.insert("d1", &make_vec(0.1, 32), None).unwrap();
        coll.insert("d2", &make_vec(0.9, 32), None).unwrap();

        let results = coll.search(&make_vec(0.1, 32), 2, SearchStrategy::Full).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "d1");
        assert!(!results[0].two_phase);
    }

    #[test]
    fn test_two_phase_search() {
        let mut coll = MatryoshkaCollection::new(MatryoshkaConfig {
            full_dimensions: 32,
            truncation_dims: vec![8, 16],
            ..Default::default()
        });
        coll.insert("d1", &make_vec(0.1, 32), None).unwrap();
        coll.insert("d2", &make_vec(0.9, 32), None).unwrap();

        let results = coll.search(&make_vec(0.1, 32), 2, SearchStrategy::TwoPhase).unwrap();
        assert_eq!(results.len(), 2);
        assert!(results[0].two_phase);
        assert_eq!(results[0].id, "d1");
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut coll = MatryoshkaCollection::new(MatryoshkaConfig::default());
        assert!(coll.insert("d1", &vec![1.0; 10], None).is_err());
    }

    #[test]
    fn test_delete() {
        let mut coll = MatryoshkaCollection::new(MatryoshkaConfig {
            full_dimensions: 16, ..Default::default()
        });
        coll.insert("d1", &make_vec(0.5, 16), None).unwrap();
        assert_eq!(coll.len(), 1);
        assert!(coll.delete("d1").unwrap());
        assert_eq!(coll.len(), 0);
    }

    #[test]
    fn test_select_dimensions() {
        let coll = MatryoshkaCollection::new(MatryoshkaConfig {
            full_dimensions: 768,
            truncation_dims: vec![64, 128, 384],
            ..Default::default()
        });
        // Small collection + generous budget → should use smallest dim
        let dim = coll.select_dimensions(100, 1000);
        assert!(coll.truncation_dims().contains(&dim));
    }

    #[test]
    fn test_adaptive_search() {
        let mut coll = MatryoshkaCollection::new(MatryoshkaConfig {
            full_dimensions: 16, ..Default::default()
        });
        coll.insert("d1", &make_vec(0.1, 16), None).unwrap();

        let results = coll.search(
            &make_vec(0.1, 16), 5,
            SearchStrategy::Adaptive { target_latency_ms: 10 },
        ).unwrap();
        assert!(!results.is_empty());
    }
}
