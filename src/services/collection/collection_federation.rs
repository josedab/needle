#![allow(clippy::unwrap_used)]
//! Cross-Collection Federation
//!
//! Federated search across multiple collections with per-collection weights,
//! result merging via RRF or weighted sum, and type-aware score normalization.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::collection_federation::{
//!     FederatedSearch, FederationConfig, MergeStrategy,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("text", 4).unwrap();
//! db.create_collection("images", 4).unwrap();
//!
//! let mut fed = FederatedSearch::new(&db, FederationConfig::default());
//!
//! let results = fed.search(
//!     &[0.5f32; 4],
//!     &["text", "images"],
//!     10,
//! ).unwrap();
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::SearchResult;
use crate::database::Database;
use crate::error::{NeedleError, Result};

// ── Configuration ────────────────────────────────────────────────────────────

/// Merge strategy for combining results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Reciprocal Rank Fusion (k=60).
    Rrf,
    /// Weighted sum of normalized scores.
    WeightedSum,
    /// Interleave results round-robin.
    Interleave,
    /// Take the minimum distance across collections.
    MinDistance,
}

impl Default for MergeStrategy {
    fn default() -> Self { Self::Rrf }
}

/// Federation configuration.
#[derive(Debug, Clone)]
pub struct FederationConfig {
    /// Merge strategy.
    pub strategy: MergeStrategy,
    /// Per-collection weights (collection_name → weight).
    pub weights: HashMap<String, f32>,
    /// Over-fetch factor per collection.
    pub overfetch_factor: usize,
    /// RRF k parameter.
    pub rrf_k: f32,
    /// Whether to deduplicate by ID across collections.
    pub dedup_by_id: bool,
}

impl Default for FederationConfig {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Rrf,
            weights: HashMap::new(),
            overfetch_factor: 3,
            rrf_k: 60.0,
            dedup_by_id: true,
        }
    }
}

impl FederationConfig {
    /// Set merge strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set weight for a collection.
    #[must_use]
    pub fn with_weight(mut self, collection: &str, weight: f32) -> Self {
        self.weights.insert(collection.into(), weight);
        self
    }
}

// ── Federated Result ─────────────────────────────────────────────────────────

/// A result from federated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// Vector ID.
    pub id: String,
    /// Source collection.
    pub collection: String,
    /// Fused score (lower = better).
    pub score: f32,
    /// Original distance from source collection.
    pub original_distance: f32,
    /// Metadata.
    pub metadata: Option<Value>,
}

// ── Federation Statistics ────────────────────────────────────────────────────

/// Statistics from a federated search.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FederationStats {
    /// Collections searched.
    pub collections_searched: usize,
    /// Total candidates before merge.
    pub total_candidates: usize,
    /// Results after merge and dedup.
    pub results_returned: usize,
    /// Per-collection result counts.
    pub per_collection: HashMap<String, usize>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

// ── Federated Search ─────────────────────────────────────────────────────────

/// Federated search across multiple collections.
pub struct FederatedSearch<'a> {
    db: &'a Database,
    config: FederationConfig,
}

impl<'a> FederatedSearch<'a> {
    /// Create a new federated search.
    pub fn new(db: &'a Database, config: FederationConfig) -> Self {
        Self { db, config }
    }

    /// Search across multiple collections.
    pub fn search(
        &self,
        query: &[f32],
        collections: &[&str],
        k: usize,
    ) -> Result<Vec<FederatedResult>> {
        let fetch_k = k * self.config.overfetch_factor;
        let mut all_results: Vec<(String, Vec<SearchResult>)> = Vec::new();

        for &name in collections {
            let coll = self.db.collection(name)?;
            let results = coll.search(query, fetch_k)?;
            all_results.push((name.into(), results));
        }

        let merged = match self.config.strategy {
            MergeStrategy::Rrf => self.merge_rrf(&all_results, k),
            MergeStrategy::WeightedSum => self.merge_weighted(&all_results, k),
            MergeStrategy::Interleave => Self::merge_interleave(&all_results, k),
            MergeStrategy::MinDistance => Self::merge_min_distance(&all_results, k),
        };

        Ok(merged)
    }

    /// Search with custom per-query weights.
    pub fn search_weighted(
        &self,
        query: &[f32],
        collection_weights: &[(&str, f32)],
        k: usize,
    ) -> Result<Vec<FederatedResult>> {
        let fetch_k = k * self.config.overfetch_factor;
        let mut all_results: Vec<(String, Vec<SearchResult>)> = Vec::new();

        for &(name, _) in collection_weights {
            let coll = self.db.collection(name)?;
            let results = coll.search(query, fetch_k)?;
            all_results.push((name.into(), results));
        }

        let weights: HashMap<String, f32> = collection_weights.iter()
            .map(|(n, w)| (n.to_string(), *w))
            .collect();

        let mut merged = self.merge_rrf(&all_results, k * 2);
        // Apply weights
        for result in &mut merged {
            let w = weights.get(&result.collection).copied().unwrap_or(1.0);
            result.score /= w.max(0.01);
        }
        merged.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        merged.truncate(k);
        Ok(merged)
    }

    fn merge_rrf(&self, results: &[(String, Vec<SearchResult>)], k: usize) -> Vec<FederatedResult> {
        let mut scores: HashMap<String, (f32, String, f32, Option<Value>)> = HashMap::new();

        for (collection, search_results) in results {
            for (rank, sr) in search_results.iter().enumerate() {
                let rrf_score = 1.0 / (self.config.rrf_k + rank as f32 + 1.0);
                let weight = self.config.weights.get(collection).copied().unwrap_or(1.0);
                let entry = scores.entry(sr.id.clone()).or_insert((0.0, collection.clone(), sr.distance, sr.metadata.clone()));
                entry.0 += rrf_score * weight;
            }
        }

        let mut results: Vec<FederatedResult> = scores.into_iter()
            .map(|(id, (score, coll, dist, meta))| FederatedResult {
                id, collection: coll, score: 1.0 / score, original_distance: dist, metadata: meta,
            })
            .collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    fn merge_weighted(&self, results: &[(String, Vec<SearchResult>)], k: usize) -> Vec<FederatedResult> {
        let mut all: Vec<FederatedResult> = Vec::new();
        for (collection, search_results) in results {
            let weight = self.config.weights.get(collection).copied().unwrap_or(1.0);
            for sr in search_results {
                all.push(FederatedResult {
                    id: sr.id.clone(), collection: collection.clone(),
                    score: sr.distance / weight.max(0.01), original_distance: sr.distance,
                    metadata: sr.metadata.clone(),
                });
            }
        }
        all.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        if self.config.dedup_by_id { dedup_by_id(&mut all); }
        all.truncate(k);
        all
    }

    fn merge_interleave(results: &[(String, Vec<SearchResult>)], k: usize) -> Vec<FederatedResult> {
        let mut merged = Vec::new();
        let max_len = results.iter().map(|(_, r)| r.len()).max().unwrap_or(0);
        for i in 0..max_len {
            for (collection, search_results) in results {
                if let Some(sr) = search_results.get(i) {
                    merged.push(FederatedResult {
                        id: sr.id.clone(), collection: collection.clone(),
                        score: sr.distance, original_distance: sr.distance,
                        metadata: sr.metadata.clone(),
                    });
                }
            }
            if merged.len() >= k { break; }
        }
        merged.truncate(k);
        merged
    }

    fn merge_min_distance(results: &[(String, Vec<SearchResult>)], k: usize) -> Vec<FederatedResult> {
        let mut best: HashMap<String, FederatedResult> = HashMap::new();
        for (collection, search_results) in results {
            for sr in search_results {
                let entry = best.entry(sr.id.clone()).or_insert(FederatedResult {
                    id: sr.id.clone(), collection: collection.clone(),
                    score: f32::MAX, original_distance: sr.distance,
                    metadata: sr.metadata.clone(),
                });
                if sr.distance < entry.score {
                    entry.score = sr.distance;
                    entry.collection = collection.clone();
                    entry.original_distance = sr.distance;
                }
            }
        }
        let mut results: Vec<FederatedResult> = best.into_values().collect();
        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

fn dedup_by_id(results: &mut Vec<FederatedResult>) {
    let mut seen = std::collections::HashSet::new();
    results.retain(|r| seen.insert(r.id.clone()));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("text", 4).unwrap();
        db.create_collection("images", 4).unwrap();

        let t = db.collection("text").unwrap();
        t.insert("t1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        t.insert("t2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();

        let i = db.collection("images").unwrap();
        i.insert("i1", &[0.5, 0.5, 0.0, 0.0], None).unwrap();
        i.insert("i2", &[0.0, 0.0, 1.0, 0.0], None).unwrap();
        db
    }

    #[test]
    fn test_rrf_federation() {
        let db = setup_db();
        let fed = FederatedSearch::new(&db, FederationConfig::default());
        let results = fed.search(&[1.0, 0.0, 0.0, 0.0], &["text", "images"], 5).unwrap();
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_weighted_federation() {
        let db = setup_db();
        let config = FederationConfig::default().with_strategy(MergeStrategy::WeightedSum);
        let fed = FederatedSearch::new(&db, config);
        let results = fed.search(&[1.0, 0.0, 0.0, 0.0], &["text", "images"], 5).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_interleave() {
        let db = setup_db();
        let config = FederationConfig::default().with_strategy(MergeStrategy::Interleave);
        let fed = FederatedSearch::new(&db, config);
        let results = fed.search(&[1.0, 0.0, 0.0, 0.0], &["text", "images"], 4).unwrap();
        // Should interleave text and images
        assert!(results.len() <= 4);
    }

    #[test]
    fn test_min_distance() {
        let db = setup_db();
        let config = FederationConfig::default().with_strategy(MergeStrategy::MinDistance);
        let fed = FederatedSearch::new(&db, config);
        let results = fed.search(&[1.0, 0.0, 0.0, 0.0], &["text", "images"], 5).unwrap();
        assert_eq!(results[0].id, "t1"); // closest to [1,0,0,0]
    }

    #[test]
    fn test_weighted_search() {
        let db = setup_db();
        let fed = FederatedSearch::new(&db, FederationConfig::default());
        let results = fed.search_weighted(
            &[1.0, 0.0, 0.0, 0.0],
            &[("text", 2.0), ("images", 0.5)],
            5,
        ).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_missing_collection() {
        let db = setup_db();
        let fed = FederatedSearch::new(&db, FederationConfig::default());
        assert!(fed.search(&[1.0; 4], &["nonexistent"], 5).is_err());
    }
}
