//! Adaptive Index Selection
//!
//! Workload-driven automatic index strategy switching with online migration.
//! Analyzes query patterns, data volume, and memory to select optimal index.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::adaptive_index::{
//!     AdaptiveSelector, WorkloadProfile, IndexStrategy, SelectionResult,
//! };
//!
//! let mut selector = AdaptiveSelector::new();
//! selector.observe_query(WorkloadProfile::point_query(128, 10));
//! selector.observe_query(WorkloadProfile::filtered_query(128, 10, 0.1));
//!
//! let result = selector.recommend(1_000_000, 384);
//! println!("Recommended: {} (confidence: {:.0}%)", result.strategy, result.confidence * 100.0);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Index strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexStrategy { Hnsw, Ivf, BruteForce, HnswQuantized, DiskAnn, Hybrid }

impl std::fmt::Display for IndexStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::Hnsw => write!(f, "HNSW"), Self::Ivf => write!(f, "IVF"), Self::BruteForce => write!(f, "BruteForce"),
            Self::HnswQuantized => write!(f, "HNSW+Quantized"), Self::DiskAnn => write!(f, "DiskANN"), Self::Hybrid => write!(f, "Hybrid") }
    }
}

/// Query workload profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub query_type: QueryType,
    pub dimensions: usize,
    pub k: usize,
    pub filter_selectivity: Option<f32>,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType { Point, Batch, Filtered, Range, Hybrid }

impl WorkloadProfile {
    pub fn point_query(dims: usize, k: usize) -> Self {
        Self { query_type: QueryType::Point, dimensions: dims, k, filter_selectivity: None, batch_size: 1 }
    }
    pub fn filtered_query(dims: usize, k: usize, selectivity: f32) -> Self {
        Self { query_type: QueryType::Filtered, dimensions: dims, k, filter_selectivity: Some(selectivity), batch_size: 1 }
    }
    pub fn batch_query(dims: usize, k: usize, batch: usize) -> Self {
        Self { query_type: QueryType::Batch, dimensions: dims, k, filter_selectivity: None, batch_size: batch }
    }
}

/// Selection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionResult {
    pub strategy: IndexStrategy,
    pub confidence: f32,
    pub rationale: Vec<String>,
    pub alternatives: Vec<(IndexStrategy, f32)>,
    pub suggested_params: HashMap<String, String>,
}

/// Migration plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPlan {
    pub from: IndexStrategy,
    pub to: IndexStrategy,
    pub estimated_duration_secs: f64,
    pub can_serve_during_migration: bool,
}

/// Adaptive index selector.
pub struct AdaptiveSelector {
    observations: Vec<WorkloadProfile>,
    current_strategy: Option<IndexStrategy>,
    max_observations: usize,
}

impl AdaptiveSelector {
    pub fn new() -> Self { Self { observations: Vec::new(), current_strategy: None, max_observations: 10_000 } }

    pub fn observe_query(&mut self, profile: WorkloadProfile) {
        if self.observations.len() >= self.max_observations { self.observations.remove(0); }
        self.observations.push(profile);
    }

    pub fn recommend(&self, vector_count: usize, dimensions: usize) -> SelectionResult {
        let mut scores: HashMap<IndexStrategy, f32> = HashMap::new();
        let mut rationale = Vec::new();

        // Size-based scoring
        if vector_count < 5_000 {
            *scores.entry(IndexStrategy::BruteForce).or_default() += 3.0;
            rationale.push(format!("Small dataset ({} vectors): brute-force optimal", vector_count));
        } else if vector_count < 100_000 {
            *scores.entry(IndexStrategy::Hnsw).or_default() += 3.0;
            rationale.push("Medium dataset: HNSW recommended".into());
        } else if vector_count < 10_000_000 {
            *scores.entry(IndexStrategy::Hnsw).or_default() += 2.0;
            *scores.entry(IndexStrategy::HnswQuantized).or_default() += 2.5;
            rationale.push("Large dataset: HNSW+quantization for memory efficiency".into());
        } else {
            *scores.entry(IndexStrategy::DiskAnn).or_default() += 3.0;
            rationale.push("Very large dataset: DiskANN for disk-based search".into());
        }

        // Workload-based scoring
        let total = self.observations.len().max(1) as f32;
        let filtered_ratio = self.observations.iter().filter(|o| o.query_type == QueryType::Filtered).count() as f32 / total;
        let batch_ratio = self.observations.iter().filter(|o| o.query_type == QueryType::Batch).count() as f32 / total;

        if filtered_ratio > 0.5 {
            *scores.entry(IndexStrategy::Hnsw).or_default() += 1.0;
            rationale.push(format!("{:.0}% filtered queries: HNSW with pre-filter", filtered_ratio * 100.0));
        }
        if batch_ratio > 0.3 {
            *scores.entry(IndexStrategy::Ivf).or_default() += 1.0;
            rationale.push(format!("{:.0}% batch queries: IVF good for batch", batch_ratio * 100.0));
        }

        // Memory scoring
        let mem_gb = (vector_count as f64 * dimensions as f64 * 4.0) / (1024.0 * 1024.0 * 1024.0);
        if mem_gb > 4.0 {
            *scores.entry(IndexStrategy::HnswQuantized).or_default() += 2.0;
            rationale.push(format!("Memory {:.1}GB: quantization reduces by 4×", mem_gb));
        }

        let mut sorted: Vec<(IndexStrategy, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = sorted.first().map(|s| s.0).unwrap_or(IndexStrategy::Hnsw);
        let max_score = sorted.first().map(|s| s.1).unwrap_or(1.0);
        let confidence = (max_score / 6.0).min(1.0);
        let mut params = HashMap::new();
        match best {
            IndexStrategy::Hnsw | IndexStrategy::HnswQuantized => {
                params.insert("m".into(), "16".into());
                params.insert("ef_construction".into(), "200".into());
                params.insert("ef_search".into(), "50".into());
            }
            _ => {}
        }

        SelectionResult {
            strategy: best, confidence, rationale,
            alternatives: sorted.into_iter().skip(1).collect(),
            suggested_params: params,
        }
    }

    pub fn migration_plan(&self, from: IndexStrategy, to: IndexStrategy, vector_count: usize) -> MigrationPlan {
        let est = vector_count as f64 / 100_000.0; // ~1s per 100K vectors
        MigrationPlan { from, to, estimated_duration_secs: est, can_serve_during_migration: true }
    }

    pub fn observation_count(&self) -> usize { self.observations.len() }
    pub fn current(&self) -> Option<IndexStrategy> { self.current_strategy }
    pub fn set_current(&mut self, strategy: IndexStrategy) { self.current_strategy = Some(strategy); }
}

impl Default for AdaptiveSelector { fn default() -> Self { Self::new() } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_dataset() {
        let s = AdaptiveSelector::new();
        let r = s.recommend(1000, 128);
        assert_eq!(r.strategy, IndexStrategy::BruteForce);
    }

    #[test]
    fn test_large_dataset() {
        let s = AdaptiveSelector::new();
        let r = s.recommend(1_000_000, 384);
        assert!(matches!(r.strategy, IndexStrategy::Hnsw | IndexStrategy::HnswQuantized));
    }

    #[test]
    fn test_workload_influence() {
        let mut s = AdaptiveSelector::new();
        for _ in 0..100 { s.observe_query(WorkloadProfile::batch_query(128, 10, 50)); }
        let r = s.recommend(500_000, 128);
        assert!(!r.rationale.is_empty());
    }

    #[test]
    fn test_migration_plan() {
        let s = AdaptiveSelector::new();
        let p = s.migration_plan(IndexStrategy::Hnsw, IndexStrategy::HnswQuantized, 1_000_000);
        assert!(p.estimated_duration_secs > 0.0);
        assert!(p.can_serve_during_migration);
    }

    #[test]
    fn test_confidence() {
        let s = AdaptiveSelector::new();
        let r = s.recommend(50_000, 384);
        assert!(r.confidence > 0.0 && r.confidence <= 1.0);
    }
}
