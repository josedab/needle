#![allow(clippy::unwrap_used)]
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

/// Full evaluation result from the adaptive index selector.
#[derive(Debug, Clone)]
pub struct IndexEvaluation {
    /// The recommended strategy and its parameters.
    pub recommended: SelectionResult,
    /// The currently active strategy, if set.
    pub current_strategy: Option<IndexStrategy>,
    /// Whether a migration is recommended.
    pub needs_migration: bool,
    /// Migration plan if migration is needed.
    pub migration_plan: Option<MigrationPlan>,
    /// Latency statistics: (p50, p95, p99) in ms.
    pub latency_stats: Option<(f64, f64, f64)>,
    /// Whether current latency meets the configured target.
    pub latency_meets_target: bool,
    /// Estimated memory usage in bytes for the current dataset.
    pub memory_estimate_bytes: usize,
    /// Whether memory usage is within the configured budget.
    pub memory_within_budget: bool,
    /// Number of workload observations collected.
    pub observation_count: usize,
}

/// Adaptive index selector.
pub struct AdaptiveSelector {
    observations: Vec<WorkloadProfile>,
    current_strategy: Option<IndexStrategy>,
    max_observations: usize,
    memory_budget_bytes: Option<usize>,
    latency_target_ms: Option<f64>,
    latency_samples: Vec<f64>,
}

impl AdaptiveSelector {
    pub fn new() -> Self { Self { observations: Vec::new(), current_strategy: None, max_observations: 10_000, memory_budget_bytes: None, latency_target_ms: None, latency_samples: Vec::new() } }

    /// Set a memory budget constraint.
    #[must_use]
    pub fn with_memory_budget(mut self, bytes: usize) -> Self { self.memory_budget_bytes = Some(bytes); self }

    /// Set a latency target in milliseconds.
    #[must_use]
    pub fn with_latency_target(mut self, ms: f64) -> Self { self.latency_target_ms = Some(ms); self }

    pub fn observe_query(&mut self, profile: WorkloadProfile) {
        if self.observations.len() >= self.max_observations { self.observations.remove(0); }
        self.observations.push(profile);
    }

    /// Record a query latency sample for tracking.
    pub fn record_latency(&mut self, latency_ms: f64) {
        self.latency_samples.push(latency_ms);
        if self.latency_samples.len() > 1000 {
            self.latency_samples.remove(0);
        }
    }

    /// Get latency statistics: (p50, p95, p99).
    pub fn latency_stats(&self) -> Option<(f64, f64, f64)> {
        if self.latency_samples.is_empty() {
            return None;
        }
        let mut sorted = self.latency_samples.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let p50 = sorted[n / 2];
        let p95 = sorted[(n as f64 * 0.95) as usize % n];
        let p99 = sorted[(n as f64 * 0.99) as usize % n];
        Some((p50, p95, p99))
    }

    /// Check whether a migration should be triggered autonomously.
    /// Returns `Some(SelectionResult)` if the current strategy is suboptimal.
    pub fn should_migrate(&self, vector_count: usize, dimensions: usize) -> Option<SelectionResult> {
        let current = self.current_strategy?;
        let rec = self.recommend(vector_count, dimensions);
        if rec.strategy != current && rec.confidence > 0.6 {
            Some(rec)
        } else {
            None
        }
    }

    /// Perform a full evaluation of the current index strategy against workload data.
    ///
    /// Returns an `IndexEvaluation` with a recommendation, latency analysis,
    /// memory analysis, and migration plan if applicable.
    pub fn evaluate(&self, vector_count: usize, dimensions: usize) -> IndexEvaluation {
        let rec = self.recommend(vector_count, dimensions);
        let latency = self.latency_stats();
        let current = self.current_strategy;

        let needs_migration = current
            .is_some_and(|c| c != rec.strategy && rec.confidence > 0.6);

        let migration_plan = if needs_migration {
            current.map(|from| self.migration_plan(from, rec.strategy, vector_count))
        } else {
            None
        };

        let latency_meets_target = match (latency, self.latency_target_ms) {
            (Some((p50, _, _)), Some(target)) => p50 <= target,
            _ => true,
        };

        let memory_estimate_bytes = vector_count.saturating_mul(dimensions).saturating_mul(4);
        let memory_within_budget = self.memory_budget_bytes
            .map_or(true, |budget| memory_estimate_bytes <= budget);

        IndexEvaluation {
            recommended: rec,
            current_strategy: current,
            needs_migration,
            migration_plan,
            latency_stats: latency,
            latency_meets_target,
            memory_estimate_bytes,
            memory_within_budget,
            observation_count: self.observations.len(),
        }
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

        // Memory budget constraint
        if let Some(budget) = self.memory_budget_bytes {
            let estimated_bytes = vector_count.saturating_mul(dimensions).saturating_mul(4);
            if estimated_bytes > budget {
                *scores.entry(IndexStrategy::HnswQuantized).or_default() += 3.0;
                *scores.entry(IndexStrategy::DiskAnn).or_default() += 2.0;
                rationale.push(format!(
                    "Exceeds memory budget ({:.0}MB > {:.0}MB): prefer quantized/disk",
                    estimated_bytes as f64 / 1_048_576.0,
                    budget as f64 / 1_048_576.0,
                ));
            }
        }

        // Latency target constraint
        if let Some(target) = self.latency_target_ms {
            if let Some((p50, _, _)) = self.latency_stats() {
                if p50 > target {
                    *scores.entry(IndexStrategy::Hnsw).or_default() += 1.5;
                    rationale.push(format!(
                        "Latency p50 {:.1}ms > target {:.1}ms: prefer low-latency index",
                        p50, target
                    ));
                }
            }
        }

        let mut sorted: Vec<(IndexStrategy, f32)> = scores.into_iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best = sorted.first().map_or(IndexStrategy::Hnsw, |s| s.0);
        let max_score = sorted.first().map_or(1.0, |s| s.1);
        let confidence = (max_score / 6.0).min(1.0);
        let mut params = HashMap::new();
        match best {
            IndexStrategy::Hnsw | IndexStrategy::HnswQuantized => {
                // Use the auto_tune infrastructure for HNSW parameters
                let tuning = crate::tuning::TuningConstraints::new(vector_count, dimensions);
                let tuned = crate::tuning::auto_tune(&tuning);
                params.insert("m".into(), tuned.config.m.to_string());
                params.insert("ef_construction".into(), tuned.config.ef_construction.to_string());
                params.insert("ef_search".into(), tuned.config.ef_search.to_string());
            }
            IndexStrategy::Ivf => {
                // Recommend nlist based on dataset size
                let nlist = ((vector_count as f64).sqrt() as usize).clamp(16, 65536);
                let nprobe = (nlist / 10).clamp(1, 256);
                params.insert("nlist".into(), nlist.to_string());
                params.insert("nprobe".into(), nprobe.to_string());
            }
            IndexStrategy::DiskAnn => {
                params.insert("max_degree".into(), "64".into());
                params.insert("l_build".into(), "100".into());
                params.insert("l_search".into(), "100".into());
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

// ── Cost Model ───────────────────────────────────────────────────────────────

/// Per-index cost model for memory and latency estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexCostModel {
    /// Strategy this model describes.
    pub strategy: IndexStrategy,
    /// Estimated memory in bytes for the given parameters.
    pub memory_bytes: usize,
    /// Estimated p50 search latency in milliseconds.
    pub p50_latency_ms: f64,
    /// Estimated p99 search latency in milliseconds.
    pub p99_latency_ms: f64,
    /// Estimated recall@10.
    pub estimated_recall: f64,
    /// Build time estimate in seconds.
    pub build_time_secs: f64,
}

/// Estimate costs for all index strategies given collection parameters.
pub fn estimate_costs(vector_count: usize, dimensions: usize) -> Vec<IndexCostModel> {
    let raw_bytes = vector_count.saturating_mul(dimensions).saturating_mul(4);
    let n = vector_count as f64;
    let d = dimensions as f64;

    vec![
        IndexCostModel {
            strategy: IndexStrategy::BruteForce,
            memory_bytes: raw_bytes,
            p50_latency_ms: n * d * 1e-9 * 1000.0, // O(n*d) comparison
            p99_latency_ms: n * d * 1.5e-9 * 1000.0,
            estimated_recall: 1.0,
            build_time_secs: 0.0,
        },
        IndexCostModel {
            strategy: IndexStrategy::Hnsw,
            // HNSW overhead: ~M*2 neighbors per node * 8 bytes each + vector storage
            memory_bytes: raw_bytes + vector_count * 16 * 2 * 8,
            p50_latency_ms: (n.ln() * d * 1e-8 * 1000.0).max(0.1),
            p99_latency_ms: (n.ln() * d * 2e-8 * 1000.0).max(0.2),
            estimated_recall: 0.95,
            build_time_secs: n * 1e-5,
        },
        IndexCostModel {
            strategy: IndexStrategy::HnswQuantized,
            // Quantized: ~4x compression on vectors
            memory_bytes: raw_bytes / 4 + vector_count * 16 * 2 * 8,
            p50_latency_ms: (n.ln() * d * 0.8e-8 * 1000.0).max(0.1),
            p99_latency_ms: (n.ln() * d * 1.6e-8 * 1000.0).max(0.2),
            estimated_recall: 0.90,
            build_time_secs: n * 1.5e-5,
        },
        IndexCostModel {
            strategy: IndexStrategy::Ivf,
            // IVF: centroids + inverted lists
            memory_bytes: raw_bytes.saturating_add((n.sqrt() as usize).saturating_mul(dimensions).saturating_mul(4)),
            p50_latency_ms: ((n / n.sqrt()) * d * 1e-9 * 1000.0).max(0.1),
            p99_latency_ms: ((n / n.sqrt()) * d * 2e-9 * 1000.0).max(0.2),
            estimated_recall: 0.85,
            build_time_secs: n * 2e-5,
        },
        IndexCostModel {
            strategy: IndexStrategy::DiskAnn,
            // DiskANN: compact in-memory graph + disk-resident vectors
            memory_bytes: vector_count * 64, // graph-only memory
            p50_latency_ms: (n.ln() * 0.5e-3).max(1.0), // includes disk I/O
            p99_latency_ms: (n.ln() * 1.5e-3).max(3.0),
            estimated_recall: 0.92,
            build_time_secs: n * 3e-5,
        },
    ]
}

/// Select the best strategy given constraints, using the cost model.
pub fn select_by_cost(
    vector_count: usize,
    dimensions: usize,
    memory_budget: Option<usize>,
    latency_target_ms: Option<f64>,
    min_recall: Option<f64>,
) -> SelectionResult {
    let costs = estimate_costs(vector_count, dimensions);
    let min_recall = min_recall.unwrap_or(0.80);

    let mut candidates: Vec<(IndexStrategy, f32, String)> = Vec::new();
    for cost in &costs {
        let mut score: f32 = 0.0;
        let mut reason = String::new();

        // Filter by memory budget
        if let Some(budget) = memory_budget {
            if cost.memory_bytes > budget {
                continue;
            }
        }

        // Filter by minimum recall
        if cost.estimated_recall < min_recall {
            continue;
        }

        // Score: prefer low latency
        score += (1.0 / (cost.p50_latency_ms + 0.01)) as f32;

        // Bonus for meeting latency target
        if let Some(target) = latency_target_ms {
            if cost.p50_latency_ms <= target {
                score += 2.0;
                reason = format!("meets latency target ({:.1}ms <= {:.1}ms)", cost.p50_latency_ms, target);
            }
        }

        // Bonus for high recall
        score += cost.estimated_recall as f32;

        // Penalty for build time
        score -= (cost.build_time_secs / 60.0) as f32 * 0.1;

        candidates.push((cost.strategy, score, reason));
    }

    if candidates.is_empty() {
        // Fallback: HNSW is always a reasonable default
        return SelectionResult {
            strategy: IndexStrategy::Hnsw,
            confidence: 0.3,
            rationale: vec!["No strategy meets all constraints; defaulting to HNSW".into()],
            alternatives: Vec::new(),
            suggested_params: HashMap::new(),
        };
    }

    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let max_score = candidates[0].1;
    let confidence = (max_score / 5.0).min(1.0);

    let mut suggested_params = HashMap::new();
    match candidates[0].0 {
        IndexStrategy::Hnsw | IndexStrategy::HnswQuantized => {
            let tuning = crate::tuning::TuningConstraints::new(vector_count, dimensions);
            let tuned = crate::tuning::auto_tune(&tuning);
            suggested_params.insert("m".into(), tuned.config.m.to_string());
            suggested_params.insert("ef_construction".into(), tuned.config.ef_construction.to_string());
            suggested_params.insert("ef_search".into(), tuned.config.ef_search.to_string());
        }
        _ => {}
    }

    SelectionResult {
        strategy: candidates[0].0,
        confidence,
        rationale: if candidates[0].2.is_empty() {
            vec![format!("Best cost-model fit for {} vectors × {} dims", vector_count, dimensions)]
        } else {
            vec![candidates[0].2.clone()]
        },
        alternatives: candidates.iter().skip(1).map(|(s, sc, _)| (*s, *sc)).collect(),
        suggested_params,
    }
}

// ── Workload Tracker ────────────────────────────────────────────────────────

/// Tracks query patterns over time for adaptive index decisions.
///
/// Attach this to a `Database` to automatically collect workload data
/// and periodically check whether an index migration is beneficial.
pub struct WorkloadTracker {
    selector: AdaptiveSelector,
    query_count: u64,
    insert_count: u64,
    evaluation_interval: u64,
    last_vector_count: usize,
    last_dimensions: usize,
}

impl WorkloadTracker {
    /// Create a new workload tracker that evaluates every `interval` queries.
    pub fn new(evaluation_interval: u64) -> Self {
        Self {
            selector: AdaptiveSelector::new(),
            query_count: 0,
            insert_count: 0,
            evaluation_interval,
            last_vector_count: 0,
            last_dimensions: 0,
        }
    }

    /// Record a search query with its latency and parameters.
    pub fn record_search(&mut self, dimensions: usize, k: usize, latency_ms: f64, had_filter: bool) {
        self.query_count += 1;
        self.last_dimensions = dimensions;
        self.selector.record_latency(latency_ms);

        let profile = if had_filter {
            WorkloadProfile::filtered_query(dimensions, k, 0.5)
        } else {
            WorkloadProfile::point_query(dimensions, k)
        };
        self.selector.observe_query(profile);
    }

    /// Record an insert operation.
    pub fn record_insert(&mut self, vector_count: usize, dimensions: usize) {
        self.insert_count += 1;
        self.last_vector_count = vector_count;
        self.last_dimensions = dimensions;
    }

    /// Check if it's time for an evaluation (every `evaluation_interval` queries).
    pub fn should_evaluate(&self) -> bool {
        self.query_count > 0 && self.query_count % self.evaluation_interval == 0
    }

    /// Perform an evaluation and return it if migration is recommended.
    pub fn check_migration(&self) -> Option<IndexEvaluation> {
        if self.last_vector_count == 0 || self.last_dimensions == 0 {
            return None;
        }
        let eval = self.selector.evaluate(self.last_vector_count, self.last_dimensions);
        if eval.needs_migration { Some(eval) } else { None }
    }

    /// Get current selector reference.
    pub fn selector(&self) -> &AdaptiveSelector {
        &self.selector
    }

    /// Get mutable selector reference.
    pub fn selector_mut(&mut self) -> &mut AdaptiveSelector {
        &mut self.selector
    }

    /// Total queries tracked.
    pub fn total_queries(&self) -> u64 {
        self.query_count
    }
}

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

    #[test]
    fn test_memory_budget_constraint() {
        let s = AdaptiveSelector::new().with_memory_budget(100 * 1024 * 1024); // 100MB
        // 1M * 384 * 4 bytes = ~1.5GB → exceeds budget
        let r = s.recommend(1_000_000, 384);
        assert!(matches!(
            r.strategy,
            IndexStrategy::HnswQuantized | IndexStrategy::DiskAnn
        ));
    }

    #[test]
    fn test_latency_tracking() {
        let mut s = AdaptiveSelector::new();
        for i in 0..100 {
            s.record_latency(i as f64 * 0.1);
        }
        let (p50, p95, p99) = s.latency_stats().unwrap();
        assert!(p50 > 0.0);
        assert!(p95 >= p50);
        assert!(p99 >= p50);
    }

    #[test]
    fn test_autonomous_should_migrate() {
        let mut s = AdaptiveSelector::new();
        s.set_current(IndexStrategy::BruteForce);
        // Very large dataset should strongly recommend migrating away from BruteForce
        let rec = s.should_migrate(5_000_000, 384);
        assert!(rec.is_some());
        assert_ne!(rec.unwrap().strategy, IndexStrategy::BruteForce);
    }

    #[test]
    fn test_no_migrate_when_optimal() {
        let mut s = AdaptiveSelector::new();
        s.set_current(IndexStrategy::Hnsw);
        // Medium dataset is good for HNSW
        let rec = s.should_migrate(50_000, 128);
        assert!(rec.is_none());
    }

    #[test]
    fn test_cost_model_estimates() {
        let costs = estimate_costs(100_000, 384);
        assert_eq!(costs.len(), 5);
        // BruteForce should have perfect recall
        let bf = costs.iter().find(|c| c.strategy == IndexStrategy::BruteForce).unwrap();
        assert!((bf.estimated_recall - 1.0).abs() < f64::EPSILON);
        assert_eq!(bf.build_time_secs, 0.0);
        // DiskANN should have lowest memory
        let da = costs.iter().find(|c| c.strategy == IndexStrategy::DiskAnn).unwrap();
        assert!(da.memory_bytes < bf.memory_bytes);
    }

    #[test]
    fn test_select_by_cost_memory_constraint() {
        // Tight memory budget should prefer DiskANN or quantized
        let r = select_by_cost(1_000_000, 384, Some(200 * 1024 * 1024), None, None);
        assert!(matches!(
            r.strategy,
            IndexStrategy::DiskAnn | IndexStrategy::HnswQuantized
        ));
    }

    #[test]
    fn test_select_by_cost_small_dataset() {
        let r = select_by_cost(1_000, 128, None, None, None);
        // Small dataset — brute force has perfect recall and no build cost
        assert_eq!(r.strategy, IndexStrategy::BruteForce);
    }
}
