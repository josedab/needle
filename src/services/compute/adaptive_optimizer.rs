//! Adaptive Query Optimizer
//!
//! Cost-based query planner that automatically selects between HNSW, IVF, DiskANN,
//! or brute-force based on collection size, filter selectivity, and query history.
//! Provides EXPLAIN plans for transparency.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::Database;
//! use needle::services::adaptive_optimizer::{
//!     AdaptiveOptimizer, OptimizerConfig, QueryProfile,
//! };
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let mut optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
//!
//! // Record query feedback to learn cost models
//! let profile = QueryProfile {
//!     collection_size: 100_000,
//!     dimensions: 128,
//!     k: 10,
//!     filter_selectivity: Some(0.1),
//!     actual_latency_us: 3200,
//!     strategy_used: "hnsw".into(),
//!     recall_estimate: Some(0.95),
//! };
//! optimizer.record_feedback(profile);
//!
//! // Plan the optimal strategy
//! let plan = optimizer.plan(100_000, 128, 10, Some(0.1));
//! println!("Strategy: {}", plan.strategy);
//! println!("Estimated latency: {}μs", plan.estimated_latency_us);
//! ```

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

// ── Configuration ────────────────────────────────────────────────────────────

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Minimum collection size to consider IVF over HNSW.
    pub ivf_threshold: usize,
    /// Minimum collection size to consider DiskANN.
    pub diskann_threshold: usize,
    /// Maximum collection size for brute-force search.
    pub brute_force_max: usize,
    /// Filter selectivity below which pre-filtering is preferred.
    pub prefilter_selectivity_threshold: f64,
    /// Number of recent feedback samples to retain.
    pub feedback_window: usize,
    /// Weight for historical latency in cost estimation (0.0–1.0).
    pub history_weight: f64,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            ivf_threshold: 100_000,
            diskann_threshold: 1_000_000,
            brute_force_max: 5_000,
            prefilter_selectivity_threshold: 0.05,
            feedback_window: 1000,
            history_weight: 0.3,
        }
    }
}

// ── Strategy ─────────────────────────────────────────────────────────────────

/// Index strategy selected by the optimizer.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Strategy {
    /// Brute-force linear scan (best for tiny collections or very selective filters).
    BruteForce,
    /// HNSW approximate nearest neighbor (default for most workloads).
    Hnsw,
    /// IVF inverted file index (large collections with moderate recall requirements).
    Ivf,
    /// DiskANN on-disk index (very large collections exceeding memory budget).
    DiskAnn,
    /// Two-phase: pre-filter metadata, then ANN on candidates.
    FilteredHnsw,
    /// Hybrid: combine HNSW + BM25 with RRF fusion.
    HybridRrf,
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BruteForce => write!(f, "brute_force"),
            Self::Hnsw => write!(f, "hnsw"),
            Self::Ivf => write!(f, "ivf"),
            Self::DiskAnn => write!(f, "diskann"),
            Self::FilteredHnsw => write!(f, "filtered_hnsw"),
            Self::HybridRrf => write!(f, "hybrid_rrf"),
        }
    }
}

/// Filter execution strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterStrategy {
    /// No filter applied.
    None,
    /// Apply filter after ANN search (post-filter).
    PostFilter,
    /// Apply filter before ANN search (pre-filter).
    PreFilter,
    /// Hybrid: pre-filter with bloom sketch, post-filter for accuracy.
    TwoPhase,
}

impl fmt::Display for FilterStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::PostFilter => write!(f, "post_filter"),
            Self::PreFilter => write!(f, "pre_filter"),
            Self::TwoPhase => write!(f, "two_phase"),
        }
    }
}

// ── Query Plan ───────────────────────────────────────────────────────────────

/// An optimized query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// The selected index strategy.
    pub strategy: Strategy,
    /// How filters will be applied.
    pub filter_strategy: FilterStrategy,
    /// Estimated latency in microseconds.
    pub estimated_latency_us: u64,
    /// Estimated recall (0.0–1.0).
    pub estimated_recall: f64,
    /// Recommended ef_search override (if applicable).
    pub recommended_ef_search: Option<usize>,
    /// Recommended nprobe (for IVF).
    pub recommended_nprobe: Option<usize>,
    /// Explanation steps for the decision.
    pub explain_steps: Vec<ExplainStep>,
    /// Cost breakdown for each considered strategy.
    pub cost_breakdown: Vec<StrategyCost>,
    /// Confidence in this plan (0.0–1.0, higher with more feedback data).
    pub confidence: f64,
}

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Query Plan ===")?;
        writeln!(f, "Strategy:          {}", self.strategy)?;
        writeln!(f, "Filter:            {}", self.filter_strategy)?;
        writeln!(f, "Est. Latency:      {}μs", self.estimated_latency_us)?;
        writeln!(
            f,
            "Est. Recall:       {:.1}%",
            self.estimated_recall * 100.0
        )?;
        writeln!(f, "Confidence:        {:.1}%", self.confidence * 100.0)?;
        if let Some(ef) = self.recommended_ef_search {
            writeln!(f, "Rec. ef_search:    {ef}")?;
        }
        if let Some(nprobe) = self.recommended_nprobe {
            writeln!(f, "Rec. nprobe:       {nprobe}")?;
        }
        writeln!(f, "\n--- Explain ---")?;
        for step in &self.explain_steps {
            writeln!(f, "  [{:?}] {}", step.phase, step.detail)?;
        }
        writeln!(f, "\n--- Cost Breakdown ---")?;
        for cost in &self.cost_breakdown {
            writeln!(
                f,
                "  {:15} → {:>8}μs (recall={:.1}%) {}",
                cost.strategy.to_string(),
                cost.estimated_latency_us,
                cost.estimated_recall * 100.0,
                if cost.selected { "← SELECTED" } else { "" }
            )?;
        }
        Ok(())
    }
}

/// A step in the query plan explanation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainStep {
    /// Phase of the optimization.
    pub phase: OptPhase,
    /// Human-readable explanation.
    pub detail: String,
}

/// Optimization phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptPhase {
    /// Gathering collection statistics.
    Statistics,
    /// Evaluating index strategies.
    CostEstimation,
    /// Selecting filter execution order.
    FilterPlanning,
    /// Applying learned adjustments from history.
    HistoryAdjustment,
    /// Final strategy selection.
    Selection,
}

/// Cost estimate for a candidate strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyCost {
    /// The candidate strategy.
    pub strategy: Strategy,
    /// Estimated latency in microseconds.
    pub estimated_latency_us: u64,
    /// Estimated recall.
    pub estimated_recall: f64,
    /// Whether this strategy was selected.
    pub selected: bool,
}

// ── Query Feedback ───────────────────────────────────────────────────────────

/// Feedback from an executed query for online learning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryProfile {
    /// Number of vectors in the collection at query time.
    pub collection_size: usize,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Number of results requested.
    pub k: usize,
    /// Filter selectivity (fraction of collection matching filter), if any.
    pub filter_selectivity: Option<f64>,
    /// Actual measured latency in microseconds.
    pub actual_latency_us: u64,
    /// Strategy that was used.
    pub strategy_used: String,
    /// Measured or estimated recall, if available.
    pub recall_estimate: Option<f64>,
}

// ── Cost Model ───────────────────────────────────────────────────────────────

/// Analytical cost model parameters learned from feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CostModel {
    /// Base latency per strategy (microseconds).
    base_latency: HashMap<String, f64>,
    /// Latency scaling factor per log2(N) for each strategy.
    scale_factor: HashMap<String, f64>,
    /// Recall estimate per strategy.
    recall_estimate: HashMap<String, f64>,
    /// Number of feedback samples per strategy.
    sample_count: HashMap<String, usize>,
}

impl Default for CostModel {
    fn default() -> Self {
        let mut base = HashMap::new();
        let mut scale = HashMap::new();
        let mut recall = HashMap::new();

        // Default analytical estimates
        base.insert("brute_force".into(), 100.0);
        scale.insert("brute_force".into(), 500.0); // linear scale
        recall.insert("brute_force".into(), 1.0);

        base.insert("hnsw".into(), 500.0);
        scale.insert("hnsw".into(), 200.0); // log scale
        recall.insert("hnsw".into(), 0.95);

        base.insert("ivf".into(), 800.0);
        scale.insert("ivf".into(), 150.0);
        recall.insert("ivf".into(), 0.90);

        base.insert("diskann".into(), 2000.0);
        scale.insert("diskann".into(), 100.0);
        recall.insert("diskann".into(), 0.92);

        base.insert("filtered_hnsw".into(), 700.0);
        scale.insert("filtered_hnsw".into(), 250.0);
        recall.insert("filtered_hnsw".into(), 0.93);

        Self {
            base_latency: base,
            scale_factor: scale,
            recall_estimate: recall,
            sample_count: HashMap::new(),
        }
    }
}

impl CostModel {
    fn estimate_latency(&self, strategy: &str, collection_size: usize, dimensions: usize) -> f64 {
        let base = self.base_latency.get(strategy).copied().unwrap_or(1000.0);
        let scale = self.scale_factor.get(strategy).copied().unwrap_or(200.0);

        let n = collection_size.max(1) as f64;
        let dim_factor = (dimensions as f64 / 128.0).sqrt();

        if strategy == "brute_force" {
            // Linear scan: O(n * d)
            base + (n / 1000.0) * scale * dim_factor
        } else {
            // Log-scale for index-based strategies
            base + n.log2() * scale * dim_factor
        }
    }

    fn estimate_recall(&self, strategy: &str) -> f64 {
        self.recall_estimate.get(strategy).copied().unwrap_or(0.90)
    }

    fn update(&mut self, profile: &QueryProfile) {
        let strategy = &profile.strategy_used;
        let count = self.sample_count.entry(strategy.clone()).or_insert(0);
        *count += 1;
        let n = *count as f64;
        let alpha = 1.0 / n.min(50.0); // Exponential moving average

        // Update base latency
        let predicted =
            self.estimate_latency(strategy, profile.collection_size, profile.dimensions);
        let actual = profile.actual_latency_us as f64;
        let error = actual - predicted;

        if let Some(base) = self.base_latency.get_mut(strategy) {
            *base += alpha * error * 0.5;
            *base = base.max(10.0);
        }

        if let Some(recall) = profile.recall_estimate {
            if let Some(r) = self.recall_estimate.get_mut(strategy) {
                *r = *r * (1.0 - alpha) + recall * alpha;
            }
        }
    }
}

// ── Adaptive Optimizer ───────────────────────────────────────────────────────

/// Adaptive query optimizer that learns from query feedback.
pub struct AdaptiveOptimizer {
    config: OptimizerConfig,
    cost_model: CostModel,
    feedback_history: Vec<QueryProfile>,
}

impl AdaptiveOptimizer {
    /// Create a new optimizer with the given configuration.
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            config,
            cost_model: CostModel::default(),
            feedback_history: Vec::new(),
        }
    }

    /// Record feedback from an executed query to improve future plans.
    pub fn record_feedback(&mut self, profile: QueryProfile) {
        self.cost_model.update(&profile);
        self.feedback_history.push(profile);

        // Trim history
        if self.feedback_history.len() > self.config.feedback_window {
            self.feedback_history
                .drain(0..self.feedback_history.len() - self.config.feedback_window);
        }
    }

    /// Generate an optimized query plan.
    pub fn plan(
        &self,
        collection_size: usize,
        dimensions: usize,
        k: usize,
        filter_selectivity: Option<f64>,
    ) -> QueryPlan {
        let mut explain = Vec::new();
        let mut costs = Vec::new();

        explain.push(ExplainStep {
            phase: OptPhase::Statistics,
            detail: format!(
                "Collection: {} vectors, {} dims, k={}",
                collection_size, dimensions, k
            ),
        });

        if let Some(sel) = filter_selectivity {
            explain.push(ExplainStep {
                phase: OptPhase::Statistics,
                detail: format!("Filter selectivity: {:.1}%", sel * 100.0),
            });
        }

        // Evaluate candidate strategies
        let candidates = self.candidate_strategies(collection_size, filter_selectivity);

        explain.push(ExplainStep {
            phase: OptPhase::CostEstimation,
            detail: format!("Evaluating {} candidate strategies", candidates.len()),
        });

        for strategy in &candidates {
            let strategy_str = strategy.to_string();
            let mut latency =
                self.cost_model
                    .estimate_latency(&strategy_str, collection_size, dimensions);
            let recall = self.cost_model.estimate_recall(&strategy_str);

            // Adjust for filter overhead
            if let Some(sel) = filter_selectivity {
                match strategy {
                    Strategy::FilteredHnsw => {
                        // Pre-filter cost scales with selectivity
                        latency *= 1.0 + (1.0 - sel) * 0.5;
                    }
                    Strategy::BruteForce => {
                        // Brute-force benefits from selective filters
                        latency *= sel.max(0.01);
                    }
                    _ => {
                        // Post-filter overhead
                        latency *= 1.0 + (1.0 - sel) * 0.2;
                    }
                }
            }

            // Adjust for k
            if k > 100 {
                latency *= 1.0 + (k as f64 / 100.0).ln();
            }

            costs.push(StrategyCost {
                strategy: strategy.clone(),
                estimated_latency_us: latency as u64,
                estimated_recall: recall,
                selected: false,
            });
        }

        // Select: minimize latency while maintaining recall >= 0.85
        costs.sort_by(|a, b| {
            // Prefer strategies with recall >= 0.85, then sort by latency
            let a_viable = a.estimated_recall >= 0.85;
            let b_viable = b.estimated_recall >= 0.85;
            match (a_viable, b_viable) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.estimated_latency_us.cmp(&b.estimated_latency_us),
            }
        });

        let selected_idx = 0;
        costs[selected_idx].selected = true;
        let selected = &costs[selected_idx];

        explain.push(ExplainStep {
            phase: OptPhase::Selection,
            detail: format!(
                "Selected {} (est. {}μs, recall={:.1}%)",
                selected.strategy,
                selected.estimated_latency_us,
                selected.estimated_recall * 100.0,
            ),
        });

        // Determine filter strategy
        let filter_strategy = self.plan_filter(filter_selectivity, &selected.strategy);
        if filter_selectivity.is_some() {
            explain.push(ExplainStep {
                phase: OptPhase::FilterPlanning,
                detail: format!("Filter strategy: {filter_strategy}"),
            });
        }

        // Recommend parameters
        let recommended_ef_search = self.recommend_ef_search(collection_size, k);
        let recommended_nprobe = if selected.strategy == Strategy::Ivf {
            Some(self.recommend_nprobe(collection_size))
        } else {
            None
        };

        // Confidence based on feedback volume
        let strategy_str = selected.strategy.to_string();
        let sample_count = self
            .cost_model
            .sample_count
            .get(&strategy_str)
            .copied()
            .unwrap_or(0);
        let confidence = (sample_count as f64 / 50.0).min(1.0) * 0.7 + 0.3;

        if sample_count > 0 {
            explain.push(ExplainStep {
                phase: OptPhase::HistoryAdjustment,
                detail: format!(
                    "Cost model calibrated from {} historical samples",
                    sample_count
                ),
            });
        }

        QueryPlan {
            strategy: selected.strategy.clone(),
            filter_strategy,
            estimated_latency_us: selected.estimated_latency_us,
            estimated_recall: selected.estimated_recall,
            recommended_ef_search,
            recommended_nprobe,
            explain_steps: explain,
            cost_breakdown: costs,
            confidence,
        }
    }

    /// Generate a human-readable EXPLAIN string.
    pub fn explain(
        &self,
        collection_size: usize,
        dimensions: usize,
        k: usize,
        filter_selectivity: Option<f64>,
    ) -> String {
        self.plan(collection_size, dimensions, k, filter_selectivity)
            .to_string()
    }

    /// Get the number of feedback samples collected.
    pub fn feedback_count(&self) -> usize {
        self.feedback_history.len()
    }

    /// Reset learned cost model to defaults.
    pub fn reset(&mut self) {
        self.cost_model = CostModel::default();
        self.feedback_history.clear();
    }

    fn candidate_strategies(
        &self,
        collection_size: usize,
        filter_selectivity: Option<f64>,
    ) -> Vec<Strategy> {
        let mut candidates = Vec::new();

        if collection_size <= self.config.brute_force_max {
            candidates.push(Strategy::BruteForce);
        }

        candidates.push(Strategy::Hnsw);

        if collection_size >= self.config.ivf_threshold {
            candidates.push(Strategy::Ivf);
        }

        if collection_size >= self.config.diskann_threshold {
            candidates.push(Strategy::DiskAnn);
        }

        if let Some(sel) = filter_selectivity {
            if sel < self.config.prefilter_selectivity_threshold {
                // Very selective filter → brute-force on filtered set may win
                if !candidates.contains(&Strategy::BruteForce) {
                    candidates.push(Strategy::BruteForce);
                }
            }
            candidates.push(Strategy::FilteredHnsw);
        }

        candidates
    }

    fn plan_filter(&self, selectivity: Option<f64>, strategy: &Strategy) -> FilterStrategy {
        match selectivity {
            None => FilterStrategy::None,
            Some(sel) if sel < self.config.prefilter_selectivity_threshold => {
                FilterStrategy::PreFilter
            }
            Some(sel) if sel > 0.5 => FilterStrategy::PostFilter,
            Some(_) => {
                if *strategy == Strategy::FilteredHnsw {
                    FilterStrategy::TwoPhase
                } else {
                    FilterStrategy::PostFilter
                }
            }
        }
    }

    fn recommend_ef_search(&self, collection_size: usize, k: usize) -> Option<usize> {
        let base_ef = k.max(50);
        let scale = if collection_size > 1_000_000 {
            2.0
        } else if collection_size > 100_000 {
            1.5
        } else {
            1.0
        };
        Some(((base_ef as f64) * scale) as usize)
    }

    fn recommend_nprobe(&self, collection_size: usize) -> usize {
        let n_clusters = (collection_size as f64).sqrt() as usize;
        // Probe ~10% of clusters, minimum 1, maximum 64
        (n_clusters / 10).max(1).min(64)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_collection_selects_brute_force() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(1000, 128, 10, None);
        assert_eq!(plan.strategy, Strategy::BruteForce);
        assert_eq!(plan.filter_strategy, FilterStrategy::None);
    }

    #[test]
    fn test_medium_collection_selects_hnsw() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(50_000, 128, 10, None);
        assert_eq!(plan.strategy, Strategy::Hnsw);
    }

    #[test]
    fn test_large_collection_considers_ivf() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(500_000, 128, 10, None);
        // IVF or HNSW depending on cost model
        assert!(
            plan.strategy == Strategy::Hnsw || plan.strategy == Strategy::Ivf,
            "Expected HNSW or IVF, got {:?}",
            plan.strategy
        );
    }

    #[test]
    fn test_selective_filter_triggers_prefilter() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(50_000, 128, 10, Some(0.01));
        assert_eq!(plan.filter_strategy, FilterStrategy::PreFilter);
    }

    #[test]
    fn test_broad_filter_triggers_postfilter() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(50_000, 128, 10, Some(0.8));
        assert_eq!(plan.filter_strategy, FilterStrategy::PostFilter);
    }

    #[test]
    fn test_feedback_improves_model() {
        let mut optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());

        // Simulate HNSW being faster than expected
        for _ in 0..20 {
            optimizer.record_feedback(QueryProfile {
                collection_size: 50_000,
                dimensions: 128,
                k: 10,
                filter_selectivity: None,
                actual_latency_us: 500,
                strategy_used: "hnsw".into(),
                recall_estimate: Some(0.97),
            });
        }

        let plan = optimizer.plan(50_000, 128, 10, None);
        assert!(plan.confidence > 0.5);
        assert_eq!(plan.strategy, Strategy::Hnsw);
    }

    #[test]
    fn test_explain_output_is_readable() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let explain = optimizer.explain(100_000, 256, 20, Some(0.1));
        assert!(explain.contains("Query Plan"));
        assert!(explain.contains("Strategy:"));
        assert!(explain.contains("Cost Breakdown"));
    }

    #[test]
    fn test_ef_search_recommendation_scales_with_collection() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let small_plan = optimizer.plan(10_000, 128, 10, None);
        let large_plan = optimizer.plan(5_000_000, 128, 10, None);
        assert!(
            large_plan.recommended_ef_search.unwrap() >= small_plan.recommended_ef_search.unwrap()
        );
    }

    #[test]
    fn test_nprobe_recommendation_for_ivf() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(500_000, 128, 10, None);
        if plan.strategy == Strategy::Ivf {
            assert!(plan.recommended_nprobe.is_some());
            assert!(plan.recommended_nprobe.unwrap() >= 1);
        }
    }

    #[test]
    fn test_plan_display() {
        let optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        let plan = optimizer.plan(50_000, 128, 10, Some(0.05));
        let display = format!("{plan}");
        assert!(!display.is_empty());
        assert!(display.contains("SELECTED"));
    }

    #[test]
    fn test_reset_clears_feedback() {
        let mut optimizer = AdaptiveOptimizer::new(OptimizerConfig::default());
        optimizer.record_feedback(QueryProfile {
            collection_size: 1000,
            dimensions: 128,
            k: 10,
            filter_selectivity: None,
            actual_latency_us: 100,
            strategy_used: "brute_force".into(),
            recall_estimate: Some(1.0),
        });
        assert_eq!(optimizer.feedback_count(), 1);
        optimizer.reset();
        assert_eq!(optimizer.feedback_count(), 0);
    }
}
