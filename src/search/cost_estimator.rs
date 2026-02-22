//! Query Cost Estimator & Optimizer
//!
//! SQL-style EXPLAIN ANALYZE for vector queries showing estimated I/O, memory,
//! candidate set sizes, and index selection rationale. Automatic index selection
//! based on query patterns and collected statistics.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::search::cost_estimator::{
//!     CostEstimator, CollectionStatistics, QueryPlan,
//!     IndexChoice, CostBreakdown,
//! };
//!
//! // Collect statistics from a collection
//! let stats = CollectionStatistics::new(1_000_000, 384, 0.1);
//!
//! let estimator = CostEstimator::new();
//!
//! // Estimate cost of a search query
//! let plan = estimator.plan(&stats, 10, Some(0.05));
//! println!("Chosen index: {:?}", plan.index_choice);
//! println!("Estimated latency: {}ms", plan.cost.estimated_latency_ms);
//! println!("Estimated memory: {}MB", plan.cost.estimated_memory_mb);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

// ── Collection Statistics ────────────────────────────────────────────────────

/// Statistics about a collection, used for query cost estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatistics {
    /// Total number of vectors.
    pub total_vectors: usize,
    /// Vector dimensionality.
    pub dimensions: usize,
    /// Fraction of vectors that have been deleted (0.0–1.0).
    pub deletion_ratio: f32,
    /// Average metadata size in bytes.
    pub avg_metadata_bytes: usize,
    /// HNSW index parameters (if available).
    pub hnsw_params: Option<HnswStats>,
    /// Filter selectivity histogram (optional).
    pub filter_selectivity: Option<Vec<SelectivityBucket>>,
}

impl CollectionStatistics {
    /// Create basic statistics.
    pub fn new(total_vectors: usize, dimensions: usize, deletion_ratio: f32) -> Self {
        Self {
            total_vectors,
            dimensions,
            deletion_ratio,
            avg_metadata_bytes: 256,
            hnsw_params: Some(HnswStats::default()),
            filter_selectivity: None,
        }
    }

    /// Set HNSW parameters.
    #[must_use]
    pub fn with_hnsw(mut self, params: HnswStats) -> Self {
        self.hnsw_params = Some(params);
        self
    }

    /// Active (non-deleted) vector count.
    pub fn active_vectors(&self) -> usize {
        ((1.0 - self.deletion_ratio) * self.total_vectors as f32) as usize
    }

    /// Estimated memory usage in bytes.
    pub fn estimated_memory_bytes(&self) -> usize {
        let vector_bytes = self.total_vectors * self.dimensions * 4; // f32
        let metadata_bytes = self.total_vectors * self.avg_metadata_bytes;
        let hnsw_bytes = if let Some(ref h) = self.hnsw_params {
            self.total_vectors * h.m * 2 * 8 // edges (avg 2*M per node, 8 bytes each)
        } else {
            0
        };
        vector_bytes + metadata_bytes + hnsw_bytes
    }
}

/// HNSW index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswStats {
    /// M parameter (connections per layer).
    pub m: usize,
    /// ef_construction parameter.
    pub ef_construction: usize,
    /// ef_search parameter.
    pub ef_search: usize,
    /// Number of layers in the graph.
    pub num_layers: usize,
}

impl Default for HnswStats {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
            num_layers: 4,
        }
    }
}

/// Selectivity bucket for filter cost estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectivityBucket {
    /// Field name.
    pub field: String,
    /// Estimated selectivity (fraction of vectors matching).
    pub selectivity: f32,
    /// Number of distinct values.
    pub cardinality: usize,
}

// ── Index Choice ─────────────────────────────────────────────────────────────

/// Which index strategy the optimizer chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexChoice {
    /// HNSW graph traversal.
    Hnsw,
    /// IVF (Inverted File Index).
    Ivf,
    /// Brute-force linear scan.
    BruteForce,
    /// HNSW with pre-filtering.
    HnswPreFilter,
    /// HNSW with post-filtering (over-fetch + filter).
    HnswPostFilter,
    /// Quantized search with re-ranking.
    QuantizedRerank,
}

impl fmt::Display for IndexChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hnsw => write!(f, "HNSW"),
            Self::Ivf => write!(f, "IVF"),
            Self::BruteForce => write!(f, "BruteForce"),
            Self::HnswPreFilter => write!(f, "HNSW+PreFilter"),
            Self::HnswPostFilter => write!(f, "HNSW+PostFilter"),
            Self::QuantizedRerank => write!(f, "Quantized+Rerank"),
        }
    }
}

// ── Cost Breakdown ───────────────────────────────────────────────────────────

/// Detailed cost breakdown for a query plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostBreakdown {
    /// Estimated latency in milliseconds.
    pub estimated_latency_ms: f64,
    /// Estimated memory usage in megabytes.
    pub estimated_memory_mb: f64,
    /// Estimated number of distance computations.
    pub distance_computations: usize,
    /// Estimated number of nodes visited (HNSW).
    pub nodes_visited: usize,
    /// Estimated I/O operations (for disk-based indices).
    pub io_operations: usize,
    /// Estimated candidate set size before final ranking.
    pub candidate_set_size: usize,
    /// Filter evaluation cost (if applicable).
    pub filter_cost: Option<FilterCost>,
}

/// Cost of evaluating a metadata filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCost {
    /// Estimated selectivity (0.0–1.0).
    pub selectivity: f32,
    /// Whether pre-filtering or post-filtering is more efficient.
    pub strategy: FilterStrategy,
    /// Estimated vectors evaluated by the filter.
    pub evaluations: usize,
}

/// Pre-filter vs post-filter strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterStrategy {
    /// Filter during HNSW traversal.
    PreFilter,
    /// Over-fetch then filter.
    PostFilter,
    /// No filter needed.
    None,
}

// ── Query Plan ───────────────────────────────────────────────────────────────

/// The optimizer's chosen query plan with cost estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Chosen index strategy.
    pub index_choice: IndexChoice,
    /// Cost breakdown.
    pub cost: CostBreakdown,
    /// Rationale for the choice.
    pub rationale: Vec<String>,
    /// Alternative plans considered.
    pub alternatives: Vec<AlternativePlan>,
}

impl fmt::Display for QueryPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Query Plan: {}", self.index_choice)?;
        writeln!(
            f,
            "  Estimated latency: {:.2}ms",
            self.cost.estimated_latency_ms
        )?;
        writeln!(
            f,
            "  Distance computations: {}",
            self.cost.distance_computations
        )?;
        writeln!(f, "  Nodes visited: {}", self.cost.nodes_visited)?;
        writeln!(
            f,
            "  Candidate set: {}",
            self.cost.candidate_set_size
        )?;
        writeln!(f, "  Rationale:")?;
        for r in &self.rationale {
            writeln!(f, "    - {}", r)?;
        }
        if !self.alternatives.is_empty() {
            writeln!(f, "  Alternatives considered:")?;
            for alt in &self.alternatives {
                writeln!(
                    f,
                    "    {} ({:.2}ms, rejected: {})",
                    alt.index_choice, alt.estimated_latency_ms, alt.rejection_reason
                )?;
            }
        }
        Ok(())
    }
}

/// An alternative plan that was considered but not chosen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativePlan {
    /// Index strategy.
    pub index_choice: IndexChoice,
    /// Estimated latency.
    pub estimated_latency_ms: f64,
    /// Why it was rejected.
    pub rejection_reason: String,
}

// ── Cost Estimator ───────────────────────────────────────────────────────────

/// Query cost estimator and optimizer.
///
/// Analyzes collection statistics and query parameters to choose the optimal
/// index strategy and estimate execution costs.
pub struct CostEstimator {
    /// Cost per distance computation in microseconds.
    distance_cost_us: f64,
    /// Cost per metadata filter evaluation in microseconds.
    filter_cost_us: f64,
    /// Cost per I/O operation in microseconds.
    io_cost_us: f64,
    /// Threshold below which brute-force is preferred.
    brute_force_threshold: usize,
}

impl Default for CostEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl CostEstimator {
    /// Create a new cost estimator with default parameters.
    pub fn new() -> Self {
        Self {
            distance_cost_us: 0.5,   // 0.5µs per distance computation (SIMD)
            filter_cost_us: 0.1,     // 0.1µs per filter evaluation
            io_cost_us: 100.0,       // 100µs per disk I/O
            brute_force_threshold: 5_000,
        }
    }

    /// Plan a query execution.
    ///
    /// - `stats`: Collection statistics
    /// - `k`: Number of results requested
    /// - `filter_selectivity`: Optional filter selectivity (0.0–1.0, fraction matching)
    pub fn plan(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        filter_selectivity: Option<f32>,
    ) -> QueryPlan {
        let mut plans = Vec::new();

        // Estimate HNSW cost
        plans.push(self.estimate_hnsw(stats, k, filter_selectivity));

        // Estimate brute-force cost
        plans.push(self.estimate_brute_force(stats, k, filter_selectivity));

        // Estimate HNSW + post-filter if filter is present
        if let Some(sel) = filter_selectivity {
            if sel < 0.5 {
                plans.push(self.estimate_hnsw_post_filter(stats, k, sel));
            }
            if sel > 0.01 {
                plans.push(self.estimate_hnsw_pre_filter(stats, k, sel));
            }
        }

        // Sort by estimated latency
        plans.sort_by(|a, b| {
            a.cost
                .estimated_latency_ms
                .partial_cmp(&b.cost.estimated_latency_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let best = plans.remove(0);
        let alternatives: Vec<AlternativePlan> = plans
            .iter()
            .map(|p| AlternativePlan {
                index_choice: p.index_choice,
                estimated_latency_ms: p.cost.estimated_latency_ms,
                rejection_reason: format!(
                    "Higher estimated latency ({:.2}ms vs {:.2}ms)",
                    p.cost.estimated_latency_ms, best.cost.estimated_latency_ms
                ),
            })
            .collect();

        QueryPlan {
            index_choice: best.index_choice,
            cost: best.cost,
            rationale: best.rationale,
            alternatives,
        }
    }

    /// Estimate the cost for a specific index choice.
    pub fn estimate_cost(
        &self,
        stats: &CollectionStatistics,
        index: IndexChoice,
        k: usize,
        filter_selectivity: Option<f32>,
    ) -> CostBreakdown {
        match index {
            IndexChoice::Hnsw => self.estimate_hnsw(stats, k, filter_selectivity).cost,
            IndexChoice::BruteForce => {
                self.estimate_brute_force(stats, k, filter_selectivity).cost
            }
            IndexChoice::HnswPostFilter => {
                let sel = filter_selectivity.unwrap_or(1.0);
                self.estimate_hnsw_post_filter(stats, k, sel).cost
            }
            IndexChoice::HnswPreFilter => {
                let sel = filter_selectivity.unwrap_or(1.0);
                self.estimate_hnsw_pre_filter(stats, k, sel).cost
            }
            _ => self.estimate_hnsw(stats, k, filter_selectivity).cost,
        }
    }

    // ── Internal cost models ─────────────────────────────────────────────────

    fn estimate_hnsw(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        _filter_selectivity: Option<f32>,
    ) -> QueryPlan {
        let n = stats.active_vectors();
        let hnsw = stats.hnsw_params.as_ref().cloned().unwrap_or_default();

        // HNSW visits ~ef_search * log2(n) nodes
        let layers = (n as f64).log2().ceil() as usize;
        let nodes_visited = hnsw.ef_search * layers.max(1);
        let distance_computations = nodes_visited * hnsw.m;
        let candidate_set = hnsw.ef_search.max(k);

        let latency = (distance_computations as f64 * self.distance_cost_us) / 1000.0;
        let memory_mb = (candidate_set * stats.dimensions * 4) as f64 / (1024.0 * 1024.0);

        let mut rationale = vec![
            format!("HNSW selected for {} vectors (log-time search)", n),
            format!("ef_search={}, M={}, layers≈{}", hnsw.ef_search, hnsw.m, layers),
            format!("~{} distance computations estimated", distance_computations),
        ];

        if n < self.brute_force_threshold {
            rationale.push(format!(
                "Note: dataset small enough for brute-force ({} < {})",
                n, self.brute_force_threshold
            ));
        }

        QueryPlan {
            index_choice: IndexChoice::Hnsw,
            cost: CostBreakdown {
                estimated_latency_ms: latency,
                estimated_memory_mb: memory_mb,
                distance_computations,
                nodes_visited,
                io_operations: 0,
                candidate_set_size: candidate_set,
                filter_cost: None,
            },
            rationale,
            alternatives: Vec::new(),
        }
    }

    fn estimate_brute_force(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        filter_selectivity: Option<f32>,
    ) -> QueryPlan {
        let n = stats.active_vectors();
        let sel = filter_selectivity.unwrap_or(1.0);
        let effective_n = (n as f32 * sel) as usize;

        let distance_computations = n; // must scan all
        let filter_evaluations = if filter_selectivity.is_some() { n } else { 0 };

        let latency = (distance_computations as f64 * self.distance_cost_us
            + filter_evaluations as f64 * self.filter_cost_us)
            / 1000.0;
        let memory_mb = (k * stats.dimensions * 4) as f64 / (1024.0 * 1024.0);

        let mut rationale =
            vec![format!("Brute-force scan of {} vectors", n)];
        if n <= self.brute_force_threshold {
            rationale
                .push("Preferred for small datasets (guaranteed 100% recall)".into());
        } else {
            rationale.push("Warning: dataset may be too large for brute-force".into());
        }

        QueryPlan {
            index_choice: IndexChoice::BruteForce,
            cost: CostBreakdown {
                estimated_latency_ms: latency,
                estimated_memory_mb: memory_mb,
                distance_computations,
                nodes_visited: n,
                io_operations: 0,
                candidate_set_size: effective_n,
                filter_cost: filter_selectivity.map(|s| FilterCost {
                    selectivity: s,
                    strategy: FilterStrategy::PreFilter,
                    evaluations: n,
                }),
            },
            rationale,
            alternatives: Vec::new(),
        }
    }

    fn estimate_hnsw_post_filter(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        selectivity: f32,
    ) -> QueryPlan {
        let n = stats.active_vectors();
        let hnsw = stats.hnsw_params.as_ref().cloned().unwrap_or_default();

        // Over-fetch factor: if selectivity is 10%, need ~10x candidates
        let over_fetch = (1.0 / selectivity.max(0.01)).ceil() as usize;
        let fetch_k = k * over_fetch.min(20); // cap at 20x
        let adjusted_ef = hnsw.ef_search.max(fetch_k);

        let layers = (n as f64).log2().ceil() as usize;
        let nodes_visited = adjusted_ef * layers.max(1);
        let distance_computations = nodes_visited * hnsw.m;
        let filter_evaluations = fetch_k;

        let latency = (distance_computations as f64 * self.distance_cost_us
            + filter_evaluations as f64 * self.filter_cost_us)
            / 1000.0;

        QueryPlan {
            index_choice: IndexChoice::HnswPostFilter,
            cost: CostBreakdown {
                estimated_latency_ms: latency,
                estimated_memory_mb: (fetch_k * stats.dimensions * 4) as f64
                    / (1024.0 * 1024.0),
                distance_computations,
                nodes_visited,
                io_operations: 0,
                candidate_set_size: fetch_k,
                filter_cost: Some(FilterCost {
                    selectivity,
                    strategy: FilterStrategy::PostFilter,
                    evaluations: filter_evaluations,
                }),
            },
            rationale: vec![
                format!(
                    "HNSW+PostFilter: fetch {}×k={} candidates, then filter",
                    over_fetch, fetch_k
                ),
                format!("Filter selectivity: {:.1}%", selectivity * 100.0),
            ],
            alternatives: Vec::new(),
        }
    }

    fn estimate_hnsw_pre_filter(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        selectivity: f32,
    ) -> QueryPlan {
        let n = stats.active_vectors();
        let hnsw = stats.hnsw_params.as_ref().cloned().unwrap_or_default();

        // Pre-filter: check filter during HNSW traversal, skip non-matching
        // Need to visit more nodes because some will be filtered out
        let visit_factor = (1.0 / selectivity.max(0.01)).sqrt().ceil() as usize;
        let adjusted_ef = hnsw.ef_search * visit_factor.min(10);

        let layers = (n as f64).log2().ceil() as usize;
        let nodes_visited = adjusted_ef * layers.max(1);
        let distance_computations = nodes_visited * hnsw.m;
        let filter_evaluations = nodes_visited;

        let latency = (distance_computations as f64 * self.distance_cost_us
            + filter_evaluations as f64 * self.filter_cost_us)
            / 1000.0;

        QueryPlan {
            index_choice: IndexChoice::HnswPreFilter,
            cost: CostBreakdown {
                estimated_latency_ms: latency,
                estimated_memory_mb: (k * stats.dimensions * 4) as f64 / (1024.0 * 1024.0),
                distance_computations,
                nodes_visited,
                io_operations: 0,
                candidate_set_size: (n as f32 * selectivity) as usize,
                filter_cost: Some(FilterCost {
                    selectivity,
                    strategy: FilterStrategy::PreFilter,
                    evaluations: filter_evaluations,
                }),
            },
            rationale: vec![
                format!(
                    "HNSW+PreFilter: evaluate filter during traversal (selectivity {:.1}%)",
                    selectivity * 100.0
                ),
                format!(
                    "Visit factor {}×, adjusted ef={}",
                    visit_factor, adjusted_ef
                ),
            ],
            alternatives: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_dataset_prefers_brute_force() {
        let stats = CollectionStatistics::new(1_000, 128, 0.0);
        let estimator = CostEstimator::new();
        let plan = estimator.plan(&stats, 10, None);

        // For small datasets, brute-force should be faster
        assert_eq!(plan.index_choice, IndexChoice::BruteForce);
    }

    #[test]
    fn test_large_dataset_prefers_hnsw() {
        let stats = CollectionStatistics::new(1_000_000, 384, 0.05);
        let estimator = CostEstimator::new();
        let plan = estimator.plan(&stats, 10, None);

        assert_eq!(plan.index_choice, IndexChoice::Hnsw);
        assert!(plan.cost.estimated_latency_ms < 100.0);
    }

    #[test]
    fn test_filter_strategy_selection() {
        let stats = CollectionStatistics::new(1_000_000, 384, 0.0);
        let estimator = CostEstimator::new();

        // High selectivity (many matches) → pre-filter available
        let plan_high = estimator.plan(&stats, 10, Some(0.5));
        // Should have considered filter-aware strategies
        assert!(!plan_high.alternatives.is_empty());

        // Low selectivity (few matches) → post-filter available as alternative
        let plan_low = estimator.plan(&stats, 10, Some(0.01));
        // At least one plan should have filter cost info
        let has_filter_plan = plan_low.cost.filter_cost.is_some()
            || plan_low
                .alternatives
                .iter()
                .any(|a| matches!(a.index_choice, IndexChoice::HnswPostFilter | IndexChoice::HnswPreFilter));
        assert!(has_filter_plan);
    }

    #[test]
    fn test_cost_breakdown() {
        let stats = CollectionStatistics::new(100_000, 256, 0.1);
        let estimator = CostEstimator::new();
        let plan = estimator.plan(&stats, 10, None);

        assert!(plan.cost.distance_computations > 0);
        assert!(plan.cost.nodes_visited > 0);
        assert!(plan.cost.estimated_latency_ms > 0.0);
    }

    #[test]
    fn test_plan_display() {
        let stats = CollectionStatistics::new(500_000, 384, 0.0);
        let estimator = CostEstimator::new();
        let plan = estimator.plan(&stats, 10, Some(0.1));

        let display = format!("{}", plan);
        assert!(display.contains("Query Plan"));
        assert!(display.contains("Estimated latency"));
    }

    #[test]
    fn test_alternatives_included() {
        let stats = CollectionStatistics::new(100_000, 128, 0.0);
        let estimator = CostEstimator::new();
        let plan = estimator.plan(&stats, 10, Some(0.1));

        // Should have alternative plans
        assert!(!plan.alternatives.is_empty());
    }

    #[test]
    fn test_estimated_memory() {
        let stats = CollectionStatistics::new(1_000_000, 384, 0.0);
        let mem = stats.estimated_memory_bytes();
        // 1M * 384 * 4 = 1.536GB for vectors alone
        assert!(mem > 1_000_000_000);
    }

    #[test]
    fn test_active_vectors() {
        let stats = CollectionStatistics::new(10_000, 64, 0.2);
        assert_eq!(stats.active_vectors(), 8_000);
    }
}
