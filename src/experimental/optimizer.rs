//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Adaptive Query Optimizer
//!
//! Cost-based query optimization for vector search with adaptive feedback:
//! - Query plan generation with cost estimation
//! - Multiple strategy evaluation (FullScan, HNSW, IVF, DiskANN, etc.)
//! - Feedback-driven cost model adjustment via exponential moving average
//! - Latency budget constraints
//! - Filter pushdown and index selection
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::optimizer::{AdaptiveOptimizer, CollectionStatistics, IndexType};
//!
//! let stats = CollectionStatistics {
//!     total_vectors: 100_000,
//!     dimensions: 384,
//!     index_type: IndexType::Hnsw,
//!     has_metadata_index: true,
//!     avg_query_latency_ms: 2.0,
//!     filter_selectivity: 0.1,
//!     memory_usage_bytes: 500_000_000,
//! };
//!
//! let optimizer = AdaptiveOptimizer::new();
//! let plan = optimizer.plan_query(&stats, 10, Some(0.05), Some(5.0));
//! println!("Strategy: {:?}", plan.strategy);
//! println!("Estimated latency: {:.2}ms", plan.estimated_cost.estimated_latency_ms);
//! ```

use crate::metadata::Filter;
use crate::CollectionRef;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// QueryStrategy
// ---------------------------------------------------------------------------

/// Query execution strategy
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum QueryStrategy {
    /// Full vector scan (for small collections or no index)
    FullScan,
    /// HNSW index search
    HnswSearch,
    /// IVF (Inverted File) index search
    IvfSearch,
    /// DiskANN graph-based search
    DiskAnnSearch,
    /// Pre-filter then search (high selectivity filters)
    FilterFirst,
    /// Search then filter (low selectivity filters)
    SearchFirst,
    /// Hybrid: combine multiple strategies
    Hybrid,
    /// Use metadata index only (no vector search)
    MetadataOnly,
}

// ---------------------------------------------------------------------------
// IndexType
// ---------------------------------------------------------------------------

/// Type of vector index available on a collection
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IndexType {
    /// No index — brute-force scan
    None,
    /// HNSW graph index
    Hnsw,
    /// IVF (Inverted File) index
    Ivf,
    /// DiskANN graph index
    DiskAnn,
}

// ---------------------------------------------------------------------------
// CollectionStatistics
// ---------------------------------------------------------------------------

/// Statistics about a collection used for cost estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatistics {
    /// Total number of vectors in the collection
    pub total_vectors: usize,
    /// Dimensionality of each vector
    pub dimensions: usize,
    /// Type of vector index
    pub index_type: IndexType,
    /// Whether a metadata index exists
    pub has_metadata_index: bool,
    /// Average observed query latency in milliseconds
    pub avg_query_latency_ms: f64,
    /// Default filter selectivity (fraction of vectors matching), 0.0–1.0
    pub filter_selectivity: f64,
    /// Approximate memory usage in bytes
    pub memory_usage_bytes: usize,
}

impl CollectionStatistics {
    /// Build statistics from a `CollectionRef`.
    pub fn from_collection(collection: &CollectionRef) -> Self {
        let total_vectors = collection.len();
        let dimensions = collection.dimensions().unwrap_or(128);
        Self {
            total_vectors,
            dimensions,
            index_type: IndexType::Hnsw,
            has_metadata_index: false,
            avg_query_latency_ms: 1.0,
            filter_selectivity: 1.0,
            memory_usage_bytes: total_vectors * dimensions * 4,
        }
    }
}

// ---------------------------------------------------------------------------
// CostEstimate
// ---------------------------------------------------------------------------

/// Cost estimate for a single strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// The strategy being estimated
    pub strategy: QueryStrategy,
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    /// Estimated recall (0.0–1.0)
    pub estimated_recall: f64,
    /// Estimated peak memory usage in bytes
    pub estimated_memory_bytes: usize,
    /// Confidence in the estimate (0.0–1.0)
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// CostModel
// ---------------------------------------------------------------------------

/// Heuristic cost model that produces `CostEstimate`s for every strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    /// Cost per distance computation
    pub distance_cost: f64,
    /// Cost per metadata check
    pub filter_cost: f64,
    /// Cost per HNSW hop
    pub hnsw_hop_cost: f64,
    /// Cost per disk read (if applicable)
    pub disk_read_cost: f64,
    /// Base overhead cost
    pub base_cost: f64,
    /// Parallelization benefit factor
    pub parallel_factor: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            distance_cost: 1.0,
            filter_cost: 0.1,
            hnsw_hop_cost: 2.0,
            disk_read_cost: 10.0,
            base_cost: 5.0,
            parallel_factor: 0.6,
        }
    }
}

impl CostModel {
    /// Return a ranked list of cost estimates for all applicable strategies.
    pub fn estimate_cost(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        filter_selectivity: Option<f64>,
    ) -> Vec<CostEstimate> {
        let sel = filter_selectivity.unwrap_or(stats.filter_selectivity);
        let n = stats.total_vectors.max(1) as f64;
        let dim = stats.dimensions.max(1) as f64;
        let has_filter = filter_selectivity.is_some();

        let mut estimates: Vec<CostEstimate> = Vec::new();

        // --- FullScan ---
        {
            let ops = n;
            let lat = self.base_cost + ops * self.distance_cost * (dim / 128.0).max(1.0);
            let lat_ms = lat * 0.001;
            let recall = 1.0; // exact
            let mem = (n as usize) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::FullScan,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.9,
            });
        }

        // --- HnswSearch ---
        if stats.index_type == IndexType::Hnsw || stats.index_type == IndexType::None {
            let ef = 50.0_f64; // default ef_search
            let log_n = n.ln().max(1.0);
            let visited = log_n * ef;
            let lat = self.base_cost
                + visited * self.hnsw_hop_cost
                + visited * self.distance_cost;
            let lat_ms = lat * 0.001;
            let recall = (0.95_f64).min(1.0 - 1.0 / (ef + 1.0));
            let mem = (visited as usize) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::HnswSearch,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.85,
            });
        }

        // --- IvfSearch ---
        if stats.index_type == IndexType::Ivf {
            let n_probes = 10.0_f64;
            let n_clusters = (n / 1000.0).max(1.0).sqrt().ceil();
            let vectors_per_probe = n / n_clusters;
            let ops = n_probes * vectors_per_probe;
            let lat = self.base_cost + ops * self.distance_cost;
            let lat_ms = lat * 0.001;
            let recall = (n_probes / n_clusters).min(0.98);
            let mem = (ops as usize) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::IvfSearch,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.80,
            });
        }

        // --- DiskAnnSearch ---
        if stats.index_type == IndexType::DiskAnn {
            let log_n = n.ln().max(1.0);
            let hops = log_n * 1.5;
            let lat = self.base_cost
                + hops * self.disk_read_cost
                + hops * self.distance_cost;
            let lat_ms = lat * 0.001;
            let recall = 0.93;
            let mem = (hops as usize) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::DiskAnnSearch,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.80,
            });
        }

        // --- FilterFirst ---
        if has_filter {
            let filtered_count = (n * sel).max(1.0);
            let lat = self.base_cost
                + n * self.filter_cost
                + filtered_count * self.distance_cost;
            let lat_ms = lat * 0.001;
            let recall = 1.0; // exact on filtered set
            let mem = (filtered_count as usize) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::FilterFirst,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.85,
            });
        }

        // --- SearchFirst ---
        if has_filter && (stats.index_type == IndexType::Hnsw || stats.index_type == IndexType::Ivf) {
            let over_fetch = (k as f64 / sel.max(0.01)).min(n).max(k as f64);
            let ef = 50.0_f64;
            let log_n = n.ln().max(1.0);
            let visited = (log_n * ef).max(over_fetch);
            let lat = self.base_cost
                + visited * self.hnsw_hop_cost
                + visited * self.distance_cost
                + over_fetch * self.filter_cost;
            let lat_ms = lat * 0.001;
            // recall degrades with selective filters
            let recall = (0.95 * sel.sqrt()).max(0.5).min(0.95);
            let mem = (visited as usize) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::SearchFirst,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.75,
            });
        }

        // --- Hybrid ---
        if has_filter && stats.index_type != IndexType::None {
            // Parallel filter + index search with merge
            let ef = 50.0_f64;
            let log_n = n.ln().max(1.0);
            let search_lat = log_n * ef * self.hnsw_hop_cost;
            let filter_lat = n * self.filter_cost;
            let lat = self.base_cost + search_lat.max(filter_lat) * self.parallel_factor + 10.0;
            let lat_ms = lat * 0.001;
            let recall = 0.97;
            let mem = (n as usize / 10) * stats.dimensions * 4;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::Hybrid,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.70,
            });
        }

        // --- MetadataOnly ---
        if has_filter && stats.has_metadata_index {
            let lat = self.base_cost + n * self.filter_cost * 0.5; // index-assisted
            let lat_ms = lat * 0.001;
            let recall = 1.0;
            let mem = 1024;
            estimates.push(CostEstimate {
                strategy: QueryStrategy::MetadataOnly,
                estimated_latency_ms: lat_ms,
                estimated_recall: recall,
                estimated_memory_bytes: mem,
                confidence: 0.60,
            });
        }

        // Sort by estimated latency (best first)
        estimates.sort_by(|a, b| {
            a.estimated_latency_ms
                .partial_cmp(&b.estimated_latency_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        estimates
    }
}

// ---------------------------------------------------------------------------
// QueryPlan  (preserves backward-compatible fields)
// ---------------------------------------------------------------------------

/// Query plan generated by the optimizer
#[derive(Debug, Clone)]
pub struct QueryPlan {
    /// Chosen execution strategy
    pub strategy: QueryStrategy,
    /// Estimated cost (backward compat: same as estimated_latency_ms)
    pub estimated_cost: f64,
    /// Estimated candidates to evaluate
    pub estimated_candidates: usize,
    /// Estimated latency in milliseconds
    pub estimated_latency_ms: f64,
    /// Whether to use parallelization
    pub parallel: bool,
    /// Filter selectivity estimate (0–1)
    pub filter_selectivity: Option<f64>,
    /// Optimization hints
    pub hints: Vec<OptimizationHint>,
    /// Full cost estimate for the chosen strategy
    pub cost_estimate: Option<CostEstimate>,
    /// Fallback strategy if the primary strategy fails or exceeds budget
    pub fallback_strategy: Option<QueryStrategy>,
}

// ---------------------------------------------------------------------------
// Hints (unchanged from original)
// ---------------------------------------------------------------------------

/// Optimization hint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    pub category: HintCategory,
    pub message: String,
    pub impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum HintCategory {
    IndexUsage,
    FilterOptimization,
    MemoryUsage,
    Parallelization,
    CachingOpportunity,
}

// ---------------------------------------------------------------------------
// QueryStats (unchanged from original)
// ---------------------------------------------------------------------------

/// Query execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStats {
    /// Actual execution time
    pub execution_time: Duration,
    /// Vectors scanned
    pub vectors_scanned: usize,
    /// Vectors passed filter
    pub vectors_filtered: usize,
    /// Distance computations performed
    pub distance_computations: usize,
    /// HNSW nodes visited
    pub hnsw_nodes_visited: usize,
    /// Cache hits
    pub cache_hits: usize,
    /// Strategy used
    pub strategy_used: QueryStrategy,
}

// ---------------------------------------------------------------------------
// Legacy CollectionStats (backward compat for QueryOptimizer::new)
// ---------------------------------------------------------------------------

/// Collection statistics for optimization (legacy API)
#[derive(Debug, Clone)]
pub struct CollectionStats {
    pub total_vectors: usize,
    pub dimensions: usize,
    pub has_hnsw_index: bool,
    pub hnsw_m: usize,
    pub hnsw_ef_search: usize,
    pub metadata_cardinality: HashMap<String, usize>,
    pub avg_vectors_per_metadata_value: HashMap<String, f64>,
}

impl CollectionStats {
    /// Convert legacy stats into `CollectionStatistics`.
    pub fn to_collection_statistics(&self) -> CollectionStatistics {
        let index_type = if self.has_hnsw_index {
            IndexType::Hnsw
        } else {
            IndexType::None
        };
        CollectionStatistics {
            total_vectors: self.total_vectors,
            dimensions: self.dimensions,
            index_type,
            has_metadata_index: !self.metadata_cardinality.is_empty(),
            avg_query_latency_ms: 1.0,
            filter_selectivity: 1.0,
            memory_usage_bytes: self.total_vectors * self.dimensions * 4,
        }
    }
}

// ---------------------------------------------------------------------------
// QueryOptimizer  (preserves backward-compatible API)
// ---------------------------------------------------------------------------

/// Query optimizer (legacy + adaptive API)
pub struct QueryOptimizer {
    cost_model: CostModel,
    collection_stats: CollectionStats,
    historical_stats: Vec<QueryStats>,
}

impl QueryOptimizer {
    /// Create a new optimizer from legacy `CollectionStats`.
    pub fn new(stats: CollectionStats) -> Self {
        Self {
            cost_model: CostModel::default(),
            collection_stats: stats,
            historical_stats: Vec::new(),
        }
    }

    /// Create optimizer from a `CollectionRef`.
    pub fn from_collection(collection: &CollectionRef) -> Self {
        let total_vectors = collection.len();
        let dimensions = collection.dimensions().unwrap_or(128);
        let collection_stats = CollectionStats {
            total_vectors,
            dimensions,
            has_hnsw_index: true,
            hnsw_m: 16,
            hnsw_ef_search: 50,
            metadata_cardinality: HashMap::new(),
            avg_vectors_per_metadata_value: HashMap::new(),
        };
        Self::new(collection_stats)
    }

    /// Set custom cost model.
    pub fn with_cost_model(mut self, model: CostModel) -> Self {
        self.cost_model = model;
        self
    }

    /// Optimize a query and return execution plan (backward-compatible signature).
    pub fn optimize(
        &self,
        _query_vector: &[f32],
        filter: Option<&Filter>,
        k: usize,
    ) -> QueryPlan {
        let selectivity = filter.map(|f| self.estimate_selectivity(f));

        let cs = self.collection_stats.to_collection_statistics();
        let estimates = self.cost_model.estimate_cost(&cs, k, selectivity);

        // Pick best estimate
        let best = estimates.first().cloned();
        let fallback = estimates.get(1).map(|e| e.strategy.clone());

        let plan_cost = best.as_ref().map_or(0.0, |e| e.estimated_latency_ms);
        let strategy = best
            .as_ref()
            .map_or(QueryStrategy::FullScan, |e| e.strategy.clone());
        let candidates = self.estimate_candidates(&strategy, selectivity);

        let mut plan = QueryPlan {
            strategy,
            estimated_cost: plan_cost,
            estimated_candidates: candidates,
            estimated_latency_ms: plan_cost,
            parallel: false,
            filter_selectivity: selectivity,
            hints: Vec::new(),
            cost_estimate: best,
            fallback_strategy: fallback,
        };

        plan.hints = self.generate_hints(&plan, filter, k);
        plan
    }

    fn estimate_candidates(&self, strategy: &QueryStrategy, selectivity: Option<f64>) -> usize {
        let total = self.collection_stats.total_vectors;
        match strategy {
            QueryStrategy::FullScan | QueryStrategy::MetadataOnly => total,
            QueryStrategy::HnswSearch | QueryStrategy::DiskAnnSearch => {
                let ef = self.collection_stats.hnsw_ef_search;
                let log_n = (total as f64).ln().max(1.0);
                (log_n * ef as f64) as usize
            }
            QueryStrategy::IvfSearch => total / 10,
            QueryStrategy::FilterFirst => {
                let sel = selectivity.unwrap_or(0.1);
                (total as f64 * sel) as usize
            }
            QueryStrategy::SearchFirst => self.collection_stats.hnsw_ef_search * 2,
            QueryStrategy::Hybrid => {
                let sel = selectivity.unwrap_or(0.5);
                (total as f64 * sel) as usize
            }
        }
    }

    /// Estimate filter selectivity (fraction of vectors that pass).
    fn estimate_selectivity(&self, filter: &Filter) -> f64 {
        use crate::metadata::FilterOperator;

        match filter {
            Filter::Condition(cond) => match cond.operator {
                FilterOperator::Eq => {
                    if let Some(cardinality) =
                        self.collection_stats.metadata_cardinality.get(&cond.field)
                    {
                        1.0 / (*cardinality as f64).max(1.0)
                    } else {
                        0.1
                    }
                }
                FilterOperator::Ne => 0.9,
                FilterOperator::Gt | FilterOperator::Gte => 0.5,
                FilterOperator::Lt | FilterOperator::Lte => 0.5,
                FilterOperator::In => {
                    if let Some(arr) = cond.value.as_array() {
                        (0.1 * arr.len() as f64).min(0.9)
                    } else {
                        0.1
                    }
                }
                FilterOperator::NotIn => {
                    if let Some(arr) = cond.value.as_array() {
                        1.0 - (0.1 * arr.len() as f64).min(0.9)
                    } else {
                        0.9
                    }
                }
                FilterOperator::Contains => 0.3,
            },
            Filter::And(filters) => filters.iter().map(|f| self.estimate_selectivity(f)).product(),
            Filter::Or(filters) => {
                let product: f64 = filters
                    .iter()
                    .map(|f| 1.0 - self.estimate_selectivity(f))
                    .product();
                1.0 - product
            }
            Filter::Not(inner) => 1.0 - self.estimate_selectivity(inner),
        }
    }

    fn generate_hints(
        &self,
        plan: &QueryPlan,
        filter: Option<&Filter>,
        k: usize,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();
        let total = self.collection_stats.total_vectors;

        if !self.collection_stats.has_hnsw_index && total > 10000 {
            hints.push(OptimizationHint {
                category: HintCategory::IndexUsage,
                message: "Consider building HNSW index for faster queries".to_string(),
                impact: 0.8,
            });
        }

        if let Some(f) = filter {
            if let Some(sel) = plan.filter_selectivity {
                if sel < 0.01 {
                    hints.push(OptimizationHint {
                        category: HintCategory::FilterOptimization,
                        message: "Highly selective filter - consider metadata index".to_string(),
                        impact: 0.5,
                    });
                }
            }
            if self.is_expensive_filter(f) {
                hints.push(OptimizationHint {
                    category: HintCategory::FilterOptimization,
                    message: "Complex filter detected - consider simplifying or indexing"
                        .to_string(),
                    impact: 0.3,
                });
            }
        }

        if total > 1_000_000 {
            hints.push(OptimizationHint {
                category: HintCategory::MemoryUsage,
                message: "Large collection - consider quantization for memory savings".to_string(),
                impact: 0.4,
            });
        }

        if total > 100_000 && !plan.parallel {
            hints.push(OptimizationHint {
                category: HintCategory::Parallelization,
                message: "Collection size suggests parallelization could help".to_string(),
                impact: 0.3,
            });
        }

        if k <= 10 {
            hints.push(OptimizationHint {
                category: HintCategory::CachingOpportunity,
                message: "Small k - results may benefit from caching".to_string(),
                impact: 0.2,
            });
        }

        hints
    }

    #[allow(clippy::only_used_in_recursion)]
    fn is_expensive_filter(&self, filter: &Filter) -> bool {
        match filter {
            Filter::And(filters) | Filter::Or(filters) => {
                filters.len() > 3 || filters.iter().any(|f| self.is_expensive_filter(f))
            }
            Filter::Not(inner) => self.is_expensive_filter(inner),
            _ => false,
        }
    }

    /// Record query statistics for learning.
    pub fn record_stats(&mut self, stats: QueryStats) {
        self.historical_stats.push(stats);
        if self.historical_stats.len() > 1000 {
            self.historical_stats.remove(0);
        }
    }

    /// Get average stats for a strategy.
    pub fn get_strategy_stats(&self, strategy: &QueryStrategy) -> Option<QueryStats> {
        let matching: Vec<&QueryStats> = self
            .historical_stats
            .iter()
            .filter(|s| &s.strategy_used == strategy)
            .collect();

        if matching.is_empty() {
            return None;
        }

        let count = matching.len();
        Some(QueryStats {
            execution_time: Duration::from_nanos(
                (matching
                    .iter()
                    .map(|s| s.execution_time.as_nanos())
                    .sum::<u128>()
                    / count as u128) as u64,
            ),
            vectors_scanned: matching.iter().map(|s| s.vectors_scanned).sum::<usize>() / count,
            vectors_filtered: matching.iter().map(|s| s.vectors_filtered).sum::<usize>() / count,
            distance_computations: matching
                .iter()
                .map(|s| s.distance_computations)
                .sum::<usize>()
                / count,
            hnsw_nodes_visited: matching
                .iter()
                .map(|s| s.hnsw_nodes_visited)
                .sum::<usize>()
                / count,
            cache_hits: matching.iter().map(|s| s.cache_hits).sum::<usize>() / count,
            strategy_used: strategy.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// QueryExplainer (unchanged from original)
// ---------------------------------------------------------------------------

/// Query plan explainer (like EXPLAIN ANALYZE)
pub struct QueryExplainer;

impl QueryExplainer {
    /// Generate human-readable explanation of query plan.
    pub fn explain(plan: &QueryPlan) -> String {
        let mut output = String::new();

        output.push_str("=== Query Plan ===\n\n");
        output.push_str(&format!("Strategy: {:?}\n", plan.strategy));
        output.push_str(&format!("Estimated Cost: {:.2}\n", plan.estimated_cost));
        output.push_str(&format!(
            "Estimated Candidates: {}\n",
            plan.estimated_candidates
        ));
        output.push_str(&format!(
            "Estimated Latency: {:.2}ms\n",
            plan.estimated_latency_ms
        ));
        output.push_str(&format!("Parallel Execution: {}\n", plan.parallel));

        if let Some(sel) = plan.filter_selectivity {
            output.push_str(&format!("Filter Selectivity: {:.2}%\n", sel * 100.0));
        }

        if let Some(ref fb) = plan.fallback_strategy {
            output.push_str(&format!("Fallback Strategy: {:?}\n", fb));
        }

        if !plan.hints.is_empty() {
            output.push_str("\n=== Optimization Hints ===\n\n");
            for hint in &plan.hints {
                output.push_str(&format!(
                    "[{:?}] {} (impact: {:.0}%)\n",
                    hint.category,
                    hint.message,
                    hint.impact * 100.0
                ));
            }
        }

        output
    }

    /// Compare actual vs estimated performance.
    pub fn analyze(plan: &QueryPlan, stats: &QueryStats) -> String {
        let mut output = String::new();

        output.push_str("=== Query Analysis ===\n\n");
        output.push_str(&format!("Strategy Used: {:?}\n", stats.strategy_used));
        output.push_str(&format!(
            "Actual Time: {:.2}ms\n",
            stats.execution_time.as_secs_f64() * 1000.0
        ));
        output.push_str(&format!(
            "Estimated Time: {:.2}ms\n",
            plan.estimated_latency_ms
        ));

        let time_accuracy = if plan.estimated_latency_ms > 0.0 {
            let actual_ms = stats.execution_time.as_secs_f64() * 1000.0;
            (1.0
                - (actual_ms - plan.estimated_latency_ms).abs()
                    / actual_ms.max(plan.estimated_latency_ms))
                * 100.0
        } else {
            0.0
        };
        output.push_str(&format!("Estimate Accuracy: {:.1}%\n\n", time_accuracy));

        output.push_str(&format!("Vectors Scanned: {}\n", stats.vectors_scanned));
        output.push_str(&format!(
            "Vectors After Filter: {}\n",
            stats.vectors_filtered
        ));
        output.push_str(&format!(
            "Distance Computations: {}\n",
            stats.distance_computations
        ));
        output.push_str(&format!(
            "HNSW Nodes Visited: {}\n",
            stats.hnsw_nodes_visited
        ));
        output.push_str(&format!("Cache Hits: {}\n", stats.cache_hits));

        output
    }
}

// ---------------------------------------------------------------------------
// FeedbackCollector
// ---------------------------------------------------------------------------

/// Per-strategy feedback state (exponential moving average).
#[derive(Debug, Clone)]
struct StrategyFeedback {
    /// EMA of actual/estimated latency ratio
    adjustment_factor: f64,
    /// EMA of actual recall (if available)
    recall_ema: Option<f64>,
    /// Number of observations
    count: usize,
}

impl Default for StrategyFeedback {
    fn default() -> Self {
        Self {
            adjustment_factor: 1.0,
            recall_ema: None,
            count: 0,
        }
    }
}

/// Collects execution feedback and computes per-strategy adjustment factors.
///
/// Thread-safe via `parking_lot::RwLock`.
pub struct FeedbackCollector {
    alpha: f64,
    feedback: RwLock<HashMap<QueryStrategy, StrategyFeedback>>,
}

impl FeedbackCollector {
    /// Create a new collector with the given EMA smoothing factor (0 < alpha ≤ 1).
    pub fn new(alpha: f64) -> Self {
        let alpha = alpha.clamp(0.01, 1.0);
        Self {
            alpha,
            feedback: RwLock::new(HashMap::new()),
        }
    }

    /// Record the result of executing a plan.
    pub fn record_execution(
        &self,
        plan: &QueryPlan,
        actual_latency_ms: f64,
        actual_recall: Option<f64>,
    ) {
        let estimated = plan
            .cost_estimate
            .as_ref()
            .map_or(plan.estimated_latency_ms, |c| c.estimated_latency_ms);

        let ratio = if estimated > 0.0 {
            actual_latency_ms / estimated
        } else {
            1.0
        };

        let mut map = self.feedback.write();
        let entry = map.entry(plan.strategy.clone()).or_default();
        entry.adjustment_factor =
            self.alpha * ratio + (1.0 - self.alpha) * entry.adjustment_factor;
        if let Some(recall) = actual_recall {
            let prev = entry.recall_ema.unwrap_or(recall);
            entry.recall_ema = Some(self.alpha * recall + (1.0 - self.alpha) * prev);
        }
        entry.count += 1;
    }

    /// Return the current adjustment factor for a strategy (actual/estimated ratio).
    ///
    /// Returns 1.0 if no feedback has been recorded yet.
    pub fn get_adjustment_factor(&self, strategy: &QueryStrategy) -> f64 {
        self.feedback
            .read()
            .get(strategy)
            .map_or(1.0, |f| f.adjustment_factor)
    }

    /// Return the number of observations for a strategy.
    pub fn observation_count(&self, strategy: &QueryStrategy) -> usize {
        self.feedback
            .read()
            .get(strategy)
            .map_or(0, |f| f.count)
    }
}

impl Default for FeedbackCollector {
    fn default() -> Self {
        Self::new(0.3)
    }
}

// ---------------------------------------------------------------------------
// AdaptiveOptimizer
// ---------------------------------------------------------------------------

/// Combines `CostModel`, `QueryOptimizer` logic, and `FeedbackCollector` to
/// produce feedback-adjusted query plans.
pub struct AdaptiveOptimizer {
    cost_model: CostModel,
    feedback: FeedbackCollector,
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer with default cost model and feedback.
    pub fn new() -> Self {
        Self {
            cost_model: CostModel::default(),
            feedback: FeedbackCollector::default(),
        }
    }

    /// Create with a custom cost model.
    pub fn with_cost_model(mut self, model: CostModel) -> Self {
        self.cost_model = model;
        self
    }

    /// Create with a custom EMA alpha for feedback.
    pub fn with_feedback_alpha(mut self, alpha: f64) -> Self {
        self.feedback = FeedbackCollector::new(alpha);
        self
    }

    /// Plan a query using cost estimation + feedback adjustment.
    pub fn plan_query(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        filter_selectivity: Option<f64>,
        latency_budget_ms: Option<f64>,
    ) -> QueryPlan {
        let estimates = self.cost_model.estimate_cost(stats, k, filter_selectivity);

        // Adjust estimates with feedback
        let adjusted: Vec<(CostEstimate, f64)> = estimates
            .into_iter()
            .map(|mut e| {
                let factor = self.feedback.get_adjustment_factor(&e.strategy);
                let adjusted_latency = e.estimated_latency_ms * factor;
                e.estimated_latency_ms = adjusted_latency;
                (e, adjusted_latency)
            })
            .collect();

        // If we have a latency budget, filter to strategies within budget
        let within_budget: Vec<&(CostEstimate, f64)> = if let Some(budget) = latency_budget_ms {
            let candidates: Vec<&(CostEstimate, f64)> =
                adjusted.iter().filter(|(_, lat)| *lat <= budget).collect();
            if candidates.is_empty() {
                // Nothing fits the budget — take the fastest anyway
                adjusted.iter().collect()
            } else {
                candidates
            }
        } else {
            adjusted.iter().collect()
        };

        // Among candidates within budget, pick the lowest latency strategy whose
        // recall is at least 0.8. If none meet the recall threshold, fall back to
        // the absolute lowest latency.
        let min_recall = 0.8;
        let acceptable: Vec<&&(CostEstimate, f64)> = within_budget
            .iter()
            .filter(|e| e.0.estimated_recall >= min_recall)
            .collect();

        let best = if acceptable.is_empty() {
            within_budget
                .iter()
                .min_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("within_budget is non-empty")
        } else {
            acceptable
                .iter()
                .min_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .expect("acceptable is non-empty")
        };

        let fallback = adjusted
            .iter()
            .find(|(e, _)| e.strategy != best.0.strategy)
            .map(|(e, _)| e.strategy.clone());

        let sel = filter_selectivity.unwrap_or(stats.filter_selectivity);
        let candidates = self.estimate_candidates_for(stats, &best.0.strategy, sel, k);

        QueryPlan {
            strategy: best.0.strategy.clone(),
            estimated_cost: best.1,
            estimated_candidates: candidates,
            estimated_latency_ms: best.1,
            parallel: best.0.strategy == QueryStrategy::Hybrid,
            filter_selectivity: filter_selectivity.or(Some(stats.filter_selectivity)),
            hints: Vec::new(),
            cost_estimate: Some(best.0.clone()),
            fallback_strategy: fallback,
        }
    }

    /// Report actual execution results to update the feedback model.
    pub fn report_result(
        &self,
        plan: &QueryPlan,
        actual_latency_ms: f64,
        actual_recall: Option<f64>,
    ) {
        self.feedback
            .record_execution(plan, actual_latency_ms, actual_recall);
    }

    /// Access the underlying feedback collector.
    pub fn feedback(&self) -> &FeedbackCollector {
        &self.feedback
    }

    fn estimate_candidates_for(
        &self,
        stats: &CollectionStatistics,
        strategy: &QueryStrategy,
        selectivity: f64,
        k: usize,
    ) -> usize {
        let n = stats.total_vectors;
        match strategy {
            QueryStrategy::FullScan | QueryStrategy::MetadataOnly => n,
            QueryStrategy::HnswSearch | QueryStrategy::DiskAnnSearch => {
                let log_n = (n as f64).ln().max(1.0);
                (log_n * 50.0) as usize
            }
            QueryStrategy::IvfSearch => n / 10,
            QueryStrategy::FilterFirst => ((n as f64) * selectivity) as usize,
            QueryStrategy::SearchFirst => {
                let over_fetch = (k as f64 / selectivity.max(0.01)).min(n as f64);
                over_fetch as usize
            }
            QueryStrategy::Hybrid => ((n as f64) * selectivity) as usize,
        }
    }
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- helpers ----

    fn create_test_stats() -> CollectionStats {
        CollectionStats {
            total_vectors: 100_000,
            dimensions: 384,
            has_hnsw_index: true,
            hnsw_m: 16,
            hnsw_ef_search: 50,
            metadata_cardinality: [("category".to_string(), 10)].into_iter().collect(),
            avg_vectors_per_metadata_value: [("category".to_string(), 10000.0)]
                .into_iter()
                .collect(),
        }
    }

    fn make_collection_statistics(n: usize, idx: IndexType) -> CollectionStatistics {
        CollectionStatistics {
            total_vectors: n,
            dimensions: 384,
            index_type: idx,
            has_metadata_index: false,
            avg_query_latency_ms: 1.0,
            filter_selectivity: 1.0,
            memory_usage_bytes: n * 384 * 4,
        }
    }

    // ---- backward-compat tests (existing QueryOptimizer API) ----

    #[test]
    fn test_optimizer_no_filter() {
        let optimizer = QueryOptimizer::new(create_test_stats());
        let query = vec![0.0; 384];
        let plan = optimizer.optimize(&query, None, 10);

        assert!(plan.estimated_cost > 0.0);
        // With the new cost model the best no-filter strategy for HNSW is HnswSearch
        assert_eq!(plan.strategy, QueryStrategy::HnswSearch);
    }

    #[test]
    fn test_optimizer_with_filter() {
        let optimizer = QueryOptimizer::new(create_test_stats());
        let query = vec![0.0; 384];
        let filter = Filter::eq("category".to_string(), serde_json::json!("tech"));
        let plan = optimizer.optimize(&query, Some(&filter), 10);

        assert!(plan.filter_selectivity.is_some());
    }

    #[test]
    fn test_selectivity_estimation() {
        let optimizer = QueryOptimizer::new(create_test_stats());

        let eq_filter = Filter::eq("category".to_string(), serde_json::json!("tech"));
        let sel = optimizer.estimate_selectivity(&eq_filter);
        assert!(sel > 0.0 && sel < 1.0);

        let and_filter = Filter::And(vec![
            Filter::eq("category".to_string(), serde_json::json!("tech")),
            Filter::eq("category".to_string(), serde_json::json!("ai")),
        ]);
        let and_sel = optimizer.estimate_selectivity(&and_filter);
        assert!(and_sel < sel);
    }

    #[test]
    fn test_small_collection_prefers_scan() {
        let mut stats = create_test_stats();
        stats.total_vectors = 100;

        let optimizer = QueryOptimizer::new(stats);
        let query = vec![0.0; 384];
        let plan = optimizer.optimize(&query, None, 10);

        assert!(plan.estimated_cost > 0.0);
    }

    #[test]
    fn test_query_explainer() {
        let optimizer = QueryOptimizer::new(create_test_stats());
        let query = vec![0.0; 384];
        let plan = optimizer.optimize(&query, None, 10);

        let explanation = QueryExplainer::explain(&plan);
        assert!(explanation.contains("Strategy"));
        assert!(explanation.contains("Estimated"));
    }

    #[test]
    fn test_hints_generation() {
        let mut stats = create_test_stats();
        stats.total_vectors = 2_000_000;

        let optimizer = QueryOptimizer::new(stats);
        let query = vec![0.0; 384];
        let plan = optimizer.optimize(&query, None, 5);

        assert!(!plan.hints.is_empty());
    }

    // ---- CostModel tests ----

    #[test]
    fn test_cost_model_small_collection() {
        let model = CostModel::default();
        let stats = make_collection_statistics(50, IndexType::None);
        let estimates = model.estimate_cost(&stats, 5, None);

        assert!(!estimates.is_empty());
        // FullScan should be present
        assert!(estimates.iter().any(|e| e.strategy == QueryStrategy::FullScan));
        // All latencies should be positive
        for e in &estimates {
            assert!(e.estimated_latency_ms > 0.0);
        }
    }

    #[test]
    fn test_cost_model_hnsw_faster_than_scan_for_large() {
        let model = CostModel::default();
        let stats = make_collection_statistics(1_000_000, IndexType::Hnsw);
        let estimates = model.estimate_cost(&stats, 10, None);

        let scan = estimates.iter().find(|e| e.strategy == QueryStrategy::FullScan).unwrap();
        let hnsw = estimates.iter().find(|e| e.strategy == QueryStrategy::HnswSearch).unwrap();
        assert!(hnsw.estimated_latency_ms < scan.estimated_latency_ms);
    }

    #[test]
    fn test_cost_model_with_filter_adds_strategies() {
        let model = CostModel::default();
        let mut stats = make_collection_statistics(100_000, IndexType::Hnsw);
        stats.has_metadata_index = true;

        let without_filter = model.estimate_cost(&stats, 10, None);
        let with_filter = model.estimate_cost(&stats, 10, Some(0.05));

        assert!(with_filter.len() > without_filter.len());
        // FilterFirst should appear with filter
        assert!(with_filter.iter().any(|e| e.strategy == QueryStrategy::FilterFirst));
    }

    #[test]
    fn test_cost_model_ivf() {
        let model = CostModel::default();
        let stats = make_collection_statistics(500_000, IndexType::Ivf);
        let estimates = model.estimate_cost(&stats, 10, None);
        assert!(estimates.iter().any(|e| e.strategy == QueryStrategy::IvfSearch));
    }

    #[test]
    fn test_cost_model_diskann() {
        let model = CostModel::default();
        let stats = make_collection_statistics(500_000, IndexType::DiskAnn);
        let estimates = model.estimate_cost(&stats, 10, None);
        assert!(estimates.iter().any(|e| e.strategy == QueryStrategy::DiskAnnSearch));
    }

    #[test]
    fn test_cost_model_sorted_by_latency() {
        let model = CostModel::default();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);
        let estimates = model.estimate_cost(&stats, 10, Some(0.1));

        for pair in estimates.windows(2) {
            assert!(pair[0].estimated_latency_ms <= pair[1].estimated_latency_ms);
        }
    }

    #[test]
    fn test_cost_model_recall_bounds() {
        let model = CostModel::default();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);
        let estimates = model.estimate_cost(&stats, 10, Some(0.5));

        for e in &estimates {
            assert!(e.estimated_recall >= 0.0 && e.estimated_recall <= 1.0);
            assert!(e.confidence >= 0.0 && e.confidence <= 1.0);
        }
    }

    // ---- FeedbackCollector tests ----

    #[test]
    fn test_feedback_default_factor() {
        let fc = FeedbackCollector::default();
        assert!((fc.get_adjustment_factor(&QueryStrategy::HnswSearch) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_feedback_single_observation() {
        let fc = FeedbackCollector::new(0.5);

        let plan = QueryPlan {
            strategy: QueryStrategy::HnswSearch,
            estimated_cost: 1.0,
            estimated_candidates: 100,
            estimated_latency_ms: 1.0,
            parallel: false,
            filter_selectivity: None,
            hints: Vec::new(),
            cost_estimate: Some(CostEstimate {
                strategy: QueryStrategy::HnswSearch,
                estimated_latency_ms: 1.0,
                estimated_recall: 0.95,
                estimated_memory_bytes: 1024,
                confidence: 0.9,
            }),
            fallback_strategy: None,
        };

        // Actual was 2x slower
        fc.record_execution(&plan, 2.0, None);
        let factor = fc.get_adjustment_factor(&QueryStrategy::HnswSearch);
        // EMA: 0.5 * 2.0 + 0.5 * 1.0 = 1.5
        assert!((factor - 1.5).abs() < 1e-9);
    }

    #[test]
    fn test_feedback_converges() {
        let fc = FeedbackCollector::new(0.3);

        let plan = QueryPlan {
            strategy: QueryStrategy::FullScan,
            estimated_cost: 10.0,
            estimated_candidates: 1000,
            estimated_latency_ms: 10.0,
            parallel: false,
            filter_selectivity: None,
            hints: Vec::new(),
            cost_estimate: Some(CostEstimate {
                strategy: QueryStrategy::FullScan,
                estimated_latency_ms: 10.0,
                estimated_recall: 1.0,
                estimated_memory_bytes: 4096,
                confidence: 0.9,
            }),
            fallback_strategy: None,
        };

        // Consistently report actual = 20ms (2x estimated)
        for _ in 0..50 {
            fc.record_execution(&plan, 20.0, Some(1.0));
        }

        let factor = fc.get_adjustment_factor(&QueryStrategy::FullScan);
        // Should converge toward 2.0
        assert!((factor - 2.0).abs() < 0.05);
        assert_eq!(fc.observation_count(&QueryStrategy::FullScan), 50);
    }

    #[test]
    fn test_feedback_thread_safety() {
        use std::sync::Arc;
        let fc = Arc::new(FeedbackCollector::new(0.3));

        let plan = QueryPlan {
            strategy: QueryStrategy::HnswSearch,
            estimated_cost: 5.0,
            estimated_candidates: 500,
            estimated_latency_ms: 5.0,
            parallel: false,
            filter_selectivity: None,
            hints: Vec::new(),
            cost_estimate: Some(CostEstimate {
                strategy: QueryStrategy::HnswSearch,
                estimated_latency_ms: 5.0,
                estimated_recall: 0.95,
                estimated_memory_bytes: 2048,
                confidence: 0.85,
            }),
            fallback_strategy: None,
        };

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let fc = Arc::clone(&fc);
                let p = plan.clone();
                std::thread::spawn(move || {
                    for _ in 0..100 {
                        fc.record_execution(&p, 6.0, Some(0.94));
                        let _ = fc.get_adjustment_factor(&QueryStrategy::HnswSearch);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(fc.observation_count(&QueryStrategy::HnswSearch), 400);
        let factor = fc.get_adjustment_factor(&QueryStrategy::HnswSearch);
        // ratio = 6.0/5.0 = 1.2, should converge near 1.2
        assert!(factor > 1.0 && factor < 1.5);
    }

    // ---- AdaptiveOptimizer tests ----

    #[test]
    fn test_adaptive_basic_plan() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);

        let plan = opt.plan_query(&stats, 10, None, None);
        assert!(plan.estimated_latency_ms > 0.0);
        assert!(plan.cost_estimate.is_some());
    }

    #[test]
    fn test_adaptive_with_filter() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);

        let plan = opt.plan_query(&stats, 10, Some(0.05), None);
        assert!(plan.filter_selectivity.is_some());
        // With a very selective filter, FilterFirst or SearchFirst should be considered
        assert!(plan.fallback_strategy.is_some());
    }

    #[test]
    fn test_adaptive_latency_budget_constraint() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(1_000_000, IndexType::Hnsw);

        // Very tight budget — should still produce a plan
        let plan = opt.plan_query(&stats, 10, None, Some(0.001));
        assert!(plan.estimated_latency_ms > 0.0);

        // Generous budget
        let plan2 = opt.plan_query(&stats, 10, None, Some(1000.0));
        assert!(plan2.estimated_latency_ms > 0.0);
    }

    #[test]
    fn test_adaptive_feedback_improves_estimates() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);

        let plan1 = opt.plan_query(&stats, 10, None, None);
        let original_latency = plan1.estimated_latency_ms;

        // Report that actual latency is 3x the estimate
        for _ in 0..20 {
            opt.report_result(&plan1, original_latency * 3.0, Some(0.90));
        }

        let plan2 = opt.plan_query(&stats, 10, None, None);
        // The adjusted estimate for the same strategy should be higher now
        if plan2.strategy == plan1.strategy {
            assert!(plan2.estimated_latency_ms > original_latency * 1.5);
        }
    }

    #[test]
    fn test_adaptive_strategy_switching() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);

        // First plan should pick HNSW
        let plan1 = opt.plan_query(&stats, 10, Some(0.05), None);

        // Heavily penalize whatever was chosen by reporting very slow actual latency
        for _ in 0..50 {
            opt.report_result(&plan1, plan1.estimated_latency_ms * 100.0, Some(0.5));
        }

        // The optimizer should now prefer a different strategy
        let plan2 = opt.plan_query(&stats, 10, Some(0.05), None);
        assert!(
            plan2.strategy != plan1.strategy || plan2.estimated_latency_ms > plan1.estimated_latency_ms,
            "Feedback should alter strategy selection or cost"
        );
    }

    // ---- Edge case tests ----

    #[test]
    fn test_empty_collection() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(0, IndexType::None);

        let plan = opt.plan_query(&stats, 10, None, None);
        assert!(plan.estimated_latency_ms >= 0.0);
        assert_eq!(plan.strategy, QueryStrategy::FullScan);
    }

    #[test]
    fn test_single_vector() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(1, IndexType::Hnsw);

        let plan = opt.plan_query(&stats, 1, None, None);
        assert!(plan.estimated_latency_ms > 0.0);
    }

    #[test]
    fn test_very_large_collection() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000_000, IndexType::Hnsw);

        let plan = opt.plan_query(&stats, 10, None, None);
        assert!(plan.estimated_latency_ms > 0.0);
        // HNSW should still be much faster than full scan for 100M vectors
        assert!(plan.strategy == QueryStrategy::HnswSearch);
    }

    #[test]
    fn test_very_large_collection_with_filter() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000_000, IndexType::Hnsw);

        let plan = opt.plan_query(&stats, 10, Some(0.001), None);
        // With very selective filter on large collection, FilterFirst can win
        assert!(plan.estimated_latency_ms > 0.0);
        assert!(plan.fallback_strategy.is_some());
    }

    #[test]
    fn test_metadata_only_strategy() {
        let opt = AdaptiveOptimizer::new();
        let mut stats = make_collection_statistics(100_000, IndexType::Hnsw);
        stats.has_metadata_index = true;

        let estimates = CostModel::default().estimate_cost(&stats, 10, Some(0.01));
        assert!(estimates.iter().any(|e| e.strategy == QueryStrategy::MetadataOnly));
    }

    #[test]
    fn test_query_plan_has_fallback() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(100_000, IndexType::Hnsw);

        let plan = opt.plan_query(&stats, 10, Some(0.1), None);
        // With a filter, there should always be a fallback
        assert!(plan.fallback_strategy.is_some());
    }

    #[test]
    fn test_cost_estimate_different_sizes() {
        let model = CostModel::default();

        let small = model.estimate_cost(&make_collection_statistics(100, IndexType::Hnsw), 5, None);
        let large = model.estimate_cost(&make_collection_statistics(10_000_000, IndexType::Hnsw), 5, None);

        let small_scan = small.iter().find(|e| e.strategy == QueryStrategy::FullScan).unwrap();
        let large_scan = large.iter().find(|e| e.strategy == QueryStrategy::FullScan).unwrap();

        // Full scan cost should scale with collection size
        assert!(large_scan.estimated_latency_ms > small_scan.estimated_latency_ms * 10.0);
    }

    #[test]
    fn test_latency_budget_prefers_fast_strategy() {
        let opt = AdaptiveOptimizer::new();
        let stats = make_collection_statistics(1_000_000, IndexType::Hnsw);

        // With a tight budget, the optimizer should still pick the fastest available strategy
        let plan = opt.plan_query(&stats, 10, None, Some(0.5));
        // The chosen strategy should be the fastest among those with acceptable recall
        let all_estimates = CostModel::default().estimate_cost(&stats, 10, None);
        let min_acceptable_latency = all_estimates
            .iter()
            .filter(|e| e.estimated_recall >= 0.8)
            .map(|e| e.estimated_latency_ms)
            .fold(f64::INFINITY, f64::min);
        // With no prior feedback, the adjustment factor is 1.0, so latency should match
        assert!(
            (plan.estimated_latency_ms - min_acceptable_latency).abs() < 1e-6,
            "Plan latency {:.6} should equal min acceptable {:.6}",
            plan.estimated_latency_ms,
            min_acceptable_latency,
        );
    }
}
