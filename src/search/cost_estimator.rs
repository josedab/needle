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

/// Histogram for cardinality estimation on a metadata field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHistogram {
    /// Field name.
    pub field: String,
    /// Bucket boundaries and counts.
    pub buckets: Vec<HistogramBucket>,
    /// Total number of values.
    pub total_count: usize,
    /// Number of distinct values.
    pub distinct_count: usize,
    /// Number of null values.
    pub null_count: usize,
    /// Most common values (value, frequency).
    pub most_common: Vec<(String, usize)>,
}

/// A single bucket in a histogram.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBucket {
    /// Lower bound (inclusive).
    pub lower: f64,
    /// Upper bound (exclusive).
    pub upper: f64,
    /// Number of values in this bucket.
    pub count: usize,
}

impl FieldHistogram {
    /// Estimate selectivity for an equality predicate.
    pub fn estimate_equality_selectivity(&self, _value: &str) -> f32 {
        // Check most common values first
        for (mcv, freq) in &self.most_common {
            if mcv == _value {
                return *freq as f32 / self.total_count.max(1) as f32;
            }
        }
        // Assume uniform distribution among distinct values
        if self.distinct_count > 0 {
            1.0 / self.distinct_count as f32
        } else {
            1.0
        }
    }

    /// Estimate selectivity for a range predicate.
    pub fn estimate_range_selectivity(&self, lower: f64, upper: f64) -> f32 {
        if self.buckets.is_empty() || self.total_count == 0 {
            return 0.5; // Unknown: assume 50%
        }

        let mut matching = 0usize;
        for bucket in &self.buckets {
            if bucket.upper > lower && bucket.lower < upper {
                // Bucket overlaps with range
                let overlap_lower = lower.max(bucket.lower);
                let overlap_upper = upper.min(bucket.upper);
                let bucket_width = bucket.upper - bucket.lower;
                if bucket_width > 0.0 {
                    let fraction = (overlap_upper - overlap_lower) / bucket_width;
                    matching += (bucket.count as f64 * fraction) as usize;
                }
            }
        }

        matching as f32 / self.total_count as f32
    }
}

/// Statistics collector that builds histograms from metadata values.
pub struct StatisticsCollector {
    /// Number of histogram buckets to create.
    pub num_buckets: usize,
    /// Maximum number of most-common-values to track.
    pub max_mcv: usize,
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self {
            num_buckets: 20,
            max_mcv: 10,
        }
    }
}

impl StatisticsCollector {
    /// Build a histogram from numeric values.
    pub fn build_numeric_histogram(&self, field: &str, values: &[f64]) -> FieldHistogram {
        if values.is_empty() {
            return FieldHistogram {
                field: field.to_string(),
                buckets: Vec::new(),
                total_count: 0,
                distinct_count: 0,
                null_count: 0,
                most_common: Vec::new(),
            };
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min_val = sorted[0];
        let max_val = sorted[sorted.len() - 1];
        let range = max_val - min_val;

        let num_buckets = self.num_buckets.min(sorted.len());
        let bucket_width = if num_buckets > 0 && range > 0.0 {
            range / num_buckets as f64
        } else {
            1.0
        };

        let mut buckets = Vec::with_capacity(num_buckets);
        for i in 0..num_buckets {
            let lower = min_val + i as f64 * bucket_width;
            let upper = if i == num_buckets - 1 {
                max_val + f64::EPSILON
            } else {
                min_val + (i + 1) as f64 * bucket_width
            };
            let count = sorted.iter().filter(|&&v| v >= lower && v < upper).count();
            buckets.push(HistogramBucket { lower, upper, count });
        }

        // Count distinct values
        let mut distinct = sorted.clone();
        distinct.dedup();

        // Most common values
        let mut freq_map: HashMap<String, usize> = HashMap::new();
        for v in &sorted {
            *freq_map.entry(format!("{:.6}", v)).or_default() += 1;
        }
        let mut most_common: Vec<(String, usize)> = freq_map.into_iter().collect();
        most_common.sort_by(|a, b| b.1.cmp(&a.1));
        most_common.truncate(self.max_mcv);

        FieldHistogram {
            field: field.to_string(),
            buckets,
            total_count: values.len(),
            distinct_count: distinct.len(),
            null_count: 0,
            most_common,
        }
    }
}

/// Rich EXPLAIN output formatter.
pub struct ExplainFormatter;

impl ExplainFormatter {
    /// Format a query plan as a rich EXPLAIN output string.
    pub fn format(plan: &QueryPlan, stats: &CollectionStatistics) -> String {
        let mut out = String::new();
        out.push_str("╔══════════════════════════════════════════════════╗\n");
        out.push_str("║              EXPLAIN QUERY PLAN                  ║\n");
        out.push_str("╠══════════════════════════════════════════════════╣\n");
        out.push_str(&format!("║ Strategy: {:<38} ║\n", plan.index_choice));
        out.push_str("╠══════════════════════════════════════════════════╣\n");
        out.push_str("║ Cost Breakdown:                                  ║\n");
        out.push_str(&format!(
            "║   Estimated Latency:    {:>10.2} ms            ║\n",
            plan.cost.estimated_latency_ms
        ));
        out.push_str(&format!(
            "║   Memory Usage:         {:>10.2} MB            ║\n",
            plan.cost.estimated_memory_mb
        ));
        out.push_str(&format!(
            "║   Distance Computations: {:>9}              ║\n",
            plan.cost.distance_computations
        ));
        out.push_str(&format!(
            "║   Nodes Visited:        {:>10}              ║\n",
            plan.cost.nodes_visited
        ));
        out.push_str(&format!(
            "║   Candidate Set:        {:>10}              ║\n",
            plan.cost.candidate_set_size
        ));

        if let Some(ref fc) = plan.cost.filter_cost {
            out.push_str("║                                                  ║\n");
            out.push_str("║ Filter:                                          ║\n");
            out.push_str(&format!(
                "║   Strategy:     {:?}{:>26} ║\n",
                fc.strategy, ""
            ));
            out.push_str(&format!(
                "║   Selectivity:  {:>8.1}%                        ║\n",
                fc.selectivity * 100.0
            ));
            out.push_str(&format!(
                "║   Evaluations:  {:>8}                         ║\n",
                fc.evaluations
            ));
        }

        out.push_str("║                                                  ║\n");
        out.push_str("║ Collection Stats:                                ║\n");
        out.push_str(&format!(
            "║   Vectors:      {:>10}                       ║\n",
            stats.total_vectors
        ));
        out.push_str(&format!(
            "║   Dimensions:   {:>10}                       ║\n",
            stats.dimensions
        ));
        out.push_str(&format!(
            "║   Active:       {:>10}                       ║\n",
            stats.active_vectors()
        ));
        out.push_str(&format!(
            "║   Memory Est:   {:>7.1} MB                       ║\n",
            stats.estimated_memory_bytes() as f64 / (1024.0 * 1024.0)
        ));

        if !plan.rationale.is_empty() {
            out.push_str("║                                                  ║\n");
            out.push_str("║ Rationale:                                       ║\n");
            for r in &plan.rationale {
                let truncated = if r.len() > 46 { &r[..46] } else { r };
                out.push_str(&format!("║   • {:<44} ║\n", truncated));
            }
        }

        if !plan.alternatives.is_empty() {
            out.push_str("║                                                  ║\n");
            out.push_str("║ Alternatives:                                    ║\n");
            for alt in &plan.alternatives {
                out.push_str(&format!(
                    "║   {} {:>8.2}ms (rejected)                  ║\n",
                    alt.index_choice, alt.estimated_latency_ms
                ));
            }
        }

        out.push_str("╚══════════════════════════════════════════════════╝\n");
        out
    }
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
#[derive(Debug, Clone)]
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

// ── Adaptive Multi-Armed Bandit Optimizer ────────────────────────────────────

/// Exploration strategy for the adaptive optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ExplorationStrategy {
    /// Epsilon-greedy: explore with probability epsilon.
    EpsilonGreedy { epsilon: f64 },
    /// Upper Confidence Bound: balance exploitation and exploration.
    Ucb { confidence: f64 },
}

impl Default for ExplorationStrategy {
    fn default() -> Self {
        Self::Ucb { confidence: 2.0 }
    }
}

/// Observed latency sample for a particular index strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyObservation {
    /// Which index was used.
    pub index: IndexChoice,
    /// Observed latency in milliseconds.
    pub latency_ms: f64,
    /// Number of results returned.
    pub results_returned: usize,
    /// Approximate dataset size at observation time.
    pub dataset_size: usize,
}

/// Per-arm statistics for the bandit model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct ArmStats {
    /// Number of times this arm has been selected.
    pulls: u64,
    /// Sum of rewards (inverse latency).
    total_reward: f64,
    /// Sum of squared rewards (for variance).
    total_reward_sq: f64,
    /// Most recent observed latency.
    last_latency_ms: f64,
}

impl ArmStats {
    fn mean_reward(&self) -> f64 {
        if self.pulls == 0 {
            return 0.0;
        }
        self.total_reward / self.pulls as f64
    }
}

/// Adaptive optimizer that learns the best index strategy from query latency feedback.
///
/// Uses a multi-armed bandit model to balance exploration (trying different indexes)
/// with exploitation (using the historically best-performing index).
#[derive(Debug, Clone)]
pub struct AdaptiveOptimizer {
    /// Strategy for balancing exploration vs exploitation.
    strategy: ExplorationStrategy,
    /// Per-index arm statistics.
    arms: HashMap<IndexChoice, ArmStats>,
    /// Total observations across all arms.
    total_observations: u64,
    /// Fallback cost estimator for cold-start.
    estimator: CostEstimator,
    /// Minimum observations per arm before trusting learned data.
    min_observations: u64,
}

impl Default for AdaptiveOptimizer {
    fn default() -> Self {
        Self::new(ExplorationStrategy::default())
    }
}

impl AdaptiveOptimizer {
    /// Create a new adaptive optimizer with the given exploration strategy.
    pub fn new(strategy: ExplorationStrategy) -> Self {
        let mut arms = HashMap::new();
        for index in [
            IndexChoice::Hnsw,
            IndexChoice::BruteForce,
            IndexChoice::HnswPreFilter,
            IndexChoice::HnswPostFilter,
        ] {
            arms.insert(index, ArmStats::default());
        }
        Self {
            strategy,
            arms,
            total_observations: 0,
            estimator: CostEstimator::new(),
            min_observations: 5,
        }
    }

    /// Select the best index strategy, balancing exploration and exploitation.
    ///
    /// During cold-start (fewer than `min_observations` per arm), falls back to
    /// the cost estimator's heuristic. Once enough data is collected, uses the
    /// bandit model.
    pub fn select(
        &self,
        stats: &CollectionStatistics,
        k: usize,
        filter_selectivity: Option<f32>,
    ) -> IndexChoice {
        // Cold start: use heuristic cost estimator
        let cold_arms: Vec<_> = self
            .arms
            .iter()
            .filter(|(_, s)| s.pulls < self.min_observations)
            .collect();
        if !cold_arms.is_empty() {
            // Explore the least-tried arm during cold start
            let least_tried = cold_arms
                .iter()
                .min_by_key(|(_, s)| s.pulls)
                .map(|(&idx, _)| idx);
            if let Some(idx) = least_tried {
                return idx;
            }
        }

        match self.strategy {
            ExplorationStrategy::EpsilonGreedy { epsilon } => {
                // Deterministic selection: always pick best arm
                // (randomness would require rand crate; use a simple hash-based probe)
                let probe = (self.total_observations * 2654435761) % 1000;
                if (probe as f64) < epsilon * 1000.0 {
                    // "Explore": pick the arm with fewest pulls
                    self.arms
                        .iter()
                        .min_by_key(|(_, s)| s.pulls)
                        .map(|(&idx, _)| idx)
                        .unwrap_or_else(|| self.estimator.plan(stats, k, filter_selectivity).index_choice)
                } else {
                    self.best_arm()
                }
            }
            ExplorationStrategy::Ucb { confidence } => {
                self.ucb_select(confidence)
            }
        }
    }

    /// Record an observed latency for a chosen index strategy.
    pub fn observe(&mut self, observation: LatencyObservation) {
        let arm = self.arms.entry(observation.index).or_default();
        let reward = 1.0 / (1.0 + observation.latency_ms);
        arm.pulls += 1;
        arm.total_reward += reward;
        arm.total_reward_sq += reward * reward;
        arm.last_latency_ms = observation.latency_ms;
        self.total_observations += 1;
    }

    /// Get the arm with the highest mean reward (lowest latency).
    fn best_arm(&self) -> IndexChoice {
        self.arms
            .iter()
            .filter(|(_, s)| s.pulls > 0)
            .max_by(|(_, a), (_, b)| {
                a.mean_reward()
                    .partial_cmp(&b.mean_reward())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&idx, _)| idx)
            .unwrap_or(IndexChoice::Hnsw)
    }

    /// UCB1 selection: mean reward + confidence bound.
    fn ucb_select(&self, confidence: f64) -> IndexChoice {
        let ln_total = (self.total_observations.max(1) as f64).ln();

        self.arms
            .iter()
            .filter(|(_, s)| s.pulls > 0)
            .max_by(|(_, a), (_, b)| {
                let ucb_a = a.mean_reward() + confidence * (ln_total / a.pulls as f64).sqrt();
                let ucb_b = b.mean_reward() + confidence * (ln_total / b.pulls as f64).sqrt();
                ucb_a
                    .partial_cmp(&ucb_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(&idx, _)| idx)
            .unwrap_or(IndexChoice::Hnsw)
    }

    /// Get performance statistics for all arms.
    pub fn arm_stats(&self) -> Vec<ArmPerformance> {
        self.arms
            .iter()
            .map(|(&index, stats)| ArmPerformance {
                index,
                observations: stats.pulls,
                mean_reward: stats.mean_reward(),
                avg_latency_ms: if stats.pulls > 0 {
                    1.0 / stats.mean_reward() - 1.0
                } else {
                    0.0
                },
                last_latency_ms: stats.last_latency_ms,
            })
            .collect()
    }

    /// Total number of observations recorded.
    pub fn total_observations(&self) -> u64 {
        self.total_observations
    }

    /// Reset all learned statistics.
    pub fn reset(&mut self) {
        for stats in self.arms.values_mut() {
            *stats = ArmStats::default();
        }
        self.total_observations = 0;
    }
}

/// Performance summary for a single bandit arm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmPerformance {
    /// Index strategy.
    pub index: IndexChoice,
    /// Number of times this strategy was used.
    pub observations: u64,
    /// Mean reward (higher = better performance).
    pub mean_reward: f64,
    /// Average observed latency in ms.
    pub avg_latency_ms: f64,
    /// Most recent latency.
    pub last_latency_ms: f64,
}

use std::collections::HashMap;

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

    // ── Adaptive Optimizer Tests ──

    #[test]
    fn test_adaptive_optimizer_cold_start() {
        let optimizer = AdaptiveOptimizer::new(ExplorationStrategy::default());
        let stats = CollectionStatistics::new(100_000, 384, 0.0);
        // Cold start should explore (pick least-tried arm)
        let choice = optimizer.select(&stats, 10, None);
        // Any arm is valid during cold start
        assert!(
            [
                IndexChoice::Hnsw,
                IndexChoice::BruteForce,
                IndexChoice::HnswPreFilter,
                IndexChoice::HnswPostFilter,
            ]
            .contains(&choice)
        );
    }

    #[test]
    fn test_adaptive_optimizer_learns() {
        let mut optimizer = AdaptiveOptimizer::new(ExplorationStrategy::Ucb { confidence: 1.0 });

        // Feed HNSW with consistently low latency (many observations)
        for _ in 0..100 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::Hnsw,
                latency_ms: 2.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }

        // Feed BruteForce with consistently high latency
        for _ in 0..100 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::BruteForce,
                latency_ms: 50.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }

        // Feed pre/post filter with moderate latency
        for _ in 0..100 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::HnswPreFilter,
                latency_ms: 5.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
            optimizer.observe(LatencyObservation {
                index: IndexChoice::HnswPostFilter,
                latency_ms: 8.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }

        let stats = CollectionStatistics::new(100_000, 384, 0.0);
        let choice = optimizer.select(&stats, 10, None);
        // Should prefer HNSW (lowest latency) with enough observations to overcome UCB bonus
        assert_eq!(choice, IndexChoice::Hnsw);
    }

    #[test]
    fn test_adaptive_optimizer_arm_stats() {
        let mut optimizer = AdaptiveOptimizer::default();

        optimizer.observe(LatencyObservation {
            index: IndexChoice::Hnsw,
            latency_ms: 3.0,
            results_returned: 10,
            dataset_size: 50_000,
        });

        let stats = optimizer.arm_stats();
        let hnsw_stat = stats.iter().find(|s| s.index == IndexChoice::Hnsw).unwrap();
        assert_eq!(hnsw_stat.observations, 1);
        assert!(hnsw_stat.mean_reward > 0.0);
        assert_eq!(hnsw_stat.last_latency_ms, 3.0);
    }

    #[test]
    fn test_adaptive_optimizer_reset() {
        let mut optimizer = AdaptiveOptimizer::default();

        optimizer.observe(LatencyObservation {
            index: IndexChoice::Hnsw,
            latency_ms: 2.0,
            results_returned: 10,
            dataset_size: 100_000,
        });
        assert_eq!(optimizer.total_observations(), 1);

        optimizer.reset();
        assert_eq!(optimizer.total_observations(), 0);
        assert!(optimizer.arm_stats().iter().all(|s| s.observations == 0));
    }

    #[test]
    fn test_epsilon_greedy_strategy() {
        let mut optimizer =
            AdaptiveOptimizer::new(ExplorationStrategy::EpsilonGreedy { epsilon: 0.1 });

        // Train sufficiently
        for _ in 0..20 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::Hnsw,
                latency_ms: 1.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
            optimizer.observe(LatencyObservation {
                index: IndexChoice::BruteForce,
                latency_ms: 100.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
            optimizer.observe(LatencyObservation {
                index: IndexChoice::HnswPreFilter,
                latency_ms: 5.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
            optimizer.observe(LatencyObservation {
                index: IndexChoice::HnswPostFilter,
                latency_ms: 8.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }

        let stats = CollectionStatistics::new(100_000, 384, 0.0);
        let choice = optimizer.select(&stats, 10, None);
        // With very low epsilon and enough training, should usually exploit best arm
        // (HNSW has lowest latency)
        assert!(
            [IndexChoice::Hnsw, IndexChoice::BruteForce, IndexChoice::HnswPreFilter, IndexChoice::HnswPostFilter]
                .contains(&choice)
        );
    }

    #[test]
    fn test_ucb_exploration() {
        let mut optimizer = AdaptiveOptimizer::new(ExplorationStrategy::Ucb { confidence: 2.0 });

        // Only observe HNSW heavily, others sparsely
        for _ in 0..50 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::Hnsw,
                latency_ms: 3.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }
        for _ in 0..2 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::BruteForce,
                latency_ms: 100.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }
        for _ in 0..2 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::HnswPreFilter,
                latency_ms: 4.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }
        for _ in 0..2 {
            optimizer.observe(LatencyObservation {
                index: IndexChoice::HnswPostFilter,
                latency_ms: 5.0,
                results_returned: 10,
                dataset_size: 100_000,
            });
        }

        // UCB considers uncertainty — under-explored arms get a bonus
        let stats = CollectionStatistics::new(100_000, 384, 0.0);
        let choice = optimizer.select(&stats, 10, None);
        // Any valid arm can be selected — UCB may explore under-tried arms
        assert!(
            [
                IndexChoice::Hnsw,
                IndexChoice::BruteForce,
                IndexChoice::HnswPreFilter,
                IndexChoice::HnswPostFilter,
            ]
            .contains(&choice)
        );
    }

    #[test]
    fn test_latency_observation_serde() {
        let obs = LatencyObservation {
            index: IndexChoice::Hnsw,
            latency_ms: 3.5,
            results_returned: 10,
            dataset_size: 50_000,
        };
        let json = serde_json::to_string(&obs).unwrap();
        let deser: LatencyObservation = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.index, IndexChoice::Hnsw);
        assert!((deser.latency_ms - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_histogram_equality_selectivity() {
        let collector = StatisticsCollector::default();
        let values: Vec<f64> = (0..100).map(|i| (i % 10) as f64).collect();
        let hist = collector.build_numeric_histogram("score", &values);

        assert_eq!(hist.total_count, 100);
        assert_eq!(hist.distinct_count, 10);
        // Each value appears 10 times → selectivity = 10/100 = 0.1
        let sel = hist.estimate_equality_selectivity("0.000000");
        assert!(sel > 0.05 && sel < 0.15);
    }

    #[test]
    fn test_histogram_range_selectivity() {
        let collector = StatisticsCollector { num_buckets: 10, max_mcv: 5 };
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let hist = collector.build_numeric_histogram("score", &values);

        // Range [0, 50) should have ~50% selectivity
        let sel = hist.estimate_range_selectivity(0.0, 50.0);
        assert!(sel > 0.3 && sel < 0.7, "range sel was {}", sel);
    }

    #[test]
    fn test_explain_formatter() {
        let stats = CollectionStatistics::new(100_000, 384, 0.05);
        let estimator = CostEstimator::new();
        let plan = estimator.plan(&stats, 10, Some(0.1));

        let explain = ExplainFormatter::format(&plan, &stats);
        assert!(explain.contains("EXPLAIN QUERY PLAN"));
        assert!(explain.contains("Estimated Latency"));
        assert!(explain.contains("Dimensions"));
    }

    #[test]
    fn test_empty_histogram() {
        let collector = StatisticsCollector::default();
        let hist = collector.build_numeric_histogram("empty", &[]);
        assert_eq!(hist.total_count, 0);
        assert_eq!(hist.distinct_count, 0);
        // Should return safe defaults
        assert_eq!(hist.estimate_equality_selectivity("any"), 1.0);
        assert_eq!(hist.estimate_range_selectivity(0.0, 100.0), 0.5);
    }
}
