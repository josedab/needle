//! Query Cost Optimizer with EXPLAIN
//!
//! SQL-like EXPLAIN for vector queries showing index scan paths, filter
//! selectivity estimates, and automatic ef_search tuning with learning
//! from query feedback.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::query_explain::*;
//!
//! let mut optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());
//!
//! // Explain a query plan
//! let plan = optimizer.explain(&QuerySpec {
//!     k: 10,
//!     dimension: 384,
//!     has_filter: true,
//!     filter_selectivity: Some(0.1),
//!     collection_size: 1_000_000,
//!     ef_search: None,
//! });
//!
//! println!("{}", plan.format_tree());
//! ```

use std::collections::VecDeque;
use std::time::Duration;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Query Specification
// ---------------------------------------------------------------------------

/// Description of a query for the optimizer to plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySpec {
    /// Number of results requested.
    pub k: usize,
    /// Vector dimensionality.
    pub dimension: usize,
    /// Whether a metadata filter is applied.
    pub has_filter: bool,
    /// Estimated filter selectivity (fraction of vectors passing the filter).
    pub filter_selectivity: Option<f64>,
    /// Number of vectors in the collection.
    pub collection_size: usize,
    /// Override ef_search (None = auto-tune).
    pub ef_search: Option<usize>,
}

// ---------------------------------------------------------------------------
// Query Plan
// ---------------------------------------------------------------------------

/// A step in the query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step name.
    pub name: String,
    /// Description of what this step does.
    pub description: String,
    /// Estimated cost (arbitrary units, lower is better).
    pub estimated_cost: f64,
    /// Estimated number of rows/candidates processed.
    pub estimated_rows: usize,
    /// Sub-steps.
    pub children: Vec<PlanStep>,
}

/// A complete query execution plan with cost estimates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplainPlan {
    /// Root steps of the plan.
    pub steps: Vec<PlanStep>,
    /// Chosen strategy.
    pub strategy: ExplainStrategy,
    /// Recommended ef_search parameter.
    pub recommended_ef_search: usize,
    /// Estimated total cost.
    pub total_cost: f64,
    /// Estimated latency.
    pub estimated_latency_ms: f64,
    /// Whether the plan uses post-filtering or pre-filtering.
    pub filter_mode: FilterMode,
    /// Warnings and suggestions.
    pub warnings: Vec<String>,
}

impl ExplainPlan {
    /// Format the plan as a human-readable tree string.
    pub fn format_tree(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "EXPLAIN (strategy: {:?}, est. latency: {:.1}ms, ef_search: {})\n",
            self.strategy, self.estimated_latency_ms, self.recommended_ef_search
        ));
        output.push_str(&format!(
            "  Filter mode: {:?}, Total cost: {:.1}\n",
            self.filter_mode, self.total_cost
        ));

        for step in &self.steps {
            Self::format_step(&mut output, step, 1);
        }

        if !self.warnings.is_empty() {
            output.push_str("\nWarnings:\n");
            for w in &self.warnings {
                output.push_str(&format!("  ⚠ {}\n", w));
            }
        }

        output
    }

    fn format_step(output: &mut String, step: &PlanStep, depth: usize) {
        let indent = "  ".repeat(depth);
        output.push_str(&format!(
            "{}→ {} (cost: {:.1}, rows: {})\n",
            indent, step.name, step.estimated_cost, step.estimated_rows
        ));
        output.push_str(&format!("{}  {}\n", indent, step.description));
        for child in &step.children {
            Self::format_step(output, child, depth + 1);
        }
    }
}

/// The chosen execution strategy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExplainStrategy {
    /// Pure HNSW search, filter applied post-search.
    HnswPostFilter,
    /// Pre-filter metadata, then search within candidates.
    PreFilterThenSearch,
    /// HNSW search + metadata filter interleaved.
    InterleavedFilter,
    /// Brute-force scan (small collections).
    BruteForce,
    /// Exact (exhaustive) search.
    ExactSearch,
}

/// How filters are applied relative to the vector search.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FilterMode {
    /// No filter applied.
    None,
    /// Filter applied after vector search.
    PostFilter,
    /// Filter applied before vector search.
    PreFilter,
    /// Filter interleaved with vector search.
    Interleaved,
}

// ---------------------------------------------------------------------------
// Cost Model
// ---------------------------------------------------------------------------

/// Parameters for the cost model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModelParams {
    /// Cost per HNSW hop (microseconds).
    pub hnsw_hop_cost_us: f64,
    /// Cost per distance computation (microseconds).
    pub distance_compute_cost_us: f64,
    /// Cost per metadata filter check (microseconds).
    pub filter_check_cost_us: f64,
    /// Cost per result marshalling (microseconds).
    pub result_marshal_cost_us: f64,
    /// Base overhead per query (microseconds).
    pub query_overhead_us: f64,
}

impl Default for CostModelParams {
    fn default() -> Self {
        Self {
            hnsw_hop_cost_us: 0.5,
            distance_compute_cost_us: 0.1,
            filter_check_cost_us: 0.05,
            result_marshal_cost_us: 1.0,
            query_overhead_us: 50.0,
        }
    }
}

/// Cost model for estimating query execution cost.
pub struct CostModel {
    params: CostModelParams,
}

impl CostModel {
    pub fn new(params: CostModelParams) -> Self {
        Self { params }
    }

    /// Estimate cost for HNSW search.
    pub fn hnsw_cost(&self, ef_search: usize, m: usize, _dimension: usize) -> f64 {
        let hops = (ef_search as f64 * (m as f64).ln()).max(1.0);
        let distance_ops = hops * m as f64;
        self.params.query_overhead_us
            + hops * self.params.hnsw_hop_cost_us
            + distance_ops * self.params.distance_compute_cost_us
    }

    /// Estimate cost for post-filter.
    pub fn post_filter_cost(&self, candidates: usize) -> f64 {
        candidates as f64 * self.params.filter_check_cost_us
    }

    /// Estimate cost for pre-filter.
    pub fn pre_filter_cost(&self, collection_size: usize) -> f64 {
        collection_size as f64 * self.params.filter_check_cost_us
    }

    /// Estimate cost for brute-force scan.
    pub fn brute_force_cost(&self, n: usize, dimension: usize) -> f64 {
        self.params.query_overhead_us
            + n as f64 * dimension as f64 * self.params.distance_compute_cost_us
    }

    /// Convert cost units to estimated milliseconds.
    pub fn cost_to_ms(&self, cost: f64) -> f64 {
        cost / 1000.0
    }
}

// ---------------------------------------------------------------------------
// Adaptive ef_search Tuner
// ---------------------------------------------------------------------------

/// Feedback from an executed query for adaptive tuning.
#[derive(Debug, Clone)]
pub struct QueryFeedback {
    pub ef_search_used: usize,
    pub actual_latency: Duration,
    pub results_returned: usize,
    pub had_filter: bool,
    pub collection_size: usize,
}

/// Adaptive tuner that learns optimal ef_search from query feedback.
pub struct AdaptiveEfTuner {
    /// Recent feedback observations.
    history: VecDeque<QueryFeedback>,
    /// Maximum history size.
    max_history: usize,
    /// Current recommended ef_search.
    current_ef: usize,
    /// Target latency.
    target_latency: Duration,
    /// Learning rate for adjustments.
    learning_rate: f64,
}

impl AdaptiveEfTuner {
    pub fn new(initial_ef: usize, target_latency: Duration) -> Self {
        Self {
            history: VecDeque::with_capacity(100),
            max_history: 100,
            current_ef: initial_ef,
            target_latency,
            learning_rate: 0.1,
        }
    }

    /// Record query feedback and adjust ef_search.
    pub fn record_feedback(&mut self, feedback: QueryFeedback) {
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(feedback.clone());

        // Adjust ef_search based on latency feedback
        if feedback.actual_latency > self.target_latency {
            // Latency too high — decrease ef
            let reduction = (self.current_ef as f64 * self.learning_rate).max(1.0) as usize;
            self.current_ef = self.current_ef.saturating_sub(reduction).max(10);
        } else if feedback.actual_latency < self.target_latency / 2 {
            // Latency has room — increase ef for better recall
            let increase = (self.current_ef as f64 * self.learning_rate).max(1.0) as usize;
            self.current_ef = (self.current_ef + increase).min(500);
        }
    }

    /// Get the current recommended ef_search.
    pub fn recommended_ef(&self) -> usize {
        self.current_ef
    }

    /// Average observed latency.
    pub fn avg_latency(&self) -> Duration {
        if self.history.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.history.iter().map(|f| f.actual_latency).sum();
        total / self.history.len() as u32
    }

    /// Number of observations.
    pub fn observation_count(&self) -> usize {
        self.history.len()
    }
}

// ---------------------------------------------------------------------------
// Optimizer
// ---------------------------------------------------------------------------

/// Configuration for the query explain optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Cost model parameters.
    pub cost_params: CostModelParams,
    /// Default ef_search.
    pub default_ef_search: usize,
    /// Default M for HNSW.
    pub default_m: usize,
    /// Threshold below which brute-force is preferred.
    pub brute_force_threshold: usize,
    /// Target query latency for adaptive tuning.
    pub target_latency_ms: u64,
    /// Enable adaptive ef_search tuning.
    pub adaptive_tuning: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            cost_params: CostModelParams::default(),
            default_ef_search: 50,
            default_m: 16,
            brute_force_threshold: 1000,
            target_latency_ms: 10,
            adaptive_tuning: true,
        }
    }
}

/// The query explain optimizer that generates execution plans and tunes parameters.
pub struct QueryExplainOptimizer {
    config: OptimizerConfig,
    cost_model: CostModel,
    tuner: RwLock<AdaptiveEfTuner>,
    queries_planned: RwLock<u64>,
}

impl QueryExplainOptimizer {
    /// Create a new optimizer.
    pub fn new(config: OptimizerConfig) -> Self {
        let cost_model = CostModel::new(config.cost_params.clone());
        let tuner = AdaptiveEfTuner::new(
            config.default_ef_search,
            Duration::from_millis(config.target_latency_ms),
        );

        Self {
            config,
            cost_model,
            tuner: RwLock::new(tuner),
            queries_planned: RwLock::new(0),
        }
    }

    /// Generate an EXPLAIN plan for a query.
    pub fn explain(&self, query: &QuerySpec) -> ExplainPlan {
        *self.queries_planned.write() += 1;

        let ef_search = query
            .ef_search
            .unwrap_or_else(|| self.tuner.read().recommended_ef());

        let mut warnings = Vec::new();

        // Decide strategy
        let (strategy, filter_mode) = self.choose_strategy(query);

        // Build plan steps
        let steps = self.build_plan_steps(query, &strategy, ef_search);

        // Estimate total cost
        let total_cost: f64 = steps.iter().map(|s| s.estimated_cost).sum();
        let estimated_latency = self.cost_model.cost_to_ms(total_cost);

        // Generate warnings
        if ef_search < 20 && query.collection_size > 100_000 {
            warnings.push(format!(
                "Low ef_search ({}) with large collection ({} vectors) may reduce recall",
                ef_search, query.collection_size
            ));
        }

        if let Some(selectivity) = query.filter_selectivity {
            if selectivity < 0.01 {
                warnings.push(format!(
                    "Very selective filter ({:.1}%) — consider pre-filtering for better performance",
                    selectivity * 100.0
                ));
            }
        }

        if query.k > ef_search {
            warnings.push(format!(
                "k ({}) > ef_search ({}) — results may have poor recall",
                query.k, ef_search
            ));
        }

        ExplainPlan {
            steps,
            strategy,
            recommended_ef_search: ef_search,
            total_cost,
            estimated_latency_ms: estimated_latency,
            filter_mode,
            warnings,
        }
    }

    /// Choose the best execution strategy for a query.
    fn choose_strategy(&self, query: &QuerySpec) -> (ExplainStrategy, FilterMode) {
        // Small collection → brute force
        if query.collection_size <= self.config.brute_force_threshold {
            if query.has_filter {
                return (ExplainStrategy::BruteForce, FilterMode::PreFilter);
            }
            return (ExplainStrategy::BruteForce, FilterMode::None);
        }

        if !query.has_filter {
            return (ExplainStrategy::HnswPostFilter, FilterMode::None);
        }

        // Choose filter strategy based on selectivity
        let selectivity = query.filter_selectivity.unwrap_or(0.5);

        if selectivity < 0.05 {
            // Very selective → pre-filter is cheaper
            (ExplainStrategy::PreFilterThenSearch, FilterMode::PreFilter)
        } else if selectivity > 0.8 {
            // Low selectivity → post-filter is fine
            (ExplainStrategy::HnswPostFilter, FilterMode::PostFilter)
        } else {
            // Medium selectivity → interleaved is best
            (ExplainStrategy::InterleavedFilter, FilterMode::Interleaved)
        }
    }

    /// Build the plan step tree.
    fn build_plan_steps(
        &self,
        query: &QuerySpec,
        strategy: &ExplainStrategy,
        ef_search: usize,
    ) -> Vec<PlanStep> {
        match strategy {
            ExplainStrategy::HnswPostFilter => {
                let hnsw_cost =
                    self.cost_model
                        .hnsw_cost(ef_search, self.config.default_m, query.dimension);
                let search_step = PlanStep {
                    name: "HNSW Search".into(),
                    description: format!(
                        "Traverse HNSW graph with ef_search={}, M={}",
                        ef_search, self.config.default_m
                    ),
                    estimated_cost: hnsw_cost,
                    estimated_rows: ef_search,
                    children: vec![],
                };

                let mut steps = vec![search_step];

                if query.has_filter {
                    let filter_cost = self.cost_model.post_filter_cost(ef_search);
                    let selectivity = query.filter_selectivity.unwrap_or(0.5);
                    steps.push(PlanStep {
                        name: "Post-Filter".into(),
                        description: format!(
                            "Apply metadata filter (selectivity: {:.0}%)",
                            selectivity * 100.0
                        ),
                        estimated_cost: filter_cost,
                        estimated_rows: (ef_search as f64 * selectivity) as usize,
                        children: vec![],
                    });
                }

                steps
            }
            ExplainStrategy::PreFilterThenSearch => {
                let filter_cost = self.cost_model.pre_filter_cost(query.collection_size);
                let selectivity = query.filter_selectivity.unwrap_or(0.05);
                let filtered_size = (query.collection_size as f64 * selectivity) as usize;

                vec![
                    PlanStep {
                        name: "Pre-Filter".into(),
                        description: format!(
                            "Scan {} metadata entries (selectivity: {:.1}%)",
                            query.collection_size,
                            selectivity * 100.0
                        ),
                        estimated_cost: filter_cost,
                        estimated_rows: filtered_size,
                        children: vec![],
                    },
                    PlanStep {
                        name: "HNSW Search (filtered)".into(),
                        description: format!("Search within {} filtered candidates", filtered_size),
                        estimated_cost: self.cost_model.hnsw_cost(
                            ef_search.min(filtered_size),
                            self.config.default_m,
                            query.dimension,
                        ),
                        estimated_rows: query.k.min(filtered_size),
                        children: vec![],
                    },
                ]
            }
            ExplainStrategy::InterleavedFilter => {
                let hnsw_cost = self.cost_model.hnsw_cost(
                    ef_search * 2, // Need to explore more due to filtering
                    self.config.default_m,
                    query.dimension,
                );
                let selectivity = query.filter_selectivity.unwrap_or(0.5);

                vec![PlanStep {
                    name: "Interleaved HNSW + Filter".into(),
                    description: format!(
                        "HNSW search with inline filter check (selectivity: {:.0}%)",
                        selectivity * 100.0
                    ),
                    estimated_cost: hnsw_cost,
                    estimated_rows: query.k,
                    children: vec![
                        PlanStep {
                            name: "HNSW Traverse".into(),
                            description: format!(
                                "ef_search={} (2x for filter overhead)",
                                ef_search * 2
                            ),
                            estimated_cost: hnsw_cost * 0.7,
                            estimated_rows: ef_search * 2,
                            children: vec![],
                        },
                        PlanStep {
                            name: "Inline Filter".into(),
                            description: "Check metadata during traversal".into(),
                            estimated_cost: hnsw_cost * 0.3,
                            estimated_rows: (ef_search as f64 * 2.0 * selectivity) as usize,
                            children: vec![],
                        },
                    ],
                }]
            }
            ExplainStrategy::BruteForce | ExplainStrategy::ExactSearch => {
                let cost = self
                    .cost_model
                    .brute_force_cost(query.collection_size, query.dimension);
                vec![PlanStep {
                    name: "Brute-Force Scan".into(),
                    description: format!(
                        "Linear scan of {} vectors ({}-dim)",
                        query.collection_size, query.dimension
                    ),
                    estimated_cost: cost,
                    estimated_rows: query.collection_size,
                    children: vec![],
                }]
            }
        }
    }

    /// Record query execution feedback for adaptive tuning.
    pub fn record_feedback(&self, feedback: QueryFeedback) {
        if self.config.adaptive_tuning {
            self.tuner.write().record_feedback(feedback);
        }
    }

    /// Get the current recommended ef_search.
    pub fn recommended_ef_search(&self) -> usize {
        self.tuner.read().recommended_ef()
    }

    /// Number of queries planned.
    pub fn queries_planned(&self) -> u64 {
        *self.queries_planned.read()
    }

    /// Get optimizer statistics.
    pub fn stats(&self) -> OptimizerStats {
        let tuner = self.tuner.read();
        OptimizerStats {
            queries_planned: *self.queries_planned.read(),
            current_ef_search: tuner.recommended_ef(),
            avg_observed_latency_ms: tuner.avg_latency().as_millis() as f64,
            feedback_observations: tuner.observation_count(),
        }
    }
}

/// Statistics from the optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerStats {
    pub queries_planned: u64,
    pub current_ef_search: usize,
    pub avg_observed_latency_ms: f64,
    pub feedback_observations: usize,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explain_simple_query() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 384,
            has_filter: false,
            filter_selectivity: None,
            collection_size: 100_000,
            ef_search: None,
        });

        assert_eq!(plan.strategy, ExplainStrategy::HnswPostFilter);
        assert_eq!(plan.filter_mode, FilterMode::None);
        assert!(!plan.steps.is_empty());
        assert!(plan.total_cost > 0.0);
    }

    #[test]
    fn test_explain_with_filter_high_selectivity() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 384,
            has_filter: true,
            filter_selectivity: Some(0.9),
            collection_size: 100_000,
            ef_search: None,
        });

        assert_eq!(plan.strategy, ExplainStrategy::HnswPostFilter);
        assert_eq!(plan.filter_mode, FilterMode::PostFilter);
    }

    #[test]
    fn test_explain_with_filter_low_selectivity() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 384,
            has_filter: true,
            filter_selectivity: Some(0.02),
            collection_size: 1_000_000,
            ef_search: None,
        });

        assert_eq!(plan.strategy, ExplainStrategy::PreFilterThenSearch);
        assert_eq!(plan.filter_mode, FilterMode::PreFilter);
    }

    #[test]
    fn test_explain_small_collection_brute_force() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 128,
            has_filter: false,
            filter_selectivity: None,
            collection_size: 500,
            ef_search: None,
        });

        assert_eq!(plan.strategy, ExplainStrategy::BruteForce);
    }

    #[test]
    fn test_explain_interleaved_filter() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 384,
            has_filter: true,
            filter_selectivity: Some(0.3),
            collection_size: 500_000,
            ef_search: None,
        });

        assert_eq!(plan.strategy, ExplainStrategy::InterleavedFilter);
        assert_eq!(plan.filter_mode, FilterMode::Interleaved);
    }

    #[test]
    fn test_explain_format_tree() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 384,
            has_filter: true,
            filter_selectivity: Some(0.3),
            collection_size: 100_000,
            ef_search: Some(100),
        });

        let tree = plan.format_tree();
        assert!(tree.contains("EXPLAIN"));
        assert!(tree.contains("est. latency"));
    }

    #[test]
    fn test_explain_warnings() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 100,
            dimension: 384,
            has_filter: false,
            filter_selectivity: None,
            collection_size: 500_000,
            ef_search: Some(10), // Very low ef_search
        });

        assert!(!plan.warnings.is_empty());
    }

    #[test]
    fn test_adaptive_ef_tuner_decrease() {
        let mut tuner = AdaptiveEfTuner::new(50, Duration::from_millis(10));

        // Slow queries → ef should decrease
        for _ in 0..10 {
            tuner.record_feedback(QueryFeedback {
                ef_search_used: 50,
                actual_latency: Duration::from_millis(20), // Over target
                results_returned: 10,
                had_filter: false,
                collection_size: 100_000,
            });
        }

        assert!(tuner.recommended_ef() < 50);
    }

    #[test]
    fn test_adaptive_ef_tuner_increase() {
        let mut tuner = AdaptiveEfTuner::new(50, Duration::from_millis(10));

        // Fast queries → ef should increase
        for _ in 0..10 {
            tuner.record_feedback(QueryFeedback {
                ef_search_used: 50,
                actual_latency: Duration::from_millis(2), // Well under target
                results_returned: 10,
                had_filter: false,
                collection_size: 100_000,
            });
        }

        assert!(tuner.recommended_ef() > 50);
    }

    #[test]
    fn test_cost_model() {
        let model = CostModel::new(CostModelParams::default());
        let hnsw_cost = model.hnsw_cost(50, 16, 384);
        assert!(hnsw_cost > 0.0);

        let bf_cost = model.brute_force_cost(1_000_000, 384);
        assert!(bf_cost > hnsw_cost);
    }

    #[test]
    fn test_optimizer_stats() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());
        optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 128,
            has_filter: false,
            filter_selectivity: None,
            collection_size: 10_000,
            ef_search: None,
        });

        let stats = optimizer.stats();
        assert_eq!(stats.queries_planned, 1);
    }

    #[test]
    fn test_optimizer_feedback_integration() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());
        let initial_ef = optimizer.recommended_ef_search();

        optimizer.record_feedback(QueryFeedback {
            ef_search_used: initial_ef,
            actual_latency: Duration::from_millis(1), // Very fast
            results_returned: 10,
            had_filter: false,
            collection_size: 50_000,
        });

        // ef should increase (room for more recall)
        assert!(optimizer.recommended_ef_search() >= initial_ef);
    }

    #[test]
    fn test_explain_custom_ef_search() {
        let optimizer = QueryExplainOptimizer::new(OptimizerConfig::default());

        let plan = optimizer.explain(&QuerySpec {
            k: 10,
            dimension: 128,
            has_filter: false,
            filter_selectivity: None,
            collection_size: 50_000,
            ef_search: Some(200),
        });

        assert_eq!(plan.recommended_ef_search, 200);
    }

    #[test]
    fn test_plan_step_serialization() {
        let step = PlanStep {
            name: "HNSW Search".into(),
            description: "Test".into(),
            estimated_cost: 42.0,
            estimated_rows: 100,
            children: vec![],
        };
        let json = serde_json::to_string(&step).unwrap();
        let decoded: PlanStep = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.name, "HNSW Search");
    }

    #[test]
    fn test_tuner_min_ef() {
        let mut tuner = AdaptiveEfTuner::new(15, Duration::from_millis(1));
        for _ in 0..100 {
            tuner.record_feedback(QueryFeedback {
                ef_search_used: 15,
                actual_latency: Duration::from_millis(100),
                results_returned: 10,
                had_filter: false,
                collection_size: 100_000,
            });
        }
        // Should not go below 10
        assert!(tuner.recommended_ef() >= 10);
    }

    #[test]
    fn test_tuner_max_ef() {
        let mut tuner = AdaptiveEfTuner::new(400, Duration::from_millis(100));
        for _ in 0..100 {
            tuner.record_feedback(QueryFeedback {
                ef_search_used: 400,
                actual_latency: Duration::from_millis(1),
                results_returned: 10,
                had_filter: false,
                collection_size: 100_000,
            });
        }
        // Should not exceed 500
        assert!(tuner.recommended_ef() <= 500);
    }
}
