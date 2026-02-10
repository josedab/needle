//! Unified Query Planner Service
//!
//! Integrates the cost-based query optimizer with Collection-level statistics
//! to provide automatic query plan selection and EXPLAIN output.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::query_planner::{QueryPlanner, QueryPlannerConfig, QueryRequest};
//! use needle::{Database, Filter};
//! use serde_json::json;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let coll = db.collection("docs").unwrap();
//! // ... insert vectors ...
//!
//! let planner = QueryPlanner::for_collection(&coll);
//!
//! let request = QueryRequest {
//!     query: vec![0.1f32; 128],
//!     k: 10,
//!     filter: None,
//!     ef_search: None,
//! };
//!
//! // Get an optimized plan
//! let plan = planner.plan(&request);
//! println!("Strategy: {:?}", plan.strategy);
//! println!("Estimated cost: {:.2}ms", plan.estimated_cost_ms);
//!
//! // Get EXPLAIN output
//! let explain = planner.explain(&request);
//! println!("{}", explain);
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::database::CollectionRef;
use crate::metadata::Filter;

/// Configuration for the query planner.
#[derive(Debug, Clone)]
pub struct QueryPlannerConfig {
    /// Weight for vector search cost in the total cost model.
    pub vector_search_weight: f64,
    /// Weight for filter cost in the total cost model.
    pub filter_weight: f64,
    /// Threshold below which pre-filtering is preferred (selectivity 0.0–1.0).
    pub pre_filter_threshold: f64,
    /// Maximum ef_search to consider for automatic tuning.
    pub max_ef_search: usize,
}

impl Default for QueryPlannerConfig {
    fn default() -> Self {
        Self {
            vector_search_weight: 1.0,
            filter_weight: 0.5,
            pre_filter_threshold: 0.1,
            max_ef_search: 500,
        }
    }
}

/// A query request to be planned.
#[derive(Debug, Clone)]
pub struct QueryRequest {
    pub query: Vec<f32>,
    pub k: usize,
    pub filter: Option<Filter>,
    pub ef_search: Option<usize>,
}

/// Strategy selected by the query planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanStrategy {
    /// Pure vector search with no filtering.
    VectorOnly,
    /// Apply filter first, then search on filtered subset.
    PreFilter,
    /// Search first, then filter results (over-retrieve and filter).
    PostFilter,
    /// Parallel filter + search, merge results.
    HybridFilter,
    /// Full scan (small collections).
    FullScan,
}

impl fmt::Display for PlanStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VectorOnly => write!(f, "VectorOnly"),
            Self::PreFilter => write!(f, "PreFilter"),
            Self::PostFilter => write!(f, "PostFilter"),
            Self::HybridFilter => write!(f, "HybridFilter"),
            Self::FullScan => write!(f, "FullScan"),
        }
    }
}

/// An optimized query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    pub strategy: PlanStrategy,
    pub estimated_cost_ms: f64,
    pub estimated_candidates: usize,
    pub ef_search: usize,
    pub steps: Vec<PlanStep>,
    pub warnings: Vec<String>,
}

/// A step in the execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub name: String,
    pub description: String,
    pub estimated_cost_ms: f64,
}

/// Collection-level statistics used for cost estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub vector_count: usize,
    pub dimension: usize,
    pub index_type: String,
    pub hnsw_m: usize,
    pub ef_search: usize,
    pub has_metadata: bool,
    pub metadata_field_count: usize,
}

impl CollectionStats {
    /// Gather stats from a CollectionRef.
    pub fn from_collection(coll: &CollectionRef) -> Self {
        let count = coll.count(None).unwrap_or(0);
        let dim = coll.dimensions().unwrap_or(0);
        Self {
            vector_count: count,
            dimension: dim,
            index_type: "HNSW".into(),
            hnsw_m: 16,
            ef_search: 50,
            has_metadata: true,
            metadata_field_count: 0,
        }
    }
}

/// The query planner that generates optimized execution plans.
pub struct QueryPlanner {
    stats: CollectionStats,
    config: QueryPlannerConfig,
}

impl QueryPlanner {
    /// Create a query planner from a collection reference.
    pub fn for_collection(coll: &CollectionRef) -> Self {
        Self {
            stats: CollectionStats::from_collection(coll),
            config: QueryPlannerConfig::default(),
        }
    }

    /// Create with explicit stats and config.
    pub fn new(stats: CollectionStats, config: QueryPlannerConfig) -> Self {
        Self { stats, config }
    }

    /// Generate an optimized query plan.
    pub fn plan(&self, request: &QueryRequest) -> QueryPlan {
        let has_filter = request.filter.is_some();
        let ef = request.ef_search.unwrap_or(self.stats.ef_search);
        let n = self.stats.vector_count;

        if n == 0 {
            return QueryPlan {
                strategy: PlanStrategy::FullScan,
                estimated_cost_ms: 0.0,
                estimated_candidates: 0,
                ef_search: ef,
                steps: vec![PlanStep {
                    name: "EmptyCollection".into(),
                    description: "Collection is empty, returning empty results".into(),
                    estimated_cost_ms: 0.0,
                }],
                warnings: vec![],
            };
        }

        // Choose strategy
        let (strategy, selectivity) = if !has_filter {
            (PlanStrategy::VectorOnly, 1.0f64)
        } else {
            let est_selectivity = self.estimate_selectivity(request.filter.as_ref().expect("filter is Some"));

            if n < 1000 {
                (PlanStrategy::FullScan, est_selectivity)
            } else if est_selectivity < self.config.pre_filter_threshold {
                (PlanStrategy::PreFilter, est_selectivity)
            } else if est_selectivity > 0.8 {
                (PlanStrategy::PostFilter, est_selectivity)
            } else {
                (PlanStrategy::HybridFilter, est_selectivity)
            }
        };

        // Estimate costs
        let (steps, total_cost) = self.estimate_steps(&strategy, n, ef, selectivity, request.k);

        // Generate warnings
        let mut warnings = Vec::new();
        if n > 1_000_000 && ef > 200 {
            warnings.push(format!(
                "High ef_search ({ef}) on large collection ({n} vectors) may cause slow queries"
            ));
        }
        if has_filter && selectivity < 0.01 {
            warnings.push("Very low filter selectivity (<1%) — consider a metadata index".into());
        }
        if request.k > n / 2 {
            warnings.push(format!(
                "k={} is more than half the collection size ({})", request.k, n
            ));
        }

        QueryPlan {
            strategy,
            estimated_cost_ms: total_cost,
            estimated_candidates: (n as f64 * selectivity) as usize,
            ef_search: ef,
            steps,
            warnings,
        }
    }

    /// Generate a human-readable EXPLAIN output.
    pub fn explain(&self, request: &QueryRequest) -> String {
        let plan = self.plan(request);
        let mut out = String::new();

        out.push_str("=== QUERY PLAN ===\n");
        out.push_str(&format!("Strategy: {}\n", plan.strategy));
        out.push_str(&format!("Estimated Cost: {:.2}ms\n", plan.estimated_cost_ms));
        out.push_str(&format!("Estimated Candidates: {}\n", plan.estimated_candidates));
        out.push_str(&format!("ef_search: {}\n", plan.ef_search));
        out.push_str(&format!(
            "Collection: {} vectors, {} dimensions\n",
            self.stats.vector_count, self.stats.dimension
        ));
        out.push_str("\nExecution Steps:\n");
        for (i, step) in plan.steps.iter().enumerate() {
            out.push_str(&format!(
                "  {}. {} — {} ({:.2}ms)\n",
                i + 1,
                step.name,
                step.description,
                step.estimated_cost_ms
            ));
        }
        if !plan.warnings.is_empty() {
            out.push_str("\nWarnings:\n");
            for w in &plan.warnings {
                out.push_str(&format!("  ⚠ {}\n", w));
            }
        }
        out
    }

    /// Generate EXPLAIN output as JSON.
    pub fn explain_json(&self, request: &QueryRequest) -> Value {
        let plan = self.plan(request);
        serde_json::to_value(&plan).unwrap_or(Value::Null)
    }

    fn estimate_selectivity(&self, _filter: &Filter) -> f64 {
        // Heuristic: assume ~30% selectivity without real statistics
        0.3
    }

    fn estimate_steps(
        &self,
        strategy: &PlanStrategy,
        n: usize,
        ef: usize,
        selectivity: f64,
        k: usize,
    ) -> (Vec<PlanStep>, f64) {
        let mut steps = Vec::new();
        let dim = self.stats.dimension;

        // Base HNSW search cost model: O(ef * log(n) * dim)
        let log_n = (n as f64).ln().max(1.0);
        let hnsw_cost = ef as f64 * log_n * dim as f64 * 0.00001; // ms

        match strategy {
            PlanStrategy::VectorOnly => {
                steps.push(PlanStep {
                    name: "HNSWSearch".into(),
                    description: format!("Search HNSW index (ef={})", ef),
                    estimated_cost_ms: hnsw_cost,
                });
            }
            PlanStrategy::PreFilter => {
                let filter_cost = n as f64 * 0.0001;
                steps.push(PlanStep {
                    name: "MetadataFilter".into(),
                    description: format!(
                        "Pre-filter metadata (est. {}% pass)",
                        (selectivity * 100.0) as u32
                    ),
                    estimated_cost_ms: filter_cost,
                });
                let reduced_cost = hnsw_cost * selectivity;
                steps.push(PlanStep {
                    name: "HNSWSearch".into(),
                    description: format!("Search filtered subset (ef={})", ef),
                    estimated_cost_ms: reduced_cost,
                });
            }
            PlanStrategy::PostFilter => {
                let over_retrieve = (k as f64 / selectivity).min(n as f64) as usize;
                steps.push(PlanStep {
                    name: "HNSWSearch".into(),
                    description: format!("Over-retrieve {} candidates (ef={})", over_retrieve, ef),
                    estimated_cost_ms: hnsw_cost,
                });
                let filter_cost = over_retrieve as f64 * 0.001;
                steps.push(PlanStep {
                    name: "PostFilter".into(),
                    description: format!("Filter {} candidates to k={}", over_retrieve, k),
                    estimated_cost_ms: filter_cost,
                });
            }
            PlanStrategy::HybridFilter => {
                let filter_cost = n as f64 * 0.0001 * 0.5; // partial scan
                steps.push(PlanStep {
                    name: "ParallelFilter".into(),
                    description: "Concurrent metadata pre-filter".into(),
                    estimated_cost_ms: filter_cost,
                });
                steps.push(PlanStep {
                    name: "HNSWSearch".into(),
                    description: format!("Search with filter hints (ef={})", ef),
                    estimated_cost_ms: hnsw_cost * 0.8,
                });
                steps.push(PlanStep {
                    name: "MergeResults".into(),
                    description: "Merge and re-rank candidates".into(),
                    estimated_cost_ms: k as f64 * 0.01,
                });
            }
            PlanStrategy::FullScan => {
                let scan_cost = n as f64 * dim as f64 * 0.00001;
                steps.push(PlanStep {
                    name: "FullScan".into(),
                    description: format!("Scan all {} vectors", n),
                    estimated_cost_ms: scan_cost,
                });
            }
        }

        let total: f64 = steps.iter().map(|s| s.estimated_cost_ms).sum();
        (steps, total)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stats(n: usize, dim: usize) -> CollectionStats {
        CollectionStats {
            vector_count: n,
            dimension: dim,
            index_type: "HNSW".into(),
            hnsw_m: 16,
            ef_search: 50,
            has_metadata: true,
            metadata_field_count: 3,
        }
    }

    #[test]
    fn test_plan_vector_only() {
        let planner = QueryPlanner::new(make_stats(10000, 128), QueryPlannerConfig::default());
        let request = QueryRequest {
            query: vec![0.1; 128],
            k: 10,
            filter: None,
            ef_search: None,
        };
        let plan = planner.plan(&request);
        assert_eq!(plan.strategy, PlanStrategy::VectorOnly);
        assert!(plan.estimated_cost_ms > 0.0);
        assert_eq!(plan.steps.len(), 1);
    }

    #[test]
    fn test_plan_with_filter() {
        let planner = QueryPlanner::new(make_stats(100000, 384), QueryPlannerConfig::default());
        let filter = Filter::eq("category", "books");
        let request = QueryRequest {
            query: vec![0.1; 384],
            k: 10,
            filter: Some(filter),
            ef_search: None,
        };
        let plan = planner.plan(&request);
        // With default heuristic selectivity of 0.3, should be HybridFilter
        assert_eq!(plan.strategy, PlanStrategy::HybridFilter);
        assert!(plan.steps.len() >= 2);
    }

    #[test]
    fn test_plan_small_collection_uses_fullscan() {
        let planner = QueryPlanner::new(make_stats(500, 128), QueryPlannerConfig::default());
        let filter = Filter::eq("status", "active");
        let request = QueryRequest {
            query: vec![0.1; 128],
            k: 10,
            filter: Some(filter),
            ef_search: None,
        };
        let plan = planner.plan(&request);
        assert_eq!(plan.strategy, PlanStrategy::FullScan);
    }

    #[test]
    fn test_plan_empty_collection() {
        let planner = QueryPlanner::new(make_stats(0, 128), QueryPlannerConfig::default());
        let request = QueryRequest {
            query: vec![0.1; 128],
            k: 10,
            filter: None,
            ef_search: None,
        };
        let plan = planner.plan(&request);
        assert_eq!(plan.strategy, PlanStrategy::FullScan);
        assert_eq!(plan.estimated_cost_ms, 0.0);
    }

    #[test]
    fn test_explain_output() {
        let planner = QueryPlanner::new(make_stats(50000, 256), QueryPlannerConfig::default());
        let request = QueryRequest {
            query: vec![0.1; 256],
            k: 10,
            filter: None,
            ef_search: Some(100),
        };
        let explain = planner.explain(&request);
        assert!(explain.contains("QUERY PLAN"));
        assert!(explain.contains("VectorOnly"));
        assert!(explain.contains("HNSWSearch"));
    }

    #[test]
    fn test_explain_json() {
        let planner = QueryPlanner::new(make_stats(1000, 128), QueryPlannerConfig::default());
        let request = QueryRequest {
            query: vec![0.1; 128],
            k: 5,
            filter: None,
            ef_search: None,
        };
        let json = planner.explain_json(&request);
        assert!(json.is_object());
        assert!(json.get("strategy").is_some());
    }

    #[test]
    fn test_warnings_for_high_k() {
        let planner = QueryPlanner::new(make_stats(100, 128), QueryPlannerConfig::default());
        let request = QueryRequest {
            query: vec![0.1; 128],
            k: 80,
            filter: None,
            ef_search: None,
        };
        let plan = planner.plan(&request);
        assert!(!plan.warnings.is_empty());
    }
}
