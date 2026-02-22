use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ast::*;

/// Statistics about a collection used by the optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStatistics {
    pub vector_count: usize,
    pub dimensions: usize,
    pub avg_metadata_fields: f64,
    pub index_type: String,
    pub hnsw_m: usize,
    pub hnsw_ef_search: usize,
    /// Selectivity estimates: field name → estimated fraction of vectors matching (0-1)
    pub selectivity: HashMap<String, f64>,
}

impl Default for CollectionStatistics {
    fn default() -> Self {
        Self {
            vector_count: 1000,
            dimensions: 384,
            avg_metadata_fields: 3.0,
            index_type: "HNSW".to_string(),
            hnsw_m: 16,
            hnsw_ef_search: 50,
            selectivity: HashMap::new(),
        }
    }
}

/// Cost estimate for a query plan node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Estimated CPU cost (arbitrary units)
    pub cpu_cost: f64,
    /// Estimated I/O cost
    pub io_cost: f64,
    /// Estimated rows to process
    pub estimated_rows: f64,
    /// Total estimated cost
    pub total_cost: f64,
}

/// An optimized query plan produced by the cost-based optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedPlan {
    pub steps: Vec<OptimizedStep>,
    pub total_cost: CostEstimate,
    pub strategy: SearchStrategy,
    pub notes: Vec<String>,
}

/// A step in the optimized plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizedStep {
    pub step_type: String,
    pub description: String,
    pub cost: CostEstimate,
}

/// High-level strategy selected by the optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Standard HNSW search, then filter
    IndexThenFilter,
    /// Filter first (if highly selective), then search the subset
    FilterThenIndex,
    /// Brute-force scan (small collection or very selective filter)
    BruteForceScan,
    /// Hybrid BM25 + vector
    HybridSearch,
}

/// Cost-based optimizer that chooses the cheapest execution strategy.
pub struct CostBasedOptimizer;

impl CostBasedOptimizer {
    /// Estimate the cost of a query and choose the best execution strategy.
    #[allow(clippy::too_many_lines)]
    pub fn optimize(query: &Query, stats: &CollectionStatistics) -> OptimizedPlan {
        let n = stats.vector_count as f64;
        let d = stats.dimensions as f64;
        let k = query.limit.unwrap_or(10) as f64;
        let ef = stats.hnsw_ef_search as f64;
        let m = stats.hnsw_m as f64;

        // Estimate filter selectivity
        let selectivity = Self::estimate_selectivity(query, stats);
        let filtered_rows = n * selectivity;

        let mut notes = Vec::new();

        // Cost model: HNSW search ≈ O(ef * log(n) * M * d) distance computations
        let hnsw_search_cost = ef * n.ln().max(1.0) * m * d * 0.001;
        // Filter cost ≈ O(n * fields)
        let filter_cost = n * stats.avg_metadata_fields * 0.0001;
        // Brute force ≈ O(n * d)
        let brute_cost = n * d * 0.001;

        // Strategy 1: Index then filter
        let itf_cost = hnsw_search_cost + filter_cost * 0.1; // filter on k*ef results only

        // Strategy 2: Filter then index (only feasible if selectivity < 0.3)
        let fti_cost = filter_cost + (filtered_rows.max(1.0).ln() * ef * m * d * 0.001);

        // Strategy 3: Brute force (attractive for small collections)
        let bf_cost = brute_cost;

        let (strategy, chosen_cost) = if n < 1000.0 {
            notes.push(format!(
                "Small collection ({} vectors): brute-force preferred",
                n as usize
            ));
            (SearchStrategy::BruteForceScan, bf_cost)
        } else if selectivity < 0.05 && n > 10_000.0 {
            notes.push(format!(
                "Highly selective filter ({:.1}%): filter-then-index preferred",
                selectivity * 100.0
            ));
            (SearchStrategy::FilterThenIndex, fti_cost)
        } else if selectivity < 0.3 && fti_cost < itf_cost {
            notes.push(format!(
                "Selective filter ({:.1}%): filter-then-index cheaper ({:.1} vs {:.1})",
                selectivity * 100.0,
                fti_cost,
                itf_cost
            ));
            (SearchStrategy::FilterThenIndex, fti_cost)
        } else {
            notes.push(format!(
                "Standard index-then-filter (selectivity {:.1}%)",
                selectivity * 100.0
            ));
            (SearchStrategy::IndexThenFilter, itf_cost)
        };

        // Check if hybrid search is applicable
        let strategy = if query.with_clause.is_some() {
            notes.push("WITH clause detected: using hybrid search".to_string());
            SearchStrategy::HybridSearch
        } else {
            strategy
        };

        let mut steps = Vec::new();
        match strategy {
            SearchStrategy::FilterThenIndex => {
                steps.push(OptimizedStep {
                    step_type: "MetadataFilter".into(),
                    description: format!(
                        "Pre-filter {} vectors to ~{}",
                        n as usize, filtered_rows as usize
                    ),
                    cost: CostEstimate {
                        cpu_cost: filter_cost,
                        io_cost: 0.0,
                        estimated_rows: filtered_rows,
                        total_cost: filter_cost,
                    },
                });
                steps.push(OptimizedStep {
                    step_type: "HnswSearch".into(),
                    description: format!("Search filtered subset for top-{}", k as usize),
                    cost: CostEstimate {
                        cpu_cost: fti_cost - filter_cost,
                        io_cost: 0.0,
                        estimated_rows: k,
                        total_cost: fti_cost - filter_cost,
                    },
                });
            }
            SearchStrategy::BruteForceScan => {
                steps.push(OptimizedStep {
                    step_type: "BruteForceScan".into(),
                    description: format!("Linear scan {} vectors", n as usize),
                    cost: CostEstimate {
                        cpu_cost: bf_cost,
                        io_cost: 0.0,
                        estimated_rows: n,
                        total_cost: bf_cost,
                    },
                });
            }
            SearchStrategy::IndexThenFilter | SearchStrategy::HybridSearch => {
                steps.push(OptimizedStep {
                    step_type: "HnswSearch".into(),
                    description: format!("HNSW ANN search (ef={}, M={})", ef as usize, m as usize),
                    cost: CostEstimate {
                        cpu_cost: hnsw_search_cost,
                        io_cost: 0.0,
                        estimated_rows: k * 2.0,
                        total_cost: hnsw_search_cost,
                    },
                });
                if query.where_clause.is_some() {
                    steps.push(OptimizedStep {
                        step_type: "PostFilter".into(),
                        description: "Apply metadata filter on candidates".into(),
                        cost: CostEstimate {
                            cpu_cost: filter_cost * 0.1,
                            io_cost: 0.0,
                            estimated_rows: k,
                            total_cost: filter_cost * 0.1,
                        },
                    });
                }
            }
        }

        let total_cost = CostEstimate {
            cpu_cost: chosen_cost,
            io_cost: 0.0,
            estimated_rows: k,
            total_cost: chosen_cost,
        };

        OptimizedPlan {
            steps,
            total_cost,
            strategy,
            notes,
        }
    }

    /// Estimate filter selectivity from WHERE clause.
    fn estimate_selectivity(query: &Query, stats: &CollectionStatistics) -> f64 {
        let Some(where_clause) = query.where_clause.as_ref() else {
            return 1.0;
        };
        Self::estimate_expr_selectivity(&where_clause.expression, stats)
    }

    fn estimate_expr_selectivity(expr: &Expression, stats: &CollectionStatistics) -> f64 {
        match expr {
            Expression::SimilarTo(_) => 1.0,
            Expression::Comparison(comp) => {
                // Use known selectivity if available
                if let Some(&sel) = stats.selectivity.get(&comp.column) {
                    return sel;
                }
                match comp.operator {
                    CompareOp::Eq => 0.1,
                    CompareOp::Ne => 0.9,
                    CompareOp::Lt | CompareOp::Le | CompareOp::Gt | CompareOp::Ge => 0.3,
                }
            }
            Expression::InList(in_expr) => {
                let per_value = 0.1;
                (per_value * in_expr.values.len() as f64).min(0.9)
            }
            Expression::Between(_) => 0.2,
            Expression::Like(_) => 0.15,
            Expression::IsNull(_) => 0.05,
            Expression::And(l, r) => {
                Self::estimate_expr_selectivity(l, stats)
                    * Self::estimate_expr_selectivity(r, stats)
            }
            Expression::Or(l, r) => {
                let sl = Self::estimate_expr_selectivity(l, stats);
                let sr = Self::estimate_expr_selectivity(r, stats);
                (sl + sr - sl * sr).min(1.0)
            }
            Expression::Not(inner) => 1.0 - Self::estimate_expr_selectivity(inner, stats),
            Expression::Grouped(inner) => Self::estimate_expr_selectivity(inner, stats),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Add tests for this module
}
