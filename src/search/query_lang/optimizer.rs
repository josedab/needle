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

    fn minimal_query() -> Query {
        Query {
            explain: false,
            select: SelectClause::All,
            from: FromClause {
                collection: "test".into(),
                alias: None,
            },
            with_clause: None,
            using_clause: None,
            where_clause: None,
            rerank_clause: None,
            order_by: None,
            limit: Some(10),
            offset: None,
        }
    }

    fn query_with_where(expr: Expression) -> Query {
        let mut q = minimal_query();
        q.where_clause = Some(WhereClause { expression: expr });
        q
    }

    fn eq_expr(col: &str) -> Expression {
        Expression::Comparison(ComparisonExpr {
            column: col.into(),
            operator: CompareOp::Eq,
            value: LiteralValue::String("v".into()),
        })
    }

    // ====================================================================
    // Strategy selection: BruteForceScan for small collections
    // ====================================================================

    #[test]
    fn test_small_collection_brute_force() {
        let q = minimal_query();
        let stats = CollectionStatistics {
            vector_count: 500,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert_eq!(plan.strategy, SearchStrategy::BruteForceScan);
    }

    #[test]
    fn test_boundary_999_brute_force() {
        let q = minimal_query();
        let stats = CollectionStatistics {
            vector_count: 999,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert_eq!(plan.strategy, SearchStrategy::BruteForceScan);
    }

    #[test]
    fn test_boundary_1000_not_brute_force() {
        let q = minimal_query();
        let stats = CollectionStatistics {
            vector_count: 1000,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert_ne!(plan.strategy, SearchStrategy::BruteForceScan);
    }

    // ====================================================================
    // Strategy: FilterThenIndex for highly selective filters
    // ====================================================================

    #[test]
    fn test_highly_selective_filter_then_index() {
        let q = query_with_where(eq_expr("category"));
        let mut stats = CollectionStatistics {
            vector_count: 100_000,
            ..Default::default()
        };
        // category has 1% selectivity → highly selective
        stats.selectivity.insert("category".into(), 0.01);

        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert_eq!(plan.strategy, SearchStrategy::FilterThenIndex);
    }

    // ====================================================================
    // Strategy: IndexThenFilter (default for large collections)
    // ====================================================================

    #[test]
    fn test_large_collection_index_then_filter() {
        let q = minimal_query();
        let stats = CollectionStatistics {
            vector_count: 50_000,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert_eq!(plan.strategy, SearchStrategy::IndexThenFilter);
    }

    #[test]
    fn test_no_where_clause_index_then_filter() {
        let q = minimal_query();
        let stats = CollectionStatistics {
            vector_count: 10_000,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&q, &stats);
        // selectivity=1.0 when no where → index then filter
        assert_eq!(plan.strategy, SearchStrategy::IndexThenFilter);
    }

    // ====================================================================
    // Strategy: HybridSearch with WITH clause
    // ====================================================================

    #[test]
    fn test_with_clause_triggers_hybrid() {
        let mut q = minimal_query();
        q.with_clause = Some(WithClause::TimeDecay(TimeDecayConfig {
            function: TimeDecayFunction::Exponential,
            params: HashMap::new(),
        }));

        let stats = CollectionStatistics {
            vector_count: 10_000,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert_eq!(plan.strategy, SearchStrategy::HybridSearch);
        assert!(plan.notes.iter().any(|n| n.contains("WITH clause")));
    }

    // ====================================================================
    // Selectivity estimation per expression type
    // ====================================================================

    #[test]
    fn test_selectivity_eq() {
        let expr = Expression::Comparison(ComparisonExpr {
            column: "x".into(),
            operator: CompareOp::Eq,
            value: LiteralValue::Number(1.0),
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_ne() {
        let expr = Expression::Comparison(ComparisonExpr {
            column: "x".into(),
            operator: CompareOp::Ne,
            value: LiteralValue::Number(1.0),
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_range_ops() {
        for op in [CompareOp::Lt, CompareOp::Le, CompareOp::Gt, CompareOp::Ge] {
            let expr = Expression::Comparison(ComparisonExpr {
                column: "x".into(),
                operator: op,
                value: LiteralValue::Number(1.0),
            });
            let stats = CollectionStatistics::default();
            let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
            assert!((sel - 0.3).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_selectivity_known_column() {
        let expr = eq_expr("status");
        let mut stats = CollectionStatistics::default();
        stats.selectivity.insert("status".into(), 0.05);

        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_in_list() {
        let expr = Expression::InList(InListExpr {
            column: "x".into(),
            values: vec![
                LiteralValue::String("a".into()),
                LiteralValue::String("b".into()),
                LiteralValue::String("c".into()),
            ],
            negated: false,
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.3).abs() < f64::EPSILON); // 3 * 0.1
    }

    #[test]
    fn test_selectivity_in_list_capped() {
        let values: Vec<LiteralValue> = (0..20).map(|i| LiteralValue::Number(i as f64)).collect();
        let expr = Expression::InList(InListExpr {
            column: "x".into(),
            values,
            negated: false,
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!(sel <= 0.9 + f64::EPSILON);
    }

    #[test]
    fn test_selectivity_between() {
        let expr = Expression::Between(BetweenExpr {
            column: "x".into(),
            low: LiteralValue::Number(0.0),
            high: LiteralValue::Number(100.0),
            negated: false,
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_like() {
        let expr = Expression::Like(LikeExpr {
            column: "x".into(),
            pattern: "%test%".into(),
            negated: false,
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.15).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_is_null() {
        let expr = Expression::IsNull(IsNullExpr {
            column: "x".into(),
            negated: false,
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.05).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_similar_to() {
        let expr = Expression::SimilarTo(SimilarToExpr {
            column: "vec".into(),
            query_param: "$q".into(),
        });
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 1.0).abs() < f64::EPSILON);
    }

    // ====================================================================
    // Compound expressions
    // ====================================================================

    #[test]
    fn test_selectivity_and() {
        let expr = Expression::And(Box::new(eq_expr("a")), Box::new(eq_expr("b")));
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.01).abs() < f64::EPSILON); // 0.1 * 0.1
    }

    #[test]
    fn test_selectivity_or() {
        let expr = Expression::Or(Box::new(eq_expr("a")), Box::new(eq_expr("b")));
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        // P(A∪B) = P(A) + P(B) - P(A)*P(B) = 0.1 + 0.1 - 0.01 = 0.19
        assert!((sel - 0.19).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_not() {
        let expr = Expression::Not(Box::new(eq_expr("a")));
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.9).abs() < f64::EPSILON); // 1.0 - 0.1
    }

    #[test]
    fn test_selectivity_grouped() {
        let inner = eq_expr("a");
        let expr = Expression::Grouped(Box::new(inner));
        let stats = CollectionStatistics::default();
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.1).abs() < f64::EPSILON);
    }

    // ====================================================================
    // Edge cases: selectivity 0 and 1
    // ====================================================================

    #[test]
    fn test_selectivity_zero() {
        let mut stats = CollectionStatistics::default();
        stats.selectivity.insert("zero".into(), 0.0);

        let expr = eq_expr("zero");
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_selectivity_one() {
        let mut stats = CollectionStatistics::default();
        stats.selectivity.insert("all".into(), 1.0);

        let expr = eq_expr("all");
        let sel = CostBasedOptimizer::estimate_expr_selectivity(&expr, &stats);
        assert!((sel - 1.0).abs() < f64::EPSILON);
    }

    // ====================================================================
    // Cost varies with collection size
    // ====================================================================

    #[test]
    fn test_cost_increases_with_size() {
        let q = minimal_query();
        let small = CollectionStatistics {
            vector_count: 5_000,
            ..Default::default()
        };
        let large = CollectionStatistics {
            vector_count: 1_000_000,
            ..Default::default()
        };

        let plan_small = CostBasedOptimizer::optimize(&q, &small);
        let plan_large = CostBasedOptimizer::optimize(&q, &large);

        assert!(plan_large.total_cost.total_cost > plan_small.total_cost.total_cost);
    }

    // ====================================================================
    // Plan structure
    // ====================================================================

    #[test]
    fn test_plan_has_steps() {
        let q = minimal_query();
        let stats = CollectionStatistics {
            vector_count: 50_000,
            ..Default::default()
        };

        let plan = CostBasedOptimizer::optimize(&q, &stats);
        assert!(!plan.steps.is_empty());
        assert!(!plan.notes.is_empty());
    }

    #[test]
    fn test_filter_then_index_has_two_steps() {
        let q = query_with_where(eq_expr("status"));
        let mut stats = CollectionStatistics {
            vector_count: 100_000,
            ..Default::default()
        };
        stats.selectivity.insert("status".into(), 0.01);

        let plan = CostBasedOptimizer::optimize(&q, &stats);
        if plan.strategy == SearchStrategy::FilterThenIndex {
            assert_eq!(plan.steps.len(), 2);
            assert_eq!(plan.steps[0].step_type, "MetadataFilter");
            assert_eq!(plan.steps[1].step_type, "HnswSearch");
        }
    }

    #[test]
    fn test_index_then_filter_with_where_has_post_filter() {
        let q = query_with_where(eq_expr("category"));
        let stats = CollectionStatistics {
            vector_count: 10_000,
            ..Default::default()
        };

        let plan = CostBasedOptimizer::optimize(&q, &stats);
        if plan.strategy == SearchStrategy::IndexThenFilter {
            assert!(plan.steps.len() >= 2);
            assert!(plan.steps.iter().any(|s| s.step_type == "PostFilter"));
        }
    }

    #[test]
    fn test_collection_statistics_default() {
        let stats = CollectionStatistics::default();
        assert_eq!(stats.vector_count, 1000);
        assert_eq!(stats.dimensions, 384);
        assert_eq!(stats.hnsw_m, 16);
        assert_eq!(stats.hnsw_ef_search, 50);
    }
}
