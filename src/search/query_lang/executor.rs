use crate::database::Database;
use crate::metadata::{Filter, FilterOperator};
use crate::SearchResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use super::ast::*;
use super::optimizer::{CostBasedOptimizer, CollectionStatistics, SearchStrategy};
use super::parser::QueryValidator;
use super::{QueryError, QueryResult};

/// Query execution context with parameters
#[derive(Debug, Clone, Default)]
pub struct QueryContext {
    /// Query parameters
    pub params: HashMap<String, Value>,
    /// Query vector (for SIMILAR TO)
    pub query_vector: Option<Vec<f32>>,
}

impl QueryContext {
    /// Create a new context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a parameter
    pub fn with_param(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.params.insert(name.into(), value.into());
        self
    }

    /// Set the query vector
    pub fn with_query_vector(mut self, vector: Vec<f32>) -> Self {
        self.query_vector = Some(vector);
        self
    }
}

/// Query execution result
#[derive(Debug, Clone)]
pub struct QueryResponse {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Query plan (for EXPLAIN ANALYZE)
    pub plan: Option<QueryPlan>,
    /// Execution statistics
    pub stats: ExecutionStats,
}

/// Query execution plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryPlan {
    /// Plan nodes
    pub nodes: Vec<PlanNode>,
}

/// Plan node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanNode {
    /// Node type
    pub node_type: String,
    /// Description
    pub description: String,
    /// Estimated cost
    pub estimated_cost: f64,
    /// Actual time (ms, if ANALYZE)
    pub actual_time_ms: Option<f64>,
    /// Rows processed
    pub rows: Option<usize>,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Total execution time in ms
    pub total_time_ms: f64,
    /// Vector search time in ms
    pub search_time_ms: f64,
    /// Filter time in ms
    pub filter_time_ms: f64,
    /// Vectors scanned
    pub vectors_scanned: usize,
    /// Vectors matched
    pub vectors_matched: usize,
}

/// Query executor
pub struct QueryExecutor {
    db: Arc<Database>,
}

impl QueryExecutor {
    /// Create a new executor
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// Execute a query
    pub fn execute(&self, query: &Query, context: &QueryContext) -> QueryResult<QueryResponse> {
        let start_time = Instant::now();

        // Validate query
        QueryValidator::validate(query)?;

        // Check collection exists
        if !self.db.has_collection(&query.from.collection) {
            return Err(QueryError::CollectionNotFound {
                name: query.from.collection.clone(),
            });
        }

        // Get query vector
        let query_vector =
            context
                .query_vector
                .clone()
                .ok_or_else(|| QueryError::MissingParameter {
                    name: "query_vector".to_string(),
                })?;

        // Build filter from WHERE clause
        let filter = if let Some(where_clause) = &query.where_clause {
            Some(Self::build_filter(&where_clause.expression, context)?)
        } else {
            None
        };

        // Determine limit
        let limit = query.limit.unwrap_or(10) as usize;

        // Run cost-based optimizer to select execution strategy
        let collection =
            self.db
                .collection(&query.from.collection)
                .map_err(|e| QueryError::ExecutionError {
                    message: e.to_string(),
                })?;

        let col_stats = {
            let s = collection.stats().map_err(|e| QueryError::ExecutionError {
                message: e.to_string(),
            })?;
            CollectionStatistics {
                vector_count: s.vector_count,
                dimensions: s.dimensions,
                avg_metadata_fields: 3.0,
                index_type: "HNSW".to_string(),
                hnsw_m: 16,
                hnsw_ef_search: 50,
                selectivity: HashMap::new(),
            }
        };
        let optimized = CostBasedOptimizer::optimize(query, &col_stats);

        // Execute search using optimizer-selected strategy
        let search_start = Instant::now();
        let results = match optimized.strategy {
            SearchStrategy::FilterThenIndex => {
                // Pre-filter then search the filtered subset
                if let Some(f) = &filter {
                    collection.search_with_filter(&query_vector, limit, f)
                } else {
                    collection.search(&query_vector, limit)
                }
            }
            SearchStrategy::BruteForceScan => {
                // For small collections, brute-force via search builder
                if let Some(f) = &filter {
                    collection.search_with_filter(&query_vector, limit, f)
                } else {
                    collection.search(&query_vector, limit)
                }
            }
            SearchStrategy::IndexThenFilter | SearchStrategy::HybridSearch => {
                if let Some(f) = &filter {
                    collection.search_with_filter(&query_vector, limit, f)
                } else {
                    collection.search(&query_vector, limit)
                }
            }
        }
        .map_err(|e| QueryError::ExecutionError {
            message: e.to_string(),
        })?;

        let search_time = search_start.elapsed();

        // Apply offset
        let results = if let Some(offset) = query.offset {
            results.into_iter().skip(offset as usize).collect()
        } else {
            results
        };

        let total_time = start_time.elapsed();

        // Build plan if EXPLAIN – include optimizer output
        let plan = if query.explain {
            Some(Self::build_plan(query, &filter, total_time.as_secs_f64() * 1000.0))
        } else {
            None
        };

        Ok(QueryResponse {
            results,
            plan,
            stats: ExecutionStats {
                total_time_ms: total_time.as_secs_f64() * 1000.0,
                search_time_ms: search_time.as_secs_f64() * 1000.0,
                filter_time_ms: 0.0,
                vectors_scanned: col_stats.vector_count.min(limit * 10),
                vectors_matched: limit,
            },
        })
    }

    /// Build a filter from an expression
    fn build_filter(expr: &Expression, context: &QueryContext) -> QueryResult<Filter> {
        match expr {
            Expression::SimilarTo(_) => {
                // SIMILAR TO is handled separately, return a pass-through filter
                Ok(Filter::And(vec![]))
            }
            Expression::Comparison(comp) => {
                let value = Self::resolve_value(&comp.value, context)?;
                let filter = match comp.operator {
                    CompareOp::Eq => Filter::eq(&comp.column, value),
                    CompareOp::Ne => Filter::ne(&comp.column, value),
                    CompareOp::Lt => Filter::lt(&comp.column, value),
                    CompareOp::Le => Filter::lte(&comp.column, value),
                    CompareOp::Gt => Filter::gt(&comp.column, value),
                    CompareOp::Ge => Filter::gte(&comp.column, value),
                };
                Ok(filter)
            }
            Expression::InList(in_expr) => {
                let values: Vec<Value> = in_expr
                    .values
                    .iter()
                    .map(|v| Self::resolve_value(v, context))
                    .collect::<QueryResult<_>>()?;

                let filter = Filter::is_in(&in_expr.column, values);
                if in_expr.negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            Expression::Between(between) => {
                let low = Self::resolve_value(&between.low, context)?;
                let high = Self::resolve_value(&between.high, context)?;

                let filter = Filter::and(vec![
                    Filter::gte(&between.column, low),
                    Filter::lte(&between.column, high),
                ]);

                if between.negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            Expression::Like(like) => {
                // Convert LIKE pattern to contains check
                // This is a simplification - full LIKE would need regex
                let pattern = like.pattern.trim_matches('%');
                let filter = Filter::Condition(crate::metadata::FilterCondition {
                    field: like.column.clone(),
                    operator: FilterOperator::Contains,
                    value: Value::String(pattern.to_string()),
                });

                if like.negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            Expression::IsNull(is_null) => {
                let filter = Filter::eq(&is_null.column, Value::Null);
                if is_null.negated {
                    Ok(Filter::Not(Box::new(filter)))
                } else {
                    Ok(filter)
                }
            }
            Expression::And(left, right) => {
                let l = Self::build_filter(left, context)?;
                let r = Self::build_filter(right, context)?;
                Ok(Filter::and(vec![l, r]))
            }
            Expression::Or(left, right) => {
                let l = Self::build_filter(left, context)?;
                let r = Self::build_filter(right, context)?;
                Ok(Filter::or(vec![l, r]))
            }
            Expression::Not(inner) => {
                let f = Self::build_filter(inner, context)?;
                Ok(Filter::Not(Box::new(f)))
            }
            Expression::Grouped(inner) => Self::build_filter(inner, context),
        }
    }

    /// Resolve a literal value, handling parameters
    fn resolve_value(value: &LiteralValue, context: &QueryContext) -> QueryResult<Value> {
        match value {
            LiteralValue::String(s) => Ok(Value::String(s.clone())),
            LiteralValue::Number(n) => Ok(serde_json::json!(*n)),
            LiteralValue::Bool(b) => Ok(Value::Bool(*b)),
            LiteralValue::Null => Ok(Value::Null),
            LiteralValue::Parameter(name) => context
                .params
                .get(name)
                .cloned()
                .ok_or_else(|| QueryError::MissingParameter { name: name.clone() }),
        }
    }

    /// Build query plan
    fn build_plan(query: &Query, filter: &Option<Filter>, actual_time: f64) -> QueryPlan {
        let mut nodes = Vec::new();

        // Collection scan node
        nodes.push(PlanNode {
            node_type: "VectorScan".to_string(),
            description: format!("Scan collection '{}'", query.from.collection),
            estimated_cost: 1.0,
            actual_time_ms: Some(actual_time * 0.8),
            rows: query.limit.map(|l| l as usize),
        });

        // Filter node if present
        if filter.is_some() {
            nodes.push(PlanNode {
                node_type: "Filter".to_string(),
                description: "Apply metadata filter".to_string(),
                estimated_cost: 0.1,
                actual_time_ms: Some(actual_time * 0.1),
                rows: None,
            });
        }

        // Similarity search node
        nodes.push(PlanNode {
            node_type: "HnswSearch".to_string(),
            description: "HNSW approximate nearest neighbor search".to_string(),
            estimated_cost: 0.5,
            actual_time_ms: Some(actual_time * 0.1),
            rows: query.limit.map(|l| l as usize),
        });

        QueryPlan { nodes }
    }

    /// Build a filter from a WHERE clause (public API for server integration)
    pub fn build_filter_from_where(
        where_clause: &WhereClause,
        context: &QueryContext,
    ) -> QueryResult<Filter> {
        Self::build_filter(&where_clause.expression, context)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::Database;

    // ---- QueryContext tests ----

    #[test]
    fn test_query_context_default() {
        let ctx = QueryContext::default();
        assert!(ctx.params.is_empty());
        assert!(ctx.query_vector.is_none());
    }

    #[test]
    fn test_query_context_with_param() {
        let ctx = QueryContext::new()
            .with_param("category", "books")
            .with_param("limit", 10);

        assert_eq!(ctx.params.get("category"), Some(&Value::String("books".to_string())));
        assert_eq!(ctx.params.get("limit"), Some(&serde_json::json!(10)));
    }

    #[test]
    fn test_query_context_with_query_vector() {
        let ctx = QueryContext::new().with_query_vector(vec![1.0, 2.0, 3.0]);
        assert_eq!(ctx.query_vector, Some(vec![1.0, 2.0, 3.0]));
    }

    // ---- resolve_value tests ----

    #[test]
    fn test_resolve_string_literal() {
        let ctx = QueryContext::new();
        let result = QueryExecutor::resolve_value(&LiteralValue::String("hello".into()), &ctx);
        assert_eq!(result.unwrap(), Value::String("hello".to_string()));
    }

    #[test]
    fn test_resolve_number_literal() {
        let ctx = QueryContext::new();
        let result = QueryExecutor::resolve_value(&LiteralValue::Number(42.0), &ctx);
        assert_eq!(result.unwrap(), serde_json::json!(42.0));
    }

    #[test]
    fn test_resolve_bool_literal() {
        let ctx = QueryContext::new();
        let result = QueryExecutor::resolve_value(&LiteralValue::Bool(true), &ctx);
        assert_eq!(result.unwrap(), Value::Bool(true));
    }

    #[test]
    fn test_resolve_null_literal() {
        let ctx = QueryContext::new();
        let result = QueryExecutor::resolve_value(&LiteralValue::Null, &ctx);
        assert_eq!(result.unwrap(), Value::Null);
    }

    #[test]
    fn test_resolve_parameter_found() {
        let ctx = QueryContext::new().with_param("name", "Alice");
        let result = QueryExecutor::resolve_value(&LiteralValue::Parameter("name".into()), &ctx);
        assert_eq!(result.unwrap(), Value::String("Alice".to_string()));
    }

    #[test]
    fn test_resolve_parameter_missing() {
        let ctx = QueryContext::new();
        let result = QueryExecutor::resolve_value(&LiteralValue::Parameter("missing".into()), &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::MissingParameter { name } => assert_eq!(name, "missing"),
            other => panic!("Expected MissingParameter, got {:?}", other),
        }
    }

    // ---- build_filter tests ----

    #[test]
    fn test_build_filter_eq_comparison() {
        let ctx = QueryContext::new();
        let expr = Expression::Comparison(ComparisonExpr {
            column: "category".to_string(),
            operator: CompareOp::Eq,
            value: LiteralValue::String("books".to_string()),
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        match filter {
            Filter::Condition(cond) => {
                assert_eq!(cond.field, "category");
                assert_eq!(cond.value, Value::String("books".to_string()));
            }
            other => panic!("Expected Condition, got {:?}", other),
        }
    }

    #[test]
    fn test_build_filter_lt_comparison() {
        let ctx = QueryContext::new();
        let expr = Expression::Comparison(ComparisonExpr {
            column: "price".to_string(),
            operator: CompareOp::Lt,
            value: LiteralValue::Number(50.0),
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        match filter {
            Filter::Condition(cond) => {
                assert_eq!(cond.field, "price");
                assert_eq!(cond.operator, FilterOperator::Lt);
            }
            other => panic!("Expected Condition, got {:?}", other),
        }
    }

    #[test]
    fn test_build_filter_and_expression() {
        let ctx = QueryContext::new();
        let left = Expression::Comparison(ComparisonExpr {
            column: "a".to_string(),
            operator: CompareOp::Eq,
            value: LiteralValue::Number(1.0),
        });
        let right = Expression::Comparison(ComparisonExpr {
            column: "b".to_string(),
            operator: CompareOp::Gt,
            value: LiteralValue::Number(2.0),
        });
        let expr = Expression::And(Box::new(left), Box::new(right));

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::And(_)));
    }

    #[test]
    fn test_build_filter_or_expression() {
        let ctx = QueryContext::new();
        let left = Expression::Comparison(ComparisonExpr {
            column: "x".to_string(),
            operator: CompareOp::Eq,
            value: LiteralValue::String("a".to_string()),
        });
        let right = Expression::Comparison(ComparisonExpr {
            column: "x".to_string(),
            operator: CompareOp::Eq,
            value: LiteralValue::String("b".to_string()),
        });
        let expr = Expression::Or(Box::new(left), Box::new(right));

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::Or(_)));
    }

    #[test]
    fn test_build_filter_not_expression() {
        let ctx = QueryContext::new();
        let inner = Expression::Comparison(ComparisonExpr {
            column: "active".to_string(),
            operator: CompareOp::Eq,
            value: LiteralValue::Bool(false),
        });
        let expr = Expression::Not(Box::new(inner));

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::Not(_)));
    }

    #[test]
    fn test_build_filter_in_list() {
        let ctx = QueryContext::new();
        let expr = Expression::InList(InListExpr {
            column: "color".to_string(),
            values: vec![
                LiteralValue::String("red".to_string()),
                LiteralValue::String("blue".to_string()),
            ],
            negated: false,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        match filter {
            Filter::Condition(cond) => {
                assert_eq!(cond.field, "color");
                assert_eq!(cond.operator, FilterOperator::In);
            }
            other => panic!("Expected Condition with In operator, got {:?}", other),
        }
    }

    #[test]
    fn test_build_filter_negated_in_list() {
        let ctx = QueryContext::new();
        let expr = Expression::InList(InListExpr {
            column: "color".to_string(),
            values: vec![LiteralValue::String("red".to_string())],
            negated: true,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::Not(_)));
    }

    #[test]
    fn test_build_filter_between() {
        let ctx = QueryContext::new();
        let expr = Expression::Between(BetweenExpr {
            column: "price".to_string(),
            low: LiteralValue::Number(10.0),
            high: LiteralValue::Number(100.0),
            negated: false,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::And(_)));
    }

    #[test]
    fn test_build_filter_negated_between() {
        let ctx = QueryContext::new();
        let expr = Expression::Between(BetweenExpr {
            column: "price".to_string(),
            low: LiteralValue::Number(10.0),
            high: LiteralValue::Number(100.0),
            negated: true,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::Not(_)));
    }

    #[test]
    fn test_build_filter_like() {
        let ctx = QueryContext::new();
        let expr = Expression::Like(LikeExpr {
            column: "name".to_string(),
            pattern: "%needle%".to_string(),
            negated: false,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        match filter {
            Filter::Condition(cond) => {
                assert_eq!(cond.field, "name");
                assert_eq!(cond.operator, FilterOperator::Contains);
                assert_eq!(cond.value, Value::String("needle".to_string()));
            }
            other => panic!("Expected Condition with Contains, got {:?}", other),
        }
    }

    #[test]
    fn test_build_filter_is_null() {
        let ctx = QueryContext::new();
        let expr = Expression::IsNull(IsNullExpr {
            column: "deleted_at".to_string(),
            negated: false,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        match filter {
            Filter::Condition(cond) => {
                assert_eq!(cond.field, "deleted_at");
                assert_eq!(cond.value, Value::Null);
            }
            other => panic!("Expected Condition, got {:?}", other),
        }
    }

    #[test]
    fn test_build_filter_is_not_null() {
        let ctx = QueryContext::new();
        let expr = Expression::IsNull(IsNullExpr {
            column: "deleted_at".to_string(),
            negated: true,
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::Not(_)));
    }

    #[test]
    fn test_build_filter_similar_to_passthrough() {
        let ctx = QueryContext::new();
        let expr = Expression::SimilarTo(SimilarToExpr {
            column: "vector".to_string(),
            query_param: "$query".to_string(),
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        // SIMILAR TO returns a pass-through Filter::And(vec![])
        assert!(matches!(filter, Filter::And(ref v) if v.is_empty()));
    }

    #[test]
    fn test_build_filter_grouped() {
        let ctx = QueryContext::new();
        let inner = Expression::Comparison(ComparisonExpr {
            column: "x".to_string(),
            operator: CompareOp::Eq,
            value: LiteralValue::Number(1.0),
        });
        let expr = Expression::Grouped(Box::new(inner));

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        assert!(matches!(filter, Filter::Condition(_)));
    }

    #[test]
    fn test_build_filter_with_parameter() {
        let ctx = QueryContext::new().with_param("min_price", 25);
        let expr = Expression::Comparison(ComparisonExpr {
            column: "price".to_string(),
            operator: CompareOp::Ge,
            value: LiteralValue::Parameter("min_price".to_string()),
        });

        let filter = QueryExecutor::build_filter(&expr, &ctx).unwrap();
        match filter {
            Filter::Condition(cond) => {
                assert_eq!(cond.field, "price");
                assert_eq!(cond.operator, FilterOperator::Gte);
                assert_eq!(cond.value, serde_json::json!(25));
            }
            other => panic!("Expected Condition, got {:?}", other),
        }
    }

    // ---- build_filter_from_where tests ----

    #[test]
    fn test_build_filter_from_where_clause() {
        let ctx = QueryContext::new();
        let where_clause = WhereClause {
            expression: Expression::Comparison(ComparisonExpr {
                column: "status".to_string(),
                operator: CompareOp::Eq,
                value: LiteralValue::String("active".to_string()),
            }),
        };

        let filter = QueryExecutor::build_filter_from_where(&where_clause, &ctx).unwrap();
        assert!(matches!(filter, Filter::Condition(_)));
    }

    // ---- execute tests ----

    #[test]
    fn test_execute_missing_collection() {
        let db = Database::in_memory();
        let executor = QueryExecutor::new(Arc::new(db));

        let query = Query {
            explain: false,
            select: SelectClause::All,
            from: FromClause {
                collection: "nonexistent".to_string(),
                alias: None,
            },
            with_clause: None,
            using_clause: None,
            where_clause: None,
            rerank_clause: None,
            order_by: None,
            limit: Some(10),
            offset: None,
        };
        let ctx = QueryContext::new().with_query_vector(vec![1.0, 2.0, 3.0]);

        let result = executor.execute(&query, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::CollectionNotFound { name } => assert_eq!(name, "nonexistent"),
            other => panic!("Expected CollectionNotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_execute_missing_query_vector() {
        let db = Database::in_memory();
        db.create_collection("test", 3).unwrap();
        let executor = QueryExecutor::new(Arc::new(db));

        let query = Query {
            explain: false,
            select: SelectClause::All,
            from: FromClause {
                collection: "test".to_string(),
                alias: None,
            },
            with_clause: None,
            using_clause: None,
            where_clause: None,
            rerank_clause: None,
            order_by: None,
            limit: Some(10),
            offset: None,
        };
        let ctx = QueryContext::new(); // No query vector

        let result = executor.execute(&query, &ctx);
        assert!(result.is_err());
        match result.unwrap_err() {
            QueryError::MissingParameter { name } => assert_eq!(name, "query_vector"),
            other => panic!("Expected MissingParameter, got {:?}", other),
        }
    }

    #[test]
    fn test_execute_basic_search() {
        let db = Database::in_memory();
        db.create_collection("docs", 3).unwrap();
        let coll = db.collection("docs").unwrap();
        coll.insert("v1", &[1.0, 0.0, 0.0], None).unwrap();
        coll.insert("v2", &[0.0, 1.0, 0.0], None).unwrap();

        let executor = QueryExecutor::new(Arc::new(db));

        let query = Query {
            explain: false,
            select: SelectClause::All,
            from: FromClause {
                collection: "docs".to_string(),
                alias: None,
            },
            with_clause: None,
            using_clause: None,
            where_clause: None,
            rerank_clause: None,
            order_by: None,
            limit: Some(5),
            offset: None,
        };
        let ctx = QueryContext::new().with_query_vector(vec![1.0, 0.0, 0.0]);

        let response = executor.execute(&query, &ctx).unwrap();
        assert!(!response.results.is_empty());
        assert!(response.plan.is_none());
        assert!(response.stats.total_time_ms >= 0.0);
    }

    #[test]
    fn test_execute_with_explain() {
        let db = Database::in_memory();
        db.create_collection("docs", 3).unwrap();

        let executor = QueryExecutor::new(Arc::new(db));

        let query = Query {
            explain: true,
            select: SelectClause::All,
            from: FromClause {
                collection: "docs".to_string(),
                alias: None,
            },
            with_clause: None,
            using_clause: None,
            where_clause: None,
            rerank_clause: None,
            order_by: None,
            limit: Some(10),
            offset: None,
        };
        let ctx = QueryContext::new().with_query_vector(vec![1.0, 2.0, 3.0]);

        let response = executor.execute(&query, &ctx).unwrap();
        assert!(response.plan.is_some());
        let plan = response.plan.unwrap();
        assert!(!plan.nodes.is_empty());
        assert!(plan.nodes.iter().any(|n| n.node_type == "VectorScan"));
        assert!(plan.nodes.iter().any(|n| n.node_type == "HnswSearch"));
    }

    #[test]
    fn test_execute_with_offset() {
        let db = Database::in_memory();
        db.create_collection("docs", 3).unwrap();
        let coll = db.collection("docs").unwrap();
        for i in 0..5 {
            coll.insert(&format!("v{}", i), &[i as f32, 0.0, 0.0], None).unwrap();
        }

        let executor = QueryExecutor::new(Arc::new(db));

        let query = Query {
            explain: false,
            select: SelectClause::All,
            from: FromClause {
                collection: "docs".to_string(),
                alias: None,
            },
            with_clause: None,
            using_clause: None,
            where_clause: None,
            rerank_clause: None,
            order_by: None,
            limit: Some(10),
            offset: Some(3),
        };
        let ctx = QueryContext::new().with_query_vector(vec![1.0, 0.0, 0.0]);

        let response = executor.execute(&query, &ctx).unwrap();
        // With 5 vectors, limit 10, offset 3 → at most 2 results
        assert!(response.results.len() <= 2);
    }

    // ---- QueryPlan / PlanNode / ExecutionStats serialization ----

    #[test]
    fn test_query_plan_serialization() {
        let plan = QueryPlan {
            nodes: vec![PlanNode {
                node_type: "VectorScan".to_string(),
                description: "Scan test".to_string(),
                estimated_cost: 1.5,
                actual_time_ms: Some(0.5),
                rows: Some(100),
            }],
        };

        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: QueryPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.nodes.len(), 1);
        assert_eq!(deserialized.nodes[0].node_type, "VectorScan");
    }

    #[test]
    fn test_execution_stats_serialization() {
        let stats = ExecutionStats {
            total_time_ms: 10.5,
            search_time_ms: 8.0,
            filter_time_ms: 2.0,
            vectors_scanned: 1000,
            vectors_matched: 10,
        };

        let json = serde_json::to_string(&stats).unwrap();
        let deserialized: ExecutionStats = serde_json::from_str(&json).unwrap();
        assert!((deserialized.total_time_ms - 10.5).abs() < f64::EPSILON);
        assert_eq!(deserialized.vectors_scanned, 1000);
    }
}
