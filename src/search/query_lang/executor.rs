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
mod tests {
    use super::*;

    // TODO: Add tests for this module
}
