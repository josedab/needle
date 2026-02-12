//! NeedleQL Query Executor Service
//!
//! Full query language execution with SELECT, WHERE, NEAREST_TO(), HYBRID_SEARCH(),
//! and EXPLAIN — building on the query_lang parser. Exposes NeedleQL execution as
//! a service for CLI, REST, and SDK consumption.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::needleql_executor::{
//!     NeedleQLExecutor, ExecutorConfig, ExecutionResult, QueryRequest,
//! };
//!
//! let executor = NeedleQLExecutor::new(ExecutorConfig::default());
//!
//! // Parse and plan a query
//! let plan = executor.parse("SELECT id, distance FROM docs NEAREST_TO([0.1, 0.2]) LIMIT 5").unwrap();
//! assert_eq!(plan.collection, "docs");
//! assert_eq!(plan.limit, 5);
//!
//! // Validate query
//! assert!(executor.validate("SELECT * FROM docs WHERE category = 'books'").is_ok());
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Re-exports from query_lang scaffold ──────────────────────────────────────

#[allow(unused_imports)]
pub use crate::search::query_lang::{
    // Parser & execution pipeline
    QueryParser, QueryValidator, QueryExecutor, CostBasedOptimizer,
    // AST types
    Query, SelectClause, FromClause, WithClause, UsingClause,
    WhereClause as QLWhereClause, Expression, ComparisonExpr, CompareOp,
    LiteralValue as QLLiteralValue, SimilarToExpr, InListExpr, BetweenExpr,
    LikeExpr, IsNullExpr, OrderByClause, SortOrder,
    // Time decay & RAG
    TimeDecayConfig, TimeDecayFunction, RagOptions,
    // Execution types
    QueryContext, QueryResponse, QueryPlan as QLQueryPlan, PlanNode,
    ExecutionStats, OptimizedPlan, SearchStrategy,
    // Cost & statistics
    CostEstimate, CollectionStatistics, OptimizedStep,
    // Session & aggregation
    QuerySession, AggregateFunction,
    // Error types
    QueryError, QueryResult,
    // Lexer & tokens
    Token, Lexer, DurationUnit,
};

// ── AST Types (NeedleQL service-layer, used by REST/CLI/SDK) ─────────────────

/// Parsed NeedleQL statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Statement {
    /// SELECT query.
    Select(SelectQuery),
    /// EXPLAIN wrapping a SELECT.
    Explain(SelectQuery),
    /// INSERT statement.
    Insert(InsertStatement),
    /// DELETE statement.
    Delete(DeleteStatement),
    /// CREATE COLLECTION statement.
    CreateCollection(CreateCollectionStatement),
    /// DROP COLLECTION statement.
    DropCollection(String),
    /// SHOW COLLECTIONS statement.
    ShowCollections,
    /// CREATE VIEW statement.
    CreateView(CreateViewStatement),
    /// DROP VIEW statement.
    DropView(String),
}

/// CREATE VIEW statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateViewStatement {
    /// View name.
    pub name: String,
    /// Source collection.
    pub source_collection: String,
    /// The full view definition query.
    pub definition: String,
    /// Result limit.
    pub limit: usize,
    /// Optional metadata filter.
    pub filter: Option<ServiceWhereClause>,
}

/// A SELECT query in NeedleQL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectQuery {
    /// Columns to return (empty = all).
    pub columns: Vec<SelectColumn>,
    /// Collection name.
    pub collection: String,
    /// WHERE filter conditions.
    pub filter: Option<ServiceWhereClause>,
    /// NEAREST_TO vector search.
    pub nearest_to: Option<NearestTo>,
    /// HYBRID_SEARCH text + vector.
    pub hybrid_search: Option<HybridSearch>,
    /// ORDER BY clause.
    pub order_by: Option<OrderBy>,
    /// LIMIT count.
    pub limit: usize,
    /// OFFSET for pagination.
    pub offset: usize,
    /// WITH options.
    pub options: HashMap<String, OptionValue>,
    /// AS OF clause for time-travel queries (Unix timestamp or time expression).
    pub as_of: Option<AsOfClause>,
}

/// AS OF clause for point-in-time time-travel queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsOfClause {
    /// Unix timestamp to query at.
    pub timestamp: Option<u64>,
    /// Time expression string (e.g. "yesterday", "2 hours ago").
    pub time_expression: Option<String>,
    /// Named snapshot to query at.
    pub snapshot: Option<String>,
    /// Version number to query at.
    pub version: Option<u64>,
}

/// Column in SELECT.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectColumn {
    /// All columns.
    Star,
    /// Named column (id, distance, metadata field).
    Named(String),
    /// Aliased expression.
    Aliased { expr: String, alias: String },
}

/// WHERE clause with filter conditions (service-layer representation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceWhereClause {
    /// Simple comparison: field op value.
    Comparison {
        field: String,
        op: ComparisonOp,
        value: LiteralValue,
    },
    /// AND of conditions.
    And(Vec<ServiceWhereClause>),
    /// OR of conditions.
    Or(Vec<ServiceWhereClause>),
    /// NOT of a condition.
    Not(Box<ServiceWhereClause>),
    /// IN list check.
    In {
        field: String,
        values: Vec<LiteralValue>,
    },
}

/// Comparison operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Like,
}

/// Literal value in NeedleQL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiteralValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
    Vector(Vec<f32>),
}

/// NEAREST_TO clause for vector similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NearestTo {
    /// Query vector.
    pub vector: Vec<f32>,
    /// Optional distance function override.
    pub distance_fn: Option<String>,
    /// ef_search parameter override.
    pub ef_search: Option<usize>,
}

/// HYBRID_SEARCH clause combining text and vector search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridSearch {
    /// Text query for BM25 component.
    pub text_query: String,
    /// Vector for similarity component.
    pub vector: Option<Vec<f32>>,
    /// Weight for vector results (0.0 to 1.0).
    pub vector_weight: f64,
    /// Weight for text results.
    pub text_weight: f64,
    /// RRF k parameter.
    pub rrf_k: f64,
}

/// ORDER BY clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBy {
    pub field: String,
    pub direction: SortDirection,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// Option value in WITH clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionValue {
    String(String),
    Number(f64),
    Bool(bool),
}

/// INSERT statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertStatement {
    pub collection: String,
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

/// DELETE statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteStatement {
    pub collection: String,
    pub filter: Option<ServiceWhereClause>,
    pub ids: Vec<String>,
}

/// CREATE COLLECTION statement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionStatement {
    pub name: String,
    pub dimensions: usize,
    pub distance_fn: Option<String>,
    pub if_not_exists: bool,
}

// ── Query Plan ──────────────────────────────────────────────────────────────

/// Service-level execution plan for a NeedleQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceQueryPlan {
    /// The parsed statement.
    pub statement: Statement,
    /// Collection targeted.
    pub collection: String,
    /// Limit for results.
    pub limit: usize,
    /// Plan steps.
    pub steps: Vec<PlanStep>,
    /// Estimated cost.
    pub estimated_cost: f64,
}

/// A step in the query execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    /// Step name.
    pub name: String,
    /// Step type.
    pub step_type: PlanStepType,
    /// Estimated rows produced.
    pub estimated_rows: usize,
    /// Estimated cost.
    pub cost: f64,
}

/// Type of plan step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PlanStepType {
    /// HNSW vector search.
    VectorSearch { ef_search: usize, k: usize },
    /// BM25 text search.
    TextSearch { query: String },
    /// Metadata filter.
    MetadataFilter { field: String },
    /// Hybrid fusion.
    HybridFusion { vector_weight: f64 },
    /// Sort operation.
    Sort { field: String },
    /// Limit operation.
    Limit { count: usize },
    /// Scan (full collection scan).
    Scan,
    /// Time-travel scan at a historical point.
    TimeTravelScan { timestamp: Option<u64>, version: Option<u64> },
    /// Insert operation.
    Insert,
    /// Delete operation.
    Delete,
    /// DDL operation (create/drop).
    Ddl,
}

// ── Execution Result ────────────────────────────────────────────────────────

/// Result of executing a NeedleQL query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Returned rows.
    pub rows: Vec<ResultRow>,
    /// Number of rows affected (for INSERT/DELETE).
    pub rows_affected: usize,
    /// Execution time in microseconds.
    pub execution_time_us: u64,
    /// Query plan (if EXPLAIN).
    pub plan: Option<ServiceQueryPlan>,
    /// Warnings generated during execution.
    pub warnings: Vec<String>,
}

/// A single result row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultRow {
    /// Column values.
    pub columns: HashMap<String, serde_json::Value>,
}

/// Query request (for REST/SDK).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryRequest {
    /// NeedleQL query string.
    pub query: String,
    /// Optional parameter bindings.
    pub params: HashMap<String, serde_json::Value>,
    /// Whether to include EXPLAIN output.
    pub explain: bool,
}

// ── Executor Config ─────────────────────────────────────────────────────────

/// Executor configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutorConfig {
    /// Maximum query length in characters.
    pub max_query_length: usize,
    /// Default result limit.
    pub default_limit: usize,
    /// Maximum result limit.
    pub max_limit: usize,
    /// Enable EXPLAIN output.
    pub enable_explain: bool,
    /// Enable HYBRID_SEARCH.
    pub enable_hybrid: bool,
    /// Default ef_search for vector queries.
    pub default_ef_search: usize,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            max_query_length: 10_000,
            default_limit: 10,
            max_limit: 10_000,
            enable_explain: true,
            enable_hybrid: true,
            default_ef_search: 50,
        }
    }
}

// ── NeedleQL Executor Service ───────────────────────────────────────────────

/// NeedleQL executor service — facade over [`QueryParser`], [`QueryValidator`],
/// [`CostBasedOptimizer`], and [`QueryExecutor`] from `query_lang`.
pub struct NeedleQLExecutor {
    config: ExecutorConfig,
    query_count: u64,
}

impl NeedleQLExecutor {
    /// Create a new executor.
    pub fn new(config: ExecutorConfig) -> Self {
        Self {
            config,
            query_count: 0,
        }
    }

    /// Parse a NeedleQL query string into a service-level query plan.
    ///
    /// For SELECT-like statements the input is first run through
    /// [`QueryParser::parse`] from `query_lang` and then mapped into the
    /// service-layer types. Non-SELECT DDL/DML statements are handled
    /// directly since `query_lang` focuses on SELECT queries.
    pub fn parse(&self, query: &str) -> Result<ServiceQueryPlan> {
        if query.len() > self.config.max_query_length {
            return Err(NeedleError::InvalidArgument(format!(
                "Query exceeds maximum length of {} characters",
                self.config.max_query_length
            )));
        }
        let query = query.trim();
        if query.is_empty() {
            return Err(NeedleError::InvalidArgument("Empty query".into()));
        }

        let upper = query.to_uppercase();

        // Delegate SELECT / EXPLAIN SELECT to QueryParser
        if upper.starts_with("EXPLAIN ") {
            let inner = &query[8..];
            let select = self.parse_select_via_query_parser(inner)?;
            let collection = select.collection.clone();
            let limit = select.limit;
            let steps = self.plan_steps(&Statement::Explain(select.clone()));
            let cost = steps.iter().map(|s| s.cost).sum();
            return Ok(ServiceQueryPlan {
                statement: Statement::Explain(select),
                collection,
                limit,
                steps,
                estimated_cost: cost,
            });
        }

        if upper.starts_with("SELECT ") || upper.starts_with("SELECT\n") {
            let (query_str, as_of) = self.extract_as_of_clause(query);
            let mut select = self.parse_select_via_query_parser(&query_str)?;
            if as_of.is_some() {
                select.as_of = as_of;
            }
            let collection = select.collection.clone();
            let limit = select.limit;
            let steps = self.plan_steps(&Statement::Select(select.clone()));
            let cost = steps.iter().map(|s| s.cost).sum();
            return Ok(ServiceQueryPlan {
                statement: Statement::Select(select),
                collection,
                limit,
                steps,
                estimated_cost: cost,
            });
        }

        if upper.starts_with("INSERT ") {
            let insert = self.parse_insert(query)?;
            let collection = insert.collection.clone();
            let steps = self.plan_steps(&Statement::Insert(insert.clone()));
            let cost = steps.iter().map(|s| s.cost).sum();
            return Ok(ServiceQueryPlan {
                statement: Statement::Insert(insert),
                collection,
                limit: 0,
                steps,
                estimated_cost: cost,
            });
        }

        if upper.starts_with("DELETE ") {
            let delete = self.parse_delete(query)?;
            let collection = delete.collection.clone();
            let steps = self.plan_steps(&Statement::Delete(delete.clone()));
            let cost = steps.iter().map(|s| s.cost).sum();
            return Ok(ServiceQueryPlan {
                statement: Statement::Delete(delete),
                collection,
                limit: 0,
                steps,
                estimated_cost: cost,
            });
        }

        if upper.starts_with("CREATE COLLECTION ") {
            let create = self.parse_create_collection(query)?;
            let name = create.name.clone();
            let steps = self.plan_steps(&Statement::CreateCollection(create.clone()));
            let cost = steps.iter().map(|s| s.cost).sum();
            return Ok(ServiceQueryPlan {
                statement: Statement::CreateCollection(create),
                collection: name,
                limit: 0,
                steps,
                estimated_cost: cost,
            });
        }

        if upper.starts_with("DROP COLLECTION ") {
            let name = query[16..].trim().trim_end_matches(';').trim().to_string();
            if name.is_empty() {
                return Err(NeedleError::InvalidArgument(
                    "Missing collection name in DROP COLLECTION".into(),
                ));
            }
            let steps = self.plan_steps(&Statement::DropCollection(name.clone()));
            let cost = steps.iter().map(|s| s.cost).sum();
            return Ok(ServiceQueryPlan {
                statement: Statement::DropCollection(name.clone()),
                collection: name,
                limit: 0,
                steps,
                estimated_cost: cost,
            });
        }

        if upper.starts_with("SHOW COLLECTIONS") {
            return Ok(ServiceQueryPlan {
                statement: Statement::ShowCollections,
                collection: String::new(),
                limit: 0,
                steps: vec![],
                estimated_cost: 0.1,
            });
        }

        if upper.starts_with("CREATE VIEW ") || upper.starts_with("CREATE MATERIALIZED VIEW ") {
            let is_materialized = upper.starts_with("CREATE MATERIALIZED VIEW ");
            let after = if is_materialized { &query[25..] } else { &query[12..] };

            let as_pos = after.to_uppercase().find(" AS ").ok_or_else(|| {
                NeedleError::InvalidArgument("Missing AS in CREATE VIEW".into())
            })?;
            let view_name = after[..as_pos].trim().to_string();

            let rest = &after[as_pos + 4..];
            let rest_upper = rest.to_uppercase();
            let from_pos = rest_upper.find("FROM ").unwrap_or(0);
            let after_from = &rest[from_pos + 5..];
            let collection_end = after_from
                .find(|c: char| c.is_whitespace())
                .unwrap_or(after_from.len());
            let collection = after_from[..collection_end].trim().to_string();

            let limit = if let Some(pos) = rest_upper.find("LIMIT ") {
                let after_limit = &rest[pos + 6..];
                let end = after_limit
                    .find(|c: char| !c.is_ascii_digit())
                    .unwrap_or(after_limit.len());
                after_limit[..end].parse().unwrap_or(10)
            } else {
                10
            };

            let filter = self.parse_where_clause_fallback(rest, &rest_upper);

            let create_view = CreateViewStatement {
                name: view_name,
                source_collection: collection.clone(),
                definition: query.to_string(),
                limit,
                filter,
            };

            return Ok(ServiceQueryPlan {
                statement: Statement::CreateView(create_view),
                collection,
                limit,
                steps: vec![PlanStep {
                    name: "Create View".into(),
                    step_type: PlanStepType::Ddl,
                    estimated_rows: 0,
                    cost: 0.1,
                }],
                estimated_cost: 0.1,
            });
        }

        if upper.starts_with("DROP VIEW ") {
            let name = query[10..].trim().trim_end_matches(';').trim().to_string();
            return Ok(ServiceQueryPlan {
                statement: Statement::DropView(name.clone()),
                collection: name,
                limit: 0,
                steps: vec![PlanStep {
                    name: "Drop View".into(),
                    step_type: PlanStepType::Ddl,
                    estimated_rows: 0,
                    cost: 0.1,
                }],
                estimated_cost: 0.1,
            });
        }

        Err(NeedleError::InvalidArgument(format!(
            "Unrecognized statement: {}",
            &query[..query.len().min(50)]
        )))
    }

    /// Validate a NeedleQL query without executing.
    ///
    /// Uses [`QueryParser`] + [`QueryValidator`] for SELECT queries.
    pub fn validate(&self, query: &str) -> Result<()> {
        self.parse(query)?;
        Ok(())
    }

    /// Get query execution statistics.
    pub fn query_count(&self) -> u64 {
        self.query_count
    }

    /// Record a query execution.
    pub fn record_query(&mut self) {
        self.query_count += 1;
    }

    /// Get config.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Convert a service-layer WHERE clause to a JSON filter (MongoDB-style).
    pub fn where_to_filter(clause: &ServiceWhereClause) -> serde_json::Value {
        match clause {
            ServiceWhereClause::Comparison { field, op, value } => {
                let val = literal_to_json(value);
                let op_str = match op {
                    ComparisonOp::Eq => "$eq",
                    ComparisonOp::Ne => "$ne",
                    ComparisonOp::Lt => "$lt",
                    ComparisonOp::Le => "$lte",
                    ComparisonOp::Gt => "$gt",
                    ComparisonOp::Ge => "$gte",
                    ComparisonOp::Like => "$eq",
                };
                if *op == ComparisonOp::Eq {
                    serde_json::json!({ field.as_str(): val })
                } else {
                    serde_json::json!({ field.as_str(): { op_str: val } })
                }
            }
            ServiceWhereClause::And(clauses) => {
                let items: Vec<_> = clauses.iter().map(Self::where_to_filter).collect();
                serde_json::json!({ "$and": items })
            }
            ServiceWhereClause::Or(clauses) => {
                let items: Vec<_> = clauses.iter().map(Self::where_to_filter).collect();
                serde_json::json!({ "$or": items })
            }
            ServiceWhereClause::Not(clause) => {
                serde_json::json!({ "$not": Self::where_to_filter(clause) })
            }
            ServiceWhereClause::In { field, values } => {
                let vals: Vec<_> = values.iter().map(literal_to_json).collect();
                serde_json::json!({ field.as_str(): { "$in": vals } })
            }
        }
    }

    // ── QueryParser delegation ──────────────────────────────────────────

    /// Parse a SELECT statement by delegating to [`QueryParser`] and mapping
    /// the resulting [`Query`] AST into the service-layer [`SelectQuery`].
    fn parse_select_via_query_parser(&self, input: &str) -> Result<SelectQuery> {
        let ql_query = QueryParser::parse(input).map_err(|e| {
            NeedleError::InvalidArgument(format!("NeedleQL parse error: {e}"))
        })?;

        // Validate via QueryValidator (catches semantic issues early).
        QueryValidator::validate(&ql_query).map_err(|e| {
            NeedleError::InvalidArgument(format!("NeedleQL validation error: {e}"))
        })?;

        // Map SelectClause
        let columns = match &ql_query.select {
            SelectClause::All => vec![SelectColumn::Star],
            SelectClause::Columns(cols) => {
                cols.iter().map(|c| SelectColumn::Named(c.clone())).collect()
            }
        };

        let collection = ql_query.from.collection.clone();

        // Map WHERE clause
        let filter = ql_query.where_clause.as_ref().map(|wc| {
            map_expression_to_service(&wc.expression)
        });

        // Detect NEAREST_TO from SIMILAR TO expression in WHERE clause
        let nearest_to = Self::extract_nearest_to(&ql_query);

        // Map ORDER BY
        let order_by = ql_query.order_by.as_ref().and_then(|ob| {
            ob.columns.first().map(|(field, dir)| OrderBy {
                field: field.clone(),
                direction: match dir {
                    SortOrder::Asc => SortDirection::Asc,
                    SortOrder::Desc => SortDirection::Desc,
                },
            })
        });

        let limit = ql_query
            .limit
            .map(|l| (l as usize).min(self.config.max_limit))
            .unwrap_or(self.config.default_limit);
        let offset = ql_query.offset.map(|o| o as usize).unwrap_or(0);

        // Map WITH options
        let mut options = HashMap::new();
        if let Some(WithClause::TimeDecay(td)) = &ql_query.with_clause {
            options.insert(
                "time_decay_function".to_string(),
                OptionValue::String(format!("{:?}", td.function)),
            );
            for (k, v) in &td.params {
                let ov = if let Some(n) = v.as_f64() {
                    OptionValue::Number(n)
                } else if let Some(b) = v.as_bool() {
                    OptionValue::Bool(b)
                } else {
                    OptionValue::String(v.to_string().trim_matches('"').to_string())
                };
                options.insert(k.clone(), ov);
            }
        }

        // Hybrid search from USING RAG clause
        let hybrid_search = ql_query.using_clause.as_ref().map(|uc| HybridSearch {
            text_query: String::new(),
            vector: None,
            vector_weight: uc.rag.hybrid_alpha.map(|a| a as f64).unwrap_or(0.5),
            text_weight: uc
                .rag
                .hybrid_alpha
                .map(|a| 1.0 - a as f64)
                .unwrap_or(0.5),
            rrf_k: 60.0,
        });

        // Parse AS OF clause from options or query string
        let as_of = options.get("as_of_timestamp").map(|v| {
            match v {
                OptionValue::Number(ts) => AsOfClause {
                    timestamp: Some(*ts as u64),
                    time_expression: None,
                    snapshot: None,
                    version: None,
                },
                OptionValue::String(expr) => AsOfClause {
                    timestamp: None,
                    time_expression: Some(expr.clone()),
                    snapshot: None,
                    version: None,
                },
                _ => AsOfClause {
                    timestamp: None,
                    time_expression: None,
                    snapshot: None,
                    version: None,
                },
            }
        });

        Ok(SelectQuery {
            columns,
            collection,
            filter,
            nearest_to,
            hybrid_search,
            order_by,
            limit,
            offset,
            options,
            as_of,
        })
    }

    /// Extract a NEAREST_TO from SIMILAR TO expression in the parsed query.
    fn extract_nearest_to(ql_query: &Query) -> Option<NearestTo> {
        if let Some(wc) = &ql_query.where_clause {
            if Self::has_similar_to(&wc.expression) {
                return Some(NearestTo {
                    vector: Vec::new(),
                    distance_fn: None,
                    ef_search: Some(50),
                });
            }
        }
        None
    }

    fn has_similar_to(expr: &Expression) -> bool {
        match expr {
            Expression::SimilarTo(_) => true,
            Expression::And(l, r) | Expression::Or(l, r) => {
                Self::has_similar_to(l) || Self::has_similar_to(r)
            }
            Expression::Not(e) | Expression::Grouped(e) => Self::has_similar_to(e),
            _ => false,
        }
    }

    // ── Fallback parsing for non-SELECT statements ──────────────────────

    fn parse_insert(&self, query: &str) -> Result<InsertStatement> {
        let upper = query.to_uppercase();
        let into_pos = upper.find("INTO ").ok_or_else(|| {
            NeedleError::InvalidArgument("Missing INTO in INSERT".into())
        })?;
        let after_into = &query[into_pos + 5..].trim();
        let (collection, _rest) = Self::extract_identifier(after_into);
        if collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "Missing collection name in INSERT".into(),
            ));
        }
        Ok(InsertStatement {
            collection,
            id: String::new(),
            vector: Vec::new(),
            metadata: None,
        })
    }

    fn parse_delete(&self, query: &str) -> Result<DeleteStatement> {
        let upper = query.to_uppercase();
        let from_pos = upper.find("FROM ").ok_or_else(|| {
            NeedleError::InvalidArgument("Missing FROM in DELETE".into())
        })?;
        let after_from = &query[from_pos + 5..].trim();
        let (collection, rest) = Self::extract_identifier(after_from);
        if collection.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "Missing collection name in DELETE".into(),
            ));
        }
        let rest_upper = rest.to_uppercase();
        let filter = self.parse_where_clause_fallback(&rest, &rest_upper);
        Ok(DeleteStatement {
            collection,
            filter,
            ids: Vec::new(),
        })
    }

    fn parse_create_collection(&self, query: &str) -> Result<CreateCollectionStatement> {
        let upper = query.to_uppercase();
        let if_not_exists = upper.contains("IF NOT EXISTS");

        let after = if if_not_exists {
            let pos = upper.find("EXISTS ").unwrap_or(18) + 7;
            &query[pos..]
        } else {
            &query[18..]
        };
        let after = after.trim();
        let (name, rest) = Self::extract_identifier(after);
        if name.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "Missing collection name in CREATE COLLECTION".into(),
            ));
        }

        let rest_upper = rest.to_uppercase();
        let dimensions = if let Some(pos) = rest_upper.find("DIMENSIONS ") {
            let after = &rest[pos + 11..];
            let end = after
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(after.len());
            after[..end].parse().unwrap_or(0)
        } else if let Some(paren_start) = rest.find('(') {
            let paren_end = rest.find(')').unwrap_or(rest.len());
            rest[paren_start + 1..paren_end]
                .trim()
                .parse()
                .unwrap_or(0)
        } else {
            0
        };

        Ok(CreateCollectionStatement {
            name,
            dimensions,
            distance_fn: None,
            if_not_exists,
        })
    }

    fn extract_identifier(s: &str) -> (String, String) {
        let s = s.trim();
        let end = s
            .find(|c: char| c.is_whitespace())
            .unwrap_or(s.len());
        let ident = s[..end].to_string();
        let rest = s[end..].to_string();
        (ident, rest)
    }

    /// Extract AS OF clause from a query string, returning the cleaned query and the clause.
    /// Supports: `AS OF TIMESTAMP <unix_ts>`, `AS OF VERSION <version>`,
    /// `AS OF SNAPSHOT '<name>'`, and `AS OF '<time_expression>'`.
    fn extract_as_of_clause(&self, query: &str) -> (String, Option<AsOfClause>) {
        let upper = query.to_uppercase();
        let as_of_pos = match upper.find(" AS OF ") {
            Some(pos) => pos,
            None => return (query.to_string(), None),
        };

        let before = &query[..as_of_pos];
        let after = query[as_of_pos + 7..].trim();
        let after_upper = after.to_uppercase();

        let (clause, rest) = if after_upper.starts_with("TIMESTAMP ") {
            let val_str = &after[10..].trim();
            let end = val_str
                .find(|c: char| c.is_whitespace())
                .unwrap_or(val_str.len());
            let ts: u64 = val_str[..end].parse().unwrap_or(0);
            (
                AsOfClause {
                    timestamp: Some(ts),
                    time_expression: None,
                    snapshot: None,
                    version: None,
                },
                val_str[end..].to_string(),
            )
        } else if after_upper.starts_with("VERSION ") {
            let val_str = &after[8..].trim();
            let end = val_str
                .find(|c: char| c.is_whitespace())
                .unwrap_or(val_str.len());
            let version: u64 = val_str[..end].parse().unwrap_or(0);
            (
                AsOfClause {
                    timestamp: None,
                    time_expression: None,
                    snapshot: None,
                    version: Some(version),
                },
                val_str[end..].to_string(),
            )
        } else if after_upper.starts_with("SNAPSHOT ") {
            let val_str = &after[9..].trim();
            let name = val_str.trim_matches('\'').trim_matches('"');
            let end = val_str
                .find(|c: char| c.is_whitespace() && c != '\'' && c != '"')
                .unwrap_or(val_str.len());
            (
                AsOfClause {
                    timestamp: None,
                    time_expression: None,
                    snapshot: Some(name.to_string()),
                    version: None,
                },
                val_str[end..].to_string(),
            )
        } else {
            // Time expression in quotes: AS OF 'yesterday'
            let expr = after.trim_matches('\'').trim_matches('"');
            let end = if after.starts_with('\'') || after.starts_with('"') {
                let quote = &after[..1];
                after[1..]
                    .find(quote)
                    .map(|p| p + 2)
                    .unwrap_or(after.len())
            } else {
                after
                    .find(|c: char| c == ';')
                    .unwrap_or(after.len())
            };
            let expr_end = expr
                .find(|c: char| c == '\'' || c == '"' || c == ';')
                .unwrap_or(expr.len());
            (
                AsOfClause {
                    timestamp: None,
                    time_expression: Some(expr[..expr_end].trim().to_string()),
                    snapshot: None,
                    version: None,
                },
                after[end..].to_string(),
            )
        };

        let cleaned_query = format!("{} {}", before.trim(), rest.trim());
        (cleaned_query.trim().to_string(), Some(clause))
    }

    /// Fallback WHERE parser for DELETE (non-SELECT) statements.
    fn parse_where_clause_fallback(
        &self,
        rest: &str,
        rest_upper: &str,
    ) -> Option<ServiceWhereClause> {
        let where_pos = rest_upper.find(" WHERE ")?;
        let after_where = &rest[where_pos + 7..];
        let end_markers = [" NEAREST_TO", " HYBRID_SEARCH", " ORDER ", " LIMIT ", " WITH ", " OFFSET "];
        let after_where_upper = after_where.to_uppercase();
        let end = end_markers
            .iter()
            .filter_map(|m| after_where_upper.find(m))
            .min()
            .unwrap_or(after_where.len());
        let clause_str = after_where[..end].trim();
        self.parse_condition(clause_str)
    }

    fn parse_condition(&self, s: &str) -> Option<ServiceWhereClause> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        let upper = s.to_uppercase();

        if let Some(pos) = upper.find(" AND ") {
            let left = self.parse_condition(&s[..pos]);
            let right = self.parse_condition(&s[pos + 5..]);
            match (left, right) {
                (Some(l), Some(r)) => return Some(ServiceWhereClause::And(vec![l, r])),
                (Some(l), None) => return Some(l),
                (None, Some(r)) => return Some(r),
                _ => return None,
            }
        }

        if let Some(pos) = upper.find(" OR ") {
            let left = self.parse_condition(&s[..pos]);
            let right = self.parse_condition(&s[pos + 4..]);
            match (left, right) {
                (Some(l), Some(r)) => return Some(ServiceWhereClause::Or(vec![l, r])),
                (Some(l), None) => return Some(l),
                (None, Some(r)) => return Some(r),
                _ => return None,
            }
        }

        if let Some(in_pos) = upper.find(" IN ") {
            let field = s[..in_pos].trim().to_string();
            let values_str = s[in_pos + 4..].trim();
            let values_str = values_str
                .trim_start_matches('(')
                .trim_end_matches(')');
            let values: Vec<LiteralValue> = values_str
                .split(',')
                .map(|v| parse_literal(v.trim()))
                .collect();
            return Some(ServiceWhereClause::In { field, values });
        }

        let ops = [
            ("!=", ComparisonOp::Ne),
            (">=", ComparisonOp::Ge),
            ("<=", ComparisonOp::Le),
            (">", ComparisonOp::Gt),
            ("<", ComparisonOp::Lt),
            ("=", ComparisonOp::Eq),
        ];
        for (op_str, op) in &ops {
            if let Some(pos) = s.find(op_str) {
                let field = s[..pos].trim().to_string();
                let value_str = s[pos + op_str.len()..].trim();
                return Some(ServiceWhereClause::Comparison {
                    field,
                    op: *op,
                    value: parse_literal(value_str),
                });
            }
        }

        None
    }

    fn plan_steps(&self, statement: &Statement) -> Vec<PlanStep> {
        match statement {
            Statement::Select(q) | Statement::Explain(q) => {
                let mut steps = Vec::new();
                if let Some(as_of) = &q.as_of {
                    steps.push(PlanStep {
                        name: "Time-Travel Scan".into(),
                        step_type: PlanStepType::TimeTravelScan {
                            timestamp: as_of.timestamp,
                            version: as_of.version,
                        },
                        estimated_rows: q.limit * 5,
                        cost: 5.0,
                    });
                }
                if q.nearest_to.is_some() {
                    let k = q.limit;
                    steps.push(PlanStep {
                        name: "HNSW Search".into(),
                        step_type: PlanStepType::VectorSearch {
                            ef_search: self.config.default_ef_search,
                            k,
                        },
                        estimated_rows: k,
                        cost: 1.0,
                    });
                }
                if let Some(hs) = &q.hybrid_search {
                    steps.push(PlanStep {
                        name: "BM25 Search".into(),
                        step_type: PlanStepType::TextSearch {
                            query: hs.text_query.clone(),
                        },
                        estimated_rows: q.limit * 2,
                        cost: 1.5,
                    });
                    steps.push(PlanStep {
                        name: "Hybrid Fusion".into(),
                        step_type: PlanStepType::HybridFusion {
                            vector_weight: hs.vector_weight,
                        },
                        estimated_rows: q.limit,
                        cost: 0.5,
                    });
                }
                if let Some(w) = &q.filter {
                    let field = match w {
                        ServiceWhereClause::Comparison { field, .. } => field.clone(),
                        _ => "compound".into(),
                    };
                    steps.push(PlanStep {
                        name: "Metadata Filter".into(),
                        step_type: PlanStepType::MetadataFilter { field },
                        estimated_rows: q.limit,
                        cost: 0.3,
                    });
                }
                if q.nearest_to.is_none() && q.hybrid_search.is_none() {
                    steps.push(PlanStep {
                        name: "Full Scan".into(),
                        step_type: PlanStepType::Scan,
                        estimated_rows: q.limit * 10,
                        cost: 10.0,
                    });
                }
                if let Some(ob) = &q.order_by {
                    steps.push(PlanStep {
                        name: "Sort".into(),
                        step_type: PlanStepType::Sort {
                            field: ob.field.clone(),
                        },
                        estimated_rows: q.limit,
                        cost: 0.2,
                    });
                }
                steps.push(PlanStep {
                    name: "Limit".into(),
                    step_type: PlanStepType::Limit { count: q.limit },
                    estimated_rows: q.limit,
                    cost: 0.01,
                });
                steps
            }
            Statement::Insert(_) => vec![PlanStep {
                name: "Insert".into(),
                step_type: PlanStepType::Insert,
                estimated_rows: 1,
                cost: 1.0,
            }],
            Statement::Delete(_) => vec![PlanStep {
                name: "Delete".into(),
                step_type: PlanStepType::Delete,
                estimated_rows: 1,
                cost: 1.0,
            }],
            Statement::CreateCollection(_) | Statement::DropCollection(_) | Statement::CreateView(_) | Statement::DropView(_) => vec![PlanStep {
                name: "DDL".into(),
                step_type: PlanStepType::Ddl,
                estimated_rows: 0,
                cost: 0.1,
            }],
            Statement::ShowCollections => vec![],
        }
    }
}

impl Default for NeedleQLExecutor {
    fn default() -> Self {
        Self::new(ExecutorConfig::default())
    }
}

// ── Mapping helpers: query_lang AST → service-layer types ───────────────────

/// Map a `query_lang::Expression` into a service-layer `ServiceWhereClause`.
fn map_expression_to_service(expr: &Expression) -> ServiceWhereClause {
    match expr {
        Expression::Comparison(comp) => ServiceWhereClause::Comparison {
            field: comp.column.clone(),
            op: match comp.operator {
                CompareOp::Eq => ComparisonOp::Eq,
                CompareOp::Ne => ComparisonOp::Ne,
                CompareOp::Lt => ComparisonOp::Lt,
                CompareOp::Le => ComparisonOp::Le,
                CompareOp::Gt => ComparisonOp::Gt,
                CompareOp::Ge => ComparisonOp::Ge,
            },
            value: map_literal(&comp.value),
        },
        Expression::And(l, r) => ServiceWhereClause::And(vec![
            map_expression_to_service(l),
            map_expression_to_service(r),
        ]),
        Expression::Or(l, r) => ServiceWhereClause::Or(vec![
            map_expression_to_service(l),
            map_expression_to_service(r),
        ]),
        Expression::Not(inner) => {
            ServiceWhereClause::Not(Box::new(map_expression_to_service(inner)))
        }
        Expression::InList(in_expr) => ServiceWhereClause::In {
            field: in_expr.column.clone(),
            values: in_expr.values.iter().map(map_literal).collect(),
        },
        Expression::Like(like) => ServiceWhereClause::Comparison {
            field: like.column.clone(),
            op: ComparisonOp::Like,
            value: LiteralValue::String(like.pattern.clone()),
        },
        Expression::IsNull(is_null) => {
            let cmp = ServiceWhereClause::Comparison {
                field: is_null.column.clone(),
                op: ComparisonOp::Eq,
                value: LiteralValue::Null,
            };
            if is_null.negated {
                ServiceWhereClause::Not(Box::new(cmp))
            } else {
                cmp
            }
        }
        Expression::SimilarTo(_) => {
            // SIMILAR TO is handled as nearest_to, emit a pass-through
            ServiceWhereClause::And(vec![])
        }
        Expression::Between(between) => ServiceWhereClause::And(vec![
            ServiceWhereClause::Comparison {
                field: between.column.clone(),
                op: ComparisonOp::Ge,
                value: map_literal(&between.low),
            },
            ServiceWhereClause::Comparison {
                field: between.column.clone(),
                op: ComparisonOp::Le,
                value: map_literal(&between.high),
            },
        ]),
        Expression::Grouped(inner) => map_expression_to_service(inner),
    }
}

/// Map a `query_lang::LiteralValue` to the service-layer `LiteralValue`.
fn map_literal(ql: &QLLiteralValue) -> LiteralValue {
    match ql {
        QLLiteralValue::String(s) => LiteralValue::String(s.clone()),
        QLLiteralValue::Number(n) => LiteralValue::Number(*n),
        QLLiteralValue::Bool(b) => LiteralValue::Bool(*b),
        QLLiteralValue::Null => LiteralValue::Null,
        QLLiteralValue::Parameter(p) => LiteralValue::String(format!("${p}")),
    }
}

fn parse_literal(s: &str) -> LiteralValue {
    let s = s.trim();
    if s.eq_ignore_ascii_case("null") {
        LiteralValue::Null
    } else if s.eq_ignore_ascii_case("true") {
        LiteralValue::Bool(true)
    } else if s.eq_ignore_ascii_case("false") {
        LiteralValue::Bool(false)
    } else if let Ok(n) = s.parse::<f64>() {
        LiteralValue::Number(n)
    } else {
        LiteralValue::String(s.trim_matches('\'').trim_matches('"').to_string())
    }
}

fn literal_to_json(lit: &LiteralValue) -> serde_json::Value {
    match lit {
        LiteralValue::String(s) => serde_json::Value::String(s.clone()),
        LiteralValue::Number(n) => serde_json::json!(n),
        LiteralValue::Bool(b) => serde_json::Value::Bool(*b),
        LiteralValue::Null => serde_json::Value::Null,
        LiteralValue::Vector(v) => serde_json::json!(v),
    }
}

// ── Type alias for backward compatibility ───────────────────────────────────

/// Backward-compatible alias — older call-sites may refer to `WhereClause`.
pub type WhereClause = ServiceWhereClause;
/// Backward-compatible alias — older call-sites may refer to `QueryPlan`.
pub type QueryPlan = ServiceQueryPlan;

#[cfg(test)]
mod tests {
    use super::*;

    fn executor() -> NeedleQLExecutor {
        NeedleQLExecutor::new(ExecutorConfig::default())
    }

    #[test]
    fn test_parse_simple_select() {
        let ex = executor();
        let plan = ex.parse("SELECT * FROM docs LIMIT 10").unwrap();
        assert_eq!(plan.collection, "docs");
        assert_eq!(plan.limit, 10);
        assert!(matches!(plan.statement, Statement::Select(_)));
    }

    #[test]
    fn test_parse_select_with_columns() {
        let ex = executor();
        let plan = ex.parse("SELECT id, distance FROM docs LIMIT 5").unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert_eq!(q.columns.len(), 2);
        } else {
            panic!("Expected Select");
        }
    }

    #[test]
    fn test_parse_nearest_to() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 5")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert!(q.nearest_to.is_some());
        }
    }

    #[test]
    fn test_parse_where_eq() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs WHERE category = 'books' LIMIT 10")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert!(q.filter.is_some());
        }
    }

    #[test]
    fn test_parse_where_and() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs WHERE category = 'books' AND price > 10 LIMIT 10")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert!(q.filter.is_some());
            // The query_lang parser produces And(Comparison, Comparison)
            // which maps to ServiceWhereClause::And(...)
            assert!(matches!(q.filter, Some(ServiceWhereClause::And(_))));
        }
    }

    #[test]
    fn test_parse_hybrid_search() {
        let ex = executor();
        let plan = ex
            .parse(
                "SELECT * FROM docs USING RAG(top_k=5, rerank=true) WHERE vector SIMILAR TO $query LIMIT 10",
            )
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert!(q.hybrid_search.is_some());
        }
    }

    #[test]
    fn test_parse_explain() {
        let ex = executor();
        let plan = ex
            .parse("EXPLAIN SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 5")
            .unwrap();
        assert!(matches!(plan.statement, Statement::Explain(_)));
        assert!(!plan.steps.is_empty());
    }

    #[test]
    fn test_parse_order_by() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs ORDER BY distance ASC LIMIT 10")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            let ob = q.order_by.as_ref().unwrap();
            assert_eq!(ob.field, "distance");
            assert_eq!(ob.direction, SortDirection::Asc);
        }
    }

    #[test]
    fn test_parse_create_collection() {
        let ex = executor();
        let plan = ex
            .parse("CREATE COLLECTION embeddings DIMENSIONS 384")
            .unwrap();
        if let Statement::CreateCollection(c) = &plan.statement {
            assert_eq!(c.name, "embeddings");
            assert_eq!(c.dimensions, 384);
        }
    }

    #[test]
    fn test_parse_drop_collection() {
        let ex = executor();
        let plan = ex.parse("DROP COLLECTION old_stuff").unwrap();
        assert!(matches!(plan.statement, Statement::DropCollection(_)));
    }

    #[test]
    fn test_parse_show_collections() {
        let ex = executor();
        let plan = ex.parse("SHOW COLLECTIONS").unwrap();
        assert!(matches!(plan.statement, Statement::ShowCollections));
    }

    #[test]
    fn test_parse_insert() {
        let ex = executor();
        let plan = ex.parse("INSERT INTO docs VALUES ()").unwrap();
        assert!(matches!(plan.statement, Statement::Insert(_)));
    }

    #[test]
    fn test_parse_delete() {
        let ex = executor();
        let plan = ex.parse("DELETE FROM docs WHERE id = 'doc1'").unwrap();
        if let Statement::Delete(d) = &plan.statement {
            assert_eq!(d.collection, "docs");
            assert!(d.filter.is_some());
        }
    }

    #[test]
    fn test_empty_query_error() {
        let ex = executor();
        assert!(ex.parse("").is_err());
    }

    #[test]
    fn test_too_long_query() {
        let ex = NeedleQLExecutor::new(ExecutorConfig {
            max_query_length: 10,
            ..Default::default()
        });
        assert!(ex.parse("SELECT * FROM very_long_collection_name LIMIT 10").is_err());
    }

    #[test]
    fn test_validate() {
        let ex = executor();
        assert!(ex.validate("SELECT * FROM docs LIMIT 5").is_ok());
        assert!(ex.validate("INVALID STUFF").is_err());
    }

    #[test]
    fn test_where_to_filter_eq() {
        let clause = ServiceWhereClause::Comparison {
            field: "category".into(),
            op: ComparisonOp::Eq,
            value: LiteralValue::String("books".into()),
        };
        let filter = NeedleQLExecutor::where_to_filter(&clause);
        assert_eq!(filter["category"], "books");
    }

    #[test]
    fn test_where_to_filter_and() {
        let clause = ServiceWhereClause::And(vec![
            ServiceWhereClause::Comparison {
                field: "a".into(),
                op: ComparisonOp::Gt,
                value: LiteralValue::Number(5.0),
            },
            ServiceWhereClause::Comparison {
                field: "b".into(),
                op: ComparisonOp::Eq,
                value: LiteralValue::String("x".into()),
            },
        ]);
        let filter = NeedleQLExecutor::where_to_filter(&clause);
        assert!(filter["$and"].is_array());
    }

    #[test]
    fn test_with_options() {
        let ex = executor();
        let plan = ex
            .parse(
                "SELECT * FROM docs WITH TIME_DECAY(EXPONENTIAL, half_life=7d) WHERE vector SIMILAR TO $query LIMIT 5",
            )
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert!(q.options.contains_key("half_life"));
        }
    }

    #[test]
    fn test_plan_steps_vector_search() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 5")
            .unwrap();
        assert!(plan.steps.iter().any(|s| matches!(s.step_type, PlanStepType::VectorSearch { .. })));
    }

    #[test]
    fn test_plan_steps_scan_fallback() {
        let ex = executor();
        let plan = ex.parse("SELECT * FROM docs LIMIT 5").unwrap();
        assert!(plan.steps.iter().any(|s| matches!(s.step_type, PlanStepType::Scan)));
    }

    #[test]
    fn test_select_with_offset() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs LIMIT 10 OFFSET 20")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            assert_eq!(q.offset, 20);
        }
    }

    #[test]
    fn test_max_limit_enforcement() {
        let ex = NeedleQLExecutor::new(ExecutorConfig {
            max_limit: 100,
            ..Default::default()
        });
        let plan = ex.parse("SELECT * FROM docs LIMIT 50000").unwrap();
        assert_eq!(plan.limit, 100);
    }

    #[test]
    fn test_as_of_timestamp() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs AS OF TIMESTAMP 1700000000 LIMIT 10")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            let as_of = q.as_of.as_ref().expect("AS OF should be present");
            assert_eq!(as_of.timestamp, Some(1_700_000_000));
        } else {
            panic!("Expected Select");
        }
        assert!(plan.steps.iter().any(|s| matches!(s.step_type, PlanStepType::TimeTravelScan { .. })));
    }

    #[test]
    fn test_as_of_version() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs AS OF VERSION 42 LIMIT 10")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            let as_of = q.as_of.as_ref().expect("AS OF should be present");
            assert_eq!(as_of.version, Some(42));
        } else {
            panic!("Expected Select");
        }
    }

    #[test]
    fn test_as_of_expression() {
        let ex = executor();
        let plan = ex
            .parse("SELECT * FROM docs AS OF 'yesterday' LIMIT 10")
            .unwrap();
        if let Statement::Select(q) = &plan.statement {
            let as_of = q.as_of.as_ref().expect("AS OF should be present");
            assert_eq!(as_of.time_expression, Some("yesterday".to_string()));
        } else {
            panic!("Expected Select");
        }
    }

    #[test]
    fn test_query_parser_delegation() {
        // Verify that QueryParser from query_lang is used
        let result = QueryParser::parse("SELECT * FROM docs LIMIT 10");
        assert!(result.is_ok());
        let q = result.unwrap();
        assert_eq!(q.from.collection, "docs");
    }

    #[test]
    fn test_query_validator_delegation() {
        // Verify QueryValidator catches semantic errors
        let q = QueryParser::parse(
            "SELECT * FROM docs USING RAG(top_k=5) WHERE category = 'science'",
        )
        .unwrap();
        let result = QueryValidator::validate(&q);
        assert!(result.is_err());
    }
}
