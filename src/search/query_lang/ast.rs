use serde_json::Value;
use std::collections::HashMap;

/// A complete NeedleQL query
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// Whether this is an EXPLAIN ANALYZE query
    pub explain: bool,
    /// SELECT columns (* or list)
    pub select: SelectClause,
    /// FROM collection
    pub from: FromClause,
    /// Optional WITH clause for modifiers
    pub with_clause: Option<WithClause>,
    /// Optional USING clause for RAG
    pub using_clause: Option<UsingClause>,
    /// Optional WHERE clause
    pub where_clause: Option<WhereClause>,
    /// Optional ORDER BY clause
    pub order_by: Option<OrderByClause>,
    /// Optional LIMIT
    pub limit: Option<u64>,
    /// Optional OFFSET
    pub offset: Option<u64>,
}

/// SELECT clause
#[derive(Debug, Clone, PartialEq)]
pub enum SelectClause {
    /// SELECT *
    All,
    /// SELECT column1, column2, ...
    Columns(Vec<String>),
}

/// FROM clause
#[derive(Debug, Clone, PartialEq)]
pub struct FromClause {
    /// Collection name
    pub collection: String,
    /// Optional alias
    pub alias: Option<String>,
}

/// WITH clause for query modifiers
#[derive(Debug, Clone, PartialEq)]
pub enum WithClause {
    /// TIME_DECAY modifier
    TimeDecay(TimeDecayConfig),
}

/// Time decay configuration
#[derive(Debug, Clone, PartialEq)]
pub struct TimeDecayConfig {
    /// Decay function type
    pub function: TimeDecayFunction,
    /// Parameters
    pub params: HashMap<String, Value>,
}

/// Time decay function types
#[derive(Debug, Clone, PartialEq)]
pub enum TimeDecayFunction {
    Linear,
    Exponential,
    Gaussian,
    Step,
}

/// USING clause for RAG integration
#[derive(Debug, Clone, PartialEq)]
pub struct UsingClause {
    /// RAG configuration
    pub rag: RagOptions,
}

/// RAG options
#[derive(Debug, Clone, PartialEq)]
pub struct RagOptions {
    /// Number of top results
    pub top_k: Option<usize>,
    /// Enable reranking
    pub rerank: Option<bool>,
    /// Hybrid search alpha
    pub hybrid_alpha: Option<f32>,
    /// Enable deduplication
    pub deduplicate: Option<bool>,
}

/// WHERE clause
#[derive(Debug, Clone, PartialEq)]
pub struct WhereClause {
    /// The filter expression
    pub expression: Expression,
}

/// Expression in WHERE clause
#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    /// Column comparison: column op value
    Comparison(ComparisonExpr),
    /// Vector similarity: vector SIMILAR TO $query
    SimilarTo(SimilarToExpr),
    /// IN expression: column IN (values)
    InList(InListExpr),
    /// BETWEEN expression: column BETWEEN a AND b
    Between(BetweenExpr),
    /// LIKE expression: column LIKE pattern
    Like(LikeExpr),
    /// IS NULL / IS NOT NULL
    IsNull(IsNullExpr),
    /// AND expression
    And(Box<Expression>, Box<Expression>),
    /// OR expression
    Or(Box<Expression>, Box<Expression>),
    /// NOT expression
    Not(Box<Expression>),
    /// Parenthesized expression
    Grouped(Box<Expression>),
}

/// Comparison expression
#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonExpr {
    pub column: String,
    pub operator: CompareOp,
    pub value: LiteralValue,
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Literal values
#[derive(Debug, Clone, PartialEq)]
pub enum LiteralValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
    Parameter(String),
}

/// SIMILAR TO expression
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarToExpr {
    pub column: String,
    pub query_param: String,
}

/// IN list expression
#[derive(Debug, Clone, PartialEq)]
pub struct InListExpr {
    pub column: String,
    pub values: Vec<LiteralValue>,
    pub negated: bool,
}

/// BETWEEN expression
#[derive(Debug, Clone, PartialEq)]
pub struct BetweenExpr {
    pub column: String,
    pub low: LiteralValue,
    pub high: LiteralValue,
    pub negated: bool,
}

/// LIKE expression
#[derive(Debug, Clone, PartialEq)]
pub struct LikeExpr {
    pub column: String,
    pub pattern: String,
    pub negated: bool,
}

/// IS NULL expression
#[derive(Debug, Clone, PartialEq)]
pub struct IsNullExpr {
    pub column: String,
    pub negated: bool,
}

/// ORDER BY clause
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByClause {
    pub columns: Vec<(String, SortOrder)>,
}

/// Sort order
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests needed: see docs/TODO-test-coverage.md
}
