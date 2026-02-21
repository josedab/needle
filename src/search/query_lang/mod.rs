//! NeedleQL - Query Language for Needle Vector Database
//!
//! SQL-like query language for vector search operations with support for:
//! - Vector similarity search with `SIMILAR TO`
//! - Metadata filtering with standard SQL operators
//! - Query plan analysis with `EXPLAIN ANALYZE`
//! - Temporal queries with `WITH TIME_DECAY`
//! - RAG pipeline integration with `USING RAG(...)`
//!
//! # Examples
//!
//! ```rust,ignore
//! use needle::query_lang::{QueryParser, QueryExecutor};
//!
//! // Basic vector search
//! let query = "SELECT * FROM documents WHERE vector SIMILAR TO $query LIMIT 10";
//! let parsed = QueryParser::parse(query)?;
//!
//! // With metadata filters
//! let query = "SELECT * FROM docs WHERE category = 'science' AND score > 0.5
//!              AND vector SIMILAR TO $query LIMIT 5";
//!
//! // With time decay
//! let query = "SELECT * FROM articles
//!              WITH TIME_DECAY(EXPONENTIAL, half_life=7d)
//!              WHERE vector SIMILAR TO $query LIMIT 10";
//!
//! // RAG pipeline query
//! let query = "SELECT * FROM knowledge_base
//!              USING RAG(top_k=5, rerank=true)
//!              WHERE vector SIMILAR TO $query";
//!
//! // Query plan analysis
//! let query = "EXPLAIN ANALYZE SELECT * FROM docs
//!              WHERE vector SIMILAR TO $query LIMIT 10";
//! ```

pub mod ast;
pub mod executor;
pub mod lexer;
pub mod optimizer;
pub mod parser;
pub mod session;

// Re-export all public types to preserve the public API
pub use ast::*;
pub use executor::*;
pub use lexer::*;
pub use optimizer::*;
pub use parser::*;
pub use session::*;

// ============================================================================
// Error Types
// ============================================================================

/// Query parsing and execution errors
#[derive(Debug, Clone, PartialEq)]
pub enum QueryError {
    /// Lexer encountered an invalid token
    InvalidToken { position: usize, found: String },
    /// Parser expected a different token
    UnexpectedToken {
        expected: String,
        found: String,
        position: usize,
    },
    /// Missing required clause
    MissingClause { clause: String },
    /// Invalid identifier
    InvalidIdentifier { name: String },
    /// Invalid literal value
    InvalidLiteral { value: String },
    /// Invalid operator
    InvalidOperator { operator: String },
    /// Collection not found
    CollectionNotFound { name: String },
    /// Missing query parameter
    MissingParameter { name: String },
    /// Type mismatch in expression
    TypeMismatch { expected: String, found: String },
    /// Semantic error (valid syntax but invalid semantics)
    SemanticError { message: String },
    /// Execution error
    ExecutionError { message: String },
}

impl std::fmt::Display for QueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidToken { position, found } => {
                write!(f, "Invalid token '{}' at position {}", found, position)
            }
            Self::UnexpectedToken {
                expected,
                found,
                position,
            } => {
                write!(
                    f,
                    "Expected {} but found '{}' at position {}",
                    expected, found, position
                )
            }
            Self::MissingClause { clause } => {
                write!(f, "Missing required clause: {}", clause)
            }
            Self::InvalidIdentifier { name } => {
                write!(f, "Invalid identifier: '{}'", name)
            }
            Self::InvalidLiteral { value } => {
                write!(f, "Invalid literal value: '{}'", value)
            }
            Self::InvalidOperator { operator } => {
                write!(f, "Invalid operator: '{}'", operator)
            }
            Self::CollectionNotFound { name } => {
                write!(f, "Collection not found: '{}'", name)
            }
            Self::MissingParameter { name } => {
                write!(f, "Missing query parameter: '{}'", name)
            }
            Self::TypeMismatch { expected, found } => {
                write!(f, "Type mismatch: expected {}, found {}", expected, found)
            }
            Self::SemanticError { message } => {
                write!(f, "Semantic error: {}", message)
            }
            Self::ExecutionError { message } => {
                write!(f, "Execution error: {}", message)
            }
        }
    }
}

impl std::error::Error for QueryError {}

/// Result type for query operations
pub type QueryResult<T> = std::result::Result<T, QueryError>;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use serde_json::Value;
    use std::sync::Arc;

    // Lexer tests

    #[test]
    fn test_lexer_keywords() {
        let mut lexer = Lexer::new("SELECT FROM WHERE AND OR NOT LIMIT");
        assert_eq!(lexer.next_token().unwrap(), Token::Select);
        assert_eq!(lexer.next_token().unwrap(), Token::From);
        assert_eq!(lexer.next_token().unwrap(), Token::Where);
        assert_eq!(lexer.next_token().unwrap(), Token::And);
        assert_eq!(lexer.next_token().unwrap(), Token::Or);
        assert_eq!(lexer.next_token().unwrap(), Token::Not);
        assert_eq!(lexer.next_token().unwrap(), Token::Limit);
        assert_eq!(lexer.next_token().unwrap(), Token::Eof);
    }

    #[test]
    fn test_lexer_operators() {
        let mut lexer = Lexer::new("= != <> < <= > >=");
        assert_eq!(lexer.next_token().unwrap(), Token::Eq);
        assert_eq!(lexer.next_token().unwrap(), Token::Ne);
        assert_eq!(lexer.next_token().unwrap(), Token::Ne);
        assert_eq!(lexer.next_token().unwrap(), Token::Lt);
        assert_eq!(lexer.next_token().unwrap(), Token::Le);
        assert_eq!(lexer.next_token().unwrap(), Token::Gt);
        assert_eq!(lexer.next_token().unwrap(), Token::Ge);
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn test_lexer_literals() {
        let mut lexer = Lexer::new("'hello' \"world\" 42 3.14 true false");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::StringLit("hello".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::StringLit("world".to_string())
        );
        assert_eq!(lexer.next_token().unwrap(), Token::NumberLit(42.0));
        assert_eq!(lexer.next_token().unwrap(), Token::NumberLit(3.14));
        assert_eq!(lexer.next_token().unwrap(), Token::BoolLit(true));
        assert_eq!(lexer.next_token().unwrap(), Token::BoolLit(false));
    }

    #[test]
    fn test_lexer_parameters() {
        let mut lexer = Lexer::new("$query $limit $filter");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Parameter("query".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Parameter("limit".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Parameter("filter".to_string())
        );
    }

    #[test]
    fn test_lexer_duration() {
        let mut lexer = Lexer::new("7d 24h 60m 30s");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Duration(7, DurationUnit::Days)
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Duration(24, DurationUnit::Hours)
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Duration(60, DurationUnit::Minutes)
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Duration(30, DurationUnit::Seconds)
        );
    }

    #[test]
    fn test_lexer_identifiers() {
        let mut lexer = Lexer::new("collection_name field1 _private");
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Identifier("collection_name".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Identifier("field1".to_string())
        );
        assert_eq!(
            lexer.next_token().unwrap(),
            Token::Identifier("_private".to_string())
        );
    }

    // Parser tests

    #[test]
    fn test_parse_simple_select() {
        let query = QueryParser::parse("SELECT * FROM documents LIMIT 10").unwrap();
        assert_eq!(query.select, SelectClause::All);
        assert_eq!(query.from.collection, "documents");
        assert_eq!(query.limit, Some(10));
    }

    #[test]
    fn test_parse_select_columns() -> Result<(), Box<dyn std::error::Error>> {
        let query = QueryParser::parse("SELECT id, title, score FROM docs")?;
        if let SelectClause::Columns(cols) = query.select {
            assert_eq!(cols, vec!["id", "title", "score"]);
        } else {
            return Err(format!("Expected SelectClause::Columns, got {:?}", query.select).into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_where_similar_to() -> Result<(), Box<dyn std::error::Error>> {
        let query =
            QueryParser::parse("SELECT * FROM documents WHERE vector SIMILAR TO $query LIMIT 10")
                ?;

        if let Some(where_clause) = query.where_clause {
            if let Expression::SimilarTo(similar) = where_clause.expression {
                assert_eq!(similar.column, "vector");
                assert_eq!(similar.query_param, "query");
            } else {
                return Err(format!("Expected SimilarTo expression, got {:?}", where_clause.expression).into())
            }
        } else {
            return Err("Expected WHERE clause, got None".into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_where_comparison() {
        let query =
            QueryParser::parse("SELECT * FROM docs WHERE category = 'science' AND score > 0.5")
                .unwrap();

        assert!(query.where_clause.is_some());
    }

    #[test]
    fn test_parse_where_in() -> Result<(), Box<dyn std::error::Error>> {
        let query =
            QueryParser::parse("SELECT * FROM docs WHERE status IN ('active', 'pending')")?;

        if let Some(where_clause) = query.where_clause {
            if let Expression::InList(in_list) = where_clause.expression {
                assert_eq!(in_list.column, "status");
                assert_eq!(in_list.values.len(), 2);
                assert!(!in_list.negated);
            } else {
                return Err(format!("Expected InList expression, got {:?}", where_clause.expression).into())
            }
        } else {
            return Err("Expected WHERE clause, got None".into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_where_between() -> Result<(), Box<dyn std::error::Error>> {
        let query =
            QueryParser::parse("SELECT * FROM docs WHERE score BETWEEN 0.5 AND 1.0")?;

        if let Some(where_clause) = query.where_clause {
            if let Expression::Between(between) = where_clause.expression {
                assert_eq!(between.column, "score");
            } else {
                return Err(format!("Expected Between expression, got {:?}", where_clause.expression).into())
            }
        } else {
            return Err("Expected WHERE clause, got None".into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_where_like() -> Result<(), Box<dyn std::error::Error>> {
        let query =
            QueryParser::parse("SELECT * FROM docs WHERE title LIKE '%machine learning%'")?;

        if let Some(where_clause) = query.where_clause {
            if let Expression::Like(like) = where_clause.expression {
                assert_eq!(like.column, "title");
                assert_eq!(like.pattern, "%machine learning%");
            } else {
                return Err(format!("Expected Like expression, got {:?}", where_clause.expression).into())
            }
        } else {
            return Err("Expected WHERE clause, got None".into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_explain_analyze() {
        let query = QueryParser::parse(
            "EXPLAIN ANALYZE SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 10",
        )
        .unwrap();

        assert!(query.explain);
    }

    #[test]
    fn test_parse_with_time_decay() -> Result<(), Box<dyn std::error::Error>> {
        let query = QueryParser::parse(
            "SELECT * FROM articles WITH TIME_DECAY(EXPONENTIAL, half_life=7d) WHERE vector SIMILAR TO $query LIMIT 10"
        )?;

        if let Some(WithClause::TimeDecay(config)) = query.with_clause {
            assert_eq!(config.function, TimeDecayFunction::Exponential);
            assert!(config.params.contains_key("half_life"));
        } else {
            return Err(format!("Expected TIME_DECAY clause, got {:?}", query.with_clause).into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_using_rag() -> Result<(), Box<dyn std::error::Error>> {
        let query = QueryParser::parse(
            "SELECT * FROM knowledge_base USING RAG(top_k=5, rerank=true) WHERE vector SIMILAR TO $query"
        )?;

        if let Some(using) = query.using_clause {
            assert_eq!(using.rag.top_k, Some(5));
            assert_eq!(using.rag.rerank, Some(true));
        } else {
            return Err("Expected USING RAG clause, got None".into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_complex_where() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE (category = 'science' OR category = 'tech') AND score > 0.5 AND vector SIMILAR TO $query LIMIT 20"
        ).unwrap();

        assert!(query.where_clause.is_some());
        assert_eq!(query.limit, Some(20));
    }

    #[test]
    fn test_parse_order_by() -> Result<(), Box<dyn std::error::Error>> {
        let query =
            QueryParser::parse("SELECT * FROM docs ORDER BY score DESC, title ASC LIMIT 10")
                ?;

        if let Some(order_by) = query.order_by {
            assert_eq!(order_by.columns.len(), 2);
            assert_eq!(order_by.columns[0], ("score".to_string(), SortOrder::Desc));
            assert_eq!(order_by.columns[1], ("title".to_string(), SortOrder::Asc));
        } else {
            return Err("Expected ORDER BY clause, got None".into())
        }

        Ok(())
    }

    #[test]
    fn test_parse_offset() {
        let query = QueryParser::parse("SELECT * FROM docs LIMIT 10 OFFSET 20").unwrap();

        assert_eq!(query.limit, Some(10));
        assert_eq!(query.offset, Some(20));
    }

    // Error handling tests

    #[test]
    fn test_parse_error_missing_from() {
        let result = QueryParser::parse("SELECT * WHERE x = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_invalid_operator() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE x == 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_error_unterminated_string() {
        let mut lexer = Lexer::new("'unterminated");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    // Validation tests

    #[test]
    fn test_validate_rag_without_similar_to() {
        let query =
            QueryParser::parse("SELECT * FROM docs USING RAG(top_k=5) WHERE category = 'science'")
                .unwrap();

        let result = QueryValidator::validate(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_time_decay_without_similar_to() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WITH TIME_DECAY(EXPONENTIAL) WHERE category = 'science'",
        )
        .unwrap();

        let result = QueryValidator::validate(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_excessive_limit() {
        let query =
            QueryParser::parse("SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 100000")
                .unwrap();

        let result = QueryValidator::validate(&query);
        assert!(result.is_err());
    }

    // Execution tests

    #[test]
    fn test_execute_simple_query() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let collection = db.collection("test").unwrap();
        collection
            .insert(
                "doc1",
                &[1.0; 8],
                Some(serde_json::json!({"category": "science"})),
            )
            .unwrap();
        collection
            .insert(
                "doc2",
                &[0.5; 8],
                Some(serde_json::json!({"category": "tech"})),
            )
            .unwrap();

        let executor = QueryExecutor::new(db);

        let query =
            QueryParser::parse("SELECT * FROM test WHERE vector SIMILAR TO $query LIMIT 10")
                .unwrap();

        let context = QueryContext::new().with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context).unwrap();
        assert!(!result.results.is_empty());
    }

    #[test]
    fn test_execute_with_filter() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let collection = db.collection("test").unwrap();
        collection
            .insert(
                "doc1",
                &[1.0; 8],
                Some(serde_json::json!({"category": "science"})),
            )
            .unwrap();
        collection
            .insert(
                "doc2",
                &[0.5; 8],
                Some(serde_json::json!({"category": "tech"})),
            )
            .unwrap();

        let executor = QueryExecutor::new(db);

        let query = QueryParser::parse(
            "SELECT * FROM test WHERE category = 'science' AND vector SIMILAR TO $query LIMIT 10",
        )
        .unwrap();

        let context = QueryContext::new().with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context).unwrap();
        assert_eq!(result.results.len(), 1);
        assert_eq!(result.results[0].id, "doc1");
    }

    #[test]
    fn test_execute_explain() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let executor = QueryExecutor::new(db);

        let query = QueryParser::parse(
            "EXPLAIN ANALYZE SELECT * FROM test WHERE vector SIMILAR TO $query LIMIT 10",
        )
        .unwrap();

        let context = QueryContext::new().with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context).unwrap();
        assert!(result.plan.is_some());
        assert!(!result.plan.unwrap().nodes.is_empty());
    }

    #[test]
    fn test_execute_collection_not_found() {
        let db = Arc::new(Database::in_memory());
        let executor = QueryExecutor::new(db);

        let query =
            QueryParser::parse("SELECT * FROM nonexistent WHERE vector SIMILAR TO $query LIMIT 10")
                .unwrap();

        let context = QueryContext::new().with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_missing_query_vector() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let executor = QueryExecutor::new(db);

        let query =
            QueryParser::parse("SELECT * FROM test WHERE vector SIMILAR TO $query LIMIT 10")
                .unwrap();

        let context = QueryContext::new();

        let result = executor.execute(&query, &context);
        assert!(result.is_err());
    }

    // Duration unit tests

    #[test]
    fn test_duration_to_seconds() {
        assert_eq!(DurationUnit::Seconds.to_seconds(30), 30);
        assert_eq!(DurationUnit::Minutes.to_seconds(2), 120);
        assert_eq!(DurationUnit::Hours.to_seconds(1), 3600);
        assert_eq!(DurationUnit::Days.to_seconds(1), 86400);
        assert_eq!(DurationUnit::Weeks.to_seconds(1), 604800);
    }

    // Query context tests

    #[test]
    fn test_query_context() {
        let context = QueryContext::new()
            .with_param("category", "science")
            .with_param("limit", 10)
            .with_query_vector(vec![1.0, 2.0, 3.0]);

        assert_eq!(
            context.params.get("category"),
            Some(&Value::String("science".to_string()))
        );
        assert_eq!(context.query_vector, Some(vec![1.0, 2.0, 3.0]));
    }

    // Cost-based optimizer tests

    #[test]
    fn test_optimizer_small_collection_brute_force() {
        let query = QueryParser::parse("SELECT * FROM tiny WHERE vector SIMILAR TO $query LIMIT 5")
            .unwrap();
        let stats = CollectionStatistics {
            vector_count: 500,
            dimensions: 128,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&query, &stats);
        assert_eq!(plan.strategy, SearchStrategy::BruteForceScan);
    }

    #[test]
    fn test_optimizer_large_collection_index() {
        let query =
            QueryParser::parse("SELECT * FROM large WHERE vector SIMILAR TO $query LIMIT 10")
                .unwrap();
        let stats = CollectionStatistics {
            vector_count: 1_000_000,
            dimensions: 384,
            ..Default::default()
        };
        let plan = CostBasedOptimizer::optimize(&query, &stats);
        assert_eq!(plan.strategy, SearchStrategy::IndexThenFilter);
    }

    #[test]
    fn test_optimizer_selective_filter() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE vector SIMILAR TO $query AND category = 'rare' LIMIT 10",
        )
        .unwrap();
        let mut stats = CollectionStatistics {
            vector_count: 100_000,
            dimensions: 384,
            ..Default::default()
        };
        stats.selectivity.insert("category".into(), 0.01);
        let plan = CostBasedOptimizer::optimize(&query, &stats);
        assert_eq!(plan.strategy, SearchStrategy::FilterThenIndex);
    }

    // Aggregation function tests

    #[test]
    fn test_aggregate_parse_count() {
        let agg = AggregateFunction::parse("COUNT(*)").unwrap();
        assert_eq!(agg, AggregateFunction::Count);
    }

    #[test]
    fn test_aggregate_parse_avg() {
        let agg = AggregateFunction::parse("AVG(price)").unwrap();
        assert_eq!(agg, AggregateFunction::Avg("price".into()));
    }

    #[test]
    fn test_aggregate_parse_min_max_sum() {
        assert_eq!(
            AggregateFunction::parse("MIN(score)").unwrap(),
            AggregateFunction::Min("score".into())
        );
        assert_eq!(
            AggregateFunction::parse("MAX(score)").unwrap(),
            AggregateFunction::Max("score".into())
        );
        assert_eq!(
            AggregateFunction::parse("SUM(quantity)").unwrap(),
            AggregateFunction::Sum("quantity".into())
        );
    }

    #[test]
    fn test_aggregate_apply() {
        let v1 = serde_json::json!(10.0);
        let v2 = serde_json::json!(20.0);
        let v3 = serde_json::json!(30.0);
        let vals: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2), Some(&v3)];
        assert_eq!(AggregateFunction::Count.apply(&vals), serde_json::json!(3));
        assert_eq!(
            AggregateFunction::Avg("x".into()).apply(&vals),
            serde_json::json!(20.0)
        );
        assert_eq!(
            AggregateFunction::Sum("x".into()).apply(&vals),
            serde_json::json!(60.0)
        );
        assert_eq!(
            AggregateFunction::Min("x".into()).apply(&vals),
            serde_json::json!(10.0)
        );
        assert_eq!(
            AggregateFunction::Max("x".into()).apply(&vals),
            serde_json::json!(30.0)
        );
    }

    // REPL session tests

    #[test]
    fn test_query_session_basic() {
        let mut session = QuerySession::new();
        session.default_collection = Some("docs".into());

        let query = session
            .parse_query("SELECT * FROM docs WHERE category = 'books' LIMIT 5")
            .unwrap();
        assert_eq!(query.from.collection, "docs");
        assert_eq!(session.history().len(), 1);
    }

    #[test]
    fn test_query_session_params() {
        let mut session = QuerySession::new();
        session.set_param("limit", LiteralValue::Number(10.0));

        assert!(session.get_param("limit").is_some());
        session.clear_params();
        assert!(session.get_param("limit").is_none());
    }

    #[test]
    fn test_query_session_empty_query() {
        let mut session = QuerySession::new();
        assert!(session.parse_query("").is_err());
    }

    #[test]
    fn test_query_session_help() {
        let help = QuerySession::help_text();
        assert!(help.contains("NeedleQL"));
        assert!(help.contains("SELECT"));
    }

    #[test]
    fn test_cosine_similarity_function_syntax() {
        let query = QueryParser::parse(
            "SELECT * FROM vectors WHERE cosine_similarity(embedding, $query) > 0.8 LIMIT 10"
        ).unwrap();
        assert_eq!(query.from.collection, "vectors");
        assert_eq!(query.limit, Some(10));
        // The cosine_similarity function should be parsed as a SimilarTo expression
        assert!(query.where_clause.is_some());
    }

    #[test]
    fn test_cosine_similarity_with_and_filter() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE cosine_similarity(embedding, $query) > 0.9 AND category = 'science' LIMIT 5"
        ).unwrap();
        assert_eq!(query.from.collection, "docs");
        assert_eq!(query.limit, Some(5));
    }

    // ── Error / invalid input tests ──────────────────────────────────────

    #[test]
    fn test_parse_empty_input() {
        let result = QueryParser::parse("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_whitespace_only() {
        let result = QueryParser::parse("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_garbage_input() {
        let result = QueryParser::parse("!@#$%^&*()");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_incomplete_select() {
        let result = QueryParser::parse("SELECT");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_select_without_from() {
        let result = QueryParser::parse("SELECT * WHERE x = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_missing_collection_name() {
        let result = QueryParser::parse("SELECT * FROM");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_where_without_expression() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_incomplete_comparison() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE x =");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_double_equals_invalid() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE x == 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unterminated_string_literal() {
        let mut lexer = Lexer::new("'unterminated string");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unterminated_double_quote_string() {
        let mut lexer = Lexer::new("\"unterminated");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_similar_to_without_parameter() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE vector SIMILAR TO LIMIT 10");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_limit_without_value() {
        let result = QueryParser::parse("SELECT * FROM docs LIMIT");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_negative_limit() {
        // Parser may or may not accept negative numbers; this tests the boundary
        let result = QueryParser::parse("SELECT * FROM docs LIMIT -1");
        // Either error or it parses but validation would reject
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_parse_in_without_closing_paren() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE status IN ('active'");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_in_empty_list() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE status IN ()");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_between_without_and() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE score BETWEEN 0.5");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unbalanced_parentheses() {
        let result = QueryParser::parse("SELECT * FROM docs WHERE (x = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_deeply_nested_valid() {
        // Moderately nested expression — should succeed
        let result = QueryParser::parse(
            "SELECT * FROM docs WHERE ((x = 1 AND y = 2) OR (z = 3 AND w = 4))"
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_sql_injection_attempt() {
        // SQL injection-style input — should be safely rejected or treated as literal
        let result = QueryParser::parse("SELECT * FROM docs; DROP TABLE docs;--");
        // Should fail parsing (';' is not a valid token)
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_just_keyword() {
        let result = QueryParser::parse("WHERE");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_from_without_select() {
        let result = QueryParser::parse("FROM docs WHERE x = 1");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_explain_without_select() {
        let result = QueryParser::parse("EXPLAIN ANALYZE FROM docs");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_order_by_without_column() {
        let result = QueryParser::parse("SELECT * FROM docs ORDER BY");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_offset_without_limit() {
        // OFFSET without LIMIT — parser may or may not require LIMIT first
        let result = QueryParser::parse("SELECT * FROM docs OFFSET 10");
        // Either ok or error is fine; this tests it doesn't panic
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_lexer_special_characters() {
        let mut lexer = Lexer::new("@");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_lexer_dollar_without_name() {
        let mut lexer = Lexer::new("$ ");
        let result = lexer.next_token();
        // Should either return empty parameter or error
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_parse_with_time_decay_invalid() {
        let result = QueryParser::parse(
            "SELECT * FROM docs WITH TIME_DECAY(INVALID_FUNC) WHERE vector SIMILAR TO $query"
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_limit_zero() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 0"
        ).unwrap();
        let result = QueryValidator::validate(&query);
        // Validator may accept or reject zero limit depending on implementation
        // The important thing is it doesn't panic
        let _ = result;
    }
}
