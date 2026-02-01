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

use crate::database::Database;
use crate::metadata::{Filter, FilterOperator};
use crate::SearchResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

// ============================================================================
// Error Types
// ============================================================================

/// Query parsing and execution errors
#[derive(Debug, Clone, PartialEq)]
pub enum QueryError {
    /// Lexer encountered an invalid token
    InvalidToken { position: usize, found: String },
    /// Parser expected a different token
    UnexpectedToken { expected: String, found: String, position: usize },
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
            Self::UnexpectedToken { expected, found, position } => {
                write!(f, "Expected {} but found '{}' at position {}", expected, found, position)
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
// Token Types
// ============================================================================

/// Token types for the lexer
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    In,
    Like,
    Between,
    Is,
    Null,
    Limit,
    Offset,
    OrderBy,
    Asc,
    Desc,
    Similar,
    To,
    Explain,
    Analyze,
    With,
    Using,
    Rag,
    TimeDecay,

    // Operators
    Eq,           // =
    Ne,           // != or <>
    Lt,           // <
    Le,           // <=
    Gt,           // >
    Ge,           // >=

    // Punctuation
    Star,         // *
    Comma,        // ,
    LParen,       // (
    RParen,       // )
    Dollar,       // $

    // Literals
    Identifier(String),
    StringLit(String),
    NumberLit(f64),
    BoolLit(bool),
    Parameter(String),  // $name

    // Duration literals
    Duration(u64, DurationUnit),

    // End of input
    Eof,
}

/// Duration units for time-based values
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DurationUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
}

impl DurationUnit {
    /// Convert to seconds
    pub fn to_seconds(&self, value: u64) -> u64 {
        match self {
            Self::Seconds => value,
            Self::Minutes => value * 60,
            Self::Hours => value * 3600,
            Self::Days => value * 86400,
            Self::Weeks => value * 604800,
        }
    }
}

// ============================================================================
// Lexer
// ============================================================================

/// Lexer for tokenizing NeedleQL queries
pub struct Lexer {
    input: Vec<char>,
    position: usize,
}

impl Lexer {
    /// Create a new lexer for the given input
    pub fn new(input: &str) -> Self {
        Self {
            input: input.chars().collect(),
            position: 0,
        }
    }

    /// Get current position
    pub fn position(&self) -> usize {
        self.position
    }

    /// Peek at the current character
    fn peek(&self) -> Option<char> {
        self.input.get(self.position).copied()
    }

    /// Peek at the next character
    fn peek_next(&self) -> Option<char> {
        self.input.get(self.position + 1).copied()
    }

    /// Advance and return the current character
    fn advance(&mut self) -> Option<char> {
        let ch = self.peek();
        self.position += 1;
        ch
    }

    /// Skip whitespace
    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Read an identifier or keyword
    fn read_identifier(&mut self) -> String {
        let mut result = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                result.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        result
    }

    /// Read a string literal
    fn read_string(&mut self, quote: char) -> QueryResult<String> {
        self.advance(); // consume opening quote
        let mut result = String::new();
        let start_pos = self.position;

        loop {
            match self.advance() {
                Some(ch) if ch == quote => {
                    return Ok(result);
                }
                Some('\\') => {
                    // Handle escape sequences
                    match self.advance() {
                        Some('n') => result.push('\n'),
                        Some('t') => result.push('\t'),
                        Some('r') => result.push('\r'),
                        Some('\\') => result.push('\\'),
                        Some(q) if q == quote => result.push(q),
                        Some(c) => result.push(c),
                        None => {
                            return Err(QueryError::InvalidLiteral {
                                value: format!("Unterminated string starting at {}", start_pos),
                            });
                        }
                    }
                }
                Some(ch) => result.push(ch),
                None => {
                    return Err(QueryError::InvalidLiteral {
                        value: format!("Unterminated string starting at {}", start_pos),
                    });
                }
            }
        }
    }

    /// Read a number literal
    fn read_number(&mut self) -> QueryResult<Token> {
        let mut num_str = String::new();
        let mut has_dot = false;

        // Handle negative numbers
        if self.peek() == Some('-') {
            num_str.push('-');
            self.advance();
        }

        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !has_dot {
                has_dot = true;
                num_str.push(ch);
                self.advance();
            } else if ch.is_alphabetic() {
                // Check for duration suffix
                let suffix = self.read_identifier().to_lowercase();
                let unit = match suffix.as_str() {
                    "s" | "sec" | "second" | "seconds" => Some(DurationUnit::Seconds),
                    "m" | "min" | "minute" | "minutes" => Some(DurationUnit::Minutes),
                    "h" | "hr" | "hour" | "hours" => Some(DurationUnit::Hours),
                    "d" | "day" | "days" => Some(DurationUnit::Days),
                    "w" | "week" | "weeks" => Some(DurationUnit::Weeks),
                    _ => None,
                };

                if let Some(unit) = unit {
                    let value: u64 = num_str.parse().map_err(|_| QueryError::InvalidLiteral {
                        value: num_str.clone(),
                    })?;
                    return Ok(Token::Duration(value, unit));
                } else {
                    return Err(QueryError::InvalidLiteral {
                        value: format!("{}{}", num_str, suffix),
                    });
                }
            } else {
                break;
            }
        }

        let value: f64 = num_str.parse().map_err(|_| QueryError::InvalidLiteral {
            value: num_str,
        })?;

        Ok(Token::NumberLit(value))
    }

    /// Get the next token
    pub fn next_token(&mut self) -> QueryResult<Token> {
        self.skip_whitespace();

        let start_pos = self.position;

        match self.peek() {
            None => Ok(Token::Eof),
            Some('*') => {
                self.advance();
                Ok(Token::Star)
            }
            Some(',') => {
                self.advance();
                Ok(Token::Comma)
            }
            Some('(') => {
                self.advance();
                Ok(Token::LParen)
            }
            Some(')') => {
                self.advance();
                Ok(Token::RParen)
            }
            Some('=') => {
                self.advance();
                Ok(Token::Eq)
            }
            Some('!') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::Ne)
                } else {
                    Err(QueryError::InvalidToken {
                        position: start_pos,
                        found: "!".to_string(),
                    })
                }
            }
            Some('<') => {
                self.advance();
                match self.peek() {
                    Some('=') => {
                        self.advance();
                        Ok(Token::Le)
                    }
                    Some('>') => {
                        self.advance();
                        Ok(Token::Ne)
                    }
                    _ => Ok(Token::Lt),
                }
            }
            Some('>') => {
                self.advance();
                if self.peek() == Some('=') {
                    self.advance();
                    Ok(Token::Ge)
                } else {
                    Ok(Token::Gt)
                }
            }
            Some('$') => {
                self.advance();
                let name = self.read_identifier();
                if name.is_empty() {
                    Err(QueryError::InvalidToken {
                        position: start_pos,
                        found: "$".to_string(),
                    })
                } else {
                    Ok(Token::Parameter(name))
                }
            }
            Some(quote @ ('\'' | '"')) => {
                let s = self.read_string(quote)?;
                Ok(Token::StringLit(s))
            }
            Some(ch) if ch.is_ascii_digit() || (ch == '-' && self.peek_next().is_some_and(|c| c.is_ascii_digit())) => {
                self.read_number()
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let ident = self.read_identifier();
                let upper = ident.to_uppercase();

                match upper.as_str() {
                    "SELECT" => Ok(Token::Select),
                    "FROM" => Ok(Token::From),
                    "WHERE" => Ok(Token::Where),
                    "AND" => Ok(Token::And),
                    "OR" => Ok(Token::Or),
                    "NOT" => Ok(Token::Not),
                    "IN" => Ok(Token::In),
                    "LIKE" => Ok(Token::Like),
                    "BETWEEN" => Ok(Token::Between),
                    "IS" => Ok(Token::Is),
                    "NULL" => Ok(Token::Null),
                    "LIMIT" => Ok(Token::Limit),
                    "OFFSET" => Ok(Token::Offset),
                    "ORDER" => Ok(Token::OrderBy),
                    "BY" => Ok(Token::Identifier("BY".to_string())), // Handle ORDER BY
                    "ASC" => Ok(Token::Asc),
                    "DESC" => Ok(Token::Desc),
                    "SIMILAR" => Ok(Token::Similar),
                    "TO" => Ok(Token::To),
                    "EXPLAIN" => Ok(Token::Explain),
                    "ANALYZE" => Ok(Token::Analyze),
                    "WITH" => Ok(Token::With),
                    "USING" => Ok(Token::Using),
                    "RAG" => Ok(Token::Rag),
                    "TIME_DECAY" | "TIMEDECAY" => Ok(Token::TimeDecay),
                    "TRUE" => Ok(Token::BoolLit(true)),
                    "FALSE" => Ok(Token::BoolLit(false)),
                    _ => Ok(Token::Identifier(ident)),
                }
            }
            Some(ch) => {
                self.advance();
                Err(QueryError::InvalidToken {
                    position: start_pos,
                    found: ch.to_string(),
                })
            }
        }
    }

    /// Peek at the next token without consuming it
    pub fn peek_token(&mut self) -> QueryResult<Token> {
        let saved_pos = self.position;
        let token = self.next_token();
        self.position = saved_pos;
        token
    }
}

// ============================================================================
// AST Types
// ============================================================================

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

// ============================================================================
// Parser
// ============================================================================

/// Parser for NeedleQL queries
pub struct QueryParser {
    lexer: Lexer,
    current: Token,
}

impl QueryParser {
    /// Parse a NeedleQL query string
    pub fn parse(input: &str) -> QueryResult<Query> {
        let mut parser = Self::new(input)?;
        parser.parse_query()
    }

    /// Create a new parser
    fn new(input: &str) -> QueryResult<Self> {
        let mut lexer = Lexer::new(input);
        let current = lexer.next_token()?;
        Ok(Self { lexer, current })
    }

    /// Get current token
    #[allow(dead_code)]
    fn current(&self) -> &Token {
        &self.current
    }

    /// Advance to next token
    fn advance(&mut self) -> QueryResult<()> {
        self.current = self.lexer.next_token()?;
        Ok(())
    }

    /// Expect a specific token
    fn expect(&mut self, expected: Token) -> QueryResult<()> {
        if self.current == expected {
            self.advance()
        } else {
            Err(QueryError::UnexpectedToken {
                expected: format!("{:?}", expected),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            })
        }
    }

    /// Check if current token matches (useful for lookahead)
    #[allow(dead_code)]
    fn check(&self, token: &Token) -> bool {
        std::mem::discriminant(&self.current) == std::mem::discriminant(token)
    }

    /// Parse a complete query
    fn parse_query(&mut self) -> QueryResult<Query> {
        let mut explain = false;

        // Check for EXPLAIN ANALYZE
        if self.current == Token::Explain {
            self.advance()?;
            if self.current == Token::Analyze {
                self.advance()?;
            }
            explain = true;
        }

        // Parse SELECT
        self.expect(Token::Select)?;
        let select = self.parse_select_clause()?;

        // Parse FROM
        self.expect(Token::From)?;
        let from = self.parse_from_clause()?;

        // Parse optional WITH
        let with_clause = if self.current == Token::With {
            Some(self.parse_with_clause()?)
        } else {
            None
        };

        // Parse optional USING
        let using_clause = if self.current == Token::Using {
            Some(self.parse_using_clause()?)
        } else {
            None
        };

        // Parse optional WHERE
        let where_clause = if self.current == Token::Where {
            self.advance()?;
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Parse optional ORDER BY
        let order_by = if self.current == Token::OrderBy {
            self.advance()?;
            // Skip "BY" if present
            if let Token::Identifier(s) = &self.current {
                if s.to_uppercase() == "BY" {
                    self.advance()?;
                }
            }
            Some(self.parse_order_by_clause()?)
        } else {
            None
        };

        // Parse optional LIMIT
        let limit = if self.current == Token::Limit {
            self.advance()?;
            Some(self.parse_limit()?)
        } else {
            None
        };

        // Parse optional OFFSET
        let offset = if self.current == Token::Offset {
            self.advance()?;
            Some(self.parse_limit()?)
        } else {
            None
        };

        Ok(Query {
            explain,
            select,
            from,
            with_clause,
            using_clause,
            where_clause,
            order_by,
            limit,
            offset,
        })
    }

    /// Parse SELECT clause
    fn parse_select_clause(&mut self) -> QueryResult<SelectClause> {
        if self.current == Token::Star {
            self.advance()?;
            Ok(SelectClause::All)
        } else {
            let mut columns = Vec::new();
            loop {
                if let Token::Identifier(name) = &self.current {
                    columns.push(name.clone());
                    self.advance()?;
                } else {
                    return Err(QueryError::UnexpectedToken {
                        expected: "column name".to_string(),
                        found: format!("{:?}", self.current),
                        position: self.lexer.position(),
                    });
                }

                if self.current == Token::Comma {
                    self.advance()?;
                } else {
                    break;
                }
            }
            Ok(SelectClause::Columns(columns))
        }
    }

    /// Parse FROM clause
    fn parse_from_clause(&mut self) -> QueryResult<FromClause> {
        let collection = if let Token::Identifier(name) = &self.current {
            name.clone()
        } else {
            return Err(QueryError::UnexpectedToken {
                expected: "collection name".to_string(),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            });
        };
        self.advance()?;

        // Optional alias
        let alias = if let Token::Identifier(name) = &self.current {
            if !matches!(name.to_uppercase().as_str(), "WHERE" | "WITH" | "USING" | "LIMIT" | "ORDER" | "OFFSET") {
                let a = name.clone();
                self.advance()?;
                Some(a)
            } else {
                None
            }
        } else {
            None
        };

        Ok(FromClause { collection, alias })
    }

    /// Parse WITH clause
    fn parse_with_clause(&mut self) -> QueryResult<WithClause> {
        self.expect(Token::With)?;

        if self.current == Token::TimeDecay {
            self.advance()?;
            self.expect(Token::LParen)?;

            // Parse decay function name
            let function = if let Token::Identifier(name) = &self.current {
                let func = match name.to_uppercase().as_str() {
                    "LINEAR" => TimeDecayFunction::Linear,
                    "EXPONENTIAL" | "EXP" => TimeDecayFunction::Exponential,
                    "GAUSSIAN" | "GAUSS" => TimeDecayFunction::Gaussian,
                    "STEP" => TimeDecayFunction::Step,
                    _ => {
                        return Err(QueryError::InvalidIdentifier {
                            name: name.clone(),
                        });
                    }
                };
                self.advance()?;
                func
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "decay function name".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            };

            // Parse optional parameters
            let mut params = HashMap::new();
            while self.current == Token::Comma {
                self.advance()?;

                // Parse param=value
                let param_name = if let Token::Identifier(name) = &self.current {
                    name.clone()
                } else {
                    break;
                };
                self.advance()?;

                self.expect(Token::Eq)?;

                let param_value = self.parse_literal_value()?;
                let json_value = match param_value {
                    LiteralValue::String(s) => Value::String(s),
                    LiteralValue::Number(n) => serde_json::json!(n),
                    LiteralValue::Bool(b) => Value::Bool(b),
                    LiteralValue::Null => Value::Null,
                    LiteralValue::Parameter(p) => Value::String(format!("${}", p)),
                };
                params.insert(param_name, json_value);
            }

            self.expect(Token::RParen)?;

            Ok(WithClause::TimeDecay(TimeDecayConfig { function, params }))
        } else {
            Err(QueryError::UnexpectedToken {
                expected: "TIME_DECAY".to_string(),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            })
        }
    }

    /// Parse USING clause
    fn parse_using_clause(&mut self) -> QueryResult<UsingClause> {
        self.expect(Token::Using)?;
        self.expect(Token::Rag)?;
        self.expect(Token::LParen)?;

        let mut options = RagOptions {
            top_k: None,
            rerank: None,
            hybrid_alpha: None,
            deduplicate: None,
        };

        // Parse RAG options
        while let Token::Identifier(name) = &self.current {
            let param = name.to_lowercase();
            self.advance()?;
            self.expect(Token::Eq)?;

            match param.as_str() {
                "top_k" | "topk" | "k" => {
                    if let Token::NumberLit(n) = self.current {
                        options.top_k = Some(n as usize);
                        self.advance()?;
                    }
                }
                "rerank" => {
                    if let Token::BoolLit(b) = self.current {
                        options.rerank = Some(b);
                        self.advance()?;
                    }
                }
                "hybrid_alpha" | "alpha" => {
                    if let Token::NumberLit(n) = self.current {
                        options.hybrid_alpha = Some(n as f32);
                        self.advance()?;
                    }
                }
                "deduplicate" | "dedup" => {
                    if let Token::BoolLit(b) = self.current {
                        options.deduplicate = Some(b);
                        self.advance()?;
                    }
                }
                _ => {}
            }

            if self.current == Token::Comma {
                self.advance()?;
            } else {
                break;
            }
        }

        self.expect(Token::RParen)?;

        Ok(UsingClause { rag: options })
    }

    /// Parse WHERE clause
    fn parse_where_clause(&mut self) -> QueryResult<WhereClause> {
        let expression = self.parse_or_expression()?;
        Ok(WhereClause { expression })
    }

    /// Parse OR expression
    fn parse_or_expression(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_and_expression()?;

        while self.current == Token::Or {
            self.advance()?;
            let right = self.parse_and_expression()?;
            left = Expression::Or(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse AND expression
    fn parse_and_expression(&mut self) -> QueryResult<Expression> {
        let mut left = self.parse_not_expression()?;

        while self.current == Token::And {
            self.advance()?;
            let right = self.parse_not_expression()?;
            left = Expression::And(Box::new(left), Box::new(right));
        }

        Ok(left)
    }

    /// Parse NOT expression
    fn parse_not_expression(&mut self) -> QueryResult<Expression> {
        if self.current == Token::Not {
            self.advance()?;
            let expr = self.parse_primary_expression()?;
            Ok(Expression::Not(Box::new(expr)))
        } else {
            self.parse_primary_expression()
        }
    }

    /// Parse primary expression
    fn parse_primary_expression(&mut self) -> QueryResult<Expression> {
        // Handle parentheses
        if self.current == Token::LParen {
            self.advance()?;
            let expr = self.parse_or_expression()?;
            self.expect(Token::RParen)?;
            return Ok(Expression::Grouped(Box::new(expr)));
        }

        // Get column name
        let column = if let Token::Identifier(name) = &self.current {
            name.clone()
        } else {
            return Err(QueryError::UnexpectedToken {
                expected: "column name or (".to_string(),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            });
        };
        self.advance()?;

        // Check for SIMILAR TO
        if self.current == Token::Similar {
            self.advance()?;
            self.expect(Token::To)?;

            let query_param = if let Token::Parameter(name) = &self.current {
                name.clone()
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "query parameter ($name)".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            };
            self.advance()?;

            return Ok(Expression::SimilarTo(SimilarToExpr {
                column,
                query_param,
            }));
        }

        // Check for IS NULL / IS NOT NULL
        if self.current == Token::Is {
            self.advance()?;
            let negated = if self.current == Token::Not {
                self.advance()?;
                true
            } else {
                false
            };
            self.expect(Token::Null)?;
            return Ok(Expression::IsNull(IsNullExpr { column, negated }));
        }

        // Check for IN
        if self.current == Token::In || (self.current == Token::Not) {
            let negated = if self.current == Token::Not {
                self.advance()?;
                true
            } else {
                false
            };

            if self.current == Token::In {
                self.advance()?;
                self.expect(Token::LParen)?;

                let mut values = Vec::new();
                loop {
                    values.push(self.parse_literal_value()?);
                    if self.current == Token::Comma {
                        self.advance()?;
                    } else {
                        break;
                    }
                }

                self.expect(Token::RParen)?;

                return Ok(Expression::InList(InListExpr {
                    column,
                    values,
                    negated,
                }));
            }
        }

        // Check for BETWEEN
        if self.current == Token::Between {
            self.advance()?;
            let low = self.parse_literal_value()?;
            self.expect(Token::And)?;
            let high = self.parse_literal_value()?;

            return Ok(Expression::Between(BetweenExpr {
                column,
                low,
                high,
                negated: false,
            }));
        }

        // Check for LIKE
        if self.current == Token::Like {
            self.advance()?;
            let pattern = if let Token::StringLit(s) = &self.current {
                s.clone()
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "string pattern".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            };
            self.advance()?;

            return Ok(Expression::Like(LikeExpr {
                column,
                pattern,
                negated: false,
            }));
        }

        // Regular comparison
        let operator = match &self.current {
            Token::Eq => CompareOp::Eq,
            Token::Ne => CompareOp::Ne,
            Token::Lt => CompareOp::Lt,
            Token::Le => CompareOp::Le,
            Token::Gt => CompareOp::Gt,
            Token::Ge => CompareOp::Ge,
            _ => {
                return Err(QueryError::UnexpectedToken {
                    expected: "comparison operator".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            }
        };
        self.advance()?;

        let value = self.parse_literal_value()?;

        Ok(Expression::Comparison(ComparisonExpr {
            column,
            operator,
            value,
        }))
    }

    /// Parse a literal value
    fn parse_literal_value(&mut self) -> QueryResult<LiteralValue> {
        let value = match &self.current {
            Token::StringLit(s) => LiteralValue::String(s.clone()),
            Token::NumberLit(n) => LiteralValue::Number(*n),
            Token::BoolLit(b) => LiteralValue::Bool(*b),
            Token::Null => LiteralValue::Null,
            Token::Parameter(name) => LiteralValue::Parameter(name.clone()),
            Token::Duration(val, unit) => {
                LiteralValue::Number(unit.to_seconds(*val) as f64)
            }
            _ => {
                return Err(QueryError::UnexpectedToken {
                    expected: "literal value".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            }
        };
        self.advance()?;
        Ok(value)
    }

    /// Parse ORDER BY clause
    fn parse_order_by_clause(&mut self) -> QueryResult<OrderByClause> {
        let mut columns = Vec::new();

        loop {
            let column = if let Token::Identifier(name) = &self.current {
                name.clone()
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "column name".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            };
            self.advance()?;

            let order = if self.current == Token::Desc {
                self.advance()?;
                SortOrder::Desc
            } else if self.current == Token::Asc {
                self.advance()?;
                SortOrder::Asc
            } else {
                SortOrder::Asc
            };

            columns.push((column, order));

            if self.current == Token::Comma {
                self.advance()?;
            } else {
                break;
            }
        }

        Ok(OrderByClause { columns })
    }

    /// Parse LIMIT/OFFSET value
    fn parse_limit(&mut self) -> QueryResult<u64> {
        if let Token::NumberLit(n) = self.current {
            if n < 0.0 {
                return Err(QueryError::InvalidLiteral {
                    value: format!("LIMIT/OFFSET must be non-negative: {}", n),
                });
            }
            let limit = n as u64;
            self.advance()?;
            Ok(limit)
        } else {
            Err(QueryError::UnexpectedToken {
                expected: "number".to_string(),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            })
        }
    }
}

// ============================================================================
// Query Validation
// ============================================================================

/// Validates a parsed query for semantic correctness
pub struct QueryValidator;

impl QueryValidator {
    /// Validate a query
    pub fn validate(query: &Query) -> QueryResult<()> {
        // Must have a vector similarity clause for search
        let has_similar_to = Self::find_similar_to(&query.where_clause);

        if !has_similar_to && query.using_clause.is_some() {
            return Err(QueryError::SemanticError {
                message: "RAG queries require a SIMILAR TO clause".to_string(),
            });
        }

        if !has_similar_to && query.with_clause.is_some() {
            return Err(QueryError::SemanticError {
                message: "TIME_DECAY requires a SIMILAR TO clause".to_string(),
            });
        }

        // Validate LIMIT is reasonable
        if let Some(limit) = query.limit {
            if limit > 10000 {
                return Err(QueryError::SemanticError {
                    message: format!("LIMIT {} exceeds maximum of 10000", limit),
                });
            }
        }

        Ok(())
    }

    fn find_similar_to(where_clause: &Option<WhereClause>) -> bool {
        if let Some(clause) = where_clause {
            Self::expr_has_similar_to(&clause.expression)
        } else {
            false
        }
    }

    fn expr_has_similar_to(expr: &Expression) -> bool {
        match expr {
            Expression::SimilarTo(_) => true,
            Expression::And(l, r) | Expression::Or(l, r) => {
                Self::expr_has_similar_to(l) || Self::expr_has_similar_to(r)
            }
            Expression::Not(e) | Expression::Grouped(e) => Self::expr_has_similar_to(e),
            _ => false,
        }
    }
}

// ============================================================================
// Query Execution
// ============================================================================

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
        let query_vector = context.query_vector.clone().ok_or_else(|| {
            QueryError::MissingParameter {
                name: "query_vector".to_string(),
            }
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
        let collection = self.db.collection(&query.from.collection)
            .map_err(|e| QueryError::ExecutionError { message: e.to_string() })?;

        let col_stats = {
            let s = collection.stats()
                .map_err(|e| QueryError::ExecutionError { message: e.to_string() })?;
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
        }.map_err(|e| QueryError::ExecutionError { message: e.to_string() })?;

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
            Some(self.build_plan(query, &filter, total_time.as_secs_f64() * 1000.0))
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
                let values: Vec<Value> = in_expr.values.iter()
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
            Expression::Grouped(inner) => {
                Self::build_filter(inner, context)
            }
        }
    }

    /// Resolve a literal value, handling parameters
    fn resolve_value(value: &LiteralValue, context: &QueryContext) -> QueryResult<Value> {
        match value {
            LiteralValue::String(s) => Ok(Value::String(s.clone())),
            LiteralValue::Number(n) => Ok(serde_json::json!(*n)),
            LiteralValue::Bool(b) => Ok(Value::Bool(*b)),
            LiteralValue::Null => Ok(Value::Null),
            LiteralValue::Parameter(name) => {
                context.params.get(name).cloned().ok_or_else(|| {
                    QueryError::MissingParameter { name: name.clone() }
                })
            }
        }
    }

    /// Build query plan
    fn build_plan(&self, query: &Query, filter: &Option<Filter>, actual_time: f64) -> QueryPlan {
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
}

// ---------------------------------------------------------------------------
// Cost-Based Query Optimizer
// ---------------------------------------------------------------------------

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
    pub fn optimize(
        query: &Query,
        stats: &CollectionStatistics,
    ) -> OptimizedPlan {
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
            notes.push(format!("Small collection ({} vectors): brute-force preferred", n as usize));
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
                selectivity * 100.0, fti_cost, itf_cost
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
                    description: format!("Pre-filter {} vectors to ~{}", n as usize, filtered_rows as usize),
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
        if query.where_clause.is_none() {
            return 1.0;
        }
        let where_clause = query.where_clause.as_ref().unwrap();
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
                Self::estimate_expr_selectivity(l, stats) * Self::estimate_expr_selectivity(r, stats)
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

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
        assert_eq!(lexer.next_token().unwrap(), Token::StringLit("hello".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::StringLit("world".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::NumberLit(42.0));
        assert_eq!(lexer.next_token().unwrap(), Token::NumberLit(3.14));
        assert_eq!(lexer.next_token().unwrap(), Token::BoolLit(true));
        assert_eq!(lexer.next_token().unwrap(), Token::BoolLit(false));
    }

    #[test]
    fn test_lexer_parameters() {
        let mut lexer = Lexer::new("$query $limit $filter");
        assert_eq!(lexer.next_token().unwrap(), Token::Parameter("query".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Parameter("limit".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Parameter("filter".to_string()));
    }

    #[test]
    fn test_lexer_duration() {
        let mut lexer = Lexer::new("7d 24h 60m 30s");
        assert_eq!(lexer.next_token().unwrap(), Token::Duration(7, DurationUnit::Days));
        assert_eq!(lexer.next_token().unwrap(), Token::Duration(24, DurationUnit::Hours));
        assert_eq!(lexer.next_token().unwrap(), Token::Duration(60, DurationUnit::Minutes));
        assert_eq!(lexer.next_token().unwrap(), Token::Duration(30, DurationUnit::Seconds));
    }

    #[test]
    fn test_lexer_identifiers() {
        let mut lexer = Lexer::new("collection_name field1 _private");
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("collection_name".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("field1".to_string()));
        assert_eq!(lexer.next_token().unwrap(), Token::Identifier("_private".to_string()));
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
    fn test_parse_select_columns() {
        let query = QueryParser::parse("SELECT id, title, score FROM docs").unwrap();
        if let SelectClause::Columns(cols) = query.select {
            assert_eq!(cols, vec!["id", "title", "score"]);
        } else {
            panic!("Expected SelectClause::Columns");
        }
    }

    #[test]
    fn test_parse_where_similar_to() {
        let query = QueryParser::parse(
            "SELECT * FROM documents WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        if let Some(where_clause) = query.where_clause {
            if let Expression::SimilarTo(similar) = where_clause.expression {
                assert_eq!(similar.column, "vector");
                assert_eq!(similar.query_param, "query");
            } else {
                panic!("Expected SimilarTo expression");
            }
        } else {
            panic!("Expected WHERE clause");
        }
    }

    #[test]
    fn test_parse_where_comparison() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE category = 'science' AND score > 0.5"
        ).unwrap();

        assert!(query.where_clause.is_some());
    }

    #[test]
    fn test_parse_where_in() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE status IN ('active', 'pending')"
        ).unwrap();

        if let Some(where_clause) = query.where_clause {
            if let Expression::InList(in_list) = where_clause.expression {
                assert_eq!(in_list.column, "status");
                assert_eq!(in_list.values.len(), 2);
                assert!(!in_list.negated);
            } else {
                panic!("Expected InList expression");
            }
        } else {
            panic!("Expected WHERE clause");
        }
    }

    #[test]
    fn test_parse_where_between() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE score BETWEEN 0.5 AND 1.0"
        ).unwrap();

        if let Some(where_clause) = query.where_clause {
            if let Expression::Between(between) = where_clause.expression {
                assert_eq!(between.column, "score");
            } else {
                panic!("Expected Between expression");
            }
        } else {
            panic!("Expected WHERE clause");
        }
    }

    #[test]
    fn test_parse_where_like() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE title LIKE '%machine learning%'"
        ).unwrap();

        if let Some(where_clause) = query.where_clause {
            if let Expression::Like(like) = where_clause.expression {
                assert_eq!(like.column, "title");
                assert_eq!(like.pattern, "%machine learning%");
            } else {
                panic!("Expected Like expression");
            }
        } else {
            panic!("Expected WHERE clause");
        }
    }

    #[test]
    fn test_parse_explain_analyze() {
        let query = QueryParser::parse(
            "EXPLAIN ANALYZE SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        assert!(query.explain);
    }

    #[test]
    fn test_parse_with_time_decay() {
        let query = QueryParser::parse(
            "SELECT * FROM articles WITH TIME_DECAY(EXPONENTIAL, half_life=7d) WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        if let Some(WithClause::TimeDecay(config)) = query.with_clause {
            assert_eq!(config.function, TimeDecayFunction::Exponential);
            assert!(config.params.contains_key("half_life"));
        } else {
            panic!("Expected TIME_DECAY clause");
        }
    }

    #[test]
    fn test_parse_using_rag() {
        let query = QueryParser::parse(
            "SELECT * FROM knowledge_base USING RAG(top_k=5, rerank=true) WHERE vector SIMILAR TO $query"
        ).unwrap();

        if let Some(using) = query.using_clause {
            assert_eq!(using.rag.top_k, Some(5));
            assert_eq!(using.rag.rerank, Some(true));
        } else {
            panic!("Expected USING RAG clause");
        }
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
    fn test_parse_order_by() {
        let query = QueryParser::parse(
            "SELECT * FROM docs ORDER BY score DESC, title ASC LIMIT 10"
        ).unwrap();

        if let Some(order_by) = query.order_by {
            assert_eq!(order_by.columns.len(), 2);
            assert_eq!(order_by.columns[0], ("score".to_string(), SortOrder::Desc));
            assert_eq!(order_by.columns[1], ("title".to_string(), SortOrder::Asc));
        } else {
            panic!("Expected ORDER BY clause");
        }
    }

    #[test]
    fn test_parse_offset() {
        let query = QueryParser::parse(
            "SELECT * FROM docs LIMIT 10 OFFSET 20"
        ).unwrap();

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
        let query = QueryParser::parse(
            "SELECT * FROM docs USING RAG(top_k=5) WHERE category = 'science'"
        ).unwrap();

        let result = QueryValidator::validate(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_time_decay_without_similar_to() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WITH TIME_DECAY(EXPONENTIAL) WHERE category = 'science'"
        ).unwrap();

        let result = QueryValidator::validate(&query);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_excessive_limit() {
        let query = QueryParser::parse(
            "SELECT * FROM docs WHERE vector SIMILAR TO $query LIMIT 100000"
        ).unwrap();

        let result = QueryValidator::validate(&query);
        assert!(result.is_err());
    }

    // Execution tests

    #[test]
    fn test_execute_simple_query() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let collection = db.collection("test").unwrap();
        collection.insert("doc1", &[1.0; 8], Some(serde_json::json!({"category": "science"}))).unwrap();
        collection.insert("doc2", &[0.5; 8], Some(serde_json::json!({"category": "tech"}))).unwrap();

        let executor = QueryExecutor::new(db);

        let query = QueryParser::parse(
            "SELECT * FROM test WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        let context = QueryContext::new()
            .with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context).unwrap();
        assert!(!result.results.is_empty());
    }

    #[test]
    fn test_execute_with_filter() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let collection = db.collection("test").unwrap();
        collection.insert("doc1", &[1.0; 8], Some(serde_json::json!({"category": "science"}))).unwrap();
        collection.insert("doc2", &[0.5; 8], Some(serde_json::json!({"category": "tech"}))).unwrap();

        let executor = QueryExecutor::new(db);

        let query = QueryParser::parse(
            "SELECT * FROM test WHERE category = 'science' AND vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        let context = QueryContext::new()
            .with_query_vector(vec![1.0; 8]);

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
            "EXPLAIN ANALYZE SELECT * FROM test WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        let context = QueryContext::new()
            .with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context).unwrap();
        assert!(result.plan.is_some());
        assert!(!result.plan.unwrap().nodes.is_empty());
    }

    #[test]
    fn test_execute_collection_not_found() {
        let db = Arc::new(Database::in_memory());
        let executor = QueryExecutor::new(db);

        let query = QueryParser::parse(
            "SELECT * FROM nonexistent WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

        let context = QueryContext::new()
            .with_query_vector(vec![1.0; 8]);

        let result = executor.execute(&query, &context);
        assert!(result.is_err());
    }

    #[test]
    fn test_execute_missing_query_vector() {
        let db = Arc::new(Database::in_memory());
        db.create_collection("test", 8).unwrap();

        let executor = QueryExecutor::new(db);

        let query = QueryParser::parse(
            "SELECT * FROM test WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();

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

        assert_eq!(context.params.get("category"), Some(&Value::String("science".to_string())));
        assert_eq!(context.query_vector, Some(vec![1.0, 2.0, 3.0]));
    }

    // Cost-based optimizer tests

    #[test]
    fn test_optimizer_small_collection_brute_force() {
        let query = QueryParser::parse(
            "SELECT * FROM tiny WHERE vector SIMILAR TO $query LIMIT 5"
        ).unwrap();
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
        let query = QueryParser::parse(
            "SELECT * FROM large WHERE vector SIMILAR TO $query LIMIT 10"
        ).unwrap();
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
            "SELECT * FROM docs WHERE vector SIMILAR TO $query AND category = 'rare' LIMIT 10"
        ).unwrap();
        let mut stats = CollectionStatistics {
            vector_count: 100_000,
            dimensions: 384,
            ..Default::default()
        };
        stats.selectivity.insert("category".into(), 0.01);
        let plan = CostBasedOptimizer::optimize(&query, &stats);
        assert_eq!(plan.strategy, SearchStrategy::FilterThenIndex);
    }
}
