use std::collections::HashMap;

use serde_json::Value;

use super::ast::*;
use super::lexer::{Lexer, Token};
use super::{QueryError, QueryResult};

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
            if !matches!(
                name.to_uppercase().as_str(),
                "WHERE" | "WITH" | "USING" | "LIMIT" | "ORDER" | "OFFSET"
            ) {
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
                        return Err(QueryError::InvalidIdentifier { name: name.clone() });
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
    #[allow(clippy::too_many_lines)]
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

        // Check for cosine_similarity(column, $param) > threshold
        // Translates to SimilarTo expression for the search pipeline
        if column.eq_ignore_ascii_case("cosine_similarity") && self.current == Token::LParen {
            self.advance()?; // consume (
            let sim_column = if let Token::Identifier(name) = &self.current {
                name.clone()
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "column name".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            };
            self.advance()?;
            self.expect(Token::Comma)?;
            let query_param = if let Token::Parameter(name) = &self.current {
                name.clone()
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "query parameter ($param)".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            };
            self.advance()?;
            self.expect(Token::RParen)?;
            // Expect comparison operator (> or >=) and threshold, but map to SimilarTo
            // The threshold is informational; actual search uses k-NN
            if self.current == Token::Gt || self.current == Token::Ge {
                self.advance()?;
                // Consume the threshold value
                let _threshold = self.parse_literal_value()?;
            }
            return Ok(Expression::SimilarTo(SimilarToExpr {
                column: sim_column,
                query_param,
            }));
        }

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
            Token::Duration(val, unit) => LiteralValue::Number(unit.to_seconds(*val) as f64),
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

#[cfg(test)]
mod tests {
    use super::*;

    // Tests needed: see docs/TODO-test-coverage.md
}
