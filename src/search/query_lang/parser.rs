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

    /// Parse a complete query (SELECT or SEARCH NEAR form)
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

        // SEARCH NEAR syntax: SEARCH NEAR $query FROM collection [WHERE ...] [LIMIT n]
        if self.current == Token::Search {
            return self.parse_search_near(explain);
        }

        // Standard SELECT syntax
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

        // Parse optional RERANK BY
        let rerank_clause = if self.current == Token::Rerank {
            self.advance()?;
            // Expect "BY"
            if let Token::Identifier(s) = &self.current {
                if s.to_uppercase() == "BY" {
                    self.advance()?;
                }
            }
            Some(self.parse_rerank_clause()?)
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
            rerank_clause,
            order_by,
            limit,
            offset,
        })
    }

    /// Parse SEARCH NEAR syntax:
    /// `SEARCH NEAR $query FROM collection [WHERE ...] [RERANK BY ...] [LIMIT n]`
    fn parse_search_near(&mut self, explain: bool) -> QueryResult<Query> {
        self.expect(Token::Search)?;
        self.expect(Token::Near)?;

        // Parse the query parameter (e.g., $query or [...])
        let query_param = if let Token::Parameter(p) = &self.current {
            let p = p.clone();
            self.advance()?;
            p
        } else if self.current == Token::Dollar {
            self.advance()?;
            if let Token::Identifier(name) = &self.current {
                let p = name.clone();
                self.advance()?;
                p
            } else {
                return Err(QueryError::UnexpectedToken {
                    expected: "parameter name after $".to_string(),
                    found: format!("{:?}", self.current),
                    position: self.lexer.position(),
                });
            }
        } else {
            return Err(QueryError::UnexpectedToken {
                expected: "$parameter".to_string(),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            });
        };

        // FROM collection
        self.expect(Token::From)?;
        let from = self.parse_from_clause()?;

        // Optional WHERE
        let where_clause = if self.current == Token::Where {
            self.advance()?;
            Some(self.parse_where_clause()?)
        } else {
            None
        };

        // Optional RERANK BY
        let rerank_clause = if self.current == Token::Rerank {
            self.advance()?;
            if let Token::Identifier(s) = &self.current {
                if s.to_uppercase() == "BY" {
                    self.advance()?;
                }
            }
            Some(self.parse_rerank_clause()?)
        } else {
            None
        };

        // Optional LIMIT
        let limit = if self.current == Token::Limit {
            self.advance()?;
            Some(self.parse_limit()?)
        } else {
            None
        };

        // Synthesize a WHERE clause with SIMILAR TO if none exists
        let where_clause = match where_clause {
            Some(w) => Some(WhereClause {
                expression: Expression::And(
                    Box::new(Expression::SimilarTo(SimilarToExpr {
                        column: "vector".to_string(),
                        query_param: query_param.clone(),
                    })),
                    Box::new(w.expression),
                ),
            }),
            None => Some(WhereClause {
                expression: Expression::SimilarTo(SimilarToExpr {
                    column: "vector".to_string(),
                    query_param,
                }),
            }),
        };

        Ok(Query {
            explain,
            select: SelectClause::All,
            from,
            with_clause: None,
            using_clause: None,
            where_clause,
            rerank_clause,
            order_by: None,
            limit,
            offset: None,
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

        // Parse RAG options (handle `rerank` keyword token as an identifier)
        loop {
            let param = match &self.current {
                Token::Identifier(name) => name.to_lowercase(),
                Token::Rerank => "rerank".to_string(),
                _ => break,
            };
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

    /// Parse RERANK BY clause.
    ///
    /// Syntax: `RERANK BY <strategy> [FETCH <n>]`
    /// Strategies:
    ///   - `<column> [ASC|DESC]` — re-rank by a metadata field
    ///   - `MMR(<lambda>)` — Maximal Marginal Relevance
    ///   - `RRF(<k>)` — Reciprocal Rank Fusion
    ///   - `CROSSENCODER('<model>')` — Cross-encoder model re-ranking
    fn parse_rerank_clause(&mut self) -> QueryResult<RerankClause> {
        let strategy = if let Token::Identifier(name) = &self.current {
            let upper = name.to_uppercase();
            match upper.as_str() {
                "MMR" => {
                    self.advance()?;
                    let lambda = if self.current == Token::LParen {
                        self.advance()?;
                        let val = if let Token::NumberLit(n) = self.current {
                            n as f32
                        } else {
                            0.5
                        };
                        self.advance()?;
                        if self.current == Token::RParen {
                            self.advance()?;
                        }
                        val
                    } else {
                        0.5
                    };
                    RerankStrategy::Mmr { lambda }
                }
                "RRF" => {
                    self.advance()?;
                    let k = if self.current == Token::LParen {
                        self.advance()?;
                        let val = if let Token::NumberLit(n) = self.current {
                            n as usize
                        } else {
                            60
                        };
                        self.advance()?;
                        if self.current == Token::RParen {
                            self.advance()?;
                        }
                        val
                    } else {
                        60
                    };
                    RerankStrategy::Rrf { k }
                }
                "CROSSENCODER" | "CROSS_ENCODER" => {
                    self.advance()?;
                    let model = if self.current == Token::LParen {
                        self.advance()?;
                        let m = if let Token::StringLit(s) = &self.current {
                            s.clone()
                        } else {
                            "default".to_string()
                        };
                        self.advance()?;
                        if self.current == Token::RParen {
                            self.advance()?;
                        }
                        m
                    } else {
                        "default".to_string()
                    };
                    RerankStrategy::CrossEncoder { model }
                }
                _ => {
                    let column = name.clone();
                    self.advance()?;
                    let order = if self.current == Token::Desc {
                        self.advance()?;
                        SortOrder::Desc
                    } else if self.current == Token::Asc {
                        self.advance()?;
                        SortOrder::Asc
                    } else {
                        SortOrder::Desc
                    };
                    RerankStrategy::Field { column, order }
                }
            }
        } else {
            return Err(QueryError::UnexpectedToken {
                expected: "rerank strategy (field name, MMR, or RRF)".to_string(),
                found: format!("{:?}", self.current),
                position: self.lexer.position(),
            });
        };

        // Optional FETCH <n>
        let fetch_k = if let Token::Identifier(s) = &self.current {
            if s.to_uppercase() == "FETCH" {
                self.advance()?;
                if let Token::NumberLit(n) = self.current {
                    self.advance()?;
                    Some(n as usize)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(RerankClause { strategy, fetch_k })
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

    fn parse(sql: &str) -> Query {
        QueryParser::parse(sql).expect("should parse")
    }

    #[test]
    fn test_parse_rerank_by_field() {
        let q = parse("SELECT * FROM docs WHERE vector SIMILAR TO $query RERANK BY score DESC LIMIT 10");
        assert!(q.rerank_clause.is_some());
        let rerank = q.rerank_clause.as_ref().expect("rerank");
        assert!(matches!(rerank.strategy, RerankStrategy::Field { ref column, order } if column == "score" && order == SortOrder::Desc));
        assert_eq!(q.limit, Some(10));
    }

    #[test]
    fn test_parse_rerank_by_mmr() {
        let q = parse("SELECT * FROM docs RERANK BY MMR(0.7) LIMIT 5");
        assert!(q.rerank_clause.is_some());
        let rerank = q.rerank_clause.as_ref().expect("rerank");
        match &rerank.strategy {
            RerankStrategy::Mmr { lambda } => assert!((lambda - 0.7).abs() < 0.01),
            other => panic!("Expected MMR, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_rerank_by_rrf_with_fetch() {
        let q = parse("SELECT * FROM docs RERANK BY RRF(60) FETCH 50 LIMIT 10");
        assert!(q.rerank_clause.is_some());
        let rerank = q.rerank_clause.as_ref().expect("rerank");
        match &rerank.strategy {
            RerankStrategy::Rrf { k } => assert_eq!(*k, 60),
            other => panic!("Expected RRF, got {:?}", other),
        }
        assert_eq!(rerank.fetch_k, Some(50));
    }

    #[test]
    fn test_parse_no_rerank() {
        let q = parse("SELECT * FROM docs LIMIT 10");
        assert!(q.rerank_clause.is_none());
    }

    #[test]
    fn test_parse_rerank_by_crossencoder() {
        let q = parse("SELECT * FROM docs RERANK BY CrossEncoder('ms-marco-MiniLM') LIMIT 5");
        let rerank = q.rerank_clause.as_ref().expect("rerank");
        match &rerank.strategy {
            RerankStrategy::CrossEncoder { model } => assert_eq!(model, "ms-marco-MiniLM"),
            other => panic!("Expected CrossEncoder, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_search_near() {
        let q = parse("SEARCH NEAR $query FROM documents LIMIT 10");
        assert_eq!(q.from.collection, "documents");
        assert_eq!(q.limit, Some(10));
        assert!(q.where_clause.is_some());
        // Should synthesize a SIMILAR TO expression
        let where_clause = q.where_clause.as_ref().expect("where");
        match &where_clause.expression {
            Expression::SimilarTo(s) => assert_eq!(s.query_param, "query"),
            _ => panic!("Expected SimilarTo expression"),
        }
    }

    #[test]
    fn test_parse_search_near_with_where() {
        let q = parse("SEARCH NEAR $query FROM docs WHERE category = 'science' LIMIT 5");
        assert_eq!(q.from.collection, "docs");
        assert!(q.where_clause.is_some());
        // Should be AND(SimilarTo, Comparison)
        let where_clause = q.where_clause.as_ref().expect("where");
        match &where_clause.expression {
            Expression::And(left, _right) => {
                assert!(matches!(left.as_ref(), Expression::SimilarTo(_)));
            }
            _ => panic!("Expected AND expression"),
        }
    }

    #[test]
    fn test_parse_search_near_with_rerank() {
        let q = parse("SEARCH NEAR $query FROM docs RERANK BY MMR(0.5) LIMIT 10");
        assert!(q.rerank_clause.is_some());
        let rerank = q.rerank_clause.as_ref().expect("rerank");
        assert!(matches!(rerank.strategy, RerankStrategy::Mmr { .. }));
    }
}
