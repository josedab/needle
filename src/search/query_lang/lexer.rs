use super::{QueryError, QueryResult};

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
    Rerank,
    Search,
    Near,

    // Operators
    Eq, // =
    Ne, // != or <>
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=

    // Punctuation
    Star,   // *
    Comma,  // ,
    LParen, // (
    RParen, // )
    Dollar, // $

    // Literals
    Identifier(String),
    StringLit(String),
    NumberLit(f64),
    BoolLit(bool),
    Parameter(String), // $name

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

        let value: f64 = num_str
            .parse()
            .map_err(|_| QueryError::InvalidLiteral { value: num_str })?;

        Ok(Token::NumberLit(value))
    }

    /// Get the next token
    #[allow(clippy::too_many_lines)]
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
            Some(ch)
                if ch.is_ascii_digit()
                    || (ch == '-' && self.peek_next().is_some_and(|c| c.is_ascii_digit())) =>
            {
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
                    "RERANK" => Ok(Token::Rerank),
                    "SEARCH" => Ok(Token::Search),
                    "NEAR" => Ok(Token::Near),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(input: &str) -> Vec<Token> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next_token().unwrap();
            if tok == Token::Eof {
                break;
            }
            tokens.push(tok);
        }
        tokens
    }

    // ====================================================================
    // Keywords
    // ====================================================================

    #[test]
    fn test_all_keywords() {
        let cases = vec![
            ("SELECT", Token::Select),
            ("FROM", Token::From),
            ("WHERE", Token::Where),
            ("AND", Token::And),
            ("OR", Token::Or),
            ("NOT", Token::Not),
            ("IN", Token::In),
            ("LIKE", Token::Like),
            ("BETWEEN", Token::Between),
            ("IS", Token::Is),
            ("NULL", Token::Null),
            ("LIMIT", Token::Limit),
            ("OFFSET", Token::Offset),
            ("ORDER", Token::OrderBy),
            ("ASC", Token::Asc),
            ("DESC", Token::Desc),
            ("SIMILAR", Token::Similar),
            ("TO", Token::To),
            ("EXPLAIN", Token::Explain),
            ("ANALYZE", Token::Analyze),
            ("WITH", Token::With),
            ("USING", Token::Using),
            ("RAG", Token::Rag),
            ("RERANK", Token::Rerank),
            ("SEARCH", Token::Search),
            ("NEAR", Token::Near),
        ];

        for (input, expected) in cases {
            let tokens = lex_all(input);
            assert_eq!(tokens.len(), 1, "Failed for: {input}");
            assert_eq!(tokens[0], expected, "Mismatch for: {input}");
        }
    }

    #[test]
    fn test_keywords_case_insensitive() {
        let tokens = lex_all("select FROM where");
        assert_eq!(tokens, vec![Token::Select, Token::From, Token::Where]);
    }

    #[test]
    fn test_time_decay_variants() {
        assert_eq!(lex_all("TIME_DECAY"), vec![Token::TimeDecay]);
        assert_eq!(lex_all("TIMEDECAY"), vec![Token::TimeDecay]);
    }

    #[test]
    fn test_boolean_literals() {
        assert_eq!(lex_all("TRUE"), vec![Token::BoolLit(true)]);
        assert_eq!(lex_all("FALSE"), vec![Token::BoolLit(false)]);
        assert_eq!(lex_all("true"), vec![Token::BoolLit(true)]);
    }

    // ====================================================================
    // Operators
    // ====================================================================

    #[test]
    fn test_comparison_operators() {
        let tokens = lex_all("= != < <= > >=");
        assert_eq!(
            tokens,
            vec![Token::Eq, Token::Ne, Token::Lt, Token::Le, Token::Gt, Token::Ge]
        );
    }

    #[test]
    fn test_ne_diamond() {
        let tokens = lex_all("<>");
        assert_eq!(tokens, vec![Token::Ne]);
    }

    // ====================================================================
    // Punctuation
    // ====================================================================

    #[test]
    fn test_punctuation() {
        let tokens = lex_all("* , ( )");
        assert_eq!(
            tokens,
            vec![Token::Star, Token::Comma, Token::LParen, Token::RParen]
        );
    }

    // ====================================================================
    // String literals
    // ====================================================================

    #[test]
    fn test_single_quoted_string() {
        let tokens = lex_all("'hello world'");
        assert_eq!(tokens, vec![Token::StringLit("hello world".into())]);
    }

    #[test]
    fn test_double_quoted_string() {
        let tokens = lex_all("\"hello\"");
        assert_eq!(tokens, vec![Token::StringLit("hello".into())]);
    }

    #[test]
    fn test_escaped_quotes_in_strings() {
        let tokens = lex_all(r#"'it\'s here'"#);
        assert_eq!(tokens, vec![Token::StringLit("it's here".into())]);
    }

    #[test]
    fn test_escape_sequences() {
        let tokens = lex_all(r#"'line1\nline2\ttab\\backslash'"#);
        assert_eq!(
            tokens,
            vec![Token::StringLit("line1\nline2\ttab\\backslash".into())]
        );
    }

    #[test]
    fn test_unterminated_string() {
        let mut lexer = Lexer::new("'unterminated");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_string() {
        let tokens = lex_all("''");
        assert_eq!(tokens, vec![Token::StringLit(String::new())]);
    }

    // ====================================================================
    // Number literals
    // ====================================================================

    #[test]
    fn test_integer() {
        let tokens = lex_all("42");
        assert_eq!(tokens, vec![Token::NumberLit(42.0)]);
    }

    #[test]
    fn test_float() {
        let tokens = lex_all("3.14");
        assert_eq!(tokens, vec![Token::NumberLit(3.14)]);
    }

    #[test]
    fn test_negative_number() {
        let tokens = lex_all("-7");
        assert_eq!(tokens, vec![Token::NumberLit(-7.0)]);
    }

    #[test]
    fn test_negative_float() {
        let tokens = lex_all("-3.5");
        assert_eq!(tokens, vec![Token::NumberLit(-3.5)]);
    }

    // ====================================================================
    // Duration literals
    // ====================================================================

    #[test]
    fn test_duration_seconds() {
        let tokens = lex_all("30s");
        assert_eq!(tokens, vec![Token::Duration(30, DurationUnit::Seconds)]);
    }

    #[test]
    fn test_duration_minutes() {
        let tokens = lex_all("5m");
        assert_eq!(tokens, vec![Token::Duration(5, DurationUnit::Minutes)]);
    }

    #[test]
    fn test_duration_hours() {
        let tokens = lex_all("2h");
        assert_eq!(tokens, vec![Token::Duration(2, DurationUnit::Hours)]);
    }

    #[test]
    fn test_duration_days() {
        let tokens = lex_all("7d");
        assert_eq!(tokens, vec![Token::Duration(7, DurationUnit::Days)]);
    }

    #[test]
    fn test_duration_weeks() {
        let tokens = lex_all("1w");
        assert_eq!(tokens, vec![Token::Duration(1, DurationUnit::Weeks)]);
    }

    #[test]
    fn test_duration_long_forms() {
        assert_eq!(lex_all("10sec"), vec![Token::Duration(10, DurationUnit::Seconds)]);
        assert_eq!(lex_all("5min"), vec![Token::Duration(5, DurationUnit::Minutes)]);
        assert_eq!(lex_all("1hour"), vec![Token::Duration(1, DurationUnit::Hours)]);
        assert_eq!(lex_all("3days"), vec![Token::Duration(3, DurationUnit::Days)]);
        assert_eq!(lex_all("2weeks"), vec![Token::Duration(2, DurationUnit::Weeks)]);
    }

    #[test]
    fn test_malformed_duration() {
        let mut lexer = Lexer::new("99xyz");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_duration_unit_to_seconds() {
        assert_eq!(DurationUnit::Seconds.to_seconds(10), 10);
        assert_eq!(DurationUnit::Minutes.to_seconds(2), 120);
        assert_eq!(DurationUnit::Hours.to_seconds(1), 3600);
        assert_eq!(DurationUnit::Days.to_seconds(1), 86400);
        assert_eq!(DurationUnit::Weeks.to_seconds(1), 604800);
    }

    // ====================================================================
    // Parameters
    // ====================================================================

    #[test]
    fn test_parameter() {
        let tokens = lex_all("$query_vec");
        assert_eq!(tokens, vec![Token::Parameter("query_vec".into())]);
    }

    #[test]
    fn test_dollar_alone_error() {
        // $ followed by non-alphanumeric → error
        let mut lexer = Lexer::new("$ ");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    // ====================================================================
    // Identifiers
    // ====================================================================

    #[test]
    fn test_identifier() {
        let tokens = lex_all("my_field");
        assert_eq!(tokens, vec![Token::Identifier("my_field".into())]);
    }

    #[test]
    fn test_identifier_with_numbers() {
        let tokens = lex_all("field123");
        assert_eq!(tokens, vec![Token::Identifier("field123".into())]);
    }

    #[test]
    fn test_long_identifier() {
        let long_id = "a".repeat(1000);
        let tokens = lex_all(&long_id);
        assert_eq!(tokens, vec![Token::Identifier(long_id)]);
    }

    // ====================================================================
    // Empty & whitespace
    // ====================================================================

    #[test]
    fn test_empty_input() {
        let tokens = lex_all("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let tokens = lex_all("   \t\n\r  ");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_whitespace_variants() {
        let tokens = lex_all("SELECT\t*\nFROM\r\ncollection");
        assert_eq!(
            tokens,
            vec![
                Token::Select,
                Token::Star,
                Token::From,
                Token::Identifier("collection".into()),
            ]
        );
    }

    // ====================================================================
    // Peek and position
    // ====================================================================

    #[test]
    fn test_peek_token_no_consume() {
        let mut lexer = Lexer::new("SELECT *");
        let peeked = lexer.peek_token().unwrap();
        assert_eq!(peeked, Token::Select);
        // Should still get Select when consuming
        let consumed = lexer.next_token().unwrap();
        assert_eq!(consumed, Token::Select);
    }

    #[test]
    fn test_position_tracking() {
        let mut lexer = Lexer::new("AB CD");
        assert_eq!(lexer.position(), 0);
        let _ = lexer.next_token().unwrap();
        // After reading "AB" and skipping whitespace check
        assert!(lexer.position() >= 2);
    }

    // ====================================================================
    // Invalid tokens
    // ====================================================================

    #[test]
    fn test_invalid_character() {
        let mut lexer = Lexer::new("@");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    #[test]
    fn test_bang_without_eq() {
        let mut lexer = Lexer::new("!");
        let result = lexer.next_token();
        assert!(result.is_err());
    }

    // ====================================================================
    // Complex expressions
    // ====================================================================

    #[test]
    fn test_full_select_query() {
        let tokens = lex_all("SELECT * FROM docs WHERE category = 'books' LIMIT 10");
        assert_eq!(tokens[0], Token::Select);
        assert_eq!(tokens[1], Token::Star);
        assert_eq!(tokens[2], Token::From);
        assert_eq!(tokens[3], Token::Identifier("docs".into()));
        assert_eq!(tokens[4], Token::Where);
        assert_eq!(tokens[5], Token::Identifier("category".into()));
        assert_eq!(tokens[6], Token::Eq);
        assert_eq!(tokens[7], Token::StringLit("books".into()));
        assert_eq!(tokens[8], Token::Limit);
        assert_eq!(tokens[9], Token::NumberLit(10.0));
    }

    #[test]
    fn test_between_expression() {
        let tokens = lex_all("price BETWEEN 10 AND 50");
        assert_eq!(
            tokens,
            vec![
                Token::Identifier("price".into()),
                Token::Between,
                Token::NumberLit(10.0),
                Token::And,
                Token::NumberLit(50.0),
            ]
        );
    }
}
