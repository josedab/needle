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
