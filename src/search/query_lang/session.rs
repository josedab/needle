use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use super::ast::{LiteralValue, Query};
use super::parser::QueryParser;
use super::QueryError;

// ============================================================================
// Aggregation Functions
// ============================================================================

/// Aggregation function for NeedleQL queries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AggregateFunction {
    /// Count vectors
    Count,
    /// Count distinct values of a metadata field
    CountDistinct(String),
    /// Average of a numeric metadata field
    Avg(String),
    /// Minimum of a numeric metadata field
    Min(String),
    /// Maximum of a numeric metadata field
    Max(String),
    /// Sum of a numeric metadata field
    Sum(String),
}

impl AggregateFunction {
    /// Parse an aggregation function from a string like "COUNT(*)" or "AVG(price)".
    pub fn parse(s: &str) -> std::result::Result<Self, QueryError> {
        let s = s.trim();
        let upper = s.to_uppercase();

        if upper == "COUNT(*)" || upper == "COUNT" {
            return Ok(AggregateFunction::Count);
        }

        // Match FUNC(field) pattern
        if let Some(paren_start) = s.find('(') {
            if let Some(paren_end) = s.find(')') {
                let func = s[..paren_start].trim().to_uppercase();
                let field = s[paren_start + 1..paren_end].trim().to_string();

                if field.is_empty() {
                    return Err(QueryError::SemanticError {
                        message: "Aggregation function requires a field name".into(),
                    });
                }

                return match func.as_str() {
                    "COUNT_DISTINCT" | "COUNT DISTINCT" => {
                        Ok(AggregateFunction::CountDistinct(field))
                    }
                    "AVG" => Ok(AggregateFunction::Avg(field)),
                    "MIN" => Ok(AggregateFunction::Min(field)),
                    "MAX" => Ok(AggregateFunction::Max(field)),
                    "SUM" => Ok(AggregateFunction::Sum(field)),
                    _ => Err(QueryError::SemanticError {
                        message: format!("Unknown aggregation function: {}", func),
                    }),
                };
            }
        }

        Err(QueryError::InvalidLiteral {
            value: format!("Invalid aggregation syntax: {}", s),
        })
    }

    /// Apply the aggregation to a set of result metadata values.
    pub fn apply(&self, values: &[Option<&serde_json::Value>]) -> serde_json::Value {
        match self {
            AggregateFunction::Count => serde_json::json!(values.len()),
            AggregateFunction::CountDistinct(_) => {
                let unique: HashSet<String> = values
                    .iter()
                    .filter_map(|v| v.map(|x| x.to_string()))
                    .collect();
                serde_json::json!(unique.len())
            }
            AggregateFunction::Avg(field) => {
                let nums: Vec<f64> = values
                    .iter()
                    .filter_map(|v| v.and_then(|x| x.as_f64()))
                    .collect();
                if nums.is_empty() {
                    serde_json::json!(null)
                } else {
                    let avg = nums.iter().sum::<f64>() / nums.len() as f64;
                    serde_json::json!(avg)
                }
            }
            AggregateFunction::Min(_) => {
                values
                    .iter()
                    .filter_map(|v| v.and_then(|x| x.as_f64()))
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(serde_json::json!(null), |v| serde_json::json!(v))
            }
            AggregateFunction::Max(_) => {
                values
                    .iter()
                    .filter_map(|v| v.and_then(|x| x.as_f64()))
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(serde_json::json!(null), |v| serde_json::json!(v))
            }
            AggregateFunction::Sum(_) => {
                let total: f64 = values
                    .iter()
                    .filter_map(|v| v.and_then(|x| x.as_f64()))
                    .sum();
                serde_json::json!(total)
            }
        }
    }
}

// ============================================================================
// Interactive REPL Support
// ============================================================================

/// A NeedleQL session that maintains state across multiple queries.
///
/// Used for interactive REPL mode in the CLI (`needle query`).
pub struct QuerySession {
    /// Named parameters persisted across queries.
    parameters: HashMap<String, LiteralValue>,
    /// History of executed queries.
    history: Vec<String>,
    /// Default collection name (avoids repeating FROM clause).
    pub default_collection: Option<String>,
    /// Maximum results per query.
    pub default_limit: u64,
}

impl QuerySession {
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
            history: Vec::new(),
            default_collection: None,
            default_limit: 10,
        }
    }

    /// Set a named parameter that persists across queries.
    pub fn set_param(&mut self, name: &str, value: LiteralValue) {
        self.parameters.insert(name.to_string(), value);
    }

    /// Get a named parameter.
    pub fn get_param(&self, name: &str) -> Option<&LiteralValue> {
        self.parameters.get(name)
    }

    /// Clear all parameters.
    pub fn clear_params(&mut self) {
        self.parameters.clear();
    }

    /// Parse and record a query in history.
    pub fn parse_query(&mut self, input: &str) -> std::result::Result<Query, QueryError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(QueryError::InvalidLiteral {
                value: "Empty query".into(),
            });
        }

        // Handle special REPL commands
        if trimmed.starts_with('\\') || trimmed.starts_with('.') {
            return Err(QueryError::InvalidLiteral {
                value: format!(
                    "Unknown command: {}. Use .help for available commands.",
                    trimmed
                ),
            });
        }

        // Inject default collection if FROM is missing
        let query_str = if !trimmed.to_uppercase().contains("FROM")
            && self.default_collection.is_some()
        {
            // Safe: checked is_some() in the condition above
            let coll = self.default_collection.as_ref().expect("checked is_some above");
            if trimmed.to_uppercase().starts_with("SELECT") {
                trimmed.replacen("SELECT", &format!("SELECT"), 1)
                    + &format!(" FROM {}", coll)
            } else {
                format!("SELECT * FROM {} {}", coll, trimmed)
            }
        } else {
            trimmed.to_string()
        };

        let query = QueryParser::parse(&query_str)?;
        self.history.push(trimmed.to_string());
        Ok(query)
    }

    /// Get query history.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Get available REPL commands as help text.
    pub fn help_text() -> &'static str {
        concat!(
            "NeedleQL Interactive Shell Commands:\n",
            "  .use <collection>  - Set default collection\n",
            "  .params            - Show current parameters\n",
            "  .set <name> <val>  - Set a parameter\n",
            "  .history           - Show query history\n",
            "  .clear             - Clear parameters\n",
            "  .help              - Show this help\n",
            "  .quit              - Exit the shell\n",
            "\n",
            "NeedleQL Syntax:\n",
            "  SELECT * FROM <collection>\n",
            "    WHERE <field> <op> <value>\n",
            "    AND vector SIMILAR TO $query\n",
            "    WITH TIME_DECAY(EXPONENTIAL, 24h)\n",
            "    ORDER BY distance ASC\n",
            "    LIMIT 10 OFFSET 0\n",
            "  EXPLAIN ANALYZE SELECT ...\n",
        )
    }
}

impl Default for QuerySession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests needed: see docs/TODO-test-coverage.md
}
