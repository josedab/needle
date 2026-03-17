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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ---- AggregateFunction tests ----

    #[test]
    fn test_parse_count_star() {
        let func = AggregateFunction::parse("COUNT(*)").unwrap();
        assert!(matches!(func, AggregateFunction::Count));
    }

    #[test]
    fn test_parse_count_keyword() {
        let func = AggregateFunction::parse("COUNT").unwrap();
        assert!(matches!(func, AggregateFunction::Count));
    }

    #[test]
    fn test_parse_count_distinct() {
        let func = AggregateFunction::parse("COUNT_DISTINCT(category)").unwrap();
        assert!(matches!(func, AggregateFunction::CountDistinct(ref f) if f == "category"));
    }

    #[test]
    fn test_parse_avg() {
        let func = AggregateFunction::parse("AVG(price)").unwrap();
        assert!(matches!(func, AggregateFunction::Avg(ref f) if f == "price"));
    }

    #[test]
    fn test_parse_min() {
        let func = AggregateFunction::parse("MIN(score)").unwrap();
        assert!(matches!(func, AggregateFunction::Min(ref f) if f == "score"));
    }

    #[test]
    fn test_parse_max() {
        let func = AggregateFunction::parse("MAX(score)").unwrap();
        assert!(matches!(func, AggregateFunction::Max(ref f) if f == "score"));
    }

    #[test]
    fn test_parse_sum() {
        let func = AggregateFunction::parse("SUM(amount)").unwrap();
        assert!(matches!(func, AggregateFunction::Sum(ref f) if f == "amount"));
    }

    #[test]
    fn test_parse_unknown_function() {
        let result = AggregateFunction::parse("MEDIAN(x)");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_empty_field() {
        let result = AggregateFunction::parse("AVG()");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_invalid_syntax() {
        let result = AggregateFunction::parse("not-a-function");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_case_insensitive() {
        let func = AggregateFunction::parse("avg(price)").unwrap();
        assert!(matches!(func, AggregateFunction::Avg(_)));

        let func = AggregateFunction::parse("count(*)").unwrap();
        assert!(matches!(func, AggregateFunction::Count));
    }

    #[test]
    fn test_parse_with_whitespace() {
        let func = AggregateFunction::parse("  AVG( price )  ").unwrap();
        assert!(matches!(func, AggregateFunction::Avg(_)));
    }

    // ---- AggregateFunction::apply tests ----

    #[test]
    fn test_apply_count() {
        let func = AggregateFunction::Count;
        let v1 = serde_json::json!(1);
        let v2 = serde_json::json!(2);
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2), None];
        assert_eq!(func.apply(&values), serde_json::json!(3));
    }

    #[test]
    fn test_apply_count_distinct() {
        let v1 = serde_json::json!("a");
        let v2 = serde_json::json!("b");
        let v3 = serde_json::json!("a");

        let func = AggregateFunction::CountDistinct("field".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2), Some(&v3), None];
        assert_eq!(func.apply(&values), serde_json::json!(2));
    }

    #[test]
    fn test_apply_avg() {
        let v1 = serde_json::json!(10.0);
        let v2 = serde_json::json!(20.0);
        let v3 = serde_json::json!(30.0);

        let func = AggregateFunction::Avg("price".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2), Some(&v3)];
        assert_eq!(func.apply(&values), serde_json::json!(20.0));
    }

    #[test]
    fn test_apply_avg_empty() {
        let func = AggregateFunction::Avg("price".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![];
        assert_eq!(func.apply(&values), serde_json::json!(null));
    }

    #[test]
    fn test_apply_min() {
        let v1 = serde_json::json!(5.0);
        let v2 = serde_json::json!(3.0);
        let v3 = serde_json::json!(8.0);

        let func = AggregateFunction::Min("score".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2), Some(&v3)];
        assert_eq!(func.apply(&values), serde_json::json!(3.0));
    }

    #[test]
    fn test_apply_max() {
        let v1 = serde_json::json!(5.0);
        let v2 = serde_json::json!(3.0);
        let v3 = serde_json::json!(8.0);

        let func = AggregateFunction::Max("score".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2), Some(&v3)];
        assert_eq!(func.apply(&values), serde_json::json!(8.0));
    }

    #[test]
    fn test_apply_sum() {
        let v1 = serde_json::json!(10.0);
        let v2 = serde_json::json!(20.0);

        let func = AggregateFunction::Sum("amount".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2)];
        assert_eq!(func.apply(&values), serde_json::json!(30.0));
    }

    #[test]
    fn test_apply_min_empty() {
        let func = AggregateFunction::Min("score".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![];
        assert_eq!(func.apply(&values), serde_json::json!(null));
    }

    #[test]
    fn test_apply_with_non_numeric() {
        let v1 = serde_json::json!("not a number");
        let v2 = serde_json::json!(10.0);

        let func = AggregateFunction::Avg("field".to_string());
        let values: Vec<Option<&serde_json::Value>> = vec![Some(&v1), Some(&v2)];
        // Only numeric values are counted
        assert_eq!(func.apply(&values), serde_json::json!(10.0));
    }

    // ---- QuerySession tests ----

    #[test]
    fn test_session_default() {
        let session = QuerySession::default();
        assert!(session.default_collection.is_none());
        assert_eq!(session.default_limit, 10);
        assert!(session.history().is_empty());
    }

    #[test]
    fn test_session_new() {
        let session = QuerySession::new();
        assert!(session.default_collection.is_none());
        assert_eq!(session.default_limit, 10);
    }

    #[test]
    fn test_session_set_and_get_param() {
        let mut session = QuerySession::new();
        session.set_param("k", LiteralValue::Number(20.0));

        assert_eq!(session.get_param("k"), Some(&LiteralValue::Number(20.0)));
        assert_eq!(session.get_param("missing"), None);
    }

    #[test]
    fn test_session_clear_params() {
        let mut session = QuerySession::new();
        session.set_param("a", LiteralValue::String("x".to_string()));
        session.set_param("b", LiteralValue::Number(1.0));

        session.clear_params();
        assert_eq!(session.get_param("a"), None);
        assert_eq!(session.get_param("b"), None);
    }

    #[test]
    fn test_session_empty_query_error() {
        let mut session = QuerySession::new();
        let result = session.parse_query("");
        assert!(result.is_err());
    }

    #[test]
    fn test_session_whitespace_query_error() {
        let mut session = QuerySession::new();
        let result = session.parse_query("   ");
        assert!(result.is_err());
    }

    #[test]
    fn test_session_repl_command_error() {
        let mut session = QuerySession::new();
        let result = session.parse_query("\\help");
        assert!(result.is_err());

        let result = session.parse_query(".quit");
        assert!(result.is_err());
    }

    #[test]
    fn test_session_history_records_queries() {
        let mut session = QuerySession::new();
        // Parse a valid query
        let _ = session.parse_query("SELECT * FROM docs");
        assert_eq!(session.history().len(), 1);
        assert_eq!(session.history()[0], "SELECT * FROM docs");
    }

    #[test]
    fn test_help_text_not_empty() {
        let help = QuerySession::help_text();
        assert!(!help.is_empty());
        assert!(help.contains(".use"));
        assert!(help.contains(".help"));
        assert!(help.contains(".quit"));
        assert!(help.contains("SELECT"));
        assert!(help.contains("WHERE"));
        assert!(help.contains("LIMIT"));
    }

    #[test]
    fn test_aggregate_function_serialization() {
        let funcs = vec![
            AggregateFunction::Count,
            AggregateFunction::CountDistinct("f".to_string()),
            AggregateFunction::Avg("f".to_string()),
            AggregateFunction::Min("f".to_string()),
            AggregateFunction::Max("f".to_string()),
            AggregateFunction::Sum("f".to_string()),
        ];

        for func in &funcs {
            let json = serde_json::to_string(func).unwrap();
            let deserialized: AggregateFunction = serde_json::from_str(&json).unwrap();
            assert_eq!(*func, deserialized);
        }
    }
}
