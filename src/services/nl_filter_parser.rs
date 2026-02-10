//! Natural Language Filter Parser
//!
//! Parses human-readable filter expressions into MongoDB-style metadata filters.
//! Supports rule-based parsing for common patterns (dates, comparisons, categories)
//! with fallback to pattern matching for complex queries.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::nl_filter_parser::{NLFilterParser, ParsedFilter};
//!
//! let parser = NLFilterParser::new();
//!
//! // Parse natural language into a filter
//! let result = parser.parse("category is 'books' and price less than 50").unwrap();
//! println!("Filter: {}", result.filter_json);
//!
//! // Parse date-based filters
//! let result = parser.parse("created after 2023-01-01").unwrap();
//! println!("Filter: {}", result.filter_json);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{NeedleError, Result};
use crate::metadata::Filter;

// ── Parsed Result ────────────────────────────────────────────────────────────

/// Result of parsing a natural language filter expression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedFilter {
    /// The generated MongoDB-style filter JSON.
    pub filter_json: Value,
    /// The parsed Filter object (ready to use with collection.search_with_filter).
    #[serde(skip)]
    pub filter: Option<Filter>,
    /// Confidence score (0.0–1.0) in the parse quality.
    pub confidence: f64,
    /// Individual clauses that were parsed.
    pub clauses: Vec<FilterClause>,
    /// Any part of the input that couldn't be parsed.
    pub unparsed_remainder: Option<String>,
}

/// A single parsed filter clause.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterClause {
    /// The field name.
    pub field: String,
    /// The operator.
    pub operator: FilterOp,
    /// The value.
    pub value: Value,
    /// The raw text that produced this clause.
    pub source_text: String,
}

/// Supported filter operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FilterOp {
    /// Equality.
    Eq,
    /// Not equal.
    Ne,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Gte,
    /// Less than.
    Lt,
    /// Less than or equal.
    Lte,
    /// In set.
    In,
    /// Not in set.
    Nin,
    /// Contains (string).
    Contains,
}

impl FilterOp {
    fn to_mongo_key(self) -> &'static str {
        match self {
            Self::Eq => "$eq",
            Self::Ne => "$ne",
            Self::Gt => "$gt",
            Self::Gte => "$gte",
            Self::Lt => "$lt",
            Self::Lte => "$lte",
            Self::In => "$in",
            Self::Nin => "$nin",
            Self::Contains => "$contains",
        }
    }
}

// ── Pattern Rules ────────────────────────────────────────────────────────────

/// A pattern rule for matching NL expressions.
#[derive(Debug, Clone)]
struct PatternRule {
    /// Keywords that trigger this rule.
    keywords: Vec<&'static str>,
    /// The operator to apply.
    operator: FilterOp,
    /// Whether the value comes before the keyword (reverse order).
    value_before: bool,
}

fn default_rules() -> Vec<PatternRule> {
    vec![
        // Equality
        PatternRule {
            keywords: vec!["is", "equals", "equal to", "=", "=="],
            operator: FilterOp::Eq,
            value_before: false,
        },
        // Not equal
        PatternRule {
            keywords: vec!["is not", "not equal to", "!=", "isn't"],
            operator: FilterOp::Ne,
            value_before: false,
        },
        // Greater than
        PatternRule {
            keywords: vec![
                "greater than",
                "more than",
                "above",
                "over",
                "exceeds",
                ">",
            ],
            operator: FilterOp::Gt,
            value_before: false,
        },
        // Greater than or equal
        PatternRule {
            keywords: vec![
                "at least",
                "greater than or equal to",
                "no less than",
                ">=",
            ],
            operator: FilterOp::Gte,
            value_before: false,
        },
        // Less than
        PatternRule {
            keywords: vec![
                "less than",
                "fewer than",
                "below",
                "under",
                "<",
            ],
            operator: FilterOp::Lt,
            value_before: false,
        },
        // Less than or equal
        PatternRule {
            keywords: vec![
                "at most",
                "less than or equal to",
                "no more than",
                "<=",
            ],
            operator: FilterOp::Lte,
            value_before: false,
        },
        // Contains
        PatternRule {
            keywords: vec!["contains", "includes", "has"],
            operator: FilterOp::Contains,
            value_before: false,
        },
        // After (date) → greater than
        PatternRule {
            keywords: vec!["after", "since", "from"],
            operator: FilterOp::Gt,
            value_before: false,
        },
        // Before (date) → less than
        PatternRule {
            keywords: vec!["before", "until", "by"],
            operator: FilterOp::Lt,
            value_before: false,
        },
    ]
}

// ── Schema Hints ─────────────────────────────────────────────────────────────

/// Schema hint for a metadata field (improves parse accuracy).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldHint {
    /// Field name.
    pub name: String,
    /// Expected value type.
    pub value_type: FieldType,
    /// Known possible values (for enum-like fields).
    pub known_values: Vec<String>,
}

/// Metadata field value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// String field.
    String,
    /// Numeric field (integer or float).
    Number,
    /// Boolean field.
    Boolean,
    /// Date/timestamp field.
    Date,
    /// Array field.
    Array,
}

// ── Parser ───────────────────────────────────────────────────────────────────

/// Natural language filter parser.
pub struct NLFilterParser {
    rules: Vec<PatternRule>,
    field_hints: HashMap<String, FieldHint>,
    /// Known field aliases (e.g., "category" → "cat", "price" → "cost")
    aliases: HashMap<String, String>,
}

impl Default for NLFilterParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NLFilterParser {
    /// Create a new parser with default rules.
    pub fn new() -> Self {
        Self {
            rules: default_rules(),
            field_hints: HashMap::new(),
            aliases: HashMap::new(),
        }
    }

    /// Add a schema hint for a field.
    pub fn add_field_hint(&mut self, hint: FieldHint) {
        self.field_hints.insert(hint.name.clone(), hint);
    }

    /// Add a field alias.
    pub fn add_alias(&mut self, alias: impl Into<String>, canonical: impl Into<String>) {
        self.aliases.insert(alias.into(), canonical.into());
    }

    /// Parse a natural language filter expression.
    pub fn parse(&self, input: &str) -> Result<ParsedFilter> {
        let input = input.trim();
        if input.is_empty() {
            return Err(NeedleError::InvalidArgument(
                "empty filter expression".into(),
            ));
        }

        let normalized = input.to_lowercase();

        // Split on "and" / "or" conjunctions
        let (conjunction, parts) = split_conjunctions(&normalized);

        let mut clauses = Vec::new();
        let mut unparsed = Vec::new();

        for part in &parts {
            match self.parse_clause(part.trim()) {
                Some(clause) => clauses.push(clause),
                None => unparsed.push(part.trim().to_string()),
            }
        }

        if clauses.is_empty() {
            return Err(NeedleError::InvalidArgument(format!(
                "could not parse any filter clauses from: '{input}'"
            )));
        }

        let filter_json = self.build_filter_json(&clauses, &conjunction);
        let confidence = clauses.len() as f64 / parts.len() as f64;
        let filter = Filter::parse(&filter_json).ok();

        let unparsed_remainder = if unparsed.is_empty() {
            None
        } else {
            Some(unparsed.join(", "))
        };

        Ok(ParsedFilter {
            filter_json,
            filter,
            confidence,
            clauses,
            unparsed_remainder,
        })
    }

    /// Parse a single clause like "category is 'books'" or "price less than 50".
    fn parse_clause(&self, text: &str) -> Option<FilterClause> {
        // Try each rule, longest keyword first to avoid partial matches
        let mut best_match: Option<(FilterClause, usize)> = None;

        for rule in &self.rules {
            for &keyword in &rule.keywords {
                if let Some(pos) = text.find(keyword) {
                    let before = text[..pos].trim();
                    let after = text[pos + keyword.len()..].trim();

                    let (field, value_str) = if rule.value_before {
                        (after, before)
                    } else {
                        (before, after)
                    };

                    if field.is_empty() || value_str.is_empty() {
                        continue;
                    }

                    let field = self.resolve_field(field);
                    let value = self.parse_value(value_str, &field);

                    let clause = FilterClause {
                        field,
                        operator: rule.operator,
                        value,
                        source_text: text.to_string(),
                    };

                    let match_quality = keyword.len();
                    if best_match.as_ref().map_or(true, |(_, q)| match_quality > *q) {
                        best_match = Some((clause, match_quality));
                    }
                }
            }
        }

        best_match.map(|(c, _)| c)
    }

    fn resolve_field(&self, raw: &str) -> String {
        let cleaned = raw
            .trim()
            .trim_matches(|c: char| c == '\'' || c == '"')
            .to_string();

        self.aliases
            .get(&cleaned)
            .cloned()
            .unwrap_or(cleaned)
    }

    fn parse_value(&self, raw: &str, field: &str) -> Value {
        let cleaned = raw
            .trim()
            .trim_matches(|c: char| c == '\'' || c == '"');

        // Check field hints for type guidance
        if let Some(hint) = self.field_hints.get(field) {
            match hint.value_type {
                FieldType::Number => {
                    if let Ok(n) = cleaned.parse::<f64>() {
                        return json!(n);
                    }
                }
                FieldType::Boolean => {
                    return match cleaned {
                        "true" | "yes" | "1" => json!(true),
                        "false" | "no" | "0" => json!(false),
                        _ => json!(cleaned),
                    };
                }
                _ => {}
            }
        }

        // Auto-detect type
        if let Ok(n) = cleaned.parse::<i64>() {
            return json!(n);
        }
        if let Ok(n) = cleaned.parse::<f64>() {
            return json!(n);
        }
        if cleaned == "true" || cleaned == "false" {
            return json!(cleaned.parse::<bool>().unwrap());
        }

        json!(cleaned)
    }

    fn build_filter_json(&self, clauses: &[FilterClause], conjunction: &str) -> Value {
        if clauses.len() == 1 {
            let c = &clauses[0];
            return json!({ &c.field: { c.operator.to_mongo_key(): c.value } });
        }

        let clause_jsons: Vec<Value> = clauses
            .iter()
            .map(|c| json!({ &c.field: { c.operator.to_mongo_key(): c.value } }))
            .collect();

        if conjunction == "or" {
            json!({ "$or": clause_jsons })
        } else {
            json!({ "$and": clause_jsons })
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn split_conjunctions(input: &str) -> (String, Vec<String>) {
    // Check for "or" first (it has lower precedence)
    let or_parts: Vec<&str> = input.split(" or ").collect();
    if or_parts.len() > 1 {
        return (
            "or".into(),
            or_parts.iter().map(|s| s.to_string()).collect(),
        );
    }

    // Split on "and"
    let and_parts: Vec<&str> = input.split(" and ").collect();
    if and_parts.len() > 1 {
        return (
            "and".into(),
            and_parts.iter().map(|s| s.to_string()).collect(),
        );
    }

    ("and".into(), vec![input.to_string()])
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_equality() {
        let parser = NLFilterParser::new();
        let result = parser.parse("category is 'books'").unwrap();
        assert_eq!(result.clauses.len(), 1);
        assert_eq!(result.clauses[0].field, "category");
        assert_eq!(result.clauses[0].operator, FilterOp::Eq);
        assert_eq!(result.clauses[0].value, json!("books"));
        assert!(result.confidence > 0.9);
    }

    #[test]
    fn test_numeric_comparison() {
        let parser = NLFilterParser::new();
        let result = parser.parse("price less than 50").unwrap();
        assert_eq!(result.clauses[0].field, "price");
        assert_eq!(result.clauses[0].operator, FilterOp::Lt);
        assert_eq!(result.clauses[0].value, json!(50));
    }

    #[test]
    fn test_compound_and() {
        let parser = NLFilterParser::new();
        let result = parser
            .parse("category is 'books' and price less than 50")
            .unwrap();
        assert_eq!(result.clauses.len(), 2);
        assert!(result.filter_json.get("$and").is_some());
    }

    #[test]
    fn test_compound_or() {
        let parser = NLFilterParser::new();
        let result = parser
            .parse("status is 'active' or status is 'pending'")
            .unwrap();
        assert_eq!(result.clauses.len(), 2);
        assert!(result.filter_json.get("$or").is_some());
    }

    #[test]
    fn test_date_filter() {
        let parser = NLFilterParser::new();
        let result = parser.parse("created after 2023-01-01").unwrap();
        assert_eq!(result.clauses[0].field, "created");
        assert_eq!(result.clauses[0].operator, FilterOp::Gt);
    }

    #[test]
    fn test_greater_than_or_equal() {
        let parser = NLFilterParser::new();
        let result = parser.parse("score at least 0.5").unwrap();
        assert_eq!(result.clauses[0].operator, FilterOp::Gte);
        assert_eq!(result.clauses[0].value, json!(0.5));
    }

    #[test]
    fn test_contains_operator() {
        let parser = NLFilterParser::new();
        let result = parser.parse("title contains 'machine learning'").unwrap();
        assert_eq!(result.clauses[0].operator, FilterOp::Contains);
        assert_eq!(result.clauses[0].value, json!("machine learning"));
    }

    #[test]
    fn test_field_aliases() {
        let mut parser = NLFilterParser::new();
        parser.add_alias("cost", "price");

        let result = parser.parse("cost less than 100").unwrap();
        assert_eq!(result.clauses[0].field, "price");
    }

    #[test]
    fn test_field_hints() {
        let mut parser = NLFilterParser::new();
        parser.add_field_hint(FieldHint {
            name: "active".into(),
            value_type: FieldType::Boolean,
            known_values: vec![],
        });

        let result = parser.parse("active is yes").unwrap();
        assert_eq!(result.clauses[0].value, json!(true));
    }

    #[test]
    fn test_empty_input_error() {
        let parser = NLFilterParser::new();
        assert!(parser.parse("").is_err());
    }

    #[test]
    fn test_unparseable_input() {
        let parser = NLFilterParser::new();
        assert!(parser.parse("xyz abc def ghi").is_err());
    }

    #[test]
    fn test_single_clause_no_and_wrapper() {
        let parser = NLFilterParser::new();
        let result = parser.parse("type is 'blog'").unwrap();
        // Single clause should NOT be wrapped in $and
        assert!(result.filter_json.get("$and").is_none());
        assert!(result.filter_json.get("type").is_some());
    }

    #[test]
    fn test_not_equal() {
        let parser = NLFilterParser::new();
        let result = parser.parse("status is not 'deleted'").unwrap();
        assert_eq!(result.clauses[0].operator, FilterOp::Ne);
    }

    #[test]
    fn test_float_parsing() {
        let parser = NLFilterParser::new();
        let result = parser.parse("score greater than 0.85").unwrap();
        assert_eq!(result.clauses[0].value, json!(0.85));
    }

    #[test]
    fn test_partial_parse_confidence() {
        let parser = NLFilterParser::new();
        let result = parser
            .parse("category is 'tech' and some unknown clause")
            .unwrap();
        assert!(result.confidence < 1.0);
        assert!(result.unparsed_remainder.is_some());
    }
}
