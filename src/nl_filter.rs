// Allow dead_code for this public API module - types are exported for library users
#![allow(dead_code)]
#![allow(clippy::wrong_self_convention)]

//! Natural Language Filters
//!
//! Convert natural language queries to structured filters:
//! - "documents from last week" → date filter
//! - "articles about ML with score > 0.8" → metadata filters
//! - "images tagged as landscape" → tag filter
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::nl_filter::{NLFilterParser, ParsedQuery};
//!
//! let parser = NLFilterParser::new();
//!
//! let query = "Show me documents from last week about machine learning";
//! let parsed = parser.parse(query)?;
//!
//! println!("Search query: {}", parsed.search_text);
//! println!("Filter: {:?}", parsed.filter);
//! ```

use crate::metadata::Filter;
use serde::{Deserialize, Serialize};

/// Parsed natural language query
#[derive(Debug, Clone)]
pub struct ParsedQuery {
    /// The search text (without filter phrases)
    pub search_text: String,
    /// Extracted filter
    pub filter: Option<Filter>,
    /// Extracted temporal constraints
    pub temporal: Option<TemporalConstraint>,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Detected intents
    pub intents: Vec<QueryIntent>,
}

/// Temporal constraint extracted from query
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalConstraint {
    /// Start time (unix timestamp)
    pub start: Option<u64>,
    /// End time (unix timestamp)
    pub end: Option<u64>,
    /// Relative time expression
    pub expression: String,
}

/// Query intent
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryIntent {
    /// Looking for similar items
    Search,
    /// Filtering by criteria
    Filter,
    /// Aggregation/counting
    Aggregate,
    /// Comparison between items
    Compare,
    /// Temporal query
    Temporal,
    /// Negation (exclude something)
    Exclude,
}

/// Pattern for extracting filters
#[derive(Debug, Clone)]
struct FilterPattern {
    /// Regex-like pattern words
    keywords: Vec<&'static str>,
    /// Field name to filter on
    field: &'static str,
    /// How to extract the value
    extractor: ValueExtractor,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
enum ValueExtractor {
    /// Next word after keyword
    NextWord,
    /// Number following keyword
    Number,
    /// Comparison (>, <, =, etc.)
    Comparison,
    /// Date/time expression
    DateTime,
    /// Boolean (true/false, yes/no)
    Boolean,
    /// List of values (comma-separated)
    List,
}

/// Natural Language Filter Parser
pub struct NLFilterParser {
    patterns: Vec<FilterPattern>,
    temporal_patterns: Vec<TemporalPattern>,
    stopwords: std::collections::HashSet<&'static str>,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct TemporalPattern {
    keywords: Vec<&'static str>,
    offset_seconds: i64,
    is_range: bool,
}

impl Default for NLFilterParser {
    fn default() -> Self {
        Self::new()
    }
}

impl NLFilterParser {
    /// Create a new parser with default patterns
    pub fn new() -> Self {
        let patterns = vec![
            // Category/type filters
            FilterPattern {
                keywords: vec!["category", "type", "kind"],
                field: "category",
                extractor: ValueExtractor::NextWord,
            },
            FilterPattern {
                keywords: vec!["tagged", "tag", "tags", "labeled"],
                field: "tags",
                extractor: ValueExtractor::List,
            },
            FilterPattern {
                keywords: vec!["by", "author", "from", "created by"],
                field: "author",
                extractor: ValueExtractor::NextWord,
            },
            // Numeric comparisons
            FilterPattern {
                keywords: vec!["score", "rating", "rank"],
                field: "score",
                extractor: ValueExtractor::Comparison,
            },
            FilterPattern {
                keywords: vec!["price", "cost"],
                field: "price",
                extractor: ValueExtractor::Comparison,
            },
            FilterPattern {
                keywords: vec!["count", "number", "quantity"],
                field: "count",
                extractor: ValueExtractor::Comparison,
            },
            // Boolean filters
            FilterPattern {
                keywords: vec!["published", "active", "enabled"],
                field: "is_published",
                extractor: ValueExtractor::Boolean,
            },
            FilterPattern {
                keywords: vec!["verified", "confirmed"],
                field: "is_verified",
                extractor: ValueExtractor::Boolean,
            },
        ];

        let temporal_patterns = vec![
            TemporalPattern {
                keywords: vec!["today"],
                offset_seconds: 0,
                is_range: true,
            },
            TemporalPattern {
                keywords: vec!["yesterday"],
                offset_seconds: -86400,
                is_range: true,
            },
            TemporalPattern {
                keywords: vec!["last hour", "past hour"],
                offset_seconds: -3600,
                is_range: false,
            },
            TemporalPattern {
                keywords: vec!["last day", "past day", "24 hours"],
                offset_seconds: -86400,
                is_range: false,
            },
            TemporalPattern {
                keywords: vec!["last week", "past week", "this week"],
                offset_seconds: -604800,
                is_range: false,
            },
            TemporalPattern {
                keywords: vec!["last month", "past month", "this month"],
                offset_seconds: -2592000,
                is_range: false,
            },
            TemporalPattern {
                keywords: vec!["last year", "past year", "this year"],
                offset_seconds: -31536000,
                is_range: false,
            },
        ];

        let stopwords: std::collections::HashSet<&'static str> = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "about", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "just", "also",
            "now", "show", "me", "find", "get", "give", "search", "look",
        ].iter().copied().collect();

        Self {
            patterns,
            temporal_patterns,
            stopwords,
        }
    }

    /// Parse a natural language query
    pub fn parse(&self, query: &str) -> ParsedQuery {
        let query_lower = query.to_lowercase();
        let words: Vec<&str> = query_lower.split_whitespace().collect();

        let mut filters = Vec::new();
        let mut temporal = None;
        let mut intents = Vec::new();
        let mut used_ranges: Vec<(usize, usize)> = Vec::new();

        // Detect intents
        if self.contains_any(&query_lower, &["show", "find", "search", "get", "look for"]) {
            intents.push(QueryIntent::Search);
        }
        if self.contains_any(&query_lower, &["filter", "where", "with", "having"]) {
            intents.push(QueryIntent::Filter);
        }
        if self.contains_any(&query_lower, &["count", "how many", "total", "sum"]) {
            intents.push(QueryIntent::Aggregate);
        }
        if self.contains_any(&query_lower, &["not", "without", "exclude", "except"]) {
            intents.push(QueryIntent::Exclude);
        }

        // Extract temporal constraints
        for pattern in &self.temporal_patterns {
            for keyword in &pattern.keywords {
                if let Some(pos) = query_lower.find(keyword) {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs();

                    temporal = Some(TemporalConstraint {
                        start: Some((now as i64 + pattern.offset_seconds) as u64),
                        end: Some(now),
                        expression: keyword.to_string(),
                    });

                    intents.push(QueryIntent::Temporal);

                    // Mark this range as used
                    used_ranges.push((pos, pos + keyword.len()));
                    break;
                }
            }
            if temporal.is_some() {
                break;
            }
        }

        // Extract field filters
        for pattern in &self.patterns {
            if let Some((filter, range)) = self.extract_filter(&query_lower, &words, pattern) {
                filters.push(filter);
                used_ranges.push(range);
            }
        }

        // Combine filters
        let combined_filter = if filters.is_empty() {
            None
        } else if filters.len() == 1 {
            Some(filters.remove(0))
        } else {
            Some(Filter::And(filters))
        };

        // Build search text by removing filter phrases
        let search_text = self.build_search_text(query, &used_ranges);

        // Calculate confidence
        let confidence = self.calculate_confidence(&intents, combined_filter.is_some(), temporal.is_some());

        ParsedQuery {
            search_text,
            filter: combined_filter,
            temporal,
            confidence,
            intents,
        }
    }

    fn contains_any(&self, text: &str, patterns: &[&str]) -> bool {
        patterns.iter().any(|p| text.contains(p))
    }

    #[allow(clippy::needless_range_loop)]
    fn extract_filter(
        &self,
        _query: &str,
        words: &[&str],
        pattern: &FilterPattern,
    ) -> Option<(Filter, (usize, usize))> {
        for keyword in &pattern.keywords {
            if let Some(keyword_pos) = self.find_word_position(words, keyword) {
                let char_start = words[..keyword_pos].iter().map(|w| w.len() + 1).sum::<usize>();

                match &pattern.extractor {
                    ValueExtractor::NextWord => {
                        if keyword_pos + 1 < words.len() {
                            let value = words[keyword_pos + 1].trim_matches(|c: char| !c.is_alphanumeric());
                            if !self.stopwords.contains(value) {
                                let char_end = char_start + keyword.len() + 1 + value.len();
                                return Some((
                                    Filter::eq(pattern.field.to_string(), serde_json::json!(value)),
                                    (char_start, char_end),
                                ));
                            }
                        }
                    }
                    ValueExtractor::Comparison => {
                        if let Some((op, value, extra_len)) = self.extract_comparison(words, keyword_pos) {
                            let char_end = char_start + keyword.len() + extra_len;
                            let filter = match op {
                                ">" => Filter::gt(pattern.field.to_string(), serde_json::json!(value)),
                                ">=" => Filter::gte(pattern.field.to_string(), serde_json::json!(value)),
                                "<" => Filter::lt(pattern.field.to_string(), serde_json::json!(value)),
                                "<=" => Filter::lte(pattern.field.to_string(), serde_json::json!(value)),
                                _ => Filter::eq(pattern.field.to_string(), serde_json::json!(value)),
                            };
                            return Some((filter, (char_start, char_end)));
                        }
                    }
                    ValueExtractor::Boolean => {
                        // Check for negation
                        let is_negated = keyword_pos > 0 &&
                            (words[keyword_pos - 1] == "not" || words[keyword_pos - 1] == "un");
                        let value = !is_negated;
                        let char_end = char_start + keyword.len();
                        return Some((
                            Filter::eq(pattern.field.to_string(), serde_json::json!(value)),
                            (char_start, char_end),
                        ));
                    }
                    ValueExtractor::List => {
                        if keyword_pos + 1 < words.len() {
                            let mut values = Vec::new();
                            let mut end_pos = keyword_pos + 1;

                            for i in (keyword_pos + 1)..words.len() {
                                let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());
                                if self.stopwords.contains(word) || word == "and" || word == "or" {
                                    continue;
                                }
                                if word.is_empty() {
                                    break;
                                }
                                values.push(serde_json::json!(word));
                                end_pos = i + 1;
                                if !words[i].ends_with(',') {
                                    break;
                                }
                            }

                            if !values.is_empty() {
                                let char_end = words[..end_pos].iter().map(|w| w.len() + 1).sum::<usize>();
                                return Some((
                                    Filter::is_in(pattern.field.to_string(), values),
                                    (char_start, char_end),
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        None
    }

    fn find_word_position(&self, words: &[&str], keyword: &str) -> Option<usize> {
        // Handle multi-word keywords
        let keyword_words: Vec<&str> = keyword.split_whitespace().collect();

        if keyword_words.len() == 1 {
            words.iter().position(|w| w.starts_with(keyword))
        } else {
            for i in 0..words.len().saturating_sub(keyword_words.len() - 1) {
                if keyword_words.iter().enumerate().all(|(j, kw)| {
                    i + j < words.len() && words[i + j].starts_with(kw)
                }) {
                    return Some(i);
                }
            }
            None
        }
    }

    fn extract_comparison(&self, words: &[&str], keyword_pos: usize) -> Option<(&'static str, f64, usize)> {
        // Look for patterns like "score > 0.8" or "score greater than 0.8"
        let mut pos = keyword_pos + 1;
        let mut op = "=";
        let mut extra_len = 0;

        if pos >= words.len() {
            return None;
        }

        // Check for comparison operators
        let word = words[pos];
        if word == ">" || word == "greater" || word == "more" || word == "above" {
            op = ">";
            extra_len += word.len() + 1;
            pos += 1;
            if pos < words.len() && words[pos] == "than" {
                extra_len += 5;
                pos += 1;
            }
        } else if word == ">=" || word == "at" {
            op = ">=";
            extra_len += word.len() + 1;
            pos += 1;
            if pos < words.len() && words[pos] == "least" {
                extra_len += 6;
                pos += 1;
            }
        } else if word == "<" || word == "less" || word == "under" || word == "below" {
            op = "<";
            extra_len += word.len() + 1;
            pos += 1;
            if pos < words.len() && words[pos] == "than" {
                extra_len += 5;
                pos += 1;
            }
        } else if word == "<=" || word == "at" {
            op = "<=";
            extra_len += word.len() + 1;
            pos += 1;
            if pos < words.len() && words[pos] == "most" {
                extra_len += 5;
                pos += 1;
            }
        } else if word == "=" || word == "equals" || word == "is" || word == "of" {
            extra_len += word.len() + 1;
            pos += 1;
        }

        // Extract the number
        if pos < words.len() {
            if let Ok(value) = words[pos].parse::<f64>() {
                extra_len += words[pos].len();
                return Some((op, value, extra_len));
            }
        }

        None
    }

    fn build_search_text(&self, query: &str, used_ranges: &[(usize, usize)]) -> String {
        if used_ranges.is_empty() {
            return self.clean_search_text(query);
        }

        // Sort ranges
        let mut sorted_ranges = used_ranges.to_vec();
        sorted_ranges.sort_by_key(|r| r.0);

        // Build text excluding used ranges
        let mut result = String::new();
        let mut last_end = 0;

        for (start, end) in sorted_ranges {
            if start > last_end {
                result.push_str(&query[last_end..start]);
            }
            last_end = end.max(last_end);
        }

        if last_end < query.len() {
            result.push_str(&query[last_end..]);
        }

        self.clean_search_text(&result)
    }

    fn clean_search_text(&self, text: &str) -> String {
        // Remove extra whitespace and clean up
        let words: Vec<&str> = text
            .split_whitespace()
            .filter(|w| !self.stopwords.contains(&w.to_lowercase().as_str()) || w.len() > 3)
            .collect();

        words.join(" ").trim().to_string()
    }

    fn calculate_confidence(&self, intents: &[QueryIntent], has_filter: bool, has_temporal: bool) -> f32 {
        let mut confidence = 0.5;

        if !intents.is_empty() {
            confidence += 0.1 * intents.len() as f32;
        }

        if has_filter {
            confidence += 0.2;
        }

        if has_temporal {
            confidence += 0.1;
        }

        confidence.min(1.0)
    }

    /// Add a custom filter pattern
    pub fn add_pattern(&mut self, keywords: Vec<&'static str>, field: &'static str) {
        self.patterns.push(FilterPattern {
            keywords,
            field,
            extractor: ValueExtractor::NextWord,
        });
    }
}

/// Builder for creating complex NL queries programmatically
#[derive(Debug, Clone)]
pub struct QueryBuilder {
    search_text: String,
    filters: Vec<Filter>,
    temporal: Option<TemporalConstraint>,
}

impl QueryBuilder {
    pub fn new(search_text: impl Into<String>) -> Self {
        Self {
            search_text: search_text.into(),
            filters: Vec::new(),
            temporal: None,
        }
    }

    pub fn with_filter(mut self, filter: Filter) -> Self {
        self.filters.push(filter);
        self
    }

    pub fn with_category(self, category: &str) -> Self {
        self.with_filter(Filter::eq("category".to_string(), serde_json::json!(category)))
    }

    pub fn with_author(self, author: &str) -> Self {
        self.with_filter(Filter::eq("author".to_string(), serde_json::json!(author)))
    }

    pub fn with_score_above(self, score: f64) -> Self {
        self.with_filter(Filter::gt("score".to_string(), serde_json::json!(score)))
    }

    pub fn from_last_days(mut self, days: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.temporal = Some(TemporalConstraint {
            start: Some(now - days * 86400),
            end: Some(now),
            expression: format!("last {} days", days),
        });
        self
    }

    pub fn build(self) -> ParsedQuery {
        let filter = if self.filters.is_empty() {
            None
        } else if self.filters.len() == 1 {
            Some(self.filters.into_iter().next().unwrap())
        } else {
            Some(Filter::And(self.filters))
        };

        ParsedQuery {
            search_text: self.search_text,
            filter,
            temporal: self.temporal,
            confidence: 1.0,
            intents: vec![QueryIntent::Search],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query() {
        let parser = NLFilterParser::new();
        let result = parser.parse("machine learning");

        assert!(result.search_text.contains("machine") || result.search_text.contains("learning"));
        assert!(result.filter.is_none());
    }

    #[test]
    fn test_category_filter() {
        let parser = NLFilterParser::new();
        let result = parser.parse("show me documents category technology");

        assert!(result.filter.is_some());
        assert!(result.intents.contains(&QueryIntent::Search));
    }

    #[test]
    fn test_temporal_filter() {
        let parser = NLFilterParser::new();
        let result = parser.parse("articles from last week");

        assert!(result.temporal.is_some());
        assert!(result.intents.contains(&QueryIntent::Temporal));

        let temporal = result.temporal.unwrap();
        assert!(temporal.start.is_some());
        assert!(temporal.end.is_some());
    }

    #[test]
    fn test_comparison_filter() {
        let parser = NLFilterParser::new();
        let result = parser.parse("products with score greater than 0.8");

        assert!(result.filter.is_some());
    }

    #[test]
    fn test_query_builder() {
        let query = QueryBuilder::new("machine learning papers")
            .with_category("research")
            .with_score_above(0.9)
            .from_last_days(30)
            .build();

        assert_eq!(query.search_text, "machine learning papers");
        assert!(query.filter.is_some());
        assert!(query.temporal.is_some());
        assert_eq!(query.confidence, 1.0);
    }

    #[test]
    fn test_tag_filter() {
        let parser = NLFilterParser::new();
        let result = parser.parse("images tagged landscape, nature");

        assert!(result.filter.is_some());
    }

    #[test]
    fn test_boolean_filter() {
        let parser = NLFilterParser::new();
        let result = parser.parse("show published articles");

        assert!(result.filter.is_some());
    }

    #[test]
    fn test_negation_detection() {
        let parser = NLFilterParser::new();
        let result = parser.parse("documents without category archived");

        assert!(result.intents.contains(&QueryIntent::Exclude));
    }
}
