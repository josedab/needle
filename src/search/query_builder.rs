//! Visual Query Builder
//!
//! A natural language interface for building vector search queries with
//! optimization hints, suggestions, and real-time feedback.
//!
//! # Features
//!
//! - **Query Intent Analysis**: Classify queries as semantic, keyword, or hybrid
//! - **Optimization Hints**: Suggest index usage, filter order, and search parameters
//! - **Query Translation**: Convert natural language to optimized NeedleQL
//! - **Quality Scoring**: Rate query effectiveness and suggest improvements
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::query_builder::{QueryBuilder, CollectionProfile};
//!
//! let profile = CollectionProfile::new("documents", 384, 100_000);
//! let builder = QueryBuilder::new(profile);
//!
//! // Parse natural language query
//! let result = builder.build("find recent articles about machine learning with score > 0.8");
//!
//! println!("NeedleQL: {}", result.needleql);
//! println!("Hints: {:?}", result.optimization_hints);
//! println!("Quality: {:.2}", result.quality_score);
//! ```

use crate::nl_filter::{NLFilterParser, ParsedQuery, TemporalConstraint};
use crate::query_lang::{Query, QueryParser, Expression};
use crate::metadata::Filter;
use serde::{Deserialize, Serialize};

// ============================================================================
// Collection Profile
// ============================================================================

/// Profile of a collection for optimization decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionProfile {
    /// Collection name
    pub name: String,
    /// Vector dimensionality
    pub dimensions: usize,
    /// Approximate vector count
    pub vector_count: usize,
    /// Available metadata fields
    pub metadata_fields: Vec<FieldProfile>,
    /// Index configuration
    pub index_config: IndexProfile,
    /// Search statistics
    pub stats: CollectionStats,
}

/// Profile of a metadata field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldProfile {
    /// Field name
    pub name: String,
    /// Field type
    pub field_type: FieldType,
    /// Cardinality (unique values)
    pub cardinality: usize,
    /// Whether field is indexed
    pub indexed: bool,
    /// Sample values for suggestions
    pub sample_values: Vec<String>,
}

/// Field data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    String,
    Number,
    Boolean,
    Array,
    DateTime,
}

/// Index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexProfile {
    /// HNSW M parameter
    pub hnsw_m: usize,
    /// HNSW ef_construction
    pub ef_construction: usize,
    /// Default ef_search
    pub ef_search: usize,
    /// Distance function
    pub distance: String,
    /// Quantization type if any
    pub quantization: Option<String>,
}

impl Default for IndexProfile {
    fn default() -> Self {
        Self {
            hnsw_m: 16,
            ef_construction: 200,
            ef_search: 50,
            distance: "cosine".to_string(),
            quantization: None,
        }
    }
}

/// Collection search statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionStats {
    /// Average query latency in ms
    pub avg_latency_ms: f64,
    /// Average recall rate
    pub avg_recall: f64,
    /// Most common filter fields
    pub common_filters: Vec<String>,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

impl CollectionProfile {
    /// Create a minimal profile
    pub fn new(name: impl Into<String>, dimensions: usize, vector_count: usize) -> Self {
        Self {
            name: name.into(),
            dimensions,
            vector_count,
            metadata_fields: Vec::new(),
            index_config: IndexProfile::default(),
            stats: CollectionStats::default(),
        }
    }

    /// Add a metadata field
    pub fn with_field(mut self, field: FieldProfile) -> Self {
        self.metadata_fields.push(field);
        self
    }

    /// Set index configuration
    pub fn with_index(mut self, config: IndexProfile) -> Self {
        self.index_config = config;
        self
    }
}

// ============================================================================
// Query Analysis
// ============================================================================

/// Analyzed query classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryClass {
    /// Pure semantic similarity search
    Semantic,
    /// Keyword-based filtering
    Keyword,
    /// Hybrid semantic + keyword
    Hybrid,
    /// Metadata-only filtering
    MetadataOnly,
    /// Aggregation query
    Aggregation,
    /// Temporal-focused query
    Temporal,
}

/// Query complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryComplexity {
    Simple,
    Moderate,
    Complex,
}

/// Detailed query analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryAnalysis {
    /// Query classification
    pub class: QueryClass,
    /// Complexity level
    pub complexity: QueryComplexity,
    /// Detected search terms
    pub search_terms: Vec<String>,
    /// Detected filter fields
    pub filter_fields: Vec<String>,
    /// Temporal constraint if any
    pub temporal: Option<TemporalConstraint>,
    /// Confidence in analysis
    pub confidence: f32,
    /// Detected language patterns
    pub patterns: Vec<DetectedPattern>,
}

/// Detected query pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Matched text
    pub matched_text: String,
    /// Start position in query
    pub start: usize,
    /// End position in query
    pub end: usize,
}

/// Types of detected patterns
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    SearchKeyword,
    FilterClause,
    TemporalExpression,
    Comparison,
    Negation,
    ListValue,
    RangeExpression,
    Aggregation,
    SortOrder,
}

/// Query analyzer
pub struct QueryAnalyzer {
    nl_parser: NLFilterParser,
    patterns: Vec<AnalyzerPattern>,
}

#[derive(Debug, Clone)]
struct AnalyzerPattern {
    keywords: Vec<&'static str>,
    pattern_type: PatternType,
}

impl Default for QueryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        let patterns = vec![
            AnalyzerPattern {
                keywords: vec!["find", "search", "show", "get", "look for", "retrieve"],
                pattern_type: PatternType::SearchKeyword,
            },
            AnalyzerPattern {
                keywords: vec!["where", "with", "having", "that have", "containing"],
                pattern_type: PatternType::FilterClause,
            },
            AnalyzerPattern {
                keywords: vec!["today", "yesterday", "last week", "last month", "recent", "latest", "from", "since", "before", "after"],
                pattern_type: PatternType::TemporalExpression,
            },
            AnalyzerPattern {
                keywords: vec!["greater than", "less than", "more than", "at least", "at most", "above", "below", ">", "<", ">=", "<="],
                pattern_type: PatternType::Comparison,
            },
            AnalyzerPattern {
                keywords: vec!["not", "without", "except", "exclude", "excluding"],
                pattern_type: PatternType::Negation,
            },
            AnalyzerPattern {
                keywords: vec!["or", "and", ",", "either"],
                pattern_type: PatternType::ListValue,
            },
            AnalyzerPattern {
                keywords: vec!["between", "from...to", "range"],
                pattern_type: PatternType::RangeExpression,
            },
            AnalyzerPattern {
                keywords: vec!["count", "how many", "total", "sum", "average", "group by"],
                pattern_type: PatternType::Aggregation,
            },
            AnalyzerPattern {
                keywords: vec!["sort by", "order by", "sorted", "ranked", "top", "best", "highest", "lowest"],
                pattern_type: PatternType::SortOrder,
            },
        ];

        Self {
            nl_parser: NLFilterParser::new(),
            patterns,
        }
    }

    /// Analyze a natural language query
    pub fn analyze(&self, query: &str) -> QueryAnalysis {
        let parsed = self.nl_parser.parse(query);
        let query_lower = query.to_lowercase();

        // Detect patterns
        let patterns = self.detect_patterns(&query_lower);

        // Classify query
        let class = self.classify_query(&parsed, &patterns);

        // Determine complexity
        let complexity = self.assess_complexity(&parsed, &patterns);

        // Extract search terms
        let search_terms = self.extract_search_terms(&parsed.search_text);

        // Extract filter fields
        let filter_fields = self.extract_filter_fields(&parsed.filter);

        QueryAnalysis {
            class,
            complexity,
            search_terms,
            filter_fields,
            temporal: parsed.temporal,
            confidence: parsed.confidence,
            patterns,
        }
    }

    fn detect_patterns(&self, query: &str) -> Vec<DetectedPattern> {
        let mut detected = Vec::new();

        for pattern in &self.patterns {
            for keyword in &pattern.keywords {
                if let Some(pos) = query.find(keyword) {
                    detected.push(DetectedPattern {
                        pattern_type: pattern.pattern_type,
                        matched_text: keyword.to_string(),
                        start: pos,
                        end: pos + keyword.len(),
                    });
                }
            }
        }

        detected.sort_by_key(|p| p.start);
        detected
    }

    fn classify_query(&self, parsed: &ParsedQuery, patterns: &[DetectedPattern]) -> QueryClass {
        let has_search = !parsed.search_text.trim().is_empty();
        let has_filter = parsed.filter.is_some();
        let has_temporal = parsed.temporal.is_some();
        let has_aggregation = patterns.iter().any(|p| p.pattern_type == PatternType::Aggregation);

        if has_aggregation {
            return QueryClass::Aggregation;
        }

        if has_temporal && !has_search && !has_filter {
            return QueryClass::Temporal;
        }

        match (has_search, has_filter) {
            (true, true) => QueryClass::Hybrid,
            (true, false) => QueryClass::Semantic,
            (false, true) => QueryClass::MetadataOnly,
            (false, false) => {
                if has_temporal {
                    QueryClass::Temporal
                } else {
                    QueryClass::Semantic
                }
            }
        }
    }

    fn assess_complexity(&self, parsed: &ParsedQuery, patterns: &[DetectedPattern]) -> QueryComplexity {
        let mut score = 0;

        // Count filter conditions
        if let Some(filter) = &parsed.filter {
            score += self.count_filter_depth(filter);
        }

        // Count detected patterns
        score += patterns.len();

        // Temporal adds complexity
        if parsed.temporal.is_some() {
            score += 2;
        }

        // Check for aggregation
        if patterns.iter().any(|p| p.pattern_type == PatternType::Aggregation) {
            score += 3;
        }

        // Check for negation
        if patterns.iter().any(|p| p.pattern_type == PatternType::Negation) {
            score += 1;
        }

        match score {
            0..=3 => QueryComplexity::Simple,
            4..=7 => QueryComplexity::Moderate,
            _ => QueryComplexity::Complex,
        }
    }

    fn count_filter_depth(&self, filter: &Filter) -> usize {
        match filter {
            Filter::And(filters) | Filter::Or(filters) => {
                1 + filters.iter().map(|f| self.count_filter_depth(f)).sum::<usize>()
            }
            Filter::Not(inner) => 1 + self.count_filter_depth(inner),
            Filter::Condition(_) => 1,
        }
    }

    fn extract_search_terms(&self, search_text: &str) -> Vec<String> {
        search_text
            .split_whitespace()
            .filter(|w| w.len() > 2)
            .map(|w| w.to_lowercase())
            .collect()
    }

    fn extract_filter_fields(&self, filter: &Option<Filter>) -> Vec<String> {
        let mut fields = Vec::new();
        if let Some(f) = filter {
            self.collect_filter_fields(f, &mut fields);
        }
        fields
    }

    fn collect_filter_fields(&self, filter: &Filter, fields: &mut Vec<String>) {
        match filter {
            Filter::And(filters) | Filter::Or(filters) => {
                for f in filters {
                    self.collect_filter_fields(f, fields);
                }
            }
            Filter::Not(inner) => self.collect_filter_fields(inner, fields),
            Filter::Condition(cond) => {
                if !fields.contains(&cond.field) {
                    fields.push(cond.field.clone());
                }
            }
        }
    }
}

// ============================================================================
// Optimization Hints
// ============================================================================

/// Optimization hint for query execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationHint {
    /// Hint category
    pub category: HintCategory,
    /// Severity level
    pub severity: HintSeverity,
    /// Human-readable message
    pub message: String,
    /// Suggested action
    pub suggestion: String,
    /// Estimated impact
    pub impact: HintImpact,
}

/// Hint categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HintCategory {
    IndexUsage,
    FilterOrder,
    SearchParameters,
    Caching,
    Quantization,
    BatchProcessing,
    MemoryUsage,
    QueryStructure,
}

/// Hint severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd, Serialize, Deserialize)]
pub enum HintSeverity {
    Info,
    Suggestion,
    Warning,
    Critical,
}

/// Estimated impact
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintImpact {
    /// Latency change estimate (negative = improvement)
    pub latency_change_percent: f32,
    /// Memory change estimate
    pub memory_change_percent: f32,
    /// Recall change estimate
    pub recall_change_percent: f32,
}

/// Hint generator
pub struct HintGenerator;

impl HintGenerator {
    /// Generate optimization hints for a query
    pub fn generate(
        analysis: &QueryAnalysis,
        profile: &CollectionProfile,
    ) -> Vec<OptimizationHint> {
        let mut hints = Vec::new();

        // Check filter field indexing
        for field in &analysis.filter_fields {
            if let Some(field_profile) = profile.metadata_fields.iter().find(|f| &f.name == field) {
                if !field_profile.indexed && field_profile.cardinality > 100 {
                    hints.push(OptimizationHint {
                        category: HintCategory::IndexUsage,
                        severity: HintSeverity::Warning,
                        message: format!("Field '{}' is not indexed but has high cardinality ({})", field, field_profile.cardinality),
                        suggestion: format!("Consider adding an index on '{}' for faster filtering", field),
                        impact: HintImpact {
                            latency_change_percent: -30.0,
                            memory_change_percent: 5.0,
                            recall_change_percent: 0.0,
                        },
                    });
                }
            }
        }

        // Check for complex query optimization
        if analysis.complexity == QueryComplexity::Complex {
            hints.push(OptimizationHint {
                category: HintCategory::QueryStructure,
                severity: HintSeverity::Suggestion,
                message: "Query has high complexity which may impact performance".to_string(),
                suggestion: "Consider breaking into multiple simpler queries or pre-filtering".to_string(),
                impact: HintImpact {
                    latency_change_percent: -20.0,
                    memory_change_percent: -10.0,
                    recall_change_percent: 0.0,
                },
            });
        }

        // Check for large result sets
        if profile.vector_count > 100_000 && analysis.filter_fields.is_empty() {
            hints.push(OptimizationHint {
                category: HintCategory::FilterOrder,
                severity: HintSeverity::Suggestion,
                message: "Large collection without pre-filters may have higher latency".to_string(),
                suggestion: "Add metadata filters to reduce search space".to_string(),
                impact: HintImpact {
                    latency_change_percent: -40.0,
                    memory_change_percent: -20.0,
                    recall_change_percent: 0.0,
                },
            });
        }

        // Suggest quantization for large collections
        if profile.vector_count > 500_000 && profile.index_config.quantization.is_none() {
            hints.push(OptimizationHint {
                category: HintCategory::Quantization,
                severity: HintSeverity::Suggestion,
                message: "Large collection without quantization uses significant memory".to_string(),
                suggestion: "Consider enabling scalar or product quantization".to_string(),
                impact: HintImpact {
                    latency_change_percent: 5.0,
                    memory_change_percent: -75.0,
                    recall_change_percent: -2.0,
                },
            });
        }

        // Check HNSW parameters for recall
        if analysis.class == QueryClass::Semantic && profile.index_config.ef_search < 100 {
            hints.push(OptimizationHint {
                category: HintCategory::SearchParameters,
                severity: HintSeverity::Info,
                message: format!("ef_search={} may trade recall for speed", profile.index_config.ef_search),
                suggestion: "Increase ef_search to 100-200 for higher recall".to_string(),
                impact: HintImpact {
                    latency_change_percent: 30.0,
                    memory_change_percent: 0.0,
                    recall_change_percent: 10.0,
                },
            });
        }

        // Caching hint for repeated queries
        if profile.stats.cache_hit_rate < 0.3 && analysis.complexity == QueryComplexity::Simple {
            hints.push(OptimizationHint {
                category: HintCategory::Caching,
                severity: HintSeverity::Info,
                message: "Simple query pattern may benefit from result caching".to_string(),
                suggestion: "Enable query result caching for repeated similar queries".to_string(),
                impact: HintImpact {
                    latency_change_percent: -80.0,
                    memory_change_percent: 10.0,
                    recall_change_percent: 0.0,
                },
            });
        }

        // Sort by severity
        hints.sort_by(|a, b| b.severity.cmp(&a.severity));
        hints
    }
}

// ============================================================================
// Query Translation
// ============================================================================

/// NeedleQL query builder result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBuildResult {
    /// Generated NeedleQL query
    pub needleql: String,
    /// Parsed query AST (if valid)
    pub parsed: Option<ParsedQueryInfo>,
    /// Query analysis
    pub analysis: QueryAnalysis,
    /// Optimization hints
    pub optimization_hints: Vec<OptimizationHint>,
    /// Quality score (0-1)
    pub quality_score: f32,
    /// Suggestions for improvement
    pub suggestions: Vec<QuerySuggestion>,
    /// Alternative query formulations
    pub alternatives: Vec<AlternativeQuery>,
}

/// Simplified parsed query info for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedQueryInfo {
    pub collection: String,
    pub has_filter: bool,
    pub has_similar_to: bool,
    pub limit: Option<u64>,
}

/// Query improvement suggestion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuerySuggestion {
    /// Suggestion type
    pub suggestion_type: SuggestionType,
    /// Human-readable message
    pub message: String,
    /// Example of improved query
    pub example: Option<String>,
}

/// Types of suggestions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SuggestionType {
    AddFilter,
    RefineSearch,
    UseHybrid,
    AddLimit,
    UseIndex,
    Simplify,
    AddTemporal,
}

/// Alternative query formulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlternativeQuery {
    /// Alternative NeedleQL
    pub needleql: String,
    /// Description of difference
    pub description: String,
    /// Expected quality score
    pub estimated_quality: f32,
}

/// Main query builder
pub struct VisualQueryBuilder {
    profile: CollectionProfile,
    analyzer: QueryAnalyzer,
}

impl VisualQueryBuilder {
    /// Create a new query builder
    pub fn new(profile: CollectionProfile) -> Self {
        Self {
            profile,
            analyzer: QueryAnalyzer::new(),
        }
    }

    /// Build a query from natural language
    pub fn build(&self, natural_query: &str) -> QueryBuildResult {
        // Analyze the query
        let analysis = self.analyzer.analyze(natural_query);

        // Generate NeedleQL
        let needleql = self.translate_to_needleql(natural_query, &analysis);

        // Parse and validate
        let parsed = self.validate_query(&needleql);

        // Generate hints
        let hints = HintGenerator::generate(&analysis, &self.profile);

        // Calculate quality score
        let quality_score = self.calculate_quality(&analysis, &hints);

        // Generate suggestions
        let suggestions = self.generate_suggestions(&analysis, quality_score);

        // Generate alternatives
        let alternatives = self.generate_alternatives(natural_query, &analysis);

        QueryBuildResult {
            needleql,
            parsed,
            analysis,
            optimization_hints: hints,
            quality_score,
            suggestions,
            alternatives,
        }
    }

    /// Translate natural language to NeedleQL
    fn translate_to_needleql(&self, _query: &str, analysis: &QueryAnalysis) -> String {
        let mut parts = Vec::new();

        // SELECT
        parts.push("SELECT *".to_string());

        // FROM
        parts.push(format!("FROM {}", self.profile.name));

        // WHERE clause
        let mut where_parts = Vec::new();

        // Add vector similarity
        if analysis.class != QueryClass::MetadataOnly && analysis.class != QueryClass::Aggregation {
            where_parts.push("vector SIMILAR TO $query".to_string());
        }

        // Add filter conditions from analysis
        for field in &analysis.filter_fields {
            if let Some(field_profile) = self.profile.metadata_fields.iter().find(|f| &f.name == field) {
                let filter_str = match field_profile.field_type {
                    FieldType::String => format!("{} = ${}Filter", field, field),
                    FieldType::Number => format!("{} >= ${}_min", field, field),
                    FieldType::Boolean => format!("{} = true", field),
                    FieldType::DateTime => format!("{} >= ${}Start", field, field),
                    FieldType::Array => format!("{} IN (${}Values)", field, field),
                };
                where_parts.push(filter_str);
            }
        }

        // Add temporal constraint
        if let Some(temporal) = &analysis.temporal {
            if let Some(start) = temporal.start {
                where_parts.push(format!("created_at >= {}", start));
            }
        }

        if !where_parts.is_empty() {
            parts.push(format!("WHERE {}", where_parts.join(" AND ")));
        }

        // LIMIT
        let limit = match analysis.complexity {
            QueryComplexity::Simple => 10,
            QueryComplexity::Moderate => 20,
            QueryComplexity::Complex => 50,
        };
        parts.push(format!("LIMIT {}", limit));

        parts.join(" ")
    }

    /// Validate generated query
    fn validate_query(&self, needleql: &str) -> Option<ParsedQueryInfo> {
        match QueryParser::parse(needleql) {
            Ok(query) => Some(ParsedQueryInfo {
                collection: query.from.collection.clone(),
                has_filter: query.where_clause.is_some(),
                has_similar_to: self.has_similar_to(&query),
                limit: query.limit,
            }),
            Err(_) => None,
        }
    }

    fn has_similar_to(&self, query: &Query) -> bool {
        if let Some(where_clause) = &query.where_clause {
            self.expr_has_similar_to(&where_clause.expression)
        } else {
            false
        }
    }

    fn expr_has_similar_to(&self, expr: &Expression) -> bool {
        match expr {
            Expression::SimilarTo(_) => true,
            Expression::And(l, r) | Expression::Or(l, r) => {
                self.expr_has_similar_to(l) || self.expr_has_similar_to(r)
            }
            Expression::Not(e) | Expression::Grouped(e) => self.expr_has_similar_to(e),
            _ => false,
        }
    }

    /// Calculate query quality score
    fn calculate_quality(&self, analysis: &QueryAnalysis, hints: &[OptimizationHint]) -> f32 {
        let mut score: f32 = 1.0;

        // Penalize for warnings and critical hints
        for hint in hints {
            match hint.severity {
                HintSeverity::Critical => score -= 0.3,
                HintSeverity::Warning => score -= 0.15,
                HintSeverity::Suggestion => score -= 0.05,
                HintSeverity::Info => {}
            }
        }

        // Bonus for clear intent
        if analysis.confidence > 0.8 {
            score += 0.1;
        }

        // Bonus for using filters on large collections
        if self.profile.vector_count > 10_000 && !analysis.filter_fields.is_empty() {
            score += 0.1;
        }

        // Penalize very complex queries
        if analysis.complexity == QueryComplexity::Complex {
            score -= 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Generate improvement suggestions
    fn generate_suggestions(&self, analysis: &QueryAnalysis, quality: f32) -> Vec<QuerySuggestion> {
        let mut suggestions = Vec::new();

        // Suggest filters for large collections
        if analysis.filter_fields.is_empty()
            && self.profile.vector_count > 10_000
            && !self.profile.metadata_fields.is_empty()
        {
            let fields: Vec<_> = self.profile.metadata_fields.iter()
                .take(3)
                .map(|f| f.name.as_str())
                .collect();
            suggestions.push(QuerySuggestion {
                suggestion_type: SuggestionType::AddFilter,
                message: format!("Consider filtering by {} to narrow results", fields.join(", ")),
                example: Some(format!("...with {} = 'value'", fields.first().unwrap_or(&"field"))),
            });
        }

        // Suggest hybrid search for keyword-heavy queries
        if analysis.class == QueryClass::Semantic && analysis.search_terms.len() > 3 {
            suggestions.push(QuerySuggestion {
                suggestion_type: SuggestionType::UseHybrid,
                message: "Multiple search terms detected - hybrid search may improve results".to_string(),
                example: Some("Enable hybrid search to combine vector similarity with BM25".to_string()),
            });
        }

        // Suggest temporal filter
        if analysis.temporal.is_none() && self.profile.metadata_fields.iter().any(|f| f.field_type == FieldType::DateTime) {
            suggestions.push(QuerySuggestion {
                suggestion_type: SuggestionType::AddTemporal,
                message: "Add a time range to find more relevant recent results".to_string(),
                example: Some("...from last week".to_string()),
            });
        }

        // Suggest simplification for complex queries
        if quality < 0.5 && analysis.complexity == QueryComplexity::Complex {
            suggestions.push(QuerySuggestion {
                suggestion_type: SuggestionType::Simplify,
                message: "Query is complex - consider breaking into multiple searches".to_string(),
                example: None,
            });
        }

        suggestions
    }

    /// Generate alternative query formulations
    fn generate_alternatives(&self, _query: &str, analysis: &QueryAnalysis) -> Vec<AlternativeQuery> {
        let mut alternatives = Vec::new();

        // More specific alternative
        if !analysis.filter_fields.is_empty() && analysis.class == QueryClass::Hybrid {
            let filter_only = format!(
                "SELECT * FROM {} WHERE {} LIMIT 100",
                self.profile.name,
                analysis.filter_fields.iter()
                    .map(|f| format!("{} = ${}", f, f))
                    .collect::<Vec<_>>()
                    .join(" AND ")
            );
            alternatives.push(AlternativeQuery {
                needleql: filter_only,
                description: "Filter-first approach: apply metadata filters before vector search".to_string(),
                estimated_quality: 0.7,
            });
        }

        // Broader search alternative
        if analysis.complexity != QueryComplexity::Simple {
            let simple = format!(
                "SELECT * FROM {} WHERE vector SIMILAR TO $query LIMIT 20",
                self.profile.name
            );
            alternatives.push(AlternativeQuery {
                needleql: simple,
                description: "Simpler query: pure vector search without filters".to_string(),
                estimated_quality: 0.6,
            });
        }

        // RAG-optimized alternative
        if analysis.class == QueryClass::Semantic || analysis.class == QueryClass::Hybrid {
            let rag = format!(
                "SELECT * FROM {} USING RAG(top_k=5, rerank=true) WHERE vector SIMILAR TO $query",
                self.profile.name
            );
            alternatives.push(AlternativeQuery {
                needleql: rag,
                description: "RAG-optimized: uses reranking for better context retrieval".to_string(),
                estimated_quality: 0.85,
            });
        }

        alternatives
    }

    /// Get field suggestions for autocomplete
    pub fn suggest_fields(&self, partial: &str) -> Vec<FieldSuggestion> {
        let partial_lower = partial.to_lowercase();

        self.profile.metadata_fields.iter()
            .filter(|f| f.name.to_lowercase().starts_with(&partial_lower))
            .map(|f| FieldSuggestion {
                name: f.name.clone(),
                field_type: f.field_type,
                sample_values: f.sample_values.clone(),
                indexed: f.indexed,
            })
            .collect()
    }

    /// Get value suggestions for a field
    pub fn suggest_values(&self, field: &str) -> Vec<String> {
        self.profile.metadata_fields.iter()
            .find(|f| f.name == field)
            .map(|f| f.sample_values.clone())
            .unwrap_or_default()
    }

    /// Explain a query in human-readable form
    pub fn explain(&self, needleql: &str) -> QueryExplanation {
        let parsed = QueryParser::parse(needleql);

        match parsed {
            Ok(query) => {
                let steps = self.generate_explanation_steps(&query);
                QueryExplanation {
                    valid: true,
                    summary: self.generate_summary(&query),
                    steps,
                    estimated_cost: self.estimate_cost(&query),
                    error: None,
                }
            }
            Err(e) => QueryExplanation {
                valid: false,
                summary: "Invalid query".to_string(),
                steps: Vec::new(),
                estimated_cost: None,
                error: Some(format!("{}", e)),
            },
        }
    }

    fn generate_summary(&self, query: &Query) -> String {
        let mut parts = Vec::new();

        parts.push(format!("Search collection '{}'", query.from.collection));

        if self.has_similar_to(query) {
            parts.push("using vector similarity".to_string());
        }

        if query.where_clause.is_some() {
            parts.push("with metadata filters".to_string());
        }

        if let Some(limit) = query.limit {
            parts.push(format!("returning up to {} results", limit));
        }

        parts.join(", ")
    }

    fn generate_explanation_steps(&self, query: &Query) -> Vec<ExplanationStep> {
        let mut steps = Vec::new();

        // Step 1: Collection access
        steps.push(ExplanationStep {
            step_number: 1,
            operation: "Access Collection".to_string(),
            description: format!("Open collection '{}' with {} vectors", query.from.collection, self.profile.vector_count),
            estimated_rows: Some(self.profile.vector_count),
        });

        // Step 2: Apply pre-filters if any
        if query.where_clause.is_some() {
            steps.push(ExplanationStep {
                step_number: 2,
                operation: "Apply Metadata Filter".to_string(),
                description: "Evaluate filter conditions on metadata".to_string(),
                estimated_rows: Some(self.profile.vector_count / 10), // Estimate 10% selectivity
            });
        }

        // Step 3: Vector search
        if self.has_similar_to(query) {
            steps.push(ExplanationStep {
                step_number: steps.len() + 1,
                operation: "HNSW Search".to_string(),
                description: format!(
                    "Approximate nearest neighbor search (M={}, ef={})",
                    self.profile.index_config.hnsw_m,
                    self.profile.index_config.ef_search
                ),
                estimated_rows: query.limit.map(|l| l as usize),
            });
        }

        // Step 4: Limit results
        if let Some(limit) = query.limit {
            steps.push(ExplanationStep {
                step_number: steps.len() + 1,
                operation: "Limit".to_string(),
                description: format!("Return top {} results", limit),
                estimated_rows: Some(limit as usize),
            });
        }

        steps
    }

    fn estimate_cost(&self, query: &Query) -> Option<CostEstimate> {
        let base_latency = match self.profile.vector_count {
            0..=10_000 => 1.0,
            10_001..=100_000 => 5.0,
            100_001..=1_000_000 => 15.0,
            _ => 50.0,
        };

        let filter_overhead = if query.where_clause.is_some() { 1.2 } else { 1.0 };

        Some(CostEstimate {
            estimated_latency_ms: base_latency * filter_overhead,
            estimated_memory_mb: (self.profile.vector_count * self.profile.dimensions * 4) as f64 / 1_000_000.0,
            scan_type: if self.has_similar_to(query) { "HNSW Index" } else { "Full Scan" }.to_string(),
        })
    }
}

/// Field suggestion for autocomplete
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSuggestion {
    pub name: String,
    pub field_type: FieldType,
    pub sample_values: Vec<String>,
    pub indexed: bool,
}

/// Human-readable query explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExplanation {
    pub valid: bool,
    pub summary: String,
    pub steps: Vec<ExplanationStep>,
    pub estimated_cost: Option<CostEstimate>,
    pub error: Option<String>,
}

/// Explanation step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationStep {
    pub step_number: usize,
    pub operation: String,
    pub description: String,
    pub estimated_rows: Option<usize>,
}

/// Cost estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostEstimate {
    pub estimated_latency_ms: f64,
    pub estimated_memory_mb: f64,
    pub scan_type: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> CollectionProfile {
        CollectionProfile::new("documents", 384, 50_000)
            .with_field(FieldProfile {
                name: "category".to_string(),
                field_type: FieldType::String,
                cardinality: 20,
                indexed: true,
                sample_values: vec!["science".to_string(), "technology".to_string(), "business".to_string()],
            })
            .with_field(FieldProfile {
                name: "score".to_string(),
                field_type: FieldType::Number,
                cardinality: 100,
                indexed: false,
                sample_values: vec!["0.5".to_string(), "0.8".to_string(), "1.0".to_string()],
            })
            .with_field(FieldProfile {
                name: "published_at".to_string(),
                field_type: FieldType::DateTime,
                cardinality: 1000,
                indexed: true,
                sample_values: vec![],
            })
    }

    #[test]
    fn test_query_analyzer_simple() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("find articles about machine learning");

        assert_eq!(analysis.class, QueryClass::Semantic);
        assert_eq!(analysis.complexity, QueryComplexity::Simple);
        assert!(!analysis.search_terms.is_empty());
    }

    #[test]
    fn test_query_analyzer_with_filter() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("show documents category technology with score greater than 0.8");

        assert_eq!(analysis.class, QueryClass::Hybrid);
        assert!(!analysis.filter_fields.is_empty());
    }

    #[test]
    fn test_query_analyzer_temporal() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("articles from last week");

        assert!(analysis.temporal.is_some());
        assert!(analysis.patterns.iter().any(|p| p.pattern_type == PatternType::TemporalExpression));
    }

    #[test]
    fn test_query_analyzer_aggregation() {
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("how many documents are in the database");

        assert_eq!(analysis.class, QueryClass::Aggregation);
    }

    #[test]
    fn test_query_builder_simple() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let result = builder.build("find articles about machine learning");

        assert!(!result.needleql.is_empty());
        assert!(result.needleql.contains("SELECT"));
        assert!(result.needleql.contains("FROM documents"));
        assert!(result.needleql.contains("SIMILAR TO"));
        assert!(result.quality_score > 0.0);
    }

    #[test]
    fn test_query_builder_with_filter() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let result = builder.build("show documents category technology");

        assert!(result.needleql.contains("WHERE"));
        assert!(!result.optimization_hints.is_empty() || result.quality_score > 0.5);
    }

    #[test]
    fn test_hint_generation() {
        let profile = CollectionProfile::new("large_collection", 384, 1_000_000);
        let analyzer = QueryAnalyzer::new();
        let analysis = analyzer.analyze("find something");

        let hints = HintGenerator::generate(&analysis, &profile);

        // Should suggest quantization for large collection
        assert!(hints.iter().any(|h| h.category == HintCategory::Quantization));
        // Should suggest filters for large collection
        assert!(hints.iter().any(|h| h.category == HintCategory::FilterOrder));
    }

    #[test]
    fn test_hint_for_unindexed_field() {
        let profile = CollectionProfile::new("test", 384, 10_000)
            .with_field(FieldProfile {
                name: "status".to_string(),
                field_type: FieldType::String,
                cardinality: 500,
                indexed: false,
                sample_values: vec![],
            });

        let analysis = QueryAnalysis {
            class: QueryClass::Hybrid,
            complexity: QueryComplexity::Simple,
            search_terms: vec!["test".to_string()],
            filter_fields: vec!["status".to_string()],
            temporal: None,
            confidence: 0.9,
            patterns: vec![],
        };

        let hints = HintGenerator::generate(&analysis, &profile);

        // Should warn about unindexed high-cardinality field
        assert!(hints.iter().any(|h| {
            h.category == HintCategory::IndexUsage && h.severity == HintSeverity::Warning
        }));
    }

    #[test]
    fn test_field_suggestions() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let suggestions = builder.suggest_fields("cat");
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].name, "category");
    }

    #[test]
    fn test_value_suggestions() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let values = builder.suggest_values("category");
        assert!(values.contains(&"science".to_string()));
        assert!(values.contains(&"technology".to_string()));
    }

    #[test]
    fn test_query_explanation() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let explanation = builder.explain("SELECT * FROM documents WHERE vector SIMILAR TO $query LIMIT 10");

        assert!(explanation.valid);
        assert!(!explanation.summary.is_empty());
        assert!(!explanation.steps.is_empty());
        assert!(explanation.estimated_cost.is_some());
    }

    #[test]
    fn test_query_explanation_invalid() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let explanation = builder.explain("INVALID QUERY SYNTAX");

        assert!(!explanation.valid);
        assert!(explanation.error.is_some());
    }

    #[test]
    fn test_alternative_generation() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        let result = builder.build("find technology articles about AI");

        // Should generate at least one alternative
        assert!(!result.alternatives.is_empty());

        // RAG alternative should be present
        assert!(result.alternatives.iter().any(|a| a.needleql.contains("RAG")));
    }

    #[test]
    fn test_quality_score_ranges() {
        let profile = test_profile();
        let builder = VisualQueryBuilder::new(profile);

        // Simple query should have good quality
        let simple = builder.build("find documents");
        assert!(simple.quality_score >= 0.5);

        // Query with filters should have good quality
        let filtered = builder.build("show category science documents");
        assert!(filtered.quality_score >= 0.5);
    }

    #[test]
    fn test_pattern_detection() {
        let analyzer = QueryAnalyzer::new();

        let analysis = analyzer.analyze("find documents with score greater than 0.8 from last week");

        assert!(analysis.patterns.iter().any(|p| p.pattern_type == PatternType::SearchKeyword));
        assert!(analysis.patterns.iter().any(|p| p.pattern_type == PatternType::Comparison));
        assert!(analysis.patterns.iter().any(|p| p.pattern_type == PatternType::TemporalExpression));
    }

    #[test]
    fn test_complexity_assessment() {
        let analyzer = QueryAnalyzer::new();

        let simple = analyzer.analyze("find documents");
        assert_eq!(simple.complexity, QueryComplexity::Simple);

        let moderate = analyzer.analyze("find technology articles from last week with score > 0.5");
        assert!(moderate.complexity != QueryComplexity::Simple);

        let complex = analyzer.analyze("find articles not in category sports or entertainment with score between 0.5 and 1.0 from last month sorted by date");
        assert_eq!(complex.complexity, QueryComplexity::Complex);
    }

    #[test]
    fn test_suggestion_generation() {
        let profile = CollectionProfile::new("large", 384, 100_000)
            .with_field(FieldProfile {
                name: "category".to_string(),
                field_type: FieldType::String,
                cardinality: 10,
                indexed: true,
                sample_values: vec![],
            });

        let builder = VisualQueryBuilder::new(profile);
        let result = builder.build("find something");

        // Should suggest adding filters for large collection
        assert!(result.suggestions.iter().any(|s| s.suggestion_type == SuggestionType::AddFilter));
    }

    #[test]
    fn test_collection_profile_builder() {
        let profile = CollectionProfile::new("test", 256, 1000)
            .with_field(FieldProfile {
                name: "field1".to_string(),
                field_type: FieldType::String,
                cardinality: 10,
                indexed: true,
                sample_values: vec![],
            })
            .with_index(IndexProfile {
                hnsw_m: 32,
                ef_construction: 400,
                ef_search: 100,
                distance: "euclidean".to_string(),
                quantization: Some("scalar".to_string()),
            });

        assert_eq!(profile.name, "test");
        assert_eq!(profile.dimensions, 256);
        assert_eq!(profile.metadata_fields.len(), 1);
        assert_eq!(profile.index_config.hnsw_m, 32);
    }
}
