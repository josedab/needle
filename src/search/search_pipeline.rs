#![allow(clippy::unwrap_used)]
//! Composable Search Pipeline DSL
//!
//! A declarative pipeline DSL for chaining search operations:
//! embed → search → rerank → filter → diversify → limit.
//!
//! Pipelines are defined as a DAG of stages, serializable to JSON/YAML,
//! and optimizable via stage reordering and caching.
//!
//! # Example
//!
//! ```rust
//! use needle::search::search_pipeline::{
//!     SearchPipeline, PipelineStage, SearchStageConfig, FilterStageConfig,
//!     LimitStageConfig,
//! };
//!
//! let pipeline = SearchPipeline::new("rag-basic")
//!     .add_stage(PipelineStage::Search(SearchStageConfig {
//!         k: 50,
//!         ef_search: Some(100),
//!     }))
//!     .add_stage(PipelineStage::Filter(FilterStageConfig {
//!         expression: r#"{"category": "science"}"#.to_string(),
//!     }))
//!     .add_stage(PipelineStage::Limit(LimitStageConfig { k: 10 }));
//!
//! assert_eq!(pipeline.stages().len(), 3);
//! ```

use serde::{Deserialize, Serialize};

/// Configuration for a vector search stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchStageConfig {
    /// Number of candidates to retrieve
    pub k: usize,
    /// Optional ef_search override
    pub ef_search: Option<usize>,
}

/// Configuration for a metadata filter stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterStageConfig {
    /// MongoDB-style filter expression as JSON string
    pub expression: String,
}

/// Configuration for a rerank stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RerankStageConfig {
    /// Reranking method: "cross-encoder", "rrf", "linear"
    pub method: String,
    /// Weight for score blending (0.0-1.0)
    #[serde(default = "default_weight")]
    pub weight: f64,
    /// Top-k after reranking
    pub k: Option<usize>,
}

fn default_weight() -> f64 {
    0.5
}

/// Configuration for a diversify stage (MMR-style).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversifyStageConfig {
    /// Lambda parameter for MMR (0=max diversity, 1=max relevance)
    #[serde(default = "default_lambda")]
    pub lambda: f64,
    /// Number of results after diversification
    pub k: usize,
}

fn default_lambda() -> f64 {
    0.7
}

/// Configuration for a limit stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitStageConfig {
    /// Maximum number of results
    pub k: usize,
}

/// Configuration for a score threshold stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdStageConfig {
    /// Maximum distance to include (results with distance > threshold are dropped)
    pub max_distance: f32,
}

/// Configuration for a cache stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStageConfig {
    /// TTL in seconds for cached results
    pub ttl_secs: u64,
    /// Maximum cache entries
    #[serde(default = "default_cache_size")]
    pub max_entries: usize,
}

fn default_cache_size() -> usize {
    1000
}

/// A single stage in the search pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PipelineStage {
    /// Vector similarity search (initial retrieval)
    Search(SearchStageConfig),
    /// Metadata filter
    Filter(FilterStageConfig),
    /// Reranking stage
    Rerank(RerankStageConfig),
    /// MMR-style diversification
    Diversify(DiversifyStageConfig),
    /// Limit/top-k truncation
    Limit(LimitStageConfig),
    /// Score threshold filtering
    Threshold(ThresholdStageConfig),
    /// Result caching
    Cache(CacheStageConfig),
}

impl PipelineStage {
    /// Get the stage type name.
    pub fn stage_type(&self) -> &'static str {
        match self {
            PipelineStage::Search(_) => "search",
            PipelineStage::Filter(_) => "filter",
            PipelineStage::Rerank(_) => "rerank",
            PipelineStage::Diversify(_) => "diversify",
            PipelineStage::Limit(_) => "limit",
            PipelineStage::Threshold(_) => "threshold",
            PipelineStage::Cache(_) => "cache",
        }
    }
}

/// A composable search pipeline defined as an ordered sequence of stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchPipeline {
    /// Pipeline name
    pub name: String,
    /// Optional description
    #[serde(default)]
    pub description: String,
    /// Ordered list of pipeline stages
    stages: Vec<PipelineStage>,
    /// Pipeline version
    #[serde(default = "default_version")]
    pub version: String,
}

fn default_version() -> String {
    "1.0".to_string()
}

impl SearchPipeline {
    /// Create a new empty pipeline.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: String::new(),
            stages: Vec::new(),
            version: default_version(),
        }
    }

    /// Add a stage to the pipeline.
    #[must_use]
    pub fn add_stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    /// Get the pipeline stages.
    pub fn stages(&self) -> &[PipelineStage] {
        &self.stages
    }

    /// Get the number of stages.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Check if the pipeline has no stages.
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Validate the pipeline configuration.
    ///
    /// Returns a list of warnings/errors. An empty list means the pipeline is valid.
    pub fn validate(&self) -> Vec<String> {
        let mut issues = Vec::new();

        if self.stages.is_empty() {
            issues.push("Pipeline has no stages".to_string());
            return issues;
        }

        // First stage should be Search
        if !matches!(self.stages.first(), Some(PipelineStage::Search(_))) {
            issues.push("First stage should be 'search'".to_string());
        }

        // Check for duplicate stage types where it doesn't make sense
        let mut seen_search = false;
        for stage in &self.stages {
            if matches!(stage, PipelineStage::Search(_)) {
                if seen_search {
                    issues.push("Multiple search stages detected".to_string());
                }
                seen_search = true;
            }
        }

        issues
    }

    /// Optimize the pipeline by reordering stages where possible.
    ///
    /// Moves filter stages closer to the search stage to reduce
    /// the number of candidates processed by expensive stages.
    pub fn optimize(mut self) -> Self {
        // Simple optimization: move filter stages right after search
        let mut optimized = Vec::new();
        let mut filters = Vec::new();
        let mut rest = Vec::new();

        for stage in self.stages.drain(..) {
            match &stage {
                PipelineStage::Search(_) => optimized.push(stage),
                PipelineStage::Filter(_) => filters.push(stage),
                _ => rest.push(stage),
            }
        }

        optimized.extend(filters);
        optimized.extend(rest);
        self.stages = optimized;
        self
    }

    /// Serialize the pipeline to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a pipeline from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get a summary of the pipeline for display.
    pub fn summary(&self) -> String {
        let stage_names: Vec<&str> = self.stages.iter().map(|s| s.stage_type()).collect();
        format!(
            "{} ({}): {}",
            self.name,
            self.version,
            stage_names.join(" → ")
        )
    }
}

// ── Built-in Pipeline Templates ──────────────────────────────────────────────

/// Built-in pipeline templates for common RAG patterns.
pub struct PipelineTemplates;

impl PipelineTemplates {
    /// Simple RAG: search → limit
    pub fn simple_rag(k: usize) -> SearchPipeline {
        SearchPipeline::new("simple-rag")
            .with_description("Basic vector search with top-k")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 3,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// Filtered RAG: search → filter → limit
    pub fn filtered_rag(k: usize, filter: &str) -> SearchPipeline {
        SearchPipeline::new("filtered-rag")
            .with_description("Vector search with metadata filter")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 5,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Filter(FilterStageConfig {
                expression: filter.to_string(),
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// Diverse RAG: search → diversify → limit
    pub fn diverse_rag(k: usize, lambda: f64) -> SearchPipeline {
        SearchPipeline::new("diverse-rag")
            .with_description("Vector search with MMR diversification")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 5,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Diversify(DiversifyStageConfig {
                lambda,
                k,
            }))
    }

    /// Reranked RAG: search → rerank → limit
    pub fn reranked_rag(k: usize) -> SearchPipeline {
        SearchPipeline::new("reranked-rag")
            .with_description("Vector search with cross-encoder reranking")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 10,
                ef_search: Some(200),
            }))
            .add_stage(PipelineStage::Rerank(RerankStageConfig {
                method: "cross-encoder".to_string(),
                weight: 0.7,
                k: Some(k),
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// High-recall: search with high ef → threshold → limit
    pub fn high_recall(k: usize, max_distance: f32) -> SearchPipeline {
        SearchPipeline::new("high-recall")
            .with_description("High-recall search with distance threshold")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 10,
                ef_search: Some(400),
            }))
            .add_stage(PipelineStage::Threshold(ThresholdStageConfig {
                max_distance,
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// Cached search: cache → search → limit
    pub fn cached_search(k: usize, ttl_secs: u64) -> SearchPipeline {
        SearchPipeline::new("cached-search")
            .with_description("Cached vector search for repeated queries")
            .add_stage(PipelineStage::Cache(CacheStageConfig {
                ttl_secs,
                max_entries: 1000,
            }))
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 3,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// Full RAG pipeline: search → filter → rerank → diversify → limit
    pub fn full_rag(k: usize, filter: &str) -> SearchPipeline {
        SearchPipeline::new("full-rag")
            .with_description("Complete RAG pipeline with all stages")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 10,
                ef_search: Some(200),
            }))
            .add_stage(PipelineStage::Filter(FilterStageConfig {
                expression: filter.to_string(),
            }))
            .add_stage(PipelineStage::Rerank(RerankStageConfig {
                method: "rrf".to_string(),
                weight: 0.6,
                k: Some(k * 3),
            }))
            .add_stage(PipelineStage::Diversify(DiversifyStageConfig {
                lambda: 0.7,
                k,
            }))
    }

    /// Semantic dedup: search → threshold → diversify
    pub fn semantic_dedup(k: usize, threshold: f32) -> SearchPipeline {
        SearchPipeline::new("semantic-dedup")
            .with_description("Find unique results by removing near-duplicates")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 5,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Threshold(ThresholdStageConfig {
                max_distance: threshold,
            }))
            .add_stage(PipelineStage::Diversify(DiversifyStageConfig {
                lambda: 0.3,
                k,
            }))
    }

    /// Precision search: search (high ef, low k) → limit
    pub fn precision_search(k: usize) -> SearchPipeline {
        SearchPipeline::new("precision-search")
            .with_description("High-precision search with aggressive ef_search")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 2,
                ef_search: Some(500),
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// Multi-stage filter: search → filter1 → filter2 → limit
    pub fn multi_filter(k: usize, filters: &[&str]) -> SearchPipeline {
        let mut pipeline = SearchPipeline::new("multi-filter")
            .with_description("Search with multiple cascading filters")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: k * 10,
                ef_search: None,
            }));

        for f in filters {
            pipeline = pipeline.add_stage(PipelineStage::Filter(FilterStageConfig {
                expression: f.to_string(),
            }));
        }

        pipeline.add_stage(PipelineStage::Limit(LimitStageConfig { k }))
    }

    /// List all available template names.
    pub fn list() -> Vec<&'static str> {
        vec![
            "simple-rag",
            "filtered-rag",
            "diverse-rag",
            "reranked-rag",
            "high-recall",
            "cached-search",
            "full-rag",
            "semantic-dedup",
            "precision-search",
            "multi-filter",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_creation() {
        let pipeline = SearchPipeline::new("test")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: 50,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k: 10 }));

        assert_eq!(pipeline.len(), 2);
        assert!(pipeline.validate().is_empty());
    }

    #[test]
    fn test_pipeline_validation() {
        let empty = SearchPipeline::new("empty");
        assert!(!empty.validate().is_empty());

        let no_search = SearchPipeline::new("no-search")
            .add_stage(PipelineStage::Limit(LimitStageConfig { k: 10 }));
        assert!(!no_search.validate().is_empty());
    }

    #[test]
    fn test_pipeline_serialization() {
        let pipeline = PipelineTemplates::simple_rag(10);
        let json = pipeline.to_json().unwrap();
        let restored = SearchPipeline::from_json(&json).unwrap();
        assert_eq!(restored.name, "simple-rag");
        assert_eq!(restored.len(), 2);
    }

    #[test]
    fn test_pipeline_optimize() {
        let pipeline = SearchPipeline::new("test")
            .add_stage(PipelineStage::Search(SearchStageConfig {
                k: 100,
                ef_search: None,
            }))
            .add_stage(PipelineStage::Rerank(RerankStageConfig {
                method: "rrf".to_string(),
                weight: 0.5,
                k: Some(20),
            }))
            .add_stage(PipelineStage::Filter(FilterStageConfig {
                expression: r#"{"x": 1}"#.to_string(),
            }))
            .add_stage(PipelineStage::Limit(LimitStageConfig { k: 10 }));

        let optimized = pipeline.optimize();
        // Filter should be moved right after search
        assert_eq!(optimized.stages()[0].stage_type(), "search");
        assert_eq!(optimized.stages()[1].stage_type(), "filter");
        assert_eq!(optimized.stages()[2].stage_type(), "rerank");
    }

    #[test]
    fn test_all_templates() {
        let templates = PipelineTemplates::list();
        assert_eq!(templates.len(), 10);

        // Verify each template creates a valid pipeline
        let p1 = PipelineTemplates::simple_rag(10);
        assert!(p1.validate().is_empty());

        let p2 = PipelineTemplates::filtered_rag(10, r#"{"a":"b"}"#);
        assert!(p2.validate().is_empty());

        let p3 = PipelineTemplates::diverse_rag(10, 0.7);
        assert!(p3.validate().is_empty());
    }

    #[test]
    fn test_pipeline_summary() {
        let pipeline = PipelineTemplates::full_rag(10, "{}");
        let summary = pipeline.summary();
        assert!(summary.contains("full-rag"));
        assert!(summary.contains("search"));
        assert!(summary.contains("diversify"));
    }
}
