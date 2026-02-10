// Allow dead_code for this public API module - types are exported for library users
#![allow(dead_code)]
#![allow(clippy::wrong_self_convention)]

//! Data Lineage - Track the origin and transformations of vector embeddings.
//!
//! Provides complete provenance tracking for vectors, including their source data,
//! embedding model, transformations, and derived outputs.
//!
//! # Features
//!
//! - **Source tracking**: Link vectors to original documents/data
//! - **Model versioning**: Track which embedding model created each vector
//! - **Transformation history**: Record all operations applied to vectors
//! - **Dependency graph**: Visualize relationships between vectors
//! - **Impact analysis**: Find all vectors affected by a source change
//!
//! # Example
//!
//! ```ignore
//! use needle::lineage::{LineageTracker, VectorLineage, SourceInfo};
//!
//! let mut tracker = LineageTracker::new();
//!
//! // Register a vector with its lineage
//! tracker.register("vec1", VectorLineage {
//!     source: SourceInfo::Document { id: "doc1", chunk: 0 },
//!     model: "text-embedding-3-small",
//!     model_version: "1.0.0",
//!     ..Default::default()
//! })?;
//!
//! // Track a transformation
//! tracker.add_transformation("vec1", Transformation::Normalize)?;
//!
//! // Find all vectors from a source
//! let derived = tracker.find_by_source("doc1");
//! ```

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Information about the source of a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceInfo {
    /// Vector from a document chunk.
    Document {
        /// Document ID.
        document_id: String,
        /// Chunk index within document.
        chunk_index: usize,
        /// Character offset start.
        char_start: Option<usize>,
        /// Character offset end.
        char_end: Option<usize>,
    },
    /// Vector from an image.
    Image {
        /// Image file ID.
        image_id: String,
        /// Region of interest (x, y, width, height).
        region: Option<(u32, u32, u32, u32)>,
    },
    /// Vector from audio.
    Audio {
        /// Audio file ID.
        audio_id: String,
        /// Start time in seconds.
        start_time: f64,
        /// End time in seconds.
        end_time: f64,
    },
    /// Vector synthesized from other vectors.
    Derived {
        /// Source vector IDs.
        source_vectors: Vec<String>,
        /// Method used.
        method: String,
    },
    /// External data source.
    External {
        /// Source system name.
        system: String,
        /// External ID.
        external_id: String,
        /// URL if applicable.
        url: Option<String>,
    },
    /// User input.
    UserInput {
        /// User ID.
        user_id: Option<String>,
        /// Session ID.
        session_id: Option<String>,
    },
    /// Unknown source.
    Unknown,
}

/// A transformation applied to a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Transformation {
    /// Created from source.
    Created,
    /// Normalized to unit length.
    Normalize,
    /// Dimensionality reduction.
    DimensionReduction {
        /// Original dimensions.
        from_dims: usize,
        /// Reduced dimensions.
        to_dims: usize,
        /// Method used.
        method: String,
    },
    /// Quantization.
    Quantize {
        /// Quantization type.
        quant_type: String,
        /// Bits per dimension.
        bits: usize,
    },
    /// Combined with other vectors.
    Merge {
        /// Vector IDs merged.
        vector_ids: Vec<String>,
        /// Merge method.
        method: String,
    },
    /// Re-embedded with different model.
    ReEmbed {
        /// Previous model.
        old_model: String,
        /// New model.
        new_model: String,
    },
    /// Updated metadata.
    MetadataUpdate {
        /// Fields changed.
        fields: Vec<String>,
    },
    /// Custom transformation.
    Custom {
        /// Transformation name.
        name: String,
        /// Parameters.
        params: HashMap<String, String>,
    },
}

/// Record of a transformation event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformationEvent {
    /// Transformation applied.
    pub transformation: Transformation,
    /// Timestamp.
    pub timestamp: u64,
    /// User who performed.
    pub user: Option<String>,
    /// Additional notes.
    pub notes: Option<String>,
}

/// Complete lineage information for a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLineage {
    /// Vector ID.
    pub vector_id: String,
    /// Source information.
    pub source: SourceInfo,
    /// Embedding model used.
    pub model: String,
    /// Model version.
    pub model_version: String,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last update timestamp.
    pub updated_at: u64,
    /// Transformation history.
    pub transformations: Vec<TransformationEvent>,
    /// Parent vectors (if derived).
    pub parents: Vec<String>,
    /// Child vectors (derived from this).
    pub children: Vec<String>,
    /// Custom tags.
    pub tags: HashSet<String>,
    /// Quality score (0-1).
    pub quality_score: Option<f32>,
    /// Confidence in source attribution.
    pub confidence: f32,
}

impl Default for VectorLineage {
    fn default() -> Self {
        Self {
            vector_id: String::new(),
            source: SourceInfo::Unknown,
            model: String::new(),
            model_version: String::new(),
            created_at: Self::now(),
            updated_at: Self::now(),
            transformations: Vec::new(),
            parents: Vec::new(),
            children: Vec::new(),
            tags: HashSet::new(),
            quality_score: None,
            confidence: 1.0,
        }
    }
}

impl VectorLineage {
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }
}

/// Lineage tracker for vectors.
pub struct LineageTracker {
    /// Lineage records by vector ID.
    lineages: HashMap<String, VectorLineage>,
    /// Index: source ID -> vector IDs.
    source_index: HashMap<String, HashSet<String>>,
    /// Index: model -> vector IDs.
    model_index: HashMap<String, HashSet<String>>,
    /// Index: tag -> vector IDs.
    tag_index: HashMap<String, HashSet<String>>,
}

impl LineageTracker {
    /// Create a new lineage tracker.
    pub fn new() -> Self {
        Self {
            lineages: HashMap::new(),
            source_index: HashMap::new(),
            model_index: HashMap::new(),
            tag_index: HashMap::new(),
        }
    }

    /// Register a vector with its lineage.
    pub fn register(&mut self, vector_id: &str, mut lineage: VectorLineage) -> Result<()> {
        if self.lineages.contains_key(vector_id) {
            return Err(NeedleError::InvalidInput(format!(
                "Vector '{}' already registered",
                vector_id
            )));
        }

        lineage.vector_id = vector_id.to_string();

        // Add creation event
        if lineage.transformations.is_empty() {
            lineage.transformations.push(TransformationEvent {
                transformation: Transformation::Created,
                timestamp: lineage.created_at,
                user: None,
                notes: None,
            });
        }

        // Update indices
        if let Some(source_id) = self.extract_source_id(&lineage.source) {
            self.source_index
                .entry(source_id)
                .or_default()
                .insert(vector_id.to_string());
        }

        self.model_index
            .entry(lineage.model.clone())
            .or_default()
            .insert(vector_id.to_string());

        for tag in &lineage.tags {
            self.tag_index
                .entry(tag.clone())
                .or_default()
                .insert(vector_id.to_string());
        }

        // Update parent-child relationships
        for parent_id in &lineage.parents {
            if let Some(parent) = self.lineages.get_mut(parent_id) {
                parent.children.push(vector_id.to_string());
            }
        }

        self.lineages.insert(vector_id.to_string(), lineage);
        Ok(())
    }

    /// Extract source ID from source info.
    fn extract_source_id(&self, source: &SourceInfo) -> Option<String> {
        match source {
            SourceInfo::Document { document_id, .. } => Some(document_id.clone()),
            SourceInfo::Image { image_id, .. } => Some(image_id.clone()),
            SourceInfo::Audio { audio_id, .. } => Some(audio_id.clone()),
            SourceInfo::External { external_id, .. } => Some(external_id.clone()),
            _ => None,
        }
    }

    /// Get lineage for a vector.
    pub fn get(&self, vector_id: &str) -> Option<&VectorLineage> {
        self.lineages.get(vector_id)
    }

    /// Add a transformation to a vector's history.
    pub fn add_transformation(
        &mut self,
        vector_id: &str,
        transformation: Transformation,
    ) -> Result<()> {
        self.add_transformation_with_details(vector_id, transformation, None, None)
    }

    /// Add a transformation with details.
    pub fn add_transformation_with_details(
        &mut self,
        vector_id: &str,
        transformation: Transformation,
        user: Option<&str>,
        notes: Option<&str>,
    ) -> Result<()> {
        let lineage = self.lineages.get_mut(vector_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", vector_id)))?;

        lineage.transformations.push(TransformationEvent {
            transformation,
            timestamp: VectorLineage::now(),
            user: user.map(|s| s.to_string()),
            notes: notes.map(|s| s.to_string()),
        });

        lineage.updated_at = VectorLineage::now();
        Ok(())
    }

    /// Add a tag to a vector.
    pub fn add_tag(&mut self, vector_id: &str, tag: &str) -> Result<()> {
        let lineage = self.lineages.get_mut(vector_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", vector_id)))?;

        lineage.tags.insert(tag.to_string());
        lineage.updated_at = VectorLineage::now();

        self.tag_index
            .entry(tag.to_string())
            .or_default()
            .insert(vector_id.to_string());

        Ok(())
    }

    /// Remove a tag from a vector.
    pub fn remove_tag(&mut self, vector_id: &str, tag: &str) -> Result<()> {
        let lineage = self.lineages.get_mut(vector_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", vector_id)))?;

        lineage.tags.remove(tag);
        lineage.updated_at = VectorLineage::now();

        if let Some(vectors) = self.tag_index.get_mut(tag) {
            vectors.remove(vector_id);
        }

        Ok(())
    }

    /// Set quality score for a vector.
    pub fn set_quality_score(&mut self, vector_id: &str, score: f32) -> Result<()> {
        let lineage = self.lineages.get_mut(vector_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", vector_id)))?;

        lineage.quality_score = Some(score.clamp(0.0, 1.0));
        lineage.updated_at = VectorLineage::now();
        Ok(())
    }

    /// Find vectors by source.
    pub fn find_by_source(&self, source_id: &str) -> Vec<&VectorLineage> {
        self.source_index
            .get(source_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.lineages.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find vectors by model.
    pub fn find_by_model(&self, model: &str) -> Vec<&VectorLineage> {
        self.model_index
            .get(model)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.lineages.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find vectors by tag.
    pub fn find_by_tag(&self, tag: &str) -> Vec<&VectorLineage> {
        self.tag_index
            .get(tag)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.lineages.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all ancestors of a vector.
    pub fn get_ancestors(&self, vector_id: &str) -> Vec<&VectorLineage> {
        let mut ancestors = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(lineage) = self.lineages.get(vector_id) {
            for parent in &lineage.parents {
                queue.push_back(parent.clone());
            }
        }

        while let Some(id) = queue.pop_front() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id.clone());

            if let Some(lineage) = self.lineages.get(&id) {
                ancestors.push(lineage);
                for parent in &lineage.parents {
                    queue.push_back(parent.clone());
                }
            }
        }

        ancestors
    }

    /// Get all descendants of a vector.
    pub fn get_descendants(&self, vector_id: &str) -> Vec<&VectorLineage> {
        let mut descendants = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if let Some(lineage) = self.lineages.get(vector_id) {
            for child in &lineage.children {
                queue.push_back(child.clone());
            }
        }

        while let Some(id) = queue.pop_front() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id.clone());

            if let Some(lineage) = self.lineages.get(&id) {
                descendants.push(lineage);
                for child in &lineage.children {
                    queue.push_back(child.clone());
                }
            }
        }

        descendants
    }

    /// Perform impact analysis for a source change.
    pub fn impact_analysis(&self, source_id: &str) -> ImpactReport {
        let direct = self.find_by_source(source_id);
        let direct_ids: HashSet<String> = direct.iter().map(|l| l.vector_id.clone()).collect();

        let mut indirect = Vec::new();
        let mut indirect_ids = HashSet::new();

        for lineage in &direct {
            let descendants = self.get_descendants(&lineage.vector_id);
            for desc in descendants {
                if !direct_ids.contains(&desc.vector_id) && !indirect_ids.contains(&desc.vector_id) {
                    indirect_ids.insert(desc.vector_id.clone());
                    indirect.push(desc.vector_id.clone());
                }
            }
        }

        ImpactReport {
            source_id: source_id.to_string(),
            direct_vectors: direct.iter().map(|l| l.vector_id.clone()).collect(),
            indirect_vectors: indirect,
            total_affected: direct.len() + indirect_ids.len(),
        }
    }

    /// Export lineage as a dependency graph.
    pub fn export_graph(&self) -> LineageGraph {
        let nodes: Vec<GraphNode> = self.lineages.values()
            .map(|l| GraphNode {
                id: l.vector_id.clone(),
                source_type: self.source_type_name(&l.source),
                model: l.model.clone(),
                transformation_count: l.transformations.len(),
            })
            .collect();

        let mut edges = Vec::new();
        for lineage in self.lineages.values() {
            for parent in &lineage.parents {
                edges.push(GraphEdge {
                    from: parent.clone(),
                    to: lineage.vector_id.clone(),
                    edge_type: "derived_from".to_string(),
                });
            }
        }

        LineageGraph { nodes, edges }
    }

    /// Get source type name.
    fn source_type_name(&self, source: &SourceInfo) -> String {
        match source {
            SourceInfo::Document { .. } => "document".to_string(),
            SourceInfo::Image { .. } => "image".to_string(),
            SourceInfo::Audio { .. } => "audio".to_string(),
            SourceInfo::Derived { .. } => "derived".to_string(),
            SourceInfo::External { .. } => "external".to_string(),
            SourceInfo::UserInput { .. } => "user_input".to_string(),
            SourceInfo::Unknown => "unknown".to_string(),
        }
    }

    /// Delete lineage for a vector.
    pub fn delete(&mut self, vector_id: &str) -> Result<()> {
        let lineage = self.lineages.remove(vector_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Vector '{}' not found", vector_id)))?;

        // Update indices
        if let Some(source_id) = self.extract_source_id(&lineage.source) {
            if let Some(vectors) = self.source_index.get_mut(&source_id) {
                vectors.remove(vector_id);
            }
        }

        if let Some(vectors) = self.model_index.get_mut(&lineage.model) {
            vectors.remove(vector_id);
        }

        for tag in &lineage.tags {
            if let Some(vectors) = self.tag_index.get_mut(tag) {
                vectors.remove(vector_id);
            }
        }

        // Update parent references
        for parent_id in &lineage.parents {
            if let Some(parent) = self.lineages.get_mut(parent_id) {
                parent.children.retain(|c| c != vector_id);
            }
        }

        // Update child references
        for child_id in &lineage.children {
            if let Some(child) = self.lineages.get_mut(child_id) {
                child.parents.retain(|p| p != vector_id);
            }
        }

        Ok(())
    }

    /// Get statistics about tracked lineages.
    pub fn stats(&self) -> LineageStats {
        let mut source_counts: HashMap<String, usize> = HashMap::new();
        let mut model_counts: HashMap<String, usize> = HashMap::new();
        let mut total_transformations = 0;

        for lineage in self.lineages.values() {
            let source_type = self.source_type_name(&lineage.source);
            *source_counts.entry(source_type).or_default() += 1;
            *model_counts.entry(lineage.model.clone()).or_default() += 1;
            total_transformations += lineage.transformations.len();
        }

        LineageStats {
            total_vectors: self.lineages.len(),
            vectors_by_source: source_counts,
            vectors_by_model: model_counts,
            total_transformations,
            unique_sources: self.source_index.len(),
            unique_models: self.model_index.len(),
            unique_tags: self.tag_index.len(),
        }
    }

    /// Get total count of tracked vectors.
    pub fn len(&self) -> usize {
        self.lineages.len()
    }

    /// Check if tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.lineages.is_empty()
    }
}

impl Default for LineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Impact analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactReport {
    /// Source ID analyzed.
    pub source_id: String,
    /// Directly affected vector IDs.
    pub direct_vectors: Vec<String>,
    /// Indirectly affected vector IDs.
    pub indirect_vectors: Vec<String>,
    /// Total affected count.
    pub total_affected: usize,
}

/// Graph representation of lineage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageGraph {
    /// Graph nodes.
    pub nodes: Vec<GraphNode>,
    /// Graph edges.
    pub edges: Vec<GraphEdge>,
}

/// A node in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Vector ID.
    pub id: String,
    /// Source type.
    pub source_type: String,
    /// Embedding model.
    pub model: String,
    /// Number of transformations.
    pub transformation_count: usize,
}

/// An edge in the lineage graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID.
    pub from: String,
    /// Target node ID.
    pub to: String,
    /// Edge type.
    pub edge_type: String,
}

/// Statistics about lineage tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageStats {
    /// Total tracked vectors.
    pub total_vectors: usize,
    /// Vectors by source type.
    pub vectors_by_source: HashMap<String, usize>,
    /// Vectors by model.
    pub vectors_by_model: HashMap<String, usize>,
    /// Total transformation events.
    pub total_transformations: usize,
    /// Unique source IDs.
    pub unique_sources: usize,
    /// Unique models.
    pub unique_models: usize,
    /// Unique tags.
    pub unique_tags: usize,
}

/// Builder for VectorLineage.
pub struct LineageBuilder {
    lineage: VectorLineage,
}

impl LineageBuilder {
    /// Create new builder.
    pub fn new(vector_id: &str) -> Self {
        let lineage = VectorLineage {
            vector_id: vector_id.to_string(),
            ..Default::default()
        };
        Self { lineage }
    }

    /// Set document source.
    pub fn from_document(mut self, doc_id: &str, chunk_index: usize) -> Self {
        self.lineage.source = SourceInfo::Document {
            document_id: doc_id.to_string(),
            chunk_index,
            char_start: None,
            char_end: None,
        };
        self
    }

    /// Set image source.
    pub fn from_image(mut self, image_id: &str) -> Self {
        self.lineage.source = SourceInfo::Image {
            image_id: image_id.to_string(),
            region: None,
        };
        self
    }

    /// Set as derived from other vectors.
    pub fn derived_from(mut self, parents: Vec<String>, method: &str) -> Self {
        self.lineage.source = SourceInfo::Derived {
            source_vectors: parents.clone(),
            method: method.to_string(),
        };
        self.lineage.parents = parents;
        self
    }

    /// Set embedding model.
    pub fn model(mut self, model: &str, version: &str) -> Self {
        self.lineage.model = model.to_string();
        self.lineage.model_version = version.to_string();
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: &str) -> Self {
        self.lineage.tags.insert(tag.to_string());
        self
    }

    /// Set confidence.
    pub fn confidence(mut self, confidence: f32) -> Self {
        self.lineage.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Set quality score.
    pub fn quality_score(mut self, score: f32) -> Self {
        self.lineage.quality_score = Some(score.clamp(0.0, 1.0));
        self
    }

    /// Build the lineage.
    pub fn build(self) -> VectorLineage {
        self.lineage
    }
}

// --- OTel-style Observability ---

static SPAN_COUNTER: AtomicU64 = AtomicU64::new(0);

fn generate_span_id() -> String {
    let count = SPAN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time = now_us();
    format!(
        "{:08x}{:08x}",
        (time >> 32) as u32 ^ count as u32,
        time as u32
    )
}

fn generate_trace_id() -> String {
    let count = SPAN_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time = now_us();
    format!("{:016x}{:016x}", time, count)
}

fn now_us() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

/// Status of a span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpanStatus {
    /// Operation completed successfully.
    Ok,
    /// Operation failed with an error message.
    Error(String),
    /// Status not set.
    Unset,
}

/// Lightweight OpenTelemetry-compatible trace context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanContext {
    /// Trace ID (32 hex chars).
    pub trace_id: String,
    /// Span ID (16 hex chars).
    pub span_id: String,
    /// Parent span ID.
    pub parent_span_id: Option<String>,
    /// Operation name.
    pub operation: String,
    /// Start time in microseconds since epoch.
    pub start_time_us: u64,
    /// End time in microseconds since epoch.
    pub end_time_us: Option<u64>,
    /// Span attributes.
    pub attributes: HashMap<String, String>,
    /// Span status.
    pub status: SpanStatus,
}

/// Records and manages spans for observability.
pub struct TraceRecorder {
    spans: RwLock<Vec<SpanContext>>,
}

impl TraceRecorder {
    /// Create a new trace recorder.
    pub fn new() -> Self {
        Self {
            spans: RwLock::new(Vec::new()),
        }
    }

    /// Start a new span, returning the created context.
    pub fn start_span(
        &self,
        operation: &str,
        trace_id: Option<&str>,
        parent: Option<&str>,
    ) -> SpanContext {
        let span = SpanContext {
            trace_id: trace_id
                .map(|s| s.to_string())
                .unwrap_or_else(generate_trace_id),
            span_id: generate_span_id(),
            parent_span_id: parent.map(|s| s.to_string()),
            operation: operation.to_string(),
            start_time_us: now_us(),
            end_time_us: None,
            attributes: HashMap::new(),
            status: SpanStatus::Unset,
        };
        self.spans.write().push(span.clone());
        span
    }

    /// End a span with the given status.
    pub fn end_span(&self, span_id: &str, status: SpanStatus) {
        let mut spans = self.spans.write();
        if let Some(span) = spans.iter_mut().find(|s| s.span_id == span_id) {
            span.end_time_us = Some(now_us());
            span.status = status;
        }
    }

    /// Get all spans belonging to a trace.
    pub fn get_trace(&self, trace_id: &str) -> Vec<SpanContext> {
        self.spans
            .read()
            .iter()
            .filter(|s| s.trace_id == trace_id)
            .cloned()
            .collect()
    }

    /// Get the most recent spans up to `limit`.
    pub fn get_recent_spans(&self, limit: usize) -> Vec<SpanContext> {
        let spans = self.spans.read();
        spans.iter().rev().take(limit).cloned().collect()
    }
}

impl Default for TraceRecorder {
    fn default() -> Self {
        Self::new()
    }
}

// --- Lineage Graph Traversal ---

/// Detailed impact report from graph traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphImpactReport {
    /// Source vector ID.
    pub source_id: String,
    /// All affected vector IDs (descendants).
    pub affected_vectors: Vec<String>,
    /// Number of affected vectors.
    pub affected_count: usize,
    /// Models used by affected vectors.
    pub affected_models: HashSet<String>,
    /// Maximum depth reached during traversal.
    pub depth: usize,
}

/// Graph traversal utilities built from a [`LineageTracker`].
pub struct LineageGraphExplorer<'a> {
    tracker: &'a LineageTracker,
}

impl<'a> LineageGraphExplorer<'a> {
    /// Build a graph explorer from a lineage tracker.
    pub fn new(tracker: &'a LineageTracker) -> Self {
        Self { tracker }
    }

    /// Find all ancestors of a vector up to `max_depth` (BFS through parents).
    pub fn ancestors(&self, vector_id: &str, max_depth: usize) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        if let Some(lineage) = self.tracker.get(vector_id) {
            for parent in &lineage.parents {
                queue.push_back((parent.clone(), 1));
            }
        }

        while let Some((id, depth)) = queue.pop_front() {
            if depth > max_depth || visited.contains(&id) {
                continue;
            }
            visited.insert(id.clone());
            result.push(id.clone());

            if let Some(lineage) = self.tracker.get(&id) {
                for parent in &lineage.parents {
                    queue.push_back((parent.clone(), depth + 1));
                }
            }
        }

        result
    }

    /// Find all descendants of a vector up to `max_depth` (BFS through children).
    pub fn descendants(&self, vector_id: &str, max_depth: usize) -> Vec<String> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        if let Some(lineage) = self.tracker.get(vector_id) {
            for child in &lineage.children {
                queue.push_back((child.clone(), 1));
            }
        }

        while let Some((id, depth)) = queue.pop_front() {
            if depth > max_depth || visited.contains(&id) {
                continue;
            }
            visited.insert(id.clone());
            result.push(id.clone());

            if let Some(lineage) = self.tracker.get(&id) {
                for child in &lineage.children {
                    queue.push_back((child.clone(), depth + 1));
                }
            }
        }

        result
    }

    /// Perform impact analysis: find all vectors affected by a change to `source_id`.
    pub fn impact_analysis(&self, source_id: &str) -> GraphImpactReport {
        let mut affected = Vec::new();
        let mut affected_models = HashSet::new();
        let mut visited = HashSet::new();
        let mut max_depth: usize = 0;
        let mut queue: VecDeque<(String, usize)> = VecDeque::new();

        if let Some(lineage) = self.tracker.get(source_id) {
            for child in &lineage.children {
                queue.push_back((child.clone(), 1));
            }
        }

        while let Some((id, depth)) = queue.pop_front() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id.clone());
            affected.push(id.clone());
            if depth > max_depth {
                max_depth = depth;
            }

            if let Some(lineage) = self.tracker.get(&id) {
                if !lineage.model.is_empty() {
                    affected_models.insert(lineage.model.clone());
                }
                for child in &lineage.children {
                    queue.push_back((child.clone(), depth + 1));
                }
            }
        }

        let affected_count = affected.len();
        GraphImpactReport {
            source_id: source_id.to_string(),
            affected_vectors: affected,
            affected_count,
            affected_models,
            depth: max_depth,
        }
    }

    /// Find the shortest path between two vectors (treating graph as undirected).
    pub fn path_between(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }

        let mut visited = HashSet::new();
        let mut queue: VecDeque<Vec<String>> = VecDeque::new();

        visited.insert(from.to_string());
        queue.push_back(vec![from.to_string()]);

        while let Some(path) = queue.pop_front() {
            let current = path.last().expect("path is non-empty");

            if let Some(lineage) = self.tracker.get(current) {
                let neighbors: Vec<&String> =
                    lineage.parents.iter().chain(lineage.children.iter()).collect();

                for neighbor in neighbors {
                    if neighbor == to {
                        let mut result = path.clone();
                        result.push(neighbor.clone());
                        return Some(result);
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(neighbor.clone());
                        queue.push_back(new_path);
                    }
                }
            }
        }

        None
    }
}

// --- Thread-safe Lineage Tracker ---

/// Thread-safe wrapper around [`LineageTracker`] using `parking_lot::RwLock`.
pub struct ThreadSafeLineageTracker {
    inner: RwLock<LineageTracker>,
}

impl ThreadSafeLineageTracker {
    /// Create a new thread-safe lineage tracker.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(LineageTracker::new()),
        }
    }

    /// Register a vector with its lineage.
    pub fn register(&self, vector_id: &str, lineage: VectorLineage) -> Result<()> {
        self.inner.write().register(vector_id, lineage)
    }

    /// Get lineage for a vector (cloned).
    pub fn get(&self, vector_id: &str) -> Option<VectorLineage> {
        self.inner.read().get(vector_id).cloned()
    }

    /// Add a transformation to a vector.
    pub fn add_transformation(
        &self,
        vector_id: &str,
        transformation: Transformation,
    ) -> Result<()> {
        self.inner.write().add_transformation(vector_id, transformation)
    }

    /// Add a transformation with details.
    pub fn add_transformation_with_details(
        &self,
        vector_id: &str,
        transformation: Transformation,
        user: Option<&str>,
        notes: Option<&str>,
    ) -> Result<()> {
        self.inner
            .write()
            .add_transformation_with_details(vector_id, transformation, user, notes)
    }

    /// Add a tag.
    pub fn add_tag(&self, vector_id: &str, tag: &str) -> Result<()> {
        self.inner.write().add_tag(vector_id, tag)
    }

    /// Remove a tag.
    pub fn remove_tag(&self, vector_id: &str, tag: &str) -> Result<()> {
        self.inner.write().remove_tag(vector_id, tag)
    }

    /// Set quality score.
    pub fn set_quality_score(&self, vector_id: &str, score: f32) -> Result<()> {
        self.inner.write().set_quality_score(vector_id, score)
    }

    /// Find vectors by source (cloned).
    pub fn find_by_source(&self, source_id: &str) -> Vec<VectorLineage> {
        self.inner
            .read()
            .find_by_source(source_id)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Find vectors by model (cloned).
    pub fn find_by_model(&self, model: &str) -> Vec<VectorLineage> {
        self.inner
            .read()
            .find_by_model(model)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Find vectors by tag (cloned).
    pub fn find_by_tag(&self, tag: &str) -> Vec<VectorLineage> {
        self.inner
            .read()
            .find_by_tag(tag)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Delete lineage for a vector.
    pub fn delete(&self, vector_id: &str) -> Result<()> {
        self.inner.write().delete(vector_id)
    }

    /// Get statistics.
    pub fn stats(&self) -> LineageStats {
        self.inner.read().stats()
    }

    /// Get count.
    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }

    /// Access the inner tracker with a read lock for graph exploration.
    pub fn with_read<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&LineageTracker) -> R,
    {
        f(&self.inner.read())
    }
}

impl Default for ThreadSafeLineageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tracker() {
        let tracker = LineageTracker::new();
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_register_vector() {
        let mut tracker = LineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("text-embedding-3-small", "1.0.0")
            .build();

        tracker.register("vec1", lineage).unwrap();

        assert_eq!(tracker.len(), 1);
        assert!(tracker.get("vec1").is_some());
    }

    #[test]
    fn test_duplicate_registration() {
        let mut tracker = LineageTracker::new();

        let lineage1 = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();

        let lineage2 = LineageBuilder::new("vec1")
            .from_document("doc2", 0)
            .model("model1", "1.0")
            .build();

        tracker.register("vec1", lineage1).unwrap();
        let result = tracker.register("vec1", lineage2);

        assert!(result.is_err());
    }

    #[test]
    fn test_find_by_source() {
        let mut tracker = LineageTracker::new();

        for i in 0..3 {
            let lineage = LineageBuilder::new(&format!("vec{}", i))
                .from_document("doc1", i)
                .model("model1", "1.0")
                .build();
            tracker.register(&format!("vec{}", i), lineage).unwrap();
        }

        let lineage = LineageBuilder::new("vec_other")
            .from_document("doc2", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec_other", lineage).unwrap();

        let found = tracker.find_by_source("doc1");
        assert_eq!(found.len(), 3);
    }

    #[test]
    fn test_find_by_model() {
        let mut tracker = LineageTracker::new();

        let lineage1 = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model_a", "1.0")
            .build();
        tracker.register("vec1", lineage1).unwrap();

        let lineage2 = LineageBuilder::new("vec2")
            .from_document("doc2", 0)
            .model("model_b", "1.0")
            .build();
        tracker.register("vec2", lineage2).unwrap();

        let found = tracker.find_by_model("model_a");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].vector_id, "vec1");
    }

    #[test]
    fn test_tags() {
        let mut tracker = LineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .tag("important")
            .build();
        tracker.register("vec1", lineage).unwrap();

        tracker.add_tag("vec1", "reviewed").unwrap();

        let by_important = tracker.find_by_tag("important");
        let by_reviewed = tracker.find_by_tag("reviewed");

        assert_eq!(by_important.len(), 1);
        assert_eq!(by_reviewed.len(), 1);

        tracker.remove_tag("vec1", "reviewed").unwrap();
        let by_reviewed = tracker.find_by_tag("reviewed");
        assert_eq!(by_reviewed.len(), 0);
    }

    #[test]
    fn test_add_transformation() {
        let mut tracker = LineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        tracker.add_transformation("vec1", Transformation::Normalize).unwrap();
        tracker.add_transformation("vec1", Transformation::DimensionReduction {
            from_dims: 1536,
            to_dims: 384,
            method: "PCA".to_string(),
        }).unwrap();

        let lineage = tracker.get("vec1").unwrap();
        assert_eq!(lineage.transformations.len(), 3); // Created + 2 transformations
    }

    #[test]
    fn test_parent_child_relationships() {
        let mut tracker = LineageTracker::new();

        // Register parent vectors
        let parent1 = LineageBuilder::new("parent1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("parent1", parent1).unwrap();

        let parent2 = LineageBuilder::new("parent2")
            .from_document("doc1", 1)
            .model("model1", "1.0")
            .build();
        tracker.register("parent2", parent2).unwrap();

        // Register derived vector
        let derived = LineageBuilder::new("derived")
            .derived_from(vec!["parent1".to_string(), "parent2".to_string()], "average")
            .model("model1", "1.0")
            .build();
        tracker.register("derived", derived).unwrap();

        // Check relationships
        let ancestors = tracker.get_ancestors("derived");
        assert_eq!(ancestors.len(), 2);

        let descendants = tracker.get_descendants("parent1");
        assert_eq!(descendants.len(), 1);
        assert_eq!(descendants[0].vector_id, "derived");
    }

    #[test]
    fn test_impact_analysis() {
        let mut tracker = LineageTracker::new();

        // Source vectors
        for i in 0..3 {
            let lineage = LineageBuilder::new(&format!("vec{}", i))
                .from_document("doc1", i)
                .model("model1", "1.0")
                .build();
            tracker.register(&format!("vec{}", i), lineage).unwrap();
        }

        // Derived vector
        let derived = LineageBuilder::new("derived")
            .derived_from(vec!["vec0".to_string()], "transform")
            .model("model1", "1.0")
            .build();
        tracker.register("derived", derived).unwrap();

        let report = tracker.impact_analysis("doc1");

        assert_eq!(report.direct_vectors.len(), 3);
        assert_eq!(report.indirect_vectors.len(), 1);
        assert_eq!(report.total_affected, 4);
    }

    #[test]
    fn test_export_graph() {
        let mut tracker = LineageTracker::new();

        let parent = LineageBuilder::new("parent")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("parent", parent).unwrap();

        let child = LineageBuilder::new("child")
            .derived_from(vec!["parent".to_string()], "copy")
            .model("model1", "1.0")
            .build();
        tracker.register("child", child).unwrap();

        let graph = tracker.export_graph();

        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.edges[0].from, "parent");
        assert_eq!(graph.edges[0].to, "child");
    }

    #[test]
    fn test_delete_lineage() {
        let mut tracker = LineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .tag("test")
            .build();
        tracker.register("vec1", lineage).unwrap();

        assert_eq!(tracker.len(), 1);

        tracker.delete("vec1").unwrap();

        assert_eq!(tracker.len(), 0);
        assert!(tracker.get("vec1").is_none());
        assert!(tracker.find_by_source("doc1").is_empty());
        assert!(tracker.find_by_tag("test").is_empty());
    }

    #[test]
    fn test_stats() {
        let mut tracker = LineageTracker::new();

        for i in 0..5 {
            let lineage = LineageBuilder::new(&format!("vec{}", i))
                .from_document(&format!("doc{}", i % 2), i)
                .model(if i < 3 { "model_a" } else { "model_b" }, "1.0")
                .tag("all")
                .build();
            tracker.register(&format!("vec{}", i), lineage).unwrap();
        }

        let stats = tracker.stats();

        assert_eq!(stats.total_vectors, 5);
        assert_eq!(stats.unique_sources, 2);
        assert_eq!(stats.unique_models, 2);
        assert_eq!(stats.unique_tags, 1);
    }

    #[test]
    fn test_quality_score() {
        let mut tracker = LineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .quality_score(0.85)
            .build();
        tracker.register("vec1", lineage).unwrap();

        assert_eq!(tracker.get("vec1").unwrap().quality_score, Some(0.85));

        tracker.set_quality_score("vec1", 0.95).unwrap();
        assert_eq!(tracker.get("vec1").unwrap().quality_score, Some(0.95));
    }

    #[test]
    fn test_different_source_types() {
        let mut tracker = LineageTracker::new();

        let doc = LineageBuilder::new("doc_vec")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("doc_vec", doc).unwrap();

        let img = LineageBuilder::new("img_vec")
            .from_image("img1")
            .model("clip", "1.0")
            .build();
        tracker.register("img_vec", img).unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.vectors_by_source.get("document"), Some(&1));
        assert_eq!(stats.vectors_by_source.get("image"), Some(&1));
    }

    #[test]
    fn test_lineage_builder() {
        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("text-embedding-3-small", "1.0.0")
            .tag("production")
            .tag("validated")
            .confidence(0.95)
            .quality_score(0.88)
            .build();

        assert_eq!(lineage.vector_id, "vec1");
        assert_eq!(lineage.model, "text-embedding-3-small");
        assert!(lineage.tags.contains("production"));
        assert!(lineage.tags.contains("validated"));
        assert_eq!(lineage.confidence, 0.95);
        assert_eq!(lineage.quality_score, Some(0.88));
    }

    // --- New tests for SpanContext, TraceRecorder, LineageGraphExplorer, ThreadSafeLineageTracker ---

    #[test]
    fn test_span_context_creation() {
        let span = SpanContext {
            trace_id: "abc123".to_string(),
            span_id: "span001".to_string(),
            parent_span_id: None,
            operation: "insert".to_string(),
            start_time_us: 1000,
            end_time_us: None,
            attributes: HashMap::new(),
            status: SpanStatus::Unset,
        };

        assert_eq!(span.trace_id, "abc123");
        assert_eq!(span.operation, "insert");
        assert!(span.end_time_us.is_none());
        assert!(matches!(span.status, SpanStatus::Unset));
    }

    #[test]
    fn test_span_context_lifecycle() {
        let mut span = SpanContext {
            trace_id: "t1".to_string(),
            span_id: "s1".to_string(),
            parent_span_id: None,
            operation: "search".to_string(),
            start_time_us: 100,
            end_time_us: None,
            attributes: HashMap::new(),
            status: SpanStatus::Unset,
        };

        span.attributes.insert("k".to_string(), "v".to_string());
        span.end_time_us = Some(200);
        span.status = SpanStatus::Ok;

        assert_eq!(span.end_time_us, Some(200));
        assert!(matches!(span.status, SpanStatus::Ok));
        assert_eq!(span.attributes.get("k").unwrap(), "v");
    }

    #[test]
    fn test_span_status_error() {
        let status = SpanStatus::Error("timeout".to_string());
        if let SpanStatus::Error(msg) = &status {
            assert_eq!(msg, "timeout");
        } else {
            panic!("expected SpanStatus::Error");
        }
    }

    #[test]
    fn test_trace_recorder_start_span() {
        let recorder = TraceRecorder::new();

        let span = recorder.start_span("insert_vector", None, None);
        assert_eq!(span.operation, "insert_vector");
        assert!(!span.trace_id.is_empty());
        assert!(!span.span_id.is_empty());
        assert!(span.parent_span_id.is_none());
        assert!(span.end_time_us.is_none());
    }

    #[test]
    fn test_trace_recorder_with_trace_id() {
        let recorder = TraceRecorder::new();

        let span = recorder.start_span("op1", Some("my-trace"), None);
        assert_eq!(span.trace_id, "my-trace");

        let child = recorder.start_span("op2", Some("my-trace"), Some(&span.span_id));
        assert_eq!(child.trace_id, "my-trace");
        assert_eq!(child.parent_span_id.as_deref(), Some(span.span_id.as_str()));
    }

    #[test]
    fn test_trace_recorder_end_span() {
        let recorder = TraceRecorder::new();

        let span = recorder.start_span("work", None, None);
        recorder.end_span(&span.span_id, SpanStatus::Ok);

        let traces = recorder.get_trace(&span.trace_id);
        assert_eq!(traces.len(), 1);
        assert!(traces[0].end_time_us.is_some());
        assert!(matches!(traces[0].status, SpanStatus::Ok));
    }

    #[test]
    fn test_trace_recorder_end_span_error() {
        let recorder = TraceRecorder::new();

        let span = recorder.start_span("fail_op", None, None);
        recorder.end_span(&span.span_id, SpanStatus::Error("bad input".to_string()));

        let traces = recorder.get_trace(&span.trace_id);
        assert!(matches!(&traces[0].status, SpanStatus::Error(msg) if msg == "bad input"));
    }

    #[test]
    fn test_trace_recorder_get_trace() {
        let recorder = TraceRecorder::new();

        let s1 = recorder.start_span("op1", Some("trace-A"), None);
        let _s2 = recorder.start_span("op2", Some("trace-A"), Some(&s1.span_id));
        let _s3 = recorder.start_span("op3", Some("trace-B"), None);

        let trace_a = recorder.get_trace("trace-A");
        assert_eq!(trace_a.len(), 2);

        let trace_b = recorder.get_trace("trace-B");
        assert_eq!(trace_b.len(), 1);
    }

    #[test]
    fn test_trace_recorder_get_recent_spans() {
        let recorder = TraceRecorder::new();

        for i in 0..5 {
            recorder.start_span(&format!("op{}", i), None, None);
        }

        let recent = recorder.get_recent_spans(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].operation, "op4");
        assert_eq!(recent[2].operation, "op2");
    }

    #[test]
    fn test_trace_recorder_unique_span_ids() {
        let recorder = TraceRecorder::new();
        let s1 = recorder.start_span("a", None, None);
        let s2 = recorder.start_span("b", None, None);
        assert_ne!(s1.span_id, s2.span_id);
    }

    // --- LineageGraphExplorer tests ---

    fn build_chain_tracker() -> LineageTracker {
        // root -> mid -> leaf
        let mut tracker = LineageTracker::new();

        let root = LineageBuilder::new("root")
            .from_document("doc1", 0)
            .model("model_a", "1.0")
            .build();
        tracker.register("root", root).unwrap();

        let mid = LineageBuilder::new("mid")
            .derived_from(vec!["root".to_string()], "transform")
            .model("model_a", "1.0")
            .build();
        tracker.register("mid", mid).unwrap();

        let leaf = LineageBuilder::new("leaf")
            .derived_from(vec!["mid".to_string()], "transform")
            .model("model_b", "2.0")
            .build();
        tracker.register("leaf", leaf).unwrap();

        tracker
    }

    #[test]
    fn test_graph_explorer_ancestors() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let ancestors = explorer.ancestors("leaf", 10);
        assert_eq!(ancestors.len(), 2);
        assert!(ancestors.contains(&"mid".to_string()));
        assert!(ancestors.contains(&"root".to_string()));
    }

    #[test]
    fn test_graph_explorer_ancestors_depth_limit() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let ancestors = explorer.ancestors("leaf", 1);
        assert_eq!(ancestors.len(), 1);
        assert!(ancestors.contains(&"mid".to_string()));
    }

    #[test]
    fn test_graph_explorer_descendants() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let descendants = explorer.descendants("root", 10);
        assert_eq!(descendants.len(), 2);
        assert!(descendants.contains(&"mid".to_string()));
        assert!(descendants.contains(&"leaf".to_string()));
    }

    #[test]
    fn test_graph_explorer_descendants_depth_limit() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let descendants = explorer.descendants("root", 1);
        assert_eq!(descendants.len(), 1);
        assert!(descendants.contains(&"mid".to_string()));
    }

    #[test]
    fn test_graph_explorer_impact_analysis() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let report = explorer.impact_analysis("root");
        assert_eq!(report.source_id, "root");
        assert_eq!(report.affected_count, 2);
        assert!(report.affected_vectors.contains(&"mid".to_string()));
        assert!(report.affected_vectors.contains(&"leaf".to_string()));
        assert!(report.affected_models.contains("model_a"));
        assert!(report.affected_models.contains("model_b"));
        assert_eq!(report.depth, 2);
    }

    #[test]
    fn test_graph_explorer_impact_analysis_leaf() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let report = explorer.impact_analysis("leaf");
        assert_eq!(report.affected_count, 0);
        assert_eq!(report.depth, 0);
    }

    #[test]
    fn test_graph_explorer_impact_analysis_wide() {
        let mut tracker = LineageTracker::new();

        let root = LineageBuilder::new("root")
            .from_document("doc1", 0)
            .model("model_a", "1.0")
            .build();
        tracker.register("root", root).unwrap();

        // Two children from root
        for i in 0..2 {
            let child = LineageBuilder::new(&format!("child{}", i))
                .derived_from(vec!["root".to_string()], "split")
                .model("model_a", "1.0")
                .build();
            tracker.register(&format!("child{}", i), child).unwrap();
        }

        // Grandchild from child0
        let gc = LineageBuilder::new("grandchild")
            .derived_from(vec!["child0".to_string()], "refine")
            .model("model_b", "2.0")
            .build();
        tracker.register("grandchild", gc).unwrap();

        let explorer = LineageGraphExplorer::new(&tracker);
        let report = explorer.impact_analysis("root");

        assert_eq!(report.affected_count, 3);
        assert_eq!(report.depth, 2);
        assert!(report.affected_models.contains("model_a"));
        assert!(report.affected_models.contains("model_b"));
    }

    #[test]
    fn test_graph_explorer_path_between_direct() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let path = explorer.path_between("root", "mid");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec!["root", "mid"]);
    }

    #[test]
    fn test_graph_explorer_path_between_transitive() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let path = explorer.path_between("root", "leaf");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "root");
        assert_eq!(path[2], "leaf");
    }

    #[test]
    fn test_graph_explorer_path_between_reverse() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        // Should find path going child->parent direction
        let path = explorer.path_between("leaf", "root");
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "leaf");
        assert_eq!(path[2], "root");
    }

    #[test]
    fn test_graph_explorer_path_between_same() {
        let tracker = build_chain_tracker();
        let explorer = LineageGraphExplorer::new(&tracker);

        let path = explorer.path_between("root", "root");
        assert_eq!(path, Some(vec!["root".to_string()]));
    }

    #[test]
    fn test_graph_explorer_path_between_disconnected() {
        let mut tracker = LineageTracker::new();

        let v1 = LineageBuilder::new("a")
            .from_document("d1", 0)
            .model("m1", "1.0")
            .build();
        tracker.register("a", v1).unwrap();

        let v2 = LineageBuilder::new("b")
            .from_document("d2", 0)
            .model("m1", "1.0")
            .build();
        tracker.register("b", v2).unwrap();

        let explorer = LineageGraphExplorer::new(&tracker);
        assert!(explorer.path_between("a", "b").is_none());
    }

    // --- ThreadSafeLineageTracker tests ---

    #[test]
    fn test_thread_safe_tracker_basic() {
        let tracker = ThreadSafeLineageTracker::new();
        assert!(tracker.is_empty());

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        assert_eq!(tracker.len(), 1);
        assert!(tracker.get("vec1").is_some());
    }

    #[test]
    fn test_thread_safe_tracker_transformations() {
        let tracker = ThreadSafeLineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        tracker
            .add_transformation("vec1", Transformation::Normalize)
            .unwrap();

        let l = tracker.get("vec1").unwrap();
        assert_eq!(l.transformations.len(), 2); // Created + Normalize
    }

    #[test]
    fn test_thread_safe_tracker_tags() {
        let tracker = ThreadSafeLineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        tracker.add_tag("vec1", "important").unwrap();
        assert_eq!(tracker.find_by_tag("important").len(), 1);

        tracker.remove_tag("vec1", "important").unwrap();
        assert_eq!(tracker.find_by_tag("important").len(), 0);
    }

    #[test]
    fn test_thread_safe_tracker_find_by_source_and_model() {
        let tracker = ThreadSafeLineageTracker::new();

        let l1 = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model_a", "1.0")
            .build();
        tracker.register("vec1", l1).unwrap();

        let l2 = LineageBuilder::new("vec2")
            .from_document("doc1", 1)
            .model("model_b", "1.0")
            .build();
        tracker.register("vec2", l2).unwrap();

        assert_eq!(tracker.find_by_source("doc1").len(), 2);
        assert_eq!(tracker.find_by_model("model_a").len(), 1);
    }

    #[test]
    fn test_thread_safe_tracker_delete() {
        let tracker = ThreadSafeLineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        tracker.delete("vec1").unwrap();
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_thread_safe_tracker_stats() {
        let tracker = ThreadSafeLineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        let stats = tracker.stats();
        assert_eq!(stats.total_vectors, 1);
    }

    #[test]
    fn test_thread_safe_tracker_quality_score() {
        let tracker = ThreadSafeLineageTracker::new();

        let lineage = LineageBuilder::new("vec1")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("vec1", lineage).unwrap();

        tracker.set_quality_score("vec1", 0.9).unwrap();
        assert_eq!(tracker.get("vec1").unwrap().quality_score, Some(0.9));
    }

    #[test]
    fn test_thread_safe_tracker_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let tracker = Arc::new(ThreadSafeLineageTracker::new());
        let mut handles = vec![];

        // Spawn writers
        for i in 0..10 {
            let t = Arc::clone(&tracker);
            handles.push(thread::spawn(move || {
                let lineage = LineageBuilder::new(&format!("vec{}", i))
                    .from_document(&format!("doc{}", i), 0)
                    .model("model1", "1.0")
                    .build();
                t.register(&format!("vec{}", i), lineage).unwrap();
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(tracker.len(), 10);

        // Concurrent reads
        let mut read_handles = vec![];
        for i in 0..10 {
            let t = Arc::clone(&tracker);
            read_handles.push(thread::spawn(move || {
                assert!(t.get(&format!("vec{}", i)).is_some());
            }));
        }

        for h in read_handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_thread_safe_tracker_with_read() {
        let tracker = ThreadSafeLineageTracker::new();

        let root = LineageBuilder::new("root")
            .from_document("doc1", 0)
            .model("model1", "1.0")
            .build();
        tracker.register("root", root).unwrap();

        let child = LineageBuilder::new("child")
            .derived_from(vec!["root".to_string()], "copy")
            .model("model1", "1.0")
            .build();
        tracker.register("child", child).unwrap();

        let descendants = tracker.with_read(|inner| {
            let explorer = LineageGraphExplorer::new(inner);
            explorer.descendants("root", 10)
        });

        assert_eq!(descendants.len(), 1);
        assert!(descendants.contains(&"child".to_string()));
    }
}
