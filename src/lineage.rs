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
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
            .unwrap()
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
}
