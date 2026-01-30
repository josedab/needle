//! Vector Lineage Tracking
//!
//! Tracks provenance chain for every vector: source document → embedding model
//! → version → transformations. Enables "explain this result" with full lineage.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::vector_lineage::{
//!     LineageTracker, VectorLineage, LineageEvent, ModelInfo,
//! };
//!
//! let mut tracker = LineageTracker::new();
//! tracker.record(VectorLineage::new("v1", "docs")
//!     .with_source("document.pdf", "page 3")
//!     .with_model(ModelInfo::new("all-MiniLM-L6-v2", 384))
//!     .with_transform("normalize"));
//!
//! let lineage = tracker.get("v1").unwrap();
//! println!("Source: {:?}, Model: {:?}", lineage.source, lineage.model);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: Option<String>,
    pub dimensions: usize,
    pub provider: Option<String>,
}

impl ModelInfo {
    pub fn new(name: &str, dimensions: usize) -> Self {
        Self { name: name.into(), version: None, dimensions, provider: None }
    }
    #[must_use] pub fn with_version(mut self, v: &str) -> Self { self.version = Some(v.into()); self }
    #[must_use] pub fn with_provider(mut self, p: &str) -> Self { self.provider = Some(p.into()); self }
}

/// Source document information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub document_id: String,
    pub location: Option<String>,
    pub content_hash: Option<String>,
}

/// A transformation applied to a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    pub name: String,
    pub params: Option<HashMap<String, String>>,
    pub timestamp: u64,
}

/// Complete lineage for a vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorLineage {
    pub vector_id: String,
    pub collection: String,
    pub source: Option<SourceInfo>,
    pub model: Option<ModelInfo>,
    pub transforms: Vec<Transform>,
    pub created_at: u64,
    pub version: u32,
}

impl VectorLineage {
    pub fn new(vector_id: &str, collection: &str) -> Self {
        Self {
            vector_id: vector_id.into(), collection: collection.into(),
            source: None, model: None, transforms: Vec::new(),
            created_at: now_secs(), version: 1,
        }
    }

    #[must_use]
    pub fn with_source(mut self, doc_id: &str, location: &str) -> Self {
        self.source = Some(SourceInfo { document_id: doc_id.into(), location: Some(location.into()), content_hash: None });
        self
    }

    #[must_use]
    pub fn with_model(mut self, model: ModelInfo) -> Self { self.model = Some(model); self }

    #[must_use]
    pub fn with_transform(mut self, name: &str) -> Self {
        self.transforms.push(Transform { name: name.into(), params: None, timestamp: now_secs() });
        self
    }
}

/// Lineage query result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageQuery {
    pub vector_id: String,
    pub lineage: VectorLineage,
    pub chain_length: usize,
}

/// Lineage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineageStats {
    pub total_tracked: usize,
    pub models_used: usize,
    pub avg_transforms: f32,
    pub sources_tracked: usize,
}

/// Vector lineage tracker.
pub struct LineageTracker {
    lineages: HashMap<String, VectorLineage>,
}

impl LineageTracker {
    pub fn new() -> Self { Self { lineages: HashMap::new() } }

    pub fn record(&mut self, lineage: VectorLineage) {
        self.lineages.insert(lineage.vector_id.clone(), lineage);
    }

    pub fn get(&self, vector_id: &str) -> Option<&VectorLineage> { self.lineages.get(vector_id) }

    pub fn query(&self, vector_id: &str) -> Option<LineageQuery> {
        self.lineages.get(vector_id).map(|l| LineageQuery {
            vector_id: vector_id.into(), lineage: l.clone(),
            chain_length: 1 + l.transforms.len() + l.model.as_ref().map_or(0, |_| 1) + l.source.as_ref().map_or(0, |_| 1),
        })
    }

    pub fn find_by_model(&self, model_name: &str) -> Vec<&VectorLineage> {
        self.lineages.values().filter(|l| l.model.as_ref().map_or(false, |m| m.name == model_name)).collect()
    }

    pub fn find_by_source(&self, doc_id: &str) -> Vec<&VectorLineage> {
        self.lineages.values().filter(|l| l.source.as_ref().map_or(false, |s| s.document_id == doc_id)).collect()
    }

    pub fn remove(&mut self, vector_id: &str) -> bool { self.lineages.remove(vector_id).is_some() }

    pub fn stats(&self) -> LineageStats {
        let mut models: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut total_transforms = 0usize;
        let mut sources = 0usize;
        for l in self.lineages.values() {
            if let Some(m) = &l.model { models.insert(m.name.clone()); }
            total_transforms += l.transforms.len();
            if l.source.is_some() { sources += 1; }
        }
        let n = self.lineages.len().max(1);
        LineageStats { total_tracked: self.lineages.len(), models_used: models.len(), avg_transforms: total_transforms as f32 / n as f32, sources_tracked: sources }
    }

    pub fn len(&self) -> usize { self.lineages.len() }
    pub fn is_empty(&self) -> bool { self.lineages.is_empty() }
}

impl Default for LineageTracker { fn default() -> Self { Self::new() } }

fn now_secs() -> u64 { SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_get() {
        let mut t = LineageTracker::new();
        t.record(VectorLineage::new("v1", "docs").with_source("doc.pdf", "p3").with_model(ModelInfo::new("minilm", 384)));
        let l = t.get("v1").unwrap();
        assert!(l.source.is_some());
        assert_eq!(l.model.as_ref().unwrap().name, "minilm");
    }

    #[test]
    fn test_query_chain() {
        let mut t = LineageTracker::new();
        t.record(VectorLineage::new("v1", "docs").with_source("d", "p").with_model(ModelInfo::new("m", 128)).with_transform("normalize"));
        let q = t.query("v1").unwrap();
        assert!(q.chain_length >= 3); // source + model + transform
    }

    #[test]
    fn test_find_by_model() {
        let mut t = LineageTracker::new();
        t.record(VectorLineage::new("v1", "docs").with_model(ModelInfo::new("minilm", 384)));
        t.record(VectorLineage::new("v2", "docs").with_model(ModelInfo::new("bge", 768)));
        assert_eq!(t.find_by_model("minilm").len(), 1);
    }

    #[test]
    fn test_find_by_source() {
        let mut t = LineageTracker::new();
        t.record(VectorLineage::new("v1", "docs").with_source("doc1.pdf", "p1"));
        t.record(VectorLineage::new("v2", "docs").with_source("doc1.pdf", "p2"));
        assert_eq!(t.find_by_source("doc1.pdf").len(), 2);
    }

    #[test]
    fn test_stats() {
        let mut t = LineageTracker::new();
        t.record(VectorLineage::new("v1", "d").with_model(ModelInfo::new("m1", 128)).with_transform("norm"));
        t.record(VectorLineage::new("v2", "d").with_model(ModelInfo::new("m2", 256)));
        let s = t.stats();
        assert_eq!(s.total_tracked, 2);
        assert_eq!(s.models_used, 2);
    }
}
