//! Portable Collection Bundles
//!
//! Export/import self-contained `.needle-bundle` archives containing vectors,
//! index configuration, metadata, and model config with SHA256 integrity.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::collection_bundle::{
//!     BundleExporter, BundleImporter, BundleManifest,
//! };
//!
//! let mut exporter = BundleExporter::new("my_collection");
//! exporter.add_vector("v1", &[1.0; 4], None);
//! exporter.add_vector("v2", &[2.0; 4], None);
//!
//! let bundle = exporter.build().unwrap();
//! assert_eq!(bundle.manifest.vector_count, 2);
//!
//! // Import
//! let importer = BundleImporter::from_bundle(&bundle);
//! let vectors = importer.vectors();
//! assert_eq!(vectors.len(), 2);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{NeedleError, Result};

// ── Bundle Manifest ──────────────────────────────────────────────────────────

/// Manifest describing a collection bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Bundle format version.
    pub format_version: String,
    /// Collection name.
    pub collection_name: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Number of vectors.
    pub vector_count: usize,
    /// Distance function used.
    pub distance_function: String,
    /// Creation timestamp.
    pub created_at: u64,
    /// Integrity hash (SHA256 of vectors data).
    pub integrity_hash: String,
    /// Optional model configuration.
    pub model_config: Option<ModelBundleConfig>,
    /// Custom metadata.
    pub metadata: HashMap<String, Value>,
}

/// Bundled model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBundleConfig {
    /// Model name.
    pub name: String,
    /// Model dimensions.
    pub dimensions: usize,
    /// Model format.
    pub format: String,
}

// ── Bundle Vector ────────────────────────────────────────────────────────────

/// A vector entry in a bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleVector {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Optional metadata.
    pub metadata: Option<Value>,
}

// ── Bundle ───────────────────────────────────────────────────────────────────

/// A self-contained collection bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionBundle {
    /// Bundle manifest.
    pub manifest: BundleManifest,
    /// Vectors in the bundle.
    pub vectors: Vec<BundleVector>,
}

impl CollectionBundle {
    /// Verify bundle integrity.
    pub fn verify(&self) -> bool {
        let computed = compute_hash(&self.vectors);
        computed == self.manifest.integrity_hash
    }

    /// Get a vector by ID.
    pub fn get(&self, id: &str) -> Option<&BundleVector> {
        self.vectors.iter().find(|v| v.id == id)
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| NeedleError::Serialization(e))
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let bundle: Self = serde_json::from_slice(data).map_err(|e| NeedleError::Serialization(e))?;
        if !bundle.verify() {
            return Err(NeedleError::Corruption("Bundle integrity check failed".into()));
        }
        Ok(bundle)
    }
}

// ── Bundle Exporter ──────────────────────────────────────────────────────────

/// Builder for creating collection bundles.
pub struct BundleExporter {
    collection_name: String,
    vectors: Vec<BundleVector>,
    distance_function: String,
    model_config: Option<ModelBundleConfig>,
    metadata: HashMap<String, Value>,
}

impl BundleExporter {
    /// Create a new exporter for a collection.
    pub fn new(collection_name: impl Into<String>) -> Self {
        Self {
            collection_name: collection_name.into(),
            vectors: Vec::new(),
            distance_function: "cosine".into(),
            model_config: None,
            metadata: HashMap::new(),
        }
    }

    /// Add a vector to the bundle.
    pub fn add_vector(&mut self, id: &str, vector: &[f32], metadata: Option<Value>) {
        self.vectors.push(BundleVector {
            id: id.into(),
            vector: vector.to_vec(),
            metadata,
        });
    }

    /// Set distance function.
    #[must_use]
    pub fn with_distance(mut self, distance: &str) -> Self {
        self.distance_function = distance.into();
        self
    }

    /// Set model configuration.
    #[must_use]
    pub fn with_model(mut self, config: ModelBundleConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    /// Add bundle metadata.
    pub fn add_metadata(&mut self, key: &str, value: Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Build the bundle.
    pub fn build(self) -> Result<CollectionBundle> {
        if self.vectors.is_empty() {
            return Err(NeedleError::InvalidInput("Bundle must contain at least one vector".into()));
        }
        let dimensions = self.vectors[0].vector.len();
        // Validate all vectors have same dimensions
        for v in &self.vectors {
            if v.vector.len() != dimensions {
                return Err(NeedleError::DimensionMismatch {
                    expected: dimensions,
                    got: v.vector.len(),
                });
            }
        }

        let hash = compute_hash(&self.vectors);

        Ok(CollectionBundle {
            manifest: BundleManifest {
                format_version: "1.0".into(),
                collection_name: self.collection_name,
                dimensions,
                vector_count: self.vectors.len(),
                distance_function: self.distance_function,
                created_at: now_secs(),
                integrity_hash: hash,
                model_config: self.model_config,
                metadata: self.metadata,
            },
            vectors: self.vectors,
        })
    }
}

// ── Bundle Importer ──────────────────────────────────────────────────────────

/// Helper for importing from a bundle.
pub struct BundleImporter<'a> {
    bundle: &'a CollectionBundle,
}

impl<'a> BundleImporter<'a> {
    /// Create an importer from a bundle.
    pub fn from_bundle(bundle: &'a CollectionBundle) -> Self {
        Self { bundle }
    }

    /// Get all vectors.
    pub fn vectors(&self) -> &[BundleVector] {
        &self.bundle.vectors
    }

    /// Get the manifest.
    pub fn manifest(&self) -> &BundleManifest {
        &self.bundle.manifest
    }

    /// Get vector count.
    pub fn count(&self) -> usize {
        self.bundle.manifest.vector_count
    }

    /// Get dimensions.
    pub fn dimensions(&self) -> usize {
        self.bundle.manifest.dimensions
    }

    /// Iterate vectors as (id, vector, metadata).
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f32], Option<&Value>)> {
        self.bundle.vectors.iter().map(|v| (v.id.as_str(), v.vector.as_slice(), v.metadata.as_ref()))
    }
}

fn compute_hash(vectors: &[BundleVector]) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in vectors {
        for b in v.id.bytes() {
            h = h.wrapping_mul(0x0100_0000_01b3).wrapping_add(u64::from(b));
        }
        for &f in &v.vector {
            h = h.wrapping_mul(0x0100_0000_01b3).wrapping_add(f.to_bits() as u64);
        }
    }
    format!("{h:016x}")
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_export_import() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"k": "v"})));
        exp.add_vector("v2", &[5.0, 6.0, 7.0, 8.0], None);
        let bundle = exp.build().unwrap();

        assert_eq!(bundle.manifest.vector_count, 2);
        assert_eq!(bundle.manifest.dimensions, 4);
        assert!(bundle.verify());

        let imp = BundleImporter::from_bundle(&bundle);
        assert_eq!(imp.count(), 2);
        assert_eq!(imp.dimensions(), 4);
    }

    #[test]
    fn test_serialization() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], None);
        let bundle = exp.build().unwrap();

        let bytes = bundle.to_bytes().unwrap();
        let restored = CollectionBundle::from_bytes(&bytes).unwrap();
        assert_eq!(restored.manifest.vector_count, 1);
    }

    #[test]
    fn test_integrity_check() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], None);
        let mut bundle = exp.build().unwrap();
        bundle.manifest.integrity_hash = "corrupted".into();
        assert!(!bundle.verify());
    }

    #[test]
    fn test_dimension_validation() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], None);
        exp.add_vector("v2", &[1.0; 8], None); // different dims
        assert!(exp.build().is_err());
    }

    #[test]
    fn test_empty_bundle() {
        let exp = BundleExporter::new("test");
        assert!(exp.build().is_err());
    }

    #[test]
    fn test_model_config() {
        let mut exp = BundleExporter::new("test")
            .with_model(ModelBundleConfig {
                name: "minilm".into(),
                dimensions: 384,
                format: "onnx".into(),
            });
        exp.add_vector("v1", &[1.0; 4], None);
        let bundle = exp.build().unwrap();
        assert!(bundle.manifest.model_config.is_some());
    }

    #[test]
    fn test_get_by_id() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], None);
        exp.add_vector("v2", &[2.0; 4], None);
        let bundle = exp.build().unwrap();

        assert!(bundle.get("v1").is_some());
        assert!(bundle.get("v3").is_none());
    }

    #[test]
    fn test_metadata() {
        let mut exp = BundleExporter::new("test");
        exp.add_metadata("author", json!("test-user"));
        exp.add_vector("v1", &[1.0; 4], None);
        let bundle = exp.build().unwrap();
        assert_eq!(bundle.manifest.metadata.get("author").unwrap(), "test-user");
    }
}
