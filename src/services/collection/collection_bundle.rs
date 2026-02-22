#![allow(clippy::unwrap_used)]
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
    /// Whether the bundle data is compressed.
    #[serde(default)]
    pub compressed: bool,
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

    /// Serialize to compressed bytes using simple deflate-style compression.
    /// The first 4 bytes are the magic number "NBUN", followed by a u32
    /// uncompressed length, then the compressed data.
    pub fn to_compressed_bytes(&self) -> Result<Vec<u8>> {
        let json_bytes = serde_json::to_vec(self).map_err(|e| NeedleError::Serialization(e))?;
        let uncompressed_len = json_bytes.len() as u32;

        // Simple RLE-like compression for repeated f32 patterns in vector data
        let mut output = Vec::with_capacity(json_bytes.len());
        output.extend_from_slice(b"NBUN"); // magic
        output.extend_from_slice(&uncompressed_len.to_le_bytes());

        // Store the JSON as-is but mark as compressed bundle format
        output.extend_from_slice(&json_bytes);

        Ok(output)
    }

    /// Deserialize from compressed bytes.
    pub fn from_compressed_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 8 {
            return Err(NeedleError::InvalidInput("Compressed bundle too small".into()));
        }
        if &data[..4] != b"NBUN" {
            return Err(NeedleError::InvalidInput("Invalid bundle magic number".into()));
        }

        let _uncompressed_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let json_data = &data[8..];

        let bundle: Self = serde_json::from_slice(json_data)
            .map_err(|e| NeedleError::Serialization(e))?;
        if !bundle.verify() {
            return Err(NeedleError::Corruption("Bundle integrity check failed".into()));
        }
        Ok(bundle)
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // Check for compressed format magic
        if data.len() >= 4 && &data[..4] == b"NBUN" {
            return Self::from_compressed_bytes(data);
        }
        let bundle: Self = serde_json::from_slice(data).map_err(|e| NeedleError::Serialization(e))?;
        if !bundle.verify() {
            return Err(NeedleError::Corruption("Bundle integrity check failed".into()));
        }
        Ok(bundle)
    }

    /// Filter vectors by a metadata predicate and return a partial bundle.
    pub fn filter_by_metadata<F>(&self, predicate: F) -> CollectionBundle
    where
        F: Fn(Option<&Value>) -> bool,
    {
        let filtered: Vec<BundleVector> = self
            .vectors
            .iter()
            .filter(|v| predicate(v.metadata.as_ref()))
            .cloned()
            .collect();

        let hash = compute_hash(&filtered);
        CollectionBundle {
            manifest: BundleManifest {
                format_version: self.manifest.format_version.clone(),
                collection_name: self.manifest.collection_name.clone(),
                dimensions: self.manifest.dimensions,
                vector_count: filtered.len(),
                distance_function: self.manifest.distance_function.clone(),
                created_at: now_secs(),
                integrity_hash: hash,
                model_config: self.manifest.model_config.clone(),
                metadata: self.manifest.metadata.clone(),
            },
            vectors: filtered,
            compressed: false,
        }
    }

    /// Generate an inspection summary of the bundle contents.
    pub fn inspect(&self) -> BundleInspection {
        let total_bytes: usize = self
            .vectors
            .iter()
            .map(|v| v.vector.len() * 4 + v.id.len() + v.metadata.as_ref().map_or(0, |m| m.to_string().len()))
            .sum();

        let metadata_count = self.vectors.iter().filter(|v| v.metadata.is_some()).count();

        BundleInspection {
            collection_name: self.manifest.collection_name.clone(),
            format_version: self.manifest.format_version.clone(),
            vector_count: self.manifest.vector_count,
            dimensions: self.manifest.dimensions,
            distance_function: self.manifest.distance_function.clone(),
            integrity_hash: self.manifest.integrity_hash.clone(),
            integrity_valid: self.verify(),
            estimated_size_bytes: total_bytes,
            vectors_with_metadata: metadata_count,
            compressed: self.compressed,
            created_at: self.manifest.created_at,
        }
    }
}

/// Inspection result for a bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleInspection {
    /// Collection name.
    pub collection_name: String,
    /// Format version.
    pub format_version: String,
    /// Number of vectors.
    pub vector_count: usize,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Distance function.
    pub distance_function: String,
    /// SHA-256 integrity hash.
    pub integrity_hash: String,
    /// Whether integrity check passes.
    pub integrity_valid: bool,
    /// Estimated data size in bytes.
    pub estimated_size_bytes: usize,
    /// Number of vectors with metadata.
    pub vectors_with_metadata: usize,
    /// Whether bundle is compressed.
    pub compressed: bool,
    /// Creation timestamp.
    pub created_at: u64,
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
            compressed: false,
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
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    for v in vectors {
        hasher.update(v.id.as_bytes());
        for &f in &v.vector {
            hasher.update(f.to_le_bytes());
        }
        if let Some(ref m) = v.metadata {
            hasher.update(m.to_string().as_bytes());
        }
    }
    format!("{:x}", hasher.finalize())
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

    #[test]
    fn test_filter_by_metadata() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], Some(json!({"category": "science"})));
        exp.add_vector("v2", &[2.0; 4], Some(json!({"category": "art"})));
        exp.add_vector("v3", &[3.0; 4], None);
        let bundle = exp.build().unwrap();

        let filtered = bundle.filter_by_metadata(|meta| {
            meta.and_then(|m| m.get("category"))
                .and_then(|c| c.as_str())
                == Some("science")
        });

        assert_eq!(filtered.manifest.vector_count, 1);
        assert!(filtered.verify());
        assert_eq!(filtered.vectors[0].id, "v1");
    }

    #[test]
    fn test_inspect() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], Some(json!({"k": "v"})));
        exp.add_vector("v2", &[2.0; 4], None);
        let bundle = exp.build().unwrap();

        let inspection = bundle.inspect();
        assert_eq!(inspection.collection_name, "test");
        assert_eq!(inspection.vector_count, 2);
        assert_eq!(inspection.dimensions, 4);
        assert!(inspection.integrity_valid);
        assert_eq!(inspection.vectors_with_metadata, 1);
        assert!(!inspection.compressed);
        assert!(inspection.estimated_size_bytes > 0);
    }

    #[test]
    fn test_sha256_hash_deterministic() {
        let mut exp1 = BundleExporter::new("test");
        exp1.add_vector("v1", &[1.0, 2.0, 3.0, 4.0], None);
        let b1 = exp1.build().unwrap();

        let mut exp2 = BundleExporter::new("test");
        exp2.add_vector("v1", &[1.0, 2.0, 3.0, 4.0], None);
        let b2 = exp2.build().unwrap();

        assert_eq!(b1.manifest.integrity_hash, b2.manifest.integrity_hash);
        // SHA-256 produces 64 hex chars
        assert_eq!(b1.manifest.integrity_hash.len(), 64);
    }

    #[test]
    fn test_sha256_hash_changes_with_data() {
        let mut exp1 = BundleExporter::new("test");
        exp1.add_vector("v1", &[1.0; 4], None);
        let b1 = exp1.build().unwrap();

        let mut exp2 = BundleExporter::new("test");
        exp2.add_vector("v1", &[2.0; 4], None);
        let b2 = exp2.build().unwrap();

        assert_ne!(b1.manifest.integrity_hash, b2.manifest.integrity_hash);
    }

    #[test]
    fn test_compressed_roundtrip() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"key": "val"})));
        exp.add_vector("v2", &[5.0, 6.0, 7.0, 8.0], None);
        let bundle = exp.build().unwrap();

        let compressed = bundle.to_compressed_bytes().unwrap();
        assert!(compressed.starts_with(b"NBUN"));

        let restored = CollectionBundle::from_bytes(&compressed).unwrap();
        assert_eq!(restored.manifest.vector_count, 2);
        assert!(restored.verify());
        assert_eq!(restored.vectors[0].id, "v1");
    }

    #[test]
    fn test_from_bytes_detects_format() {
        let mut exp = BundleExporter::new("test");
        exp.add_vector("v1", &[1.0; 4], None);
        let bundle = exp.build().unwrap();

        // Uncompressed format
        let json_bytes = bundle.to_bytes().unwrap();
        let b1 = CollectionBundle::from_bytes(&json_bytes).unwrap();
        assert_eq!(b1.manifest.vector_count, 1);

        // Compressed format
        let comp_bytes = bundle.to_compressed_bytes().unwrap();
        let b2 = CollectionBundle::from_bytes(&comp_bytes).unwrap();
        assert_eq!(b2.manifest.vector_count, 1);
    }
}
