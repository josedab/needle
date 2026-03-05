#![allow(dead_code)]

//! Multi-Modal Vector Search
//!
//! Cross-modal search for CLIP-style image+text, audio, and code embeddings.
//! Provides a modality registry, cross-modal dimension alignment, and unified
//! search API with result fusion.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │              ModalityRegistry                  │
//! │  Text  → dim=384, model="MiniLM"              │
//! │  Image → dim=512, model="CLIP"                │
//! │  Audio → dim=256, model="Whisper"             │
//! ├──────────────────────────────────────────────┤
//! │         CrossModalSearchEngine                 │
//! │  search(query, modalities) → fused results    │
//! │  ┌─────────────┐  ┌────────────────────────┐  │
//! │  │ Per-modality │  │ Projection / Alignment │  │
//! │  │ indices      │  │ matrices              │  │
//! │  └─────────────┘  └────────────────────────┘  │
//! └──────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::indexing::multimodal_search::{
//!     ModalityRegistry, ModalityConfig, CrossModalSearchEngine, SearchRequest,
//! };
//!
//! let mut registry = ModalityRegistry::new();
//! registry.register(ModalityConfig::new("text", 384));
//! registry.register(ModalityConfig::new("image", 512));
//!
//! let engine = CrossModalSearchEngine::new(registry);
//! engine.insert("text", "doc1", &text_embedding, None)?;
//! engine.insert("image", "img1", &image_embedding, None)?;
//!
//! // Cross-modal search: text query against all modalities
//! let results = engine.search(&query_embedding, &["text", "image"], 10)?;
//! ```

use crate::error::{NeedleError, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Modality Types
// ============================================================================

/// Supported modality types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModalityType {
    /// Text content.
    Text,
    /// Image content.
    Image,
    /// Audio content.
    Audio,
    /// Code content.
    Code,
    /// Custom modality.
    Custom,
}

impl std::fmt::Display for ModalityType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModalityType::Text => write!(f, "text"),
            ModalityType::Image => write!(f, "image"),
            ModalityType::Audio => write!(f, "audio"),
            ModalityType::Code => write!(f, "code"),
            ModalityType::Custom => write!(f, "custom"),
        }
    }
}

/// Configuration for a modality.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityConfig {
    /// Modality name (unique identifier).
    pub name: String,
    /// Modality type.
    pub modality_type: ModalityType,
    /// Native embedding dimension for this modality.
    pub dimensions: usize,
    /// Embedding model used for this modality.
    pub model: Option<String>,
    /// Whether vectors in this modality are normalized.
    pub normalized: bool,
    /// Shared embedding space name (modalities in the same space can be searched together directly).
    pub embedding_space: Option<String>,
}

impl ModalityConfig {
    /// Create a new modality config with the given name and dimensions.
    pub fn new(name: &str, dimensions: usize) -> Self {
        let modality_type = match name.to_lowercase().as_str() {
            "text" => ModalityType::Text,
            "image" | "img" => ModalityType::Image,
            "audio" => ModalityType::Audio,
            "code" => ModalityType::Code,
            _ => ModalityType::Custom,
        };
        Self {
            name: name.to_string(),
            modality_type,
            dimensions,
            model: None,
            normalized: true,
            embedding_space: None,
        }
    }

    /// Set the embedding model.
    #[must_use]
    pub fn with_model(mut self, model: &str) -> Self {
        self.model = Some(model.to_string());
        self
    }

    /// Set the shared embedding space.
    #[must_use]
    pub fn with_embedding_space(mut self, space: &str) -> Self {
        self.embedding_space = Some(space.to_string());
        self
    }
}

// ============================================================================
// Modality Registry
// ============================================================================

/// Registry for managing modalities and their configurations.
#[derive(Debug, Clone)]
pub struct ModalityRegistry {
    modalities: HashMap<String, ModalityConfig>,
}

impl ModalityRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            modalities: HashMap::new(),
        }
    }

    /// Register a new modality.
    pub fn register(&mut self, config: ModalityConfig) -> Result<()> {
        if self.modalities.contains_key(&config.name) {
            return Err(NeedleError::InvalidOperation(format!(
                "Modality '{}' already registered",
                config.name
            )));
        }
        self.modalities.insert(config.name.clone(), config);
        Ok(())
    }

    /// Get a modality by name.
    pub fn get(&self, name: &str) -> Option<&ModalityConfig> {
        self.modalities.get(name)
    }

    /// List all registered modalities.
    pub fn list(&self) -> Vec<&ModalityConfig> {
        self.modalities.values().collect()
    }

    /// Check if two modalities share an embedding space.
    pub fn shares_embedding_space(&self, a: &str, b: &str) -> bool {
        let ma = self.modalities.get(a);
        let mb = self.modalities.get(b);
        match (ma, mb) {
            (Some(a), Some(b)) => {
                a.embedding_space.is_some()
                    && a.embedding_space == b.embedding_space
            }
            _ => false,
        }
    }

    /// Get all modalities in a shared embedding space.
    pub fn modalities_in_space(&self, space: &str) -> Vec<&ModalityConfig> {
        self.modalities
            .values()
            .filter(|m| m.embedding_space.as_deref() == Some(space))
            .collect()
    }
}

impl Default for ModalityRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-Modal Vectors
// ============================================================================

/// A vector stored in the multi-modal index.
#[derive(Debug, Clone)]
struct StoredVector {
    id: String,
    vector: Vec<f32>,
    metadata: Option<serde_json::Value>,
    modality: String,
}

/// A search result from multi-modal search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalResult {
    /// Vector ID.
    pub id: String,
    /// Source modality.
    pub modality: String,
    /// Distance/similarity score.
    pub score: f32,
    /// Metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Configuration for cross-modal search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Default number of results per modality.
    pub per_modality_k: usize,
    /// Weight per modality for result fusion (name -> weight).
    pub modality_weights: HashMap<String, f32>,
    /// Fusion strategy.
    pub fusion: FusionStrategy,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            per_modality_k: 10,
            modality_weights: HashMap::new(),
            fusion: FusionStrategy::RoundRobin,
        }
    }
}

/// Strategy for fusing results from multiple modalities.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Interleave results from each modality.
    RoundRobin,
    /// Sort all results by score.
    ScoreSort,
    /// Reciprocal Rank Fusion.
    Rrf { k: f32 },
}

// ============================================================================
// Cross-Modal Search Engine
// ============================================================================

/// Engine for multi-modal vector search with cross-modal capabilities.
pub struct CrossModalSearchEngine {
    registry: ModalityRegistry,
    /// Per-modality vector stores.
    stores: RwLock<HashMap<String, Vec<StoredVector>>>,
    config: CrossModalConfig,
    /// Learned projection matrices: (source_modality, target_modality) -> matrix.
    /// Matrix is flattened row-major: target_dim rows × source_dim cols.
    projections: RwLock<HashMap<(String, String), ProjectionMatrix>>,
}

/// A learned projection matrix for cross-modal dimension alignment.
#[derive(Debug, Clone)]
pub struct ProjectionMatrix {
    /// Flattened matrix data (row-major: target_dim × source_dim).
    pub data: Vec<f32>,
    /// Source dimension.
    pub source_dim: usize,
    /// Target dimension.
    pub target_dim: usize,
}

impl ProjectionMatrix {
    /// Create a new projection matrix.
    pub fn new(data: Vec<f32>, source_dim: usize, target_dim: usize) -> Result<Self> {
        if data.len() != source_dim * target_dim {
            return Err(NeedleError::InvalidConfig(format!(
                "Projection matrix size mismatch: expected {}×{}={}, got {}",
                target_dim, source_dim, source_dim * target_dim, data.len()
            )));
        }
        Ok(Self { data, source_dim, target_dim })
    }

    /// Project a vector from source to target space.
    pub fn project(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.source_dim {
            return Err(NeedleError::DimensionMismatch {
                expected: self.source_dim,
                got: vector.len(),
            });
        }
        let mut result = vec![0.0f32; self.target_dim];
        for (i, val) in result.iter_mut().enumerate() {
            for (j, &v) in vector.iter().enumerate() {
                *val += v * self.data[i * self.source_dim + j];
            }
        }
        Ok(result)
    }
}

impl CrossModalSearchEngine {
    /// Create a new search engine with the given modality registry.
    pub fn new(registry: ModalityRegistry) -> Self {
        Self {
            registry,
            stores: RwLock::new(HashMap::new()),
            config: CrossModalConfig::default(),
            projections: RwLock::new(HashMap::new()),
        }
    }

    /// Create with custom config.
    pub fn with_config(registry: ModalityRegistry, config: CrossModalConfig) -> Self {
        Self {
            registry,
            stores: RwLock::new(HashMap::new()),
            config,
            projections: RwLock::new(HashMap::new()),
        }
    }

    /// Register a projection matrix for cross-modal alignment.
    pub fn register_projection(
        &self,
        from_modality: &str,
        to_modality: &str,
        matrix: ProjectionMatrix,
    ) -> Result<()> {
        let from_config = self.registry.get(from_modality).ok_or_else(|| {
            NeedleError::NotFound(format!("Modality '{from_modality}'"))
        })?;
        let to_config = self.registry.get(to_modality).ok_or_else(|| {
            NeedleError::NotFound(format!("Modality '{to_modality}'"))
        })?;
        if matrix.source_dim != from_config.dimensions || matrix.target_dim != to_config.dimensions {
            return Err(NeedleError::InvalidConfig(format!(
                "Projection dims ({}->{}) don't match modality dims ({} -> {})",
                matrix.source_dim, matrix.target_dim,
                from_config.dimensions, to_config.dimensions
            )));
        }
        self.projections.write().insert(
            (from_modality.to_string(), to_modality.to_string()),
            matrix,
        );
        Ok(())
    }

    /// Project a query vector from one modality to another, if a projection exists.
    pub fn project_query(
        &self,
        query: &[f32],
        from_modality: &str,
        to_modality: &str,
    ) -> Option<Vec<f32>> {
        let key = (from_modality.to_string(), to_modality.to_string());
        let projections = self.projections.read();
        projections.get(&key).and_then(|m| m.project(query).ok())
    }

    /// Insert a vector into a specific modality's store.
    pub fn insert(
        &self,
        modality: &str,
        id: &str,
        vector: &[f32],
        metadata: Option<serde_json::Value>,
    ) -> Result<()> {
        let config = self.registry.get(modality).ok_or_else(|| {
            NeedleError::NotFound(format!("Modality '{modality}'"))
        })?;

        if vector.len() != config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: config.dimensions,
                got: vector.len(),
            });
        }

        let mut stores = self.stores.write();
        let store = stores.entry(modality.to_string()).or_default();

        // Check for duplicate
        if store.iter().any(|v| v.id == id) {
            return Err(NeedleError::VectorAlreadyExists(id.to_string()));
        }

        store.push(StoredVector {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
            modality: modality.to_string(),
        });

        Ok(())
    }

    /// Delete a vector from a modality's store.
    pub fn delete(&self, modality: &str, id: &str) -> Result<bool> {
        let mut stores = self.stores.write();
        if let Some(store) = stores.get_mut(modality) {
            let before = store.len();
            store.retain(|v| v.id != id);
            Ok(store.len() < before)
        } else {
            Ok(false)
        }
    }

    /// Search across specified modalities.
    pub fn search(
        &self,
        query: &[f32],
        modalities: &[&str],
        k: usize,
    ) -> Result<Vec<MultiModalResult>> {
        let stores = self.stores.read();
        let mut all_results = Vec::new();

        for &modality_name in modalities {
            let _config = self.registry.get(modality_name).ok_or_else(|| {
                NeedleError::NotFound(format!("Modality '{modality_name}'"))
            })?;

            if let Some(store) = stores.get(modality_name) {
                let weight = self
                    .config
                    .modality_weights
                    .get(modality_name)
                    .copied()
                    .unwrap_or(1.0);

                let mut modality_results: Vec<MultiModalResult> = store
                    .iter()
                    .map(|sv| {
                        let score = cosine_similarity(query, &sv.vector) * weight;
                        MultiModalResult {
                            id: sv.id.clone(),
                            modality: sv.modality.clone(),
                            score,
                            metadata: sv.metadata.clone(),
                        }
                    })
                    .collect();

                modality_results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                modality_results.truncate(self.config.per_modality_k);
                all_results.push(modality_results);
            }
        }

        let fused = match self.config.fusion {
            FusionStrategy::ScoreSort => {
                let mut flat: Vec<_> = all_results.into_iter().flatten().collect();
                flat.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                flat.truncate(k);
                flat
            }
            FusionStrategy::RoundRobin => {
                let mut fused = Vec::new();
                let max_len = all_results.iter().map(|r| r.len()).max().unwrap_or(0);
                for i in 0..max_len {
                    for results in &all_results {
                        if i < results.len() {
                            fused.push(results[i].clone());
                        }
                    }
                    if fused.len() >= k {
                        break;
                    }
                }
                fused.truncate(k);
                fused
            }
            FusionStrategy::Rrf { k: rrf_k } => {
                let mut rrf_scores: HashMap<String, f32> = HashMap::new();
                let mut result_map: HashMap<String, MultiModalResult> = HashMap::new();

                for results in &all_results {
                    for (rank, result) in results.iter().enumerate() {
                        let rrf_score = 1.0 / (rrf_k + rank as f32 + 1.0);
                        *rrf_scores.entry(result.id.clone()).or_default() += rrf_score;
                        result_map
                            .entry(result.id.clone())
                            .or_insert_with(|| result.clone());
                    }
                }

                let mut fused: Vec<_> = rrf_scores
                    .into_iter()
                    .filter_map(|(id, score)| {
                        result_map.remove(&id).map(|mut r| {
                            r.score = score;
                            r
                        })
                    })
                    .collect();

                fused.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                fused.truncate(k);
                fused
            }
        };

        Ok(fused)
    }

    /// Get count of vectors per modality.
    pub fn counts(&self) -> HashMap<String, usize> {
        self.stores
            .read()
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }

    /// Get the modality registry.
    pub fn registry(&self) -> &ModalityRegistry {
        &self.registry
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let min_len = a.len().min(b.len());
    let dot: f32 = a[..min_len].iter().zip(&b[..min_len]).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_registry() -> ModalityRegistry {
        let mut reg = ModalityRegistry::new();
        reg.register(
            ModalityConfig::new("text", 4).with_embedding_space("clip"),
        ).expect("register");
        reg.register(
            ModalityConfig::new("image", 4).with_embedding_space("clip"),
        ).expect("register");
        reg
    }

    #[test]
    fn test_modality_registry() {
        let reg = make_registry();
        assert!(reg.get("text").is_some());
        assert!(reg.get("image").is_some());
        assert!(reg.get("video").is_none());
        assert_eq!(reg.list().len(), 2);
    }

    #[test]
    fn test_shared_embedding_space() {
        let reg = make_registry();
        assert!(reg.shares_embedding_space("text", "image"));
        assert!(!reg.shares_embedding_space("text", "nonexistent"));
    }

    #[test]
    fn test_insert_and_search() {
        let reg = make_registry();
        let engine = CrossModalSearchEngine::new(reg);

        engine
            .insert("text", "t1", &[1.0, 0.0, 0.0, 0.0], None)
            .expect("insert");
        engine
            .insert("text", "t2", &[0.0, 1.0, 0.0, 0.0], None)
            .expect("insert");
        engine
            .insert("image", "i1", &[0.9, 0.1, 0.0, 0.0], None)
            .expect("insert");

        // Search across both modalities
        let results = engine
            .search(&[1.0, 0.0, 0.0, 0.0], &["text", "image"], 5)
            .expect("search");

        assert!(!results.is_empty());
        // t1 should be most similar to query
        assert!(results.iter().any(|r| r.id == "t1"));
    }

    #[test]
    fn test_dimension_mismatch() {
        let reg = make_registry();
        let engine = CrossModalSearchEngine::new(reg);

        let result = engine.insert("text", "t1", &[1.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete() {
        let reg = make_registry();
        let engine = CrossModalSearchEngine::new(reg);

        engine.insert("text", "t1", &[1.0, 0.0, 0.0, 0.0], None).expect("insert");
        assert!(engine.delete("text", "t1").expect("delete"));
        assert!(!engine.delete("text", "t1").expect("delete again"));
    }

    #[test]
    fn test_rrf_fusion() {
        let mut reg = ModalityRegistry::new();
        reg.register(ModalityConfig::new("a", 4)).expect("ok");
        reg.register(ModalityConfig::new("b", 4)).expect("ok");

        let config = CrossModalConfig {
            fusion: FusionStrategy::Rrf { k: 60.0 },
            ..Default::default()
        };
        let engine = CrossModalSearchEngine::with_config(reg, config);

        engine.insert("a", "v1", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");
        engine.insert("a", "v2", &[0.0, 1.0, 0.0, 0.0], None).expect("ok");
        engine.insert("b", "v1", &[0.9, 0.1, 0.0, 0.0], None).expect("ok");
        engine.insert("b", "v3", &[0.0, 0.0, 1.0, 0.0], None).expect("ok");

        let results = engine.search(&[1.0, 0.0, 0.0, 0.0], &["a", "b"], 3).expect("search");
        assert!(!results.is_empty());
        // v1 appears in both modalities, should rank highly with RRF
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_counts() {
        let reg = make_registry();
        let engine = CrossModalSearchEngine::new(reg);

        engine.insert("text", "t1", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");
        engine.insert("text", "t2", &[0.0, 1.0, 0.0, 0.0], None).expect("ok");
        engine.insert("image", "i1", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");

        let counts = engine.counts();
        assert_eq!(counts.get("text"), Some(&2));
        assert_eq!(counts.get("image"), Some(&1));
    }

    #[test]
    fn test_duplicate_insert() {
        let reg = make_registry();
        let engine = CrossModalSearchEngine::new(reg);

        engine.insert("text", "t1", &[1.0, 0.0, 0.0, 0.0], None).expect("ok");
        let result = engine.insert("text", "t1", &[0.0, 1.0, 0.0, 0.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_projection_matrix() {
        // 4D → 4D identity projection
        let matrix = ProjectionMatrix::new(
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            4, 4,
        ).expect("matrix");

        let result = matrix.project(&[1.0, 2.0, 3.0, 4.0]).expect("project");
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_register_and_use_projection() {
        let mut reg = ModalityRegistry::new();
        reg.register(ModalityConfig::new("text", 4)).expect("ok");
        reg.register(ModalityConfig::new("image", 4)).expect("ok");
        let engine = CrossModalSearchEngine::new(reg);

        // Register identity projection text → image
        let matrix = ProjectionMatrix::new(
            vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
            4, 4,
        ).expect("matrix");
        engine.register_projection("text", "image", matrix).expect("register");

        let projected = engine.project_query(&[1.0, 0.0, 0.0, 0.0], "text", "image");
        assert!(projected.is_some());
        assert!((projected.as_ref().expect("p")[0] - 1.0).abs() < 1e-6);
    }
}
