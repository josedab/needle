#![allow(clippy::unwrap_used)]
//! v1.0 API Stability Manifest
//!
//! Tracks stability annotations for public types, manages deprecation
//! lifecycle, and provides migration helpers for API evolution.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::api_stability::{
//!     ApiManifest, StabilityLevel, ApiEntry, DeprecationInfo,
//! };
//!
//! let manifest = ApiManifest::default_manifest();
//! assert!(manifest.is_stable("Database"));
//! assert!(manifest.is_stable("Collection"));
//!
//! let deprecated = manifest.deprecated_items();
//! for item in deprecated {
//!     println!("Deprecated: {} → use {} instead", item.name, item.replacement.as_deref().unwrap_or("N/A"));
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Stability Levels ─────────────────────────────────────────────────────────

/// API stability classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StabilityLevel {
    /// Covered by semver. Breaking changes only in major versions.
    Stable,
    /// Functional but may change in minor versions. Use with caution.
    Beta,
    /// May change or be removed without notice.
    Experimental,
    /// Scheduled for removal. Use the replacement instead.
    Deprecated,
    /// Internal implementation detail. Not part of public API.
    Internal,
}

impl std::fmt::Display for StabilityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Stable => write!(f, "stable"),
            Self::Beta => write!(f, "beta"),
            Self::Experimental => write!(f, "experimental"),
            Self::Deprecated => write!(f, "deprecated"),
            Self::Internal => write!(f, "internal"),
        }
    }
}

// ── API Entry ────────────────────────────────────────────────────────────────

/// A single public API item tracked in the manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEntry {
    /// Fully qualified name (e.g., "Database", "Collection::search").
    pub name: String,
    /// Stability level.
    pub stability: StabilityLevel,
    /// Module path.
    pub module: String,
    /// API kind (struct, fn, trait, enum).
    pub kind: ApiKind,
    /// Version when this item was introduced.
    pub since: String,
    /// Deprecation info (if deprecated).
    pub deprecation: Option<DeprecationInfo>,
    /// Brief description.
    pub description: String,
}

/// Kind of API item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiKind {
    Struct,
    Enum,
    Trait,
    Function,
    Method,
    Module,
    Type,
}

/// Deprecation information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeprecationInfo {
    /// Version when deprecated.
    pub since: String,
    /// Version when it will be removed.
    pub removal: Option<String>,
    /// Replacement to use instead.
    pub replacement: Option<String>,
    /// Migration guide URL or instructions.
    pub migration: Option<String>,
}

// ── Migration Helper ─────────────────────────────────────────────────────────

/// A migration step for upgrading between versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationStep {
    /// From version.
    pub from: String,
    /// To version.
    pub to: String,
    /// Items changed.
    pub changes: Vec<ApiChange>,
}

/// A single API change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiChange {
    /// What changed.
    pub kind: ChangeKind,
    /// Old name/signature.
    pub old: String,
    /// New name/signature.
    pub new: String,
    /// Migration instructions.
    pub instructions: String,
}

/// Kind of API change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeKind {
    Renamed,
    Removed,
    SignatureChanged,
    MovedModule,
    BehaviorChanged,
}

// ── API Manifest ─────────────────────────────────────────────────────────────

/// Registry of all public API items with stability annotations.
pub struct ApiManifest {
    entries: HashMap<String, ApiEntry>,
    migrations: Vec<MigrationStep>,
    version: String,
}

impl ApiManifest {
    /// Create an empty manifest.
    pub fn new(version: &str) -> Self {
        Self {
            entries: HashMap::new(),
            migrations: Vec::new(),
            version: version.into(),
        }
    }

    /// Create the default manifest for Needle v0.1.0 with all core types classified.
    pub fn default_manifest() -> Self {
        let mut m = Self::new("0.1.0");

        // Core stable types
        for (name, module, kind, desc) in STABLE_CORE {
            m.register(ApiEntry {
                name: name.to_string(),
                stability: StabilityLevel::Stable,
                module: module.to_string(),
                kind: **kind,
                since: "0.1.0".into(),
                deprecation: None,
                description: desc.to_string(),
            });
        }

        // Beta types
        for (name, module, kind, desc) in BETA_TYPES {
            m.register(ApiEntry {
                name: name.to_string(),
                stability: StabilityLevel::Beta,
                module: module.to_string(),
                kind: **kind,
                since: "0.1.0".into(),
                deprecation: None,
                description: desc.to_string(),
            });
        }

        // Deprecated items
        m.register(ApiEntry {
            name: "drop_collection".into(),
            stability: StabilityLevel::Deprecated,
            module: "database".into(),
            kind: ApiKind::Method,
            since: "0.1.0".into(),
            deprecation: Some(DeprecationInfo {
                since: "0.1.0".into(),
                removal: Some("1.0.0".into()),
                replacement: Some("delete_collection".into()),
                migration: Some("Replace db.drop_collection(name) with db.delete_collection(name)".into()),
            }),
            description: "Delete a collection (deprecated alias)".into(),
        });

        m
    }

    /// Register an API entry.
    pub fn register(&mut self, entry: ApiEntry) {
        self.entries.insert(entry.name.clone(), entry);
    }

    /// Check if an item is stable.
    pub fn is_stable(&self, name: &str) -> bool {
        self.entries.get(name).map_or(false, |e| e.stability == StabilityLevel::Stable)
    }

    /// Check if an item is deprecated.
    pub fn is_deprecated(&self, name: &str) -> bool {
        self.entries.get(name).map_or(false, |e| e.stability == StabilityLevel::Deprecated)
    }

    /// Get all deprecated items.
    pub fn deprecated_items(&self) -> Vec<&ApiEntry> {
        self.entries.values()
            .filter(|e| e.stability == StabilityLevel::Deprecated)
            .collect()
    }

    /// Get all items at a given stability level.
    pub fn items_at_level(&self, level: StabilityLevel) -> Vec<&ApiEntry> {
        self.entries.values().filter(|e| e.stability == level).collect()
    }

    /// Get an entry by name.
    pub fn get(&self, name: &str) -> Option<&ApiEntry> {
        self.entries.get(name)
    }

    /// Total entry count.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether manifest is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Add a migration step.
    pub fn add_migration(&mut self, step: MigrationStep) {
        self.migrations.push(step);
    }

    /// Get migrations between two versions.
    pub fn migrations_between(&self, from: &str, to: &str) -> Vec<&MigrationStep> {
        self.migrations.iter()
            .filter(|m| m.from == from && m.to == to)
            .collect()
    }

    /// Generate a stability report.
    pub fn report(&self) -> StabilityReport {
        let mut counts = HashMap::new();
        for entry in self.entries.values() {
            *counts.entry(entry.stability).or_insert(0usize) += 1;
        }
        StabilityReport {
            version: self.version.clone(),
            total_items: self.entries.len(),
            stable: *counts.get(&StabilityLevel::Stable).unwrap_or(&0),
            beta: *counts.get(&StabilityLevel::Beta).unwrap_or(&0),
            experimental: *counts.get(&StabilityLevel::Experimental).unwrap_or(&0),
            deprecated: *counts.get(&StabilityLevel::Deprecated).unwrap_or(&0),
            coverage: if self.entries.is_empty() { 0.0 } else {
                *counts.get(&StabilityLevel::Stable).unwrap_or(&0) as f32 / self.entries.len() as f32
            },
        }
    }
}

/// Stability report summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityReport {
    pub version: String,
    pub total_items: usize,
    pub stable: usize,
    pub beta: usize,
    pub experimental: usize,
    pub deprecated: usize,
    pub coverage: f32,
}

const STABLE_CORE: &[(&str, &str, &ApiKind, &str)] = &[
    ("Database", "database", &ApiKind::Struct, "Main entry point for managing collections"),
    ("DatabaseConfig", "database", &ApiKind::Struct, "Database configuration options"),
    ("Collection", "collection", &ApiKind::Struct, "Vector collection with HNSW index"),
    ("CollectionConfig", "collection::config", &ApiKind::Struct, "Collection configuration"),
    ("CollectionRef", "database::collection_ref", &ApiKind::Struct, "Thread-safe collection handle"),
    ("SearchResult", "collection::search", &ApiKind::Struct, "Search result with id, distance, metadata"),
    ("SearchParams", "database::collection_ref", &ApiKind::Struct, "Fluent search parameter builder"),
    ("Filter", "metadata", &ApiKind::Struct, "MongoDB-style metadata filter"),
    ("MetadataStore", "metadata", &ApiKind::Struct, "Metadata storage engine"),
    ("DistanceFunction", "distance", &ApiKind::Enum, "Distance metric selection"),
    ("NeedleError", "error", &ApiKind::Enum, "Structured error type"),
    ("ErrorCode", "error", &ApiKind::Enum, "Numeric error code categories"),
    ("Result", "error", &ApiKind::Type, "Type alias for Result<T, NeedleError>"),
    ("HnswConfig", "indexing::hnsw", &ApiKind::Struct, "HNSW index configuration"),
    ("HnswIndex", "indexing::hnsw", &ApiKind::Struct, "HNSW index implementation"),
    ("ScalarQuantizer", "indexing::quantization", &ApiKind::Struct, "Scalar quantization (4x compression)"),
    ("ProductQuantizer", "indexing::quantization", &ApiKind::Struct, "Product quantization (8-32x)"),
    ("SparseVector", "indexing::sparse", &ApiKind::Struct, "Sparse vector representation"),
    ("SparseIndex", "indexing::sparse", &ApiKind::Struct, "Sparse vector index"),
    ("MultiVector", "indexing::multivec", &ApiKind::Struct, "Multi-vector document (ColBERT)"),
];

const BETA_TYPES: &[(&str, &str, &ApiKind, &str)] = &[
    ("Bm25Index", "hybrid", &ApiKind::Struct, "BM25 text search index"),
    ("AsyncDatabase", "async_api", &ApiKind::Struct, "Async database wrapper"),
    ("ServerConfig", "server", &ApiKind::Struct, "HTTP server configuration"),
    ("TextEmbedder", "embeddings", &ApiKind::Struct, "ONNX embedding generator"),
    ("AutoEmbedder", "ml::auto_embed", &ApiKind::Struct, "Automatic embedding selection"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_manifest() {
        let m = ApiManifest::default_manifest();
        assert!(m.is_stable("Database"));
        assert!(m.is_stable("Collection"));
        assert!(m.is_stable("Filter"));
        assert!(m.is_stable("NeedleError"));
        assert!(!m.is_stable("Bm25Index")); // beta
        assert!(m.is_deprecated("drop_collection"));
    }

    #[test]
    fn test_stability_report() {
        let m = ApiManifest::default_manifest();
        let report = m.report();
        assert!(report.stable >= 15);
        assert!(report.beta >= 3);
        assert!(report.deprecated >= 1);
        assert!(report.coverage > 0.5);
    }

    #[test]
    fn test_deprecated_items() {
        let m = ApiManifest::default_manifest();
        let deprecated = m.deprecated_items();
        assert!(!deprecated.is_empty());
        assert!(deprecated[0].deprecation.is_some());
        assert!(deprecated[0].deprecation.as_ref().unwrap().replacement.is_some());
    }

    #[test]
    fn test_items_by_level() {
        let m = ApiManifest::default_manifest();
        let stable = m.items_at_level(StabilityLevel::Stable);
        assert!(stable.len() >= 15);
    }

    #[test]
    fn test_migration() {
        let mut m = ApiManifest::default_manifest();
        m.add_migration(MigrationStep {
            from: "0.1.0".into(),
            to: "1.0.0".into(),
            changes: vec![ApiChange {
                kind: ChangeKind::Renamed,
                old: "drop_collection".into(),
                new: "delete_collection".into(),
                instructions: "Simple rename".into(),
            }],
        });
        let migrations = m.migrations_between("0.1.0", "1.0.0");
        assert_eq!(migrations.len(), 1);
    }

    #[test]
    fn test_custom_entry() {
        let mut m = ApiManifest::new("0.2.0");
        m.register(ApiEntry {
            name: "CustomType".into(),
            stability: StabilityLevel::Experimental,
            module: "experimental".into(),
            kind: ApiKind::Struct,
            since: "0.2.0".into(),
            deprecation: None,
            description: "Test".into(),
        });
        assert!(!m.is_stable("CustomType"));
        assert_eq!(m.len(), 1);
    }
}
