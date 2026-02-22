#![allow(clippy::unwrap_used)]
//! Plugin Marketplace & Extension API
//!
//! Stable plugin trait API with versioned ABI, manifest format, and registry
//! discovery. Enables community extensions for distance functions, index types,
//! storage backends, and embedding providers.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::plugin_api::{
//!     PluginManifest, PluginRegistry, PluginCapability,
//!     PluginVersion, DistancePluginAdapter, IndexPluginAdapter,
//! };
//!
//! let mut registry = PluginRegistry::new();
//!
//! // Register a custom distance function plugin
//! let manifest = PluginManifest::builder("hamming-distance")
//!     .version(PluginVersion::new(1, 0, 0))
//!     .author("Community")
//!     .description("Hamming distance for binary vectors")
//!     .capability(PluginCapability::Distance)
//!     .build();
//!
//! registry.register(manifest).unwrap();
//!
//! let plugins = registry.list();
//! assert_eq!(plugins.len(), 1);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Plugin Version ───────────────────────────────────────────────────────────

/// Semantic version for a plugin.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PluginVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (new features, backward compatible).
    pub minor: u32,
    /// Patch version (bug fixes).
    pub patch: u32,
}

impl PluginVersion {
    /// Create a new version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Check if this version is compatible with a required version.
    /// Compatible if same major and >= minor.patch.
    pub fn is_compatible_with(&self, required: &PluginVersion) -> bool {
        self.major == required.major
            && (self.minor > required.minor
                || (self.minor == required.minor && self.patch >= required.patch))
    }
}

impl fmt::Display for PluginVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

// ── Plugin Capabilities ──────────────────────────────────────────────────────

/// What a plugin provides.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginCapability {
    /// Custom distance function.
    Distance,
    /// Custom index implementation.
    Index,
    /// Custom storage backend.
    Storage,
    /// Custom embedding provider.
    Embedding,
    /// Custom pre-processing pipeline.
    PreProcessor,
    /// Custom post-processing pipeline.
    PostProcessor,
    /// Custom metadata filter.
    Filter,
    /// Custom serialization format.
    Serializer,
}

impl fmt::Display for PluginCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Distance => write!(f, "distance"),
            Self::Index => write!(f, "index"),
            Self::Storage => write!(f, "storage"),
            Self::Embedding => write!(f, "embedding"),
            Self::PreProcessor => write!(f, "pre-processor"),
            Self::PostProcessor => write!(f, "post-processor"),
            Self::Filter => write!(f, "filter"),
            Self::Serializer => write!(f, "serializer"),
        }
    }
}

// ── Plugin Status ────────────────────────────────────────────────────────────

/// Registration status of a plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginStatus {
    /// Plugin is registered and available.
    Active,
    /// Plugin is registered but disabled.
    Disabled,
    /// Plugin failed to load.
    Failed,
    /// Plugin is deprecated.
    Deprecated,
}

// ── Plugin Manifest ──────────────────────────────────────────────────────────

/// Metadata describing a plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Plugin version.
    pub version: PluginVersion,
    /// Author or organization.
    pub author: String,
    /// Description.
    pub description: String,
    /// Capabilities provided.
    pub capabilities: Vec<PluginCapability>,
    /// License (SPDX identifier).
    pub license: Option<String>,
    /// Homepage URL.
    pub homepage: Option<String>,
    /// Repository URL.
    pub repository: Option<String>,
    /// Minimum Needle version required.
    pub min_needle_version: Option<PluginVersion>,
    /// Plugin dependencies (plugin_id → required version).
    pub dependencies: HashMap<String, PluginVersion>,
    /// Plugin configuration schema (JSON Schema).
    pub config_schema: Option<serde_json::Value>,
}

impl PluginManifest {
    /// Create a builder.
    pub fn builder(id: impl Into<String>) -> PluginManifestBuilder {
        PluginManifestBuilder::new(id)
    }
}

/// Builder for `PluginManifest`.
pub struct PluginManifestBuilder {
    manifest: PluginManifest,
}

impl PluginManifestBuilder {
    fn new(id: impl Into<String>) -> Self {
        let id = id.into();
        Self {
            manifest: PluginManifest {
                id: id.clone(),
                name: id,
                version: PluginVersion::new(0, 1, 0),
                author: String::new(),
                description: String::new(),
                capabilities: Vec::new(),
                license: None,
                homepage: None,
                repository: None,
                min_needle_version: None,
                dependencies: HashMap::new(),
                config_schema: None,
            },
        }
    }

    /// Set version.
    #[must_use]
    pub fn version(mut self, version: PluginVersion) -> Self {
        self.manifest.version = version;
        self
    }

    /// Set human-readable name.
    #[must_use]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.manifest.name = name.into();
        self
    }

    /// Set author.
    #[must_use]
    pub fn author(mut self, author: impl Into<String>) -> Self {
        self.manifest.author = author.into();
        self
    }

    /// Set description.
    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.manifest.description = desc.into();
        self
    }

    /// Add a capability.
    #[must_use]
    pub fn capability(mut self, cap: PluginCapability) -> Self {
        self.manifest.capabilities.push(cap);
        self
    }

    /// Set license.
    #[must_use]
    pub fn license(mut self, license: impl Into<String>) -> Self {
        self.manifest.license = Some(license.into());
        self
    }

    /// Set homepage.
    #[must_use]
    pub fn homepage(mut self, url: impl Into<String>) -> Self {
        self.manifest.homepage = Some(url.into());
        self
    }

    /// Set repository.
    #[must_use]
    pub fn repository(mut self, url: impl Into<String>) -> Self {
        self.manifest.repository = Some(url.into());
        self
    }

    /// Add a dependency.
    #[must_use]
    pub fn depends_on(mut self, plugin_id: impl Into<String>, version: PluginVersion) -> Self {
        self.manifest.dependencies.insert(plugin_id.into(), version);
        self
    }

    /// Set min Needle version.
    #[must_use]
    pub fn min_needle_version(mut self, version: PluginVersion) -> Self {
        self.manifest.min_needle_version = Some(version);
        self
    }

    /// Build the manifest.
    pub fn build(self) -> PluginManifest {
        self.manifest
    }
}

// ── Plugin Traits ────────────────────────────────────────────────────────────

/// Trait for custom distance function plugins.
pub trait DistancePluginAdapter: Send + Sync {
    /// Compute distance between two vectors.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;

    /// Name of this distance function.
    fn name(&self) -> &str;

    /// Whether lower values mean more similar (true for most distances).
    fn lower_is_better(&self) -> bool {
        true
    }
}

/// Trait for custom index plugins.
pub trait IndexPluginAdapter: Send + Sync {
    /// Insert a vector.
    fn insert(&mut self, id: &str, vector: &[f32]) -> Result<()>;

    /// Search for nearest neighbors.
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>>;

    /// Delete a vector.
    fn delete(&mut self, id: &str) -> Result<bool>;

    /// Name of this index type.
    fn name(&self) -> &str;

    /// Number of indexed vectors.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Trait for custom storage backend plugins.
pub trait StoragePluginAdapter: Send + Sync {
    /// Write data to storage.
    fn write(&mut self, key: &str, data: &[u8]) -> Result<()>;

    /// Read data from storage.
    fn read(&self, key: &str) -> Result<Option<Vec<u8>>>;

    /// Delete data from storage.
    fn delete(&mut self, key: &str) -> Result<bool>;

    /// List all keys.
    fn list_keys(&self) -> Result<Vec<String>>;

    /// Name of this storage backend.
    fn name(&self) -> &str;
}

/// Trait for custom embedding provider plugins.
pub trait EmbeddingPluginAdapter: Send + Sync {
    /// Generate embeddings for a batch of texts.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Output dimensions.
    fn dimensions(&self) -> usize;

    /// Model name.
    fn model_name(&self) -> &str;
}

// ── Plugin Registration Entry ────────────────────────────────────────────────

/// A registered plugin entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginEntry {
    /// Plugin manifest.
    pub manifest: PluginManifest,
    /// Registration status.
    pub status: PluginStatus,
    /// When this plugin was registered.
    pub registered_at: SystemTime,
    /// Number of times this plugin was invoked.
    pub invocation_count: u64,
}

// ── Plugin Registry ──────────────────────────────────────────────────────────

/// Registry for managing plugins.
pub struct PluginRegistry {
    plugins: HashMap<String, PluginEntry>,
    distance_plugins: HashMap<String, Box<dyn DistancePluginAdapter>>,
}

impl PluginRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            distance_plugins: HashMap::new(),
        }
    }

    /// Register a plugin manifest (metadata only).
    pub fn register(&mut self, manifest: PluginManifest) -> Result<()> {
        if self.plugins.contains_key(&manifest.id) {
            return Err(NeedleError::Conflict(format!(
                "Plugin '{}' is already registered",
                manifest.id
            )));
        }

        // Check dependencies
        for (dep_id, required_version) in &manifest.dependencies {
            match self.plugins.get(dep_id) {
                None => {
                    return Err(NeedleError::NotFound(format!(
                        "Plugin dependency '{}' not found",
                        dep_id
                    )));
                }
                Some(dep) => {
                    if !dep.manifest.version.is_compatible_with(required_version) {
                        return Err(NeedleError::InvalidConfig(format!(
                            "Plugin '{}' requires {} >= {}, found {}",
                            manifest.id, dep_id, required_version, dep.manifest.version
                        )));
                    }
                }
            }
        }

        self.plugins.insert(
            manifest.id.clone(),
            PluginEntry {
                manifest,
                status: PluginStatus::Active,
                registered_at: SystemTime::now(),
                invocation_count: 0,
            },
        );

        Ok(())
    }

    /// Register a distance function plugin with implementation.
    pub fn register_distance(
        &mut self,
        manifest: PluginManifest,
        adapter: Box<dyn DistancePluginAdapter>,
    ) -> Result<()> {
        let id = manifest.id.clone();
        self.register(manifest)?;
        self.distance_plugins.insert(id, adapter);
        Ok(())
    }

    /// Get a distance plugin adapter.
    pub fn get_distance(&self, id: &str) -> Option<&dyn DistancePluginAdapter> {
        self.distance_plugins.get(id).map(|b| b.as_ref())
    }

    /// Unregister a plugin.
    pub fn unregister(&mut self, id: &str) -> Result<bool> {
        // Check if any other plugin depends on this one
        for (other_id, entry) in &self.plugins {
            if other_id != id && entry.manifest.dependencies.contains_key(id) {
                return Err(NeedleError::InvalidOperation(format!(
                    "Cannot unregister '{}': plugin '{}' depends on it",
                    id, other_id
                )));
            }
        }

        self.distance_plugins.remove(id);
        Ok(self.plugins.remove(id).is_some())
    }

    /// Enable or disable a plugin.
    pub fn set_status(&mut self, id: &str, status: PluginStatus) -> Result<()> {
        let entry = self
            .plugins
            .get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin '{id}' not found")))?;
        entry.status = status;
        Ok(())
    }

    /// List all registered plugins.
    pub fn list(&self) -> Vec<&PluginEntry> {
        self.plugins.values().collect()
    }

    /// Find plugins by capability.
    pub fn find_by_capability(&self, capability: &PluginCapability) -> Vec<&PluginEntry> {
        self.plugins
            .values()
            .filter(|e| e.manifest.capabilities.contains(capability))
            .collect()
    }

    /// Get a plugin by ID.
    pub fn get(&self, id: &str) -> Option<&PluginEntry> {
        self.plugins.get(id)
    }

    /// Get the number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Validate all plugin dependencies are satisfied.
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (id, entry) in &self.plugins {
            for (dep_id, required) in &entry.manifest.dependencies {
                match self.plugins.get(dep_id) {
                    None => errors.push(format!(
                        "Plugin '{}': missing dependency '{}'",
                        id, dep_id
                    )),
                    Some(dep) => {
                        if !dep.manifest.version.is_compatible_with(required) {
                            errors.push(format!(
                                "Plugin '{}': dependency '{}' version {} incompatible with required {}",
                                id, dep_id, dep.manifest.version, required
                            ));
                        }
                    }
                }
            }
        }
        errors
    }

    /// Hot-reload a plugin: unregister and re-register with updated manifest.
    /// Preserves dependent relationships if versions are compatible.
    pub fn hot_reload(&mut self, manifest: PluginManifest) -> Result<()> {
        let id = manifest.id.clone();
        // Check if new version is compatible with dependents
        if let Some(old_entry) = self.plugins.get(&id) {
            let old_version = &old_entry.manifest.version;
            if manifest.version.major != old_version.major {
                // Check if any dependent needs the old major version
                for (dep_id, entry) in &self.plugins {
                    if let Some(required) = entry.manifest.dependencies.get(&id) {
                        if !manifest.version.is_compatible_with(required) {
                            return Err(NeedleError::InvalidArgument(format!(
                                "Plugin '{dep_id}' requires '{id}' version {required}, new version {} incompatible",
                                manifest.version
                            )));
                        }
                    }
                }
            }
        }

        // Remove old entry (skip dependency check for reload)
        self.plugins.remove(&id);
        self.distance_plugins.remove(&id);

        // Re-register
        let entry = PluginEntry {
            manifest,
            status: PluginStatus::Active,
            registered_at: SystemTime::now(),
            invocation_count: 0,
        };
        self.plugins.insert(id, entry);
        Ok(())
    }

    /// Get a health summary of all plugins.
    pub fn health(&self) -> PluginRegistryHealth {
        let total = self.plugins.len();
        let active = self.plugins.values().filter(|e| e.status == PluginStatus::Active).count();
        let disabled = self.plugins.values().filter(|e| e.status == PluginStatus::Disabled).count();
        let errors = self.validate();
        PluginRegistryHealth {
            total_plugins: total,
            active,
            disabled,
            dependency_errors: errors.len(),
            error_details: errors,
        }
    }
}

/// Plugin registry health summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginRegistryHealth {
    pub total_plugins: usize,
    pub active: usize,
    pub disabled: usize,
    pub dependency_errors: usize,
    pub error_details: Vec<String>,
}

impl Default for PluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_manifest(id: &str) -> PluginManifest {
        PluginManifest::builder(id)
            .version(PluginVersion::new(1, 0, 0))
            .author("Test Author")
            .description("Test plugin")
            .capability(PluginCapability::Distance)
            .build()
    }

    #[test]
    fn test_register() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("test-plugin")).unwrap();
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_duplicate_register() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("test")).unwrap();
        assert!(registry.register(test_manifest("test")).is_err());
    }

    #[test]
    fn test_unregister() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("test")).unwrap();
        assert!(registry.unregister("test").unwrap());
        assert!(registry.is_empty());
    }

    #[test]
    fn test_find_by_capability() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("distance-1")).unwrap();
        registry
            .register(
                PluginManifest::builder("index-1")
                    .capability(PluginCapability::Index)
                    .build(),
            )
            .unwrap();

        let distance_plugins = registry.find_by_capability(&PluginCapability::Distance);
        assert_eq!(distance_plugins.len(), 1);
    }

    #[test]
    fn test_dependency_check() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("base")).unwrap();

        let dependent = PluginManifest::builder("dependent")
            .depends_on("base", PluginVersion::new(1, 0, 0))
            .build();
        registry.register(dependent).unwrap();

        // Can't unregister base because dependent depends on it
        assert!(registry.unregister("base").is_err());
    }

    #[test]
    fn test_missing_dependency() {
        let mut registry = PluginRegistry::new();

        let plugin = PluginManifest::builder("needs-base")
            .depends_on("nonexistent", PluginVersion::new(1, 0, 0))
            .build();

        assert!(registry.register(plugin).is_err());
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = PluginVersion::new(1, 2, 3);
        assert!(v1.is_compatible_with(&PluginVersion::new(1, 2, 0)));
        assert!(v1.is_compatible_with(&PluginVersion::new(1, 2, 3)));
        assert!(!v1.is_compatible_with(&PluginVersion::new(2, 0, 0)));
        assert!(!v1.is_compatible_with(&PluginVersion::new(1, 3, 0)));
    }

    #[test]
    fn test_set_status() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("test")).unwrap();

        registry
            .set_status("test", PluginStatus::Disabled)
            .unwrap();
        assert_eq!(
            registry.get("test").unwrap().status,
            PluginStatus::Disabled
        );
    }

    #[test]
    fn test_validate() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("base")).unwrap();

        let dependent = PluginManifest::builder("dep")
            .depends_on("base", PluginVersion::new(1, 0, 0))
            .build();
        registry.register(dependent).unwrap();

        let errors = registry.validate();
        assert!(errors.is_empty());
    }

    #[test]
    fn test_version_display() {
        assert_eq!(
            format!("{}", PluginVersion::new(1, 2, 3)),
            "1.2.3"
        );
    }

    #[test]
    fn test_capability_display() {
        assert_eq!(format!("{}", PluginCapability::Distance), "distance");
        assert_eq!(format!("{}", PluginCapability::Index), "index");
    }

    struct HammingDistance;
    impl DistancePluginAdapter for HammingDistance {
        fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter()
                .zip(b.iter())
                .filter(|(&x, &y)| (x - y).abs() > 0.5)
                .count() as f32
        }
        fn name(&self) -> &str {
            "hamming"
        }
    }

    #[test]
    fn test_distance_plugin() {
        let mut registry = PluginRegistry::new();
        let manifest = PluginManifest::builder("hamming")
            .capability(PluginCapability::Distance)
            .build();

        registry
            .register_distance(manifest, Box::new(HammingDistance))
            .unwrap();

        let plugin = registry.get_distance("hamming").unwrap();
        assert_eq!(plugin.name(), "hamming");

        let dist = plugin.compute(&[1.0, 0.0, 1.0], &[0.0, 0.0, 1.0]);
        assert!((dist - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hot_reload() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("reloadable")).unwrap();

        let updated = PluginManifest::builder("reloadable")
            .version(PluginVersion::new(1, 1, 0))
            .build();
        registry.hot_reload(updated).unwrap();

        let entry = registry.get("reloadable").unwrap();
        assert_eq!(entry.manifest.version, PluginVersion::new(1, 1, 0));
    }

    #[test]
    fn test_health_summary() {
        let mut registry = PluginRegistry::new();
        registry.register(test_manifest("a")).unwrap();
        registry.register(test_manifest("b")).unwrap();
        registry.set_status("b", PluginStatus::Disabled).unwrap();

        let health = registry.health();
        assert_eq!(health.total_plugins, 2);
        assert_eq!(health.active, 1);
        assert_eq!(health.disabled, 1);
        assert_eq!(health.dependency_errors, 0);
    }
}
