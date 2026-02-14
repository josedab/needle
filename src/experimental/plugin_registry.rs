//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Plugin Ecosystem & Marketplace
//!
//! WASM-based plugin system for custom distance functions, pre/post-search
//! hooks, data transforms — with a registry for community plugins.
//!
//! # Plugin Types
//!
//! - **Distance plugins**: Custom distance/similarity functions
//! - **Pre-search hooks**: Transform queries before search
//! - **Post-search hooks**: Filter/re-rank results after search
//! - **Transform plugins**: Transform data during ingestion
//! - **Storage plugins**: Custom storage backends
//!
//! # Registry
//!
//! The plugin registry is a simple in-memory store of plugin manifests.
//! A production implementation would back this with HTTP and persistent storage.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::plugin_registry::*;
//!
//! let mut registry = PluginRegistry::new(RegistryConfig::default());
//!
//! // Register a plugin
//! let manifest = PluginManifest {
//!     id: "my-distance".into(),
//!     name: "Custom L3 Distance".into(),
//!     version: "1.0.0".into(),
//!     plugin_type: RegistryPluginType::Distance,
//!     description: "L3 norm distance function".into(),
//!     author: "user".into(),
//!     license: "MIT".into(),
//!     checksum: "abc123".into(),
//!     size_bytes: 1024,
//!     capabilities: vec!["distance".into()],
//!     min_needle_version: "0.1.0".into(),
//!     dependencies: vec![],
//! };
//!
//! registry.publish(manifest).unwrap();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::collection::SearchResult;
use crate::error::{NeedleError, Result};

// ---------------------------------------------------------------------------
// Plugin Types
// ---------------------------------------------------------------------------

/// Plugin type in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegistryPluginType {
    /// Custom distance function.
    Distance,
    /// Pre-search query transformation.
    PreSearchHook,
    /// Post-search result transformation.
    PostSearchHook,
    /// Data ingestion transformation.
    Transform,
    /// Custom storage backend.
    Storage,
    /// Embedding provider.
    EmbeddingProvider,
    /// Index implementation.
    IndexProvider,
    /// General-purpose extension.
    Extension,
}

impl std::fmt::Display for RegistryPluginType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Distance => write!(f, "distance"),
            Self::PreSearchHook => write!(f, "pre-search-hook"),
            Self::PostSearchHook => write!(f, "post-search-hook"),
            Self::Transform => write!(f, "transform"),
            Self::Storage => write!(f, "storage"),
            Self::EmbeddingProvider => write!(f, "embedding-provider"),
            Self::IndexProvider => write!(f, "index-provider"),
            Self::Extension => write!(f, "extension"),
        }
    }
}

// ---------------------------------------------------------------------------
// Plugin Manifest
// ---------------------------------------------------------------------------

/// Metadata about a plugin in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin identifier (e.g. "com.example.my-plugin").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Plugin type.
    pub plugin_type: RegistryPluginType,
    /// Description.
    pub description: String,
    /// Author name or organization.
    pub author: String,
    /// License (SPDX identifier).
    pub license: String,
    /// SHA-256 checksum of the WASM binary.
    pub checksum: String,
    /// Size of the WASM binary in bytes.
    pub size_bytes: u64,
    /// Capabilities this plugin provides.
    pub capabilities: Vec<String>,
    /// Minimum Needle version required.
    pub min_needle_version: String,
    /// Plugin dependencies (by ID).
    pub dependencies: Vec<String>,
}

/// A published plugin with registry metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedPlugin {
    pub manifest: PluginManifest,
    pub published_at: u64,
    pub downloads: u64,
    pub verified: bool,
    pub deprecated: bool,
}

// ---------------------------------------------------------------------------
// Plugin Instance (Runtime)
// ---------------------------------------------------------------------------

/// Status of a loaded plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PluginStatus {
    /// Plugin is loaded and active.
    Active,
    /// Plugin is loaded but disabled.
    Disabled,
    /// Plugin failed to load.
    Failed,
    /// Plugin is being loaded.
    Loading,
}

/// A loaded plugin instance.
pub struct LoadedPlugin {
    pub manifest: PluginManifest,
    pub status: PluginStatus,
    pub loaded_at: Instant,
    pub invocations: u64,
    pub total_time: Duration,
}

impl LoadedPlugin {
    /// Average invocation time.
    pub fn avg_invocation_time(&self) -> Duration {
        if self.invocations > 0 {
            self.total_time / self.invocations as u32
        } else {
            Duration::ZERO
        }
    }

    /// Record an invocation.
    pub fn record_invocation(&mut self, duration: Duration) {
        self.invocations += 1;
        self.total_time += duration;
    }
}

// ---------------------------------------------------------------------------
// Plugin Runtime
// ---------------------------------------------------------------------------

/// Trait for distance function plugins.
pub trait DistanceFn: Send + Sync {
    /// Compute distance between two vectors.
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    /// Name of this distance function.
    fn name(&self) -> &str;
}

/// Trait for pre-search hook plugins.
pub trait PreSearchHookFn: Send + Sync {
    /// Transform a query vector before search.
    fn transform_query(&self, query: &[f32]) -> Vec<f32>;
    /// Name of this hook.
    fn name(&self) -> &str;
}

/// Trait for post-search hook plugins.
pub trait PostSearchHookFn: Send + Sync {
    /// Transform search results after search.
    fn transform_results(&self, results: Vec<SearchResult>) -> Vec<SearchResult>;
    /// Name of this hook.
    fn name(&self) -> &str;
}

/// Trait for data transform plugins.
pub trait TransformFn: Send + Sync {
    /// Transform a vector during ingestion.
    fn transform(&self, vector: &[f32]) -> Vec<f32>;
    /// Name of this transform.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Plugin Manager (Runtime)
// ---------------------------------------------------------------------------

/// Manages loaded plugins at runtime.
pub struct PluginRuntimeManager {
    distance_plugins: RwLock<HashMap<String, Box<dyn DistanceFn>>>,
    pre_hooks: RwLock<Vec<Box<dyn PreSearchHookFn>>>,
    post_hooks: RwLock<Vec<Box<dyn PostSearchHookFn>>>,
    transforms: RwLock<Vec<Box<dyn TransformFn>>>,
    loaded: RwLock<HashMap<String, LoadedPlugin>>,
}

impl PluginRuntimeManager {
    pub fn new() -> Self {
        Self {
            distance_plugins: RwLock::new(HashMap::new()),
            pre_hooks: RwLock::new(Vec::new()),
            post_hooks: RwLock::new(Vec::new()),
            transforms: RwLock::new(Vec::new()),
            loaded: RwLock::new(HashMap::new()),
        }
    }

    /// Register a custom distance function.
    pub fn register_distance(&self, id: &str, plugin: Box<dyn DistanceFn>) {
        self.distance_plugins.write().insert(id.to_string(), plugin);
    }

    /// Register a pre-search hook.
    pub fn register_pre_hook(&self, hook: Box<dyn PreSearchHookFn>) {
        self.pre_hooks.write().push(hook);
    }

    /// Register a post-search hook.
    pub fn register_post_hook(&self, hook: Box<dyn PostSearchHookFn>) {
        self.post_hooks.write().push(hook);
    }

    /// Register a data transform.
    pub fn register_transform(&self, transform: Box<dyn TransformFn>) {
        self.transforms.write().push(transform);
    }

    /// Compute distance using a named plugin.
    pub fn compute_distance(&self, plugin_id: &str, a: &[f32], b: &[f32]) -> Result<f32> {
        let plugins = self.distance_plugins.read();
        let plugin = plugins
            .get(plugin_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Distance plugin: {}", plugin_id)))?;
        Ok(plugin.compute(a, b))
    }

    /// Apply all pre-search hooks to a query.
    pub fn apply_pre_hooks(&self, query: &[f32]) -> Vec<f32> {
        let hooks = self.pre_hooks.read();
        let mut result = query.to_vec();
        for hook in hooks.iter() {
            result = hook.transform_query(&result);
        }
        result
    }

    /// Apply all post-search hooks to results.
    pub fn apply_post_hooks(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let hooks = self.post_hooks.read();
        let mut result = results;
        for hook in hooks.iter() {
            result = hook.transform_results(result);
        }
        result
    }

    /// Apply all transforms to a vector.
    pub fn apply_transforms(&self, vector: &[f32]) -> Vec<f32> {
        let transforms = self.transforms.read();
        let mut result = vector.to_vec();
        for t in transforms.iter() {
            result = t.transform(&result);
        }
        result
    }

    /// Number of loaded distance plugins.
    pub fn distance_plugin_count(&self) -> usize {
        self.distance_plugins.read().len()
    }

    /// Number of loaded pre-search hooks.
    pub fn pre_hook_count(&self) -> usize {
        self.pre_hooks.read().len()
    }

    /// Number of loaded post-search hooks.
    pub fn post_hook_count(&self) -> usize {
        self.post_hooks.read().len()
    }

    /// Number of loaded transforms.
    pub fn transform_count(&self) -> usize {
        self.transforms.read().len()
    }

    /// List names of registered distance plugins.
    pub fn list_distance_plugins(&self) -> Vec<String> {
        self.distance_plugins.read().keys().cloned().collect()
    }
}

impl Default for PluginRuntimeManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Plugin Registry (Marketplace)
// ---------------------------------------------------------------------------

/// Configuration for the plugin registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of plugins.
    pub max_plugins: usize,
    /// Whether to allow unverified plugins.
    pub allow_unverified: bool,
    /// Maximum plugin WASM binary size.
    pub max_plugin_size: u64,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_plugins: 1000,
            allow_unverified: true,
            max_plugin_size: 10 * 1024 * 1024, // 10MB
        }
    }
}

/// In-memory plugin registry (marketplace).
pub struct PluginRegistry {
    config: RegistryConfig,
    plugins: RwLock<HashMap<String, PublishedPlugin>>,
}

impl PluginRegistry {
    /// Create a new registry.
    pub fn new(config: RegistryConfig) -> Self {
        Self {
            config,
            plugins: RwLock::new(HashMap::new()),
        }
    }

    /// Publish a plugin to the registry.
    pub fn publish(&self, manifest: PluginManifest) -> Result<()> {
        let plugins = self.plugins.read();
        if plugins.len() >= self.config.max_plugins {
            return Err(NeedleError::CapacityExceeded("Registry at capacity".into()));
        }
        if manifest.size_bytes > self.config.max_plugin_size {
            return Err(NeedleError::InvalidInput(format!(
                "Plugin too large: {} bytes (max: {})",
                manifest.size_bytes, self.config.max_plugin_size
            )));
        }
        drop(plugins);

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        self.plugins.write().insert(
            manifest.id.clone(),
            PublishedPlugin {
                manifest,
                published_at: now,
                downloads: 0,
                verified: false,
                deprecated: false,
            },
        );

        Ok(())
    }

    /// Get a plugin by ID.
    pub fn get(&self, id: &str) -> Option<PublishedPlugin> {
        self.plugins.read().get(id).cloned()
    }

    /// Search plugins by type.
    pub fn search_by_type(&self, plugin_type: RegistryPluginType) -> Vec<PublishedPlugin> {
        self.plugins
            .read()
            .values()
            .filter(|p| p.manifest.plugin_type == plugin_type)
            .cloned()
            .collect()
    }

    /// Search plugins by keyword.
    pub fn search(&self, query: &str) -> Vec<PublishedPlugin> {
        let lower = query.to_lowercase();
        self.plugins
            .read()
            .values()
            .filter(|p| {
                p.manifest.name.to_lowercase().contains(&lower)
                    || p.manifest.description.to_lowercase().contains(&lower)
                    || p.manifest
                        .capabilities
                        .iter()
                        .any(|c| c.to_lowercase().contains(&lower))
            })
            .cloned()
            .collect()
    }

    /// List all plugins.
    pub fn list_all(&self) -> Vec<PublishedPlugin> {
        self.plugins.read().values().cloned().collect()
    }

    /// Mark a plugin as verified.
    pub fn verify(&self, id: &str) -> Result<()> {
        let mut plugins = self.plugins.write();
        let plugin = plugins
            .get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin: {}", id)))?;
        plugin.verified = true;
        Ok(())
    }

    /// Deprecate a plugin.
    pub fn deprecate(&self, id: &str) -> Result<()> {
        let mut plugins = self.plugins.write();
        let plugin = plugins
            .get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin: {}", id)))?;
        plugin.deprecated = true;
        Ok(())
    }

    /// Record a download.
    pub fn record_download(&self, id: &str) {
        if let Some(plugin) = self.plugins.write().get_mut(id) {
            plugin.downloads += 1;
        }
    }

    /// Remove a plugin from the registry.
    pub fn remove(&self, id: &str) -> Result<()> {
        self.plugins
            .write()
            .remove(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin: {}", id)))?;
        Ok(())
    }

    /// Number of plugins in the registry.
    pub fn count(&self) -> usize {
        self.plugins.read().len()
    }

    /// Get most downloaded plugins.
    pub fn popular(&self, limit: usize) -> Vec<PublishedPlugin> {
        let mut all = self.list_all();
        all.sort_by(|a, b| b.downloads.cmp(&a.downloads));
        all.truncate(limit);
        all
    }
}

// ---------------------------------------------------------------------------
// Built-in Plugins
// ---------------------------------------------------------------------------

/// L3 norm distance function.
pub struct L3Distance;

impl DistanceFn for L3Distance {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs().powi(3))
            .sum::<f32>()
            .cbrt()
    }

    fn name(&self) -> &str {
        "L3 Distance"
    }
}

/// Normalize query vectors before search.
pub struct NormalizeHook;

impl PreSearchHookFn for NormalizeHook {
    fn transform_query(&self, query: &[f32]) -> Vec<f32> {
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            query.iter().map(|x| x / norm).collect()
        } else {
            query.to_vec()
        }
    }

    fn name(&self) -> &str {
        "Normalize"
    }
}

/// Truncate results to top-k based on a threshold.
pub struct ThresholdFilter {
    pub max_distance: f32,
}

impl PostSearchHookFn for ThresholdFilter {
    fn transform_results(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        results
            .into_iter()
            .filter(|r| r.distance <= self.max_distance)
            .collect()
    }

    fn name(&self) -> &str {
        "Threshold Filter"
    }
}

/// Dimensionality reduction via truncation.
pub struct TruncateTransform {
    pub target_dim: usize,
}

impl TransformFn for TruncateTransform {
    fn transform(&self, vector: &[f32]) -> Vec<f32> {
        vector.iter().take(self.target_dim).copied().collect()
    }

    fn name(&self) -> &str {
        "Truncate"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest(id: &str) -> PluginManifest {
        PluginManifest {
            id: id.into(),
            name: format!("Test Plugin {}", id),
            version: "1.0.0".into(),
            plugin_type: RegistryPluginType::Distance,
            description: "A test plugin".into(),
            author: "test".into(),
            license: "MIT".into(),
            checksum: "abc123".into(),
            size_bytes: 1024,
            capabilities: vec!["distance".into()],
            min_needle_version: "0.1.0".into(),
            dependencies: vec![],
        }
    }

    #[test]
    fn test_registry_publish_and_get() {
        let registry = PluginRegistry::new(RegistryConfig::default());
        registry.publish(sample_manifest("test-1")).unwrap();

        let plugin = registry.get("test-1").unwrap();
        assert_eq!(plugin.manifest.id, "test-1");
        assert_eq!(plugin.downloads, 0);
        assert!(!plugin.verified);
    }

    #[test]
    fn test_registry_search_by_type() {
        let registry = PluginRegistry::new(RegistryConfig::default());

        let mut m1 = sample_manifest("dist-1");
        m1.plugin_type = RegistryPluginType::Distance;
        registry.publish(m1).unwrap();

        let mut m2 = sample_manifest("hook-1");
        m2.plugin_type = RegistryPluginType::PreSearchHook;
        registry.publish(m2).unwrap();

        let distances = registry.search_by_type(RegistryPluginType::Distance);
        assert_eq!(distances.len(), 1);
    }

    #[test]
    fn test_registry_keyword_search() {
        let registry = PluginRegistry::new(RegistryConfig::default());
        let mut m = sample_manifest("custom-dist");
        m.description = "Custom L3 norm distance function".into();
        registry.publish(m).unwrap();

        let results = registry.search("L3 norm");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_registry_verify_and_deprecate() {
        let registry = PluginRegistry::new(RegistryConfig::default());
        registry.publish(sample_manifest("p1")).unwrap();

        registry.verify("p1").unwrap();
        assert!(registry.get("p1").unwrap().verified);

        registry.deprecate("p1").unwrap();
        assert!(registry.get("p1").unwrap().deprecated);
    }

    #[test]
    fn test_registry_download_count() {
        let registry = PluginRegistry::new(RegistryConfig::default());
        registry.publish(sample_manifest("p1")).unwrap();

        registry.record_download("p1");
        registry.record_download("p1");

        assert_eq!(registry.get("p1").unwrap().downloads, 2);
    }

    #[test]
    fn test_registry_remove() {
        let registry = PluginRegistry::new(RegistryConfig::default());
        registry.publish(sample_manifest("p1")).unwrap();
        assert_eq!(registry.count(), 1);

        registry.remove("p1").unwrap();
        assert_eq!(registry.count(), 0);
        assert!(registry.get("p1").is_none());
    }

    #[test]
    fn test_registry_capacity() {
        let config = RegistryConfig {
            max_plugins: 2,
            ..Default::default()
        };
        let registry = PluginRegistry::new(config);

        registry.publish(sample_manifest("p1")).unwrap();
        registry.publish(sample_manifest("p2")).unwrap();
        assert!(registry.publish(sample_manifest("p3")).is_err());
    }

    #[test]
    fn test_registry_size_limit() {
        let config = RegistryConfig {
            max_plugin_size: 500,
            ..Default::default()
        };
        let registry = PluginRegistry::new(config);

        let mut m = sample_manifest("big");
        m.size_bytes = 1000;
        assert!(registry.publish(m).is_err());
    }

    #[test]
    fn test_registry_popular() {
        let registry = PluginRegistry::new(RegistryConfig::default());
        registry.publish(sample_manifest("p1")).unwrap();
        registry.publish(sample_manifest("p2")).unwrap();

        registry.record_download("p1");
        registry.record_download("p2");
        registry.record_download("p2");

        let popular = registry.popular(10);
        assert_eq!(popular[0].manifest.id, "p2"); // Most downloads
    }

    #[test]
    fn test_l3_distance() {
        let d = L3Distance;
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let dist = d.compute(&a, &b);
        assert!(dist > 0.0);
        assert_eq!(d.name(), "L3 Distance");
    }

    #[test]
    fn test_normalize_hook() {
        let hook = NormalizeHook;
        let result = hook.transform_query(&[3.0, 4.0]);
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_threshold_filter() {
        let filter = ThresholdFilter { max_distance: 0.5 };
        let results = vec![
            SearchResult::new("a", 0.3, None),
            SearchResult::new("b", 0.7, None),
            SearchResult::new("c", 0.4, None),
        ];
        let filtered = filter.transform_results(results);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_truncate_transform() {
        let t = TruncateTransform { target_dim: 3 };
        let result = t.transform(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(result.len(), 3);
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_runtime_manager() {
        let manager = PluginRuntimeManager::new();

        manager.register_distance("l3", Box::new(L3Distance));
        assert_eq!(manager.distance_plugin_count(), 1);

        let dist = manager
            .compute_distance("l3", &[1.0, 0.0], &[0.0, 1.0])
            .unwrap();
        assert!(dist > 0.0);
    }

    #[test]
    fn test_runtime_hooks() {
        let manager = PluginRuntimeManager::new();

        manager.register_pre_hook(Box::new(NormalizeHook));
        assert_eq!(manager.pre_hook_count(), 1);

        let query = manager.apply_pre_hooks(&[3.0, 4.0]);
        let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_runtime_post_hooks() {
        let manager = PluginRuntimeManager::new();
        manager.register_post_hook(Box::new(ThresholdFilter { max_distance: 0.5 }));

        let results = vec![
            SearchResult::new("a", 0.3, None),
            SearchResult::new("b", 0.8, None),
        ];
        let filtered = manager.apply_post_hooks(results);
        assert_eq!(filtered.len(), 1);
    }

    #[test]
    fn test_runtime_transforms() {
        let manager = PluginRuntimeManager::new();
        manager.register_transform(Box::new(TruncateTransform { target_dim: 2 }));

        let result = manager.apply_transforms(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_runtime_missing_distance() {
        let manager = PluginRuntimeManager::new();
        assert!(manager.compute_distance("nope", &[1.0], &[2.0]).is_err());
    }

    #[test]
    fn test_plugin_type_display() {
        assert_eq!(format!("{}", RegistryPluginType::Distance), "distance");
        assert_eq!(format!("{}", RegistryPluginType::Transform), "transform");
    }

    #[test]
    fn test_manifest_serialization() {
        let m = sample_manifest("test");
        let json = serde_json::to_string(&m).unwrap();
        let decoded: PluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.id, "test");
    }

    #[test]
    fn test_loaded_plugin_stats() {
        let mut loaded = LoadedPlugin {
            manifest: sample_manifest("p1"),
            status: PluginStatus::Active,
            loaded_at: Instant::now(),
            invocations: 0,
            total_time: Duration::ZERO,
        };

        assert_eq!(loaded.avg_invocation_time(), Duration::ZERO);
        loaded.record_invocation(Duration::from_millis(10));
        loaded.record_invocation(Duration::from_millis(20));
        assert_eq!(loaded.invocations, 2);
        assert_eq!(loaded.avg_invocation_time(), Duration::from_millis(15));
    }

    #[test]
    fn test_list_distance_plugins() {
        let manager = PluginRuntimeManager::new();
        manager.register_distance("l3", Box::new(L3Distance));
        let list = manager.list_distance_plugins();
        assert!(list.contains(&"l3".to_string()));
    }
}
