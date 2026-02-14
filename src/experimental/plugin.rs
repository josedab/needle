//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Plugin/Extension System
//!
//! Provides a modular plugin architecture for extending Needle with custom
//! distance functions, search hooks, and other capabilities.
//!
//! # Overview
//!
//! The plugin system is built around a few core abstractions:
//! - [`Plugin`] trait — lifecycle and metadata for all plugins
//! - [`DistancePlugin`] trait — custom distance functions
//! - [`PreSearchHook`] / [`PostSearchHook`] — intercept and modify searches
//! - [`PluginManager`] — thread-safe registry for loading and querying plugins
//!
//! # Example
//!
//! ```rust
//! use needle::plugin::{
//!     Plugin, PluginType, PluginManager, DistancePlugin, PluginError, HookResult,
//! };
//!
//! struct MyDistance;
//!
//! impl Plugin for MyDistance {
//!     fn name(&self) -> &str { "my-distance" }
//!     fn version(&self) -> &str { "0.1.0" }
//!     fn description(&self) -> &str { "Custom weighted distance" }
//!     fn plugin_type(&self) -> PluginType { PluginType::Distance }
//!     fn on_load(&mut self) -> HookResult<()> { Ok(()) }
//!     fn on_unload(&mut self) -> HookResult<()> { Ok(()) }
//! }
//!
//! impl DistancePlugin for MyDistance {
//!     fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
//!         a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2) * 2.0).sum::<f32>().sqrt()
//!     }
//!     fn distance_name(&self) -> &str { "weighted-euclidean" }
//! }
//!
//! let manager = PluginManager::new();
//! manager.register(Box::new(MyDistance)).unwrap();
//! assert_eq!(manager.list_plugins().len(), 1);
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to the plugin subsystem.
#[derive(Error, Debug)]
pub enum PluginError {
    /// A plugin with the given name is already registered.
    #[error("plugin '{0}' is already registered")]
    AlreadyRegistered(String),

    /// No plugin found with the given name.
    #[error("plugin '{0}' not found")]
    NotFound(String),

    /// Plugin failed during its lifecycle hook.
    #[error("plugin lifecycle error: {0}")]
    LifecycleError(String),

    /// A required dependency is missing.
    #[error("missing dependency '{0}'")]
    MissingDependency(String),

    /// Hook execution failed.
    #[error("hook error: {0}")]
    HookError(String),
}

/// Convenience result type for hook and plugin operations.
pub type HookResult<T> = std::result::Result<T, PluginError>;

// ---------------------------------------------------------------------------
// Plugin metadata
// ---------------------------------------------------------------------------

/// Categories of plugins recognised by the system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginType {
    /// Custom distance/similarity function.
    Distance,
    /// Hook executed before a search query.
    PreSearchHook,
    /// Hook executed after search results are computed.
    PostSearchHook,
    /// Alternative storage backend.
    StorageBackend,
    /// Uncategorised / user-defined.
    Custom,
}

impl fmt::Display for PluginType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Distance => write!(f, "Distance"),
            Self::PreSearchHook => write!(f, "PreSearchHook"),
            Self::PostSearchHook => write!(f, "PostSearchHook"),
            Self::StorageBackend => write!(f, "StorageBackend"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Static metadata describing a plugin, useful for cataloguing and
/// dependency resolution without instantiating the plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin name (should match [`Plugin::name`]).
    pub name: String,
    /// SemVer version string.
    pub version: String,
    /// Plugin author.
    pub author: String,
    /// Human-readable description.
    pub description: String,
    /// Category of plugin.
    pub plugin_type: PluginType,
    /// Names of plugins that must be loaded first.
    pub dependencies: Vec<String>,
}

impl PluginManifest {
    /// Create a new manifest with the required fields.
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        author: impl Into<String>,
        description: impl Into<String>,
        plugin_type: PluginType,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            author: author.into(),
            description: description.into(),
            plugin_type,
            dependencies: Vec::new(),
        }
    }

    /// Add dependency names.
    pub fn with_dependencies(mut self, deps: Vec<String>) -> Self {
        self.dependencies = deps;
        self
    }
}

// ---------------------------------------------------------------------------
// Core traits
// ---------------------------------------------------------------------------

/// Core trait implemented by every plugin.
pub trait Plugin: Send + Sync {
    /// Unique name identifying this plugin.
    fn name(&self) -> &str;

    /// SemVer version string.
    fn version(&self) -> &str;

    /// Human-readable description.
    fn description(&self) -> &str;

    /// The category of this plugin.
    fn plugin_type(&self) -> PluginType;

    /// Called when the plugin is loaded into a [`PluginManager`].
    fn on_load(&mut self) -> HookResult<()>;

    /// Called when the plugin is unloaded from a [`PluginManager`].
    fn on_unload(&mut self) -> HookResult<()>;
}

/// A single search result as seen by search hooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHookResult {
    /// Vector identifier.
    pub id: String,
    /// Distance from the query vector.
    pub distance: f32,
    /// Optional JSON metadata attached to the vector.
    pub metadata: Option<serde_json::Value>,
}

/// Plugin that provides a custom distance function.
pub trait DistancePlugin: Plugin {
    /// Compute the distance between two vectors.
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32;

    /// Short name for the distance metric (e.g. `"weighted-euclidean"`).
    fn distance_name(&self) -> &str;
}

/// Hook executed **before** a search is performed.
///
/// Implementations may modify the query vector and/or the number of results
/// requested (`k`).
pub trait PreSearchHook: Plugin {
    /// Receives the original query and `k`, returns (possibly modified) values.
    fn before_search(&self, query: &[f32], k: usize) -> HookResult<(Vec<f32>, usize)>;
}

/// Hook executed **after** search results have been computed.
///
/// Implementations may filter, reorder, or augment results.
pub trait PostSearchHook: Plugin {
    /// Receives search results and returns (possibly modified) results.
    fn after_search(&self, results: Vec<SearchHookResult>) -> HookResult<Vec<SearchHookResult>>;
}

// ---------------------------------------------------------------------------
// Plugin manager
// ---------------------------------------------------------------------------

/// Thread-safe registry that manages the lifecycle of plugins.
///
/// Plugins are keyed by their [`Plugin::name`]. The manager ensures that
/// `on_load` / `on_unload` are called at the appropriate times.
pub struct PluginManager {
    plugins: RwLock<HashMap<String, Box<dyn Plugin>>>,
}

impl fmt::Debug for PluginManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let plugins = self.plugins.read();
        f.debug_struct("PluginManager")
            .field("plugin_count", &plugins.len())
            .field("plugins", &plugins.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create an empty plugin manager.
    pub fn new() -> Self {
        Self {
            plugins: RwLock::new(HashMap::new()),
        }
    }

    /// Register a plugin.
    ///
    /// Calls [`Plugin::on_load`] before inserting. Returns an error if a
    /// plugin with the same name is already registered or if `on_load` fails.
    pub fn register(&self, mut plugin: Box<dyn Plugin>) -> HookResult<()> {
        let name = plugin.name().to_string();

        // Check for duplicates while holding a read lock (fast path).
        {
            let plugins = self.plugins.read();
            if plugins.contains_key(&name) {
                return Err(PluginError::AlreadyRegistered(name));
            }
        }

        plugin.on_load().map_err(|e| {
            PluginError::LifecycleError(format!("on_load failed for '{}': {}", name, e))
        })?;

        let mut plugins = self.plugins.write();
        // Re-check after acquiring write lock to avoid TOCTOU.
        if plugins.contains_key(&name) {
            return Err(PluginError::AlreadyRegistered(name));
        }
        plugins.insert(name, plugin);
        Ok(())
    }

    /// Unregister a plugin by name.
    ///
    /// Calls [`Plugin::on_unload`] before removing. Returns the plugin on
    /// success so the caller may inspect or re-register it.
    pub fn unregister(&self, name: &str) -> HookResult<Box<dyn Plugin>> {
        let mut plugins = self.plugins.write();
        let mut plugin = plugins
            .remove(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;

        if let Err(e) = plugin.on_unload() {
            // Put it back if unload fails so the registry stays consistent.
            let err_msg = format!("on_unload failed for '{}': {}", name, e);
            plugins.insert(name.to_string(), plugin);
            return Err(PluginError::LifecycleError(err_msg));
        }

        Ok(plugin)
    }

    /// Retrieve a reference to a plugin by name.
    ///
    /// The returned guard holds a read lock; prefer short-lived usage.
    pub fn get(&self, name: &str) -> HookResult<PluginRef<'_>> {
        let plugins = self.plugins.read();
        if !plugins.contains_key(name) {
            return Err(PluginError::NotFound(name.to_string()));
        }
        Ok(PluginRef {
            guard: plugins,
            name: name.to_string(),
        })
    }

    /// List metadata for all registered plugins.
    pub fn list_plugins(&self) -> Vec<PluginInfo> {
        let plugins = self.plugins.read();
        plugins
            .values()
            .map(|p| PluginInfo {
                name: p.name().to_string(),
                version: p.version().to_string(),
                description: p.description().to_string(),
                plugin_type: p.plugin_type(),
            })
            .collect()
    }

    /// List plugins filtered by [`PluginType`].
    pub fn list_by_type(&self, plugin_type: PluginType) -> Vec<PluginInfo> {
        let plugins = self.plugins.read();
        plugins
            .values()
            .filter(|p| p.plugin_type() == plugin_type)
            .map(|p| PluginInfo {
                name: p.name().to_string(),
                version: p.version().to_string(),
                description: p.description().to_string(),
                plugin_type: p.plugin_type(),
            })
            .collect()
    }

    /// Returns the number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.read().len()
    }

    /// Returns `true` if no plugins are registered.
    pub fn is_empty(&self) -> bool {
        self.plugins.read().is_empty()
    }
}

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

/// Lightweight snapshot of plugin metadata returned by listing methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginInfo {
    /// Plugin name.
    pub name: String,
    /// Plugin version.
    pub version: String,
    /// Plugin description.
    pub description: String,
    /// Plugin category.
    pub plugin_type: PluginType,
}

/// RAII guard that provides read access to a single plugin inside the manager.
pub struct PluginRef<'a> {
    guard: parking_lot::RwLockReadGuard<'a, HashMap<String, Box<dyn Plugin>>>,
    name: String,
}

impl<'a> fmt::Debug for PluginRef<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PluginRef")
            .field("name", &self.name)
            .finish()
    }
}

impl<'a> PluginRef<'a> {
    /// Access the underlying [`Plugin`] trait object.
    pub fn plugin(&self) -> &dyn Plugin {
        // Safety: we verified the key exists before constructing PluginRef.
        self.guard
            .get(&self.name)
            .expect("plugin was registered")
            .as_ref()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    // -- helpers ----------------------------------------------------------

    /// Minimal plugin used across multiple tests.
    struct DummyPlugin {
        loaded: bool,
    }

    impl DummyPlugin {
        fn new() -> Self {
            Self { loaded: false }
        }
    }

    impl Plugin for DummyPlugin {
        fn name(&self) -> &str {
            "dummy"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "A dummy plugin for testing"
        }
        fn plugin_type(&self) -> PluginType {
            PluginType::Custom
        }
        fn on_load(&mut self) -> HookResult<()> {
            self.loaded = true;
            Ok(())
        }
        fn on_unload(&mut self) -> HookResult<()> {
            self.loaded = false;
            Ok(())
        }
    }

    /// Distance plugin that computes a weighted Euclidean distance.
    struct WeightedEuclidean {
        weight: f32,
    }

    impl Plugin for WeightedEuclidean {
        fn name(&self) -> &str {
            "weighted-euclidean"
        }
        fn version(&self) -> &str {
            "0.1.0"
        }
        fn description(&self) -> &str {
            "Euclidean distance scaled by a constant weight"
        }
        fn plugin_type(&self) -> PluginType {
            PluginType::Distance
        }
        fn on_load(&mut self) -> HookResult<()> {
            Ok(())
        }
        fn on_unload(&mut self) -> HookResult<()> {
            Ok(())
        }
    }

    impl DistancePlugin for WeightedEuclidean {
        fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
            let sum: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
            (sum * self.weight).sqrt()
        }

        fn distance_name(&self) -> &str {
            "weighted-euclidean"
        }
    }

    /// Pre-search hook that normalises the query vector.
    struct NormalizeQueryHook;

    impl Plugin for NormalizeQueryHook {
        fn name(&self) -> &str {
            "normalize-query"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "Normalises the query vector before search"
        }
        fn plugin_type(&self) -> PluginType {
            PluginType::PreSearchHook
        }
        fn on_load(&mut self) -> HookResult<()> {
            Ok(())
        }
        fn on_unload(&mut self) -> HookResult<()> {
            Ok(())
        }
    }

    impl PreSearchHook for NormalizeQueryHook {
        fn before_search(&self, query: &[f32], k: usize) -> HookResult<(Vec<f32>, usize)> {
            let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm == 0.0 {
                return Err(PluginError::HookError("zero-length query vector".into()));
            }
            let normalized: Vec<f32> = query.iter().map(|x| x / norm).collect();
            Ok((normalized, k))
        }
    }

    /// Post-search hook that filters out results beyond a distance threshold.
    struct DistanceThresholdHook {
        max_distance: f32,
    }

    impl Plugin for DistanceThresholdHook {
        fn name(&self) -> &str {
            "distance-threshold"
        }
        fn version(&self) -> &str {
            "1.0.0"
        }
        fn description(&self) -> &str {
            "Filters results exceeding a distance threshold"
        }
        fn plugin_type(&self) -> PluginType {
            PluginType::PostSearchHook
        }
        fn on_load(&mut self) -> HookResult<()> {
            Ok(())
        }
        fn on_unload(&mut self) -> HookResult<()> {
            Ok(())
        }
    }

    impl PostSearchHook for DistanceThresholdHook {
        fn after_search(
            &self,
            results: Vec<SearchHookResult>,
        ) -> HookResult<Vec<SearchHookResult>> {
            Ok(results
                .into_iter()
                .filter(|r| r.distance <= self.max_distance)
                .collect())
        }
    }

    // -- register / unregister -------------------------------------------

    #[test]
    fn test_register_and_list() {
        let manager = PluginManager::new();
        assert!(manager.is_empty());

        manager.register(Box::new(DummyPlugin::new())).unwrap();
        assert_eq!(manager.len(), 1);

        let list = manager.list_plugins();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "dummy");
        assert_eq!(list[0].version, "1.0.0");
        assert_eq!(list[0].plugin_type, PluginType::Custom);
    }

    #[test]
    fn test_register_duplicate_fails() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();

        let err = manager.register(Box::new(DummyPlugin::new())).unwrap_err();
        assert!(matches!(err, PluginError::AlreadyRegistered(_)));
    }

    #[test]
    fn test_unregister() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();
        assert_eq!(manager.len(), 1);

        let plugin = manager.unregister("dummy").unwrap();
        assert_eq!(plugin.name(), "dummy");
        assert!(manager.is_empty());
    }

    #[test]
    fn test_unregister_not_found() {
        let manager = PluginManager::new();
        match manager.unregister("nonexistent") {
            Err(PluginError::NotFound(_)) => {}
            other => panic!("expected NotFound, got {:?}", other.err()),
        }
    }

    #[test]
    fn test_get_plugin() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();

        let plugin_ref = manager.get("dummy").unwrap();
        assert_eq!(plugin_ref.plugin().name(), "dummy");
    }

    #[test]
    fn test_get_not_found() {
        let manager = PluginManager::new();
        let err = manager.get("missing").unwrap_err();
        assert!(matches!(err, PluginError::NotFound(_)));
    }

    // -- list by type ----------------------------------------------------

    #[test]
    fn test_list_by_type() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();
        manager
            .register(Box::new(WeightedEuclidean { weight: 2.0 }))
            .unwrap();

        let distance_plugins = manager.list_by_type(PluginType::Distance);
        assert_eq!(distance_plugins.len(), 1);
        assert_eq!(distance_plugins[0].name, "weighted-euclidean");

        let custom_plugins = manager.list_by_type(PluginType::Custom);
        assert_eq!(custom_plugins.len(), 1);
        assert_eq!(custom_plugins[0].name, "dummy");

        let hook_plugins = manager.list_by_type(PluginType::PreSearchHook);
        assert!(hook_plugins.is_empty());
    }

    // -- distance plugin -------------------------------------------------

    #[test]
    fn test_distance_plugin() {
        let plugin = WeightedEuclidean { weight: 4.0 };

        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 2.0];

        // sqrt((1+4+4) * 4) = sqrt(36) = 6
        let dist = plugin.compute_distance(&a, &b);
        assert!((dist - 6.0).abs() < 1e-6);
        assert_eq!(plugin.distance_name(), "weighted-euclidean");
    }

    // -- pre-search hook -------------------------------------------------

    #[test]
    fn test_pre_search_hook() {
        let hook = NormalizeQueryHook;
        let query = vec![3.0, 4.0];

        let (normalized, k) = hook.before_search(&query, 10).unwrap();
        assert_eq!(k, 10);
        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_pre_search_hook_zero_vector() {
        let hook = NormalizeQueryHook;
        let query = vec![0.0, 0.0];

        let err = hook.before_search(&query, 5).unwrap_err();
        assert!(matches!(err, PluginError::HookError(_)));
    }

    // -- post-search hook ------------------------------------------------

    #[test]
    fn test_post_search_hook() {
        let hook = DistanceThresholdHook { max_distance: 0.5 };

        let results = vec![
            SearchHookResult {
                id: "a".into(),
                distance: 0.1,
                metadata: None,
            },
            SearchHookResult {
                id: "b".into(),
                distance: 0.6,
                metadata: None,
            },
            SearchHookResult {
                id: "c".into(),
                distance: 0.3,
                metadata: None,
            },
        ];

        let filtered = hook.after_search(results).unwrap();
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].id, "a");
        assert_eq!(filtered[1].id, "c");
    }

    // -- lifecycle -------------------------------------------------------

    #[test]
    fn test_lifecycle_on_load_called() {
        let manager = PluginManager::new();
        // DummyPlugin sets `loaded = true` in on_load; we verify indirectly
        // by checking that register succeeds (on_load returns Ok).
        manager.register(Box::new(DummyPlugin::new())).unwrap();
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_lifecycle_on_load_failure() {
        struct FailOnLoad;
        impl Plugin for FailOnLoad {
            fn name(&self) -> &str {
                "fail-load"
            }
            fn version(&self) -> &str {
                "0.0.1"
            }
            fn description(&self) -> &str {
                "Always fails to load"
            }
            fn plugin_type(&self) -> PluginType {
                PluginType::Custom
            }
            fn on_load(&mut self) -> HookResult<()> {
                Err(PluginError::LifecycleError("boom".into()))
            }
            fn on_unload(&mut self) -> HookResult<()> {
                Ok(())
            }
        }

        let manager = PluginManager::new();
        let err = manager.register(Box::new(FailOnLoad)).unwrap_err();
        assert!(matches!(err, PluginError::LifecycleError(_)));
        assert!(manager.is_empty());
    }

    #[test]
    fn test_lifecycle_on_unload_called() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();

        // Unregister — on_unload is called internally.
        let _plugin = manager.unregister("dummy").unwrap();
        assert!(manager.is_empty());
    }

    // -- thread safety ---------------------------------------------------

    #[test]
    fn test_concurrent_access() {
        let manager = Arc::new(PluginManager::new());

        // Register a plugin from the main thread.
        manager.register(Box::new(DummyPlugin::new())).unwrap();

        let done = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::new();

        // Spawn readers.
        for _ in 0..4 {
            let mgr = Arc::clone(&manager);
            let flag = Arc::clone(&done);
            handles.push(std::thread::spawn(move || {
                while !flag.load(Ordering::Relaxed) {
                    let _list = mgr.list_plugins();
                    std::thread::yield_now();
                }
            }));
        }

        // Perform some writes while readers are active.
        for i in 0..10 {
            let name = format!("temp-{}", i);
            struct Temp(String);
            impl Plugin for Temp {
                fn name(&self) -> &str {
                    &self.0
                }
                fn version(&self) -> &str {
                    "0.0.1"
                }
                fn description(&self) -> &str {
                    "temporary"
                }
                fn plugin_type(&self) -> PluginType {
                    PluginType::Custom
                }
                fn on_load(&mut self) -> HookResult<()> {
                    Ok(())
                }
                fn on_unload(&mut self) -> HookResult<()> {
                    Ok(())
                }
            }

            manager.register(Box::new(Temp(name.clone()))).unwrap();
            manager.unregister(&name).unwrap();
        }

        done.store(true, Ordering::Relaxed);
        for h in handles {
            h.join().unwrap();
        }

        // Only the original plugin should remain.
        assert_eq!(manager.len(), 1);
    }

    // -- plugin manifest -------------------------------------------------

    #[test]
    fn test_plugin_manifest() {
        let manifest = PluginManifest::new(
            "my-plugin",
            "1.2.3",
            "Test Author",
            "A test plugin",
            PluginType::Distance,
        )
        .with_dependencies(vec!["dep-a".into(), "dep-b".into()]);

        assert_eq!(manifest.name, "my-plugin");
        assert_eq!(manifest.version, "1.2.3");
        assert_eq!(manifest.author, "Test Author");
        assert_eq!(manifest.plugin_type, PluginType::Distance);
        assert_eq!(manifest.dependencies.len(), 2);
    }

    #[test]
    fn test_plugin_manifest_serialization() {
        let manifest = PluginManifest::new(
            "ser-plugin",
            "0.1.0",
            "Author",
            "Serializable plugin",
            PluginType::PostSearchHook,
        );

        let json = serde_json::to_string(&manifest).unwrap();
        let deserialized: PluginManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "ser-plugin");
        assert_eq!(deserialized.plugin_type, PluginType::PostSearchHook);
    }

    // -- PluginType display & debug --------------------------------------

    #[test]
    fn test_plugin_type_display() {
        assert_eq!(PluginType::Distance.to_string(), "Distance");
        assert_eq!(PluginType::PreSearchHook.to_string(), "PreSearchHook");
        assert_eq!(PluginType::PostSearchHook.to_string(), "PostSearchHook");
        assert_eq!(PluginType::StorageBackend.to_string(), "StorageBackend");
        assert_eq!(PluginType::Custom.to_string(), "Custom");
    }

    // -- PluginManager debug ---------------------------------------------

    #[test]
    fn test_plugin_manager_debug() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();
        let debug_str = format!("{:?}", manager);
        assert!(debug_str.contains("PluginManager"));
        assert!(debug_str.contains("dummy"));
    }

    // -- SearchHookResult ------------------------------------------------

    #[test]
    fn test_search_hook_result_with_metadata() {
        let result = SearchHookResult {
            id: "vec-1".into(),
            distance: 0.42,
            metadata: Some(serde_json::json!({"category": "test"})),
        };
        assert_eq!(result.id, "vec-1");
        assert!((result.distance - 0.42).abs() < 1e-6);
        assert!(result.metadata.is_some());
    }
}
