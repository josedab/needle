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
use std::collections::{HashMap, HashSet};
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
    /// Custom embedding transformer (modifies vectors before/after embedding).
    EmbeddingTransformer,
    /// Custom index type.
    IndexBackend,
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
            Self::EmbeddingTransformer => write!(f, "EmbeddingTransformer"),
            Self::IndexBackend => write!(f, "IndexBackend"),
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
    #[must_use]
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

    /// Return the plugin manifest including dependency information.
    fn manifest(&self) -> PluginManifest {
        PluginManifest::new(
            self.name(),
            self.version(),
            "",
            self.description(),
            self.plugin_type(),
        )
    }
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

/// Trait for plugins that transform embedding vectors.
///
/// Used for custom normalization, dimension reduction, augmentation,
/// or domain-specific vector modifications.
pub trait EmbeddingTransformerPlugin: Plugin {
    /// Transform a vector before it is indexed (called during insert).
    fn transform_for_index(&self, vector: &[f32]) -> HookResult<Vec<f32>>;

    /// Transform a query vector before search.
    fn transform_for_query(&self, vector: &[f32]) -> HookResult<Vec<f32>> {
        // Default: same transformation as indexing
        self.transform_for_index(vector)
    }

    /// Whether this transformer changes the vector dimensions.
    fn changes_dimensions(&self) -> bool {
        false
    }

    /// Output dimensions (if `changes_dimensions` returns true).
    fn output_dimensions(&self, input_dimensions: usize) -> usize {
        input_dimensions
    }
}

// ---------------------------------------------------------------------------
// Plugin directory manager
// ---------------------------------------------------------------------------

/// Manages the local plugin directory (~/.needle/plugins/).
pub struct PluginDirectory {
    /// Root directory for plugins.
    root: std::path::PathBuf,
}

impl PluginDirectory {
    /// Create a plugin directory manager for the default path.
    pub fn default_path() -> Self {
        let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
        Self {
            root: std::path::PathBuf::from(home).join(".needle").join("plugins"),
        }
    }

    /// Create a plugin directory at a custom path.
    pub fn new(root: std::path::PathBuf) -> Self {
        Self { root }
    }

    /// Ensure the plugin directory exists.
    pub fn ensure_exists(&self) -> HookResult<()> {
        std::fs::create_dir_all(&self.root).map_err(|e| {
            PluginError::LifecycleError(format!("Failed to create plugin dir: {e}"))
        })
    }

    /// List installed plugin names (subdirectories).
    pub fn list_installed(&self) -> Vec<String> {
        if !self.root.exists() {
            return Vec::new();
        }
        std::fs::read_dir(&self.root)
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .filter_map(|e| e.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the path for a specific plugin.
    pub fn plugin_path(&self, name: &str) -> std::path::PathBuf {
        self.root.join(name)
    }

    /// Remove a plugin's directory.
    pub fn remove(&self, name: &str) -> HookResult<()> {
        let path = self.plugin_path(name);
        if !path.exists() {
            return Err(PluginError::NotFound(name.to_string()));
        }
        std::fs::remove_dir_all(&path).map_err(|e| {
            PluginError::LifecycleError(format!("Failed to remove plugin '{name}': {e}"))
        })
    }

    /// Get the root directory path.
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }
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

    /// Register multiple plugins respecting dependency order.
    ///
    /// Performs topological sort on plugin dependencies and registers them
    /// in the correct order. Returns an error if there is a circular
    /// dependency or a missing dependency.
    pub fn register_with_dependencies(
        &self,
        mut plugins: Vec<Box<dyn Plugin>>,
    ) -> HookResult<usize> {
        // Build dependency graph
        let mut dep_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut name_map: HashMap<String, usize> = HashMap::new();

        for (i, p) in plugins.iter().enumerate() {
            let name = p.name().to_string();
            let deps = p.manifest().dependencies.clone();
            dep_map.insert(name.clone(), deps);
            name_map.insert(name, i);
        }

        // Topological sort (Kahn's algorithm)
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for (name, deps) in &dep_map {
            in_degree.entry(name.clone()).or_insert(0);
            for dep in deps {
                *in_degree.entry(dep.clone()).or_insert(0) += 0;
                if name_map.contains_key(dep) {
                    *in_degree.entry(name.clone()).or_insert(0) += 1;
                }
            }
        }

        let mut queue: Vec<String> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(name, _)| name.clone())
            .collect();
        queue.sort(); // deterministic order

        let mut order = Vec::new();
        while let Some(name) = queue.pop() {
            order.push(name.clone());
            for (other, deps) in &dep_map {
                if deps.contains(&name) {
                    if let Some(deg) = in_degree.get_mut(other) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push(other.clone());
                        }
                    }
                }
            }
        }

        if order.len() < dep_map.len() {
            return Err(PluginError::MissingDependency(
                "Circular dependency detected".into(),
            ));
        }

        // Check for missing external dependencies
        let known_names: HashSet<&String> = name_map.keys().collect();
        let loaded: std::collections::HashSet<String> =
            self.plugins.read().keys().cloned().collect();

        for (name, deps) in &dep_map {
            for dep in deps {
                if !known_names.contains(dep) && !loaded.contains(dep) {
                    return Err(PluginError::MissingDependency(format!(
                        "Plugin '{}' requires '{}' which is not available",
                        name, dep
                    )));
                }
            }
        }

        // Register in dependency order
        // We need to extract plugins by name from the vec
        let mut plugin_map: HashMap<String, Box<dyn Plugin>> = HashMap::new();
        for p in plugins.drain(..) {
            plugin_map.insert(p.name().to_string(), p);
        }

        let mut registered = 0;
        for name in &order {
            if let Some(plugin) = plugin_map.remove(name) {
                self.register(plugin)?;
                registered += 1;
            }
        }

        Ok(registered)
    }

    /// Hot-reload a plugin: unregister the old version and register the new one.
    ///
    /// If registration of the new plugin fails, the old plugin is NOT restored
    /// (it was already unloaded). Returns the old plugin on success.
    pub fn hot_reload(
        &self,
        new_plugin: Box<dyn Plugin>,
    ) -> HookResult<Box<dyn Plugin>> {
        let name = new_plugin.name().to_string();

        // Unregister old
        let old = self.unregister(&name)?;

        // Register new
        if let Err(e) = self.register(new_plugin) {
            // Try to restore old plugin on failure
            if let Err(restore_err) = self.register(old) {
                tracing::error!("Failed to restore plugin '{}' after hot-reload failure: {}", name, restore_err);
            }
            return Err(PluginError::LifecycleError(format!(
                "Hot-reload failed for '{}': {}",
                name, e
            )));
        }

        Ok(old)
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
// Plugin API Version & Hook Version Negotiation
// ---------------------------------------------------------------------------

/// Plugin API version for compatibility checks.
/// Plugins declare which API version they support; the host can reject
/// incompatible plugins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ApiVersion {
    /// Major version (breaking changes).
    pub major: u32,
    /// Minor version (backward-compatible additions).
    pub minor: u32,
}

impl ApiVersion {
    /// The current plugin API version.
    pub const CURRENT: Self = Self { major: 1, minor: 0 };

    /// Create a new API version.
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Check if a plugin's API version is compatible with the host.
    pub fn is_compatible_with(&self, host: &ApiVersion) -> bool {
        // Same major version, plugin minor <= host minor
        self.major == host.major && self.minor <= host.minor
    }
}

impl fmt::Display for ApiVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}.{}", self.major, self.minor)
    }
}

/// Extended plugin trait with API version support.
pub trait VersionedPlugin: Plugin {
    /// The API version this plugin was built against.
    fn api_version(&self) -> ApiVersion {
        ApiVersion::CURRENT
    }
}

// ---------------------------------------------------------------------------
// Plugin Permissions
// ---------------------------------------------------------------------------

/// Permissions a plugin can request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginPermission {
    /// Read vectors from collections.
    ReadVectors,
    /// Write/modify vectors.
    WriteVectors,
    /// Access metadata.
    ReadMetadata,
    /// Modify metadata.
    WriteMetadata,
    /// Access filesystem.
    FileSystemAccess,
    /// Make network requests.
    NetworkAccess,
    /// Access system metrics.
    MetricsAccess,
}

/// Extended manifest with permissions and API version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtendedManifest {
    /// Base manifest.
    #[serde(flatten)]
    pub base: PluginManifest,
    /// API version.
    pub api_version: ApiVersion,
    /// Required permissions.
    pub permissions: Vec<PluginPermission>,
    /// Homepage URL.
    pub homepage: Option<String>,
    /// License identifier (e.g., "MIT", "Apache-2.0").
    pub license: Option<String>,
    /// Minimum Needle version required.
    pub min_needle_version: Option<String>,
}

impl ExtendedManifest {
    /// Parse an extended manifest from a TOML string.
    pub fn from_toml(toml_str: &str) -> HookResult<Self> {
        // Simple TOML parser for plugin manifests.
        // Parses key = "value" lines within [plugin] section.
        let mut name = String::new();
        let mut version = String::new();
        let mut author = String::new();
        let mut description = String::new();
        let mut plugin_type = PluginType::Custom;
        let mut api_major = 1u32;
        let mut api_minor = 0u32;
        let mut permissions = Vec::new();
        let mut homepage = None;
        let mut license = None;
        let mut min_needle_version = None;
        let dependencies = Vec::new();

        for line in toml_str.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
                continue;
            }
            if let Some((key, val)) = line.split_once('=') {
                let key = key.trim();
                let val = val.trim().trim_matches('"');
                match key {
                    "name" => name = val.to_string(),
                    "version" => version = val.to_string(),
                    "author" => author = val.to_string(),
                    "description" => description = val.to_string(),
                    "type" => {
                        plugin_type = match val {
                            "distance" => PluginType::Distance,
                            "pre_search" => PluginType::PreSearchHook,
                            "post_search" => PluginType::PostSearchHook,
                            "embedding_transformer" => PluginType::EmbeddingTransformer,
                            "index" => PluginType::IndexBackend,
                            "storage" => PluginType::StorageBackend,
                            _ => PluginType::Custom,
                        };
                    }
                    "api_major" => api_major = val.parse().unwrap_or(1),
                    "api_minor" => api_minor = val.parse().unwrap_or(0),
                    "homepage" => homepage = Some(val.to_string()),
                    "license" => license = Some(val.to_string()),
                    "min_needle_version" => min_needle_version = Some(val.to_string()),
                    _ => {}
                }
            }
        }

        if name.is_empty() {
            return Err(PluginError::LifecycleError("Missing 'name' in manifest".into()));
        }

        Ok(Self {
            base: PluginManifest {
                name,
                version,
                author,
                description,
                plugin_type,
                dependencies,
            },
            api_version: ApiVersion::new(api_major, api_minor),
            permissions,
            homepage,
            license,
            min_needle_version,
        })
    }
}

// ---------------------------------------------------------------------------
// WASM Plugin Sandbox (placeholder)
// ---------------------------------------------------------------------------

/// Configuration for the WASM plugin sandbox.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSandboxConfig {
    /// Maximum memory in bytes (default: 64MB).
    pub max_memory_bytes: u64,
    /// Maximum execution time in milliseconds (default: 5000).
    pub max_execution_ms: u64,
    /// Allowed permissions.
    pub allowed_permissions: HashSet<PluginPermission>,
    /// Enable fuel-based CPU limiting.
    pub fuel_limit: Option<u64>,
}

impl Default for WasmSandboxConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64 MB
            max_execution_ms: 5000,
            allowed_permissions: HashSet::new(),
            fuel_limit: Some(1_000_000),
        }
    }
}

/// State of a WASM plugin instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WasmPluginState {
    /// Not yet instantiated.
    Unloaded,
    /// Loaded and ready.
    Ready,
    /// Currently executing.
    Running,
    /// Encountered an error.
    Error,
    /// Terminated (exceeded limits).
    Terminated,
}

/// Placeholder for a WASM plugin sandbox.
/// Full implementation requires the `wasmtime` crate.
#[derive(Debug)]
pub struct WasmSandbox {
    config: WasmSandboxConfig,
    state: WasmPluginState,
    plugin_name: Option<String>,
}

impl WasmSandbox {
    /// Create a new WASM sandbox.
    pub fn new(config: WasmSandboxConfig) -> Self {
        Self {
            config,
            state: WasmPluginState::Unloaded,
            plugin_name: None,
        }
    }

    /// Load a WASM module from bytes.
    /// This is a placeholder — actual implementation requires wasmtime.
    pub fn load(&mut self, name: &str, _wasm_bytes: &[u8]) -> HookResult<()> {
        self.plugin_name = Some(name.to_string());
        self.state = WasmPluginState::Ready;
        Ok(())
    }

    /// Check if the sandbox is ready to execute.
    pub fn is_ready(&self) -> bool {
        self.state == WasmPluginState::Ready
    }

    /// Get the current state.
    pub fn state(&self) -> WasmPluginState {
        self.state
    }

    /// Get the loaded plugin name.
    pub fn plugin_name(&self) -> Option<&str> {
        self.plugin_name.as_deref()
    }

    /// Terminate the sandbox.
    pub fn terminate(&mut self) {
        self.state = WasmPluginState::Terminated;
    }
}

// ---------------------------------------------------------------------------
// Local Plugin Registry (Marketplace)
// ---------------------------------------------------------------------------

/// A local plugin registry that manages installed plugins.
pub struct LocalPluginRegistry {
    /// Plugin directory on disk.
    directory: PluginDirectory,
    /// In-memory index of installed plugins.
    installed: RwLock<HashMap<String, InstalledPlugin>>,
}

/// Metadata about an installed plugin in the local registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledPlugin {
    /// Plugin manifest.
    pub manifest: PluginManifest,
    /// Installation timestamp (Unix epoch).
    pub installed_at: u64,
    /// Whether the plugin is enabled.
    pub enabled: bool,
    /// File size in bytes.
    pub size_bytes: u64,
    /// SHA-256 hash of the WASM binary.
    pub hash: Option<String>,
    /// Sandbox configuration override.
    pub sandbox_config: Option<WasmSandboxConfig>,
}

impl LocalPluginRegistry {
    /// Create a new local registry with the default plugin directory.
    pub fn new() -> Self {
        Self {
            directory: PluginDirectory::default_path(),
            installed: RwLock::new(HashMap::new()),
        }
    }

    /// Create a registry at a custom path.
    pub fn with_directory(dir: PluginDirectory) -> Self {
        Self {
            directory: dir,
            installed: RwLock::new(HashMap::new()),
        }
    }

    /// Install a plugin from manifest data.
    pub fn install(&self, manifest: PluginManifest) -> HookResult<()> {
        self.directory.ensure_exists()?;
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let installed = InstalledPlugin {
            manifest: manifest.clone(),
            installed_at: now,
            enabled: true,
            size_bytes: 0,
            hash: None,
            sandbox_config: None,
        };
        self.installed.write().insert(manifest.name.clone(), installed);
        Ok(())
    }

    /// Remove an installed plugin.
    pub fn remove(&self, name: &str) -> HookResult<()> {
        let removed = self.installed.write().remove(name);
        if removed.is_none() {
            return Err(PluginError::NotFound(name.to_string()));
        }
        Ok(())
    }

    /// List all installed plugins.
    pub fn list(&self) -> Vec<InstalledPlugin> {
        self.installed.read().values().cloned().collect()
    }

    /// Get an installed plugin by name.
    pub fn get(&self, name: &str) -> Option<InstalledPlugin> {
        self.installed.read().get(name).cloned()
    }

    /// Enable a plugin.
    pub fn enable(&self, name: &str) -> HookResult<()> {
        let mut installed = self.installed.write();
        let plugin = installed.get_mut(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        plugin.enabled = true;
        Ok(())
    }

    /// Disable a plugin.
    pub fn disable(&self, name: &str) -> HookResult<()> {
        let mut installed = self.installed.write();
        let plugin = installed.get_mut(name)
            .ok_or_else(|| PluginError::NotFound(name.to_string()))?;
        plugin.enabled = false;
        Ok(())
    }

    /// Get count of installed plugins.
    pub fn count(&self) -> usize {
        self.installed.read().len()
    }

    /// Get count of enabled plugins.
    pub fn enabled_count(&self) -> usize {
        self.installed.read().values().filter(|p| p.enabled).count()
    }
}

impl Default for LocalPluginRegistry {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_dependency_resolution() {
        struct PluginA;
        impl Plugin for PluginA {
            fn name(&self) -> &str { "plugin-a" }
            fn version(&self) -> &str { "1.0.0" }
            fn description(&self) -> &str { "Base plugin" }
            fn plugin_type(&self) -> PluginType { PluginType::Custom }
            fn on_load(&mut self) -> HookResult<()> { Ok(()) }
            fn on_unload(&mut self) -> HookResult<()> { Ok(()) }
        }

        struct PluginB;
        impl Plugin for PluginB {
            fn name(&self) -> &str { "plugin-b" }
            fn version(&self) -> &str { "1.0.0" }
            fn description(&self) -> &str { "Depends on A" }
            fn plugin_type(&self) -> PluginType { PluginType::Custom }
            fn on_load(&mut self) -> HookResult<()> { Ok(()) }
            fn on_unload(&mut self) -> HookResult<()> { Ok(()) }
            fn manifest(&self) -> PluginManifest {
                PluginManifest::new("plugin-b", "1.0.0", "", "Depends on A", PluginType::Custom)
                    .with_dependencies(vec!["plugin-a".into()])
            }
        }

        let manager = PluginManager::new();
        // Register B before A - should resolve via dependency ordering
        let plugins: Vec<Box<dyn Plugin>> = vec![Box::new(PluginB), Box::new(PluginA)];
        let count = manager.register_with_dependencies(plugins).unwrap();
        assert_eq!(count, 2);
        assert_eq!(manager.len(), 2);
    }

    #[test]
    fn test_missing_dependency_error() {
        struct PluginC;
        impl Plugin for PluginC {
            fn name(&self) -> &str { "plugin-c" }
            fn version(&self) -> &str { "1.0.0" }
            fn description(&self) -> &str { "Depends on missing" }
            fn plugin_type(&self) -> PluginType { PluginType::Custom }
            fn on_load(&mut self) -> HookResult<()> { Ok(()) }
            fn on_unload(&mut self) -> HookResult<()> { Ok(()) }
            fn manifest(&self) -> PluginManifest {
                PluginManifest::new("plugin-c", "1.0.0", "", "Missing dep", PluginType::Custom)
                    .with_dependencies(vec!["nonexistent".into()])
            }
        }

        let manager = PluginManager::new();
        let result = manager.register_with_dependencies(vec![Box::new(PluginC)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hot_reload() {
        let manager = PluginManager::new();
        manager.register(Box::new(DummyPlugin::new())).unwrap();

        // Hot-reload with a new dummy instance
        let old = manager.hot_reload(Box::new(DummyPlugin::new())).unwrap();
        assert_eq!(old.name(), "dummy");
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn test_api_version_compatibility() {
        let v1_0 = ApiVersion::new(1, 0);
        let v1_1 = ApiVersion::new(1, 1);
        let v2_0 = ApiVersion::new(2, 0);

        assert!(v1_0.is_compatible_with(&v1_0));
        assert!(v1_0.is_compatible_with(&v1_1)); // plugin 1.0 works with host 1.1
        assert!(!v1_1.is_compatible_with(&v1_0)); // plugin 1.1 doesn't work with host 1.0
        assert!(!v2_0.is_compatible_with(&v1_0)); // different major
    }

    #[test]
    fn test_extended_manifest_from_toml() {
        let toml = r#"
[plugin]
name = "my-distance"
version = "0.1.0"
author = "Test Author"
description = "Custom weighted distance"
type = "distance"
api_major = "1"
api_minor = "0"
license = "MIT"
homepage = "https://example.com"
"#;
        let manifest = ExtendedManifest::from_toml(toml).expect("parse");
        assert_eq!(manifest.base.name, "my-distance");
        assert_eq!(manifest.base.version, "0.1.0");
        assert_eq!(manifest.base.plugin_type, PluginType::Distance);
        assert_eq!(manifest.api_version, ApiVersion::new(1, 0));
        assert_eq!(manifest.license, Some("MIT".into()));
        assert_eq!(manifest.homepage, Some("https://example.com".into()));
    }

    #[test]
    fn test_extended_manifest_missing_name() {
        let toml = r#"
version = "0.1.0"
"#;
        let result = ExtendedManifest::from_toml(toml);
        assert!(result.is_err());
    }

    #[test]
    fn test_wasm_sandbox_lifecycle() {
        let config = WasmSandboxConfig::default();
        let mut sandbox = WasmSandbox::new(config);

        assert_eq!(sandbox.state(), WasmPluginState::Unloaded);
        assert!(!sandbox.is_ready());

        sandbox.load("test-plugin", b"fake-wasm-bytes").expect("load");
        assert_eq!(sandbox.state(), WasmPluginState::Ready);
        assert!(sandbox.is_ready());
        assert_eq!(sandbox.plugin_name(), Some("test-plugin"));

        sandbox.terminate();
        assert_eq!(sandbox.state(), WasmPluginState::Terminated);
        assert!(!sandbox.is_ready());
    }

    #[test]
    fn test_wasm_sandbox_config_defaults() {
        let config = WasmSandboxConfig::default();
        assert_eq!(config.max_memory_bytes, 64 * 1024 * 1024);
        assert_eq!(config.max_execution_ms, 5000);
        assert!(config.fuel_limit.is_some());
    }

    #[test]
    fn test_local_plugin_registry() {
        let dir = tempfile::tempdir().unwrap();
        let registry = LocalPluginRegistry::with_directory(
            PluginDirectory::new(dir.path().to_path_buf()),
        );

        let manifest = PluginManifest::new(
            "test-distance",
            "1.0.0",
            "test-author",
            "A test distance plugin",
            PluginType::Distance,
        );

        registry.install(manifest).unwrap();
        assert_eq!(registry.count(), 1);
        assert_eq!(registry.enabled_count(), 1);

        let plugins = registry.list();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].manifest.name, "test-distance");
        assert!(plugins[0].enabled);
    }

    #[test]
    fn test_registry_enable_disable() {
        let dir = tempfile::tempdir().unwrap();
        let registry = LocalPluginRegistry::with_directory(
            PluginDirectory::new(dir.path().to_path_buf()),
        );

        let manifest = PluginManifest::new("p1", "1.0", "", "test", PluginType::Custom);
        registry.install(manifest).unwrap();

        registry.disable("p1").unwrap();
        assert_eq!(registry.enabled_count(), 0);

        registry.enable("p1").unwrap();
        assert_eq!(registry.enabled_count(), 1);
    }

    #[test]
    fn test_registry_remove() {
        let dir = tempfile::tempdir().unwrap();
        let registry = LocalPluginRegistry::with_directory(
            PluginDirectory::new(dir.path().to_path_buf()),
        );

        let manifest = PluginManifest::new("p1", "1.0", "", "test", PluginType::Custom);
        registry.install(manifest).unwrap();
        assert_eq!(registry.count(), 1);

        registry.remove("p1").unwrap();
        assert_eq!(registry.count(), 0);

        assert!(registry.remove("nonexistent").is_err());
    }
}
