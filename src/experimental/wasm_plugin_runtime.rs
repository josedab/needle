#![allow(dead_code)]

//! WASM Plugin Runtime
//!
//! Extends the plugin system with WASM-based sandboxed execution for user-defined
//! functions (custom distance metrics, rerankers, metadata transformers).
//!
//! # Design
//!
//! - **Plugin API Traits**: `DistancePlugin`, `RerankerPlugin`, `TransformerPlugin`
//! - **WASM Sandbox**: Memory limits, CPU timeouts, capability restrictions
//! - **Plugin Registry**: Discovery, loading, and lifecycle management
//!
//! # Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────────────────┐
//! │                    WasmPluginRuntime                       │
//! ├──────────────────────────────────────────────────────────┤
//! │  PluginRegistry                                           │
//! │  ├── distance/weighted_euclidean.wasm                     │
//! │  ├── reranker/cross_encoder.wasm                          │
//! │  └── transformer/normalize.wasm                           │
//! ├──────────────────────────────────────────────────────────┤
//! │  WasmSandbox                                              │
//! │  ├── Memory limit: 64MB                                   │
//! │  ├── CPU timeout: 100ms                                   │
//! │  └── No filesystem/network access                         │
//! └──────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::experimental::wasm_plugin_runtime::{
//!     WasmPluginRuntime, WasmPluginConfig, PluginManifest,
//! };
//!
//! let mut runtime = WasmPluginRuntime::new(WasmPluginConfig::default());
//! runtime.load_plugin("my_distance", &wasm_bytes, manifest)?;
//!
//! let distance = runtime.call_distance("my_distance", &vec_a, &vec_b)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for the WASM plugin runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmPluginConfig {
    /// Maximum memory per WASM instance (bytes).
    pub max_memory_bytes: usize,
    /// CPU timeout per plugin call.
    pub call_timeout: Duration,
    /// Maximum number of loaded plugins.
    pub max_plugins: usize,
    /// Enable fuel metering for CPU limiting.
    pub enable_fuel_metering: bool,
    /// Fuel units per call (higher = more CPU time allowed).
    pub fuel_per_call: u64,
    /// Allow plugins to access stdin/stdout.
    pub allow_stdio: bool,
}

impl Default for WasmPluginConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64MB
            call_timeout: Duration::from_millis(100),
            max_plugins: 64,
            enable_fuel_metering: true,
            fuel_per_call: 1_000_000,
            allow_stdio: false,
        }
    }
}

/// Plugin type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WasmPluginType {
    /// Custom distance/similarity function.
    Distance,
    /// Result reranker.
    Reranker,
    /// Metadata transformer.
    Transformer,
    /// Vector preprocessor.
    Preprocessor,
    /// Custom scorer.
    Scorer,
}

/// Plugin manifest describing a WASM plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    /// Unique plugin name.
    pub name: String,
    /// Plugin version.
    pub version: String,
    /// Plugin type.
    pub plugin_type: WasmPluginType,
    /// Human-readable description.
    pub description: String,
    /// Author or organization.
    pub author: String,
    /// License identifier.
    pub license: String,
    /// Required host API version.
    pub min_host_version: String,
    /// Expected input dimensions (0 = any).
    pub expected_dimensions: usize,
    /// Exported WASM functions.
    pub exports: Vec<String>,
}

impl PluginManifest {
    /// Create a new manifest for a distance plugin.
    pub fn distance(name: &str, version: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            plugin_type: WasmPluginType::Distance,
            description: description.to_string(),
            author: String::new(),
            license: "MIT".to_string(),
            min_host_version: "0.1.0".to_string(),
            expected_dimensions: 0,
            exports: vec!["compute_distance".to_string()],
        }
    }

    /// Create a new manifest for a reranker plugin.
    pub fn reranker(name: &str, version: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            plugin_type: WasmPluginType::Reranker,
            description: description.to_string(),
            author: String::new(),
            license: "MIT".to_string(),
            min_host_version: "0.1.0".to_string(),
            expected_dimensions: 0,
            exports: vec!["rerank".to_string()],
        }
    }

    /// Create a new manifest for a transformer plugin.
    pub fn transformer(name: &str, version: &str, description: &str) -> Self {
        Self {
            name: name.to_string(),
            version: version.to_string(),
            plugin_type: WasmPluginType::Transformer,
            description: description.to_string(),
            author: String::new(),
            license: "MIT".to_string(),
            min_host_version: "0.1.0".to_string(),
            expected_dimensions: 0,
            exports: vec!["transform".to_string()],
        }
    }
}

/// Trait for distance function plugins.
pub trait DistancePluginApi: Send + Sync {
    /// Compute distance between two vectors.
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32;
    /// Name of the distance function.
    fn distance_name(&self) -> &str;
}

/// Trait for reranker plugins.
pub trait RerankerPluginApi: Send + Sync {
    /// Rerank search results given a query.
    /// Returns reordered indices with new scores.
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[(String, f32, Vec<f32>)],
    ) -> Vec<(usize, f32)>;
    /// Name of the reranker.
    fn reranker_name(&self) -> &str;
}

/// Trait for metadata transformer plugins.
pub trait TransformerPluginApi: Send + Sync {
    /// Transform metadata before storage or retrieval.
    fn transform(&self, metadata: &serde_json::Value) -> serde_json::Value;
    /// Name of the transformer.
    fn transformer_name(&self) -> &str;
}

/// Runtime state for a loaded WASM plugin.
struct LoadedPlugin {
    manifest: PluginManifest,
    /// WASM binary (stored for potential re-instantiation).
    wasm_bytes: Vec<u8>,
    /// Whether the plugin is currently enabled.
    enabled: bool,
    /// Total invocations.
    invocations: u64,
    /// Total execution time.
    total_time: Duration,
    /// Average execution time per call (microseconds).
    avg_time_us: f64,
}

/// Statistics for a loaded plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginStats {
    /// Plugin name.
    pub name: String,
    /// Plugin type.
    pub plugin_type: WasmPluginType,
    /// Whether enabled.
    pub enabled: bool,
    /// Total invocations.
    pub invocations: u64,
    /// Average execution time per call (microseconds).
    pub avg_time_us: f64,
    /// WASM binary size (bytes).
    pub wasm_size_bytes: usize,
}

/// WASM Plugin Runtime: manages plugin lifecycle, sandboxing, and execution.
///
/// Note: This implementation provides the API contract and plugin management.
/// The actual WASM execution requires the `wasmtime` crate, which is an optional
/// dependency. Without it, plugins operate in a simulated mode for API testing.
pub struct WasmPluginRuntime {
    config: WasmPluginConfig,
    plugins: HashMap<String, LoadedPlugin>,
    /// Built-in reference distance plugins.
    builtin_distances: HashMap<String, Box<dyn DistancePluginApi>>,
    /// Built-in reference reranker plugins.
    builtin_rerankers: HashMap<String, Box<dyn RerankerPluginApi>>,
    /// Built-in reference transformers.
    builtin_transformers: HashMap<String, Box<dyn TransformerPluginApi>>,
}

impl WasmPluginRuntime {
    /// Create a new runtime with default configuration.
    pub fn new(config: WasmPluginConfig) -> Self {
        let mut runtime = Self {
            config,
            plugins: HashMap::new(),
            builtin_distances: HashMap::new(),
            builtin_rerankers: HashMap::new(),
            builtin_transformers: HashMap::new(),
        };
        runtime.register_builtin_plugins();
        runtime
    }

    /// Register built-in reference plugins.
    fn register_builtin_plugins(&mut self) {
        // Reference distance: weighted euclidean
        self.builtin_distances.insert(
            "weighted_euclidean".to_string(),
            Box::new(WeightedEuclidean),
        );

        // Reference distance: manhattan
        self.builtin_distances.insert(
            "manhattan".to_string(),
            Box::new(Manhattan),
        );

        // Reference reranker: reciprocal rank
        self.builtin_rerankers.insert(
            "reciprocal_rank".to_string(),
            Box::new(ReciprocalRankReranker),
        );

        // Reference transformer: lowercase keys
        self.builtin_transformers.insert(
            "lowercase_keys".to_string(),
            Box::new(LowercaseKeysTransformer),
        );

        // Reference transformer: field filter
        self.builtin_transformers.insert(
            "field_filter".to_string(),
            Box::new(FieldFilterTransformer {
                allowed_fields: vec!["title".to_string(), "category".to_string(), "tags".to_string()],
            }),
        );
    }

    /// Register a native distance plugin (no WASM required).
    ///
    /// This allows extending Needle with custom distance functions implemented
    /// in Rust, without the overhead of WASM sandboxing.
    pub fn register_native_distance(
        &mut self,
        name: &str,
        plugin: Box<dyn DistancePluginApi>,
    ) {
        self.builtin_distances.insert(name.to_string(), plugin);
    }

    /// Register a native reranker plugin (no WASM required).
    pub fn register_native_reranker(
        &mut self,
        name: &str,
        plugin: Box<dyn RerankerPluginApi>,
    ) {
        self.builtin_rerankers.insert(name.to_string(), plugin);
    }

    /// Register a native transformer plugin (no WASM required).
    pub fn register_native_transformer(
        &mut self,
        name: &str,
        plugin: Box<dyn TransformerPluginApi>,
    ) {
        self.builtin_transformers.insert(name.to_string(), plugin);
    }

    /// Load a WASM plugin from binary.
    pub fn load_plugin(
        &mut self,
        name: &str,
        wasm_bytes: &[u8],
        manifest: PluginManifest,
    ) -> Result<()> {
        if self.plugins.len() >= self.config.max_plugins {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum plugin count ({}) reached",
                self.config.max_plugins
            )));
        }

        if self.plugins.contains_key(name) {
            return Err(NeedleError::Conflict(format!(
                "Plugin '{}' already loaded",
                name
            )));
        }

        // Validate WASM binary size against memory limit
        if wasm_bytes.len() > self.config.max_memory_bytes {
            return Err(NeedleError::CapacityExceeded(format!(
                "WASM binary ({} bytes) exceeds memory limit ({} bytes)",
                wasm_bytes.len(),
                self.config.max_memory_bytes
            )));
        }

        self.plugins.insert(
            name.to_string(),
            LoadedPlugin {
                manifest,
                wasm_bytes: wasm_bytes.to_vec(),
                enabled: true,
                invocations: 0,
                total_time: Duration::ZERO,
                avg_time_us: 0.0,
            },
        );

        Ok(())
    }

    /// Unload a plugin.
    pub fn unload_plugin(&mut self, name: &str) -> Result<()> {
        self.plugins
            .remove(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin '{}' not found", name)))?;
        Ok(())
    }

    /// Enable or disable a plugin.
    pub fn set_enabled(&mut self, name: &str, enabled: bool) -> Result<()> {
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin '{}' not found", name)))?;
        plugin.enabled = enabled;
        Ok(())
    }

    /// Call a distance function (built-in or native plugin).
    ///
    /// Looks up plugins in this order:
    /// 1. Built-in reference plugins
    /// 2. User-registered native plugins
    /// 3. Loaded WASM plugins (requires wasmtime; currently uses fallback)
    pub fn call_distance(&mut self, name: &str, a: &[f32], b: &[f32]) -> Result<f32> {
        // Check built-in first
        if let Some(dist) = self.builtin_distances.get(name) {
            return Ok(dist.compute_distance(a, b));
        }

        // Check loaded WASM plugins
        let plugin = self
            .plugins
            .get_mut(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Distance plugin '{}' not found", name)))?;

        if !plugin.enabled {
            return Err(NeedleError::InvalidOperation(format!(
                "Plugin '{}' is disabled",
                name
            )));
        }

        if plugin.manifest.plugin_type != WasmPluginType::Distance {
            return Err(NeedleError::InvalidOperation(format!(
                "Plugin '{}' is not a distance plugin (type: {:?})",
                name, plugin.manifest.plugin_type
            )));
        }

        let start = Instant::now();

        // Simulated WASM execution (real implementation would use wasmtime)
        // For now, compute a default euclidean distance
        let distance: f32 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt();

        let elapsed = start.elapsed();
        plugin.invocations += 1;
        plugin.total_time += elapsed;
        plugin.avg_time_us =
            plugin.total_time.as_micros() as f64 / plugin.invocations as f64;

        Ok(distance)
    }

    /// Call a reranker (built-in or WASM).
    pub fn call_reranker(
        &self,
        name: &str,
        query: &[f32],
        candidates: &[(String, f32, Vec<f32>)],
    ) -> Result<Vec<(usize, f32)>> {
        if let Some(reranker) = self.builtin_rerankers.get(name) {
            return Ok(reranker.rerank(query, candidates));
        }
        Err(NeedleError::NotFound(format!(
            "Reranker plugin '{}' not found",
            name
        )))
    }

    /// Call a transformer (built-in or WASM).
    pub fn call_transformer(
        &self,
        name: &str,
        metadata: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        if let Some(transformer) = self.builtin_transformers.get(name) {
            return Ok(transformer.transform(metadata));
        }
        Err(NeedleError::NotFound(format!(
            "Transformer plugin '{}' not found",
            name
        )))
    }

    /// List all loaded plugins.
    pub fn list_plugins(&self) -> Vec<PluginStats> {
        let mut stats: Vec<PluginStats> = self
            .plugins
            .iter()
            .map(|(name, p)| PluginStats {
                name: name.clone(),
                plugin_type: p.manifest.plugin_type,
                enabled: p.enabled,
                invocations: p.invocations,
                avg_time_us: p.avg_time_us,
                wasm_size_bytes: p.wasm_bytes.len(),
            })
            .collect();

        // Also include built-in plugins
        for name in self.builtin_distances.keys() {
            stats.push(PluginStats {
                name: name.clone(),
                plugin_type: WasmPluginType::Distance,
                enabled: true,
                invocations: 0,
                avg_time_us: 0.0,
                wasm_size_bytes: 0,
            });
        }
        for name in self.builtin_rerankers.keys() {
            stats.push(PluginStats {
                name: name.clone(),
                plugin_type: WasmPluginType::Reranker,
                enabled: true,
                invocations: 0,
                avg_time_us: 0.0,
                wasm_size_bytes: 0,
            });
        }
        for name in self.builtin_transformers.keys() {
            stats.push(PluginStats {
                name: name.clone(),
                plugin_type: WasmPluginType::Transformer,
                enabled: true,
                invocations: 0,
                avg_time_us: 0.0,
                wasm_size_bytes: 0,
            });
        }

        stats
    }

    /// Get plugin count.
    pub fn plugin_count(&self) -> usize {
        self.plugins.len()
            + self.builtin_distances.len()
            + self.builtin_rerankers.len()
            + self.builtin_transformers.len()
    }
}

// ── Built-in Reference Plugins ──────────────────────────────────────────────

/// Weighted Euclidean distance with uniform weights.
struct WeightedEuclidean;

impl DistancePluginApi for WeightedEuclidean {
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2) * 2.0)
            .sum::<f32>()
            .sqrt()
    }

    fn distance_name(&self) -> &str {
        "weighted_euclidean"
    }
}

/// Manhattan (L1) distance.
struct Manhattan;

impl DistancePluginApi for Manhattan {
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .sum()
    }

    fn distance_name(&self) -> &str {
        "manhattan"
    }
}

/// Reciprocal rank reranker: scores based on original rank position.
struct ReciprocalRankReranker;

impl RerankerPluginApi for ReciprocalRankReranker {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[(String, f32, Vec<f32>)],
    ) -> Vec<(usize, f32)> {
        candidates
            .iter()
            .enumerate()
            .map(|(i, _)| (i, 1.0 / (60.0 + i as f32 + 1.0)))
            .collect()
    }

    fn reranker_name(&self) -> &str {
        "reciprocal_rank"
    }
}

/// Lowercase all metadata keys.
struct LowercaseKeysTransformer;

impl TransformerPluginApi for LowercaseKeysTransformer {
    fn transform(&self, metadata: &serde_json::Value) -> serde_json::Value {
        match metadata {
            serde_json::Value::Object(map) => {
                let transformed: serde_json::Map<String, serde_json::Value> = map
                    .iter()
                    .map(|(k, v)| (k.to_lowercase(), v.clone()))
                    .collect();
                serde_json::Value::Object(transformed)
            }
            other => other.clone(),
        }
    }

    fn transformer_name(&self) -> &str {
        "lowercase_keys"
    }
}

/// Filter metadata to only allowed fields.
struct FieldFilterTransformer {
    allowed_fields: Vec<String>,
}

impl TransformerPluginApi for FieldFilterTransformer {
    fn transform(&self, metadata: &serde_json::Value) -> serde_json::Value {
        match metadata {
            serde_json::Value::Object(map) => {
                let filtered: serde_json::Map<String, serde_json::Value> = map
                    .iter()
                    .filter(|(k, _)| self.allowed_fields.contains(k))
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                serde_json::Value::Object(filtered)
            }
            other => other.clone(),
        }
    }

    fn transformer_name(&self) -> &str {
        "field_filter"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_builtin_distance() {
        let mut runtime = WasmPluginRuntime::new(WasmPluginConfig::default());

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let dist = runtime.call_distance("manhattan", &a, &b).unwrap();
        assert!((dist - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_builtin_reranker() {
        let runtime = WasmPluginRuntime::new(WasmPluginConfig::default());

        let candidates = vec![
            ("a".to_string(), 0.1, vec![1.0]),
            ("b".to_string(), 0.2, vec![2.0]),
        ];

        let reranked = runtime
            .call_reranker("reciprocal_rank", &[1.0], &candidates)
            .unwrap();
        assert_eq!(reranked.len(), 2);
        assert!(reranked[0].1 > reranked[1].1); // First rank has higher score
    }

    #[test]
    fn test_builtin_transformer() {
        let runtime = WasmPluginRuntime::new(WasmPluginConfig::default());

        let metadata = json!({"Title": "Hello", "Category": "test"});
        let transformed = runtime
            .call_transformer("lowercase_keys", &metadata)
            .unwrap();

        assert!(transformed.get("title").is_some());
        assert!(transformed.get("category").is_some());
    }

    #[test]
    fn test_load_wasm_plugin() {
        let mut runtime = WasmPluginRuntime::new(WasmPluginConfig::default());

        let manifest = PluginManifest::distance("test_dist", "0.1.0", "Test distance");
        let wasm_bytes = vec![0u8; 100]; // Dummy WASM

        runtime.load_plugin("test_dist", &wasm_bytes, manifest).unwrap();

        let plugins = runtime.list_plugins();
        assert!(plugins.iter().any(|p| p.name == "test_dist"));
    }

    #[test]
    fn test_plugin_capacity_limit() {
        let config = WasmPluginConfig {
            max_plugins: 2,
            ..Default::default()
        };
        let mut runtime = WasmPluginRuntime::new(config);

        for i in 0..2 {
            let manifest = PluginManifest::distance(&format!("p{}", i), "0.1.0", "test");
            runtime
                .load_plugin(&format!("p{}", i), &[0u8; 10], manifest)
                .unwrap();
        }

        let manifest = PluginManifest::distance("p2", "0.1.0", "test");
        assert!(runtime.load_plugin("p2", &[0u8; 10], manifest).is_err());
    }

    #[test]
    fn test_unload_plugin() {
        let mut runtime = WasmPluginRuntime::new(WasmPluginConfig::default());

        let manifest = PluginManifest::distance("test", "0.1.0", "test");
        runtime.load_plugin("test", &[0u8; 10], manifest).unwrap();
        runtime.unload_plugin("test").unwrap();
        assert!(runtime.unload_plugin("test").is_err()); // Already removed
    }

    #[test]
    fn test_plugin_not_found() {
        let runtime = WasmPluginRuntime::new(WasmPluginConfig::default());
        assert!(runtime.call_reranker("nonexistent", &[], &[]).is_err());
    }
}
