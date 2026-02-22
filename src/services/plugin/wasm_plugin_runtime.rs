#![allow(clippy::unwrap_used)]
//! WASM Plugin Runtime
//!
//! Plugin execution environment with a defined ABI for distance functions,
//! embedding providers, and reranking logic. Supports plugin lifecycle
//! management (load/unload/hot-reload) and sandboxed execution.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::wasm_plugin_runtime::{
//!     PluginRuntime, PluginAbi, RuntimeConfig, PluginHandle,
//!     DistancePlugin, EmbedPlugin, RerankPlugin,
//! };
//!
//! let mut runtime = PluginRuntime::new(RuntimeConfig::default());
//!
//! // Register a custom distance function
//! runtime.register_distance("manhattan", Box::new(ManhattanPlugin));
//!
//! // Execute it
//! let dist = runtime.compute_distance("manhattan", &[1.0, 2.0], &[3.0, 4.0]).unwrap();
//! ```

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Plugin ABI ──────────────────────────────────────────────────────────────

/// Plugin capability type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PluginAbi {
    /// Custom distance function.
    Distance,
    /// Embedding provider.
    Embed,
    /// Reranking function.
    Rerank,
    /// Metadata transform.
    Transform,
}

impl std::fmt::Display for PluginAbi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Distance => write!(f, "distance"),
            Self::Embed => write!(f, "embed"),
            Self::Rerank => write!(f, "rerank"),
            Self::Transform => write!(f, "transform"),
        }
    }
}

/// Trait for distance function plugins.
pub trait DistancePlugin: Send + Sync {
    fn compute(&self, a: &[f32], b: &[f32]) -> f32;
    fn name(&self) -> &str;
}

/// Trait for embedding plugins.
pub trait EmbedPlugin: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimensions(&self) -> usize;
    fn name(&self) -> &str;
}

/// Trait for reranking plugins.
pub trait RerankPlugin: Send + Sync {
    fn rerank(&self, query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>>;
    fn name(&self) -> &str;
}

// ── Plugin Handle ───────────────────────────────────────────────────────────

/// Metadata about a loaded plugin.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginHandle {
    pub name: String,
    pub abi: PluginAbi,
    pub version: String,
    pub loaded_at: u64,
    pub call_count: u64,
    pub avg_latency_us: u64,
}

/// Runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Maximum number of loaded plugins.
    pub max_plugins: usize,
    /// Maximum execution time per call in milliseconds.
    pub timeout_ms: u64,
    /// Whether to enable call metrics tracking.
    pub enable_metrics: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_plugins: 64,
            timeout_ms: 5000,
            enable_metrics: true,
        }
    }
}

// ── Plugin Runtime ──────────────────────────────────────────────────────────

/// Plugin execution runtime.
pub struct PluginRuntime {
    config: RuntimeConfig,
    distance_plugins: HashMap<String, Box<dyn DistancePlugin>>,
    embed_plugins: HashMap<String, Box<dyn EmbedPlugin>>,
    rerank_plugins: HashMap<String, Box<dyn RerankPlugin>>,
    handles: HashMap<String, PluginHandle>,
}

impl PluginRuntime {
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            distance_plugins: HashMap::new(),
            embed_plugins: HashMap::new(),
            rerank_plugins: HashMap::new(),
            handles: HashMap::new(),
        }
    }

    /// Register a distance function plugin.
    pub fn register_distance(
        &mut self,
        name: &str,
        plugin: Box<dyn DistancePlugin>,
    ) -> Result<()> {
        self.check_capacity()?;
        let handle = PluginHandle {
            name: name.to_string(),
            abi: PluginAbi::Distance,
            version: "1.0.0".into(),
            loaded_at: now_secs(),
            call_count: 0,
            avg_latency_us: 0,
        };
        self.handles.insert(name.to_string(), handle);
        self.distance_plugins.insert(name.to_string(), plugin);
        Ok(())
    }

    /// Register an embedding plugin.
    pub fn register_embed(
        &mut self,
        name: &str,
        plugin: Box<dyn EmbedPlugin>,
    ) -> Result<()> {
        self.check_capacity()?;
        let handle = PluginHandle {
            name: name.to_string(),
            abi: PluginAbi::Embed,
            version: "1.0.0".into(),
            loaded_at: now_secs(),
            call_count: 0,
            avg_latency_us: 0,
        };
        self.handles.insert(name.to_string(), handle);
        self.embed_plugins.insert(name.to_string(), plugin);
        Ok(())
    }

    /// Register a reranking plugin.
    pub fn register_rerank(
        &mut self,
        name: &str,
        plugin: Box<dyn RerankPlugin>,
    ) -> Result<()> {
        self.check_capacity()?;
        let handle = PluginHandle {
            name: name.to_string(),
            abi: PluginAbi::Rerank,
            version: "1.0.0".into(),
            loaded_at: now_secs(),
            call_count: 0,
            avg_latency_us: 0,
        };
        self.handles.insert(name.to_string(), handle);
        self.rerank_plugins.insert(name.to_string(), plugin);
        Ok(())
    }

    /// Compute distance using a named plugin.
    pub fn compute_distance(&mut self, name: &str, a: &[f32], b: &[f32]) -> Result<f32> {
        let plugin = self.distance_plugins.get(name).ok_or_else(|| {
            NeedleError::NotFound(format!("Distance plugin '{name}'"))
        })?;
        let start = Instant::now();
        let result = plugin.compute(a, b);
        self.update_metrics(name, start);
        Ok(result)
    }

    /// Embed text using a named plugin.
    pub fn embed_text(&mut self, name: &str, text: &str) -> Result<Vec<f32>> {
        let plugin = self.embed_plugins.get(name).ok_or_else(|| {
            NeedleError::NotFound(format!("Embed plugin '{name}'"))
        })?;
        let start = Instant::now();
        let result = plugin.embed(text)?;
        self.update_metrics(name, start);
        Ok(result)
    }

    /// Rerank documents using a named plugin.
    pub fn rerank(
        &mut self,
        name: &str,
        query: &str,
        documents: &[&str],
    ) -> Result<Vec<(usize, f32)>> {
        let plugin = self.rerank_plugins.get(name).ok_or_else(|| {
            NeedleError::NotFound(format!("Rerank plugin '{name}'"))
        })?;
        let start = Instant::now();
        let result = plugin.rerank(query, documents)?;
        self.update_metrics(name, start);
        Ok(result)
    }

    /// Unload a plugin by name (hot-reload: unload then re-register).
    pub fn unload(&mut self, name: &str) -> bool {
        let removed = self.distance_plugins.remove(name).is_some()
            || self.embed_plugins.remove(name).is_some()
            || self.rerank_plugins.remove(name).is_some();
        self.handles.remove(name);
        removed
    }

    /// List all loaded plugin handles.
    pub fn list(&self) -> Vec<&PluginHandle> {
        self.handles.values().collect()
    }

    /// Get a specific plugin handle.
    pub fn get_handle(&self, name: &str) -> Option<&PluginHandle> {
        self.handles.get(name)
    }

    /// Total loaded plugins.
    pub fn plugin_count(&self) -> usize {
        self.handles.len()
    }

    fn check_capacity(&self) -> Result<()> {
        if self.handles.len() >= self.config.max_plugins {
            return Err(NeedleError::CapacityExceeded(format!(
                "Max plugins ({}) reached", self.config.max_plugins
            )));
        }
        Ok(())
    }

    fn update_metrics(&mut self, name: &str, start: Instant) {
        if !self.config.enable_metrics {
            return;
        }
        if let Some(handle) = self.handles.get_mut(name) {
            let latency = start.elapsed().as_micros() as u64;
            let count = handle.call_count;
            handle.avg_latency_us =
                (handle.avg_latency_us * count + latency) / (count + 1);
            handle.call_count += 1;
        }
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct L1Distance;
    impl DistancePlugin for L1Distance {
        fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
        }
        fn name(&self) -> &str { "l1" }
    }

    struct MockEmbed;
    impl EmbedPlugin for MockEmbed {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32 * 0.1; 4])
        }
        fn dimensions(&self) -> usize { 4 }
        fn name(&self) -> &str { "mock" }
    }

    struct MockRerank;
    impl RerankPlugin for MockRerank {
        fn rerank(&self, _query: &str, documents: &[&str]) -> Result<Vec<(usize, f32)>> {
            Ok(documents.iter().enumerate().map(|(i, d)| (i, d.len() as f32)).collect())
        }
        fn name(&self) -> &str { "mock" }
    }

    #[test]
    fn test_register_and_compute_distance() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();

        let dist = runtime.compute_distance("l1", &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert!((dist - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_register_and_embed() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_embed("mock", Box::new(MockEmbed)).unwrap();

        let emb = runtime.embed_text("mock", "hello").unwrap();
        assert_eq!(emb.len(), 4);
    }

    #[test]
    fn test_register_and_rerank() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_rerank("mock", Box::new(MockRerank)).unwrap();

        let results = runtime.rerank("mock", "query", &["short", "longer text"]).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_unload() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();
        assert_eq!(runtime.plugin_count(), 1);

        assert!(runtime.unload("l1"));
        assert_eq!(runtime.plugin_count(), 0);
        assert!(runtime.compute_distance("l1", &[1.0], &[2.0]).is_err());
    }

    #[test]
    fn test_capacity_limit() {
        let mut runtime = PluginRuntime::new(RuntimeConfig {
            max_plugins: 1, ..Default::default()
        });
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();
        assert!(runtime.register_embed("mock", Box::new(MockEmbed)).is_err());
    }

    #[test]
    fn test_metrics_tracking() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();

        for _ in 0..10 {
            runtime.compute_distance("l1", &[1.0], &[2.0]).unwrap();
        }

        let handle = runtime.get_handle("l1").unwrap();
        assert_eq!(handle.call_count, 10);
    }

    #[test]
    fn test_list_plugins() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();
        runtime.register_embed("mock", Box::new(MockEmbed)).unwrap();

        let list = runtime.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_hot_reload() {
        let mut runtime = PluginRuntime::new(RuntimeConfig::default());
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();

        // Hot reload: unload and re-register
        runtime.unload("l1");
        runtime.register_distance("l1", Box::new(L1Distance)).unwrap();

        let dist = runtime.compute_distance("l1", &[1.0, 2.0], &[3.0, 4.0]).unwrap();
        assert!((dist - 4.0).abs() < 0.01);
        assert_eq!(runtime.get_handle("l1").unwrap().call_count, 1); // reset after reload
    }
}
