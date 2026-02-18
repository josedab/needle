//! Plugin Runtime Service
//!
//! Runtime plugin management that integrates the plugin system with the
//! Database/Collection pipeline, supporting dynamic registration of custom
//! distance functions, pre/post-search hooks, and transform plugins.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::plugin_runtime::{PluginRuntime, PluginRuntimeConfig, PluginHook};
//! use needle::plugin::{Plugin, PluginManifest, PluginType};
//! use needle::Database;
//!
//! let db = Database::in_memory();
//! db.create_collection("docs", 128).unwrap();
//!
//! let config = PluginRuntimeConfig::builder()
//!     .max_plugins(50)
//!     .enable_sandboxing(false)
//!     .build();
//!
//! let mut runtime = PluginRuntime::new(config);
//!
//! // Register a custom pre-search hook
//! runtime.register_hook(
//!     "normalize_query",
//!     PluginHook::PreSearch(Box::new(|query| {
//!         let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
//!         if norm > 0.0 {
//!             query.iter().map(|x| x / norm).collect()
//!         } else {
//!             query.to_vec()
//!         }
//!     })),
//! );
//!
//! // Execute hooks in pipeline
//! let query = vec![0.3f32; 128];
//! let processed = runtime.run_pre_search_hooks(&query);
//! ```

use std::collections::HashMap;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::collection::SearchResult;
use crate::error::{NeedleError, Result};

/// Configuration for the plugin runtime.
#[derive(Debug, Clone)]
pub struct PluginRuntimeConfig {
    pub max_plugins: usize,
    pub enable_sandboxing: bool,
    pub max_hook_duration_ms: u64,
    pub enable_metrics: bool,
}

impl Default for PluginRuntimeConfig {
    fn default() -> Self {
        Self {
            max_plugins: 100,
            enable_sandboxing: false,
            max_hook_duration_ms: 100,
            enable_metrics: true,
        }
    }
}

pub struct PluginRuntimeConfigBuilder {
    config: PluginRuntimeConfig,
}

impl PluginRuntimeConfig {
    pub fn builder() -> PluginRuntimeConfigBuilder {
        PluginRuntimeConfigBuilder {
            config: Self::default(),
        }
    }
}

impl PluginRuntimeConfigBuilder {
    #[must_use]
    pub fn max_plugins(mut self, max: usize) -> Self {
        self.config.max_plugins = max;
        self
    }

    #[must_use]
    pub fn enable_sandboxing(mut self, enable: bool) -> Self {
        self.config.enable_sandboxing = enable;
        self
    }

    #[must_use]
    pub fn max_hook_duration_ms(mut self, ms: u64) -> Self {
        self.config.max_hook_duration_ms = ms;
        self
    }

    #[must_use]
    pub fn enable_metrics(mut self, enable: bool) -> Self {
        self.config.enable_metrics = enable;
        self
    }

    pub fn build(self) -> PluginRuntimeConfig {
        self.config
    }
}

/// Type alias for pre-search hook function.
pub type PreSearchFn = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

/// Type alias for post-search hook function.
pub type PostSearchFn = Box<dyn Fn(Vec<SearchResult>) -> Vec<SearchResult> + Send + Sync>;

/// Type alias for custom distance function.
pub type DistanceFn = Box<dyn Fn(&[f32], &[f32]) -> f32 + Send + Sync>;

/// Type alias for vector transform function.
pub type TransformFn = Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>;

/// Hook types that can be registered with the runtime.
pub enum PluginHook {
    /// Pre-search: transform query vector before searching.
    PreSearch(PreSearchFn),
    /// Post-search: filter/rerank results after searching.
    PostSearch(PostSearchFn),
    /// Custom distance function.
    Distance(DistanceFn),
    /// Vector transform (applied on insert).
    Transform(TransformFn),
}

/// Metadata about a registered hook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookInfo {
    pub name: String,
    pub hook_type: String,
    pub enabled: bool,
    pub invocation_count: u64,
    pub total_duration_us: u64,
    pub avg_duration_us: u64,
}

/// Plugin runtime managing hooks and their execution.
pub struct PluginRuntime {
    config: PluginRuntimeConfig,
    pre_search_hooks: RwLock<Vec<(String, PreSearchFn)>>,
    post_search_hooks: RwLock<Vec<(String, PostSearchFn)>>,
    distance_fns: RwLock<HashMap<String, DistanceFn>>,
    transform_fns: RwLock<Vec<(String, TransformFn)>>,
    hook_stats: RwLock<HashMap<String, (u64, u64)>>, // (count, total_us)
    enabled_hooks: RwLock<HashMap<String, bool>>,
}

impl PluginRuntime {
    /// Create a new plugin runtime.
    pub fn new(config: PluginRuntimeConfig) -> Self {
        Self {
            config,
            pre_search_hooks: RwLock::new(Vec::new()),
            post_search_hooks: RwLock::new(Vec::new()),
            distance_fns: RwLock::new(HashMap::new()),
            transform_fns: RwLock::new(Vec::new()),
            hook_stats: RwLock::new(HashMap::new()),
            enabled_hooks: RwLock::new(HashMap::new()),
        }
    }

    /// Register a hook with the runtime.
    pub fn register_hook(&self, name: impl Into<String>, hook: PluginHook) -> Result<()> {
        let name = name.into();
        let total = self.total_hooks();
        if total >= self.config.max_plugins {
            return Err(NeedleError::InvalidArgument(format!(
                "max plugins ({}) reached",
                self.config.max_plugins
            )));
        }

        self.enabled_hooks.write().insert(name.clone(), true);

        match hook {
            PluginHook::PreSearch(f) => {
                self.pre_search_hooks.write().push((name, f));
            }
            PluginHook::PostSearch(f) => {
                self.post_search_hooks.write().push((name, f));
            }
            PluginHook::Distance(f) => {
                self.distance_fns.write().insert(name, f);
            }
            PluginHook::Transform(f) => {
                self.transform_fns.write().push((name, f));
            }
        }
        Ok(())
    }

    /// Unregister a hook by name.
    pub fn unregister_hook(&self, name: &str) -> bool {
        let mut removed = false;

        self.pre_search_hooks.write().retain(|(n, _)| {
            if n == name {
                removed = true;
                false
            } else {
                true
            }
        });
        self.post_search_hooks.write().retain(|(n, _)| {
            if n == name {
                removed = true;
                false
            } else {
                true
            }
        });
        if self.distance_fns.write().remove(name).is_some() {
            removed = true;
        }
        self.transform_fns.write().retain(|(n, _)| {
            if n == name {
                removed = true;
                false
            } else {
                true
            }
        });
        self.enabled_hooks.write().remove(name);
        removed
    }

    /// Enable or disable a hook.
    pub fn set_enabled(&self, name: &str, enabled: bool) -> bool {
        if let Some(entry) = self.enabled_hooks.write().get_mut(name) {
            *entry = enabled;
            true
        } else {
            false
        }
    }

    /// Run all pre-search hooks on a query vector.
    pub fn run_pre_search_hooks(&self, query: &[f32]) -> Vec<f32> {
        let hooks = self.pre_search_hooks.read();
        let enabled = self.enabled_hooks.read();
        let mut result = query.to_vec();

        for (name, hook) in hooks.iter() {
            if !enabled.get(name).copied().unwrap_or(true) {
                continue;
            }
            let start = Instant::now();
            result = hook(&result);
            self.record_invocation(name, start.elapsed());
        }
        result
    }

    /// Run all post-search hooks on search results.
    pub fn run_post_search_hooks(&self, results: Vec<SearchResult>) -> Vec<SearchResult> {
        let hooks = self.post_search_hooks.read();
        let enabled = self.enabled_hooks.read();
        let mut results = results;

        for (name, hook) in hooks.iter() {
            if !enabled.get(name).copied().unwrap_or(true) {
                continue;
            }
            let start = Instant::now();
            results = hook(results);
            self.record_invocation(name, start.elapsed());
        }
        results
    }

    /// Run all transform hooks on a vector (for insert-time transforms).
    pub fn run_transform_hooks(&self, vector: &[f32]) -> Vec<f32> {
        let hooks = self.transform_fns.read();
        let enabled = self.enabled_hooks.read();
        let mut result = vector.to_vec();

        for (name, hook) in hooks.iter() {
            if !enabled.get(name).copied().unwrap_or(true) {
                continue;
            }
            let start = Instant::now();
            result = hook(&result);
            self.record_invocation(name, start.elapsed());
        }
        result
    }

    /// Compute distance using a named custom distance function.
    pub fn compute_distance(&self, name: &str, a: &[f32], b: &[f32]) -> Option<f32> {
        let fns = self.distance_fns.read();
        fns.get(name).map(|f| {
            let start = Instant::now();
            let d = f(a, b);
            self.record_invocation(name, start.elapsed());
            d
        })
    }

    /// List all registered hooks with statistics.
    pub fn list_hooks(&self) -> Vec<HookInfo> {
        let stats = self.hook_stats.read();
        let enabled = self.enabled_hooks.read();
        let mut hooks = Vec::new();

        let collect = |name: &str, hook_type: &str| {
            let (count, total_us) = stats.get(name).copied().unwrap_or((0, 0));
            HookInfo {
                name: name.to_string(),
                hook_type: hook_type.to_string(),
                enabled: enabled.get(name).copied().unwrap_or(true),
                invocation_count: count,
                total_duration_us: total_us,
                avg_duration_us: if count > 0 { total_us / count } else { 0 },
            }
        };

        for (name, _) in self.pre_search_hooks.read().iter() {
            hooks.push(collect(name, "PreSearch"));
        }
        for (name, _) in self.post_search_hooks.read().iter() {
            hooks.push(collect(name, "PostSearch"));
        }
        for name in self.distance_fns.read().keys() {
            hooks.push(collect(name, "Distance"));
        }
        for (name, _) in self.transform_fns.read().iter() {
            hooks.push(collect(name, "Transform"));
        }

        hooks
    }

    /// Total number of registered hooks.
    pub fn total_hooks(&self) -> usize {
        self.pre_search_hooks.read().len()
            + self.post_search_hooks.read().len()
            + self.distance_fns.read().len()
            + self.transform_fns.read().len()
    }

    fn record_invocation(&self, name: &str, duration: std::time::Duration) {
        if self.config.enable_metrics {
            let us = duration.as_micros() as u64;
            let mut stats = self.hook_stats.write();
            let entry = stats.entry(name.to_string()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += us;
        }
    }
}

// ── Event Bus & Serverless Function Runtime ─────────────────────────────────

/// Events that can trigger serverless functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CollectionEvent {
    /// A vector was inserted.
    VectorInserted {
        collection: String,
        id: String,
        dimensions: usize,
    },
    /// A vector was updated.
    VectorUpdated {
        collection: String,
        id: String,
        dimensions: usize,
    },
    /// A vector was deleted.
    VectorDeleted {
        collection: String,
        id: String,
    },
    /// A collection was created.
    CollectionCreated {
        name: String,
        dimensions: usize,
    },
    /// A collection was dropped.
    CollectionDropped {
        name: String,
    },
}

impl CollectionEvent {
    /// Get the event type name.
    pub fn event_type(&self) -> &'static str {
        match self {
            Self::VectorInserted { .. } => "vector.inserted",
            Self::VectorUpdated { .. } => "vector.updated",
            Self::VectorDeleted { .. } => "vector.deleted",
            Self::CollectionCreated { .. } => "collection.created",
            Self::CollectionDropped { .. } => "collection.dropped",
        }
    }

    /// Get the collection name associated with the event.
    pub fn collection(&self) -> &str {
        match self {
            Self::VectorInserted { collection, .. }
            | Self::VectorUpdated { collection, .. }
            | Self::VectorDeleted { collection, .. } => collection,
            Self::CollectionCreated { name, .. }
            | Self::CollectionDropped { name, .. } => name,
        }
    }
}

/// Type alias for event handler function.
pub type EventHandlerFn = Box<dyn Fn(&CollectionEvent) -> Result<()> + Send + Sync>;

/// A registered serverless function.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ServerlessFunction {
    /// Function name.
    pub name: String,
    /// Event types this function listens to.
    pub event_filters: Vec<String>,
    /// Optional collection filter (empty means all collections).
    pub collection_filter: Option<String>,
    /// Whether the function is enabled.
    pub enabled: bool,
    /// Maximum execution time in milliseconds.
    pub timeout_ms: u64,
    /// Number of times invoked.
    pub invocations: u64,
    /// Number of errors.
    pub errors: u64,
    /// Deployment timestamp.
    pub deployed_at: u64,
    /// Last invocation timestamp.
    pub last_invoked_at: Option<u64>,
}

/// Configuration for the serverless function runtime.
#[derive(Debug, Clone)]
pub struct ServerlessFunctionConfig {
    /// Maximum number of deployed functions.
    pub max_functions: usize,
    /// Default timeout for function execution (ms).
    pub default_timeout_ms: u64,
    /// Maximum timeout allowed (ms).
    pub max_timeout_ms: u64,
    /// Maximum log entries to retain per function.
    pub max_log_entries: usize,
}

impl Default for ServerlessFunctionConfig {
    fn default() -> Self {
        Self {
            max_functions: 100,
            default_timeout_ms: 5000,
            max_timeout_ms: 30000,
            max_log_entries: 1000,
        }
    }
}

/// Log entry from a function invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionLogEntry {
    /// Function name.
    pub function_name: String,
    /// Event type that triggered the invocation.
    pub event_type: String,
    /// Whether the invocation succeeded.
    pub success: bool,
    /// Error message (if failed).
    pub error: Option<String>,
    /// Execution duration in microseconds.
    pub duration_us: u64,
    /// Timestamp.
    pub timestamp: u64,
}

/// Serverless function runtime with event-driven execution.
pub struct ServerlessFunctionRuntime {
    config: ServerlessFunctionConfig,
    functions: RwLock<HashMap<String, ServerlessFunction>>,
    handlers: RwLock<HashMap<String, EventHandlerFn>>,
    logs: RwLock<Vec<FunctionLogEntry>>,
}

impl ServerlessFunctionRuntime {
    /// Create a new serverless function runtime.
    pub fn new(config: ServerlessFunctionConfig) -> Self {
        Self {
            config,
            functions: RwLock::new(HashMap::new()),
            handlers: RwLock::new(HashMap::new()),
            logs: RwLock::new(Vec::new()),
        }
    }

    /// Deploy a new serverless function.
    pub fn deploy(
        &self,
        name: &str,
        event_filters: Vec<String>,
        collection_filter: Option<String>,
        handler: EventHandlerFn,
    ) -> Result<()> {
        let functions = self.functions.read();
        if functions.len() >= self.config.max_functions {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum functions ({}) reached",
                self.config.max_functions
            )));
        }
        if functions.contains_key(name) {
            return Err(NeedleError::DuplicateId(format!(
                "Function '{}' already deployed",
                name
            )));
        }
        drop(functions);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let func = ServerlessFunction {
            name: name.to_string(),
            event_filters,
            collection_filter,
            enabled: true,
            timeout_ms: self.config.default_timeout_ms,
            invocations: 0,
            errors: 0,
            deployed_at: now,
            last_invoked_at: None,
        };

        self.functions.write().insert(name.to_string(), func);
        self.handlers.write().insert(name.to_string(), handler);
        Ok(())
    }

    /// Remove a deployed function.
    pub fn remove(&self, name: &str) -> Result<()> {
        self.functions
            .write()
            .remove(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Function '{name}' not found")))?;
        self.handlers.write().remove(name);
        Ok(())
    }

    /// Dispatch an event to all matching functions.
    pub fn dispatch(&self, event: &CollectionEvent) -> Vec<FunctionLogEntry> {
        let event_type = event.event_type().to_string();
        let collection = event.collection().to_string();
        let mut log_entries = Vec::new();

        let functions = self.functions.read();
        let handlers = self.handlers.read();

        for (name, func) in functions.iter() {
            if !func.enabled {
                continue;
            }

            // Check event filter
            let event_matches = func.event_filters.is_empty()
                || func.event_filters.iter().any(|f| f == &event_type || f == "*");

            // Check collection filter
            let collection_matches = func
                .collection_filter
                .as_ref()
                .is_none_or(|c| c == &collection);

            if !event_matches || !collection_matches {
                continue;
            }

            let start = Instant::now();
            let result = handlers.get(name).map(|h| h(event));

            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();

            let entry = FunctionLogEntry {
                function_name: name.clone(),
                event_type: event_type.clone(),
                success: result.as_ref().is_some_and(|r| r.is_ok()),
                error: result
                    .as_ref()
                    .and_then(|r| r.as_ref().err().map(|e| e.to_string())),
                duration_us: start.elapsed().as_micros() as u64,
                timestamp: now,
            };

            log_entries.push(entry);
        }

        // Update stats (we need to drop reads first)
        drop(functions);
        drop(handlers);

        let mut funcs = self.functions.write();
        for entry in &log_entries {
            if let Some(func) = funcs.get_mut(&entry.function_name) {
                func.invocations += 1;
                func.last_invoked_at = Some(entry.timestamp);
                if !entry.success {
                    func.errors += 1;
                }
            }
        }
        drop(funcs);

        // Store logs
        let mut logs = self.logs.write();
        for entry in &log_entries {
            logs.push(entry.clone());
        }
        while logs.len() > self.config.max_log_entries {
            logs.remove(0);
        }

        log_entries
    }

    /// List all deployed functions.
    pub fn list(&self) -> Vec<ServerlessFunction> {
        self.functions.read().values().cloned().collect()
    }

    /// Get function logs, optionally filtered by function name.
    pub fn logs(&self, function_name: Option<&str>) -> Vec<FunctionLogEntry> {
        let logs = self.logs.read();
        match function_name {
            Some(name) => logs.iter().filter(|l| l.function_name == name).cloned().collect(),
            None => logs.clone(),
        }
    }

    /// Enable or disable a function.
    pub fn set_enabled(&self, name: &str, enabled: bool) -> Result<()> {
        self.functions
            .write()
            .get_mut(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Function '{name}' not found")))?
            .enabled = enabled;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_run_pre_search() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook(
                "normalize",
                PluginHook::PreSearch(Box::new(|query| {
                    let norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        query.iter().map(|x| x / norm).collect()
                    } else {
                        query.to_vec()
                    }
                })),
            )
            .unwrap();

        let query = vec![3.0, 4.0];
        let result = runtime.run_pre_search_hooks(&query);
        assert!((result[0] - 0.6).abs() < 0.01);
        assert!((result[1] - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_register_and_run_post_search() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook(
                "top_3",
                PluginHook::PostSearch(Box::new(|mut results| {
                    results.truncate(3);
                    results
                })),
            )
            .unwrap();

        let results: Vec<SearchResult> = (0..10)
            .map(|i| SearchResult {
                id: format!("v{}", i),
                distance: i as f32,
                metadata: None,
            })
            .collect();

        let filtered = runtime.run_post_search_hooks(results);
        assert_eq!(filtered.len(), 3);
    }

    #[test]
    fn test_custom_distance() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook(
                "l1",
                PluginHook::Distance(Box::new(|a, b| {
                    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
                })),
            )
            .unwrap();

        let d = runtime
            .compute_distance("l1", &[1.0, 2.0], &[3.0, 4.0])
            .unwrap();
        assert!((d - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_unregister() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook("dummy", PluginHook::PreSearch(Box::new(|q| q.to_vec())))
            .unwrap();
        assert_eq!(runtime.total_hooks(), 1);

        assert!(runtime.unregister_hook("dummy"));
        assert_eq!(runtime.total_hooks(), 0);
    }

    #[test]
    fn test_disable_hook() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook(
                "double",
                PluginHook::PreSearch(Box::new(|q| q.iter().map(|x| x * 2.0).collect())),
            )
            .unwrap();

        let q = vec![1.0, 2.0];
        let result = runtime.run_pre_search_hooks(&q);
        assert_eq!(result, vec![2.0, 4.0]);

        // Disable
        runtime.set_enabled("double", false);
        let result = runtime.run_pre_search_hooks(&q);
        assert_eq!(result, vec![1.0, 2.0]); // unchanged
    }

    #[test]
    fn test_max_plugins() {
        let config = PluginRuntimeConfig::builder().max_plugins(2).build();
        let runtime = PluginRuntime::new(config);

        runtime
            .register_hook("h1", PluginHook::PreSearch(Box::new(|q| q.to_vec())))
            .unwrap();
        runtime
            .register_hook("h2", PluginHook::PreSearch(Box::new(|q| q.to_vec())))
            .unwrap();

        let result = runtime.register_hook("h3", PluginHook::PreSearch(Box::new(|q| q.to_vec())));
        assert!(result.is_err());
    }

    #[test]
    fn test_list_hooks() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook("pre1", PluginHook::PreSearch(Box::new(|q| q.to_vec())))
            .unwrap();
        runtime
            .register_hook("dist1", PluginHook::Distance(Box::new(|_, _| 0.0)))
            .unwrap();

        let hooks = runtime.list_hooks();
        assert_eq!(hooks.len(), 2);
        assert!(hooks.iter().any(|h| h.name == "pre1"));
        assert!(hooks.iter().any(|h| h.name == "dist1"));
    }

    #[test]
    fn test_transform_hooks() {
        let runtime = PluginRuntime::new(PluginRuntimeConfig::default());

        runtime
            .register_hook(
                "scale",
                PluginHook::Transform(Box::new(|v| v.iter().map(|x| x * 10.0).collect())),
            )
            .unwrap();

        let v = vec![0.1, 0.2];
        let result = runtime.run_transform_hooks(&v);
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!((result[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_serverless_deploy_and_dispatch() {
        let runtime = ServerlessFunctionRuntime::new(ServerlessFunctionConfig::default());

        let invoked = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let invoked_clone = invoked.clone();

        runtime.deploy(
            "on_insert",
            vec!["vector.inserted".to_string()],
            None,
            Box::new(move |_event| {
                invoked_clone.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            }),
        ).expect("deploy should succeed");

        assert_eq!(runtime.list().len(), 1);

        let event = CollectionEvent::VectorInserted {
            collection: "docs".to_string(),
            id: "doc1".to_string(),
            dimensions: 4,
        };

        let logs = runtime.dispatch(&event);
        assert_eq!(logs.len(), 1);
        assert!(logs[0].success);
        assert!(invoked.load(std::sync::atomic::Ordering::SeqCst));
    }

    #[test]
    fn test_serverless_event_filter() {
        let runtime = ServerlessFunctionRuntime::new(ServerlessFunctionConfig::default());

        runtime.deploy(
            "only_deletes",
            vec!["vector.deleted".to_string()],
            None,
            Box::new(|_| Ok(())),
        ).expect("deploy");

        // Insert event should not trigger
        let insert_event = CollectionEvent::VectorInserted {
            collection: "docs".to_string(),
            id: "doc1".to_string(),
            dimensions: 4,
        };
        let logs = runtime.dispatch(&insert_event);
        assert!(logs.is_empty());

        // Delete event should trigger
        let delete_event = CollectionEvent::VectorDeleted {
            collection: "docs".to_string(),
            id: "doc1".to_string(),
        };
        let logs = runtime.dispatch(&delete_event);
        assert_eq!(logs.len(), 1);
    }

    #[test]
    fn test_serverless_collection_filter() {
        let runtime = ServerlessFunctionRuntime::new(ServerlessFunctionConfig::default());

        runtime.deploy(
            "only_docs",
            vec!["*".to_string()],
            Some("docs".to_string()),
            Box::new(|_| Ok(())),
        ).expect("deploy");

        let docs_event = CollectionEvent::VectorInserted {
            collection: "docs".to_string(),
            id: "d1".to_string(),
            dimensions: 4,
        };
        assert_eq!(runtime.dispatch(&docs_event).len(), 1);

        let other_event = CollectionEvent::VectorInserted {
            collection: "other".to_string(),
            id: "o1".to_string(),
            dimensions: 4,
        };
        assert!(runtime.dispatch(&other_event).is_empty());
    }

    #[test]
    fn test_serverless_remove() {
        let runtime = ServerlessFunctionRuntime::new(ServerlessFunctionConfig::default());
        runtime.deploy("f1", vec![], None, Box::new(|_| Ok(()))).expect("deploy");
        assert_eq!(runtime.list().len(), 1);

        runtime.remove("f1").expect("remove");
        assert!(runtime.list().is_empty());
        assert!(runtime.remove("f1").is_err());
    }

    #[test]
    fn test_serverless_disable() {
        let runtime = ServerlessFunctionRuntime::new(ServerlessFunctionConfig::default());
        runtime.deploy("f1", vec!["*".to_string()], None, Box::new(|_| Ok(()))).expect("deploy");

        runtime.set_enabled("f1", false).expect("disable");

        let event = CollectionEvent::VectorInserted {
            collection: "docs".to_string(),
            id: "d1".to_string(),
            dimensions: 4,
        };
        assert!(runtime.dispatch(&event).is_empty());
    }
}
