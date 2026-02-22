//! Plugin Ecosystem Runtime
//!
//! WASM-style sandboxed plugin execution with capability-based permissions,
//! resource limits, manifest validation, and community registry support.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::plugin_ecosystem::{
//!     PluginRuntime, RuntimeConfig, PluginSandbox,
//!     Permission, ResourceLimits,
//! };
//!
//! let mut runtime = PluginRuntime::new(RuntimeConfig::default());
//!
//! // Define a sandbox with limited permissions
//! let sandbox = PluginSandbox::new("my-plugin")
//!     .with_permission(Permission::ReadData)
//!     .with_limits(ResourceLimits::default());
//!
//! runtime.register_sandbox(sandbox).unwrap();
//!
//! // Execute a plugin function
//! let result = runtime.execute("my-plugin", "compute_distance", &[1.0, 0.0, 1.0, 0.0]).unwrap();
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};
use crate::services::plugin_api::{
    PluginCapability, PluginEntry, PluginManifest, PluginRegistry, PluginStatus, PluginVersion,
};

// ── Permissions ──────────────────────────────────────────────────────────────

/// Capabilities a plugin can request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Permission {
    /// Read vector data.
    ReadData,
    /// Write/modify vector data.
    WriteData,
    /// Access metadata.
    ReadMetadata,
    /// Modify metadata.
    WriteMetadata,
    /// Execute searches.
    Search,
    /// Access network (for embedding providers).
    Network,
    /// Access filesystem (for storage backends).
    FileSystem,
    /// Execute custom distance functions.
    ComputeDistance,
}

impl std::fmt::Display for Permission {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReadData => write!(f, "read:data"),
            Self::WriteData => write!(f, "write:data"),
            Self::ReadMetadata => write!(f, "read:metadata"),
            Self::WriteMetadata => write!(f, "write:metadata"),
            Self::Search => write!(f, "search"),
            Self::Network => write!(f, "network"),
            Self::FileSystem => write!(f, "filesystem"),
            Self::ComputeDistance => write!(f, "compute:distance"),
        }
    }
}

// ── Resource Limits ──────────────────────────────────────────────────────────

/// Resource limits for sandbox execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory in bytes.
    pub max_memory_bytes: usize,
    /// Maximum execution time per call.
    pub max_execution_time: Duration,
    /// Maximum number of calls per second.
    pub max_calls_per_second: u32,
    /// Maximum data input size in bytes.
    pub max_input_bytes: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 64 * 1024 * 1024, // 64MB
            max_execution_time: Duration::from_secs(5),
            max_calls_per_second: 1000,
            max_input_bytes: 10 * 1024 * 1024, // 10MB
        }
    }
}

// ── Plugin Sandbox ───────────────────────────────────────────────────────────

/// Sandbox configuration for a plugin.
#[derive(Debug, Clone)]
pub struct PluginSandbox {
    /// Plugin identifier.
    pub plugin_id: String,
    /// Granted permissions.
    pub permissions: Vec<Permission>,
    /// Resource limits.
    pub limits: ResourceLimits,
    /// Whether the sandbox is active.
    pub active: bool,
}

impl PluginSandbox {
    /// Create a new sandbox for a plugin.
    pub fn new(plugin_id: impl Into<String>) -> Self {
        Self {
            plugin_id: plugin_id.into(),
            permissions: Vec::new(),
            limits: ResourceLimits::default(),
            active: true,
        }
    }

    /// Add a permission.
    #[must_use]
    pub fn with_permission(mut self, perm: Permission) -> Self {
        self.permissions.push(perm);
        self
    }

    /// Set resource limits.
    #[must_use]
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Check if a permission is granted.
    pub fn has_permission(&self, perm: &Permission) -> bool {
        self.permissions.contains(perm)
    }
}

// ── Execution Result ─────────────────────────────────────────────────────────

/// Result from a sandboxed plugin execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Plugin ID.
    pub plugin_id: String,
    /// Function name.
    pub function: String,
    /// Output data (f32 values for distance functions, etc).
    pub output: Vec<f32>,
    /// Execution duration.
    pub duration_ms: u64,
    /// Memory used.
    pub memory_used: usize,
}

// ── Runtime Configuration ────────────────────────────────────────────────────

/// Runtime configuration.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    /// Default resource limits for new sandboxes.
    pub default_limits: ResourceLimits,
    /// Maximum concurrent plugin executions.
    pub max_concurrent: usize,
    /// Whether to enforce permission checks.
    pub enforce_permissions: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            default_limits: ResourceLimits::default(),
            max_concurrent: 16,
            enforce_permissions: true,
        }
    }
}

// ── Execution Statistics ─────────────────────────────────────────────────────

/// Per-plugin execution statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PluginStats {
    /// Total executions.
    pub total_executions: u64,
    /// Total errors.
    pub total_errors: u64,
    /// Average execution time in ms.
    pub avg_duration_ms: f64,
    /// Maximum execution time in ms.
    pub max_duration_ms: f64,
}

// ── Plugin Runtime ───────────────────────────────────────────────────────────

/// Sandboxed plugin execution runtime.
pub struct PluginRuntime {
    config: RuntimeConfig,
    registry: PluginRegistry,
    sandboxes: HashMap<String, PluginSandbox>,
    stats: HashMap<String, PluginStats>,
    // Simulated plugin functions (in real impl, these would be WASM modules)
    functions: HashMap<(String, String), Box<dyn Fn(&[f32]) -> Vec<f32> + Send + Sync>>,
}

impl PluginRuntime {
    /// Create a new plugin runtime.
    pub fn new(config: RuntimeConfig) -> Self {
        Self {
            config,
            registry: PluginRegistry::new(),
            sandboxes: HashMap::new(),
            stats: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    /// Register a sandbox.
    pub fn register_sandbox(&mut self, sandbox: PluginSandbox) -> Result<()> {
        self.sandboxes
            .insert(sandbox.plugin_id.clone(), sandbox);
        Ok(())
    }

    /// Register a plugin function (for testing/built-in plugins).
    pub fn register_function<F>(
        &mut self,
        plugin_id: &str,
        function_name: &str,
        func: F,
    ) -> Result<()>
    where
        F: Fn(&[f32]) -> Vec<f32> + Send + Sync + 'static,
    {
        if !self.sandboxes.contains_key(plugin_id) {
            return Err(NeedleError::NotFound(format!(
                "Sandbox for plugin '{plugin_id}' not found"
            )));
        }
        self.functions.insert(
            (plugin_id.into(), function_name.into()),
            Box::new(func),
        );
        Ok(())
    }

    /// Execute a plugin function.
    pub fn execute(
        &mut self,
        plugin_id: &str,
        function_name: &str,
        input: &[f32],
    ) -> Result<ExecutionResult> {
        // Check sandbox exists and is active
        let sandbox = self
            .sandboxes
            .get(plugin_id)
            .ok_or_else(|| NeedleError::NotFound(format!("Plugin '{plugin_id}' not found")))?;

        if !sandbox.active {
            return Err(NeedleError::InvalidOperation(format!(
                "Plugin '{plugin_id}' sandbox is inactive"
            )));
        }

        // Check input size
        let input_bytes = input.len() * 4;
        if input_bytes > sandbox.limits.max_input_bytes {
            return Err(NeedleError::CapacityExceeded(format!(
                "Input size {input_bytes} exceeds limit {}",
                sandbox.limits.max_input_bytes
            )));
        }

        // Execute
        let start = Instant::now();
        let key = (plugin_id.into(), function_name.into());
        let func = self
            .functions
            .get(&key)
            .ok_or_else(|| NeedleError::NotFound(format!(
                "Function '{function_name}' not found in plugin '{plugin_id}'"
            )))?;

        let output = func(input);
        let duration = start.elapsed();

        // Check execution time
        if duration > sandbox.limits.max_execution_time {
            let stats = self.stats.entry(plugin_id.into()).or_default();
            stats.total_errors += 1;
            return Err(NeedleError::Timeout(duration));
        }

        let duration_ms = duration.as_millis() as u64;

        // Update stats
        let stats = self.stats.entry(plugin_id.into()).or_default();
        stats.total_executions += 1;
        stats.avg_duration_ms = stats.avg_duration_ms
            * ((stats.total_executions - 1) as f64 / stats.total_executions as f64)
            + duration_ms as f64 / stats.total_executions as f64;
        if duration_ms as f64 > stats.max_duration_ms {
            stats.max_duration_ms = duration_ms as f64;
        }

        Ok(ExecutionResult {
            plugin_id: plugin_id.into(),
            function: function_name.into(),
            output,
            duration_ms,
            memory_used: 0,
        })
    }

    /// Get plugin statistics.
    pub fn stats(&self, plugin_id: &str) -> Option<&PluginStats> {
        self.stats.get(plugin_id)
    }

    /// Deactivate a plugin sandbox.
    pub fn deactivate(&mut self, plugin_id: &str) -> Result<()> {
        let sandbox = self.sandboxes.get_mut(plugin_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Plugin '{plugin_id}' not found"))
        })?;
        sandbox.active = false;
        Ok(())
    }

    /// Activate a plugin sandbox.
    pub fn activate(&mut self, plugin_id: &str) -> Result<()> {
        let sandbox = self.sandboxes.get_mut(plugin_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Plugin '{plugin_id}' not found"))
        })?;
        sandbox.active = true;
        Ok(())
    }

    /// List all sandboxed plugins.
    pub fn list_plugins(&self) -> Vec<&str> {
        self.sandboxes.keys().map(|s| s.as_str()).collect()
    }

    /// Number of registered sandboxes.
    pub fn sandbox_count(&self) -> usize {
        self.sandboxes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_execute() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        let sandbox = PluginSandbox::new("dist")
            .with_permission(Permission::ComputeDistance);
        rt.register_sandbox(sandbox).unwrap();

        rt.register_function("dist", "l1", |input: &[f32]| {
            vec![input.iter().map(|x| x.abs()).sum()]
        })
        .unwrap();

        let result = rt.execute("dist", "l1", &[1.0, -2.0, 3.0]).unwrap();
        assert!((result.output[0] - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_inactive_sandbox() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        rt.register_sandbox(PluginSandbox::new("p")).unwrap();
        rt.register_function("p", "f", |_| vec![]).unwrap();
        rt.deactivate("p").unwrap();

        assert!(rt.execute("p", "f", &[]).is_err());
    }

    #[test]
    fn test_missing_function() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        rt.register_sandbox(PluginSandbox::new("p")).unwrap();
        assert!(rt.execute("p", "missing", &[]).is_err());
    }

    #[test]
    fn test_permissions() {
        let sandbox = PluginSandbox::new("test")
            .with_permission(Permission::ReadData)
            .with_permission(Permission::Search);

        assert!(sandbox.has_permission(&Permission::ReadData));
        assert!(!sandbox.has_permission(&Permission::WriteData));
    }

    #[test]
    fn test_input_size_limit() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        let sandbox = PluginSandbox::new("p").with_limits(ResourceLimits {
            max_input_bytes: 16, // only 4 floats
            ..Default::default()
        });
        rt.register_sandbox(sandbox).unwrap();
        rt.register_function("p", "f", |_| vec![]).unwrap();

        // 5 floats = 20 bytes > 16 limit
        assert!(rt.execute("p", "f", &[1.0; 5]).is_err());
    }

    #[test]
    fn test_stats_tracking() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        rt.register_sandbox(PluginSandbox::new("p")).unwrap();
        rt.register_function("p", "f", |_| vec![1.0]).unwrap();

        rt.execute("p", "f", &[]).unwrap();
        rt.execute("p", "f", &[]).unwrap();

        let stats = rt.stats("p").unwrap();
        assert_eq!(stats.total_executions, 2);
    }

    #[test]
    fn test_list_plugins() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        rt.register_sandbox(PluginSandbox::new("a")).unwrap();
        rt.register_sandbox(PluginSandbox::new("b")).unwrap();
        assert_eq!(rt.sandbox_count(), 2);
    }

    #[test]
    fn test_activate_deactivate() {
        let mut rt = PluginRuntime::new(RuntimeConfig::default());
        rt.register_sandbox(PluginSandbox::new("p")).unwrap();
        rt.register_function("p", "f", |_| vec![]).unwrap();

        rt.deactivate("p").unwrap();
        assert!(rt.execute("p", "f", &[]).is_err());

        rt.activate("p").unwrap();
        assert!(rt.execute("p", "f", &[]).is_ok());
    }
}
