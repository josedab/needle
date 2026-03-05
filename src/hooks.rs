//! Programmable Pre/Post-Processing Hooks
//!
//! Register Rust closures as hooks on insert (pre-processing) and search
//! (post-processing). Ships with 5 built-in hooks for common operations.
//!
//! # Hook Types
//!
//! - **Insert hooks**: Transform vector/metadata before indexing
//!   - Pre-insert: normalize, validate, augment metadata
//! - **Search hooks**: Transform results after retrieval
//!   - Post-search: rerank, filter, transform results
//!
//! # Built-in Hooks
//!
//! 1. `normalize_l2` — L2-normalize vectors on insert
//! 2. `validate_dimensions` — Reject vectors with wrong dimensions
//! 3. `add_timestamp` — Add insertion timestamp to metadata
//! 4. `dedup_results` — Remove duplicate IDs from search results
//! 5. `score_threshold` — Filter search results below a distance threshold
//!
//! # Example
//!
//! ```rust
//! use needle::hooks::*;
//!
//! let mut registry = HookRegistry::new();
//! registry.register_insert_hook("normalize", builtin::normalize_l2());
//! registry.register_search_hook("dedup", builtin::dedup_results());
//!
//! // Apply insert hooks
//! let mut ctx = InsertHookContext {
//!     id: "doc1".to_string(),
//!     vector: vec![3.0, 4.0],
//!     metadata: None,
//! };
//! registry.run_insert_hooks(&mut ctx);
//! // Vector is now L2-normalized
//! ```

use serde_json::Value;
use std::collections::HashMap;

// ============================================================================
// Hook Context Types
// ============================================================================

/// Context passed to insert hooks (mutable).
#[derive(Debug, Clone)]
pub struct InsertHookContext {
    /// Vector ID.
    pub id: String,
    /// Vector data (may be modified by hooks).
    pub vector: Vec<f32>,
    /// Metadata (may be modified by hooks).
    pub metadata: Option<Value>,
}

/// A single search result for post-processing hooks.
#[derive(Debug, Clone)]
pub struct SearchResultEntry {
    /// Vector ID.
    pub id: String,
    /// Distance from query.
    pub distance: f32,
    /// Metadata.
    pub metadata: Option<Value>,
}

/// Context passed to search hooks (mutable).
#[derive(Debug, Clone)]
pub struct SearchHookContext {
    /// Query vector used for the search.
    pub query: Vec<f32>,
    /// Search results (may be modified by hooks).
    pub results: Vec<SearchResultEntry>,
    /// Number of results originally requested.
    pub k: usize,
}

// ============================================================================
// Hook Traits
// ============================================================================

/// A hook that runs before vector insertion.
pub trait InsertHook: Send + Sync {
    /// Hook name for identification.
    fn name(&self) -> &str;
    /// Process the insert context. Return Ok(()) to continue, Err to abort.
    fn process(&self, ctx: &mut InsertHookContext) -> std::result::Result<(), String>;
}

/// A hook that runs after search.
pub trait SearchHook: Send + Sync {
    /// Hook name for identification.
    fn name(&self) -> &str;
    /// Process search results. Can filter, reorder, or transform results.
    fn process(&self, ctx: &mut SearchHookContext) -> std::result::Result<(), String>;
}

// ============================================================================
// Closure-based Hook Implementations
// ============================================================================

/// Insert hook backed by a closure.
struct ClosureInsertHook {
    name: String,
    func: Box<dyn Fn(&mut InsertHookContext) -> std::result::Result<(), String> + Send + Sync>,
}

impl InsertHook for ClosureInsertHook {
    fn name(&self) -> &str {
        &self.name
    }
    fn process(&self, ctx: &mut InsertHookContext) -> std::result::Result<(), String> {
        (self.func)(ctx)
    }
}

/// Search hook backed by a closure.
struct ClosureSearchHook {
    name: String,
    func: Box<dyn Fn(&mut SearchHookContext) -> std::result::Result<(), String> + Send + Sync>,
}

impl SearchHook for ClosureSearchHook {
    fn name(&self) -> &str {
        &self.name
    }
    fn process(&self, ctx: &mut SearchHookContext) -> std::result::Result<(), String> {
        (self.func)(ctx)
    }
}

// ============================================================================
// Hook Registry
// ============================================================================

/// Registry for managing pre-insert and post-search hooks.
pub struct HookRegistry {
    insert_hooks: Vec<Box<dyn InsertHook>>,
    search_hooks: Vec<Box<dyn SearchHook>>,
    /// Hook execution stats: name -> invocation count.
    stats: HashMap<String, u64>,
}

impl HookRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            insert_hooks: Vec::new(),
            search_hooks: Vec::new(),
            stats: HashMap::new(),
        }
    }

    /// Register an insert hook.
    pub fn register_insert_hook(
        &mut self,
        name: impl Into<String>,
        func: impl Fn(&mut InsertHookContext) -> std::result::Result<(), String> + Send + Sync + 'static,
    ) {
        let name = name.into();
        self.insert_hooks.push(Box::new(ClosureInsertHook {
            name,
            func: Box::new(func),
        }));
    }

    /// Register a search hook.
    pub fn register_search_hook(
        &mut self,
        name: impl Into<String>,
        func: impl Fn(&mut SearchHookContext) -> std::result::Result<(), String> + Send + Sync + 'static,
    ) {
        let name = name.into();
        self.search_hooks.push(Box::new(ClosureSearchHook {
            name,
            func: Box::new(func),
        }));
    }

    /// Register a trait-object insert hook.
    pub fn register_insert_hook_trait(&mut self, hook: Box<dyn InsertHook>) {
        self.insert_hooks.push(hook);
    }

    /// Register a trait-object search hook.
    pub fn register_search_hook_trait(&mut self, hook: Box<dyn SearchHook>) {
        self.search_hooks.push(hook);
    }

    /// Run all insert hooks in order. Returns first error if any.
    pub fn run_insert_hooks(
        &mut self,
        ctx: &mut InsertHookContext,
    ) -> std::result::Result<(), String> {
        for hook in &self.insert_hooks {
            *self.stats.entry(hook.name().to_string()).or_insert(0) += 1;
            hook.process(ctx)?;
        }
        Ok(())
    }

    /// Run all search hooks in order. Returns first error if any.
    pub fn run_search_hooks(
        &mut self,
        ctx: &mut SearchHookContext,
    ) -> std::result::Result<(), String> {
        for hook in &self.search_hooks {
            *self.stats.entry(hook.name().to_string()).or_insert(0) += 1;
            hook.process(ctx)?;
        }
        Ok(())
    }

    /// List registered hook names.
    pub fn list_hooks(&self) -> (Vec<&str>, Vec<&str>) {
        let insert: Vec<&str> = self.insert_hooks.iter().map(|h| h.name()).collect();
        let search: Vec<&str> = self.search_hooks.iter().map(|h| h.name()).collect();
        (insert, search)
    }

    /// Get hook execution statistics.
    pub fn stats(&self) -> &HashMap<String, u64> {
        &self.stats
    }

    /// Remove an insert hook by name.
    pub fn remove_insert_hook(&mut self, name: &str) -> bool {
        let len = self.insert_hooks.len();
        self.insert_hooks.retain(|h| h.name() != name);
        self.insert_hooks.len() < len
    }

    /// Remove a search hook by name.
    pub fn remove_search_hook(&mut self, name: &str) -> bool {
        let len = self.search_hooks.len();
        self.search_hooks.retain(|h| h.name() != name);
        self.search_hooks.len() < len
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Built-in Hooks
// ============================================================================

/// Built-in hook implementations.
pub mod builtin {
    use super::*;

    /// L2-normalize vectors on insert.
    pub fn normalize_l2() -> impl Fn(&mut InsertHookContext) -> std::result::Result<(), String> + Send + Sync {
        |ctx| {
            let norm: f32 = ctx.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut ctx.vector {
                    *x /= norm;
                }
            }
            Ok(())
        }
    }

    /// Validate vector dimensions.
    pub fn validate_dimensions(expected: usize) -> impl Fn(&mut InsertHookContext) -> std::result::Result<(), String> + Send + Sync {
        move |ctx| {
            if ctx.vector.len() != expected {
                Err(format!(
                    "Dimension mismatch: expected {}, got {}",
                    expected,
                    ctx.vector.len()
                ))
            } else {
                Ok(())
            }
        }
    }

    /// Add insertion timestamp to metadata.
    pub fn add_timestamp() -> impl Fn(&mut InsertHookContext) -> std::result::Result<(), String> + Send + Sync {
        |ctx| {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            let meta = ctx
                .metadata
                .get_or_insert_with(|| serde_json::json!({}));
            if let Some(obj) = meta.as_object_mut() {
                obj.insert("_inserted_at".to_string(), Value::from(ts));
            }
            Ok(())
        }
    }

    /// Remove duplicate IDs from search results (keeps first occurrence).
    pub fn dedup_results() -> impl Fn(&mut SearchHookContext) -> std::result::Result<(), String> + Send + Sync {
        |ctx| {
            let mut seen = std::collections::HashSet::new();
            ctx.results.retain(|r| seen.insert(r.id.clone()));
            Ok(())
        }
    }

    /// Filter search results by maximum distance threshold.
    pub fn score_threshold(max_distance: f32) -> impl Fn(&mut SearchHookContext) -> std::result::Result<(), String> + Send + Sync {
        move |ctx| {
            ctx.results.retain(|r| r.distance <= max_distance);
            Ok(())
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_l2_hook() {
        let mut registry = HookRegistry::new();
        registry.register_insert_hook("normalize", builtin::normalize_l2());

        let mut ctx = InsertHookContext {
            id: "d1".into(),
            vector: vec![3.0, 4.0],
            metadata: None,
        };
        registry.run_insert_hooks(&mut ctx).expect("hooks");

        let norm: f32 = ctx.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_validate_dimensions_hook() {
        let mut registry = HookRegistry::new();
        registry.register_insert_hook("validate", builtin::validate_dimensions(3));

        let mut ctx = InsertHookContext {
            id: "d1".into(),
            vector: vec![1.0, 2.0, 3.0],
            metadata: None,
        };
        assert!(registry.run_insert_hooks(&mut ctx).is_ok());

        let mut bad_ctx = InsertHookContext {
            id: "d2".into(),
            vector: vec![1.0, 2.0],
            metadata: None,
        };
        assert!(registry.run_insert_hooks(&mut bad_ctx).is_err());
    }

    #[test]
    fn test_add_timestamp_hook() {
        let mut registry = HookRegistry::new();
        registry.register_insert_hook("timestamp", builtin::add_timestamp());

        let mut ctx = InsertHookContext {
            id: "d1".into(),
            vector: vec![1.0],
            metadata: None,
        };
        registry.run_insert_hooks(&mut ctx).expect("hooks");

        let meta = ctx.metadata.as_ref().expect("should have metadata");
        assert!(meta.get("_inserted_at").is_some());
    }

    #[test]
    fn test_dedup_results_hook() {
        let mut registry = HookRegistry::new();
        registry.register_search_hook("dedup", builtin::dedup_results());

        let mut ctx = SearchHookContext {
            query: vec![1.0],
            results: vec![
                SearchResultEntry { id: "a".into(), distance: 0.1, metadata: None },
                SearchResultEntry { id: "b".into(), distance: 0.2, metadata: None },
                SearchResultEntry { id: "a".into(), distance: 0.3, metadata: None },
            ],
            k: 3,
        };
        registry.run_search_hooks(&mut ctx).expect("hooks");
        assert_eq!(ctx.results.len(), 2);
    }

    #[test]
    fn test_score_threshold_hook() {
        let mut registry = HookRegistry::new();
        registry.register_search_hook("threshold", builtin::score_threshold(0.5));

        let mut ctx = SearchHookContext {
            query: vec![1.0],
            results: vec![
                SearchResultEntry { id: "a".into(), distance: 0.1, metadata: None },
                SearchResultEntry { id: "b".into(), distance: 0.8, metadata: None },
                SearchResultEntry { id: "c".into(), distance: 0.3, metadata: None },
            ],
            k: 3,
        };
        registry.run_search_hooks(&mut ctx).expect("hooks");
        assert_eq!(ctx.results.len(), 2); // b filtered out
    }

    #[test]
    fn test_hook_chaining() {
        let mut registry = HookRegistry::new();
        registry.register_insert_hook("normalize", builtin::normalize_l2());
        registry.register_insert_hook("timestamp", builtin::add_timestamp());

        let mut ctx = InsertHookContext {
            id: "d1".into(),
            vector: vec![3.0, 4.0],
            metadata: None,
        };
        registry.run_insert_hooks(&mut ctx).expect("hooks");

        // Vector should be normalized
        let norm: f32 = ctx.vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        // Metadata should have timestamp
        assert!(ctx.metadata.is_some());
    }

    #[test]
    fn test_list_and_remove_hooks() {
        let mut registry = HookRegistry::new();
        registry.register_insert_hook("h1", |_ctx| Ok(()));
        registry.register_search_hook("h2", |_ctx| Ok(()));

        let (insert, search) = registry.list_hooks();
        assert_eq!(insert, vec!["h1"]);
        assert_eq!(search, vec!["h2"]);

        assert!(registry.remove_insert_hook("h1"));
        assert!(!registry.remove_insert_hook("h1"));
    }

    #[test]
    fn test_hook_stats() {
        let mut registry = HookRegistry::new();
        registry.register_insert_hook("counter", |_ctx| Ok(()));

        for _ in 0..5 {
            let mut ctx = InsertHookContext {
                id: "d".into(),
                vector: vec![1.0],
                metadata: None,
            };
            registry.run_insert_hooks(&mut ctx).expect("hooks");
        }

        assert_eq!(registry.stats().get("counter"), Some(&5));
    }
}
