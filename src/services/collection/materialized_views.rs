//! Materialized Search Views
//!
//! Pre-computed search results for frequent query patterns. Like SQL
//! materialized views but for vector queries — auto-refreshes when underlying
//! data changes beyond a configurable drift threshold.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::materialized_views::{
//!     ViewManager, MaterializedView, ViewConfig,
//! };
//!
//! let mut mgr = ViewManager::new();
//!
//! // Define a materialized view
//! mgr.create_view(MaterializedView::new(
//!     "trending",
//!     vec![0.5f32; 4],
//!     10,
//! ).with_staleness_threshold(0.1));
//!
//! // Populate from search results
//! mgr.refresh("trending", &[("d1".into(), 0.1), ("d2".into(), 0.2)]).unwrap();
//!
//! // Serve from cache (sub-microsecond)
//! let results = mgr.query("trending").unwrap();
//! assert_eq!(results.len(), 2);
//! ```

use std::collections::HashMap;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── View Configuration ───────────────────────────────────────────────────────

/// Configuration for a materialized view.
#[derive(Debug, Clone)]
pub struct ViewConfig {
    /// Maximum staleness before auto-refresh (in seconds).
    pub max_staleness_secs: u64,
    /// Drift threshold (cosine distance) to trigger refresh.
    pub drift_threshold: f32,
    /// Maximum cached results.
    pub max_results: usize,
}

impl Default for ViewConfig {
    fn default() -> Self {
        Self {
            max_staleness_secs: 300,
            drift_threshold: 0.1,
            max_results: 100,
        }
    }
}

// ── Materialized View ────────────────────────────────────────────────────────

/// A materialized search view definition.
#[derive(Debug, Clone)]
pub struct MaterializedView {
    /// View name.
    pub name: String,
    /// Source collection name.
    pub source_collection: Option<String>,
    /// Query embedding that defines this view.
    pub query: Vec<f32>,
    /// Number of results to cache.
    pub k: usize,
    /// Staleness threshold for drift-based refresh.
    pub staleness_threshold: f32,
    /// Metadata filter in MongoDB-style JSON (applied during refresh).
    pub metadata_filter: Option<serde_json::Value>,
    /// NeedleQL view definition (the CREATE VIEW statement).
    pub definition: Option<String>,
    /// Cached results (id, distance).
    results: Vec<(String, f32)>,
    /// Last refresh time.
    last_refresh: Option<u64>,
    /// Number of times served from cache.
    hit_count: u64,
    /// Whether the view needs refresh.
    stale: bool,
    /// Number of mutations since last refresh.
    mutations_since_refresh: u64,
    /// Mutation threshold to trigger auto-refresh.
    pub auto_refresh_threshold: u64,
}

impl MaterializedView {
    /// Create a new view.
    pub fn new(name: impl Into<String>, query: Vec<f32>, k: usize) -> Self {
        Self {
            name: name.into(),
            source_collection: None,
            query,
            k,
            staleness_threshold: 0.1,
            metadata_filter: None,
            definition: None,
            results: Vec::new(),
            last_refresh: None,
            hit_count: 0,
            stale: true,
            mutations_since_refresh: 0,
            auto_refresh_threshold: 100,
        }
    }

    /// Set staleness threshold.
    #[must_use]
    pub fn with_staleness_threshold(mut self, threshold: f32) -> Self {
        self.staleness_threshold = threshold;
        self
    }

    /// Set source collection name.
    #[must_use]
    pub fn with_source_collection(mut self, collection: impl Into<String>) -> Self {
        self.source_collection = Some(collection.into());
        self
    }

    /// Set metadata filter for the view.
    #[must_use]
    pub fn with_metadata_filter(mut self, filter: serde_json::Value) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Set the NeedleQL view definition string.
    #[must_use]
    pub fn with_definition(mut self, def: impl Into<String>) -> Self {
        self.definition = Some(def.into());
        self
    }

    /// Set the mutation count threshold for auto-refresh.
    #[must_use]
    pub fn with_auto_refresh_threshold(mut self, threshold: u64) -> Self {
        self.auto_refresh_threshold = threshold;
        self
    }

    /// Record a mutation on the source collection.
    pub fn record_mutation(&mut self) {
        self.mutations_since_refresh += 1;
        if self.mutations_since_refresh >= self.auto_refresh_threshold {
            self.stale = true;
        }
    }

    /// Check if auto-refresh should be triggered.
    pub fn needs_refresh(&self) -> bool {
        if self.stale {
            return true;
        }

        // Check time-based staleness
        if let Some(last) = self.last_refresh {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            return now.saturating_sub(last) > 300; // 5 minute default staleness
        }

        true
    }
}

// ── View Statistics ──────────────────────────────────────────────────────────

/// Statistics for a materialized view.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewStats {
    /// View name.
    pub name: String,
    /// Number of cached results.
    pub cached_results: usize,
    /// Cache hit count.
    pub hit_count: u64,
    /// Whether the view is stale.
    pub is_stale: bool,
    /// Last refresh timestamp.
    pub last_refresh: Option<u64>,
}

// ── View Manager ─────────────────────────────────────────────────────────────

/// Manages materialized search views.
pub struct ViewManager {
    views: HashMap<String, MaterializedView>,
    config: ViewConfig,
}

impl ViewManager {
    /// Create a new view manager.
    pub fn new() -> Self {
        Self {
            views: HashMap::new(),
            config: ViewConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ViewConfig) -> Self {
        Self {
            views: HashMap::new(),
            config,
        }
    }

    /// Create a materialized view.
    pub fn create_view(&mut self, view: MaterializedView) -> Result<()> {
        if self.views.contains_key(&view.name) {
            return Err(NeedleError::Conflict(format!("View '{}' already exists", view.name)));
        }
        self.views.insert(view.name.clone(), view);
        Ok(())
    }

    /// Drop a view.
    pub fn drop_view(&mut self, name: &str) -> bool {
        self.views.remove(name).is_some()
    }

    /// Refresh a view with new results.
    pub fn refresh(&mut self, name: &str, results: &[(String, f32)]) -> Result<()> {
        let view = self.views.get_mut(name)
            .ok_or_else(|| NeedleError::NotFound(format!("View '{name}' not found")))?;

        view.results = results.iter().take(view.k).cloned().collect();
        view.last_refresh = Some(now_secs());
        view.stale = false;
        Ok(())
    }

    /// Query a materialized view (returns cached results).
    pub fn query(&mut self, name: &str) -> Result<Vec<(String, f32)>> {
        let view = self.views.get_mut(name)
            .ok_or_else(|| NeedleError::NotFound(format!("View '{name}' not found")))?;

        if view.stale || view.results.is_empty() {
            return Err(NeedleError::InvalidState(format!("View '{name}' needs refresh")));
        }

        view.hit_count += 1;
        Ok(view.results.clone())
    }

    /// Mark a view as stale (e.g., after data changes).
    pub fn invalidate(&mut self, name: &str) {
        if let Some(view) = self.views.get_mut(name) {
            view.stale = true;
        }
    }

    /// Invalidate all views.
    pub fn invalidate_all(&mut self) {
        for view in self.views.values_mut() {
            view.stale = true;
        }
    }

    /// Check if a query matches a materialized view.
    pub fn find_matching_view(&self, query: &[f32], k: usize) -> Option<&str> {
        for view in self.views.values() {
            if !view.stale && view.k >= k && query.len() == view.query.len() {
                let dist = cosine_distance(query, &view.query);
                if dist < view.staleness_threshold {
                    return Some(&view.name);
                }
            }
        }
        None
    }

    /// Get statistics for all views.
    pub fn stats(&self) -> Vec<ViewStats> {
        self.views.values().map(|v| ViewStats {
            name: v.name.clone(),
            cached_results: v.results.len(),
            hit_count: v.hit_count,
            is_stale: v.stale,
            last_refresh: v.last_refresh,
        }).collect()
    }

    /// View count.
    pub fn len(&self) -> usize {
        self.views.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.views.is_empty()
    }

    /// Total cache hit count across all views.
    pub fn total_hits(&self) -> u64 {
        self.views.values().map(|v| v.hit_count).sum()
    }

    /// Parse and create a view from a NeedleQL CREATE VIEW statement.
    /// Format: `CREATE VIEW <name> AS SELECT ... FROM <collection> NEAREST_TO([...]) LIMIT <k>`
    pub fn create_view_from_ql(&mut self, statement: &str) -> Result<()> {
        let upper = statement.to_uppercase();
        if !upper.starts_with("CREATE VIEW ") && !upper.starts_with("CREATE MATERIALIZED VIEW ") {
            return Err(NeedleError::InvalidArgument(
                "Expected CREATE VIEW or CREATE MATERIALIZED VIEW".into(),
            ));
        }

        // Extract view name
        let after_view = if upper.starts_with("CREATE MATERIALIZED VIEW ") {
            &statement[25..]
        } else {
            &statement[12..]
        };

        let as_pos = after_view.to_uppercase().find(" AS ").ok_or_else(|| {
            NeedleError::InvalidArgument("Missing AS in CREATE VIEW".into())
        })?;
        let view_name = after_view[..as_pos].trim().to_string();

        if view_name.is_empty() {
            return Err(NeedleError::InvalidArgument("Missing view name".into()));
        }

        // Extract collection from FROM clause
        let rest = &after_view[as_pos + 4..];
        let from_pos = rest.to_uppercase().find("FROM ").unwrap_or(0);
        let after_from = &rest[from_pos + 5..];
        let collection_end = after_from
            .find(|c: char| c.is_whitespace())
            .unwrap_or(after_from.len());
        let collection = after_from[..collection_end].trim().to_string();

        // Extract LIMIT
        let limit = if let Some(pos) = rest.to_uppercase().find("LIMIT ") {
            let after_limit = &rest[pos + 6..];
            let end = after_limit
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(after_limit.len());
            after_limit[..end].parse().unwrap_or(10)
        } else {
            10
        };

        let view = MaterializedView::new(view_name, Vec::new(), limit)
            .with_source_collection(collection)
            .with_definition(statement);

        self.create_view(view)
    }

    /// Record a mutation on a source collection (for change tracking).
    /// Marks matching views as potentially stale.
    pub fn record_collection_mutation(&mut self, collection: &str) {
        for view in self.views.values_mut() {
            if view.source_collection.as_deref() == Some(collection) {
                view.record_mutation();
            }
        }
    }

    /// Get views that need refresh.
    pub fn views_needing_refresh(&self) -> Vec<&str> {
        self.views
            .values()
            .filter(|v| v.needs_refresh())
            .map(|v| v.name.as_str())
            .collect()
    }

    /// List views for a specific source collection.
    pub fn views_for_collection(&self, collection: &str) -> Vec<ViewStats> {
        self.views
            .values()
            .filter(|v| v.source_collection.as_deref() == Some(collection))
            .map(|v| ViewStats {
                name: v.name.clone(),
                cached_results: v.results.len(),
                hit_count: v.hit_count,
                is_stale: v.stale,
                last_refresh: v.last_refresh,
            })
            .collect()
    }
}

impl Default for ViewManager {
    fn default() -> Self { Self::new() }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON { return 1.0; }
    1.0 - (dot / (na * nb))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_query() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        mgr.refresh("v1", &[("d1".into(), 0.1), ("d2".into(), 0.2)]).unwrap();
        let results = mgr.query("v1").unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_stale_view() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        assert!(mgr.query("v1").is_err()); // not refreshed
    }

    #[test]
    fn test_invalidate() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        mgr.refresh("v1", &[("d1".into(), 0.1)]).unwrap();
        mgr.invalidate("v1");
        assert!(mgr.query("v1").is_err());
    }

    #[test]
    fn test_find_matching() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0, 0.0, 0.0, 0.0], 10).with_staleness_threshold(0.2)).unwrap();
        mgr.refresh("v1", &[("d1".into(), 0.1)]).unwrap();

        assert!(mgr.find_matching_view(&[1.0, 0.0, 0.0, 0.0], 5).is_some());
        assert!(mgr.find_matching_view(&[0.0, 1.0, 0.0, 0.0], 5).is_none());
    }

    #[test]
    fn test_hit_counting() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        mgr.refresh("v1", &[("d1".into(), 0.1)]).unwrap();
        mgr.query("v1").unwrap();
        mgr.query("v1").unwrap();
        assert_eq!(mgr.total_hits(), 2);
    }

    #[test]
    fn test_drop_view() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        assert!(mgr.drop_view("v1"));
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_duplicate_view() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        assert!(mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).is_err());
    }

    #[test]
    fn test_create_view_from_ql() {
        let mut mgr = ViewManager::new();
        mgr.create_view_from_ql(
            "CREATE VIEW trending AS SELECT * FROM docs NEAREST_TO([0.1, 0.2]) LIMIT 20"
        ).unwrap();
        assert_eq!(mgr.len(), 1);
        let stats = mgr.stats();
        assert_eq!(stats[0].name, "trending");
    }

    #[test]
    fn test_create_materialized_view_from_ql() {
        let mut mgr = ViewManager::new();
        mgr.create_view_from_ql(
            "CREATE MATERIALIZED VIEW hot_topics AS SELECT * FROM articles LIMIT 50"
        ).unwrap();
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_mutation_tracking() {
        let mut mgr = ViewManager::new();
        let view = MaterializedView::new("v1", vec![1.0; 4], 10)
            .with_source_collection("docs")
            .with_auto_refresh_threshold(3);
        mgr.create_view(view).unwrap();
        mgr.refresh("v1", &[("d1".into(), 0.1)]).unwrap();

        // Not stale after refresh
        assert!(mgr.query("v1").is_ok());

        // Record mutations
        mgr.record_collection_mutation("docs");
        mgr.record_collection_mutation("docs");
        assert!(mgr.query("v1").is_ok()); // Still fresh

        mgr.record_collection_mutation("docs"); // 3rd mutation triggers staleness
        assert!(mgr.query("v1").is_err()); // Now stale
    }

    #[test]
    fn test_views_needing_refresh() {
        let mut mgr = ViewManager::new();
        mgr.create_view(MaterializedView::new("v1", vec![1.0; 4], 10)).unwrap();
        mgr.create_view(MaterializedView::new("v2", vec![0.0; 4], 10)).unwrap();

        // Both new views need refresh
        assert_eq!(mgr.views_needing_refresh().len(), 2);

        mgr.refresh("v1", &[("d1".into(), 0.1)]).unwrap();
        assert_eq!(mgr.views_needing_refresh().len(), 1);
    }

    #[test]
    fn test_views_for_collection() {
        let mut mgr = ViewManager::new();
        mgr.create_view(
            MaterializedView::new("v1", vec![1.0; 4], 10).with_source_collection("docs")
        ).unwrap();
        mgr.create_view(
            MaterializedView::new("v2", vec![1.0; 4], 10).with_source_collection("other")
        ).unwrap();

        let docs_views = mgr.views_for_collection("docs");
        assert_eq!(docs_views.len(), 1);
        assert_eq!(docs_views[0].name, "v1");
    }

    #[test]
    fn test_metadata_filter_view() {
        let mut mgr = ViewManager::new();
        let view = MaterializedView::new("filtered", vec![1.0; 4], 10)
            .with_metadata_filter(serde_json::json!({"category": "science"}))
            .with_source_collection("docs");
        mgr.create_view(view).unwrap();

        let stats = mgr.stats();
        assert_eq!(stats.len(), 1);
    }
}
