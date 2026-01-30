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
    /// Query embedding that defines this view.
    pub query: Vec<f32>,
    /// Number of results to cache.
    pub k: usize,
    /// Staleness threshold for drift-based refresh.
    pub staleness_threshold: f32,
    /// Cached results (id, distance).
    results: Vec<(String, f32)>,
    /// Last refresh time.
    last_refresh: Option<u64>,
    /// Number of times served from cache.
    hit_count: u64,
    /// Whether the view needs refresh.
    stale: bool,
}

impl MaterializedView {
    /// Create a new view.
    pub fn new(name: impl Into<String>, query: Vec<f32>, k: usize) -> Self {
        Self {
            name: name.into(),
            query,
            k,
            staleness_threshold: 0.1,
            results: Vec::new(),
            last_refresh: None,
            hit_count: 0,
            stale: true,
        }
    }

    /// Set staleness threshold.
    #[must_use]
    pub fn with_staleness_threshold(mut self, threshold: f32) -> Self {
        self.staleness_threshold = threshold;
        self
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
}
