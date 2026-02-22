//! Web UI Dashboard
//!
//! Browser-based interface for monitoring and managing Needle vector database.
//!
//! This module provides a server-side rendered web dashboard that allows users to:
//! - View database statistics and health status
//! - Browse and inspect collections
//! - Execute vector queries through an interactive playground
//! - Monitor metrics and performance
//!
//! # Features
//!
//! - **Dashboard**: Overview of database health, statistics, and recent activity
//! - **Collections Browser**: List, inspect, and manage vector collections
//! - **Query Playground**: Interactive interface for testing vector searches
//! - **Metrics Dashboard**: Visualization of performance metrics
//!
//! # Usage
//!
//! ```rust,ignore
//! use needle::web_ui::{WebUiConfig, serve_web_ui};
//! use needle::Database;
//! use std::sync::Arc;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let db = Database::open("vectors.needle")?;
//!     let config = WebUiConfig::default();
//!     serve_web_ui(Arc::new(db), config).await?;
//!     Ok(())
//! }
//! ```

mod state;
mod templates;
mod routes;

// Re-export all public types from state
pub use state::{
    ActionParam, AdminAction, AdminSection, Alert, AlertConfig, AlertSeverity,
    CollectionHealthScore, CollectionStatsResponse, FilterCondition, LatencyBucket, LatencyHeatmap,
    MonitoringSnapshot, SearchQuery, StatsResponse, VisualQueryState, WebUiConfig, WebUiState,
};

// Re-export all public functions from templates
pub use templates::{
    generate_admin_dashboard_html, generate_monitoring_dashboard_html,
    generate_query_builder_html, get_admin_sections,
};

// Re-export all public functions from routes
pub use routes::{
    check_alerts, compute_monitoring_snapshot, create_web_ui_router, serve_web_ui,
    serve_web_ui_default,
};
