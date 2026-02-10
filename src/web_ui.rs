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

use crate::database::Database;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the Web UI server
#[derive(Debug, Clone)]
pub struct WebUiConfig {
    /// Address to bind the server to
    pub addr: SocketAddr,
    /// Application title shown in the browser
    pub title: String,
    /// Whether to enable the query playground
    pub enable_query_playground: bool,
    /// Refresh interval for auto-updating dashboards (in seconds)
    pub refresh_interval: u64,
}

impl Default for WebUiConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8081".parse().expect("valid socket addr"),
            title: "Needle Dashboard".to_string(),
            enable_query_playground: true,
            refresh_interval: 30,
        }
    }
}

impl WebUiConfig {
    /// Create a new configuration with the specified address
    pub fn new(addr: &str) -> Self {
        Self {
            addr: addr.parse().expect("Invalid address"),
            ..Default::default()
        }
    }

    /// Set a custom title for the dashboard
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Enable or disable the query playground
    pub fn with_query_playground(mut self, enabled: bool) -> Self {
        self.enable_query_playground = enabled;
        self
    }
}

// ============================================================================
// Application State
// ============================================================================

/// Shared state for the Web UI application
pub struct WebUiState {
    /// Reference to the Needle database
    pub db: RwLock<Database>,
    /// Configuration
    pub config: WebUiConfig,
    /// Server start time for uptime calculation
    pub start_time: u64,
}

impl WebUiState {
    /// Create a new Web UI state
    pub fn new(db: Database, config: WebUiConfig) -> Self {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            db: RwLock::new(db),
            config,
            start_time,
        }
    }

    /// Get the server uptime in seconds
    pub fn uptime(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now - self.start_time
    }
}

// ============================================================================
// CSS Styles
// ============================================================================

/// Inline CSS styles for the dashboard
const CSS_STYLES: &str = r#"
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #64748b;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: #475569;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Navigation */
nav {
    background-color: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

nav .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.nav-links {
    display: flex;
    gap: 1.5rem;
    list-style: none;
}

.nav-links a {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    transition: all 0.2s;
}

.nav-links a:hover,
.nav-links a.active {
    color: var(--text-primary);
    background-color: var(--bg-card);
}

/* Cards */
.card {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

.card-title {
    font-size: 1.125rem;
    font-weight: 600;
}

/* Grid layouts */
.grid {
    display: grid;
    gap: 1rem;
}

.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

@media (max-width: 768px) {
    .grid-2, .grid-3, .grid-4 {
        grid-template-columns: 1fr;
    }
}

/* Stats */
.stat {
    text-align: center;
    padding: 1rem;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

.stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

th {
    background-color: var(--bg-card);
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

tr:hover {
    background-color: var(--bg-card);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-success { background-color: var(--success); color: white; }
.badge-warning { background-color: var(--warning); color: black; }
.badge-danger { background-color: var(--danger); color: white; }
.badge-primary { background-color: var(--primary); color: white; }

/* Buttons */
.btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
}

.btn-primary {
    background-color: var(--primary);
    color: white;
}

.btn-primary:hover {
    background-color: var(--primary-dark);
}

.btn-secondary {
    background-color: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background-color: var(--border-color);
}

/* Forms */
.form-group {
    margin-bottom: 1rem;
}

.form-label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.form-input, .form-textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    background-color: var(--bg-card);
    color: var(--text-primary);
    font-size: 1rem;
}

.form-input:focus, .form-textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
}

.form-textarea {
    min-height: 120px;
    font-family: monospace;
    resize: vertical;
}

/* Health indicator */
.health-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.health-dot {
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.health-dot.healthy { background-color: var(--success); }
.health-dot.warning { background-color: var(--warning); }
.health-dot.error { background-color: var(--danger); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Main content */
main {
    padding: 2rem 0;
}

.page-header {
    margin-bottom: 2rem;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.page-description {
    color: var(--text-secondary);
}

/* Code blocks */
pre, code {
    font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
    background-color: var(--bg-card);
    border-radius: 0.375rem;
}

pre {
    padding: 1rem;
    overflow-x: auto;
}

code {
    padding: 0.125rem 0.375rem;
    font-size: 0.875em;
}

/* Results */
.results-list {
    list-style: none;
}

.result-item {
    padding: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    margin-bottom: 0.5rem;
    background-color: var(--bg-card);
}

.result-item:hover {
    border-color: var(--primary);
}

.result-id {
    font-weight: 600;
    color: var(--primary);
}

.result-distance {
    color: var(--text-secondary);
    font-size: 0.875rem;
}

/* Progress bar */
.progress-bar {
    height: 0.5rem;
    background-color: var(--bg-card);
    border-radius: 9999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--primary);
    transition: width 0.3s ease;
}

/* Footer */
footer {
    padding: 2rem 0;
    margin-top: 2rem;
    border-top: 1px solid var(--border-color);
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.875rem;
}
"#;

// ============================================================================
// Template Helpers
// ============================================================================

/// Generate the base HTML layout with navigation
fn base_layout(title: &str, content: &str, active_page: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Needle Dashboard</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <nav>
        <div class="container">
            <a href="/" class="logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                Needle
            </a>
            <ul class="nav-links">
                <li><a href="/" class="{dashboard_active}">Dashboard</a></li>
                <li><a href="/collections" class="{collections_active}">Collections</a></li>
                <li><a href="/query" class="{query_active}">Query</a></li>
                <li><a href="/monitoring" class="{monitoring_active}">Monitoring</a></li>
            </ul>
        </div>
    </nav>
    <main>
        <div class="container">
            {content}
        </div>
    </main>
    <footer>
        <div class="container">
            <p>Needle Vector Database &copy; 2024 - v{version}</p>
        </div>
    </footer>
</body>
</html>"#,
        title = title,
        content = content,
        version = env!("CARGO_PKG_VERSION"),
        dashboard_active = if active_page == "dashboard" { "active" } else { "" },
        collections_active = if active_page == "collections" { "active" } else { "" },
        query_active = if active_page == "query" { "active" } else { "" },
        monitoring_active = if active_page == "monitoring" { "active" } else { "" },
    )
}

/// Format a number with thousands separators
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i).is_multiple_of(3) {
            result.push(',');
        }
        result.push(*c);
    }
    result
}

/// Format bytes into human-readable size
fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format uptime into human-readable duration
fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Generate a stat card HTML component
fn stat_card(label: &str, value: &str, icon: &str) -> String {
    format!(
        r#"<div class="card stat">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>"#,
        icon = icon,
        value = value,
        label = label
    )
}

/// Generate a health badge based on status
#[allow(dead_code)]
fn health_badge(healthy: bool) -> String {
    if healthy {
        r#"<span class="badge badge-success">Healthy</span>"#.to_string()
    } else {
        r#"<span class="badge badge-danger">Unhealthy</span>"#.to_string()
    }
}

// ============================================================================
// Request/Response Types
// ============================================================================

/// Query parameters for search requests
#[derive(Debug, Deserialize)]
pub struct SearchQuery {
    /// Collection to search in
    pub collection: Option<String>,
    /// Query vector as comma-separated values
    pub vector: Option<String>,
    /// Number of results to return
    pub k: Option<usize>,
}

/// API statistics response
#[derive(Debug, Serialize)]
pub struct StatsResponse {
    /// Server health status
    pub healthy: bool,
    /// Server uptime in seconds
    pub uptime_seconds: u64,
    /// Total number of collections
    pub total_collections: usize,
    /// Total number of vectors across all collections
    pub total_vectors: usize,
    /// Per-collection statistics
    pub collections: Vec<CollectionStatsResponse>,
    /// Server version
    pub version: String,
}

/// Per-collection statistics
#[derive(Debug, Serialize)]
pub struct CollectionStatsResponse {
    /// Collection name
    pub name: String,
    /// Number of vectors
    pub vector_count: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Number of deleted vectors pending compaction
    pub deleted_count: usize,
    /// Whether compaction is needed
    pub needs_compaction: bool,
}

// ============================================================================
// Route Handlers
// ============================================================================

/// GET / - Main dashboard page
async fn dashboard_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();
    let total_vectors = db.total_vectors();
    let uptime = state.uptime();

    let collection_rows: String = collections
        .iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let deleted = coll.deleted_count();
            let needs_compact = coll.needs_compaction(0.2);

            Some(format!(
                r#"<tr>
                    <td><a href="/collections/{name}">{name}</a></td>
                    <td>{count}</td>
                    <td>{dims}</td>
                    <td>{deleted}</td>
                    <td>{status}</td>
                </tr>"#,
                name = name,
                count = format_number(count),
                dims = dims,
                deleted = deleted,
                status = if needs_compact {
                    r#"<span class="badge badge-warning">Needs Compaction</span>"#
                } else {
                    r#"<span class="badge badge-success">OK</span>"#
                }
            ))
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Dashboard</h1>
            <p class="page-description">Overview of your Needle vector database</p>
        </div>

        <div class="grid grid-4">
            {stat_health}
            {stat_collections}
            {stat_vectors}
            {stat_uptime}
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Collections Overview</h2>
                <a href="/collections" class="btn btn-secondary">View All</a>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Vectors</th>
                        <th>Dimensions</th>
                        <th>Deleted</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {collection_rows}
                </tbody>
            </table>
            {empty_message}
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Quick Actions</h2>
                </div>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                    <a href="/query" class="btn btn-primary">Query Playground</a>
                    <a href="/api/stats" class="btn btn-secondary">View API Stats</a>
                    <a href="/monitoring" class="btn btn-secondary">Metrics Dashboard</a>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">System Info</h2>
                </div>
                <p><strong>Version:</strong> {version}</p>
                <p><strong>Collections:</strong> {num_collections}</p>
                <p><strong>Total Vectors:</strong> {total_vectors}</p>
                <p><strong>Uptime:</strong> {uptime_formatted}</p>
            </div>
        </div>
        "#,
        stat_health = stat_card("Health", "Healthy", ""),
        stat_collections = stat_card("Collections", &format_number(collections.len()), ""),
        stat_vectors = stat_card("Total Vectors", &format_number(total_vectors), ""),
        stat_uptime = stat_card("Uptime", &format_uptime(uptime), ""),
        collection_rows = collection_rows,
        empty_message = if collections.is_empty() {
            r#"<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No collections yet. Create one using the API.</p>"#
        } else {
            ""
        },
        version = env!("CARGO_PKG_VERSION"),
        num_collections = collections.len(),
        total_vectors = format_number(total_vectors),
        uptime_formatted = format_uptime(uptime),
    );

    Html(base_layout("Dashboard", &content, "dashboard"))
}

/// GET /collections - List all collections
async fn collections_list_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let collection_cards: String = collections
        .iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let deleted = coll.deleted_count();

            Some(format!(
                r#"<div class="card">
                    <div class="card-header">
                        <h3 class="card-title">{name}</h3>
                        <a href="/collections/{name}" class="btn btn-primary">View Details</a>
                    </div>
                    <div class="grid grid-3">
                        <div class="stat">
                            <div class="stat-value">{count}</div>
                            <div class="stat-label">Vectors</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{dims}</div>
                            <div class="stat-label">Dimensions</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{deleted}</div>
                            <div class="stat-label">Deleted</div>
                        </div>
                    </div>
                </div>"#,
                name = name,
                count = format_number(count),
                dims = dims,
                deleted = deleted,
            ))
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Collections</h1>
            <p class="page-description">Browse and manage your vector collections</p>
        </div>

        {cards}

        {empty_message}
        "#,
        cards = collection_cards,
        empty_message = if collections.is_empty() {
            r#"<div class="card" style="text-align: center; padding: 3rem;">
                <h3>No Collections Found</h3>
                <p style="color: var(--text-secondary); margin: 1rem 0;">
                    Create your first collection using the REST API or CLI.
                </p>
                <pre style="text-align: left; display: inline-block;">
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_collection", "dimensions": 384}'
                </pre>
            </div>"#
        } else {
            ""
        },
    );

    Html(base_layout("Collections", &content, "collections"))
}

/// GET /collections/:name - Collection detail view
async fn collection_detail_handler(
    State(state): State<Arc<WebUiState>>,
    Path(name): Path<String>,
) -> Result<Html<String>, (StatusCode, Html<String>)> {
    let db = state.db.read().await;

    let coll = match db.collection(&name) {
        Ok(c) => c,
        Err(_) => {
            let error_content = format!(
                r#"
                <div class="card" style="text-align: center; padding: 3rem;">
                    <h2 style="color: var(--danger);">Collection Not Found</h2>
                    <p style="margin: 1rem 0;">The collection "{}" does not exist.</p>
                    <a href="/collections" class="btn btn-primary">Back to Collections</a>
                </div>
                "#,
                name
            );
            return Err((
                StatusCode::NOT_FOUND,
                Html(base_layout("Not Found", &error_content, "collections")),
            ));
        }
    };

    let dims = coll.dimensions().unwrap_or(0);
    let count = coll.len();
    let deleted = coll.deleted_count();
    let needs_compact = coll.needs_compaction(0.2);

    // Get sample vector IDs
    let sample_ids: Vec<String> = coll.ids().ok()
        .map(|ids| ids.into_iter().take(10).collect())
        .unwrap_or_default();

    let sample_rows: String = sample_ids
        .iter()
        .map(|id| {
            format!(
                r#"<tr>
                    <td><code>{id}</code></td>
                    <td>-</td>
                </tr>"#,
                id = id
            )
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="page-title">{name}</h1>
                    <p class="page-description">Collection details and statistics</p>
                </div>
                <a href="/collections" class="btn btn-secondary">Back to Collections</a>
            </div>
        </div>

        <div class="grid grid-4">
            {stat_vectors}
            {stat_dims}
            {stat_deleted}
            {stat_status}
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Collection Info</h2>
                </div>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Name</td><td><code>{name}</code></td></tr>
                    <tr><td>Dimensions</td><td>{dims}</td></tr>
                    <tr><td>Vector Count</td><td>{count}</td></tr>
                    <tr><td>Deleted Count</td><td>{deleted}</td></tr>
                    <tr><td>Needs Compaction</td><td>{compact_status}</td></tr>
                </table>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Quick Search</h2>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Use the Query Playground to search this collection.
                </p>
                <a href="/query?collection={name}" class="btn btn-primary">
                    Open Query Playground
                </a>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Sample Vectors</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Metadata</th>
                    </tr>
                </thead>
                <tbody>
                    {sample_rows}
                </tbody>
            </table>
            {empty_sample}
        </div>
        "#,
        name = name,
        stat_vectors = stat_card("Vectors", &format_number(count), ""),
        stat_dims = stat_card("Dimensions", &dims.to_string(), ""),
        stat_deleted = stat_card("Deleted", &deleted.to_string(), ""),
        stat_status = stat_card(
            "Status",
            if needs_compact { "Needs Compaction" } else { "OK" },
            ""
        ),
        dims = dims,
        count = format_number(count),
        deleted = deleted,
        compact_status = if needs_compact {
            r#"<span class="badge badge-warning">Yes</span>"#
        } else {
            r#"<span class="badge badge-success">No</span>"#
        },
        sample_rows = sample_rows,
        empty_sample = if sample_ids.is_empty() {
            r#"<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No vectors in this collection.</p>"#
        } else {
            ""
        },
    );

    Ok(Html(base_layout(&format!("Collection: {}", name), &content, "collections")))
}

/// GET /query - Query playground
async fn query_playground_handler(
    State(state): State<Arc<WebUiState>>,
    Query(params): Query<SearchQuery>,
) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let collection_options: String = collections
        .iter()
        .map(|name| {
            let selected = params.collection.as_ref() == Some(name);
            format!(
                r#"<option value="{name}" {selected}>{name}</option>"#,
                name = name,
                selected = if selected { "selected" } else { "" }
            )
        })
        .collect();

    let default_k = params.k.unwrap_or(10);
    let default_vector = params.vector.as_deref().unwrap_or("");

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Query Playground</h1>
            <p class="page-description">Test vector searches interactively</p>
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Search Parameters</h2>
                </div>
                <form id="search-form">
                    <div class="form-group">
                        <label class="form-label" for="collection">Collection</label>
                        <select id="collection" name="collection" class="form-input">
                            <option value="">Select a collection</option>
                            {collection_options}
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="vector">Query Vector (comma-separated)</label>
                        <textarea
                            id="vector"
                            name="vector"
                            class="form-textarea"
                            placeholder="0.1, 0.2, 0.3, ..."
                        >{default_vector}</textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="k">Number of Results (k)</label>
                        <input
                            type="number"
                            id="k"
                            name="k"
                            class="form-input"
                            value="{default_k}"
                            min="1"
                            max="100"
                        />
                    </div>
                    <button type="submit" class="btn btn-primary">
                        Search
                    </button>
                </form>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Results</h2>
                </div>
                <div id="results">
                    <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                        Enter a query vector and click Search to see results.
                    </p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">API Usage</h2>
            </div>
            <p style="margin-bottom: 1rem; color: var(--text-secondary);">
                Use this curl command to search via the API:
            </p>
            <pre>
curl -X POST http://localhost:8080/collections/YOUR_COLLECTION/search \
  -H "Content-Type: application/json" \
  -d '{{
    "vector": [0.1, 0.2, 0.3, ...],
    "k": 10
  }}'
            </pre>
        </div>

        <script>
        document.getElementById('search-form').addEventListener('submit', async (e) => {{
            e.preventDefault();

            const collection = document.getElementById('collection').value;
            const vectorStr = document.getElementById('vector').value;
            const k = parseInt(document.getElementById('k').value) || 10;

            if (!collection) {{
                alert('Please select a collection');
                return;
            }}

            if (!vectorStr.trim()) {{
                alert('Please enter a query vector');
                return;
            }}

            const vector = vectorStr.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

            if (vector.length === 0) {{
                alert('Invalid vector format. Use comma-separated numbers.');
                return;
            }}

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p style="text-align: center; padding: 2rem;">Searching...</p>';

            try {{
                const response = await fetch(`/collections/${{collection}}/search`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ vector, k }})
                }});

                const data = await response.json();

                if (response.ok && data.results) {{
                    if (data.results.length === 0) {{
                        resultsDiv.innerHTML = '<p style="text-align: center; padding: 2rem; color: var(--text-secondary);">No results found.</p>';
                    }} else {{
                        let html = '<ul class="results-list">';
                        data.results.forEach((r, i) => {{
                            html += `
                                <li class="result-item">
                                    <div class="result-id">#${{i + 1}} ${{r.id}}</div>
                                    <div class="result-distance">Distance: ${{r.distance.toFixed(6)}} | Score: ${{r.score.toFixed(4)}}</div>
                                </li>
                            `;
                        }});
                        html += '</ul>';
                        resultsDiv.innerHTML = html;
                    }}
                }} else {{
                    resultsDiv.innerHTML = `<p style="color: var(--danger); padding: 1rem;">Error: ${{data.error || 'Unknown error'}}</p>`;
                }}
            }} catch (err) {{
                resultsDiv.innerHTML = `<p style="color: var(--danger); padding: 1rem;">Error: ${{err.message}}</p>`;
            }}
        }});
        </script>
        "#,
        collection_options = collection_options,
        default_vector = default_vector,
        default_k = default_k,
    );

    Html(base_layout("Query Playground", &content, "query"))
}

/// GET /monitoring - Metrics dashboard
async fn monitoring_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();
    let total_vectors = db.total_vectors();
    let uptime = state.uptime();

    // Calculate memory estimates
    let mut total_memory: usize = 0;
    let mut collection_memory: Vec<(String, usize, usize)> = Vec::new();

    for name in &collections {
        if let Ok(coll) = db.collection(name) {
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let vector_memory = count * dims * std::mem::size_of::<f32>();
            total_memory += vector_memory;
            collection_memory.push((name.clone(), count, vector_memory));
        }
    }

    let memory_rows: String = collection_memory
        .iter()
        .map(|(name, count, mem)| {
            format!(
                r#"<tr>
                    <td>{name}</td>
                    <td>{count}</td>
                    <td>{memory}</td>
                    <td>
                        <div class="progress-bar" style="width: 100px;">
                            <div class="progress-fill" style="width: {pct}%;"></div>
                        </div>
                    </td>
                </tr>"#,
                name = name,
                count = format_number(*count),
                memory = format_bytes(*mem),
                pct = if total_memory > 0 {
                    (*mem as f64 / total_memory as f64 * 100.0).round() as usize
                } else {
                    0
                }
            )
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Monitoring</h1>
            <p class="page-description">System metrics and performance data</p>
        </div>

        <div class="grid grid-4">
            {stat_status}
            {stat_uptime}
            {stat_memory}
            {stat_vectors}
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Memory Usage by Collection</h2>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Collection</th>
                            <th>Vectors</th>
                            <th>Memory</th>
                            <th>Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {memory_rows}
                    </tbody>
                </table>
                {empty_memory}
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">System Health</h2>
                </div>
                <div style="padding: 1rem;">
                    <div class="health-indicator" style="margin-bottom: 1rem;">
                        <span class="health-dot healthy"></span>
                        <span>Database Status: <strong>Healthy</strong></span>
                    </div>
                    <div class="health-indicator" style="margin-bottom: 1rem;">
                        <span class="health-dot healthy"></span>
                        <span>API Server: <strong>Running</strong></span>
                    </div>
                    <div class="health-indicator">
                        <span class="health-dot healthy"></span>
                        <span>Web UI: <strong>Active</strong></span>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Performance Metrics</h2>
            </div>
            <p style="color: var(--text-secondary); text-align: center; padding: 3rem;">
                Performance metrics visualization coming soon.<br/>
                Enable the <code>metrics</code> feature for Prometheus-compatible metrics.
            </p>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">API Endpoints</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Endpoint</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><code>GET /api/stats</code></td><td>JSON statistics for programmatic access</td></tr>
                    <tr><td><code>GET /health</code></td><td>Health check endpoint (API server)</td></tr>
                    <tr><td><code>GET /metrics</code></td><td>Prometheus metrics (if enabled)</td></tr>
                </tbody>
            </table>
        </div>
        "#,
        stat_status = stat_card("Status", "Healthy", ""),
        stat_uptime = stat_card("Uptime", &format_uptime(uptime), ""),
        stat_memory = stat_card("Est. Memory", &format_bytes(total_memory), ""),
        stat_vectors = stat_card("Vectors", &format_number(total_vectors), ""),
        memory_rows = memory_rows,
        empty_memory = if collections.is_empty() {
            r#"<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No collections to display.</p>"#
        } else {
            ""
        },
    );

    Html(base_layout("Monitoring", &content, "monitoring"))
}

/// GET /api/stats - JSON stats endpoint
async fn api_stats_handler(State(state): State<Arc<WebUiState>>) -> Json<StatsResponse> {
    let db = state.db.read().await;
    let collection_names = db.list_collections();
    let total_vectors = db.total_vectors();
    let uptime = state.uptime();

    let collections: Vec<CollectionStatsResponse> = collection_names
        .iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            Some(CollectionStatsResponse {
                name: name.clone(),
                vector_count: coll.len(),
                dimensions: coll.dimensions().unwrap_or(0),
                deleted_count: coll.deleted_count(),
                needs_compaction: coll.needs_compaction(0.2),
            })
        })
        .collect();

    Json(StatsResponse {
        healthy: true,
        uptime_seconds: uptime,
        total_collections: collection_names.len(),
        total_vectors,
        collections,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "web-ui",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

// ============================================================================
// Router and Server
// ============================================================================

/// Create the Web UI router
pub fn create_web_ui_router(state: Arc<WebUiState>) -> Router {
    Router::new()
        // HTML pages
        .route("/", get(dashboard_handler))
        .route("/collections", get(collections_list_handler))
        .route("/collections/{name}", get(collection_detail_handler))
        .route("/query", get(query_playground_handler))
        .route("/monitoring", get(monitoring_handler))
        // API endpoints
        .route("/api/stats", get(api_stats_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

/// Start the Web UI server
///
/// # Arguments
///
/// * `db` - The Needle database instance
/// * `config` - Web UI configuration
///
/// # Example
///
/// ```rust,ignore
/// use needle::web_ui::{WebUiConfig, serve_web_ui};
/// use needle::Database;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let db = Database::open("vectors.needle")?;
///     let config = WebUiConfig::default();
///     serve_web_ui(db, config).await?;
///     Ok(())
/// }
/// ```
pub async fn serve_web_ui(db: Database, config: WebUiConfig) -> Result<(), Box<dyn std::error::Error>> {
    let addr = config.addr;
    let state = Arc::new(WebUiState::new(db, config));
    let app = create_web_ui_router(state);

    println!("Needle Web UI starting on http://{}", addr);
    println!("  Dashboard:   http://{}/", addr);
    println!("  Collections: http://{}/collections", addr);
    println!("  Query:       http://{}/query", addr);
    println!("  Monitoring:  http://{}/monitoring", addr);
    println!("  API Stats:   http://{}/api/stats", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the Web UI with default configuration
pub async fn serve_web_ui_default(db: Database) -> Result<(), Box<dyn std::error::Error>> {
    serve_web_ui(db, WebUiConfig::default()).await
}

// ============================================================================
// Visual Query Builder & Admin Dashboard (Next-Gen)
// ============================================================================

/// Visual query builder state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualQueryState {
    /// Selected collection
    pub collection: Option<String>,
    /// Query vector (can be pasted or generated)
    pub query_vector: Option<Vec<f32>>,
    /// Number of results
    pub limit: usize,
    /// Filter conditions
    pub filters: Vec<FilterCondition>,
    /// Distance function
    pub distance: String,
    /// Include metadata in results
    pub include_metadata: bool,
    /// Use HNSW index
    pub use_index: bool,
}

impl Default for VisualQueryState {
    fn default() -> Self {
        Self {
            collection: None,
            query_vector: None,
            limit: 10,
            filters: Vec::new(),
            distance: "cosine".to_string(),
            include_metadata: true,
            use_index: true,
        }
    }
}

/// Filter condition for the visual builder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterCondition {
    /// Field name
    pub field: String,
    /// Operator (eq, ne, gt, gte, lt, lte, in, contains)
    pub operator: String,
    /// Value
    pub value: serde_json::Value,
}

impl FilterCondition {
    /// Convert to Filter
    pub fn to_filter(&self) -> Option<crate::metadata::Filter> {
        Some(match self.operator.as_str() {
            "eq" => crate::metadata::Filter::eq(self.field.clone(), self.value.clone()),
            "ne" => crate::metadata::Filter::ne(self.field.clone(), self.value.clone()),
            "gt" => crate::metadata::Filter::gt(self.field.clone(), self.value.clone()),
            "gte" => crate::metadata::Filter::gte(self.field.clone(), self.value.clone()),
            "lt" => crate::metadata::Filter::lt(self.field.clone(), self.value.clone()),
            "lte" => crate::metadata::Filter::lte(self.field.clone(), self.value.clone()),
            "in" => {
                if let serde_json::Value::Array(arr) = &self.value {
                    crate::metadata::Filter::In {
                        field: self.field.clone(),
                        values: arr.clone(),
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        })
    }
}

/// Admin dashboard section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminSection {
    /// Section name
    pub name: String,
    /// Section description
    pub description: String,
    /// Available actions
    pub actions: Vec<AdminAction>,
}

/// Admin action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminAction {
    /// Action ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// HTTP method
    pub method: String,
    /// Endpoint
    pub endpoint: String,
    /// Required parameters
    pub params: Vec<ActionParam>,
    /// Whether action is dangerous
    pub dangerous: bool,
}

/// Action parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionParam {
    /// Parameter name
    pub name: String,
    /// Parameter type (string, number, boolean, array)
    pub param_type: String,
    /// Whether required
    pub required: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
    /// Description
    pub description: String,
}

/// Generate admin sections
pub fn get_admin_sections() -> Vec<AdminSection> {
    vec![
        AdminSection {
            name: "Collection Management".to_string(),
            description: "Create, modify, and delete collections".to_string(),
            actions: vec![
                AdminAction {
                    id: "create_collection".to_string(),
                    name: "Create Collection".to_string(),
                    description: "Create a new vector collection".to_string(),
                    method: "POST".to_string(),
                    endpoint: "/api/collections".to_string(),
                    params: vec![
                        ActionParam {
                            name: "name".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            default: None,
                            description: "Collection name".to_string(),
                        },
                        ActionParam {
                            name: "dimensions".to_string(),
                            param_type: "number".to_string(),
                            required: true,
                            default: Some(serde_json::json!(384)),
                            description: "Vector dimensions".to_string(),
                        },
                        ActionParam {
                            name: "distance".to_string(),
                            param_type: "string".to_string(),
                            required: false,
                            default: Some(serde_json::json!("cosine")),
                            description: "Distance function".to_string(),
                        },
                    ],
                    dangerous: false,
                },
                AdminAction {
                    id: "delete_collection".to_string(),
                    name: "Delete Collection".to_string(),
                    description: "Delete a collection and all its data".to_string(),
                    method: "DELETE".to_string(),
                    endpoint: "/api/collections/{name}".to_string(),
                    params: vec![
                        ActionParam {
                            name: "name".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            default: None,
                            description: "Collection to delete".to_string(),
                        },
                    ],
                    dangerous: true,
                },
                AdminAction {
                    id: "compact_collection".to_string(),
                    name: "Compact Collection".to_string(),
                    description: "Remove deleted vectors and reclaim space".to_string(),
                    method: "POST".to_string(),
                    endpoint: "/api/collections/{name}/compact".to_string(),
                    params: vec![
                        ActionParam {
                            name: "name".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            default: None,
                            description: "Collection to compact".to_string(),
                        },
                    ],
                    dangerous: false,
                },
            ],
        },
        AdminSection {
            name: "Data Management".to_string(),
            description: "Import, export, and backup data".to_string(),
            actions: vec![
                AdminAction {
                    id: "export_collection".to_string(),
                    name: "Export Collection".to_string(),
                    description: "Export collection data to JSON".to_string(),
                    method: "GET".to_string(),
                    endpoint: "/api/collections/{name}/export".to_string(),
                    params: vec![
                        ActionParam {
                            name: "name".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            default: None,
                            description: "Collection to export".to_string(),
                        },
                    ],
                    dangerous: false,
                },
                AdminAction {
                    id: "save_database".to_string(),
                    name: "Save Database".to_string(),
                    description: "Persist all changes to disk".to_string(),
                    method: "POST".to_string(),
                    endpoint: "/api/save".to_string(),
                    params: vec![],
                    dangerous: false,
                },
            ],
        },
        AdminSection {
            name: "Monitoring".to_string(),
            description: "View metrics and health status".to_string(),
            actions: vec![
                AdminAction {
                    id: "health_check".to_string(),
                    name: "Health Check".to_string(),
                    description: "Check database health status".to_string(),
                    method: "GET".to_string(),
                    endpoint: "/health".to_string(),
                    params: vec![],
                    dangerous: false,
                },
                AdminAction {
                    id: "get_stats".to_string(),
                    name: "Get Statistics".to_string(),
                    description: "Get database statistics".to_string(),
                    method: "GET".to_string(),
                    endpoint: "/api/stats".to_string(),
                    params: vec![],
                    dangerous: false,
                },
            ],
        },
    ]
}

/// Generate the visual query builder HTML
pub fn generate_query_builder_html(state: &VisualQueryState, collections: &[String]) -> String {
    let collection_options: String = collections
        .iter()
        .map(|c| {
            let selected = state.collection.as_ref() == Some(c);
            format!(
                r#"<option value="{}" {}>{}</option>"#,
                c,
                if selected { "selected" } else { "" },
                c
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let filters_html: String = state
        .filters
        .iter()
        .enumerate()
        .map(|(i, f)| {
            format!(
                r#"
                <div class="filter-row" data-index="{}">
                    <input type="text" name="filter_field_{}" value="{}" placeholder="Field" class="field-input">
                    <select name="filter_op_{}">
                        <option value="eq" {}>equals</option>
                        <option value="ne" {}>not equals</option>
                        <option value="gt" {}>greater than</option>
                        <option value="gte" {}>greater or equal</option>
                        <option value="lt" {}>less than</option>
                        <option value="lte" {}>less or equal</option>
                        <option value="in" {}>in list</option>
                    </select>
                    <input type="text" name="filter_value_{}" value="{}" placeholder="Value" class="value-input">
                    <button type="button" class="remove-filter" onclick="removeFilter({})">✕</button>
                </div>
                "#,
                i,
                i, f.field,
                i,
                if f.operator == "eq" { "selected" } else { "" },
                if f.operator == "ne" { "selected" } else { "" },
                if f.operator == "gt" { "selected" } else { "" },
                if f.operator == "gte" { "selected" } else { "" },
                if f.operator == "lt" { "selected" } else { "" },
                if f.operator == "lte" { "selected" } else { "" },
                if f.operator == "in" { "selected" } else { "" },
                i, serde_json::to_string(&f.value).unwrap_or_default(),
                i
            )
        })
        .collect();

    format!(
        r#"
        <div class="query-builder">
            <h2>Visual Query Builder</h2>
            
            <div class="builder-section">
                <label>Collection</label>
                <select id="collection-select" name="collection">
                    <option value="">Select collection...</option>
                    {}
                </select>
            </div>

            <div class="builder-section">
                <label>Query Vector</label>
                <div class="vector-input-group">
                    <textarea id="query-vector" name="query_vector" 
                        placeholder="Paste vector as JSON array, e.g., [0.1, 0.2, ...]"
                        rows="3">{}</textarea>
                    <button type="button" onclick="generateRandomVector()">Generate Random</button>
                </div>
            </div>

            <div class="builder-section">
                <label>Filters</label>
                <div id="filters-container">
                    {}
                </div>
                <button type="button" onclick="addFilter()">+ Add Filter</button>
            </div>

            <div class="builder-section options">
                <div class="option-group">
                    <label>Results Limit</label>
                    <input type="number" name="limit" value="{}" min="1" max="1000">
                </div>
                <div class="option-group">
                    <label>Distance Function</label>
                    <select name="distance">
                        <option value="cosine" {}>Cosine</option>
                        <option value="euclidean" {}>Euclidean</option>
                        <option value="dot" {}>Dot Product</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>
                        <input type="checkbox" name="include_metadata" {}> Include Metadata
                    </label>
                </div>
                <div class="option-group">
                    <label>
                        <input type="checkbox" name="use_index" {}> Use HNSW Index
                    </label>
                </div>
            </div>

            <div class="builder-actions">
                <button type="submit" class="primary">Execute Query</button>
                <button type="button" onclick="resetBuilder()">Reset</button>
                <button type="button" onclick="generateCurl()">Generate cURL</button>
            </div>
        </div>

        <script>
        let filterIndex = {};
        
        function addFilter() {{
            const container = document.getElementById('filters-container');
            const html = `
                <div class="filter-row" data-index="${{filterIndex}}">
                    <input type="text" name="filter_field_${{filterIndex}}" placeholder="Field" class="field-input">
                    <select name="filter_op_${{filterIndex}}">
                        <option value="eq">equals</option>
                        <option value="ne">not equals</option>
                        <option value="gt">greater than</option>
                        <option value="gte">greater or equal</option>
                        <option value="lt">less than</option>
                        <option value="lte">less or equal</option>
                        <option value="in">in list</option>
                    </select>
                    <input type="text" name="filter_value_${{filterIndex}}" placeholder="Value" class="value-input">
                    <button type="button" class="remove-filter" onclick="removeFilter(${{filterIndex}})">✕</button>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', html);
            filterIndex++;
        }}

        function removeFilter(index) {{
            const row = document.querySelector(`.filter-row[data-index="${{index}}"]`);
            if (row) row.remove();
        }}

        function generateRandomVector() {{
            const collection = document.getElementById('collection-select').value;
            // Default to 384 dimensions
            const dims = 384;
            const vector = Array.from({{length: dims}}, () => Math.random() * 2 - 1);
            document.getElementById('query-vector').value = JSON.stringify(vector);
        }}

        function resetBuilder() {{
            document.getElementById('collection-select').value = '';
            document.getElementById('query-vector').value = '';
            document.getElementById('filters-container').innerHTML = '';
            filterIndex = 0;
        }}

        function generateCurl() {{
            const collection = document.getElementById('collection-select').value;
            const vector = document.getElementById('query-vector').value;
            const limit = document.querySelector('input[name="limit"]').value;
            
            if (!collection || !vector) {{
                alert('Please select a collection and enter a query vector');
                return;
            }}

            const curl = `curl -X POST http://localhost:8080/collections/${{collection}}/search \\
  -H "Content-Type: application/json" \\
  -d '{{"vector": ${{vector}}, "k": ${{limit}}}}'`;
            
            navigator.clipboard.writeText(curl);
            alert('cURL command copied to clipboard!');
        }}
        </script>
        "#,
        collection_options,
        state.query_vector.as_ref().map(|v| serde_json::to_string(v).unwrap_or_default()).unwrap_or_default(),
        filters_html,
        state.limit,
        if state.distance == "cosine" { "selected" } else { "" },
        if state.distance == "euclidean" { "selected" } else { "" },
        if state.distance == "dot" { "selected" } else { "" },
        if state.include_metadata { "checked" } else { "" },
        if state.use_index { "checked" } else { "" },
        state.filters.len()
    )
}

/// Generate the admin dashboard HTML
pub fn generate_admin_dashboard_html() -> String {
    let sections = get_admin_sections();

    let sections_html: String = sections
        .iter()
        .map(|section| {
            let actions_html: String = section
                .actions
                .iter()
                .map(|action| {
                    let params_html: String = action
                        .params
                        .iter()
                        .map(|p| {
                            let input_type = match p.param_type.as_str() {
                                "number" => "number",
                                "boolean" => "checkbox",
                                _ => "text",
                            };
                            format!(
                                r#"
                                <div class="param-row">
                                    <label>{}{}</label>
                                    <input type="{}" name="{}" value="{}" placeholder="{}">
                                </div>
                                "#,
                                p.name,
                                if p.required { " *" } else { "" },
                                input_type,
                                p.name,
                                p.default.as_ref().map(|v| v.to_string()).unwrap_or_default(),
                                p.description
                            )
                        })
                        .collect();

                    let danger_class = if action.dangerous { "danger" } else { "" };

                    format!(
                        r#"
                        <div class="action-card {}">
                            <h4>{}</h4>
                            <p class="description">{}</p>
                            <div class="action-details">
                                <span class="method method-{}">{}</span>
                                <code>{}</code>
                            </div>
                            <form class="action-form" data-endpoint="{}" data-method="{}">
                                {}
                                <button type="submit" class="{}">{}</button>
                            </form>
                        </div>
                        "#,
                        danger_class,
                        action.name,
                        action.description,
                        action.method.to_lowercase(),
                        action.method,
                        action.endpoint,
                        action.endpoint,
                        action.method,
                        params_html,
                        danger_class,
                        if action.dangerous { "⚠ Execute" } else { "Execute" }
                    )
                })
                .collect();

            format!(
                r#"
                <div class="admin-section">
                    <h3>{}</h3>
                    <p class="section-desc">{}</p>
                    <div class="actions-grid">
                        {}
                    </div>
                </div>
                "#,
                section.name, section.description, actions_html
            )
        })
        .collect();

    format!(
        r#"
        <div class="admin-dashboard">
            <h2>Admin Dashboard</h2>
            <p class="warning">⚠️ Actions here can modify or delete data. Use with caution.</p>
            {}
        </div>

        <style>
        .admin-dashboard {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .warning {{ color: #f59e0b; background: #fef3c7; padding: 10px; border-radius: 4px; }}
        .admin-section {{ margin: 30px 0; padding: 20px; background: #f8fafc; border-radius: 8px; }}
        .section-desc {{ color: #64748b; margin-bottom: 20px; }}
        .actions-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
        .action-card {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .action-card.danger {{ border-color: #ef4444; }}
        .action-card h4 {{ margin: 0 0 10px 0; }}
        .action-card .description {{ color: #64748b; font-size: 14px; }}
        .action-details {{ margin: 15px 0; }}
        .method {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .method-get {{ background: #22c55e; color: white; }}
        .method-post {{ background: #3b82f6; color: white; }}
        .method-delete {{ background: #ef4444; color: white; }}
        .action-form {{ margin-top: 15px; }}
        .param-row {{ margin: 10px 0; }}
        .param-row label {{ display: block; font-size: 14px; margin-bottom: 4px; }}
        .param-row input {{ width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 4px; }}
        .action-form button {{ margin-top: 10px; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }}
        .action-form button {{ background: #3b82f6; color: white; }}
        .action-form button.danger {{ background: #ef4444; }}
        </style>

        <script>
        document.querySelectorAll('.action-form').forEach(form => {{
            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                const endpoint = form.dataset.endpoint;
                const method = form.dataset.method;
                
                // Confirm dangerous actions
                if (form.querySelector('button.danger')) {{
                    if (!confirm('Are you sure you want to perform this action?')) {{
                        return;
                    }}
                }}

                // Build the request
                const formData = new FormData(form);
                let url = endpoint;
                let body = null;

                // Replace path params
                formData.forEach((value, key) => {{
                    if (url.includes(`{{${{key}}}`)) {{
                        url = url.replace(`{{${{key}}}}`, value);
                    }}
                }});

                if (method !== 'GET') {{
                    const data = {{}};
                    formData.forEach((value, key) => {{
                        if (!endpoint.includes(`{{${{key}}}`)) {{
                            data[key] = value;
                        }}
                    }});
                    body = JSON.stringify(data);
                }}

                try {{
                    const response = await fetch(url, {{
                        method: method,
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: body
                    }});
                    const result = await response.json();
                    alert(response.ok ? 'Success!' : 'Error: ' + JSON.stringify(result));
                }} catch (err) {{
                    alert('Error: ' + err.message);
                }}
            }});
        }});
        </script>
        "#,
        sections_html
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_uptime() {
        assert_eq!(format_uptime(30), "30s");
        assert_eq!(format_uptime(90), "1m 30s");
        assert_eq!(format_uptime(3661), "1h 1m 1s");
        assert_eq!(format_uptime(86400), "1d 0h 0m");
        assert_eq!(format_uptime(90061), "1d 1h 1m");
    }

    #[test]
    fn test_web_ui_config_default() {
        let config = WebUiConfig::default();
        assert_eq!(config.addr.port(), 8081);
        assert_eq!(config.title, "Needle Dashboard");
        assert!(config.enable_query_playground);
    }

    #[test]
    fn test_web_ui_config_builder() {
        let config = WebUiConfig::new("127.0.0.1:9000")
            .with_title("My Dashboard")
            .with_query_playground(false);

        assert_eq!(config.addr.port(), 9000);
        assert_eq!(config.title, "My Dashboard");
        assert!(!config.enable_query_playground);
    }

    #[test]
    fn test_stats_response_serialization() {
        let stats = StatsResponse {
            healthy: true,
            uptime_seconds: 3600,
            total_collections: 2,
            total_vectors: 1000,
            collections: vec![
                CollectionStatsResponse {
                    name: "test".to_string(),
                    vector_count: 500,
                    dimensions: 384,
                    deleted_count: 10,
                    needs_compaction: false,
                },
            ],
            version: "0.1.0".to_string(),
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"healthy\":true"));
        assert!(json.contains("\"total_vectors\":1000"));
    }
}
