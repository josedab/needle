//! HTTP REST API server for Needle
//!
//! Provides a REST API for vector database operations, making Needle
//! accessible from any language or tool that can make HTTP requests.
//!
//! # Authentication
//!
//! The server supports multiple authentication methods:
//!
//! ## API Key Authentication
//!
//! ```rust,ignore
//! use needle::server::{ServerConfig, AuthConfig, ApiKey};
//!
//! let auth_config = AuthConfig::new()
//!     .with_api_key(ApiKey::new("my-api-key").with_role("admin"))
//!     .require_auth(true);
//!
//! let config = ServerConfig::default()
//!     .with_auth(auth_config);
//! ```
//!
//! API keys are passed via the `X-API-Key` header:
//! ```bash
//! curl -H "X-API-Key: my-api-key" http://localhost:8080/collections
//! ```
//!
//! ## JWT Authentication
//!
//! ```rust,ignore
//! let auth_config = AuthConfig::new()
//!     .with_jwt_secret("your-secret-key")
//!     .require_auth(true);
//! ```
//!
//! JWT tokens are passed via the `Authorization: Bearer <token>` header.


mod auth;
mod handlers;
mod middleware;
mod types;

pub use auth::*;
pub use types::*;

use handlers::*;
use middleware::*;

use crate::database::Database;
use crate::error::NeedleError;
use crate::metadata::Filter;
use crate::security::{Role, User};
use axum::{
    body::Body,
    extract::{ConnectInfo, Path, Query, State},
    http::{header, Method, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{delete, get, post, put},
    Json, Router,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use governor::{clock::DefaultClock, state::keyed::DashMapStateStore, Quota, RateLimiter};
use hmac::{Hmac, Mac};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};

#[cfg(feature = "metrics")]
use crate::metrics::{http_metrics, metrics};

// ── Server defaults ──────────────────────────────────────────────────────────
/// Default maximum request body size (10 MB).
const DEFAULT_MAX_BODY_SIZE: usize = 10 * 1024 * 1024;
/// Default maximum items per batch operation.
const DEFAULT_MAX_BATCH_SIZE: usize = 10_000;
/// Default request timeout in seconds.
const DEFAULT_REQUEST_TIMEOUT_SECS: u64 = 30;
/// Default CORS preflight cache duration in seconds.
const DEFAULT_CORS_MAX_AGE_SECS: u64 = 3600;
/// Default server listen port.
const DEFAULT_PORT: u16 = 8080;

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Address to bind to
    pub addr: SocketAddr,
    /// CORS configuration
    pub cors_config: CorsConfig,
    /// Database path (None for in-memory)
    pub db_path: Option<String>,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// Authentication configuration
    pub auth: AuthConfig,
    /// Maximum request body size in bytes (default: 100MB)
    pub max_body_size: usize,
    /// Maximum number of items in a batch operation (default: 10000)
    /// Prevents memory exhaustion from large batch requests
    pub max_batch_size: usize,
    /// Request timeout in seconds (default: 30)
    /// Requests exceeding this duration will be terminated
    pub request_timeout_secs: u64,
    /// Trusted proxies for honoring X-Forwarded-For/X-Real-IP headers.
    /// Only requests from these IPs may override the client IP.
    pub trusted_proxies: Vec<IpAddr>,
}

/// CORS configuration
#[derive(Debug, Clone)]
pub struct CorsConfig {
    /// Enable CORS
    pub enabled: bool,
    /// Allowed origins (None = allow all, Some([]) = deny all external)
    pub allowed_origins: Option<Vec<String>>,
    /// Allow credentials
    pub allow_credentials: bool,
    /// Max age for preflight cache (in seconds)
    pub max_age_secs: u64,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            // Default to localhost only for security
            allowed_origins: Some(vec![
                "http://localhost:3000".to_string(),
                "http://localhost:8080".to_string(),
                "http://127.0.0.1:3000".to_string(),
                "http://127.0.0.1:8080".to_string(),
            ]),
            allow_credentials: false,
            max_age_secs: DEFAULT_CORS_MAX_AGE_SECS,
        }
    }
}

impl CorsConfig {
    /// Create a permissive CORS config (not recommended for production)
    ///
    /// Note: Credentials are NOT enabled by default even in permissive mode.
    /// Combining allow-all-origins with credentials is a CSRF vulnerability.
    /// Use `with_credentials(true)` only with specific origins for secure apps.
    pub fn permissive() -> Self {
        Self {
            enabled: true,
            allowed_origins: None,    // Allow all
            allow_credentials: false, // SECURITY: Never combine wildcard origins with credentials
            max_age_secs: DEFAULT_CORS_MAX_AGE_SECS,
        }
    }

    /// Enable credentials (cookies, auth headers).
    ///
    /// WARNING: Only enable credentials with specific allowed_origins, not with
    /// wildcard origins. Combining both creates a CSRF vulnerability where any
    /// website can make authenticated requests on behalf of users.
    pub fn with_credentials(mut self, allow: bool) -> Self {
        self.allow_credentials = allow;
        self
    }

    /// Create a restrictive CORS config
    pub fn restrictive() -> Self {
        Self {
            enabled: true,
            allowed_origins: Some(vec![]), // No external origins
            allow_credentials: false,
            max_age_secs: 0,
        }
    }

    /// Add an allowed origin
    pub fn with_origin(mut self, origin: impl Into<String>) -> Self {
        let origins = self.allowed_origins.get_or_insert_with(Vec::new);
        origins.push(origin.into());
        self
    }

    /// Validate the CORS configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.allow_credentials && self.allowed_origins.is_none() {
            return Err(
                "CORS: allow_credentials cannot be combined with wildcard origins — \
                 this is insecure and rejected by browsers. \
                 Specify explicit allowed_origins instead."
                    .to_string(),
            );
        }
        Ok(())
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Requests per second (global limit)
    pub requests_per_second: u32,
    /// Burst size (allows short bursts above the rate)
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            requests_per_second: 100,
            burst_size: 50,
        }
    }
}

impl RateLimitConfig {
    /// Disable rate limiting
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            requests_per_second: 0,
            burst_size: 0,
        }
    }

    /// Set requests per second
    pub fn with_rate(mut self, rps: u32) -> Self {
        self.requests_per_second = rps;
        self
    }

    /// Set burst size
    pub fn with_burst(mut self, burst: u32) -> Self {
        self.burst_size = burst;
        self
    }
}

fn default_trusted_proxies() -> Vec<IpAddr> {
    vec![
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V6(Ipv6Addr::LOCALHOST),
    ]
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: SocketAddr::from(([127, 0, 0, 1], DEFAULT_PORT)),
            cors_config: CorsConfig::default(),
            db_path: None,
            rate_limit: RateLimitConfig::default(),
            auth: AuthConfig::default(),
            max_body_size: DEFAULT_MAX_BODY_SIZE,
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            request_timeout_secs: DEFAULT_REQUEST_TIMEOUT_SECS,
            trusted_proxies: default_trusted_proxies(),
        }
    }
}

impl ServerConfig {
    pub fn new(addr: &str) -> Result<Self, std::net::AddrParseError> {
        Ok(Self {
            addr: addr.parse()?,
            ..Default::default()
        })
    }

    pub fn with_db_path(mut self, path: impl Into<String>) -> Self {
        self.db_path = Some(path.into());
        self
    }

    pub fn with_cors(mut self, config: CorsConfig) -> Self {
        self.cors_config = config;
        self
    }

    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = config;
        self
    }

    pub fn with_max_body_size(mut self, bytes: usize) -> Self {
        self.max_body_size = bytes;
        self
    }

    pub fn with_max_batch_size(mut self, size: usize) -> Self {
        self.max_batch_size = size;
        self
    }

    pub fn with_request_timeout(mut self, secs: u64) -> Self {
        self.request_timeout_secs = secs;
        self
    }

    /// Add a trusted proxy IP for forwarded header handling.
    pub fn with_trusted_proxy(mut self, proxy: IpAddr) -> Self {
        self.trusted_proxies.push(proxy);
        self
    }

    /// Replace the trusted proxy list. Use an empty list to disable trust.
    pub fn with_trusted_proxies(mut self, proxies: Vec<IpAddr>) -> Self {
        self.trusted_proxies = proxies;
        self
    }

    /// Set authentication configuration.
    pub fn with_auth(mut self, config: AuthConfig) -> Self {
        self.auth = config;
        self
    }
}

/// Per-IP rate limiter type
type PerIpRateLimiter = RateLimiter<IpAddr, DashMapStateStore<IpAddr>, DefaultClock>;

/// Shared application state
pub struct AppState {
    db: RwLock<Database>,
    rate_limiter: Option<Arc<PerIpRateLimiter>>,
    /// Authentication configuration
    auth: AuthConfig,
    /// Maximum items allowed in batch operations
    max_batch_size: usize,
    /// Trusted proxies for forwarded header handling
    trusted_proxies: Vec<IpAddr>,
    /// Optional embedding provider for auto-embed text endpoints
    #[cfg(feature = "embedding-providers")]
    embed_provider: Option<Arc<dyn crate::embeddings_provider::EmbeddingProvider>>,
}

impl AppState {
    /// Create a new AppState with the given database and no rate limiting
    pub fn new(db: Database) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: None,
            auth: AuthConfig::default(),
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            trusted_proxies: default_trusted_proxies(),
            #[cfg(feature = "embedding-providers")]
            embed_provider: None,
        }
    }

    /// Create a new AppState with the given database and rate limiting config
    pub fn with_rate_limit(db: Database, config: &RateLimitConfig) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: create_rate_limiter(config),
            auth: AuthConfig::default(),
            max_batch_size: DEFAULT_MAX_BATCH_SIZE,
            trusted_proxies: default_trusted_proxies(),
            #[cfg(feature = "embedding-providers")]
            embed_provider: None,
        }
    }

    /// Create a new AppState with full configuration
    pub fn with_config(db: Database, config: &ServerConfig) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: create_rate_limiter(&config.rate_limit),
            auth: config.auth.clone(),
            max_batch_size: config.max_batch_size,
            trusted_proxies: config.trusted_proxies.clone(),
            #[cfg(feature = "embedding-providers")]
            embed_provider: None,
        }
    }
}

/// Build the router with configuration
#[allow(deprecated)]
pub fn create_router_with_config(state: Arc<AppState>, config: &ServerConfig) -> std::result::Result<Router, String> {
    let cors_layer = build_cors_layer(&config.cors_config)?;
    let timeout_layer = TimeoutLayer::new(Duration::from_secs(config.request_timeout_secs));

    #[allow(unused_mut)]
    let mut router = Router::new()
        // Health & Info
        .route("/health", get(health))
        .route("/", get(get_info))
        .route("/info", get(get_info))
        // Collections
        .route("/collections", get(list_collections))
        .route("/collections", post(create_collection))
        .route("/collections/:name", get(get_collection))
        .route("/collections/:name", delete(delete_collection))
        .route("/collections/:name/compact", post(compact_collection))
        .route("/collections/:name/export", get(export_collection))
        // Vectors
        .route("/collections/:collection/vectors", get(list_vectors))
        .route("/collections/:collection/vectors", post(insert_vector))
        .route("/collections/:collection/vectors/batch", post(batch_insert))
        .route("/collections/:collection/vectors/upsert", post(upsert_vector))
        .route("/collections/:collection/vectors/:id", get(get_vector))
        .route("/collections/:collection/vectors/:id", delete(delete_vector))
        .route("/collections/:collection/vectors/:id/metadata", post(update_metadata))
        // Text insertion (auto-embed) — built-in hash embeddings or external provider
        .route("/collections/:collection/texts", post(insert_text_handler))
        .route("/collections/:collection/texts/batch", post(batch_insert_text_handler))
        .route("/collections/:collection/texts/search", post(search_text_handler))
        // Search
        .route("/collections/:collection/search", post(search))
        .route("/collections/:collection/search/batch", post(batch_search))
        .route("/collections/:collection/search/radius", post(radius_search))
        // GraphRAG search
        .route("/collections/:collection/search/graph", post(graph_search_handler))
        // Matryoshka two-phase search
        .route("/collections/:collection/search/matryoshka", post(matryoshka_search_handler))
        // Semantic cache
        .route("/collections/:collection/cache/lookup", post(cache_lookup_handler))
        .route("/collections/:collection/cache/store", post(cache_store_handler))
        // Streaming ingestion
        .route("/collections/:collection/ingest", post(streaming_insert_handler))
        // Time-travel queries
        .route("/collections/:collection/search/time-travel", post(time_travel_search_handler))
        .route("/collections/:collection/snapshots/diff", post(snapshot_diff_handler))
        // Agentic memory protocol
        .route("/collections/:collection/memory/remember", post(remember_handler))
        .route("/collections/:collection/memory/recall", post(recall_handler))
        .route("/collections/:collection/memory/:memory_id/forget", delete(forget_handler))
        // Query cost estimation
        .route("/collections/:collection/search/estimate", post(cost_estimate_handler))
        // Vector diff
        .route("/collections/:collection/diff", post(vector_diff_handler))
        // CDC change feed (JSON) and SSE streaming
        .route("/collections/:collection/changes", get(change_feed_handler))
        .route("/collections/:collection/changes/stream", get(change_stream_sse_handler))
        // In-process benchmark
        .route("/collections/:collection/benchmark", post(benchmark_handler))
        // Index status (incremental index / WAL)
        .route("/collections/:collection/index/status", get(index_status_handler))
        // Cluster/shard topology
        .route("/cluster/status", get(cluster_status_handler))
        // gRPC schema definitions
        .route("/grpc/schema", get(grpc_schema_handler))
        // OpenTelemetry tracing status
        .route("/tracing/status", get(tracing_status_handler))
        // Webhook management
        .route("/webhooks", post(create_webhook_handler))
        .route("/webhooks", get(list_webhooks_handler))
        .route("/webhooks/:id", delete(delete_webhook_handler))
        // Embedding model router
        .route("/embeddings/router/status", get(embedding_router_status_handler))
        // Database operations
        .route("/save", post(save_database))
        // Aliases
        .route("/aliases", post(create_alias_handler))
        .route("/aliases", get(list_aliases_handler))
        .route("/aliases/:alias", get(get_alias_handler))
        .route("/aliases/:alias", delete(delete_alias_handler))
        .route("/aliases/:alias", put(update_alias_handler))
        // TTL endpoints
        .route("/collections/:name/expire", post(expire_vectors_handler))
        .route("/collections/:name/ttl-stats", get(ttl_stats_handler))
        // OpenAPI spec
        .route("/openapi.json", get(serve_openapi_spec))
        // Dashboard
        .route("/dashboard", get(serve_dashboard))
        // Snapshot endpoints
        .route("/collections/:name/snapshots", get(list_snapshots_handler))
        .route("/collections/:name/snapshots", post(create_snapshot_handler))
        .route("/collections/:name/snapshots/:snapshot/restore", post(restore_snapshot_handler))
        // MCP over HTTP
        .route("/mcp", post(mcp_http_handler))
        .route("/mcp/config", get(mcp_config_handler))
        // Interactive playground
        .route("/playground", get(serve_playground));

    // Add metrics endpoint when metrics feature is enabled
    #[cfg(feature = "metrics")]
    {
        router = router.route("/metrics", get(get_metrics));
    }

    // Apply authentication and rate limiting (auth runs first, then rate limiting)
    let router = router
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            rate_limit_middleware,
        ))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .with_state(state);

    // Build final router with all layers
    // Note: metrics middleware is applied here (outermost) so it captures all requests
    #[cfg(feature = "metrics")]
    let router = router.layer(axum::middleware::from_fn(metrics_middleware));

    Ok(router
        .layer(axum::middleware::from_fn(api_stability_middleware))
        .layer(axum::middleware::from_fn(security_headers_middleware))
        .layer(TraceLayer::new_for_http())
        .layer(timeout_layer)
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(cors_layer))
}

/// Build the router (legacy, uses permissive CORS for backwards compatibility)
#[deprecated(since = "0.5.0", note = "Use `create_router_with_config` with restrictive CORS defaults instead")]
pub fn create_router(state: Arc<AppState>) -> Router {
    warn!("create_router() uses permissive CORS (AllowOrigin::any()). Migrate to create_router_with_config() with restrictive CORS for production use.");
    let config = ServerConfig {
        cors_config: CorsConfig::permissive(),
        ..Default::default()
    };
    create_router_with_config(state, &config).expect("default permissive CORS config is valid")
}


pub async fn serve(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Validate auth configuration at startup
    if let Err(e) = config.auth.validate() {
        return Err(format!("Invalid auth configuration: {e}").into());
    }

    // Initialize tracing subscriber with environment filter (may already be initialized by CLI)
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .try_init();

    info!(
        addr = %config.addr,
        timeout_secs = config.request_timeout_secs,
        rate_limit_enabled = config.rate_limit.enabled,
        "Starting Needle server"
    );

    if !config.auth.require_auth {
        warn!("⚠️  Authentication is DISABLED — all API endpoints are exposed without authentication. \
               Set require_auth=true and configure API keys or JWT for production use.");
    }

    let db = if let Some(path) = &config.db_path {
        info!(path = %path, "Opening database file");
        Database::open(path)?
    } else {
        info!("Using in-memory database");
        Database::in_memory()
    };

    let state = Arc::new(AppState::with_config(db, &config));
    let app = create_router_with_config(state.clone(), &config)?;

    info!("Listening on http://{}", config.addr);

    let listener = tokio::net::TcpListener::bind(&config.addr).await?;

    // Graceful shutdown handling
    let shutdown_signal = async {
        let ctrl_c = async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                tracing::error!("Failed to install Ctrl+C handler: {e}");
            }
        };

        #[cfg(unix)]
        let terminate = async {
            match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
                Ok(mut sig) => { sig.recv().await; }
                Err(e) => tracing::error!("Failed to install SIGTERM handler: {e}"),
            }
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        info!("Shutdown signal received, starting graceful shutdown");
    };

    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(shutdown_signal)
    .await?;

    // Save database on shutdown
    info!("Saving database before shutdown");
    {
        let mut db = state.db.write().await;
        if let Err(e) = db.save() {
            error!(error = %e, "Failed to save database during shutdown");
        } else {
            info!("Database saved successfully");
        }
    }

    info!("Server shutdown complete");
    Ok(())
}

/// Start server with default config
pub async fn serve_default() -> Result<(), Box<dyn std::error::Error>> {
    serve(ServerConfig::default()).await
}


// OpenAPI Specification Generator
// ---------------------------------------------------------------------------

/// Generates an OpenAPI 3.1 specification for the Needle REST API.
pub fn generate_openapi_spec() -> serde_json::Value {
    let mut spec = serde_json::Map::new();
    spec.insert("openapi".into(), serde_json::json!("3.1.0"));
    spec.insert("info".into(), openapi_info());
    spec.insert("servers".into(), serde_json::json!([{ "url": "http://localhost:8080", "description": "Local development" }]));
    spec.insert("tags".into(), openapi_tags());
    spec.insert("paths".into(), openapi_paths());
    spec.insert("components".into(), openapi_components());
    spec.insert("security".into(), serde_json::json!([{ "ApiKeyAuth": [] }, { "BearerAuth": [] }]));
    serde_json::Value::Object(spec)
}

fn openapi_info() -> serde_json::Value {
    serde_json::json!({
        "title": "Needle Vector Database API",
        "description": "REST API for the Needle embedded vector database",
        "version": "0.1.0",
        "license": { "name": "MIT" },
        "contact": { "name": "Needle Team" }
    })
}

fn openapi_tags() -> serde_json::Value {
    serde_json::json!([
        { "name": "System", "description": "Health, info, and database-level operations" },
        { "name": "Collections", "description": "Collection management" },
        { "name": "Vectors", "description": "Vector CRUD operations" },
        { "name": "Text", "description": "Text insertion and search with auto-embedding" },
        { "name": "Search", "description": "Vector similarity search" },
        { "name": "Semantic Cache", "description": "LLM response caching by semantic similarity" },
        { "name": "Streaming", "description": "Streaming ingestion" },
        { "name": "Time Travel", "description": "Temporal queries and snapshot diffing" },
        { "name": "Memory", "description": "Agentic memory protocol for LLM agents" },
        { "name": "Analytics", "description": "Cost estimation, benchmarking, and index status" },
        { "name": "Cluster", "description": "Cluster topology and status" },
        { "name": "Webhooks", "description": "Webhook management" },
        { "name": "Aliases", "description": "Collection alias management" },
        { "name": "TTL", "description": "Time-to-live and vector expiration" },
        { "name": "Snapshots", "description": "Point-in-time snapshot management" },
        { "name": "MCP", "description": "Model Context Protocol over HTTP" },
        { "name": "UI", "description": "Dashboard and playground" },
        { "name": "Observability", "description": "Tracing, metrics, and embedding router status" }
    ])
}

fn openapi_paths() -> serde_json::Value {
    let mut paths = serde_json::Map::new();
    openapi_paths_system(&mut paths);
    openapi_paths_collections(&mut paths);
    openapi_paths_vectors(&mut paths);
    openapi_paths_text(&mut paths);
    openapi_paths_search(&mut paths);
    openapi_paths_cache(&mut paths);
    openapi_paths_streaming(&mut paths);
    openapi_paths_time_travel(&mut paths);
    openapi_paths_memory(&mut paths);
    openapi_paths_analytics(&mut paths);
    openapi_paths_cluster(&mut paths);
    openapi_paths_webhooks(&mut paths);
    openapi_paths_aliases(&mut paths);
    openapi_paths_ttl(&mut paths);
    openapi_paths_snapshots(&mut paths);
    openapi_paths_mcp(&mut paths);
    openapi_paths_ui(&mut paths);
    openapi_paths_observability(&mut paths);
    serde_json::Value::Object(paths)
}

fn col_param() -> serde_json::Value {
    serde_json::json!({ "name": "collection", "in": "path", "required": true, "schema": { "type": "string" } })
}

fn name_param() -> serde_json::Value {
    serde_json::json!({ "name": "name", "in": "path", "required": true, "schema": { "type": "string" } })
}

fn schema_ref(name: &str) -> serde_json::Value {
    serde_json::json!({ "application/json": { "schema": { "$ref": format!("#/components/schemas/{name}") } } })
}

fn openapi_paths_system(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/health".into(), serde_json::json!({
        "get": { "summary": "Health check", "operationId": "healthCheck", "tags": ["System"],
            "responses": { "200": { "description": "Server is healthy", "content": schema_ref("HealthResponse") } } }
    }));
    paths.insert("/".into(), serde_json::json!({
        "get": { "summary": "Server info", "operationId": "getRoot", "tags": ["System"],
            "responses": { "200": { "description": "Server info and version" } } }
    }));
    paths.insert("/info".into(), serde_json::json!({
        "get": { "summary": "Server info (alias)", "operationId": "getInfo", "tags": ["System"],
            "responses": { "200": { "description": "Server info and version" } } }
    }));
    paths.insert("/save".into(), serde_json::json!({
        "post": { "summary": "Save database to disk", "operationId": "saveDatabase", "tags": ["System"],
            "responses": { "200": { "description": "Database saved" } } }
    }));
    paths.insert("/openapi.json".into(), serde_json::json!({
        "get": { "summary": "OpenAPI specification", "operationId": "getOpenApiSpec", "tags": ["System"],
            "responses": { "200": { "description": "OpenAPI 3.1 JSON spec" } } }
    }));
    paths.insert("/grpc/schema".into(), serde_json::json!({
        "get": { "summary": "gRPC schema definitions", "operationId": "grpcSchema", "tags": ["System"],
            "responses": { "200": { "description": "Protobuf schema" } } }
    }));
}

fn openapi_paths_collections(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections".into(), serde_json::json!({
        "get": { "summary": "List all collections", "operationId": "listCollections", "tags": ["Collections"],
            "responses": { "200": { "description": "List of collections", "content": { "application/json": {
                "schema": { "type": "array", "items": { "$ref": "#/components/schemas/CollectionInfo" } } } } } } },
        "post": { "summary": "Create a new collection", "operationId": "createCollection", "tags": ["Collections"],
            "requestBody": { "required": true, "content": schema_ref("CreateCollectionRequest") },
            "responses": { "201": { "description": "Collection created" }, "409": { "description": "Collection already exists" } } }
    }));
    paths.insert("/collections/{name}".into(), serde_json::json!({
        "get": { "summary": "Get collection info", "operationId": "getCollection", "tags": ["Collections"],
            "parameters": [name_param()],
            "responses": { "200": { "description": "Collection info", "content": schema_ref("CollectionInfo") }, "404": { "description": "Collection not found" } } },
        "delete": { "summary": "Delete a collection", "operationId": "deleteCollection", "tags": ["Collections"],
            "parameters": [name_param()],
            "responses": { "200": { "description": "Collection deleted" }, "404": { "description": "Collection not found" } } }
    }));
    paths.insert("/collections/{name}/compact".into(), serde_json::json!({
        "post": { "summary": "Compact a collection", "description": "Remove deleted vectors and reclaim storage space",
            "operationId": "compactCollection", "tags": ["Collections"], "parameters": [name_param()],
            "responses": { "200": { "description": "Compaction complete" }, "404": { "description": "Collection not found" } } }
    }));
    paths.insert("/collections/{name}/export".into(), serde_json::json!({
        "get": { "summary": "Export collection data", "description": "Export all vectors and metadata as JSON",
            "operationId": "exportCollection", "tags": ["Collections"], "parameters": [name_param()],
            "responses": { "200": { "description": "Collection data as JSON array" }, "404": { "description": "Collection not found" } } }
    }));
}

fn openapi_paths_vectors(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/vectors".into(), serde_json::json!({
        "get": { "summary": "List vectors in collection", "operationId": "listVectors", "tags": ["Vectors"],
            "parameters": [col_param(), { "name": "limit", "in": "query", "schema": { "type": "integer", "default": 100 } },
                { "name": "offset", "in": "query", "schema": { "type": "integer", "default": 0 } }],
            "responses": { "200": { "description": "List of vectors" } } },
        "post": { "summary": "Insert a vector", "operationId": "insertVector", "tags": ["Vectors"],
            "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("InsertVectorRequest") },
            "responses": { "201": { "description": "Vector inserted" }, "400": { "description": "Invalid input" } } }
    }));
    paths.insert("/collections/{collection}/vectors/batch".into(), serde_json::json!({
        "post": { "summary": "Batch insert vectors", "description": "Insert multiple vectors in a single request",
            "operationId": "batchInsert", "tags": ["Vectors"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("BatchInsertRequest") },
            "responses": { "200": { "description": "Vectors inserted", "content": schema_ref("BatchInsertResponse") }, "400": { "description": "Invalid input" } } }
    }));
    paths.insert("/collections/{collection}/vectors/upsert".into(), serde_json::json!({
        "post": { "summary": "Upsert a vector", "description": "Insert or update a vector by ID",
            "operationId": "upsertVector", "tags": ["Vectors"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("InsertVectorRequest") },
            "responses": { "200": { "description": "Vector upserted" }, "400": { "description": "Invalid input" } } }
    }));
    paths.insert("/collections/{collection}/vectors/{id}".into(), serde_json::json!({
        "get": { "summary": "Get a vector by ID", "operationId": "getVector", "tags": ["Vectors"],
            "parameters": [col_param(), { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Vector data", "content": schema_ref("VectorResponse") }, "404": { "description": "Vector not found" } } },
        "delete": { "summary": "Delete a vector", "operationId": "deleteVector", "tags": ["Vectors"],
            "parameters": [col_param(), { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Vector deleted" }, "404": { "description": "Not found" } } }
    }));
    paths.insert("/collections/{collection}/vectors/{id}/metadata".into(), serde_json::json!({
        "post": { "summary": "Update vector metadata", "description": "Merge or replace metadata on an existing vector",
            "operationId": "updateMetadata", "tags": ["Vectors"],
            "parameters": [col_param(), { "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "type": "object", "additionalProperties": true } } } },
            "responses": { "200": { "description": "Metadata updated" }, "404": { "description": "Vector not found" } } }
    }));
}

fn openapi_paths_text(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/texts".into(), serde_json::json!({
        "post": { "summary": "Insert text with auto-embedding", "description": "Insert a text document; the server generates an embedding automatically",
            "operationId": "insertText", "tags": ["Text"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("InsertTextRequest") },
            "responses": { "201": { "description": "Text inserted" }, "400": { "description": "Invalid input" } } }
    }));
    paths.insert("/collections/{collection}/texts/batch".into(), serde_json::json!({
        "post": { "summary": "Batch insert texts with auto-embedding", "operationId": "batchInsertTexts", "tags": ["Text"],
            "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("BatchInsertTextRequest") },
            "responses": { "200": { "description": "Texts inserted" }, "400": { "description": "Invalid input" } } }
    }));
    paths.insert("/collections/{collection}/texts/search".into(), serde_json::json!({
        "post": { "summary": "Search by text with auto-embedding", "description": "Provide a text query; the server embeds it and searches",
            "operationId": "searchText", "tags": ["Text"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("TextSearchRequest") },
            "responses": { "200": { "description": "Search results", "content": schema_ref("SearchResponse") } } }
    }));
}

fn openapi_paths_search(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/search".into(), serde_json::json!({
        "post": { "summary": "Search for similar vectors", "operationId": "searchVectors", "tags": ["Search"],
            "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("SearchRequest") },
            "responses": { "200": { "description": "Search results", "content": schema_ref("SearchResponse") } } }
    }));
    paths.insert("/collections/{collection}/search/batch".into(), serde_json::json!({
        "post": { "summary": "Batch search", "description": "Search for multiple query vectors in a single request",
            "operationId": "batchSearch", "tags": ["Search"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("BatchSearchRequest") },
            "responses": { "200": { "description": "Batch search results", "content": schema_ref("BatchSearchResponse") } } }
    }));
    paths.insert("/collections/{collection}/search/radius".into(), serde_json::json!({
        "post": { "summary": "Radius search", "description": "Find all vectors within a distance threshold",
            "operationId": "radiusSearch", "tags": ["Search"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("RadiusSearchRequest") },
            "responses": { "200": { "description": "Vectors within radius", "content": schema_ref("SearchResponse") } } }
    }));
    paths.insert("/collections/{collection}/search/graph".into(), serde_json::json!({
        "post": { "summary": "GraphRAG search", "description": "Graph-augmented retrieval combining vector similarity with knowledge graph traversal",
            "operationId": "graphSearch", "tags": ["Search"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("GraphSearchRequest") },
            "responses": { "200": { "description": "Graph search results" } } }
    }));
    paths.insert("/collections/{collection}/search/matryoshka".into(), serde_json::json!({
        "post": { "summary": "Matryoshka two-phase search", "description": "Two-phase search using truncated embeddings for coarse filtering, then full embeddings for re-ranking",
            "operationId": "matryoshkaSearch", "tags": ["Search"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("MatryoshkaSearchRequest") },
            "responses": { "200": { "description": "Search results", "content": schema_ref("SearchResponse") } } }
    }));
}

fn openapi_paths_cache(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/cache/lookup".into(), serde_json::json!({
        "post": { "summary": "Semantic cache lookup", "description": "Look up a cached response by semantic similarity to the query",
            "operationId": "cacheLookup", "tags": ["Semantic Cache"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("CacheLookupRequest") },
            "responses": { "200": { "description": "Cache hit or miss", "content": schema_ref("CacheLookupResponse") } } }
    }));
    paths.insert("/collections/{collection}/cache/store".into(), serde_json::json!({
        "post": { "summary": "Store in semantic cache", "description": "Store a query-response pair in the semantic cache",
            "operationId": "cacheStore", "tags": ["Semantic Cache"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("CacheStoreRequest") },
            "responses": { "200": { "description": "Cached successfully" } } }
    }));
}

fn openapi_paths_streaming(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/ingest".into(), serde_json::json!({
        "post": { "summary": "Streaming ingest", "description": "Stream vectors into a collection via NDJSON",
            "operationId": "streamingIngest", "tags": ["Streaming"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": { "application/x-ndjson": { "schema": { "$ref": "#/components/schemas/InsertVectorRequest" } } } },
            "responses": { "200": { "description": "Ingestion summary", "content": schema_ref("IngestResponse") } } }
    }));
}

fn openapi_paths_time_travel(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/search/time-travel".into(), serde_json::json!({
        "post": { "summary": "Time-travel search", "description": "Search against a historical snapshot of the collection",
            "operationId": "timeTravelSearch", "tags": ["Time Travel"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("TimeTravelSearchRequest") },
            "responses": { "200": { "description": "Search results at specified point in time", "content": schema_ref("SearchResponse") } } }
    }));
    paths.insert("/collections/{collection}/snapshots/diff".into(), serde_json::json!({
        "post": { "summary": "Snapshot diff", "description": "Compare two snapshots and return added/removed/modified vectors",
            "operationId": "snapshotDiff", "tags": ["Time Travel"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("SnapshotDiffRequest") },
            "responses": { "200": { "description": "Diff between snapshots", "content": schema_ref("SnapshotDiffResponse") } } }
    }));
}

fn openapi_paths_memory(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/memory/remember".into(), serde_json::json!({
        "post": { "summary": "Store a memory", "description": "Store context for an LLM agent to recall later",
            "operationId": "remember", "tags": ["Memory"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("RememberRequest") },
            "responses": { "201": { "description": "Memory stored", "content": schema_ref("RememberResponse") } } }
    }));
    paths.insert("/collections/{collection}/memory/recall".into(), serde_json::json!({
        "post": { "summary": "Recall memories", "description": "Retrieve relevant memories by semantic similarity",
            "operationId": "recall", "tags": ["Memory"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("RecallRequest") },
            "responses": { "200": { "description": "Recalled memories", "content": schema_ref("RecallResponse") } } }
    }));
    paths.insert("/collections/{collection}/memory/{memory_id}/forget".into(), serde_json::json!({
        "delete": { "summary": "Forget a memory", "description": "Delete a specific memory by ID",
            "operationId": "forget", "tags": ["Memory"],
            "parameters": [col_param(), { "name": "memory_id", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Memory forgotten" }, "404": { "description": "Memory not found" } } }
    }));
}

fn openapi_paths_analytics(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{collection}/search/estimate".into(), serde_json::json!({
        "post": { "summary": "Query cost estimation", "description": "Estimate the cost of a search query without executing it",
            "operationId": "costEstimate", "tags": ["Analytics"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("SearchRequest") },
            "responses": { "200": { "description": "Estimated cost", "content": schema_ref("CostEstimateResponse") } } }
    }));
    paths.insert("/collections/{collection}/diff".into(), serde_json::json!({
        "post": { "summary": "Vector diff", "description": "Compare two sets of vector IDs and return differences",
            "operationId": "vectorDiff", "tags": ["Analytics"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("VectorDiffRequest") },
            "responses": { "200": { "description": "Diff results" } } }
    }));
    paths.insert("/collections/{collection}/changes".into(), serde_json::json!({
        "get": { "summary": "SSE change feed", "description": "Server-Sent Events stream of collection changes",
            "operationId": "changeFeed", "tags": ["Analytics"], "parameters": [col_param()],
            "responses": { "200": { "description": "Event stream", "content": { "text/event-stream": { "schema": { "type": "string" } } } } } }
    }));
    paths.insert("/collections/{collection}/benchmark".into(), serde_json::json!({
        "post": { "summary": "In-process benchmark", "description": "Run a search benchmark and return latency statistics",
            "operationId": "benchmark", "tags": ["Analytics"], "parameters": [col_param()],
            "requestBody": { "required": true, "content": schema_ref("BenchmarkRequest") },
            "responses": { "200": { "description": "Benchmark results", "content": schema_ref("BenchmarkResponse") } } }
    }));
    paths.insert("/collections/{collection}/index/status".into(), serde_json::json!({
        "get": { "summary": "Index status", "description": "Check HNSW index build status and WAL state",
            "operationId": "indexStatus", "tags": ["Analytics"], "parameters": [col_param()],
            "responses": { "200": { "description": "Index status", "content": schema_ref("IndexStatusResponse") } } }
    }));
}

fn openapi_paths_cluster(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/cluster/status".into(), serde_json::json!({
        "get": { "summary": "Cluster status", "description": "Get cluster topology, shard distribution, and node health",
            "operationId": "clusterStatus", "tags": ["Cluster"],
            "responses": { "200": { "description": "Cluster status", "content": schema_ref("ClusterStatusResponse") } } }
    }));
}

fn openapi_paths_webhooks(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/webhooks".into(), serde_json::json!({
        "post": { "summary": "Create a webhook", "description": "Register a webhook for collection events",
            "operationId": "createWebhook", "tags": ["Webhooks"],
            "requestBody": { "required": true, "content": schema_ref("CreateWebhookRequest") },
            "responses": { "201": { "description": "Webhook created", "content": schema_ref("WebhookResponse") } } },
        "get": { "summary": "List webhooks", "operationId": "listWebhooks", "tags": ["Webhooks"],
            "responses": { "200": { "description": "List of webhooks" } } }
    }));
    paths.insert("/webhooks/{id}".into(), serde_json::json!({
        "delete": { "summary": "Delete a webhook", "operationId": "deleteWebhook", "tags": ["Webhooks"],
            "parameters": [{ "name": "id", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Webhook deleted" }, "404": { "description": "Webhook not found" } } }
    }));
}

fn openapi_paths_aliases(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/aliases".into(), serde_json::json!({
        "post": { "summary": "Create an alias", "description": "Create a named alias pointing to a collection",
            "operationId": "createAlias", "tags": ["Aliases"],
            "requestBody": { "required": true, "content": schema_ref("CreateAliasRequest") },
            "responses": { "201": { "description": "Alias created" }, "409": { "description": "Alias already exists" } } },
        "get": { "summary": "List aliases", "operationId": "listAliases", "tags": ["Aliases"],
            "responses": { "200": { "description": "List of aliases" } } }
    }));
    paths.insert("/aliases/{alias}".into(), serde_json::json!({
        "get": { "summary": "Get alias target", "operationId": "getAlias", "tags": ["Aliases"],
            "parameters": [{ "name": "alias", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Alias details" }, "404": { "description": "Alias not found" } } },
        "delete": { "summary": "Delete an alias", "operationId": "deleteAlias", "tags": ["Aliases"],
            "parameters": [{ "name": "alias", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Alias deleted" }, "404": { "description": "Alias not found" } } },
        "put": { "summary": "Update an alias", "description": "Change the target collection of an alias",
            "operationId": "updateAlias", "tags": ["Aliases"],
            "parameters": [{ "name": "alias", "in": "path", "required": true, "schema": { "type": "string" } }],
            "requestBody": { "required": true, "content": schema_ref("UpdateAliasRequest") },
            "responses": { "200": { "description": "Alias updated" }, "404": { "description": "Alias not found" } } }
    }));
}

fn openapi_paths_ttl(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{name}/expire".into(), serde_json::json!({
        "post": { "summary": "Expire vectors by TTL", "description": "Delete vectors that have exceeded their time-to-live",
            "operationId": "expireVectors", "tags": ["TTL"], "parameters": [name_param()],
            "requestBody": { "required": true, "content": schema_ref("ExpireRequest") },
            "responses": { "200": { "description": "Expired vectors count" } } }
    }));
    paths.insert("/collections/{name}/ttl-stats".into(), serde_json::json!({
        "get": { "summary": "TTL statistics", "description": "Get TTL distribution and expiration statistics",
            "operationId": "ttlStats", "tags": ["TTL"], "parameters": [name_param()],
            "responses": { "200": { "description": "TTL statistics", "content": schema_ref("TtlStatsResponse") } } }
    }));
}

fn openapi_paths_snapshots(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/collections/{name}/snapshots".into(), serde_json::json!({
        "get": { "summary": "List snapshots", "description": "List all point-in-time snapshots for a collection",
            "operationId": "listSnapshots", "tags": ["Snapshots"], "parameters": [name_param()],
            "responses": { "200": { "description": "List of snapshots" } } },
        "post": { "summary": "Create a snapshot", "description": "Create a point-in-time snapshot of the collection",
            "operationId": "createSnapshot", "tags": ["Snapshots"], "parameters": [name_param()],
            "responses": { "201": { "description": "Snapshot created" } } }
    }));
    paths.insert("/collections/{name}/snapshots/{snapshot}/restore".into(), serde_json::json!({
        "post": { "summary": "Restore from snapshot", "description": "Restore a collection to a previous snapshot state",
            "operationId": "restoreSnapshot", "tags": ["Snapshots"],
            "parameters": [name_param(), { "name": "snapshot", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Snapshot restored" }, "404": { "description": "Snapshot not found" } } }
    }));
}

fn openapi_paths_mcp(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/mcp".into(), serde_json::json!({
        "post": { "summary": "MCP tool invocation", "description": "Invoke a Model Context Protocol tool over HTTP",
            "operationId": "mcpInvoke", "tags": ["MCP"],
            "requestBody": { "required": true, "content": schema_ref("McpRequest") },
            "responses": { "200": { "description": "MCP tool result", "content": schema_ref("McpResponse") } } }
    }));
    paths.insert("/mcp/config".into(), serde_json::json!({
        "get": { "summary": "MCP configuration", "description": "Get the MCP server configuration and available tools",
            "operationId": "mcpConfig", "tags": ["MCP"],
            "responses": { "200": { "description": "MCP configuration" } } }
    }));
}

fn openapi_paths_ui(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/dashboard".into(), serde_json::json!({
        "get": { "summary": "Web dashboard", "description": "Serve the web-based admin dashboard UI",
            "operationId": "dashboard", "tags": ["UI"],
            "responses": { "200": { "description": "HTML dashboard page" } } }
    }));
    paths.insert("/playground".into(), serde_json::json!({
        "get": { "summary": "Interactive playground", "description": "Serve the interactive vector search playground",
            "operationId": "playground", "tags": ["UI"],
            "responses": { "200": { "description": "HTML playground page" } } }
    }));
}

fn openapi_paths_observability(paths: &mut serde_json::Map<String, serde_json::Value>) {
    paths.insert("/tracing/status".into(), serde_json::json!({
        "get": { "summary": "Tracing status", "description": "Check OpenTelemetry tracing configuration and export status",
            "operationId": "tracingStatus", "tags": ["Observability"],
            "responses": { "200": { "description": "Tracing configuration" } } }
    }));
    paths.insert("/embeddings/router/status".into(), serde_json::json!({
        "get": { "summary": "Embedding router status", "description": "Get the status of configured embedding model providers",
            "operationId": "embeddingRouterStatus", "tags": ["Observability"],
            "responses": { "200": { "description": "Embedding router status" } } }
    }));
    paths.insert("/metrics".into(), serde_json::json!({
        "get": { "summary": "Prometheus metrics", "description": "Export Prometheus metrics (requires metrics feature)",
            "operationId": "getMetrics", "tags": ["Observability"],
            "responses": { "200": { "description": "Prometheus metrics in text exposition format",
                "content": { "text/plain": { "schema": { "type": "string" } } } } } }
    }));
}

fn openapi_components() -> serde_json::Value {
    let mut components = serde_json::Map::new();
    let mut schemas = serde_json::Map::new();
    openapi_schemas_core(&mut schemas);
    openapi_schemas_search(&mut schemas);
    openapi_schemas_text(&mut schemas);
    openapi_schemas_cache(&mut schemas);
    openapi_schemas_advanced(&mut schemas);
    openapi_schemas_management(&mut schemas);
    components.insert("schemas".into(), serde_json::Value::Object(schemas));
    components.insert("securitySchemes".into(), serde_json::json!({
        "ApiKeyAuth": { "type": "apiKey", "in": "header", "name": "X-API-Key" },
        "BearerAuth": { "type": "http", "scheme": "bearer", "bearerFormat": "JWT" }
    }));
    serde_json::Value::Object(components)
}

fn openapi_schemas_core(s: &mut serde_json::Map<String, serde_json::Value>) {
    s.insert("HealthResponse".into(), serde_json::json!({
        "type": "object", "properties": { "status": { "type": "string", "example": "healthy" } }
    }));
    s.insert("CollectionInfo".into(), serde_json::json!({
        "type": "object", "properties": {
            "name": { "type": "string" }, "dimensions": { "type": "integer" },
            "vector_count": { "type": "integer" },
            "distance_function": { "type": "string", "enum": ["cosine", "euclidean", "dot_product", "manhattan"] }
        }
    }));
    s.insert("CreateCollectionRequest".into(), serde_json::json!({
        "type": "object", "required": ["name", "dimensions"], "properties": {
            "name": { "type": "string" }, "dimensions": { "type": "integer", "minimum": 1 },
            "distance": { "type": "string", "enum": ["cosine", "euclidean", "dot_product", "manhattan"], "default": "cosine" },
            "m": { "type": "integer", "default": 16 }, "ef_construction": { "type": "integer", "default": 200 }
        }
    }));
    s.insert("InsertVectorRequest".into(), serde_json::json!({
        "type": "object", "required": ["id", "vector"], "properties": {
            "id": { "type": "string" },
            "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
            "metadata": { "type": "object", "additionalProperties": true }
        }
    }));
    s.insert("BatchInsertRequest".into(), serde_json::json!({
        "type": "object", "required": ["vectors"], "properties": {
            "vectors": { "type": "array", "items": { "$ref": "#/components/schemas/InsertVectorRequest" } }
        }
    }));
    s.insert("BatchInsertResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "inserted": { "type": "integer" },
            "errors": { "type": "array", "items": { "type": "object", "properties": { "id": { "type": "string" }, "error": { "type": "string" } } } }
        }
    }));
    s.insert("VectorResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "id": { "type": "string" },
            "vector": { "type": "array", "items": { "type": "number" } },
            "metadata": { "type": "object", "nullable": true }
        }
    }));
}

fn openapi_schemas_search(s: &mut serde_json::Map<String, serde_json::Value>) {
    s.insert("SearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["vector"], "properties": {
            "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
            "k": { "type": "integer", "default": 10, "minimum": 1 },
            "filter": { "type": "object", "description": "MongoDB-style metadata filter" },
            "ef_search": { "type": "integer" }
        }
    }));
    s.insert("SearchResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "results": { "type": "array", "items": { "type": "object", "properties": {
                "id": { "type": "string" }, "distance": { "type": "number" }, "metadata": { "type": "object", "nullable": true }
            }}}
        }
    }));
    s.insert("BatchSearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["queries"], "properties": {
            "queries": { "type": "array", "items": { "$ref": "#/components/schemas/SearchRequest" } }
        }
    }));
    s.insert("BatchSearchResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "results": { "type": "array", "items": { "$ref": "#/components/schemas/SearchResponse" } }
        }
    }));
    s.insert("RadiusSearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["vector", "radius"], "properties": {
            "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
            "radius": { "type": "number", "description": "Maximum distance threshold" },
            "filter": { "type": "object", "description": "MongoDB-style metadata filter" },
            "limit": { "type": "integer", "default": 100 }
        }
    }));
    s.insert("GraphSearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["vector"], "properties": {
            "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
            "k": { "type": "integer", "default": 10 },
            "depth": { "type": "integer", "default": 2, "description": "Graph traversal depth" }
        }
    }));
    s.insert("MatryoshkaSearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["vector"], "properties": {
            "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
            "k": { "type": "integer", "default": 10 },
            "coarse_dimensions": { "type": "integer", "description": "Dimensions for first-pass coarse search" }
        }
    }));
}

fn openapi_schemas_text(s: &mut serde_json::Map<String, serde_json::Value>) {
    s.insert("InsertTextRequest".into(), serde_json::json!({
        "type": "object", "required": ["id", "text"], "properties": {
            "id": { "type": "string" }, "text": { "type": "string" },
            "metadata": { "type": "object", "additionalProperties": true }
        }
    }));
    s.insert("BatchInsertTextRequest".into(), serde_json::json!({
        "type": "object", "required": ["documents"], "properties": {
            "documents": { "type": "array", "items": { "$ref": "#/components/schemas/InsertTextRequest" } }
        }
    }));
    s.insert("TextSearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["text"], "properties": {
            "text": { "type": "string" }, "k": { "type": "integer", "default": 10 },
            "filter": { "type": "object", "description": "MongoDB-style metadata filter" }
        }
    }));
}

fn openapi_schemas_cache(s: &mut serde_json::Map<String, serde_json::Value>) {
    s.insert("CacheLookupRequest".into(), serde_json::json!({
        "type": "object", "required": ["query"], "properties": {
            "query": { "type": "string", "description": "The query to look up in the cache" },
            "threshold": { "type": "number", "description": "Similarity threshold for cache hits", "default": 0.95 }
        }
    }));
    s.insert("CacheLookupResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "hit": { "type": "boolean" }, "response": { "type": "string", "nullable": true },
            "similarity": { "type": "number", "nullable": true }
        }
    }));
    s.insert("CacheStoreRequest".into(), serde_json::json!({
        "type": "object", "required": ["query", "response"], "properties": {
            "query": { "type": "string" }, "response": { "type": "string" },
            "ttl_seconds": { "type": "integer", "description": "Optional TTL in seconds" }
        }
    }));
}

fn openapi_schemas_advanced(s: &mut serde_json::Map<String, serde_json::Value>) {
    s.insert("IngestResponse".into(), serde_json::json!({
        "type": "object", "properties": { "inserted": { "type": "integer" }, "errors": { "type": "integer" } }
    }));
    s.insert("TimeTravelSearchRequest".into(), serde_json::json!({
        "type": "object", "required": ["vector", "timestamp"], "properties": {
            "vector": { "type": "array", "items": { "type": "number", "format": "float" } },
            "timestamp": { "type": "string", "format": "date-time", "description": "Point-in-time to search against" },
            "k": { "type": "integer", "default": 10 }
        }
    }));
    s.insert("SnapshotDiffRequest".into(), serde_json::json!({
        "type": "object", "required": ["from", "to"], "properties": {
            "from": { "type": "string", "description": "Source snapshot ID or timestamp" },
            "to": { "type": "string", "description": "Target snapshot ID or timestamp" }
        }
    }));
    s.insert("SnapshotDiffResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "added": { "type": "array", "items": { "type": "string" } },
            "removed": { "type": "array", "items": { "type": "string" } },
            "modified": { "type": "array", "items": { "type": "string" } }
        }
    }));
    s.insert("RememberRequest".into(), serde_json::json!({
        "type": "object", "required": ["content"], "properties": {
            "content": { "type": "string", "description": "Memory content to store" },
            "metadata": { "type": "object", "additionalProperties": true }
        }
    }));
    s.insert("RememberResponse".into(), serde_json::json!({
        "type": "object", "properties": { "memory_id": { "type": "string" } }
    }));
    s.insert("RecallRequest".into(), serde_json::json!({
        "type": "object", "required": ["query"], "properties": {
            "query": { "type": "string" }, "k": { "type": "integer", "default": 5 }
        }
    }));
    s.insert("RecallResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "memories": { "type": "array", "items": { "type": "object", "properties": {
                "id": { "type": "string" }, "content": { "type": "string" }, "similarity": { "type": "number" }
            }}}
        }
    }));
    s.insert("CostEstimateResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "estimated_latency_ms": { "type": "number" }, "estimated_comparisons": { "type": "integer" },
            "index_type": { "type": "string" }
        }
    }));
    s.insert("VectorDiffRequest".into(), serde_json::json!({
        "type": "object", "required": ["ids_a", "ids_b"], "properties": {
            "ids_a": { "type": "array", "items": { "type": "string" } },
            "ids_b": { "type": "array", "items": { "type": "string" } }
        }
    }));
    s.insert("BenchmarkRequest".into(), serde_json::json!({
        "type": "object", "properties": {
            "num_queries": { "type": "integer", "default": 100 },
            "k": { "type": "integer", "default": 10 }, "dimensions": { "type": "integer" }
        }
    }));
    s.insert("BenchmarkResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "qps": { "type": "number", "description": "Queries per second" },
            "mean_latency_ms": { "type": "number" }, "p99_latency_ms": { "type": "number" },
            "recall": { "type": "number" }
        }
    }));
    s.insert("IndexStatusResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "indexed_vectors": { "type": "integer" }, "pending_vectors": { "type": "integer" },
            "index_type": { "type": "string" }, "build_progress": { "type": "number" }
        }
    }));
    s.insert("ClusterStatusResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "nodes": { "type": "integer" }, "shards": { "type": "integer" },
            "status": { "type": "string", "enum": ["healthy", "degraded", "unavailable"] }
        }
    }));
}

fn openapi_schemas_management(s: &mut serde_json::Map<String, serde_json::Value>) {
    s.insert("CreateWebhookRequest".into(), serde_json::json!({
        "type": "object", "required": ["url", "events"], "properties": {
            "url": { "type": "string", "format": "uri" },
            "events": { "type": "array", "items": { "type": "string", "enum": ["insert", "delete", "update", "compact"] } },
            "collection": { "type": "string", "description": "Optional: scope to a specific collection" }
        }
    }));
    s.insert("WebhookResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "id": { "type": "string" }, "url": { "type": "string" },
            "events": { "type": "array", "items": { "type": "string" } }
        }
    }));
    s.insert("CreateAliasRequest".into(), serde_json::json!({
        "type": "object", "required": ["alias", "collection"], "properties": {
            "alias": { "type": "string" }, "collection": { "type": "string" }
        }
    }));
    s.insert("UpdateAliasRequest".into(), serde_json::json!({
        "type": "object", "required": ["collection"], "properties": { "collection": { "type": "string" } }
    }));
    s.insert("ExpireRequest".into(), serde_json::json!({
        "type": "object", "properties": {
            "max_age_seconds": { "type": "integer", "description": "Delete vectors older than this many seconds" }
        }
    }));
    s.insert("TtlStatsResponse".into(), serde_json::json!({
        "type": "object", "properties": {
            "total_with_ttl": { "type": "integer" }, "expired": { "type": "integer" },
            "earliest_expiry": { "type": "string", "format": "date-time", "nullable": true }
        }
    }));
    s.insert("McpRequest".into(), serde_json::json!({
        "type": "object", "required": ["method"], "properties": {
            "method": { "type": "string", "description": "MCP tool method name" },
            "params": { "type": "object", "additionalProperties": true }
        }
    }));
    s.insert("McpResponse".into(), serde_json::json!({
        "type": "object", "properties": { "result": { "type": "object", "additionalProperties": true } }
    }));
}

/// Return the OpenAPI spec as a formatted JSON string.
pub fn openapi_spec_json() -> String {
    serde_json::to_string_pretty(&generate_openapi_spec()).unwrap_or_default()
}


#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    const TEST_API_KEY: &str = "test-key";

    // RateLimitConfig tests
    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert!(config.enabled);
        assert_eq!(config.requests_per_second, 100);
        assert_eq!(config.burst_size, 50);
    }

    #[test]
    fn test_rate_limit_config_disabled() {
        let config = RateLimitConfig::disabled();
        assert!(!config.enabled);
        assert_eq!(config.requests_per_second, 0);
        assert_eq!(config.burst_size, 0);
    }

    #[test]
    fn test_rate_limit_config_with_rate() {
        let config = RateLimitConfig::default().with_rate(200);
        assert_eq!(config.requests_per_second, 200);
        assert_eq!(config.burst_size, 50); // unchanged
    }

    #[test]
    fn test_rate_limit_config_with_burst() {
        let config = RateLimitConfig::default().with_burst(100);
        assert_eq!(config.requests_per_second, 100); // unchanged
        assert_eq!(config.burst_size, 100);
    }

    #[test]
    fn test_rate_limit_config_chained() {
        let config = RateLimitConfig::default().with_rate(500).with_burst(250);
        assert!(config.enabled);
        assert_eq!(config.requests_per_second, 500);
        assert_eq!(config.burst_size, 250);
    }

    // CorsConfig tests
    #[test]
    fn test_cors_config_default() {
        let config = CorsConfig::default();
        assert!(config.enabled);
        assert!(!config.allow_credentials);
        assert_eq!(config.max_age_secs, 3600);
        let origins = config.allowed_origins.unwrap();
        assert!(origins.contains(&"http://localhost:3000".to_string()));
        assert!(origins.contains(&"http://localhost:8080".to_string()));
    }

    #[test]
    fn test_cors_config_permissive() {
        let config = CorsConfig::permissive();
        assert!(config.enabled);
        assert!(!config.allow_credentials); // SECURITY: credentials disabled even in permissive mode
        assert!(config.allowed_origins.is_none()); // Allow all
        assert_eq!(config.max_age_secs, 3600);
    }

    #[test]
    fn test_cors_config_with_credentials() {
        // Credentials should only be used with specific origins
        let config = CorsConfig::default().with_credentials(true);
        assert!(config.allow_credentials);
    }

    #[test]
    fn test_cors_config_restrictive() {
        let config = CorsConfig::restrictive();
        assert!(config.enabled);
        assert!(!config.allow_credentials);
        assert_eq!(config.max_age_secs, 0);
        let origins = config.allowed_origins.unwrap();
        assert!(origins.is_empty()); // No external origins
    }

    #[test]
    fn test_cors_config_with_origin() {
        let config = CorsConfig::restrictive().with_origin("https://example.com");
        let origins = config.allowed_origins.unwrap();
        assert_eq!(origins.len(), 1);
        assert!(origins.contains(&"https://example.com".to_string()));
    }

    #[test]
    fn test_cors_config_with_multiple_origins() {
        let config = CorsConfig::restrictive()
            .with_origin("https://example.com")
            .with_origin("https://api.example.com");
        let origins = config.allowed_origins.unwrap();
        assert_eq!(origins.len(), 2);
    }

    // Error response formatting tests
    #[test]
    fn test_error_collection_not_found() {
        let err = NeedleError::CollectionNotFound("test".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json.code, "COLLECTION_NOT_FOUND");
    }

    #[test]
    fn test_error_vector_not_found() {
        let err = NeedleError::VectorNotFound("vec1".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(json.code, "VECTOR_NOT_FOUND");
    }

    #[test]
    fn test_error_collection_already_exists() {
        let err = NeedleError::CollectionAlreadyExists("test".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(json.code, "COLLECTION_EXISTS");
    }

    #[test]
    fn test_error_vector_already_exists() {
        let err = NeedleError::VectorAlreadyExists("vec1".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(json.code, "VECTOR_EXISTS");
    }

    #[test]
    fn test_error_dimension_mismatch() {
        let err = NeedleError::DimensionMismatch {
            expected: 384,
            got: 128,
        };
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json.code, "DIMENSION_MISMATCH");
        assert!(json.error.contains("384"));
        assert!(json.error.contains("128"));
    }

    #[test]
    fn test_error_invalid_vector() {
        let err = NeedleError::InvalidVector("contains NaN".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json.code, "INVALID_VECTOR");
    }

    #[test]
    fn test_error_invalid_config() {
        let err = NeedleError::InvalidConfig("bad M value".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(json.code, "INVALID_CONFIG");
    }

    #[test]
    fn test_error_internal_fallback() {
        let err = NeedleError::Corruption("data corrupted".to_string());
        let (status, json): (StatusCode, Json<ApiError>) = err.into();
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(json.code, "INTERNAL_ERROR");
    }

    // ServerConfig tests
    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.max_body_size, 10 * 1024 * 1024); // 10MB
        assert!(config.db_path.is_none());
        assert!(config.cors_config.enabled);
        assert!(config.rate_limit.enabled);
    }

    // Authentication tests
    #[test]
    fn test_api_key_creation() {
        let key = ApiKey::new("test-key-123");
        assert_eq!(key.key, "test-key-123");
        assert_eq!(key.roles, vec!["reader".to_string()]);
        assert!(key.active);
        assert!(key.expires_at.is_none());
    }

    #[test]
    fn test_api_key_with_role() {
        let key = ApiKey::new("admin-key").with_role("admin");
        assert_eq!(key.roles, vec!["admin".to_string()]);
    }

    #[test]
    fn test_api_key_with_name() {
        let key = ApiKey::new(TEST_API_KEY).with_name("Production API Key");
        assert_eq!(key.name, Some("Production API Key".to_string()));
    }

    #[test]
    fn test_api_key_expiration() {
        // Key expires in the past
        let expired_key = ApiKey::new("expired").expires_at(0); // Unix epoch = expired
        assert!(expired_key.is_expired());
        assert!(!expired_key.is_valid());

        // Key expires in the future
        let future_ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600; // 1 hour from now
        let valid_key = ApiKey::new("valid").expires_at(future_ts);
        assert!(!valid_key.is_expired());
        assert!(valid_key.is_valid());
    }

    #[test]
    fn test_api_key_to_user() {
        let key = ApiKey::new(TEST_API_KEY)
            .with_name("Test Key")
            .with_role("writer");
        let user = key.to_user();
        assert!(user.id.starts_with("apikey:"));
        assert!(user.name.is_some());
        assert!(!user.roles.is_empty());
    }

    #[test]
    fn test_auth_config_creation() {
        let config = AuthConfig::new();
        assert!(!config.require_auth);
        assert!(config.api_keys.is_empty());
        assert!(config.jwt_secret.is_none());
        // Health and root are public by default
        assert!(config.is_public_endpoint("/health"));
        assert!(config.is_public_endpoint("/"));
    }

    #[test]
    fn test_auth_config_with_api_key() {
        let config = AuthConfig::new().with_api_key(ApiKey::new(TEST_API_KEY));
        assert_eq!(config.api_keys.len(), 1);
    }

    #[test]
    fn test_auth_config_validate_api_key() {
        let config = AuthConfig::new().with_api_key(ApiKey::new("valid-key").with_role("admin"));

        // Valid key
        let user = config.validate_api_key("valid-key");
        assert!(user.is_some());

        // Invalid key
        let invalid = config.validate_api_key("wrong-key");
        assert!(invalid.is_none());
    }

    #[test]
    fn test_auth_config_require_auth() {
        let config = AuthConfig::new().require_auth(true);
        assert!(config.require_auth);
    }

    #[test]
    fn test_auth_config_public_endpoints() {
        let config = AuthConfig::new().with_public_endpoint("/metrics");
        assert!(config.is_public_endpoint("/health"));
        assert!(config.is_public_endpoint("/metrics"));
        assert!(!config.is_public_endpoint("/collections"));
    }

    #[test]
    fn test_jwt_claims_creation() {
        let claims = JwtClaims::new("user123", 3600);
        assert_eq!(claims.sub, "user123");
        assert!(!claims.is_expired());
        assert_eq!(claims.roles, vec!["reader".to_string()]);
    }

    #[test]
    fn test_jwt_claims_with_roles() {
        let claims = JwtClaims::new("admin", 3600)
            .with_roles(vec!["admin".to_string(), "writer".to_string()]);
        assert_eq!(claims.roles.len(), 2);
        assert!(claims.roles.contains(&"admin".to_string()));
    }

    #[test]
    fn test_jwt_claims_to_user() {
        let claims =
            JwtClaims::new("user@example.com", 3600).with_roles(vec!["writer".to_string()]);
        let user = claims.to_user();
        assert_eq!(user.id, "user@example.com");
        assert!(!user.roles.is_empty());
    }

    #[test]
    fn test_jwt_generate_and_validate() {
        let config = AuthConfig::new().with_jwt_secret("super-secret-key-for-testing");

        let claims = JwtClaims::new("testuser", 3600).with_roles(vec!["admin".to_string()]);

        // Generate token
        let token = config.generate_jwt(&claims).expect("should generate token");
        assert!(!token.is_empty());
        assert_eq!(token.split('.').count(), 3); // header.payload.signature

        // Validate token
        let validated = config.validate_jwt(&token).expect("should validate token");
        assert_eq!(validated.sub, "testuser");
        assert!(validated.roles.contains(&"admin".to_string()));
    }

    #[test]
    fn test_jwt_invalid_signature() {
        let config = AuthConfig::new().with_jwt_secret("correct-secret");

        let claims = JwtClaims::new("user", 3600);
        let token = config.generate_jwt(&claims).unwrap();

        // Try validating with a different secret
        let wrong_config = AuthConfig::new().with_jwt_secret("wrong-secret");

        let result = wrong_config.validate_jwt(&token);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_expired_token() {
        let config = AuthConfig::new().with_jwt_secret("secret");

        // Create claims that are already expired (negative expiration)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let mut claims = JwtClaims::new("user", 0);
        claims.exp = now - 100; // Expired 100 seconds ago

        let token = config.generate_jwt(&claims).unwrap();
        let result = config.validate_jwt(&token);
        assert!(matches!(result, Err(AuthError::TokenExpired)));
    }

    #[test]
    fn test_jwt_no_secret_configured() {
        let config = AuthConfig::new(); // No JWT secret
        let claims = JwtClaims::new("user", 3600);

        let result = config.generate_jwt(&claims);
        assert!(matches!(result, Err(AuthError::NoJwtSecret)));
    }

    #[test]
    fn test_auth_method_equality() {
        assert_eq!(AuthMethod::ApiKey, AuthMethod::ApiKey);
        assert_eq!(AuthMethod::Jwt, AuthMethod::Jwt);
        assert_eq!(AuthMethod::None, AuthMethod::None);
        assert_ne!(AuthMethod::ApiKey, AuthMethod::Jwt);
    }

    #[test]
    fn test_auth_error_display() {
        assert_eq!(
            AuthError::MissingCredentials.to_string(),
            "Authentication required"
        );
        assert_eq!(AuthError::InvalidApiKey.to_string(), "Invalid API key");
        assert_eq!(AuthError::TokenExpired.to_string(), "Token has expired");
        assert_eq!(
            AuthError::InvalidSignature.to_string(),
            "Invalid token signature"
        );
    }

    #[test]
    fn test_server_config_with_auth() {
        let auth_config = AuthConfig::new()
            .with_api_key(ApiKey::new(TEST_API_KEY))
            .require_auth(true);

        let server_config = ServerConfig::default().with_auth(auth_config);

        assert!(server_config.auth.require_auth);
        assert_eq!(server_config.auth.api_keys.len(), 1);
    }

    #[test]
    fn test_openapi_spec_generation() {
        let spec = generate_openapi_spec();
        assert_eq!(spec["openapi"], "3.1.0");
        assert_eq!(spec["info"]["title"], "Needle Vector Database API");
        assert!(spec["paths"]["/health"].is_object());
        assert!(spec["paths"]["/collections"].is_object());
        assert!(spec["paths"]["/collections/{collection}/search"].is_object());
        assert!(spec["components"]["schemas"]["SearchRequest"].is_object());
        assert!(spec["components"]["schemas"]["SearchResponse"].is_object());
    }

    #[test]
    fn test_openapi_spec_json_string() {
        let json = openapi_spec_json();
        assert!(json.contains("Needle Vector Database API"));
        assert!(json.contains("searchVectors"));
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert!(parsed.is_object());
    }
}
