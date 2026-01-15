//! HTTP REST API server for Needle
//!
//! Provides a REST API for vector database operations, making Needle
//! accessible from any language or tool that can make HTTP requests.

use crate::database::Database;
use crate::error::NeedleError;
use crate::metadata::Filter;
use axum::{
    body::Body,
    extract::{Path, Query, State},
    http::{header, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{delete, get, post},
    Json, Router,
};
use governor::{
    clock::DefaultClock,
    state::keyed::DashMapStateStore,
    Quota, RateLimiter,
};
use std::net::IpAddr;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::timeout::TimeoutLayer;
use tower_http::trace::TraceLayer;
use tracing::{info, warn, error};

#[cfg(feature = "metrics")]
use crate::metrics::{http_metrics, metrics};

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
    /// Maximum request body size in bytes (default: 100MB)
    pub max_body_size: usize,
    /// Maximum number of items in a batch operation (default: 10000)
    /// Prevents memory exhaustion from large batch requests
    pub max_batch_size: usize,
    /// Request timeout in seconds (default: 30)
    /// Requests exceeding this duration will be terminated
    pub request_timeout_secs: u64,
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
            max_age_secs: 3600,
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
            allowed_origins: None, // Allow all
            allow_credentials: false, // SECURITY: Never combine wildcard origins with credentials
            max_age_secs: 3600,
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

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            addr: "127.0.0.1:8080"
                .parse()
                .unwrap_or_else(|_| SocketAddr::from(([127, 0, 0, 1], 8080))),
            cors_config: CorsConfig::default(),
            db_path: None,
            rate_limit: RateLimitConfig::default(),
            max_body_size: 100 * 1024 * 1024, // 100MB
            max_batch_size: 10_000, // 10k items max per batch
            request_timeout_secs: 30, // 30 seconds default timeout
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
}

/// Per-IP rate limiter type
type PerIpRateLimiter = RateLimiter<IpAddr, DashMapStateStore<IpAddr>, DefaultClock>;

/// Shared application state
pub struct AppState {
    db: RwLock<Database>,
    rate_limiter: Option<Arc<PerIpRateLimiter>>,
    /// Maximum items allowed in batch operations
    max_batch_size: usize,
}

impl AppState {
    /// Create a new AppState with the given database and no rate limiting
    pub fn new(db: Database) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: None,
            max_batch_size: 10_000, // default
        }
    }

    /// Create a new AppState with the given database and rate limiting config
    pub fn with_rate_limit(db: Database, config: &RateLimitConfig) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: create_rate_limiter(config),
            max_batch_size: 10_000, // default
        }
    }

    /// Create a new AppState with full configuration
    pub fn with_config(db: Database, config: &ServerConfig) -> Self {
        Self {
            db: RwLock::new(db),
            rate_limiter: create_rate_limiter(&config.rate_limit),
            max_batch_size: config.max_batch_size,
        }
    }
}

/// API error response
#[derive(Debug, Serialize)]
struct ApiError {
    error: String,
    code: String,
}

impl From<NeedleError> for (StatusCode, Json<ApiError>) {
    fn from(err: NeedleError) -> Self {
        let (status, code) = match &err {
            NeedleError::CollectionNotFound(_) => (StatusCode::NOT_FOUND, "COLLECTION_NOT_FOUND"),
            NeedleError::VectorNotFound(_) => (StatusCode::NOT_FOUND, "VECTOR_NOT_FOUND"),
            NeedleError::CollectionAlreadyExists(_) => (StatusCode::CONFLICT, "COLLECTION_EXISTS"),
            NeedleError::VectorAlreadyExists(_) => (StatusCode::CONFLICT, "VECTOR_EXISTS"),
            NeedleError::DimensionMismatch { .. } => (StatusCode::BAD_REQUEST, "DIMENSION_MISMATCH"),
            NeedleError::InvalidVector(_) => (StatusCode::BAD_REQUEST, "INVALID_VECTOR"),
            NeedleError::InvalidConfig(_) => (StatusCode::BAD_REQUEST, "INVALID_CONFIG"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
        };

        (
            status,
            Json(ApiError {
                error: err.to_string(),
                code: code.to_string(),
            }),
        )
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Deserialize)]
struct CreateCollectionRequest {
    name: String,
    dimensions: usize,
    #[serde(default)]
    distance: Option<String>,
    #[serde(default)]
    m: Option<usize>,
    #[serde(default)]
    ef_construction: Option<usize>,
}

#[derive(Debug, Serialize)]
struct CollectionInfo {
    name: String,
    dimensions: usize,
    count: usize,
    deleted_count: usize,
}

#[derive(Debug, Deserialize)]
struct InsertRequest {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct BatchInsertRequest {
    vectors: Vec<InsertRequest>,
}

#[derive(Debug, Deserialize)]
struct UpsertRequest {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
    #[serde(default)]
    include_vectors: bool,
    #[serde(default)]
    explain: bool,
}

fn default_k() -> usize {
    10
}

#[derive(Debug, Serialize)]
struct SearchResponse {
    results: Vec<SearchResultResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    explanation: Option<SearchExplanation>,
}

#[derive(Debug, Serialize)]
struct SearchResultResponse {
    id: String,
    distance: f32,
    score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
struct SearchExplanation {
    query_norm: f32,
    distance_metric: String,
    top_dimensions: Vec<DimensionContribution>,
}

#[derive(Debug, Serialize)]
struct DimensionContribution {
    dimension: usize,
    query_value: f32,
    contribution: f32,
}

#[derive(Debug, Deserialize)]
struct BatchSearchRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<Value>,
}

#[derive(Debug, Serialize)]
struct VectorResponse {
    id: String,
    vector: Vec<f32>,
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct UpdateMetadataRequest {
    metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct QueryParams {
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

// ============ Handlers ============

/// Health check endpoint
async fn health() -> impl IntoResponse {
    Json(json!({"status": "healthy", "version": env!("CARGO_PKG_VERSION")}))
}

/// Get database info
async fn get_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    Json(json!({
        "collections": collections.len(),
        "total_vectors": db.total_vectors(),
    }))
}

/// List all collections
async fn list_collections(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections: Vec<CollectionInfo> = db
        .list_collections()
        .into_iter()
        .filter_map(|name| {
            let coll = db.collection(&name).ok()?;
            Some(CollectionInfo {
                name,
                dimensions: coll.dimensions().unwrap_or(0),
                count: coll.len(),
                deleted_count: coll.deleted_count(),
            })
        })
        .collect();

    Json(json!({"collections": collections}))
}

/// Create a new collection
async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;

    let mut config = crate::CollectionConfig::new(&req.name, req.dimensions);

    if let Some(dist) = &req.distance {
        config = config.with_distance(match dist.to_lowercase().as_str() {
            "euclidean" | "l2" => crate::DistanceFunction::Euclidean,
            "dot" | "dotproduct" => crate::DistanceFunction::DotProduct,
            "manhattan" | "l1" => crate::DistanceFunction::Manhattan,
            _ => crate::DistanceFunction::Cosine,
        });
    }

    if let Some(m) = req.m {
        config = config.with_m(m);
    }

    if let Some(ef) = req.ef_construction {
        config = config.with_ef_construction(ef);
    }

    db.create_collection_with_config(config).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({"created": req.name})),
    ))
}

/// Get collection info
async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db.collection(&name).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "name": name,
        "dimensions": coll.dimensions(),
        "count": coll.len(),
        "deleted_count": coll.deleted_count(),
        "needs_compaction": coll.needs_compaction(0.2),
    })))
}

/// Delete a collection
async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let dropped = db.drop_collection(&name).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if dropped {
        Ok(Json(json!({"deleted": name})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: format!("Collection '{}' not found", name),
                code: "COLLECTION_NOT_FOUND".to_string(),
            }),
        ))
    }
}

/// Insert a vector
async fn insert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    coll.insert(&req.id, &req.vector, req.metadata)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"inserted": req.id}))))
}

/// Batch insert vectors
async fn batch_insert(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    // Validate batch size to prevent memory exhaustion
    if req.vectors.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError {
                error: format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.vectors.len(),
                    state.max_batch_size
                ),
                code: "BATCH_TOO_LARGE".to_string(),
            }),
        ));
    }

    let db = state.db.write().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let mut inserted = 0;
    let mut errors = Vec::new();

    for item in req.vectors {
        match coll.insert(&item.id, &item.vector, item.metadata) {
            Ok(_) => inserted += 1,
            Err(e) => errors.push(json!({"id": item.id, "error": e.to_string()})),
        }
    }

    Ok(Json(json!({
        "inserted": inserted,
        "errors": errors,
    })))
}

/// Upsert a vector
async fn upsert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Check if exists and update or insert
    let existed = if coll.get(&req.id).is_some() {
        coll.delete(&req.id).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        true
    } else {
        false
    };

    coll.insert(&req.id, &req.vector, req.metadata)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "id": req.id,
        "updated": existed,
    })))
}

/// Get a vector by ID
async fn get_vector(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    match coll.get(&id) {
        Some((vector, metadata)) => Ok(Json(VectorResponse {
            id,
            vector,
            metadata,
        })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: format!("Vector '{}' not found", id),
                code: "VECTOR_NOT_FOUND".to_string(),
            }),
        )),
    }
}

/// Delete a vector
async fn delete_vector(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let deleted = coll.delete(&id).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if deleted {
        Ok(Json(json!({"deleted": id})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: format!("Vector '{}' not found", id),
                code: "VECTOR_NOT_FOUND".to_string(),
            }),
        ))
    }
}

/// Update vector metadata
async fn update_metadata(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
    Json(req): Json<UpdateMetadataRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Get existing vector
    let (vector, _) = coll.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError {
                error: format!("Vector '{}' not found", id),
                code: "VECTOR_NOT_FOUND".to_string(),
            }),
        )
    })?;

    // Delete and re-insert with new metadata
    coll.delete(&id).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    coll.insert(&id, &vector, req.metadata)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({"updated": id})))
}

/// Search for similar vectors
async fn search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let raw_results = if let Some(filter_value) = &req.filter {
        let filter = Filter::parse(filter_value)
            .map_err(|e| (StatusCode::BAD_REQUEST, Json(ApiError { error: format!("Invalid filter: {}", e), code: "INVALID_FILTER".to_string() })))?;
        coll.search_with_filter(&req.vector, req.k, &filter)
    } else {
        coll.search(&req.vector, req.k)
    }
    .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Convert to response format with optional vectors
    let results: Vec<SearchResultResponse> = raw_results
        .into_iter()
        .map(|r| {
            let vector = if req.include_vectors {
                coll.get(&r.id).map(|(v, _)| v)
            } else {
                None
            };

            SearchResultResponse {
                id: r.id,
                distance: r.distance,
                score: 1.0 / (1.0 + r.distance), // Convert distance to similarity score
                metadata: r.metadata,
                vector,
            }
        })
        .collect();

    // Generate explanation if requested
    let explanation = if req.explain {
        let query_norm: f32 = req.vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Find dimensions that contribute most to similarity
        let mut contributions: Vec<(usize, f32, f32)> = req
            .vector
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v, v.abs()))
            .collect();
        contributions.sort_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        });

        let top_dims: Vec<DimensionContribution> = contributions
            .into_iter()
            .take(10)
            .map(|(dim, val, contrib)| DimensionContribution {
                dimension: dim,
                query_value: val,
                contribution: contrib / query_norm,
            })
            .collect();

        Some(SearchExplanation {
            query_norm,
            distance_metric: "cosine".to_string(),
            top_dimensions: top_dims,
        })
    } else {
        None
    };

    Ok(Json(SearchResponse {
        results,
        explanation,
    }))
}

/// Batch search
async fn batch_search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    // Validate batch size to prevent memory exhaustion
    if req.vectors.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError {
                error: format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.vectors.len(),
                    state.max_batch_size
                ),
                code: "BATCH_TOO_LARGE".to_string(),
            }),
        ));
    }

    let db = state.db.read().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let all_results: Vec<Vec<SearchResultResponse>> = if let Some(filter_value) = &req.filter {
        let filter = Filter::parse(filter_value)
            .map_err(|e| (StatusCode::BAD_REQUEST, Json(ApiError { error: format!("Invalid filter: {}", e), code: "INVALID_FILTER".to_string() })))?;

        let mut results = Vec::new();
        for query in &req.vectors {
            let r = coll
                .search_with_filter(query, req.k, &filter)
                .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
            results.push(r);
        }
        results
    } else {
        req.vectors
            .iter()
            .map(|q| coll.search(q, req.k))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?
    }
    .into_iter()
    .map(|results| {
        results
            .into_iter()
            .map(|r| SearchResultResponse {
                id: r.id,
                distance: r.distance,
                score: 1.0 / (1.0 + r.distance),
                metadata: r.metadata,
                vector: None,
            })
            .collect()
    })
    .collect();

    Ok(Json(json!({"results": all_results})))
}

/// Compact a collection
async fn compact_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let removed = coll.compact().map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "compacted": collection,
        "removed": removed,
    })))
}

/// List vector IDs in a collection
async fn list_vectors(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Query(params): Query<QueryParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let ids = coll.ids().map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(100).min(1000);

    let page: Vec<String> = ids.into_iter().skip(offset).take(limit).collect();

    Ok(Json(json!({
        "ids": page,
        "offset": offset,
        "limit": limit,
        "total": coll.len(),
    })))
}

/// Export collection
async fn export_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db.collection(&collection).map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let vectors = coll.export_all().map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let export: Vec<Value> = vectors
        .into_iter()
        .map(|(id, vec, meta)| {
            json!({
                "id": id,
                "vector": vec,
                "metadata": meta,
            })
        })
        .collect();

    Ok(Json(json!({
        "collection": collection,
        "dimensions": coll.dimensions(),
        "count": export.len(),
        "vectors": export,
    })))
}

/// Save database to disk
async fn save_database(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let mut db = state.db.write().await;
    db.save().map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    Ok(Json(json!({"saved": true})))
}

/// Build CORS layer from configuration
fn build_cors_layer(config: &CorsConfig) -> CorsLayer {
    if !config.enabled {
        return CorsLayer::new();
    }

    let mut cors = CorsLayer::new()
        .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::PUT, Method::OPTIONS])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
        .max_age(std::time::Duration::from_secs(config.max_age_secs));

    // Set allowed origins
    cors = match &config.allowed_origins {
        None => cors.allow_origin(AllowOrigin::any()),
        Some(origins) if origins.is_empty() => cors,
        Some(origins) => {
            let origins: Vec<_> = origins
                .iter()
                .filter_map(|o| o.parse().ok())
                .collect();
            cors.allow_origin(origins)
        }
    };

    if config.allow_credentials {
        cors = cors.allow_credentials(true);
    }

    cors
}

/// Build the router with configuration
#[allow(deprecated)]
pub fn create_router_with_config(state: Arc<AppState>, config: &ServerConfig) -> Router {
    let cors_layer = build_cors_layer(&config.cors_config);
    let timeout_layer = TimeoutLayer::new(Duration::from_secs(config.request_timeout_secs));

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
        // Search
        .route("/collections/:collection/search", post(search))
        .route("/collections/:collection/search/batch", post(batch_search))
        // Database operations
        .route("/save", post(save_database));

    // Add metrics endpoint when metrics feature is enabled
    #[cfg(feature = "metrics")]
    {
        router = router.route("/metrics", get(get_metrics));
    }

    // Apply rate limiting and state
    let router = router
        .layer(middleware::from_fn_with_state(state.clone(), rate_limit_middleware))
        .with_state(state);

    // Build final router with all layers
    // Note: metrics middleware is applied here (outermost) so it captures all requests
    #[cfg(feature = "metrics")]
    let router = router.layer(middleware::from_fn(metrics_middleware));

    router
        .layer(TraceLayer::new_for_http())
        .layer(timeout_layer)
        .layer(RequestBodyLimitLayer::new(config.max_body_size))
        .layer(cors_layer)
}

/// Build the router (legacy, uses permissive CORS for backwards compatibility)
pub fn create_router(state: Arc<AppState>) -> Router {
    let config = ServerConfig {
        cors_config: CorsConfig::permissive(),
        ..Default::default()
    };
    create_router_with_config(state, &config)
}

/// Create a per-IP rate limiter from configuration
fn create_rate_limiter(config: &RateLimitConfig) -> Option<Arc<PerIpRateLimiter>> {
    if !config.enabled || config.requests_per_second == 0 {
        return None;
    }

    // Create quota: requests_per_second with burst_size burst capacity
    let quota = Quota::per_second(
        NonZeroU32::new(config.requests_per_second).unwrap_or(NonZeroU32::new(100).unwrap()),
    )
    .allow_burst(
        NonZeroU32::new(config.burst_size).unwrap_or(NonZeroU32::new(1).unwrap()),
    );

    Some(Arc::new(RateLimiter::dashmap(quota)))
}

/// Extract client IP from request, checking X-Forwarded-For header first
fn extract_client_ip(request: &Request<Body>) -> IpAddr {
    // Check X-Forwarded-For header first (for proxied requests)
    if let Some(forwarded_for) = request.headers().get("x-forwarded-for") {
        if let Ok(value) = forwarded_for.to_str() {
            // X-Forwarded-For can contain multiple IPs, take the first one
            if let Some(first_ip) = value.split(',').next() {
                if let Ok(ip) = first_ip.trim().parse::<IpAddr>() {
                    return ip;
                }
            }
        }
    }

    // Check X-Real-IP header
    if let Some(real_ip) = request.headers().get("x-real-ip") {
        if let Ok(value) = real_ip.to_str() {
            if let Ok(ip) = value.trim().parse::<IpAddr>() {
                return ip;
            }
        }
    }

    // Fallback to localhost if no IP found
    IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))
}

/// Rate limiting middleware - returns 429 if rate limit exceeded (per-IP)
async fn rate_limit_middleware(
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if let Some(limiter) = &state.rate_limiter {
        let client_ip = extract_client_ip(&request);
        match limiter.check_key(&client_ip) {
            Ok(_) => next.run(request).await,
            Err(_) => {
                warn!(
                    client_ip = %client_ip,
                    "Rate limit exceeded"
                );
                let error = ApiError {
                    error: "Rate limit exceeded. Please slow down.".to_string(),
                    code: "RATE_LIMIT_EXCEEDED".to_string(),
                };
                (StatusCode::TOO_MANY_REQUESTS, Json(error)).into_response()
            }
        }
    } else {
        next.run(request).await
    }
}

/// Metrics middleware - records HTTP request metrics (when metrics feature is enabled)
#[cfg(feature = "metrics")]
async fn metrics_middleware(
    request: Request<Body>,
    next: Next,
) -> Response {
    let method = request.method().to_string();
    let path = request.uri().path().to_string();

    let mut timer = http_metrics().start_request(&method, &path);
    let response = next.run(request).await;
    timer.set_status(response.status().as_u16());

    response
}

/// Handler for /metrics endpoint - exports Prometheus metrics
#[cfg(feature = "metrics")]
async fn get_metrics() -> impl IntoResponse {
    let output = metrics().export();
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")],
        output,
    )
}

/// Start the HTTP server
pub async fn serve(config: ServerConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing subscriber with environment filter
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into())
        )
        .init();

    info!(
        addr = %config.addr,
        timeout_secs = config.request_timeout_secs,
        rate_limit_enabled = config.rate_limit.enabled,
        "Starting Needle server"
    );

    let db = if let Some(path) = &config.db_path {
        info!(path = %path, "Opening database file");
        Database::open(path)?
    } else {
        info!("Using in-memory database");
        Database::in_memory()
    };

    let state = Arc::new(AppState::with_config(db, &config));
    let app = create_router_with_config(state.clone(), &config);

    info!("Listening on http://{}", config.addr);

    let listener = tokio::net::TcpListener::bind(&config.addr).await?;

    // Graceful shutdown handling
    let shutdown_signal = async {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("Failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("Failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {},
            _ = terminate => {},
        }

        info!("Shutdown signal received, starting graceful shutdown");
    };

    axum::serve(listener, app)
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

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

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
        let config = RateLimitConfig::default()
            .with_rate(500)
            .with_burst(250);
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
        let config = CorsConfig::restrictive()
            .with_origin("https://example.com");
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
        let err = NeedleError::DimensionMismatch { expected: 384, got: 128 };
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
        assert_eq!(config.max_body_size, 100 * 1024 * 1024); // 100MB
        assert!(config.db_path.is_none());
        assert!(config.cors_config.enabled);
        assert!(config.rate_limit.enabled);
    }
}
