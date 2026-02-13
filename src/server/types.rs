//! Request/response types and DTOs for the Needle HTTP server.

use axum::{http::StatusCode, Json};
use crate::error::NeedleError;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use tracing::error;

/// API error response
#[derive(Debug, Serialize)]
pub struct ApiError {
    pub error: String,
    pub code: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub help: String,
}

impl ApiError {
    pub fn new(error: impl Into<String>, code: impl Into<String>) -> Self {
        Self {
            error: error.into(),
            code: code.into(),
            help: String::new(),
        }
    }
}

impl From<NeedleError> for (StatusCode, Json<ApiError>) {
    fn from(err: NeedleError) -> Self {
        let (status, code) = match &err {
            NeedleError::CollectionNotFound(_) => (StatusCode::NOT_FOUND, "COLLECTION_NOT_FOUND"),
            NeedleError::VectorNotFound(_) => (StatusCode::NOT_FOUND, "VECTOR_NOT_FOUND"),
            NeedleError::CollectionAlreadyExists(_) => (StatusCode::CONFLICT, "COLLECTION_EXISTS"),
            NeedleError::VectorAlreadyExists(_) => (StatusCode::CONFLICT, "VECTOR_EXISTS"),
            NeedleError::DimensionMismatch { .. } => {
                (StatusCode::BAD_REQUEST, "DIMENSION_MISMATCH")
            }
            NeedleError::InvalidVector(_) => (StatusCode::BAD_REQUEST, "INVALID_VECTOR"),
            NeedleError::InvalidConfig(_) => (StatusCode::BAD_REQUEST, "INVALID_CONFIG"),
            NeedleError::AliasNotFound(_) => (StatusCode::NOT_FOUND, "ALIAS_NOT_FOUND"),
            NeedleError::AliasAlreadyExists(_) => (StatusCode::CONFLICT, "ALIAS_EXISTS"),
            NeedleError::CollectionHasAliases(_) => {
                (StatusCode::CONFLICT, "COLLECTION_HAS_ALIASES")
            }
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
        };

        let help = err.help();

        // Sanitize internal errors: log full details server-side, return generic message to clients
        let error_message = if status == StatusCode::INTERNAL_SERVER_ERROR {
            error!(error = %err, "Internal server error");
            "An internal error occurred".to_string()
        } else {
            err.to_string()
        };

        (
            status,
            Json(ApiError {
                error: error_message,
                code: code.to_string(),
                help,
            }),
        )
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimensions: usize,
    #[serde(default)]
    pub distance: Option<String>,
    #[serde(default)]
    pub m: Option<usize>,
    #[serde(default)]
    pub ef_construction: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimensions: usize,
    pub count: usize,
    pub deleted_count: usize,
}

#[derive(Debug, Deserialize)]
pub struct InsertRequest {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<Value>,
    /// Optional TTL in seconds; if not provided, uses collection default
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct BatchInsertRequest {
    pub vectors: Vec<InsertRequest>,
}

#[derive(Debug, Deserialize)]
pub struct UpsertRequest {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<Value>,
    /// Optional TTL in seconds; if not provided, uses collection default
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    #[serde(default = "default_k")]
    pub k: usize,
    /// Pre-filter: applied during ANN search (efficient, reduces candidates)
    #[serde(default)]
    pub filter: Option<Value>,
    /// Post-filter: applied after ANN search (guarantees k candidates before filtering)
    #[serde(default)]
    pub post_filter: Option<Value>,
    /// Over-fetch factor for post-filtering (default: 3)
    /// Search fetches k * post_filter_factor candidates before post-filtering
    #[serde(default = "default_post_filter_factor")]
    pub post_filter_factor: usize,
    #[serde(default)]
    pub include_vectors: bool,
    #[serde(default)]
    pub explain: bool,
    /// Override distance function for this query ("cosine", "euclidean", "dot", "manhattan")
    /// When different from the collection's index, uses brute-force search
    #[serde(default)]
    pub distance: Option<String>,
}

fn default_k() -> usize {
    10
}

fn default_post_filter_factor() -> usize {
    3
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<SearchExplanation>,
}

#[derive(Debug, Serialize)]
pub struct SearchResultResponse {
    pub id: String,
    pub distance: f32,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector: Option<Vec<f32>>,
}

#[derive(Debug, Serialize)]
pub struct SearchExplanation {
    pub query_norm: f32,
    pub distance_metric: String,
    pub top_dimensions: Vec<DimensionContribution>,
    /// Detailed profiling data (when available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ProfilingData>,
}

#[derive(Debug, Serialize)]
pub struct ProfilingData {
    /// Total search time in microseconds
    pub total_time_us: u64,
    /// Time spent in HNSW index traversal (microseconds)
    pub index_time_us: u64,
    /// Time spent evaluating metadata filters (microseconds)
    pub filter_time_us: u64,
    /// Time spent enriching results with metadata (microseconds)
    pub enrich_time_us: u64,
    /// Number of candidates before filtering
    pub candidates_before_filter: usize,
    /// Number of candidates after filtering
    pub candidates_after_filter: usize,
    /// HNSW index statistics
    pub hnsw_stats: HnswStatsResponse,
    /// Collection dimensions
    pub dimensions: usize,
    /// Collection vector count
    pub collection_size: usize,
    /// Requested k value
    pub requested_k: usize,
    /// Effective k (clamped to collection size)
    pub effective_k: usize,
    /// ef_search parameter used
    pub ef_search: usize,
    /// Whether a filter was applied
    pub filter_applied: bool,
}

#[derive(Debug, Serialize)]
pub struct HnswStatsResponse {
    /// Number of nodes visited during the search
    pub visited_nodes: usize,
    /// Number of layers traversed (including layer 0)
    pub layers_traversed: usize,
    /// Number of distance computations performed
    pub distance_computations: usize,
    /// Time spent in HNSW traversal (microseconds)
    pub traversal_time_us: u64,
}

#[derive(Debug, Serialize)]
pub struct DimensionContribution {
    pub dimension: usize,
    pub query_value: f32,
    pub contribution: f32,
}

#[derive(Debug, Deserialize)]
pub struct BatchSearchRequest {
    pub vectors: Vec<Vec<f32>>,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub filter: Option<Value>,
}

/// Request for radius-based (range) search
#[derive(Debug, Deserialize)]
pub struct RadiusSearchRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Maximum distance from query (all vectors within this distance are returned)
    pub max_distance: f32,
    /// Maximum number of results to return
    #[serde(default = "default_radius_limit")]
    pub limit: usize,
    /// Optional metadata filter
    #[serde(default)]
    pub filter: Option<Value>,
    /// Include vector data in response
    #[serde(default)]
    pub include_vectors: bool,
}

fn default_radius_limit() -> usize {
    1000
}

#[derive(Debug, Serialize)]
pub struct VectorResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct UpdateMetadataRequest {
    pub metadata: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct QueryParams {
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

// ============ Alias Request/Response Types ============

#[derive(Debug, Deserialize)]
pub struct CreateAliasRequest {
    pub alias: String,
    pub collection: String,
}

#[derive(Debug, Deserialize)]
pub struct UpdateAliasRequest {
    pub collection: String,
}

#[derive(Debug, Serialize)]
pub struct AliasInfo {
    pub alias: String,
    pub collection: String,
}

#[derive(Deserialize)]
pub struct SnapshotRequest {
    pub name: String,
}



/// Request body for text insertion (auto-embed).
///
/// When an embedding provider is configured, this endpoint automatically
/// generates vector embeddings from text before insertion.
#[derive(Deserialize)]
pub struct InsertTextRequest {
    /// Unique ID for the vector
    pub id: String,
    /// Text content to embed
    pub text: String,
    /// Optional metadata
    pub metadata: Option<Value>,
}

/// Batch text insertion request.
#[derive(Deserialize)]
pub struct BatchInsertTextRequest {
    pub texts: Vec<InsertTextRequest>,
}

/// Text search request — search using text instead of a vector.
#[derive(Deserialize)]
pub struct TextSearchRequest {
    /// Query text to embed and search
    pub text: String,
    /// Number of results
    #[serde(default = "default_k")]
    pub k: usize,
    /// Optional metadata filter
    #[serde(default)]
    pub filter: Option<Value>,
}

/// GraphRAG search — combines vector similarity with knowledge graph traversal.
#[derive(Deserialize)]
pub struct GraphSearchRequest {
    /// Query vector embedding
    pub vector: Vec<f32>,
    /// Number of results
    #[serde(default = "default_k")]
    pub k: usize,
    /// Maximum graph traversal hops (default: 2)
    #[serde(default = "default_max_hops")]
    pub max_hops: usize,
}

fn default_max_hops() -> usize { 2 }

/// Matryoshka search request — two-phase dimensional reduction search.
#[derive(Deserialize)]
pub struct MatryoshkaSearchRequest {
    /// Full-dimension query vector
    pub vector: Vec<f32>,
    /// Number of results
    #[serde(default = "default_k")]
    pub k: usize,
    /// Truncated dimension count for coarse search phase
    pub coarse_dims: usize,
    /// Oversampling multiplier for candidate set (default: 4)
    #[serde(default = "default_oversample")]
    pub oversample: usize,
    /// Include vectors in response
    #[serde(default)]
    pub include_vectors: bool,
}

fn default_oversample() -> usize { 4 }

/// Semantic cache lookup request.
#[derive(Deserialize)]
pub struct CacheLookupRequest {
    /// Query vector to find cached responses for
    pub vector: Vec<f32>,
    /// Similarity threshold (0.0-1.0, higher = more strict, default: 0.95)
    #[serde(default = "default_cache_threshold")]
    pub threshold: f32,
}

fn default_cache_threshold() -> f32 { 0.95 }

/// Semantic cache store request.
#[derive(Deserialize)]
pub struct CacheStoreRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Response to cache
    pub response: String,
    /// Optional model name for namespace isolation
    #[serde(default)]
    pub model: Option<String>,
    /// TTL in seconds (default: no expiry)
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

/// Streaming batch insert with backpressure feedback.
#[derive(Deserialize)]
pub struct StreamingInsertRequest {
    /// Batch of vectors to insert
    pub vectors: Vec<StreamingVector>,
    /// Sequence ID for exactly-once dedup (optional)
    #[serde(default)]
    pub sequence_id: Option<String>,
    /// If true, flush to index immediately (slower, more durable)
    #[serde(default)]
    pub flush: bool,
}

#[derive(Deserialize)]
pub struct StreamingVector {
    pub id: String,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub metadata: Option<Value>,
}

/// Time-travel search request — query collection state at a point in time.
#[derive(Deserialize)]
pub struct TimeTravelSearchRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of results
    #[serde(default = "default_k")]
    pub k: usize,
    /// Snapshot name to search against (optional if timestamp provided)
    #[serde(default)]
    pub snapshot: String,
    /// Unix timestamp to search as-of (alternative to snapshot)
    pub as_of_timestamp: Option<u64>,
    /// Version number to search at (alternative to snapshot/timestamp)
    pub as_of_version: Option<u64>,
    /// Natural language time expression (e.g. "yesterday", "2 hours ago")
    pub as_of_expression: Option<String>,
}


/// Snapshot diff — compare collection state between two snapshots.
#[derive(Deserialize)]
pub struct SnapshotDiffRequest {
    /// First snapshot name (or "current" for current state)
    pub from: String,
    /// Second snapshot name (or "current" for current state)
    pub to: String,
}

/// Cost estimation request — predict query cost before execution.
#[derive(Deserialize)]
pub struct CostEstimateRequest {
    /// Query vector (used for dimension validation)
    pub vector: Vec<f32>,
    /// Number of results to return
    #[serde(default = "default_k")]
    pub k: usize,
    /// Optional filter (used to estimate selectivity)
    #[serde(default)]
    pub filter: Option<Value>,
    /// ef_search override (if not specified, uses collection default)
    #[serde(default)]
    pub ef_search: Option<usize>,
}

/// Compare two collections and return differences.
#[derive(Deserialize)]
pub struct VectorDiffRequest {
    /// Name of the second collection to compare against
    pub other_collection: String,
    /// Maximum number of differences to return
    #[serde(default = "default_diff_limit")]
    pub limit: usize,
}

fn default_diff_limit() -> usize { 1000 }

/// Subscribe to collection changes — returns current change stream config.
/// Full SSE streaming requires the `async` feature and a persistent connection.
/// This endpoint provides the change feed metadata and recent events.
#[derive(Deserialize)]
pub struct ChangeStreamQuery {
    /// Maximum events to return (default: 50)
    #[serde(default = "default_change_limit")]
    pub limit: usize,
    /// Resume from event ID (for cursor-based pagination)
    #[serde(default)]
    pub after: Option<u64>,
    /// Filter by event type: "insert", "update", "delete"
    #[serde(default)]
    pub event_type: Option<String>,
}

fn default_change_limit() -> usize { 50 }

/// Run a quick in-process benchmark on the specified collection.
#[derive(Deserialize)]
pub struct BenchmarkRequest {
    /// Number of random queries to run
    #[serde(default = "default_bench_queries")]
    pub num_queries: usize,
    /// k value for search
    #[serde(default = "default_k")]
    pub k: usize,
}

fn default_bench_queries() -> usize { 100 }

/// Remember request — store a memory entry for an AI agent.
#[derive(Deserialize)]
pub struct RememberRequest {
    /// Memory content text
    pub content: String,
    /// Memory vector embedding
    pub vector: Vec<f32>,
    /// Memory tier: "episodic", "semantic", or "procedural"
    #[serde(default = "default_memory_tier")]
    pub tier: String,
    /// Importance score (0.0-1.0)
    #[serde(default = "default_importance")]
    pub importance: f32,
    /// Optional session ID for scoping
    #[serde(default)]
    pub session_id: Option<String>,
    /// Optional metadata
    #[serde(default)]
    pub metadata: Option<Value>,
}

fn default_memory_tier() -> String { "episodic".to_string() }
fn default_importance() -> f32 { 0.5 }

/// Recall request — retrieve relevant memories.
#[derive(Deserialize)]
pub struct RecallRequest {
    /// Query vector
    pub vector: Vec<f32>,
    /// Number of memories to retrieve
    #[serde(default = "default_k")]
    pub k: usize,
    /// Filter by tier
    #[serde(default)]
    pub tier: Option<String>,
    /// Filter by session
    #[serde(default)]
    pub session_id: Option<String>,
    /// Minimum importance threshold
    #[serde(default)]
    pub min_importance: Option<f32>,
}


#[derive(Deserialize)]
pub struct CreateWebhookRequest {
    /// URL to deliver events to
    pub url: String,
    /// Optional HMAC-SHA256 secret for payload signing
    #[serde(default)]
    pub secret: Option<String>,
    /// Collection filter (empty = all collections)
    #[serde(default)]
    pub collections: Vec<String>,
    /// Event type filter (empty = all events)
    #[serde(default)]
    pub event_types: Vec<String>,
}

impl CreateWebhookRequest {
    /// Validate the webhook URL to prevent SSRF attacks.
    /// Only HTTPS URLs are allowed, and private/loopback/link-local IPs are rejected.
    pub fn validate_url(&self) -> Result<(), String> {
        let url = self.url.trim();

        // Scheme check
        if !url.starts_with("https://") {
            return Err("Only HTTPS URLs are allowed for webhooks".to_string());
        }

        // Extract host portion: strip scheme, take up to first '/' or ':'
        let after_scheme = &url["https://".len()..];
        let host_port = after_scheme.split('/').next().unwrap_or("");
        // Strip port if present
        let host = if host_port.starts_with('[') {
            // IPv6: [::1]:8080
            host_port.split(']').next().unwrap_or("").trim_start_matches('[')
        } else {
            host_port.split(':').next().unwrap_or("")
        };

        if host.is_empty() {
            return Err("URL must contain a valid host".to_string());
        }

        // Reject known loopback hostnames
        if host == "localhost" || host == "127.0.0.1" || host == "::1" {
            return Err("Loopback addresses are not allowed".to_string());
        }

        // Parse as IP and reject private ranges
        if let Ok(ip) = host.parse::<std::net::IpAddr>() {
            if is_private_ip(&ip) {
                return Err("Private/internal IP addresses are not allowed".to_string());
            }
        }

        Ok(())
    }
}

/// Check if an IP address is in a private, loopback, or link-local range.
fn is_private_ip(ip: &std::net::IpAddr) -> bool {
    match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()           // 127.0.0.0/8
            || v4.is_private()         // 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16
            || v4.is_link_local()      // 169.254.0.0/16
            || v4.is_unspecified()     // 0.0.0.0
        }
        std::net::IpAddr::V6(v6) => {
            v6.is_loopback()           // ::1
            || v6.is_unspecified()     // ::
        }
    }
}

