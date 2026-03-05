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

/// Request body for creating a new vector collection.
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

/// Summary information about a collection.
#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimensions: usize,
    pub count: usize,
    pub deleted_count: usize,
}

/// Request body for inserting a single vector into a collection.
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

/// Request body for inserting multiple vectors in a single batch.
#[derive(Debug, Deserialize)]
pub struct BatchInsertRequest {
    pub vectors: Vec<InsertRequest>,
}

/// Request body for upserting (insert or update) a single vector.
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

/// Cursor for paginating through search results.
///
/// Use the `next_cursor` from a previous search response to continue from
/// where the last page left off. Results after this (distance, id) pair are returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchCursor {
    /// Distance value of the last result in the previous page.
    pub distance: f32,
    /// ID of the last result in the previous page.
    pub id: String,
}

/// Request body for approximate nearest neighbor search.
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
    /// Cursor for pagination. Pass the `next_cursor` from a previous search response
    /// to get the next page of results.
    #[serde(default)]
    pub search_after: Option<SearchCursor>,
}

fn default_k() -> usize {
    10
}

fn default_post_filter_factor() -> usize {
    3
}

/// Response body for a search operation, containing ranked results and optional explanation.
#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultResponse>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub explanation: Option<SearchExplanation>,
    /// Cursor to pass as `search_after` in the next request to get the next page.
    /// `None` when there are no more results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<SearchCursor>,
    /// Whether more results are available beyond this page.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub has_more: Option<bool>,
}

/// A single search result with distance, score, and optional metadata/vector.
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

/// Detailed explanation of how a search was executed, including profiling data.
#[derive(Debug, Serialize)]
pub struct SearchExplanation {
    pub query_norm: f32,
    pub distance_metric: String,
    pub top_dimensions: Vec<DimensionContribution>,
    /// Detailed profiling data (when available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profiling: Option<ProfilingData>,
}

/// Timing and candidate-count profiling data for a search operation.
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

/// HNSW-specific statistics collected during a search traversal.
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

/// Contribution of a single dimension to the query vector norm.
#[derive(Debug, Serialize)]
pub struct DimensionContribution {
    pub dimension: usize,
    pub query_value: f32,
    pub contribution: f32,
}

/// Request body for searching multiple vectors in a single batch.
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

/// Response body for a single vector retrieval.
#[derive(Debug, Serialize)]
pub struct VectorResponse {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

/// Request body for updating a vector's metadata.
#[derive(Debug, Deserialize)]
pub struct UpdateMetadataRequest {
    pub metadata: Option<Value>,
    /// When true, performs a full replacement of metadata.
    /// When false (default), performs a JSON merge patch: new keys are added,
    /// existing keys are overwritten, keys set to null are removed.
    #[serde(default)]
    pub replace: bool,
}

/// Query string parameters for paginated list endpoints.
#[derive(Debug, Deserialize)]
pub struct QueryParams {
    #[serde(default)]
    pub offset: Option<usize>,
    #[serde(default)]
    pub limit: Option<usize>,
}

/// Generic paginated response envelope for list endpoints.
///
/// Wraps any list response with pagination metadata. Used by endpoints that
/// return collections of items (vectors, collections, etc.).
#[derive(Debug, Serialize)]
pub struct PaginatedResponse<T: Serialize> {
    /// The list of items in this page.
    pub data: T,
    /// Pagination metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pagination: Option<PaginationMeta>,
}

/// Pagination metadata for list responses.
#[derive(Debug, Serialize)]
pub struct PaginationMeta {
    /// Number of items in this page.
    pub count: usize,
    /// Offset of the first item in this page (for offset-based pagination).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset: Option<usize>,
    /// Total number of items (if known).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total: Option<usize>,
    /// Whether there are more items after this page.
    pub has_more: bool,
}

impl<T: Serialize> PaginatedResponse<T> {
    /// Create a paginated response with metadata.
    pub fn new(data: T, count: usize, offset: Option<usize>, total: Option<usize>, has_more: bool) -> Self {
        Self {
            data,
            pagination: Some(PaginationMeta { count, offset, total, has_more }),
        }
    }

    /// Create a simple response without pagination metadata.
    pub fn unpaginated(data: T) -> Self {
        Self { data, pagination: None }
    }
}

// ============ Alias Request/Response Types ============

/// Request body for creating a collection alias.
#[derive(Debug, Deserialize)]
pub struct CreateAliasRequest {
    pub alias: String,
    pub collection: String,
}

/// Request body for updating an alias to point to a different collection.
#[derive(Debug, Deserialize)]
pub struct UpdateAliasRequest {
    pub collection: String,
}

/// Summary information about a collection alias.
#[derive(Debug, Serialize)]
pub struct AliasInfo {
    pub alias: String,
    pub collection: String,
}

/// Request body for creating a collection snapshot.
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

/// A single vector in a streaming batch insert.
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

/// Delta sync request — pull incremental changes from a given LSN.
#[derive(Deserialize)]
pub struct DeltaSyncRequest {
    /// Last LSN the replica has seen. Entries after this LSN will be returned.
    pub from_lsn: u64,
    /// Optional replica identifier for tracking.
    #[serde(default)]
    pub replica_id: Option<String>,
}

/// Delta sync response.
#[derive(Serialize)]
pub struct DeltaSyncResponse {
    /// Response type: "delta", "up_to_date", or "snapshot_required".
    pub status: String,
    /// Starting LSN of the delta (if status == "delta").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub from_lsn: Option<u64>,
    /// Ending LSN of the delta (if status == "delta").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to_lsn: Option<u64>,
    /// Number of entries in the delta.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entry_count: Option<usize>,
    /// Current server LSN.
    pub current_lsn: u64,
    /// Serialised delta entries (JSON array).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entries: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;
    use serde_json::json;

    // ── CreateCollectionRequest deserialization ───────────────────────────

    #[test]
    fn test_create_collection_request_minimal() {
        let json = json!({"name": "test", "dimensions": 128});
        let req: CreateCollectionRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.name, "test");
        assert_eq!(req.dimensions, 128);
        assert!(req.distance.is_none());
        assert!(req.m.is_none());
        assert!(req.ef_construction.is_none());
    }

    #[test]
    fn test_create_collection_request_full() {
        let json = json!({
            "name": "test",
            "dimensions": 384,
            "distance": "cosine",
            "m": 32,
            "ef_construction": 400
        });
        let req: CreateCollectionRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.name, "test");
        assert_eq!(req.dimensions, 384);
        assert_eq!(req.distance.as_deref(), Some("cosine"));
        assert_eq!(req.m, Some(32));
        assert_eq!(req.ef_construction, Some(400));
    }

    #[test]
    fn test_create_collection_request_extra_fields_ignored() {
        let json = json!({"name": "test", "dimensions": 4, "extra_field": "ignored"});
        let req: CreateCollectionRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.name, "test");
    }

    #[test]
    fn test_create_collection_request_missing_required() {
        let json = json!({"name": "test"});
        let result = serde_json::from_value::<CreateCollectionRequest>(json);
        assert!(result.is_err());
    }

    // ── SearchRequest deserialization ─────────────────────────────────────

    #[test]
    fn test_search_request_minimal() {
        let json = json!({"vector": [1.0, 0.0, 0.0]});
        let req: SearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(req.k, 10); // default
        assert!(req.filter.is_none());
        assert!(!req.include_vectors);
        assert!(!req.explain);
    }

    #[test]
    fn test_search_request_all_optional_fields() {
        let json = json!({
            "vector": [1.0, 0.0],
            "k": 5,
            "filter": {"category": "books"},
            "post_filter": {"price": {"$lt": 50}},
            "post_filter_factor": 5,
            "include_vectors": true,
            "explain": true,
            "distance": "euclidean"
        });
        let req: SearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.k, 5);
        assert!(req.filter.is_some());
        assert!(req.post_filter.is_some());
        assert_eq!(req.post_filter_factor, 5);
        assert!(req.include_vectors);
        assert!(req.explain);
        assert_eq!(req.distance.as_deref(), Some("euclidean"));
    }

    // ── ApiError::from(NeedleError) mapping ──────────────────────────────

    #[test]
    fn test_api_error_from_collection_not_found() {
        let err = NeedleError::CollectionNotFound("test".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(api_err.code, "COLLECTION_NOT_FOUND");
    }

    #[test]
    fn test_api_error_from_vector_not_found() {
        let err = NeedleError::VectorNotFound("v1".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(api_err.code, "VECTOR_NOT_FOUND");
    }

    #[test]
    fn test_api_error_from_collection_already_exists() {
        let err = NeedleError::CollectionAlreadyExists("test".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(api_err.code, "COLLECTION_EXISTS");
    }

    #[test]
    fn test_api_error_from_vector_already_exists() {
        let err = NeedleError::VectorAlreadyExists("v1".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(api_err.code, "VECTOR_EXISTS");
    }

    #[test]
    fn test_api_error_from_dimension_mismatch() {
        let err = NeedleError::DimensionMismatch { expected: 128, got: 64 };
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(api_err.code, "DIMENSION_MISMATCH");
    }

    #[test]
    fn test_api_error_from_invalid_vector() {
        let err = NeedleError::InvalidVector("contains NaN".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(api_err.code, "INVALID_VECTOR");
    }

    #[test]
    fn test_api_error_from_invalid_config() {
        let err = NeedleError::InvalidConfig("bad config".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(api_err.code, "INVALID_CONFIG");
    }

    #[test]
    fn test_api_error_from_alias_not_found() {
        let err = NeedleError::AliasNotFound("a1".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_eq!(api_err.code, "ALIAS_NOT_FOUND");
    }

    #[test]
    fn test_api_error_from_alias_already_exists() {
        let err = NeedleError::AliasAlreadyExists("a1".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(api_err.code, "ALIAS_EXISTS");
    }

    #[test]
    fn test_api_error_from_collection_has_aliases() {
        let err = NeedleError::CollectionHasAliases("test".into());
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::CONFLICT);
        assert_eq!(api_err.code, "COLLECTION_HAS_ALIASES");
    }

    #[test]
    fn test_api_error_from_io_error() {
        let err = NeedleError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"));
        let (status, Json(api_err)) = <(StatusCode, Json<ApiError>)>::from(err);
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(api_err.code, "INTERNAL_ERROR");
        assert_eq!(api_err.error, "An internal error occurred");
    }

    // ── InsertRequest / BatchInsertRequest ────────────────────────────────

    #[test]
    fn test_insert_request_deserialization() {
        let json = json!({"id": "v1", "vector": [1.0, 2.0, 3.0]});
        let req: InsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.id, "v1");
        assert_eq!(req.vector, vec![1.0, 2.0, 3.0]);
        assert!(req.metadata.is_none());
    }

    #[test]
    fn test_insert_request_with_metadata() {
        let json = json!({"id": "v1", "vector": [1.0], "metadata": {"key": "val"}});
        let req: InsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert!(req.metadata.is_some());
    }

    #[test]
    fn test_batch_insert_request() {
        let json = json!({
            "vectors": [
                {"id": "v1", "vector": [1.0]},
                {"id": "v2", "vector": [2.0]}
            ]
        });
        let req: BatchInsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vectors.len(), 2);
    }

    #[test]
    fn test_insert_request_empty_vector() {
        let json = json!({"id": "v1", "vector": []});
        let req: InsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert!(req.vector.is_empty());
    }

    // ── SearchResponse serialization round-trip ──────────────────────────

    #[test]
    fn test_search_response_serialization() {
        let resp = SearchResponse {
            results: vec![SearchResultResponse {
                id: "v1".to_string(),
                distance: 0.5,
                score: 0.95,
                metadata: Some(json!({"key": "val"})),
                vector: None,
            }],
            explanation: None,
            next_cursor: None,
            has_more: None,
        };
        let json = serde_json::to_value(&resp).expect("should serialize");
        assert_eq!(json["results"][0]["id"], "v1");
        assert_eq!(json["results"][0]["distance"], 0.5);
        assert!(json.get("explanation").is_none());
        // next_cursor and has_more should be omitted when None
        assert!(json.get("next_cursor").is_none());
        assert!(json.get("has_more").is_none());
    }

    #[test]
    fn test_search_response_with_explanation() {
        let resp = SearchResponse {
            results: vec![],
            explanation: Some(SearchExplanation {
                query_norm: 1.0,
                distance_metric: "cosine".to_string(),
                top_dimensions: vec![],
                profiling: None,
            }),
            next_cursor: None,
            has_more: Some(false),
        };
        let json = serde_json::to_value(&resp).expect("should serialize");
        assert!(json.get("explanation").is_some());
    }

    // ── QueryParams defaults ─────────────────────────────────────────────

    #[test]
    fn test_query_params_defaults() {
        let json = json!({});
        let params: QueryParams = serde_json::from_value(json).expect("should deserialize");
        assert!(params.offset.is_none());
        assert!(params.limit.is_none());
    }

    #[test]
    fn test_query_params_with_values() {
        let json = json!({"offset": 10, "limit": 50});
        let params: QueryParams = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(params.offset, Some(10));
        assert_eq!(params.limit, Some(50));
    }

    // ── RadiusSearchRequest ──────────────────────────────────────────────

    #[test]
    fn test_radius_search_request_defaults() {
        let json = json!({"vector": [1.0, 0.0], "max_distance": 0.5});
        let req: RadiusSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.limit, 1000);
        assert!(!req.include_vectors);
    }

    #[test]
    fn test_radius_search_request_zero_radius() {
        let json = json!({"vector": [1.0], "max_distance": 0.0});
        let req: RadiusSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.max_distance, 0.0);
    }

    #[test]
    fn test_radius_search_request_negative_radius() {
        let json = json!({"vector": [1.0], "max_distance": -1.0});
        let req: RadiusSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.max_distance, -1.0);
    }

    // ── ApiError construction ────────────────────────────────────────────

    #[test]
    fn test_api_error_new() {
        let err = ApiError::new("something went wrong", "BAD_REQUEST");
        assert_eq!(err.error, "something went wrong");
        assert_eq!(err.code, "BAD_REQUEST");
        assert!(err.help.is_empty());
    }

    #[test]
    fn test_api_error_serialization() {
        let err = ApiError::new("test error", "TEST_CODE");
        let json = serde_json::to_value(&err).expect("should serialize");
        assert_eq!(json["error"], "test error");
        assert_eq!(json["code"], "TEST_CODE");
        assert!(json.get("help").is_none());
    }

    // ── CreateWebhookRequest URL validation ──────────────────────────────

    #[test]
    fn test_webhook_url_validation_public() {
        let req = CreateWebhookRequest {
            url: "https://example.com/webhook".to_string(),
            secret: None,
            collections: vec![],
            event_types: vec!["insert".to_string()],
        };
        assert!(req.validate_url().is_ok());
    }

    #[test]
    fn test_webhook_url_validation_localhost_rejected() {
        let req = CreateWebhookRequest {
            url: "http://localhost/webhook".to_string(),
            secret: None,
            collections: vec![],
            event_types: vec!["insert".to_string()],
        };
        assert!(req.validate_url().is_err());
    }

    #[test]
    fn test_webhook_url_validation_private_ip_rejected() {
        let req = CreateWebhookRequest {
            url: "http://192.168.1.1/webhook".to_string(),
            secret: None,
            collections: vec![],
            event_types: vec!["insert".to_string()],
        };
        assert!(req.validate_url().is_err());
    }

    // ── BatchSearchRequest ───────────────────────────────────────────────

    #[test]
    fn test_batch_search_request_deserialization() {
        let json = json!({
            "vectors": [[1.0, 0.0], [0.0, 1.0]],
            "k": 5,
            "filter": {"category": "books"}
        });
        let req: BatchSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vectors.len(), 2);
        assert_eq!(req.k, 5);
        assert!(req.filter.is_some());
    }

    #[test]
    fn test_batch_search_request_defaults() {
        let json = json!({"vectors": [[1.0]]});
        let req: BatchSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.k, 10); // default
        assert!(req.filter.is_none());
    }

    // ── UpsertRequest ────────────────────────────────────────────────────

    #[test]
    fn test_upsert_request_deserialization() {
        let json = json!({
            "id": "u1",
            "vector": [1.0, 2.0, 3.0],
            "metadata": {"key": "val"},
            "ttl_seconds": 3600
        });
        let req: UpsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.id, "u1");
        assert_eq!(req.vector.len(), 3);
        assert!(req.metadata.is_some());
        assert_eq!(req.ttl_seconds, Some(3600));
    }

    #[test]
    fn test_upsert_request_minimal() {
        let json = json!({"id": "u1", "vector": [1.0]});
        let req: UpsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert!(req.metadata.is_none());
        assert!(req.ttl_seconds.is_none());
    }

    // ── StreamingInsertRequest / StreamingVector ─────────────────────────

    #[test]
    fn test_streaming_insert_request() {
        let json = json!({
            "vectors": [
                {"id": "s1", "vector": [1.0, 0.0]},
                {"id": "s2", "vector": [0.0, 1.0], "metadata": {"k": "v"}}
            ],
            "sequence_id": "seq-001",
            "flush": true
        });
        let req: StreamingInsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vectors.len(), 2);
        assert_eq!(req.sequence_id, Some("seq-001".to_string()));
        assert!(req.flush);
    }

    #[test]
    fn test_streaming_insert_defaults() {
        let json = json!({"vectors": []});
        let req: StreamingInsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert!(req.vectors.is_empty());
        assert!(req.sequence_id.is_none());
        assert!(!req.flush);
    }

    // ── GraphSearchRequest ───────────────────────────────────────────────

    #[test]
    fn test_graph_search_request() {
        let json = json!({
            "vector": [1.0, 0.0, 0.0],
            "k": 5,
            "max_hops": 3
        });
        let req: GraphSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vector.len(), 3);
        assert_eq!(req.k, 5);
    }

    // ── MatryoshkaSearchRequest ──────────────────────────────────────────

    #[test]
    fn test_matryoshka_search_request() {
        let json = json!({
            "vector": [1.0, 0.0, 0.0, 0.0],
            "k": 10,
            "coarse_dims": 64,
            "oversample": 4
        });
        let req: MatryoshkaSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vector.len(), 4);
        assert_eq!(req.k, 10);
    }

    // ── CacheLookupRequest / CacheStoreRequest ──────────────────────────

    #[test]
    fn test_cache_lookup_request() {
        let json = json!({"vector": [1.0, 0.0], "threshold": 0.95});
        let req: CacheLookupRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.vector.len(), 2);
    }

    #[test]
    fn test_cache_store_request() {
        let json = json!({
            "vector": [1.0],
            "response": "cached answer",
            "model": "gpt-4",
            "ttl_seconds": 600
        });
        let req: CacheStoreRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.model, Some("gpt-4".to_string()));
    }

    // ── TimeTravelSearchRequest ──────────────────────────────────────────

    #[test]
    fn test_time_travel_search_request() {
        let json = json!({
            "vector": [1.0, 0.0],
            "k": 5,
            "snapshot": "snap_v1"
        });
        let req: TimeTravelSearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.k, 5);
    }

    // ── SnapshotDiffRequest ──────────────────────────────────────────────

    #[test]
    fn test_snapshot_diff_request() {
        let json = json!({"from": "snap1", "to": "snap2"});
        let req: SnapshotDiffRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.from, "snap1");
        assert_eq!(req.to, "snap2");
    }

    // ── BenchmarkRequest ─────────────────────────────────────────────────

    #[test]
    fn test_benchmark_request() {
        let json = json!({"num_queries": 100, "k": 10});
        let req: BenchmarkRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.num_queries, 100);
        assert_eq!(req.k, 10);
    }

    // ── RememberRequest / RecallRequest ──────────────────────────────────

    #[test]
    fn test_remember_request() {
        let json = json!({
            "content": "important fact",
            "vector": [1.0, 0.0, 0.0, 0.0]
        });
        let req: RememberRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.content, "important fact");
    }

    #[test]
    fn test_recall_request() {
        let json = json!({
            "vector": [1.0, 0.0],
            "k": 5
        });
        let req: RecallRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.k, 5);
    }

    // ── VectorResponse serialization ─────────────────────────────────────

    #[test]
    fn test_vector_response_serialization() {
        let resp = VectorResponse {
            id: "v1".to_string(),
            vector: vec![1.0, 0.0],
            metadata: Some(json!({"key": "val"})),
        };
        let json = serde_json::to_value(&resp).expect("should serialize");
        assert_eq!(json["id"], "v1");
        assert_eq!(json["vector"], json!([1.0, 0.0]));
    }

    // ── CollectionInfo serialization ─────────────────────────────────────

    #[test]
    fn test_collection_info_serialization() {
        let info = CollectionInfo {
            name: "test".to_string(),
            dimensions: 128,
            count: 1000,
            deleted_count: 50,
        };
        let json = serde_json::to_value(&info).expect("should serialize");
        assert_eq!(json["name"], "test");
        assert_eq!(json["dimensions"], 128);
        assert_eq!(json["count"], 1000);
        assert_eq!(json["deleted_count"], 50);
    }

    // ── AliasInfo serialization ──────────────────────────────────────────

    #[test]
    fn test_alias_info_serialization() {
        let info = AliasInfo {
            alias: "my_alias".to_string(),
            collection: "my_coll".to_string(),
        };
        let json = serde_json::to_value(&info).expect("should serialize");
        assert_eq!(json["alias"], "my_alias");
        assert_eq!(json["collection"], "my_coll");
    }

    // ── CreateAliasRequest ───────────────────────────────────────────────

    #[test]
    fn test_create_alias_request() {
        let json = json!({"alias": "a1", "collection": "c1"});
        let req: CreateAliasRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.alias, "a1");
        assert_eq!(req.collection, "c1");
    }

    // ── UpdateAliasRequest ───────────────────────────────────────────────

    #[test]
    fn test_update_alias_request() {
        let json = json!({"collection": "new_coll"});
        let req: UpdateAliasRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.collection, "new_coll");
    }

    // ── UpdateMetadataRequest ────────────────────────────────────────────

    #[test]
    fn test_update_metadata_request_null() {
        let json = json!({"metadata": null});
        let req: UpdateMetadataRequest = serde_json::from_value(json).expect("should deserialize");
        assert!(req.metadata.is_none());
    }

    // ── InsertRequest with ttl ───────────────────────────────────────────

    #[test]
    fn test_insert_request_with_ttl() {
        let json = json!({"id": "v1", "vector": [1.0], "ttl_seconds": 300});
        let req: InsertRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.ttl_seconds, Some(300));
    }

    // ── SearchRequest with distance override ─────────────────────────────

    #[test]
    fn test_search_request_with_distance_override() {
        let json = json!({
            "vector": [1.0],
            "distance": "manhattan",
            "k": 3
        });
        let req: SearchRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.distance.as_deref(), Some("manhattan"));
    }

    // ── CostEstimateRequest ──────────────────────────────────────────────

    #[test]
    fn test_cost_estimate_request() {
        let json = json!({
            "vector": [1.0, 0.0],
            "k": 10
        });
        let req: CostEstimateRequest = serde_json::from_value(json).expect("should deserialize");
        assert_eq!(req.k, 10);
    }
}

