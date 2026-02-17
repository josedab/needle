//! gRPC Service Schema
//!
//! Protobuf-equivalent type definitions and service traits for a gRPC API layer.
//! These types mirror what a `.proto` file would generate, enabling typed
//! client/server implementations once a gRPC runtime (tonic) is integrated.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::grpc_schema::{
//!     NeedleService, SearchRequest, SearchResponse, VectorResult,
//!     CreateCollectionRequest, InsertRequest,
//! };
//!
//! // Implement the service trait for your handler
//! struct MyHandler;
//! impl NeedleService for MyHandler {
//!     fn create_collection(&self, req: CreateCollectionRequest) -> GrpcResult<CreateCollectionResponse> {
//!         Ok(CreateCollectionResponse { success: true, message: "Created".into() })
//!     }
//!     // ... implement other methods
//! #   fn insert(&self, req: InsertRequest) -> GrpcResult<InsertResponse> { unimplemented!() }
//! #   fn search(&self, req: SearchRequest) -> GrpcResult<SearchResponse> { unimplemented!() }
//! #   fn get(&self, req: GetRequest) -> GrpcResult<GetResponse> { unimplemented!() }
//! #   fn delete(&self, req: DeleteRequest) -> GrpcResult<DeleteResponse> { unimplemented!() }
//! #   fn list_collections(&self, req: ListCollectionsRequest) -> GrpcResult<ListCollectionsResponse> { unimplemented!() }
//! #   fn batch_insert(&self, req: BatchInsertRequest) -> GrpcResult<BatchInsertResponse> { unimplemented!() }
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// gRPC status code (mirrors grpc-status).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GrpcStatus {
    Ok = 0,
    InvalidArgument = 3,
    NotFound = 5,
    AlreadyExists = 6,
    Internal = 13,
}

/// gRPC error type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpcError {
    pub code: GrpcStatus,
    pub message: String,
}

impl std::fmt::Display for GrpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "gRPC {:?}: {}", self.code, self.message)
    }
}

/// gRPC result type.
pub type GrpcResult<T> = std::result::Result<T, GrpcError>;

// ── Request/Response Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimensions: u32,
    pub distance: String,
    pub hnsw_m: Option<u32>,
    pub hnsw_ef_construction: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateCollectionResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertRequest {
    pub collection: String,
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResponse {
    pub success: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertRequest {
    pub collection: String,
    pub vectors: Vec<VectorEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertResponse {
    pub inserted: u32,
    pub failed: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub collection: String,
    pub query: Vec<f32>,
    pub k: u32,
    pub ef_search: Option<u32>,
    pub filter: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<VectorResult>,
    pub latency_us: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetRequest {
    pub collection: String,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GetResponse {
    pub found: bool,
    pub vector: Vec<f32>,
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub collection: String,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteResponse {
    pub deleted: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListCollectionsRequest {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimensions: u32,
    pub vector_count: u64,
    pub distance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ListCollectionsResponse {
    pub collections: Vec<CollectionInfo>,
}

// ── Service Trait ────────────────────────────────────────────────────────────

/// Trait defining the Needle gRPC service API.
/// Implement this to create a gRPC server handler.
pub trait NeedleService {
    fn create_collection(&self, req: CreateCollectionRequest) -> GrpcResult<CreateCollectionResponse>;
    fn insert(&self, req: InsertRequest) -> GrpcResult<InsertResponse>;
    fn batch_insert(&self, req: BatchInsertRequest) -> GrpcResult<BatchInsertResponse>;
    fn search(&self, req: SearchRequest) -> GrpcResult<SearchResponse>;
    fn get(&self, req: GetRequest) -> GrpcResult<GetResponse>;
    fn delete(&self, req: DeleteRequest) -> GrpcResult<DeleteResponse>;
    fn list_collections(&self, req: ListCollectionsRequest) -> GrpcResult<ListCollectionsResponse>;
}

// ── In-Memory Implementation ────────────────────────────────────────────────

/// In-memory implementation of the gRPC service backed by a Database.
pub struct DatabaseService<'a> {
    db: &'a crate::database::Database,
}

impl<'a> DatabaseService<'a> {
    pub fn new(db: &'a crate::database::Database) -> Self {
        Self { db }
    }
}

impl<'a> NeedleService for DatabaseService<'a> {
    fn create_collection(&self, req: CreateCollectionRequest) -> GrpcResult<CreateCollectionResponse> {
        self.db
            .create_collection(&req.name, req.dimensions as usize)
            .map(|_| CreateCollectionResponse {
                success: true,
                message: format!("Collection '{}' created", req.name),
            })
            .map_err(|e| GrpcError {
                code: GrpcStatus::Internal,
                message: e.to_string(),
            })
    }

    fn insert(&self, req: InsertRequest) -> GrpcResult<InsertResponse> {
        let coll = self.db.collection(&req.collection).map_err(|e| GrpcError {
            code: GrpcStatus::NotFound,
            message: e.to_string(),
        })?;
        coll.insert(&req.id, &req.vector, req.metadata)
            .map(|_| InsertResponse { success: true })
            .map_err(|e| GrpcError {
                code: GrpcStatus::Internal,
                message: e.to_string(),
            })
    }

    fn batch_insert(&self, req: BatchInsertRequest) -> GrpcResult<BatchInsertResponse> {
        let coll = self.db.collection(&req.collection).map_err(|e| GrpcError {
            code: GrpcStatus::NotFound,
            message: e.to_string(),
        })?;
        let mut inserted = 0u32;
        let mut failed = 0u32;
        for entry in &req.vectors {
            match coll.insert(&entry.id, &entry.vector, entry.metadata.clone()) {
                Ok(_) => inserted += 1,
                Err(_) => failed += 1,
            }
        }
        Ok(BatchInsertResponse { inserted, failed })
    }

    fn search(&self, req: SearchRequest) -> GrpcResult<SearchResponse> {
        let start = std::time::Instant::now();
        let coll = self.db.collection(&req.collection).map_err(|e| GrpcError {
            code: GrpcStatus::NotFound,
            message: e.to_string(),
        })?;
        let results = coll.search(&req.query, req.k as usize).map_err(|e| GrpcError {
            code: GrpcStatus::Internal,
            message: e.to_string(),
        })?;
        Ok(SearchResponse {
            results: results
                .into_iter()
                .map(|r| VectorResult {
                    id: r.id,
                    distance: r.distance,
                    metadata: r.metadata,
                })
                .collect(),
            latency_us: start.elapsed().as_micros() as u64,
        })
    }

    fn get(&self, req: GetRequest) -> GrpcResult<GetResponse> {
        let coll = self.db.collection(&req.collection).map_err(|e| GrpcError {
            code: GrpcStatus::NotFound,
            message: e.to_string(),
        })?;
        match coll.get(&req.id) {
            Some((vec, meta)) => Ok(GetResponse {
                found: true,
                vector: vec.to_vec(),
                metadata: meta.map(|v| v.clone()),
            }),
            None => Ok(GetResponse {
                found: false,
                vector: Vec::new(),
                metadata: None,
            }),
        }
    }

    fn delete(&self, req: DeleteRequest) -> GrpcResult<DeleteResponse> {
        let coll = self.db.collection(&req.collection).map_err(|e| GrpcError {
            code: GrpcStatus::NotFound,
            message: e.to_string(),
        })?;
        let deleted = coll.delete(&req.id).map_err(|e| GrpcError {
            code: GrpcStatus::Internal,
            message: e.to_string(),
        })?;
        Ok(DeleteResponse { deleted })
    }

    fn list_collections(&self, _req: ListCollectionsRequest) -> GrpcResult<ListCollectionsResponse> {
        let names = self.db.list_collections();
        let mut collections = Vec::new();
        for name in names {
            if let Ok(coll) = self.db.collection(&name) {
                collections.push(CollectionInfo {
                    name,
                    dimensions: coll.dimensions().unwrap_or(0) as u32,
                    vector_count: coll.len() as u64,
                    distance: "cosine".into(),
                });
            }
        }
        Ok(ListCollectionsResponse { collections })
    }
}

// ── Streaming Support ───────────────────────────────────────────────────────

/// A streaming insert request chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInsertChunk {
    pub collection: String,
    pub vectors: Vec<VectorEntry>,
}

/// Response for a streaming insert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamInsertResponse {
    pub total_inserted: u64,
    pub total_failed: u64,
    pub chunks_processed: u64,
}

/// A streaming search request (for bidirectional streaming).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSearchRequest {
    pub collection: String,
    pub query: Vec<f32>,
    pub k: u32,
    pub request_id: String,
}

/// A streaming search response (sent back per query in the stream).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamSearchResponse {
    pub request_id: String,
    pub results: Vec<VectorResult>,
    pub latency_us: u64,
}

/// Streaming gRPC service trait for bidirectional streaming operations.
pub trait StreamingNeedleService: NeedleService {
    /// Process a stream of insert chunks and return aggregate results.
    fn streaming_insert(&self, chunks: Vec<StreamInsertChunk>) -> GrpcResult<StreamInsertResponse>;

    /// Process a stream of search requests and return per-query results.
    fn streaming_search(
        &self,
        requests: Vec<StreamSearchRequest>,
    ) -> GrpcResult<Vec<StreamSearchResponse>>;
}

impl<'a> StreamingNeedleService for DatabaseService<'a> {
    fn streaming_insert(&self, chunks: Vec<StreamInsertChunk>) -> GrpcResult<StreamInsertResponse> {
        let mut total_inserted = 0u64;
        let mut total_failed = 0u64;
        let mut chunks_processed = 0u64;

        for chunk in chunks {
            let coll = self.db.collection(&chunk.collection).map_err(|e| GrpcError {
                code: GrpcStatus::NotFound,
                message: e.to_string(),
            })?;

            for entry in &chunk.vectors {
                match coll.insert(&entry.id, &entry.vector, entry.metadata.clone()) {
                    Ok(_) => total_inserted += 1,
                    Err(_) => total_failed += 1,
                }
            }
            chunks_processed += 1;
        }

        Ok(StreamInsertResponse {
            total_inserted,
            total_failed,
            chunks_processed,
        })
    }

    fn streaming_search(
        &self,
        requests: Vec<StreamSearchRequest>,
    ) -> GrpcResult<Vec<StreamSearchResponse>> {
        let mut responses = Vec::with_capacity(requests.len());

        for req in requests {
            let start = std::time::Instant::now();
            let coll = self.db.collection(&req.collection).map_err(|e| GrpcError {
                code: GrpcStatus::NotFound,
                message: e.to_string(),
            })?;

            let results = coll
                .search(&req.query, req.k as usize)
                .map_err(|e| GrpcError {
                    code: GrpcStatus::Internal,
                    message: e.to_string(),
                })?;

            responses.push(StreamSearchResponse {
                request_id: req.request_id,
                results: results
                    .into_iter()
                    .map(|r| VectorResult {
                        id: r.id,
                        distance: r.distance,
                        metadata: r.metadata,
                    })
                    .collect(),
                latency_us: start.elapsed().as_micros() as u64,
            });
        }

        Ok(responses)
    }
}

// ── Connection Context ──────────────────────────────────────────────────────

/// gRPC connection context carrying authentication and request metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConnectionContext {
    /// Authenticated user or service identity.
    pub identity: Option<String>,
    /// Bearer token from metadata.
    pub auth_token: Option<String>,
    /// Request ID for distributed tracing.
    pub request_id: Option<String>,
    /// Client-specified timeout in milliseconds.
    pub timeout_ms: Option<u64>,
    /// Arbitrary metadata key-value pairs from gRPC headers.
    pub metadata: HashMap<String, String>,
}

impl ConnectionContext {
    /// Create a context from gRPC-style metadata headers.
    pub fn from_metadata(headers: &HashMap<String, String>) -> Self {
        Self {
            identity: headers.get("x-identity").cloned(),
            auth_token: headers.get("authorization")
                .map(|v| v.strip_prefix("Bearer ").unwrap_or(v).to_string()),
            request_id: headers.get("x-request-id").cloned(),
            timeout_ms: headers.get("grpc-timeout")
                .and_then(|v| v.strip_suffix('m'))
                .and_then(|v| v.parse().ok()),
            metadata: headers.clone(),
        }
    }

    /// Check if the connection has a valid auth token.
    pub fn is_authenticated(&self) -> bool {
        self.auth_token.is_some()
    }
}

/// Server-streaming search: yields results in batches.
///
/// Useful for large result sets where the client wants to process
/// results incrementally without waiting for the full result set.
pub struct ServerStreamingSearch<'a> {
    db: &'a crate::database::Database,
    collection: String,
    query: Vec<f32>,
    total_k: usize,
    batch_size: usize,
    offset: usize,
    done: bool,
}

impl<'a> ServerStreamingSearch<'a> {
    /// Create a new server-streaming search.
    pub fn new(
        db: &'a crate::database::Database,
        collection: &str,
        query: Vec<f32>,
        total_k: usize,
        batch_size: usize,
    ) -> Self {
        Self {
            db,
            collection: collection.to_string(),
            query,
            total_k,
            batch_size: batch_size.max(1),
            offset: 0,
            done: false,
        }
    }

    /// Get the next batch of results.
    /// Returns `None` when all results have been yielded.
    pub fn next_batch(&mut self) -> GrpcResult<Option<StreamSearchResponse>> {
        if self.done || self.offset >= self.total_k {
            return Ok(None);
        }

        let coll = self.db.collection(&self.collection).map_err(|e| GrpcError {
            code: GrpcStatus::NotFound,
            message: e.to_string(),
        })?;

        // Fetch a larger set and slice the window
        let fetch_k = (self.offset + self.batch_size).min(self.total_k);
        let all_results = coll.search(&self.query, fetch_k).map_err(|e| GrpcError {
            code: GrpcStatus::Internal,
            message: e.to_string(),
        })?;

        if self.offset >= all_results.len() {
            self.done = true;
            return Ok(None);
        }

        let end = (self.offset + self.batch_size).min(all_results.len());
        let batch: Vec<VectorResult> = all_results[self.offset..end]
            .iter()
            .map(|r| VectorResult {
                id: r.id.clone(),
                distance: r.distance,
                metadata: r.metadata.clone(),
            })
            .collect();

        self.offset = end;
        if self.offset >= self.total_k || batch.is_empty() {
            self.done = true;
        }

        Ok(Some(StreamSearchResponse {
            request_id: format!("batch-{}", self.offset / self.batch_size),
            results: batch,
            latency_us: 0,
        }))
    }
}

// ── Health Service ──────────────────────────────────────────────────────────

/// gRPC health check status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ServingStatus {
    Unknown,
    Serving,
    NotServing,
}

/// Health check response for gRPC health protocol (grpc.health.v1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckResponse {
    pub status: ServingStatus,
    pub version: String,
    pub uptime_secs: u64,
    pub collections: usize,
    pub total_vectors: u64,
}

/// Health service that reports server readiness.
pub struct HealthService<'a> {
    db: &'a crate::database::Database,
    start_time: std::time::Instant,
}

impl<'a> HealthService<'a> {
    pub fn new(db: &'a crate::database::Database) -> Self {
        Self {
            db,
            start_time: std::time::Instant::now(),
        }
    }

    pub fn check(&self) -> HealthCheckResponse {
        let collections = self.db.list_collections();
        let total_vectors: u64 = collections.iter()
            .filter_map(|name| self.db.collection(name).ok())
            .map(|c| c.len() as u64)
            .sum();

        HealthCheckResponse {
            status: ServingStatus::Serving,
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_secs: self.start_time.elapsed().as_secs(),
            collections: collections.len(),
            total_vectors,
        }
    }
}

// ── Batch Insert with Progress ──────────────────────────────────────────────

/// Progress update during a batch insert operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchInsertProgress {
    /// Total vectors in this batch.
    pub total: usize,
    /// Vectors inserted so far.
    pub completed: usize,
    /// Vectors that failed.
    pub failed: usize,
    /// Whether the batch is complete.
    pub done: bool,
    /// Elapsed time in milliseconds.
    pub elapsed_ms: u64,
}

/// Batch inserter that reports progress after each sub-batch.
pub struct ProgressBatchInserter<'a> {
    service: &'a DatabaseService<'a>,
    chunk_size: usize,
}

impl<'a> ProgressBatchInserter<'a> {
    pub fn new(service: &'a DatabaseService<'a>, chunk_size: usize) -> Self {
        Self {
            service,
            chunk_size: chunk_size.max(1),
        }
    }

    /// Insert vectors in chunks, yielding progress after each chunk.
    pub fn insert_with_progress(
        &self,
        req: BatchInsertRequest,
    ) -> Vec<BatchInsertProgress> {
        let start = std::time::Instant::now();
        let total = req.vectors.len();
        let mut progress_reports = Vec::new();
        let mut completed = 0usize;
        let mut failed = 0usize;

        for chunk in req.vectors.chunks(self.chunk_size) {
            let chunk_req = BatchInsertRequest {
                collection: req.collection.clone(),
                vectors: chunk.to_vec(),
            };

            match self.service.batch_insert(chunk_req) {
                Ok(resp) => {
                    completed += resp.inserted as usize;
                    failed += resp.failed as usize;
                }
                Err(_) => {
                    failed += chunk.len();
                }
            }

            progress_reports.push(BatchInsertProgress {
                total,
                completed,
                failed,
                done: completed + failed >= total,
                elapsed_ms: start.elapsed().as_millis() as u64,
            });
        }

        // Ensure final report marks done
        if let Some(last) = progress_reports.last_mut() {
            last.done = true;
        }

        progress_reports
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;

    #[test]
    fn test_create_and_insert() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);

        let resp = svc.create_collection(CreateCollectionRequest {
            name: "test".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();
        assert!(resp.success);

        let resp = svc.insert(InsertRequest {
            collection: "test".into(), id: "v1".into(),
            vector: vec![1.0; 4], metadata: None,
        }).unwrap();
        assert!(resp.success);
    }

    #[test]
    fn test_search() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        svc.create_collection(CreateCollectionRequest {
            name: "test".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();

        svc.insert(InsertRequest {
            collection: "test".into(), id: "v1".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0], metadata: None,
        }).unwrap();

        let resp = svc.search(SearchRequest {
            collection: "test".into(), query: vec![1.0, 0.0, 0.0, 0.0],
            k: 5, ef_search: None, filter: None,
        }).unwrap();
        assert_eq!(resp.results.len(), 1);
        assert_eq!(resp.results[0].id, "v1");
    }

    #[test]
    fn test_batch_insert() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        svc.create_collection(CreateCollectionRequest {
            name: "test".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();

        let resp = svc.batch_insert(BatchInsertRequest {
            collection: "test".into(),
            vectors: vec![
                VectorEntry { id: "v1".into(), vector: vec![1.0; 4], metadata: None },
                VectorEntry { id: "v2".into(), vector: vec![2.0; 4], metadata: None },
            ],
        }).unwrap();
        assert_eq!(resp.inserted, 2);
        assert_eq!(resp.failed, 0);
    }

    #[test]
    fn test_get_and_delete() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        svc.create_collection(CreateCollectionRequest {
            name: "test".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();
        svc.insert(InsertRequest {
            collection: "test".into(), id: "v1".into(),
            vector: vec![1.0; 4], metadata: None,
        }).unwrap();

        let get = svc.get(GetRequest { collection: "test".into(), id: "v1".into() }).unwrap();
        assert!(get.found);

        let del = svc.delete(DeleteRequest { collection: "test".into(), id: "v1".into() }).unwrap();
        assert!(del.deleted);

        let get2 = svc.get(GetRequest { collection: "test".into(), id: "v1".into() }).unwrap();
        assert!(!get2.found);
    }

    #[test]
    fn test_list_collections() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        svc.create_collection(CreateCollectionRequest {
            name: "a".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();
        svc.create_collection(CreateCollectionRequest {
            name: "b".into(), dimensions: 8, distance: "euclidean".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();

        let resp = svc.list_collections(ListCollectionsRequest {}).unwrap();
        assert_eq!(resp.collections.len(), 2);
    }

    #[test]
    fn test_not_found_error() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        let err = svc.search(SearchRequest {
            collection: "nonexistent".into(), query: vec![1.0; 4],
            k: 5, ef_search: None, filter: None,
        }).unwrap_err();
        assert_eq!(err.code, GrpcStatus::NotFound);
    }

    #[test]
    fn test_grpc_error_display() {
        let err = GrpcError { code: GrpcStatus::NotFound, message: "test".into() };
        assert!(format!("{err}").contains("NotFound"));
    }

    #[test]
    fn test_streaming_insert() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        svc.create_collection(CreateCollectionRequest {
            name: "test".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();

        let chunks = vec![
            StreamInsertChunk {
                collection: "test".into(),
                vectors: vec![
                    VectorEntry { id: "v1".into(), vector: vec![1.0; 4], metadata: None },
                    VectorEntry { id: "v2".into(), vector: vec![0.5; 4], metadata: None },
                ],
            },
            StreamInsertChunk {
                collection: "test".into(),
                vectors: vec![
                    VectorEntry { id: "v3".into(), vector: vec![0.1; 4], metadata: None },
                ],
            },
        ];

        let resp = svc.streaming_insert(chunks).unwrap();
        assert_eq!(resp.total_inserted, 3);
        assert_eq!(resp.total_failed, 0);
        assert_eq!(resp.chunks_processed, 2);
    }

    #[test]
    fn test_streaming_search() {
        let db = Database::in_memory();
        let svc = DatabaseService::new(&db);
        svc.create_collection(CreateCollectionRequest {
            name: "test".into(), dimensions: 4, distance: "cosine".into(),
            hnsw_m: None, hnsw_ef_construction: None,
        }).unwrap();

        svc.insert(InsertRequest {
            collection: "test".into(), id: "v1".into(),
            vector: vec![1.0, 0.0, 0.0, 0.0], metadata: None,
        }).unwrap();

        let requests = vec![
            StreamSearchRequest {
                collection: "test".into(),
                query: vec![1.0, 0.0, 0.0, 0.0],
                k: 5,
                request_id: "q1".into(),
            },
            StreamSearchRequest {
                collection: "test".into(),
                query: vec![0.0, 1.0, 0.0, 0.0],
                k: 5,
                request_id: "q2".into(),
            },
        ];

        let responses = svc.streaming_search(requests).unwrap();
        assert_eq!(responses.len(), 2);
        assert_eq!(responses[0].request_id, "q1");
        assert_eq!(responses[1].request_id, "q2");
        assert!(!responses[0].results.is_empty());
    }
}
