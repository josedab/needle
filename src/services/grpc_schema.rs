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
}
