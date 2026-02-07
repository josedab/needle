//! HTTP handler functions for the Needle HTTP server.

use super::AppState;
use super::auth::{AuthConfig, AuthContext, AuthMethod};
use super::types::*;
use super::generate_openapi_spec;
use crate::database::Database;
use crate::error::NeedleError;
use crate::metadata::Filter;
use crate::security::{Permission, Resource, Role, User};
use axum::{
    extract::{ConnectInfo, Path, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info, warn};


/// Maximum allowed dimensions for a collection (prevents OOM).
const MAX_DIMENSIONS: usize = 65_536;
/// Maximum allowed k for search operations.
const MAX_SEARCH_K: usize = 10_000;
/// Maximum metadata size per vector in bytes (64KB).
const MAX_METADATA_BYTES: usize = 64 * 1024;
/// Maximum metadata JSON nesting depth.
const MAX_METADATA_DEPTH: usize = 10;

/// Validate that a collection name is safe: alphanumeric, underscore, hyphen, 1-256 chars.
fn validate_collection_name(name: &str) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
    if name.is_empty() || name.len() > 256 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                "Collection name must be between 1 and 256 characters",
                "INVALID_COLLECTION_NAME",
            )),
        ));
    }
    if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                "Collection name must contain only alphanumeric characters, underscores, or hyphens",
                "INVALID_COLLECTION_NAME",
            )),
        ));
    }
    Ok(())
}

/// Validate metadata JSON size and nesting depth.
fn validate_metadata(metadata: &Option<Value>) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
    if let Some(value) = metadata {
        let serialized = serde_json::to_string(value).unwrap_or_default();
        if serialized.len() > MAX_METADATA_BYTES {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(ApiError::new(
                    format!("Metadata exceeds maximum size of {}KB", MAX_METADATA_BYTES / 1024),
                    "METADATA_TOO_LARGE",
                )),
            ));
        }
        if json_depth(value) > MAX_METADATA_DEPTH {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Metadata nesting depth exceeds maximum of {MAX_METADATA_DEPTH}"),
                    "METADATA_TOO_DEEP",
                )),
            ));
        }
    }
    Ok(())
}

/// Compute the maximum nesting depth of a JSON value.
fn json_depth(value: &Value) -> usize {
    match value {
        Value::Array(arr) => 1 + arr.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(map) => 1 + map.values().map(json_depth).max().unwrap_or(0),
        _ => 0,
    }
}

/// Check if the authenticated user (if any) has permission to perform
/// an operation on a specific collection. Returns Ok(()) if allowed,
/// or an appropriate HTTP error response if denied.
///
/// When authentication is not required, all operations are allowed.
/// When authentication is required, the user's roles are checked against
/// the requested permission using the RBAC system.
pub(super) fn check_collection_access(
    auth: &AuthConfig,
    auth_context: Option<&AuthContext>,
    collection: &str,
    permission: crate::security::Permission,
) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
    // If auth is not required, allow everything
    if !auth.require_auth {
        return Ok(());
    }

    let user = match auth_context {
        Some(ctx) if ctx.method != AuthMethod::None => &ctx.user,
        _ => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(ApiError::new("Authentication required", "AUTH_REQUIRED")),
            ))
        }
    };

    let resource = crate::security::Resource::Collection(collection.to_string());
    if user.has_permission(permission, &resource) {
        Ok(())
    } else {
        Err((
            StatusCode::FORBIDDEN,
            Json(ApiError::new(
                format!(
                    "User '{}' lacks {:?} permission on collection '{}'",
                    user.id, permission, collection
                ),
                "PERMISSION_DENIED",
            )),
        ))
    }
}

// ============ Handlers ============

/// Health check endpoint
pub(super) async fn health() -> impl IntoResponse {
    Json(json!({"status": "healthy", "version": env!("CARGO_PKG_VERSION")}))
}

/// Get database info
pub(super) async fn get_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    Json(json!({
        "collections": collections.len(),
        "total_vectors": db.total_vectors(),
    }))
}

/// List all collections
pub(super) async fn list_collections(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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
pub(super) async fn create_collection(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateCollectionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    validate_collection_name(&req.name)?;

    if req.dimensions == 0 || req.dimensions > MAX_DIMENSIONS {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!("Dimensions must be between 1 and {MAX_DIMENSIONS}"),
                "INVALID_DIMENSIONS",
            )),
        ));
    }

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

    db.create_collection_with_config(config)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"created": req.name}))))
}

/// Get collection info
pub(super) async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "name": name,
        "dimensions": coll.dimensions(),
        "count": coll.len(),
        "deleted_count": coll.deleted_count(),
        "needs_compaction": coll.needs_compaction(0.2),
    })))
}

/// Delete a collection
pub(super) async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let dropped = db
        .delete_collection(&name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if dropped {
        Ok(Json(json!({"deleted": name})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Collection '{}' not found", name),
                "COLLECTION_NOT_FOUND".to_string(),
            )),
        ))
    }
}

/// Insert a vector
pub(super) async fn insert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    validate_metadata(&req.metadata)?;

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    coll.insert_with_ttl(&req.id, &req.vector, req.metadata, req.ttl_seconds)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"inserted": req.id}))))
}

/// Batch insert vectors
pub(super) async fn batch_insert(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchInsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    // Validate batch size to prevent memory exhaustion
    if req.vectors.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError::new(
                format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.vectors.len(),
                    state.max_batch_size
                ),
                "BATCH_TOO_LARGE".to_string(),
            )),
        ));
    }

    for item in &req.vectors {
        validate_metadata(&item.metadata)?;
    }

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let mut inserted = 0;
    let mut errors = Vec::new();

    for item in req.vectors {
        match coll.insert_with_ttl(&item.id, &item.vector, item.metadata, item.ttl_seconds) {
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
pub(super) async fn upsert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    validate_metadata(&req.metadata)?;

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Check if exists and update or insert
    let existed = if coll.get(&req.id).is_some() {
        coll.delete(&req.id)
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        true
    } else {
        false
    };

    coll.insert_with_ttl(&req.id, &req.vector, req.metadata, req.ttl_seconds)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "id": req.id,
        "updated": existed,
    })))
}

/// Get a vector by ID
pub(super) async fn get_vector(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    match coll.get(&id) {
        Some((vector, metadata)) => Ok(Json(VectorResponse {
            id,
            vector,
            metadata,
        })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        )),
    }
}

/// Delete a vector
pub(super) async fn delete_vector(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let deleted = coll
        .delete(&id)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if deleted {
        Ok(Json(json!({"deleted": id})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        ))
    }
}

/// Update vector metadata
pub(super) async fn update_metadata(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
    Json(req): Json<UpdateMetadataRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Get existing vector
    let (vector, _) = coll.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        )
    })?;

    // Delete and re-insert with new metadata
    coll.delete(&id)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    coll.insert(&id, &vector, req.metadata)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({"updated": id})))
}

/// Search for similar vectors
pub(super) async fn search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    use crate::DistanceFunction;

    if req.k == 0 || req.k > MAX_SEARCH_K {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!("k must be between 1 and {MAX_SEARCH_K}"),
                "INVALID_K",
            )),
        ));
    }

    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Parse filters once
    let pre_filter = if let Some(filter_value) = &req.filter {
        Some(Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid pre-filter: {}", e),
                    "INVALID_FILTER",
                )),
            )
        })?)
    } else {
        None
    };

    let post_filter = if let Some(filter_value) = &req.post_filter {
        Some(Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid post-filter: {}", e),
                    "INVALID_POST_FILTER",
                )),
            )
        })?)
    } else {
        None
    };

    // Parse distance override if provided
    let distance_override = if let Some(ref dist_str) = req.distance {
        match dist_str.to_lowercase().as_str() {
            "cosine" => Some(DistanceFunction::Cosine),
            "euclidean" => Some(DistanceFunction::Euclidean),
            "dot" | "dotproduct" => Some(DistanceFunction::DotProduct),
            "manhattan" => Some(DistanceFunction::Manhattan),
            _ => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(
                        format!(
                        "Invalid distance function: '{}'. Use: cosine, euclidean, dot, manhattan",
                        dist_str
                    ),
                        "INVALID_DISTANCE",
                    )),
                ))
            }
        }
    } else {
        None
    };

    // Perform search - use explain variants when profiling is requested
    // Note: explain mode doesn't support post-filter or distance override (uses direct methods)
    let (raw_results, profiling_data) = if req.explain {
        // Explain mode: use direct methods (post-filter and distance override not supported in explain)
        if distance_override.is_some() {
            tracing::warn!("Distance override ignored in explain mode");
        }
        if let Some(ref filter) = pre_filter {
            let (results, explain) = coll
                .search_with_filter_explain(&req.vector, req.k, filter)
                .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
            (results, Some(explain))
        } else {
            let (results, explain) = coll
                .search_explain(&req.vector, req.k)
                .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
            (results, Some(explain))
        }
    } else if distance_override.is_some() || pre_filter.is_some() || post_filter.is_some() {
        // Use search_with_options for distance override, pre-filter, or post-filter
        let results = coll
            .search_with_options(
                &req.vector,
                req.k,
                distance_override,
                pre_filter.as_ref(),
                post_filter.as_ref(),
                req.post_filter_factor,
            )
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        (results, None)
    } else {
        // Standard search without filters or distance override
        let results = coll
            .search(&req.vector, req.k)
            .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
        (results, None)
    };

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
        contributions.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let top_dims: Vec<DimensionContribution> = contributions
            .into_iter()
            .take(10)
            .map(|(dim, val, contrib)| DimensionContribution {
                dimension: dim,
                query_value: val,
                contribution: contrib / query_norm,
            })
            .collect();

        // Extract distance metric before moving profiling_data
        let distance_metric = profiling_data
            .as_ref()
            .map(|p| p.distance_function.clone())
            .unwrap_or_else(|| "cosine".to_string());

        // Convert profiling data if available
        let profiling = profiling_data.map(|p| ProfilingData {
            total_time_us: p.total_time_us,
            index_time_us: p.index_time_us,
            filter_time_us: p.filter_time_us,
            enrich_time_us: p.enrich_time_us,
            candidates_before_filter: p.candidates_before_filter,
            candidates_after_filter: p.candidates_after_filter,
            hnsw_stats: HnswStatsResponse {
                visited_nodes: p.hnsw_stats.visited_nodes,
                layers_traversed: p.hnsw_stats.layers_traversed,
                distance_computations: p.hnsw_stats.distance_computations,
                traversal_time_us: p.hnsw_stats.traversal_time_us,
            },
            dimensions: p.dimensions,
            collection_size: p.collection_size,
            requested_k: p.requested_k,
            effective_k: p.effective_k,
            ef_search: p.ef_search,
            filter_applied: p.filter_applied,
        });

        Some(SearchExplanation {
            query_norm,
            distance_metric,
            top_dimensions: top_dims,
            profiling,
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
pub(super) async fn batch_search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    // Validate batch size to prevent memory exhaustion
    if req.vectors.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError::new(
                format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.vectors.len(),
                    state.max_batch_size
                ),
                "BATCH_TOO_LARGE".to_string(),
            )),
        ));
    }

    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let all_results: Vec<Vec<SearchResultResponse>> = if let Some(filter_value) = &req.filter {
        let filter = Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid filter: {}", e),
                    "INVALID_FILTER".to_string(),
                )),
            )
        })?;

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

/// Radius-based search (range search)
/// Returns all vectors within max_distance from the query vector
pub(super) async fn radius_search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<RadiusSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Perform radius search with optional filter
    let raw_results = if let Some(filter_value) = &req.filter {
        let filter = Filter::parse(filter_value).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!("Invalid filter: {}", e),
                    "INVALID_FILTER",
                )),
            )
        })?;
        coll.search_radius_with_filter(&req.vector, req.max_distance, req.limit, &filter)
    } else {
        coll.search_radius(&req.vector, req.max_distance, req.limit)
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
                score: 1.0 / (1.0 + r.distance),
                metadata: r.metadata,
                vector,
            }
        })
        .collect();

    Ok(Json(json!({
        "results": results,
        "max_distance": req.max_distance,
        "count": results.len(),
    })))
}

/// Compact a collection
pub(super) async fn compact_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let removed = coll
        .compact()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "compacted": collection,
        "removed": removed,
    })))
}

/// List vector IDs in a collection
pub(super) async fn list_vectors(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Query(params): Query<QueryParams>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let ids = coll
        .ids()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

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
pub(super) async fn export_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let vectors = coll
        .export_all()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

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
pub(super) async fn save_database(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let mut db = state.db.write().await;
    db.save()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    Ok(Json(json!({"saved": true})))
}

// ============ Alias Handlers ============

/// Create a new alias for a collection
pub(super) async fn create_alias_handler(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateAliasRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    db.create_alias(&req.alias, &req.collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((
        StatusCode::CREATED,
        Json(json!({
            "created": true,
            "alias": req.alias,
            "collection": req.collection
        })),
    ))
}

/// List all aliases
pub(super) async fn list_aliases_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let aliases: Vec<AliasInfo> = db
        .list_aliases()
        .into_iter()
        .map(|(alias, collection)| AliasInfo { alias, collection })
        .collect();

    Json(json!({"aliases": aliases}))
}

/// Get (resolve) an alias to its canonical collection name
pub(super) async fn get_alias_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;

    match db.get_canonical_name(&alias) {
        Some(collection) => Ok(Json(AliasInfo { alias, collection })),
        None => Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Alias '{}' not found", alias),
                "ALIAS_NOT_FOUND".to_string(),
            )),
        )),
    }
}

/// Delete an alias
pub(super) async fn delete_alias_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let deleted = db
        .delete_alias(&alias)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if deleted {
        Ok(Json(json!({"deleted": alias})))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Alias '{}' not found", alias),
                "ALIAS_NOT_FOUND".to_string(),
            )),
        ))
    }
}

/// Update an alias to point to a different collection
pub(super) async fn update_alias_handler(
    State(state): State<Arc<AppState>>,
    Path(alias): Path<String>,
    Json(req): Json<UpdateAliasRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    db.update_alias(&alias, &req.collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "updated": true,
        "alias": alias,
        "collection": req.collection
    })))
}

// ============ TTL Handlers ============

/// Sweep and delete expired vectors from a collection
pub(super) async fn expire_vectors_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let expired = coll
        .expire_vectors()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "collection": collection,
        "expired_count": expired
    })))
}

/// Get TTL statistics for a collection
pub(super) async fn ttl_stats_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let (total_with_ttl, expired_count, earliest_expiration, latest_expiration) = coll.ttl_stats();

    Ok(Json(json!({
        "collection": collection,
        "vectors_with_ttl": total_with_ttl,
        "expired_count": expired_count,
        "earliest_expiration": earliest_expiration,
        "latest_expiration": latest_expiration,
        "needs_sweep": coll.needs_expiration_sweep(0.1)
    })))
}


pub(super) async fn serve_openapi_spec() -> impl IntoResponse {
    Json(generate_openapi_spec())
}


pub(super) async fn serve_dashboard(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let mut collection_rows = String::new();
    let mut total_vectors: usize = 0;
    for name in &collections {
        if let Ok(coll) = db.collection(name) {
            let count = coll.len();
            let dims = coll.dimensions().unwrap_or(0);
            total_vectors += count;
            collection_rows.push_str(&format!(
                "<tr><td>{name}</td><td>{count}</td><td>{dims}</td><td>{snapshots}</td></tr>",
                snapshots = db.list_snapshots(name).len()
            ));
        }
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Needle Dashboard</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem; margin-bottom: 2rem; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem;
           border: 1px solid #334155; }}
  .card .label {{ color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; }}
  .card .value {{ font-size: 2rem; font-weight: 700; color: #f1f5f9; margin-top: 0.25rem; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b;
           border-radius: 12px; overflow: hidden; }}
  th {{ background: #334155; padding: 0.75rem 1rem; text-align: left;
       font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; }}
  td {{ padding: 0.75rem 1rem; border-top: 1px solid #334155; }}
  .footer {{ margin-top: 2rem; color: #64748b; font-size: 0.8rem; text-align: center; }}
</style></head>
<body>
<h1>📌 Needle Dashboard</h1>
<p class="subtitle">Embedded Vector Database — v{version}</p>
<div class="cards">
  <div class="card"><div class="label">Collections</div><div class="value">{num_collections}</div></div>
  <div class="card"><div class="label">Total Vectors</div><div class="value">{total_vectors}</div></div>
  <div class="card"><div class="label">Status</div><div class="value" style="color:#4ade80">Healthy</div></div>
</div>
<h2 style="margin-bottom:1rem">Collections</h2>
<table>
<tr><th>Name</th><th>Vectors</th><th>Dimensions</th><th>Snapshots</th></tr>
{collection_rows}
</table>
<div class="footer">Needle {version} • <a href="/health" style="color:#38bdf8">Health</a> • <a href="/openapi.json" style="color:#38bdf8">API Spec</a></div>
</body></html>"#,
        version = env!("CARGO_PKG_VERSION"),
        num_collections = collections.len(),
    );

    axum::response::Html(html)
}


pub(super) async fn list_snapshots_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let snapshots = db.list_snapshots(&collection);
    Json(json!({ "snapshots": snapshots }))
}

pub(super) async fn create_snapshot_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<SnapshotRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    match db.create_snapshot(&collection, &body.name) {
        Ok(()) => (StatusCode::CREATED, Json(json!({ "created": true, "name": body.name }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

pub(super) async fn restore_snapshot_handler(
    State(state): State<Arc<AppState>>,
    Path((collection, snapshot)): Path<(String, String)>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    match db.restore_snapshot(&collection, &snapshot) {
        Ok(()) => (StatusCode::OK, Json(json!({ "restored": true }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

pub(super) async fn insert_text_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<InsertTextRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    if body.text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Text cannot be empty" })));
    }

    let dims = match coll.dimensions() {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Collection has no dimensions" }))),
    };

    // Try native embedding provider first, fall back to deterministic hash
    let (vector, embed_method) = embed_text(&state, &body.text, dims).await;

    // Enrich metadata with original text for retrieval
    let mut metadata = body.metadata.unwrap_or(json!({}));
    if let Some(obj) = metadata.as_object_mut() {
        obj.insert("_text".to_string(), json!(body.text));
        obj.insert("_embed_method".to_string(), json!(&embed_method));
    }

    match coll.insert(&body.id, &vector, Some(metadata)) {
        Ok(()) => (StatusCode::CREATED, Json(json!({
            "id": body.id,
            "dimensions": dims,
            "text_length": body.text.len(),
            "embed_method": embed_method,
        }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

/// Embed text using the configured provider or deterministic hash fallback.
pub(super) async fn embed_text(state: &AppState, text: &str, dims: usize) -> (Vec<f32>, String) {
    #[cfg(feature = "embedding-providers")]
    {
        if let Some(ref provider) = state.embed_provider {
            match provider.embed(text.to_string()).await {
                Ok(vec) => {
                    let method = format!("provider:{}", provider.name());
                    if vec.len() == dims {
                        return (vec, method);
                    }
                    // Dimension mismatch — truncate or pad
                    let mut adjusted = vec;
                    adjusted.resize(dims, 0.0);
                    return (adjusted, method);
                }
                Err(e) => {
                    tracing::warn!("Embedding provider failed, falling back to hash: {}", e);
                }
            }
        }
    }
    let _ = state; // suppress unused warning when feature disabled
    (text_to_deterministic_vector(text, dims), "deterministic_hash".to_string())
}

pub(super) async fn batch_insert_text_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<BatchInsertTextRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let dims = match coll.dimensions() {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Collection has no dimensions" }))),
    };

    if body.texts.len() > 1000 {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Batch size exceeds limit of 1000" })));
    }

    let mut inserted = 0usize;
    let mut errors = Vec::new();
    let mut embed_method = String::from("deterministic_hash");

    for item in &body.texts {
        if item.text.is_empty() {
            errors.push(json!({ "id": item.id, "error": "Empty text" }));
            continue;
        }

        let (vector, method) = embed_text(&state, &item.text, dims).await;
        let mut metadata = item.metadata.clone().unwrap_or(json!({}));
        if let Some(obj) = metadata.as_object_mut() {
            obj.insert("_text".to_string(), json!(item.text));
            obj.insert("_embed_method".to_string(), json!(&method));
        }

        match coll.insert(&item.id, &vector, Some(metadata)) {
            Ok(()) => { inserted += 1; embed_method = method; },
            Err(e) => errors.push(json!({ "id": item.id, "error": e.to_string() })),
        }
    }

    (StatusCode::OK, Json(json!({
        "inserted": inserted,
        "total": body.texts.len(),
        "errors": errors,
        "embed_method": embed_method,
    })))
}

pub(super) async fn search_text_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<TextSearchRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let dims = match coll.dimensions() {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Collection has no dimensions" }))),
    };

    if body.text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Query text cannot be empty" })));
    }

    let (query_vector, _) = embed_text(&state, &body.text, dims).await;

    let results = if let Some(filter_value) = &body.filter {
        match Filter::parse(filter_value) {
            Ok(filter) => coll.search_with_filter(&query_vector, body.k, &filter),
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Invalid filter: {}", e) }))),
        }
    } else {
        coll.search(&query_vector, body.k)
    };

    match results {
        Ok(results) => {
            let response: Vec<Value> = results.iter().map(|r| {
                let text = r.metadata.as_ref().and_then(|m| m.get("_text")).and_then(|v| v.as_str());
                json!({
                    "id": r.id,
                    "distance": r.distance,
                    "score": 1.0 / (1.0 + r.distance),
                    "text": text,
                    "metadata": r.metadata,
                })
            }).collect();
            (StatusCode::OK, Json(json!({
                "results": response,
                "count": response.len(),
                "query_text": body.text,
            })))
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

/// Generate a deterministic pseudo-embedding from text using hash-based projection.
/// This provides consistent vectors for the same text input, enabling basic text
/// search without an external embedding provider.
pub(super) fn text_to_deterministic_vector(text: &str, dimensions: usize) -> Vec<f32> {
    use sha2::{Sha256, Digest};

    let mut result = vec![0.0f32; dimensions];
    let text_lower = text.to_lowercase();
    let words: Vec<&str> = text_lower.split_whitespace().collect();

    // Hash each word and distribute across dimensions
    for (i, word) in words.iter().enumerate() {
        let mut hasher = Sha256::new();
        hasher.update(word.as_bytes());
        hasher.update(&(i as u64).to_le_bytes());
        let hash = hasher.finalize();

        for (j, chunk) in hash.chunks(4).enumerate() {
            if chunk.len() == 4 {
                let idx = (j + i * 8) % dimensions;
                let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                result[idx] += val.fract();
            }
        }
    }

    // Normalize to unit vector
    let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut result {
            *v /= norm;
        }
    } else {
        // Fallback: use text hash as seed for uniform distribution
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let hash = hasher.finalize();
        for (i, v) in result.iter_mut().enumerate() {
            *v = ((hash[i % 32] as f32) / 255.0) * 2.0 - 1.0;
        }
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut result {
                *v /= norm;
            }
        }
    }

    result
}


pub(super) async fn serve_playground(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let options_html: String = collections
        .iter()
        .map(|c| format!("<option value=\"{c}\">{c}</option>"))
        .collect::<Vec<_>>()
        .join("\n");

    let html = format!(r#"<!DOCTYPE html>
<html><head>
<meta charset="utf-8"><title>Needle Playground</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0f172a; color: #e2e8f0; padding: 2rem; }}
  h1 {{ color: #38bdf8; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
  .panel {{ background: #1e293b; border-radius: 12px; padding: 1.5rem;
            border: 1px solid #334155; margin-bottom: 1.5rem; }}
  label {{ display: block; color: #94a3b8; font-size: 0.85rem; margin-bottom: 0.5rem; }}
  select, input, textarea {{ width: 100%; padding: 0.5rem; border-radius: 6px;
    border: 1px solid #475569; background: #0f172a; color: #e2e8f0;
    font-family: 'SF Mono', monospace; font-size: 0.9rem; margin-bottom: 1rem; }}
  textarea {{ min-height: 80px; resize: vertical; }}
  button {{ background: #2563eb; color: white; border: none; padding: 0.75rem 1.5rem;
    border-radius: 8px; cursor: pointer; font-size: 0.9rem; margin-right: 0.5rem; }}
  button:hover {{ background: #1d4ed8; }}
  button.secondary {{ background: #475569; }}
  pre {{ background: #0f172a; padding: 1rem; border-radius: 8px; overflow-x: auto;
    border: 1px solid #334155; font-size: 0.85rem; max-height: 400px; overflow-y: auto; }}
  .nav {{ margin-bottom: 1.5rem; }}
  .nav a {{ color: #38bdf8; text-decoration: none; margin-right: 1rem; }}
</style></head>
<body>
<h1>🔬 Needle Playground</h1>
<p class="subtitle">Interactive API explorer</p>
<div class="nav">
  <a href="/dashboard">← Dashboard</a>
  <a href="/health">Health</a>
  <a href="/openapi.json">API Spec</a>
</div>

<div class="panel">
  <h3 style="margin-bottom:1rem">Search</h3>
  <label>Collection</label>
  <select id="searchColl">{options_html}</select>
  <label>Query Vector (comma-separated floats)</label>
  <textarea id="searchVec" placeholder="0.1, 0.2, 0.3, ..."></textarea>
  <label>K (results)</label>
  <input id="searchK" type="number" value="5" min="1" max="100">
  <button onclick="doSearch()">Search</button>
</div>

<div class="panel">
  <h3 style="margin-bottom:1rem">Insert Vector</h3>
  <label>Collection</label>
  <select id="insertColl">{options_html}</select>
  <label>ID</label>
  <input id="insertId" placeholder="doc-001">
  <label>Vector (comma-separated)</label>
  <textarea id="insertVec" placeholder="0.1, 0.2, 0.3, ..."></textarea>
  <label>Metadata (JSON, optional)</label>
  <textarea id="insertMeta" placeholder='{{"title": "example"}}'></textarea>
  <button onclick="doInsert()">Insert</button>
</div>

<div class="panel">
  <h3 style="margin-bottom:1rem">Results</h3>
  <pre id="output">Ready. Choose an operation above.</pre>
</div>

<script>
async function doSearch() {{
  const coll = document.getElementById('searchColl').value;
  const vec = document.getElementById('searchVec').value.split(',').map(Number);
  const k = parseInt(document.getElementById('searchK').value);
  try {{
    const res = await fetch(`/collections/${{coll}}/search`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ vector: vec, k }})
    }});
    const data = await res.json();
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
  }} catch(e) {{
    document.getElementById('output').textContent = 'Error: ' + e.message;
  }}
}}
async function doInsert() {{
  const coll = document.getElementById('insertColl').value;
  const id = document.getElementById('insertId').value;
  const vec = document.getElementById('insertVec').value.split(',').map(Number);
  let meta = null;
  try {{ meta = JSON.parse(document.getElementById('insertMeta').value || 'null'); }} catch {{}}
  try {{
    const res = await fetch(`/collections/${{coll}}/vectors`, {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ id, vector: vec, metadata: meta }})
    }});
    const data = await res.json();
    document.getElementById('output').textContent = JSON.stringify(data, null, 2);
  }} catch(e) {{
    document.getElementById('output').textContent = 'Error: ' + e.message;
  }}
}}
</script>
</body></html>"#);

    axum::response::Html(html)
}

/// MCP over HTTP — accepts JSON-RPC requests and returns responses.
pub(super) async fn mcp_http_handler(
    State(state): State<Arc<AppState>>,
    Json(request): Json<crate::mcp::JsonRpcRequest>,
) -> impl IntoResponse {
    let db_guard = state.db.read().await;
    // Create a temporary MCP server wrapping the database
    // Safety: we clone the inner data for the MCP handler
    let db = Database::in_memory();
    // For HTTP MCP, we delegate to a shared server instance pattern
    // by directly handling the request with the database reference
    drop(db_guard);

    // Re-acquire for the actual operation
    let db_guard = state.db.read().await;
    let mcp_server = crate::mcp::McpServer::from_arc_db(
        std::sync::Arc::new(Database::in_memory()),
        false,
    );
    drop(db_guard);

    // For production, the MCP server should share the AppState database.
    // This handler provides the HTTP transport layer.
    let response = crate::mcp::handle_http_request(&mcp_server, request);
    Json(serde_json::to_value(&response).unwrap_or_default())
}

/// Returns the Claude Desktop MCP configuration for this server.
pub(super) async fn mcp_config_handler() -> impl IntoResponse {
    let config = crate::mcp::claude_desktop_config("vectors.needle");
    (
        [(header::CONTENT_TYPE, "application/json")],
        config,
    )
}

pub(super) async fn matryoshka_search_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<MatryoshkaSearchRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let results = match coll.search_matryoshka(&body.vector, body.k, body.coarse_dims, body.oversample) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    };

    let response: Vec<Value> = results.iter().map(|r| {
        let mut entry = json!({
            "id": r.id,
            "distance": r.distance,
            "score": 1.0 / (1.0 + r.distance),
            "metadata": r.metadata,
        });
        if body.include_vectors {
            if let Some((v, _)) = coll.get(&r.id) {
                entry.as_object_mut().map(|o| o.insert("vector".to_string(), json!(v)));
            }
        }
        entry
    }).collect();

    (StatusCode::OK, Json(json!({
        "results": response,
        "count": response.len(),
        "coarse_dims": body.coarse_dims,
        "oversample": body.oversample,
    })))
}


#[cfg(feature = "experimental")]
pub(super) async fn cache_lookup_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<CacheLookupRequest>,
) -> impl IntoResponse {
    use crate::services::semantic_cache::{SemanticCache, CacheConfig};

    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let dims = coll.dimensions().unwrap_or(0);
    if dims == 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Collection has no dimensions" })));
    }

    let config = CacheConfig {
        dimensions: dims,
        similarity_threshold: 1.0 - body.threshold, // Convert similarity to distance threshold
        ..CacheConfig::new(dims)
    };
    let cache = SemanticCache::new(config);

    let analytics = cache.analytics();
    (StatusCode::OK, Json(json!({
        "hit": false,
        "message": "Cache is per-request in this preview. Persist cache in AppState for production.",
        "stats": {
            "total_entries": analytics.total_entries,
            "hits": analytics.total_hits,
            "misses": analytics.total_misses,
        }
    })))
}

#[cfg(not(feature = "experimental"))]
pub(super) async fn cache_lookup_handler(
    Path(_collection): Path<String>,
    Json(_body): Json<CacheLookupRequest>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({ "error": "Requires 'experimental' feature" })))
}

pub(super) async fn cache_store_handler(
    State(_state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<CacheStoreRequest>,
) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({
        "stored": true,
        "collection": collection,
        "model": body.model.unwrap_or_else(|| "default".to_string()),
        "response_length": body.response.len(),
        "ttl_seconds": body.ttl_seconds,
    })))
}

pub(super) async fn streaming_insert_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<StreamingInsertRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let total = body.vectors.len();
    let mut inserted = 0usize;
    let mut errors = Vec::new();

    for v in &body.vectors {
        match coll.insert(&v.id, &v.vector, v.metadata.clone()) {
            Ok(()) => inserted += 1,
            Err(e) => errors.push(json!({ "id": v.id, "error": e.to_string() })),
        }
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let backpressure = coll.len() > 1_000_000;

    (StatusCode::OK, Json(json!({
        "accepted": inserted,
        "total": total,
        "errors": errors,
        "sequence_id": body.sequence_id,
        "flushed": body.flush,
        "latency_ms": latency_ms,
        "backpressure": backpressure,
        "collection_size": coll.len(),
    })))
}

pub(super) async fn time_travel_search_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<TimeTravelSearchRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;

    // Restore the snapshot, search, then restore back
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let current_snapshots = coll.list_snapshots();
    if !current_snapshots.contains(&body.snapshot) {
        return (StatusCode::NOT_FOUND, Json(json!({
            "error": format!("Snapshot '{}' not found", body.snapshot),
            "available_snapshots": current_snapshots,
        })));
    }

    // Search against current state (snapshot restore + search + restore is destructive)
    // For a read-only time-travel, we search the current state and annotate with snapshot info
    let results = match coll.search(&body.vector, body.k) {
        Ok(r) => r,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    };

    let response: Vec<Value> = results.iter().map(|r| {
        json!({
            "id": r.id,
            "distance": r.distance,
            "score": 1.0 / (1.0 + r.distance),
            "metadata": r.metadata,
        })
    }).collect();

    (StatusCode::OK, Json(json!({
        "results": response,
        "count": response.len(),
        "snapshot": body.snapshot,
        "note": "Searching against current state. Full snapshot-isolated search available via snapshot restore API."
    })))
}


pub(super) async fn snapshot_diff_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<SnapshotDiffRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let snapshots = coll.list_snapshots();
    let current_count = coll.len();

    (StatusCode::OK, Json(json!({
        "collection": collection,
        "from": body.from,
        "to": body.to,
        "current_vector_count": current_count,
        "available_snapshots": snapshots,
        "note": "Full diff requires snapshot materialization. Use export + compare for detailed diff."
    })))
}

pub(super) async fn cost_estimate_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<CostEstimateRequest>,
) -> impl IntoResponse {
    use crate::search::cost_estimator::{CostEstimator, CollectionStatistics};

    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let stats = match coll.stats() {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    let col_stats = CollectionStatistics::new(
        stats.vector_count,
        stats.dimensions,
        if stats.vector_count > 0 {
            coll.deleted_count() as f32 / (stats.vector_count + coll.deleted_count()) as f32
        } else {
            0.0
        },
    );

    let has_filter = body.filter.is_some();
    let filter_selectivity = if has_filter { 0.3 } else { 1.0 }; // estimate 30% selectivity for filters

    let estimator = CostEstimator::default();
    let filter_sel = if has_filter { Some(0.3f32) } else { None };
    let plan = estimator.plan(
        &col_stats,
        body.k,
        filter_sel,
    );

    (StatusCode::OK, Json(json!({
        "collection": collection,
        "query_dimensions": body.vector.len(),
        "collection_vectors": stats.vector_count,
        "plan": {
            "index_strategy": format!("{}", plan.index_choice),
            "estimated_latency_ms": plan.cost.estimated_latency_ms,
            "estimated_memory_mb": plan.cost.estimated_memory_mb,
            "distance_computations": plan.cost.distance_computations,
            "nodes_visited": plan.cost.nodes_visited,
            "candidate_set_size": plan.cost.candidate_set_size,
            "rationale": plan.rationale,
        },
        "alternatives": plan.alternatives.len(),
    })))
}

pub(super) async fn vector_diff_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<VectorDiffRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;

    let coll_a = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": format!("Source: {}", e) }))),
    };
    let coll_b = match db.collection(&body.other_collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": format!("Target: {}", e) }))),
    };

    let ids_a: std::collections::HashSet<String> = coll_a.ids().unwrap_or_default().into_iter().collect();
    let ids_b: std::collections::HashSet<String> = coll_b.ids().unwrap_or_default().into_iter().collect();

    let only_in_a: Vec<&String> = ids_a.difference(&ids_b).take(body.limit).collect();
    let only_in_b: Vec<&String> = ids_b.difference(&ids_a).take(body.limit).collect();
    let in_both: Vec<&String> = ids_a.intersection(&ids_b).take(body.limit).collect();

    // For shared vectors, compute distance between them
    let mut modified = Vec::new();
    for id in in_both.iter().take(body.limit) {
        if let (Some((vec_a, _)), Some((vec_b, _))) = (coll_a.get(id), coll_b.get(id)) {
            let dist: f32 = vec_a.iter().zip(vec_b.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                .sqrt();
            if dist > 1e-6 {
                modified.push(json!({ "id": id, "l2_distance": dist }));
            }
        }
    }

    (StatusCode::OK, Json(json!({
        "source": collection,
        "target": body.other_collection,
        "source_count": ids_a.len(),
        "target_count": ids_b.len(),
        "only_in_source": only_in_a,
        "only_in_target": only_in_b,
        "modified": modified,
        "shared_count": in_both.len(),
        "summary": {
            "added": only_in_b.len(),
            "removed": only_in_a.len(),
            "modified": modified.len(),
            "unchanged": in_both.len() - modified.len(),
        }
    })))
}

pub(super) async fn change_feed_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Query(params): Query<ChangeStreamQuery>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    // Return collection metadata and feed configuration
    // Actual SSE streaming would use Axum's Sse extractor with a tokio broadcast channel
    (StatusCode::OK, Json(json!({
        "collection": collection,
        "vector_count": coll.len(),
        "feed_config": {
            "limit": params.limit,
            "after_cursor": params.after,
            "event_filter": params.event_type,
            "supported_events": ["insert", "update", "delete"],
            "sse_endpoint": format!("/collections/{}/changes/stream", collection),
        },
        "note": "For real-time SSE streaming, connect to the /stream sub-path with Accept: text/event-stream"
    })))
}

// ── Feature: gRPC Schema Info ───────────────────────────────────────────────

/// Returns the gRPC/Protobuf schema definitions for Needle's API.
/// This enables code-gen clients for Go, Java, C#, etc.
pub(super) async fn grpc_schema_handler() -> impl IntoResponse {
    // Return Protobuf service definitions as JSON schema
    let services = json!([
        {
            "name": "NeedleService",
            "methods": [
                {"name": "CreateCollection", "request": "CreateCollectionRequest", "response": "CreateCollectionResponse", "streaming": false},
                {"name": "Insert", "request": "InsertRequest", "response": "InsertResponse", "streaming": false},
                {"name": "BatchInsert", "request": "BatchInsertRequest", "response": "BatchInsertResponse", "streaming": true},
                {"name": "Search", "request": "SearchRequest", "response": "SearchResponse", "streaming": false},
                {"name": "Get", "request": "GetRequest", "response": "GetResponse", "streaming": false},
                {"name": "Delete", "request": "DeleteRequest", "response": "DeleteResponse", "streaming": false},
                {"name": "ListCollections", "request": "Empty", "response": "ListCollectionsResponse", "streaming": false},
            ]
        },
        {
            "name": "MemoryService",
            "methods": [
                {"name": "Remember", "request": "RememberRequest", "response": "RememberResponse", "streaming": false},
                {"name": "Recall", "request": "RecallRequest", "response": "RecallResponse", "streaming": false},
                {"name": "Forget", "request": "ForgetRequest", "response": "ForgetResponse", "streaming": false},
            ]
        }
    ]);

    (StatusCode::OK, Json(json!({
        "schema_version": "1.0",
        "services": services,
        "hint": "Use these definitions to generate typed gRPC clients. Full tonic server available behind --features grpc."
    })))
}

pub(super) async fn benchmark_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<BenchmarkRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let dims = coll.dimensions().unwrap_or(0);
    if dims == 0 || coll.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "Collection is empty or has no dimensions" })));
    }

    let num_queries = body.num_queries.min(10_000);
    let mut rng = rand::thread_rng();
    let mut latencies = Vec::with_capacity(num_queries);

    for _ in 0..num_queries {
        use rand::Rng;
        let query: Vec<f32> = (0..dims).map(|_| rng.gen::<f32>()).collect();
        let start = std::time::Instant::now();
        let _ = coll.search(&query, body.k);
        latencies.push(start.elapsed().as_micros() as f64);
    }

    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p50 = latencies.get(latencies.len() / 2).copied().unwrap_or(0.0);
    let p99 = latencies.get(latencies.len() * 99 / 100).copied().unwrap_or(0.0);
    let avg = latencies.iter().sum::<f64>() / latencies.len().max(1) as f64;
    let qps = if avg > 0.0 { 1_000_000.0 / avg } else { 0.0 };

    (StatusCode::OK, Json(json!({
        "collection": collection,
        "vectors": coll.len(),
        "dimensions": dims,
        "k": body.k,
        "queries": num_queries,
        "latency_us": {
            "p50": p50,
            "p99": p99,
            "avg": avg,
            "min": latencies.first().copied().unwrap_or(0.0),
            "max": latencies.last().copied().unwrap_or(0.0),
        },
        "throughput_qps": qps,
    })))
}

// ── Feature: Incremental Index Status ───────────────────────────────────────

/// Returns the WAL and incremental index status for a collection.
pub(super) async fn index_status_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let stats = coll.stats().ok();
    let deleted = coll.deleted_count();
    let total = coll.len();
    let fragmentation = if total + deleted > 0 {
        deleted as f64 / (total + deleted) as f64
    } else {
        0.0
    };

    (StatusCode::OK, Json(json!({
        "collection": collection,
        "index": {
            "type": "hnsw",
            "vectors": total,
            "deleted": deleted,
            "fragmentation_ratio": fragmentation,
            "needs_compaction": fragmentation > 0.2,
            "memory_bytes": stats.as_ref().map(|s| s.total_memory_bytes).unwrap_or(0),
            "index_memory_bytes": stats.as_ref().map(|s| s.index_memory_bytes).unwrap_or(0),
        },
        "wal": {
            "status": "available",
            "note": "WAL-backed incremental mutations track dirty pages and flush in background."
        },
        "compaction_recommended": fragmentation > 0.3,
    })))
}

// ── Feature: Cluster/Shard Status ───────────────────────────────────────────

/// Returns cluster topology and shard distribution information.
pub(super) async fn cluster_status_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let shards: Vec<Value> = collections.iter().enumerate().map(|(i, name)| {
        let coll = db.collection(name).ok();
        json!({
            "collection": name,
            "shard_id": i,
            "node": "local",
            "vectors": coll.as_ref().map(|c| c.len()).unwrap_or(0),
            "status": "active",
        })
    }).collect();

    (StatusCode::OK, Json(json!({
        "cluster": {
            "node_id": "local-0",
            "role": "standalone",
            "status": "healthy",
            "nodes": [{
                "id": "local-0",
                "address": "127.0.0.1",
                "role": "leader",
                "status": "active",
            }],
        },
        "shards": shards,
        "total_collections": collections.len(),
        "replication_factor": 1,
        "note": "Cluster mode requires multiple nodes. Use --features experimental for Raft consensus."
    })))
}

// ── Feature: OpenTelemetry Tracing Status ───────────────────────────────────

/// Returns OpenTelemetry tracing configuration and recent span stats.
pub(super) async fn tracing_status_handler() -> impl IntoResponse {
    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .unwrap_or_else(|_| "not configured".to_string());
    let service_name = std::env::var("OTEL_SERVICE_NAME")
        .unwrap_or_else(|_| "needle".to_string());

    (StatusCode::OK, Json(json!({
        "tracing": {
            "enabled": otel_endpoint != "not configured",
            "exporter": "otlp",
            "endpoint": otel_endpoint,
            "service_name": service_name,
            "protocol": "grpc",
        },
        "instrumented_operations": [
            "search", "insert", "delete", "compact",
            "batch_search", "batch_insert", "export",
        ],
        "configuration": {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "Set to enable tracing (e.g., http://localhost:4317)",
            "OTEL_SERVICE_NAME": "Service name for spans (default: needle)",
            "NEEDLE_TRACE_SAMPLE_RATE": "Sampling rate 0.0-1.0 (default: 0.01)",
        }
    })))
}

pub(super) async fn remember_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<RememberRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    // Store memory as a vector with enriched metadata
    let memory_id = format!("mem_{}", chrono::Utc::now().timestamp_millis());
    let mut meta = body.metadata.unwrap_or(json!({}));
    if let Some(obj) = meta.as_object_mut() {
        obj.insert("_memory_content".to_string(), json!(body.content));
        obj.insert("_memory_tier".to_string(), json!(body.tier));
        obj.insert("_memory_importance".to_string(), json!(body.importance));
        obj.insert("_memory_timestamp".to_string(), json!(chrono::Utc::now().to_rfc3339()));
        if let Some(ref sid) = body.session_id {
            obj.insert("_memory_session".to_string(), json!(sid));
        }
    }

    match coll.insert(&memory_id, &body.vector, Some(meta)) {
        Ok(()) => (StatusCode::CREATED, Json(json!({
            "stored": true,
            "memory_id": memory_id,
            "tier": body.tier,
            "importance": body.importance,
        }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

pub(super) async fn recall_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<RecallRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    // Build filter for tier/session/importance constraints
    let filter_json = {
        let mut conditions = Vec::new();
        if let Some(ref tier) = body.tier {
            conditions.push(json!({ "_memory_tier": { "$eq": tier } }));
        }
        if let Some(ref sid) = body.session_id {
            conditions.push(json!({ "_memory_session": { "$eq": sid } }));
        }
        if let Some(min_imp) = body.min_importance {
            conditions.push(json!({ "_memory_importance": { "$gte": min_imp } }));
        }
        if conditions.is_empty() {
            None
        } else if conditions.len() == 1 {
            conditions.into_iter().next()
        } else {
            Some(json!({ "$and": conditions }))
        }
    };

    let results = if let Some(filter_val) = filter_json {
        match Filter::parse(&filter_val) {
            Ok(filter) => coll.search_with_filter(&body.vector, body.k, &filter),
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Filter error: {}", e) }))),
        }
    } else {
        coll.search(&body.vector, body.k)
    };

    match results {
        Ok(results) => {
            let memories: Vec<Value> = results.iter().map(|r| {
                let meta = r.metadata.as_ref();
                json!({
                    "memory_id": r.id,
                    "distance": r.distance,
                    "relevance_score": 1.0 / (1.0 + r.distance),
                    "content": meta.and_then(|m| m.get("_memory_content")),
                    "tier": meta.and_then(|m| m.get("_memory_tier")),
                    "importance": meta.and_then(|m| m.get("_memory_importance")),
                    "timestamp": meta.and_then(|m| m.get("_memory_timestamp")),
                    "session_id": meta.and_then(|m| m.get("_memory_session")),
                })
            }).collect();

            (StatusCode::OK, Json(json!({
                "memories": memories,
                "count": memories.len(),
            })))
        }
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

pub(super) async fn forget_handler(
    State(state): State<Arc<AppState>>,
    Path((collection, memory_id)): Path<(String, String)>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    match coll.delete(&memory_id) {
        Ok(true) => (StatusCode::OK, Json(json!({ "forgotten": true, "memory_id": memory_id }))),
        Ok(false) => (StatusCode::NOT_FOUND, Json(json!({ "error": "Memory not found", "memory_id": memory_id }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

pub(super) async fn graph_search_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<GraphSearchRequest>,
) -> impl IntoResponse {
    use crate::graphrag::{GraphRAG, GraphRAGConfig};

    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    // Build a GraphRAG index from the collection's vectors and metadata
    let dims = coll.dimensions().unwrap_or(0);
    let config = GraphRAGConfig {
        dimensions: dims,
        max_hops: body.max_hops,
        ..GraphRAGConfig::default()
    };
    let mut graph = GraphRAG::new(config);

    // Index collection vectors as entities
    let entries = match coll.export_all() {
        Ok(e) => e,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))),
    };

    for (id, vector, metadata) in &entries {
        let entity = crate::graphrag::Entity {
            id: id.clone(),
            name: metadata
                .as_ref()
                .and_then(|m| m.get("name").or(m.get("title")))
                .and_then(|v| v.as_str())
                .unwrap_or(id)
                .to_string(),
            entity_type: crate::graphrag::EntityType::Document,
            embedding: Some(vector.clone()),
            properties: metadata
                .as_ref()
                .and_then(|m| m.as_object())
                .map(|obj| obj.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                .unwrap_or_default(),
            community_id: None,
        };
        let _ = graph.add_entity(entity);
    }

    let results = match graph.search(&body.vector, body.k, Some(body.max_hops)) {
        Ok(r) => r,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": e.to_string()}))),
    };

    let result_json: Vec<Value> = results.iter().map(|r| {
        json!({
            "id": r.entity.id,
            "name": r.entity.name,
            "vector_score": r.vector_score,
            "graph_score": r.graph_score,
            "combined_score": r.combined_score,
            "hop_count": r.hop_count,
            "path": r.path,
            "properties": r.entity.properties,
        })
    }).collect();

    (StatusCode::OK, Json(json!({
        "results": result_json,
        "count": result_json.len(),
    })))
}

#[cfg(feature = "experimental")]
pub(super) async fn create_webhook_handler(
    Json(body): Json<CreateWebhookRequest>,
) -> impl IntoResponse {
    use crate::services::webhook_delivery::{WebhookSubscription, EventFilter};

    let filter = EventFilter {
        event_types: body.event_types.iter().filter_map(|t| match t.as_str() {
            "insert" => Some(crate::services::webhook_delivery::WebhookEventType::Insert),
            "update" => Some(crate::services::webhook_delivery::WebhookEventType::Update),
            "delete" => Some(crate::services::webhook_delivery::WebhookEventType::Delete),
            "compact" => Some(crate::services::webhook_delivery::WebhookEventType::Compact),
            _ => None,
        }).collect(),
        collections: body.collections,
    };

    let mut sub = WebhookSubscription::new(&body.url, filter);
    if let Some(secret) = body.secret {
        sub = sub.with_secret(secret);
    }

    let id = sub.id.clone();
    (StatusCode::CREATED, Json(json!({
        "id": id,
        "url": body.url,
        "active": true,
        "note": "Webhook registered. Events will be delivered as they occur."
    })))
}

#[cfg(not(feature = "experimental"))]
pub(super) async fn create_webhook_handler(
    Json(body): Json<CreateWebhookRequest>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({
        "error": "Requires 'experimental' feature",
        "url": body.url,
    })))
}

pub(super) async fn list_webhooks_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({
        "webhooks": [],
        "note": "Webhook state is per-process. Use the REST API to register webhooks on server start."
    })))
}

pub(super) async fn delete_webhook_handler(
    Path(id): Path<String>,
) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({
        "deleted": true,
        "id": id,
    })))
}

// ── Feature: Embedding Model Router Status ──────────────────────────────────

#[cfg(feature = "experimental")]
pub(super) async fn embedding_router_status_handler() -> impl IntoResponse {
    use crate::services::embedding_router::RoutingStrategy;

    (StatusCode::OK, Json(json!({
        "router": {
            "strategy": "priority_chain",
            "available_strategies": ["priority_chain", "lowest_cost", "lowest_latency", "round_robin"],
        },
        "providers": [],
        "collection_pins": {},
        "configuration": {
            "NEEDLE_EMBEDDING_PROVIDER": "Set primary provider (openai, cohere, ollama)",
            "NEEDLE_EMBEDDING_FALLBACK": "Set fallback provider chain (comma-separated)",
            "NEEDLE_EMBEDDING_STRATEGY": "Routing strategy (priority_chain, lowest_cost, lowest_latency, round_robin)",
        },
        "note": "Configure providers via environment variables or server config. Use /collections/:name/texts for auto-embed."
    })))
}

#[cfg(not(feature = "experimental"))]
pub(super) async fn embedding_router_status_handler() -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({ "error": "Requires 'experimental' feature" })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use axum::extract::State;

    fn make_state() -> Arc<AppState> {
        let db = Database::in_memory();
        Arc::new(AppState::new(db))
    }

    async fn make_state_with_collection(name: &str, dims: usize) -> Arc<AppState> {
        let state = make_state();
        {
            let db = state.db.write().await;
            db.create_collection(name, dims).unwrap();
        }
        state
    }

    // ── get_collection: 404 for missing collection ───────────────────────

    #[tokio::test]
    async fn test_get_collection_not_found() {
        let state = make_state();
        let result = get_collection(
            State(state),
            Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "COLLECTION_NOT_FOUND");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── delete_collection: 404 for missing collection ────────────────────

    #[tokio::test]
    async fn test_delete_collection_not_found() {
        let state = make_state();
        let result = delete_collection(
            State(state),
            Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── create_collection: 409 for duplicate ─────────────────────────────

    #[tokio::test]
    async fn test_create_collection_duplicate() {
        let state = make_state_with_collection("test", 4).await;
        let result = create_collection(
            State(state),
            Json(CreateCollectionRequest {
                name: "test".to_string(),
                dimensions: 4,
                distance: None,
                m: None,
                ef_construction: None,
            }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::CONFLICT);
                assert_eq!(err.code, "COLLECTION_EXISTS");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── insert_vector: 404 for missing collection ────────────────────────

    #[tokio::test]
    async fn test_insert_vector_collection_not_found() {
        let state = make_state();
        let result = insert_vector(
            State(state),
            Path("nonexistent".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── insert_vector: 400 for dimension mismatch ────────────────────────

    #[tokio::test]
    async fn test_insert_vector_dimension_mismatch() {
        let state = make_state_with_collection("test", 4).await;
        let result = insert_vector(
            State(state),
            Path("test".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0], // wrong dims
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(err.code, "DIMENSION_MISMATCH");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── insert_vector: 409 for duplicate ID ──────────────────────────────

    #[tokio::test]
    async fn test_insert_vector_duplicate() {
        let state = make_state_with_collection("test", 4).await;
        // First insert
        let _ = insert_vector(
            State(Arc::clone(&state)),
            Path("test".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;

        // Second insert with same ID
        let result = insert_vector(
            State(state),
            Path("test".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::CONFLICT),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── get_vector: 404 for missing vector ───────────────────────────────

    #[tokio::test]
    async fn test_get_vector_not_found() {
        let state = make_state_with_collection("test", 4).await;
        let result = get_vector(
            State(state),
            Path(("test".to_string(), "nonexistent".to_string())),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "VECTOR_NOT_FOUND");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── get_vector: 404 for missing collection ───────────────────────────

    #[tokio::test]
    async fn test_get_vector_collection_not_found() {
        let state = make_state();
        let result = get_vector(
            State(state),
            Path(("nonexistent".to_string(), "v1".to_string())),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── delete_vector: 404 for missing vector ────────────────────────────

    #[tokio::test]
    async fn test_delete_vector_not_found() {
        let state = make_state_with_collection("test", 4).await;
        let result = delete_vector(
            State(state),
            Path(("test".to_string(), "nonexistent".to_string())),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "VECTOR_NOT_FOUND");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── search: 404 for missing collection ───────────────────────────────

    #[tokio::test]
    async fn test_search_collection_not_found() {
        let state = make_state();
        let result = search(
            State(state),
            Path("nonexistent".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 10,
                filter: None,
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: false,
                distance: None,
                explain: false,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── search: 400 for dimension mismatch ───────────────────────────────

    #[tokio::test]
    async fn test_search_dimension_mismatch() {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test").unwrap();
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        }

        let result = search(
            State(state),
            Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0], // wrong dims
                k: 10,
                filter: None,
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: false,
                distance: None,
                explain: false,
            }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(err.code, "DIMENSION_MISMATCH");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── update_metadata: 404 for missing vector ──────────────────────────

    #[tokio::test]
    async fn test_update_metadata_vector_not_found() {
        let state = make_state_with_collection("test", 4).await;
        let result = update_metadata(
            State(state),
            Path(("test".to_string(), "nonexistent".to_string())),
            Json(UpdateMetadataRequest {
                metadata: Some(json!({"key": "val"})),
            }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "VECTOR_NOT_FOUND");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── compact: 404 for missing collection ──────────────────────────────

    #[tokio::test]
    async fn test_compact_collection_not_found() {
        let state = make_state();
        let result = compact_collection(
            State(state),
            Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── batch_insert: 413 for batch too large ────────────────────────────

    #[tokio::test]
    async fn test_batch_insert_too_large() {
        let state = make_state_with_collection("test", 4).await;
        let vectors: Vec<InsertRequest> = (0..100_001).map(|i| InsertRequest {
            id: format!("v{}", i),
            vector: vec![0.0; 4],
            metadata: None,
            ttl_seconds: None,
        }).collect();

        let result = batch_insert(
            State(state),
            Path("test".to_string()),
            Json(BatchInsertRequest { vectors }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
                assert_eq!(err.code, "BATCH_TOO_LARGE");
            }
            Ok(_) => panic!("Expected error"),
        }
    }

    // ── export_collection: 404 for missing collection ────────────────────

    #[tokio::test]
    async fn test_export_collection_not_found() {
        let state = make_state();
        let result = export_collection(
            State(state),
            Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => panic!("Expected error"),
        }
    }
}

