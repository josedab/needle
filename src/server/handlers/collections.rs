//! Collection CRUD, alias, snapshot, and TTL handlers.

use crate::server::AppState;
use crate::server::types::*;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::warn;

use super::{validate_collection_name, MAX_DIMENSIONS, MAX_EXPORT_VECTORS};

// ============ Collection CRUD ============

/// List all collections in the database.
///
/// Supports optional `offset` and `limit` query parameters for pagination.
/// Without parameters, returns all collections (backward compatible).
pub(in crate::server) async fn list_collections(
    State(state): State<Arc<AppState>>,
    Query(params): Query<QueryParams>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let all_names = db.list_collections();
    let total = all_names.len();

    let offset = params.offset.unwrap_or(0);
    let limit = params.limit.unwrap_or(total);

    let page_names: Vec<_> = all_names
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect();

    let collections: Vec<CollectionInfo> = page_names
        .into_iter()
        .filter_map(|name| {
            let coll = db.collection(&name).map_err(|e| {
                warn!("Collection lookup failed for '{}': {e}", name);
                e
            }).ok()?;
            Some(CollectionInfo {
                name,
                dimensions: coll.dimensions().unwrap_or(0),
                count: coll.len(),
                deleted_count: coll.deleted_count(),
            })
        })
        .collect();

    let count = collections.len();
    let has_more = offset + count < total;

    Json(json!({
        "collections": collections,
        "pagination": {
            "count": count,
            "offset": offset,
            "total": total,
            "has_more": has_more
        }
    }))
}

/// Create a new vector collection.
///
/// `POST /collections` — accepts name, dimensions, and optional HNSW parameters.
/// Returns `201 Created` on success.
pub(in crate::server) async fn create_collection(
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
        config = config.with_distance(
            dist.parse().unwrap_or(crate::DistanceFunction::Cosine),
        );
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

/// Get detailed information about a collection.
///
/// `GET /collections/:name` — returns dimensions, vector count, deleted count,
/// and compaction status.
pub(in crate::server) async fn get_collection(
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

/// Delete a collection and all its vectors.
///
/// `DELETE /collections/:name` — returns 404 if the collection does not exist.
pub(in crate::server) async fn delete_collection(
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

/// Rename a collection.
///
/// `POST /collections/:name/rename` — renames the collection and updates aliases.
pub(in crate::server) async fn rename_collection(
    State(state): State<Arc<AppState>>,
    Path(old_name): Path<String>,
    Json(req): Json<RenameCollectionRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    validate_collection_name(&req.new_name)?;

    let db = state.db.write().await;
    db.rename_collection(&old_name, &req.new_name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "old_name": old_name,
        "new_name": req.new_name,
    })))
}

/// Compact a collection by removing deleted vectors and reclaiming space.
///
/// `POST /collections/:name/compact` — returns the number of removed entries.
pub(in crate::server) async fn compact_collection(
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

/// Export all vectors from a collection as JSON.
///
/// `GET /collections/:name/export` — returns up to `MAX_EXPORT_VECTORS`
/// entries with their IDs, vectors, and metadata.
pub(in crate::server) async fn export_collection(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let count = coll.len();
    if count > MAX_EXPORT_VECTORS {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError::new(
                format!(
                    "Collection has {} vectors, exceeding export limit of {}. Use list_vectors + get_vector for large exports.",
                    count, MAX_EXPORT_VECTORS
                ),
                "EXPORT_TOO_LARGE".to_string(),
            )),
        ));
    }

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

/// List vector IDs in a collection with pagination.
///
/// `GET /collections/:name/vectors` — accepts `offset` and `limit` query
/// parameters. Returns paginated IDs and total count.
pub(in crate::server) async fn list_vectors(
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

// ============ Alias Handlers ============

/// Create a new alias pointing to a collection.
///
/// `POST /aliases` — maps an alias name to a collection name.
/// Returns `201 Created`.
pub(in crate::server) async fn create_alias_handler(
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

/// List all collection aliases.
///
/// `GET /aliases` — returns an array of alias-to-collection mappings.
pub(in crate::server) async fn list_aliases_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let aliases: Vec<AliasInfo> = db
        .list_aliases()
        .into_iter()
        .map(|(alias, collection)| AliasInfo { alias, collection })
        .collect();

    Json(json!({"aliases": aliases}))
}

/// Resolve an alias to its canonical collection name.
///
/// `GET /aliases/:alias` — returns 404 if the alias does not exist.
pub(in crate::server) async fn get_alias_handler(
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

/// Delete an alias.
///
/// `DELETE /aliases/:alias` — returns 404 if the alias does not exist.
pub(in crate::server) async fn delete_alias_handler(
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

/// Update an alias to point to a different collection.
///
/// `PUT /aliases/:alias` — changes the target collection for an existing alias.
pub(in crate::server) async fn update_alias_handler(
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

// ============ Snapshot Handlers ============

/// List snapshots for a collection.
///
/// `GET /collections/:name/snapshots` — returns an array of snapshot names.
pub(in crate::server) async fn list_snapshots_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let snapshots = db.list_snapshots(&collection);
    Json(json!({ "snapshots": snapshots }))
}

/// Create a named snapshot of a collection.
///
/// `POST /collections/:name/snapshots` — creates a point-in-time snapshot.
/// Returns `201 Created` on success.
pub(in crate::server) async fn create_snapshot_handler(
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

/// Restore a collection from a named snapshot.
///
/// `POST /collections/:name/snapshots/:snapshot/restore` — replaces the
/// collection contents with the snapshot state.
pub(in crate::server) async fn restore_snapshot_handler(
    State(state): State<Arc<AppState>>,
    Path((collection, snapshot)): Path<(String, String)>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    match db.restore_snapshot(&collection, &snapshot) {
        Ok(()) => (StatusCode::OK, Json(json!({ "restored": true }))),
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
    }
}

// ============ TTL Handlers ============

/// Sweep and delete expired vectors from a collection.
///
/// `POST /collections/:name/expire` — removes vectors whose TTL has elapsed
/// and returns the count of expired entries.
pub(in crate::server) async fn expire_vectors_handler(
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

/// Get TTL statistics for a collection.
///
/// `GET /collections/:name/ttl` — returns counts of vectors with TTL, expired
/// vectors, earliest/latest expirations, and whether a sweep is recommended.
pub(in crate::server) async fn ttl_stats_handler(
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

/// Get the TTL for a specific vector.
///
/// `GET /collections/:name/vectors/:id/ttl` — returns the expiration timestamp
/// for the vector, or `null` if no TTL is set.
pub(in crate::server) async fn get_vector_ttl(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let ttl = coll.get_ttl(&id);

    Ok(Json(json!({
        "id": id,
        "collection": collection,
        "expiration_timestamp": ttl
    })))
}

/// Set or remove the TTL for a specific vector.
///
/// `PUT /collections/:name/vectors/:id/ttl` — sets the TTL in seconds.
/// Pass `{"ttl_seconds": null}` to remove the TTL.
pub(in crate::server) async fn set_vector_ttl(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
    Json(body): Json<serde_json::Value>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let ttl_seconds = body.get("ttl_seconds").and_then(|v| v.as_u64());

    coll.set_ttl(&id, ttl_seconds)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({
        "id": id,
        "collection": collection,
        "ttl_seconds": ttl_seconds,
        "status": "updated"
    })))
}

/// Get index advisor recommendations with what-if analysis.
///
/// `GET /collections/:name/advise` — returns cost/benefit preview for each index type.
pub(in crate::server) async fn advise_collection_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    use crate::tuning::what_if_analysis;

    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let num_vectors = coll.len();
    let dimensions = coll.dimensions().unwrap_or(0);

    if num_vectors == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                "Collection is empty. Insert vectors first.",
                "EMPTY_COLLECTION",
            )),
        ));
    }

    let analysis = what_if_analysis(num_vectors, dimensions, None, None);
    Ok(Json(json!(analysis)))
}

/// Scan a collection for near-duplicate vectors.
///
/// `POST /collections/:name/dedup/scan` — accepts optional threshold override.
/// Returns groups of duplicate vectors.
pub(in crate::server) async fn dedup_scan_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<Value>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let threshold = req.get("threshold").and_then(|v| v.as_f64()).map(|t| t as f32);

    let result = coll.dedup_scan(threshold)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    Ok(Json(json!(result)))
}

// ============ Stats Handlers ============

/// Get per-field metadata statistics for a collection.
///
/// `GET /collections/:name/stats/fields` — returns cardinality, indexing status,
/// and high-cardinality flag for each metadata field.
pub(in crate::server) async fn collection_field_stats(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let stats = coll.all_field_stats();
    Ok(Json(json!(stats)))
}

/// Get estimated memory usage breakdown for a collection.
///
/// `GET /collections/:name/stats/memory` — returns bytes used by vectors,
/// index, metadata, caches, and total.
pub(in crate::server) async fn collection_memory_usage(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.read().await;
    let coll = db
        .collection(&name)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let stats = coll
        .memory_usage()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    Ok(Json(json!(stats)))
}

#[cfg(test)]
mod tests {
    use super::*;
}
