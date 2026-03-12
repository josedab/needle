//! Vector CRUD and text embedding handlers.

use crate::metadata::Filter;
use crate::server::AppState;
use crate::server::types::*;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::{error, warn};

use super::{validate_metadata, validate_vector_id, validate_vector_dimensions};

/// Maximum allowed size of a single text input in bytes.
const MAX_TEXT_BYTES: usize = 100_000;

// ============ Vector CRUD ============

/// Insert a single vector into a collection.
///
/// `POST /collections/:name/vectors` — accepts id, vector, optional metadata
/// and optional TTL. Returns `201 Created`.
pub(in crate::server) async fn insert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<InsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    validate_vector_id(&req.id)?;
    validate_metadata(&req.metadata)?;
    validate_vector_dimensions(&req.vector)?;

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if let Some(expected) = coll.dimensions() {
        if req.vector.len() != expected {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        expected,
                        req.vector.len()
                    ),
                    "DIMENSION_MISMATCH",
                )),
            ));
        }
    }

    coll.insert_with_ttl(&req.id, &req.vector, req.metadata, req.ttl_seconds)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"inserted": req.id}))))
}

/// Batch insert vectors
pub(in crate::server) async fn batch_insert(
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
        validate_vector_id(&item.id)?;
        validate_metadata(&item.metadata)?;
        validate_vector_dimensions(&item.vector)?;
    }

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if let Some(expected) = coll.dimensions() {
        for item in &req.vectors {
            if item.vector.len() != expected {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(
                        format!(
                            "Vector '{}' dimension mismatch: expected {}, got {}",
                            item.id, expected, item.vector.len()
                        ),
                        "DIMENSION_MISMATCH",
                    )),
                ));
            }
        }
    }

    let mut inserted = 0;
    let mut errors = Vec::new();

    for item in req.vectors {
        match coll.insert_with_ttl(&item.id, &item.vector, item.metadata, item.ttl_seconds) {
            Ok(_) => inserted += 1,
            Err(e) => {
                warn!(id = %item.id, error = %e, "Batch insert failed for vector");
                errors.push(json!({"id": item.id, "error": e.to_string()}));
            }
        }
    }

    Ok(Json(json!({
        "inserted": inserted,
        "errors": errors,
    })))
}

/// Insert or update a vector (upsert).
///
/// `PUT /collections/:name/vectors` — if the ID already exists, deletes and
/// re-inserts. Returns `{"updated": true}` if it replaced an existing vector.
pub(in crate::server) async fn upsert_vector(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<UpsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    validate_metadata(&req.metadata)?;
    validate_vector_dimensions(&req.vector)?;

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if let Some(expected) = coll.dimensions() {
        if req.vector.len() != expected {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(ApiError::new(
                    format!(
                        "Vector dimension mismatch: expected {}, got {}",
                        expected,
                        req.vector.len()
                    ),
                    "DIMENSION_MISMATCH",
                )),
            ));
        }
    }

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

/// Retrieve a vector and its metadata by ID.
///
/// `GET /collections/:name/vectors/:id` — returns the vector, metadata, and ID.
/// Returns 404 if not found.
pub(in crate::server) async fn get_vector(
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

/// Delete a vector by ID.
///
/// `DELETE /collections/:name/vectors/:id` — returns 404 if not found.
pub(in crate::server) async fn delete_vector(
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
        Ok(StatusCode::NO_CONTENT.into_response())
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

/// Update the metadata of an existing vector.
///
/// `POST /collections/:name/vectors/:id/metadata` — by default performs a
/// JSON merge patch (new keys added, null keys removed). Pass `"replace": true`
/// to fully replace metadata.
pub(in crate::server) async fn update_metadata(
    State(state): State<Arc<AppState>>,
    Path((collection, id)): Path<(String, String)>,
    Json(req): Json<UpdateMetadataRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    // Get existing vector and save original metadata for rollback
    let (vector, original_metadata) = coll.get(&id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            Json(ApiError::new(
                format!("Vector '{}' not found", id),
                "VECTOR_NOT_FOUND".to_string(),
            )),
        )
    })?;

    let new_metadata = if req.replace {
        // Full replace mode
        req.metadata
    } else if let Some(ref patch) = req.metadata {
        // Merge patch mode: merge new fields into existing metadata
        match (&original_metadata, patch) {
            (Some(Value::Object(existing)), Value::Object(patch_map)) => {
                let mut merged = existing.clone();
                for (key, value) in patch_map {
                    if value.is_null() {
                        merged.remove(key);
                    } else {
                        merged.insert(key.clone(), value.clone());
                    }
                }
                Some(Value::Object(merged))
            }
            _ => req.metadata,
        }
    } else {
        original_metadata.clone()
    };

    // Delete and re-insert with new metadata; rollback on insert failure
    coll.delete(&id)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    if let Err(e) = coll.insert(&id, &vector, new_metadata) {
        // Rollback: re-insert with original data
        warn!(id = %id, error = %e, "Metadata update insert failed, rolling back");
        if let Err(rollback_err) = coll.insert(&id, &vector, original_metadata) {
            error!(id = %id, error = %rollback_err, "Rollback failed — vector may be missing from collection");
        }
        return Err(Into::<(StatusCode, Json<ApiError>)>::into(e));
    }

    Ok(Json(json!({"updated": id})))
}

/// Streaming batch insert with backpressure signalling.
///
/// `POST /collections/:name/vectors/stream` — inserts vectors in streaming
/// fashion, returning per-batch acknowledgements with latency and
/// backpressure indicators.
pub(in crate::server) async fn streaming_insert_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<StreamingInsertRequest>,
) -> impl IntoResponse {
    let start = std::time::Instant::now();
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!(ApiError::new(e.to_string(), "NOT_FOUND")))),
    };

    let total = body.vectors.len();
    let mut inserted = 0usize;
    let mut errors = Vec::new();

    for v in &body.vectors {
        match coll.insert(&v.id, &v.vector, v.metadata.clone()) {
            Ok(()) => inserted += 1,
            Err(e) => {
                warn!(id = %v.id, error = %e, "Batch insert failed for vector");
                errors.push(json!({ "id": v.id, "error": e.to_string() }));
            }
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

// ============ Text Handlers ============

/// Insert a vector by providing raw text instead of a pre-computed embedding.
///
/// `POST /collections/:name/texts` — embeds the text via the configured
/// embedding provider (or deterministic hash fallback) and inserts the
/// resulting vector with enriched metadata.
pub(in crate::server) async fn insert_text_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<InsertTextRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!(ApiError::new(e.to_string(), "NOT_FOUND")))),
    };

    if body.text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new("Text cannot be empty", "EMPTY_TEXT"))));
    }

    if body.text.len() > MAX_TEXT_BYTES {
        return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new(
            format!("Text exceeds maximum size of {MAX_TEXT_BYTES} bytes"),
            "TEXT_TOO_LARGE",
        ))));
    }

    let dims = match coll.dimensions() {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new("Collection has no dimensions", "INVALID_COLLECTION")))),
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
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!(ApiError::new(e.to_string(), "BAD_REQUEST")))),
    }
}

/// Embed text using the configured provider or deterministic hash fallback.
pub(in crate::server) async fn embed_text(state: &AppState, text: &str, dims: usize) -> (Vec<f32>, String) {
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

/// Batch insert vectors from text, embedding each via the configured provider.
///
/// `POST /collections/:name/texts/batch` — accepts an array of text documents,
/// embeds them, and inserts the resulting vectors. Returns per-item results.
pub(in crate::server) async fn batch_insert_text_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<BatchInsertTextRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!(ApiError::new(e.to_string(), "NOT_FOUND")))),
    };

    let dims = match coll.dimensions() {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new("Collection has no dimensions", "INVALID_COLLECTION")))),
    };

    if body.texts.len() > 1000 {
        return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new("Batch size exceeds limit of 1000", "BATCH_TOO_LARGE"))));
    }

    let mut inserted = 0usize;
    let mut errors = Vec::new();
    let mut embed_method = String::from("deterministic_hash");

    for item in &body.texts {
        if item.text.is_empty() {
            errors.push(json!({ "id": item.id, "error": "Empty text" }));
            continue;
        }

        if item.text.len() > MAX_TEXT_BYTES {
            errors.push(json!({ "id": item.id, "error": format!("Text exceeds maximum size of {} bytes", MAX_TEXT_BYTES) }));
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
            Err(e) => {
                warn!(id = %item.id, error = %e, "Batch embed insert failed for vector");
                errors.push(json!({ "id": item.id, "error": e.to_string() }));
            }
        }
    }

    (StatusCode::OK, Json(json!({
        "inserted": inserted,
        "total": body.texts.len(),
        "errors": errors,
        "embed_method": embed_method,
    })))
}

/// Search a collection using a text query instead of a raw vector.
///
/// `POST /collections/:name/texts/search` — embeds the query text and performs
/// ANN search, returning results with relevance scores.
pub(in crate::server) async fn search_text_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<TextSearchRequest>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!(ApiError::new(e.to_string(), "NOT_FOUND")))),
    };

    let dims = match coll.dimensions() {
        Some(d) => d,
        None => return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new("Collection has no dimensions", "INVALID_COLLECTION")))),
    };

    if body.text.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new("Query text cannot be empty", "EMPTY_TEXT"))));
    }

    let (query_vector, _) = embed_text(&state, &body.text, dims).await;

    let results = if let Some(filter_value) = &body.filter {
        match Filter::parse(filter_value) {
            Ok(filter) => coll.search_with_filter(&query_vector, body.k, &filter),
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!(ApiError::new(format!("Invalid filter: {e}"), "INVALID_FILTER")))),
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
        Err(e) => (StatusCode::BAD_REQUEST, Json(json!(ApiError::new(e.to_string(), "BAD_REQUEST")))),
    }
}

/// Generate a deterministic pseudo-embedding from text using hash-based projection.
/// This provides consistent vectors for the same text input, enabling basic text
/// search without an external embedding provider.
pub(in crate::server) fn text_to_deterministic_vector(text: &str, dimensions: usize) -> Vec<f32> {
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

/// Insert text using the collection's auto-embed provider.
///
/// `POST /collections/:name/texts/auto` — embeds text via the collection's
/// configured auto-embed provider and inserts the resulting vector.
pub(in crate::server) async fn insert_auto_text(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<AutoTextRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    coll.insert_auto_text(&body.id, &body.text, body.metadata)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok((StatusCode::CREATED, Json(json!({"inserted": body.id}))))
}

/// Delete multiple vectors by ID.
///
/// `POST /collections/:name/vectors/delete-batch` — accepts a list of IDs and
/// deletes them. Returns the count of successfully deleted vectors.
/// IDs that don't exist are silently skipped.
pub(in crate::server) async fn batch_delete(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<BatchDeleteRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    if req.ids.len() > state.max_batch_size {
        return Err((
            StatusCode::PAYLOAD_TOO_LARGE,
            Json(ApiError::new(
                format!(
                    "Batch size {} exceeds maximum allowed {}",
                    req.ids.len(),
                    state.max_batch_size
                ),
                "BATCH_TOO_LARGE".to_string(),
            )),
        ));
    }

    for id in &req.ids {
        validate_vector_id(id)?;
    }

    let db = state.db.write().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    let id_refs: Vec<&str> = req.ids.iter().map(String::as_str).collect();
    let deleted_count = coll
        .batch_delete(&id_refs)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    Ok(Json(json!({ "deleted_count": deleted_count })))
}

#[cfg(test)]
mod tests {
    use super::*;
}
