//! HTTP handler functions for the Needle HTTP server.

mod admin;
mod collections;
mod search;
mod vectors;

// Re-export all handlers for use by the server router
pub(super) use admin::*;
pub(super) use collections::*;
pub(super) use search::*;
pub(super) use vectors::*;

use super::auth::{AuthConfig, AuthContext, AuthMethod};
use super::types::*;
use axum::{
    http::StatusCode,
    Json,
};
use serde_json::Value;
use tracing::warn;


/// Maximum number of vectors allowed in a single export response.
pub(in crate::server) const MAX_EXPORT_VECTORS: usize = 100_000;
pub(in crate::server) const MAX_DIMENSIONS: usize = 65_536;
/// Maximum allowed k for search operations.
pub(in crate::server) const MAX_SEARCH_K: usize = 10_000;

/// HTML-escape a string to prevent XSS when interpolating into HTML.
pub(in crate::server) fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
/// Maximum metadata size per vector in bytes (64KB).
const MAX_METADATA_BYTES: usize = 64 * 1024;
/// Maximum metadata JSON nesting depth.
const MAX_METADATA_DEPTH: usize = 10;
/// Maximum vector ID length in bytes.
const MAX_VECTOR_ID_BYTES: usize = 1024;

/// Validate vector dimensions are within the allowed maximum.
pub(in crate::server) fn validate_vector_dimensions(vector: &[f32]) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
    if vector.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                "Vector must not be empty",
                "INVALID_VECTOR",
            )),
        ));
    }
    if vector.len() > MAX_DIMENSIONS {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!(
                    "Vector dimensions {} exceed maximum allowed {}",
                    vector.len(),
                    MAX_DIMENSIONS
                ),
                "DIMENSIONS_TOO_LARGE",
            )),
        ));
    }
    Ok(())
}

/// Validate that a vector ID is safe: non-empty, no control characters, bounded length.
pub(in crate::server) fn validate_vector_id(id: &str) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
    if id.is_empty() || id.len() > MAX_VECTOR_ID_BYTES {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!("Vector ID must be between 1 and {MAX_VECTOR_ID_BYTES} bytes"),
                "INVALID_VECTOR_ID",
            )),
        ));
    }
    if id.chars().any(|c| c.is_control()) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                "Vector ID must not contain control characters",
                "INVALID_VECTOR_ID",
            )),
        ));
    }
    Ok(())
}

/// Validate that a collection name is safe: alphanumeric, underscore, hyphen, 1-256 chars.
pub(in crate::server) fn validate_collection_name(name: &str) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
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
pub(in crate::server) fn validate_metadata(metadata: &Option<Value>) -> std::result::Result<(), (StatusCode, Json<ApiError>)> {
    if let Some(value) = metadata {
        // Estimate JSON size without full re-serialization
        let estimated_size = json_byte_size(value);
        if estimated_size > MAX_METADATA_BYTES {
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

/// Estimate the serialized byte size of a JSON value without allocating a string.
fn json_byte_size(value: &Value) -> usize {
    match value {
        Value::Null => 4,   // "null"
        Value::Bool(b) => if *b { 4 } else { 5 }, // "true" or "false"
        Value::Number(n) => {
            // Estimate number length (format as string length)
            let s = n.to_string();
            s.len()
        }
        Value::String(s) => s.len() + 2, // quotes + content (conservative for escapes)
        Value::Array(arr) => {
            // brackets + commas + element sizes
            2 + arr.iter().map(json_byte_size).sum::<usize>()
                + arr.len().saturating_sub(1) // commas
        }
        Value::Object(map) => {
            // braces + commas + key/value sizes
            2 + map.iter().map(|(k, v)| k.len() + 2 + 1 + json_byte_size(v)).sum::<usize>()
                + map.len().saturating_sub(1) // commas
        }
    }
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
        warn!(
            user_id = %user.id,
            permission = ?permission,
            collection = %collection,
            "Permission denied on collection"
        );
        Err((
            StatusCode::FORBIDDEN,
            Json(ApiError::new(
                "Access denied: insufficient permissions",
                "PERMISSION_DENIED",
            )),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::Database;
    use axum::extract::State;

    fn make_state() -> std::sync::Arc<super::super::AppState> {
        let db = Database::in_memory();
        std::sync::Arc::new(super::super::AppState::new(db))
    }

    async fn make_state_with_collection(name: &str, dims: usize) -> std::sync::Arc<super::super::AppState> {
        let state = make_state();
        {
            let db = state.db.write().await;
            db.create_collection(name, dims).expect("test collection creation should succeed");
        }
        state
    }

    // ── get_collection: 404 for missing collection ───────────────────────

    #[tokio::test]
    async fn test_get_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = get_collection(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "COLLECTION_NOT_FOUND");
            }
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── delete_collection: 404 for missing collection ────────────────────

    #[tokio::test]
    async fn test_delete_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = delete_collection(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── create_collection: 409 for duplicate ─────────────────────────────

    #[tokio::test]
    async fn test_create_collection_duplicate() -> Result<(), Box<dyn std::error::Error>> {
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
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── insert_vector: 404 for missing collection ────────────────────────

    #[tokio::test]
    async fn test_insert_vector_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = insert_vector(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── insert_vector: 400 for dimension mismatch ────────────────────────

    #[tokio::test]
    async fn test_insert_vector_dimension_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = insert_vector(
            State(state),
            axum::extract::Path("test".to_string()),
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
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── insert_vector: 409 for duplicate ID ──────────────────────────────

    #[tokio::test]
    async fn test_insert_vector_duplicate() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        // First insert
        let _ = insert_vector(
            State(std::sync::Arc::clone(&state)),
            axum::extract::Path("test".to_string()),
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
            axum::extract::Path("test".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::CONFLICT),
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── get_vector: 404 for missing vector ───────────────────────────────

    #[tokio::test]
    async fn test_get_vector_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = get_vector(
            State(state),
            axum::extract::Path(("test".to_string(), "nonexistent".to_string())),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "VECTOR_NOT_FOUND");
            }
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── get_vector: 404 for missing collection ───────────────────────────

    #[tokio::test]
    async fn test_get_vector_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = get_vector(
            State(state),
            axum::extract::Path(("nonexistent".to_string(), "v1".to_string())),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── delete_vector: 404 for missing vector ────────────────────────────

    #[tokio::test]
    async fn test_delete_vector_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = delete_vector(
            State(state),
            axum::extract::Path(("test".to_string(), "nonexistent".to_string())),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "VECTOR_NOT_FOUND");
            }
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── search: 404 for missing collection ───────────────────────────────

    #[tokio::test]
    async fn test_search_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = search(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
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
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── search: 400 for dimension mismatch ───────────────────────────────

    #[tokio::test]
    async fn test_search_dimension_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }

        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
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
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── update_metadata: 404 for missing vector ──────────────────────────

    #[tokio::test]
    async fn test_update_metadata_vector_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = update_metadata(
            State(state),
            axum::extract::Path(("test".to_string(), "nonexistent".to_string())),
            Json(UpdateMetadataRequest {
                metadata: Some(serde_json::json!({"key": "val"})),
            }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert_eq!(err.code, "VECTOR_NOT_FOUND");
            }
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── compact: 404 for missing collection ──────────────────────────────

    #[tokio::test]
    async fn test_compact_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = compact_collection(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── batch_insert: 413 for batch too large ────────────────────────────

    #[tokio::test]
    async fn test_batch_insert_too_large() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let vectors: Vec<InsertRequest> = (0..100_001).map(|i| InsertRequest {
            id: format!("v{}", i),
            vector: vec![0.0; 4],
            metadata: None,
            ttl_seconds: None,
        }).collect();

        let result = batch_insert(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(BatchInsertRequest { vectors }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
                assert_eq!(err.code, "BATCH_TOO_LARGE");
            }
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── export_collection: 404 for missing collection ────────────────────

    #[tokio::test]
    async fn test_export_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = export_collection(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }

        Ok(())
    }

    // ── create_collection: success with valid payload ────────────────────

    #[tokio::test]
    async fn test_create_collection_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = create_collection(
            State(state.clone()),
            Json(CreateCollectionRequest {
                name: "test".to_string(),
                dimensions: 128,
                distance: None,
                m: None,
                ef_construction: None,
            }),
        ).await;
        assert!(result.is_ok());

        let db = state.db.read().await;
        assert!(db.has_collection("test"));
        Ok(())
    }

    // ── create_collection: 400 for invalid name ──────────────────────────

    #[tokio::test]
    async fn test_create_collection_invalid_name() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = create_collection(
            State(state),
            Json(CreateCollectionRequest {
                name: "bad name!@#".to_string(),
                dimensions: 4,
                distance: None,
                m: None,
                ef_construction: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::BAD_REQUEST),
            Ok(_) => return Err("Expected error for invalid name".into()),
        }
        Ok(())
    }

    // ── create_collection: 400 for zero dimensions ───────────────────────

    #[tokio::test]
    async fn test_create_collection_zero_dimensions() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = create_collection(
            State(state),
            Json(CreateCollectionRequest {
                name: "test".to_string(),
                dimensions: 0,
                distance: None,
                m: None,
                ef_construction: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::BAD_REQUEST),
            Ok(_) => return Err("Expected error for zero dimensions".into()),
        }
        Ok(())
    }

    // ── insert_vector: success with valid payload ────────────────────────

    #[tokio::test]
    async fn test_insert_vector_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = insert_vector(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: Some(serde_json::json!({"key": "value"})),
                ttl_seconds: None,
            }),
        ).await;
        assert!(result.is_ok());

        // Verify vector was inserted
        let db = state.db.read().await;
        let coll = db.collection("test").expect("test collection should exist");
        assert!(coll.get("v1").is_some());
        Ok(())
    }

    // ── batch_insert: success with valid batch ───────────────────────────

    #[tokio::test]
    async fn test_batch_insert_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let vectors = vec![
            InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            },
            InsertRequest {
                id: "v2".to_string(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            },
        ];
        let result = batch_insert(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
            Json(BatchInsertRequest { vectors }),
        ).await;
        assert!(result.is_ok());

        let db = state.db.read().await;
        let coll = db.collection("test").expect("test collection should exist");
        assert!(coll.get("v1").is_some());
        assert!(coll.get("v2").is_some());
        Ok(())
    }

    // ── batch_insert: 404 for missing collection ─────────────────────────

    #[tokio::test]
    async fn test_batch_insert_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = batch_insert(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            Json(BatchInsertRequest { vectors: vec![
                InsertRequest {
                    id: "v1".to_string(),
                    vector: vec![1.0],
                    metadata: None,
                    ttl_seconds: None,
                },
            ] }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── batch_insert: dimension mismatch rejected early ────────────────

    #[tokio::test]
    async fn test_batch_insert_dimension_mismatch() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let vectors = vec![
            InsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            },
            InsertRequest {
                id: "v2".to_string(),
                vector: vec![1.0, 0.0], // wrong dims
                metadata: None,
                ttl_seconds: None,
            },
        ];
        let result = batch_insert(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
            Json(BatchInsertRequest { vectors }),
        ).await;
        // Dimension mismatch is now rejected at the API layer before any inserts
        assert!(result.is_err(), "Batch insert should fail with dimension mismatch");
        if let Err((status, _)) = result {
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }
        Ok(())
    }

    // ── search: empty collection returns ok ─────────────────────────────

    #[tokio::test]
    async fn test_search_empty_collection() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
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
        assert!(result.is_ok(), "Search on empty collection should succeed");
        Ok(())
    }

    // ── search: with valid filter ────────────────────────────────────────

    #[tokio::test]
    async fn test_search_with_filter() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(serde_json::json!({"category": "a"})))?;
            coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(serde_json::json!({"category": "b"})))?;
        }

        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 10,
                filter: Some(serde_json::json!({"category": "a"})),
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: false,
                distance: None,
                explain: false,
            }),
        ).await;
        assert!(result.is_ok(), "Search with filter should succeed");
        Ok(())
    }

    // ── delete_vector: success for existing vector ───────────────────────

    #[tokio::test]
    async fn test_delete_vector_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }
        let result = delete_vector(
            State(state.clone()),
            axum::extract::Path(("test".to_string(), "v1".to_string())),
        ).await;
        assert!(result.is_ok());

        let db = state.db.read().await;
        let coll = db.collection("test")?;
        assert!(coll.get("v1").is_none());
        Ok(())
    }

    // ── health endpoint ──────────────────────────────────────────────────

    #[tokio::test]
    async fn test_health_endpoint() {
        let _response = health().await;
        // health() returns impl IntoResponse (Json); just verify it doesn't panic
    }

    // ── save_database endpoint ───────────────────────────────────────────

    #[tokio::test]
    async fn test_save_database_in_memory() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = save_database(State(state)).await;
        // in-memory databases may succeed or fail depending on implementation
        // but should not panic
        let _ = result;
        Ok(())
    }

    // ── error response format validation ─────────────────────────────────

    #[tokio::test]
    async fn test_error_response_format() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = get_collection(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, Json(err))) => {
                assert_eq!(status, StatusCode::NOT_FOUND);
                assert!(!err.error.is_empty());
                assert!(!err.code.is_empty());
                assert_eq!(err.code, "COLLECTION_NOT_FOUND");
            }
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── text_to_deterministic_vector consistency ─────────────────────────

    #[test]
    fn test_text_to_deterministic_vector_consistency() {
        let v1 = text_to_deterministic_vector("hello world", 128);
        let v2 = text_to_deterministic_vector("hello world", 128);
        assert_eq!(v1.len(), 128);
        assert_eq!(v1, v2, "Same input should produce same vector");

        let v3 = text_to_deterministic_vector("different text", 128);
        assert_ne!(v1, v3, "Different input should produce different vector");
    }

    #[test]
    fn test_text_to_deterministic_vector_normalization() {
        let v = text_to_deterministic_vector("test normalization", 64);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Vector should be unit normalized, got {}", norm);
    }

    #[test]
    fn test_text_to_deterministic_vector_empty_input() {
        let v = text_to_deterministic_vector("", 32);
        assert_eq!(v.len(), 32);
        // Empty text should still produce a valid vector
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(norm > 0.0 || v.iter().all(|&x| x == 0.0));
    }

    // ── validate_collection_name edge cases ──────────────────────────────

    #[test]
    fn test_validate_collection_name_empty() {
        assert!(validate_collection_name("").is_err());
    }

    #[test]
    fn test_validate_collection_name_too_long() {
        let name = "a".repeat(257);
        assert!(validate_collection_name(&name).is_err());
    }

    #[test]
    fn test_validate_collection_name_valid() {
        assert!(validate_collection_name("my-collection_123").is_ok());
    }

    // ── search: with explain enabled ─────────────────────────────────────

    #[tokio::test]
    async fn test_search_with_explain() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }

        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 1,
                filter: None,
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: false,
                distance: None,
                explain: true,
            }),
        ).await;
        assert!(result.is_ok(), "Search with explain should succeed");
        Ok(())
    }

    // ── html_escape test ─────────────────────────────────────────────────

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<script>"), "&lt;script&gt;");
        assert_eq!(html_escape("a&b"), "a&amp;b");
        assert_eq!(html_escape("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(html_escape("safe text"), "safe text");
    }

    // ── list_collections handler ─────────────────────────────────────────

    #[tokio::test]
    async fn test_list_collections() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let _response = list_collections(State(state)).await;
        // Returns impl IntoResponse, verify no panic
        Ok(())
    }

    // ── get_collection: success for existing ─────────────────────────────

    #[tokio::test]
    async fn test_get_collection_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = get_collection(
            State(state),
            axum::extract::Path("test".to_string()),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── delete_collection: success for existing ──────────────────────────

    #[tokio::test]
    async fn test_delete_collection_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = delete_collection(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
        ).await;
        assert!(result.is_ok());

        let db = state.db.read().await;
        assert!(!db.has_collection("test"));
        Ok(())
    }

    // ── search with include_vectors ──────────────────────────────────────

    #[tokio::test]
    async fn test_search_include_vectors() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }

        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 1,
                filter: None,
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: true,
                distance: None,
                explain: false,
            }),
        ).await;
        assert!(result.is_ok(), "Search with include_vectors should succeed");
        Ok(())
    }

    // ── upsert_vector: insert then update ────────────────────────────────

    #[tokio::test]
    async fn test_upsert_vector_insert() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = upsert_vector(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
            Json(UpsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        assert!(result.is_ok());
        let db = state.db.read().await;
        let coll = db.collection("test")?;
        assert!(coll.get("v1").is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_vector_update_existing() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }
        let result = upsert_vector(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
            Json(UpsertRequest {
                id: "v1".to_string(),
                vector: vec![0.0, 1.0, 0.0, 0.0],
                metadata: Some(serde_json::json!({"updated": true})),
                ttl_seconds: None,
            }),
        ).await;
        assert!(result.is_ok());
        let db = state.db.read().await;
        let coll = db.collection("test")?;
        let (vec, meta) = coll.get("v1").expect("vector v1 should exist");
        assert!((vec[1] - 1.0).abs() < 0.001);
        assert!(meta.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_upsert_vector_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = upsert_vector(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            Json(UpsertRequest {
                id: "v1".to_string(),
                vector: vec![1.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── batch_search ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_batch_search_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
            coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
        }
        let result = batch_search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(BatchSearchRequest {
                vectors: vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.0, 1.0, 0.0, 0.0],
                ],
                k: 1,
                filter: None,
            }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_search_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = batch_search(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            Json(BatchSearchRequest {
                vectors: vec![vec![1.0]],
                k: 5,
                filter: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── radius_search ────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_radius_search_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }
        let result = radius_search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(RadiusSearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                max_distance: 0.5,
                limit: 100,
                filter: None,
                include_vectors: false,
            }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_radius_search_invalid_limit() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = radius_search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(RadiusSearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                max_distance: 0.5,
                limit: 0,
                filter: None,
                include_vectors: false,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::BAD_REQUEST),
            Ok(_) => return Err("Expected error for limit=0".into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_radius_search_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = radius_search(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            Json(RadiusSearchRequest {
                vector: vec![1.0],
                max_distance: 1.0,
                limit: 10,
                filter: None,
                include_vectors: false,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── streaming_insert_handler ─────────────────────────────────────────

    #[tokio::test]
    async fn test_streaming_insert_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let _response = streaming_insert_handler(
            State(state.clone()),
            axum::extract::Path("test".to_string()),
            Json(StreamingInsertRequest {
                vectors: vec![
                    StreamingVector {
                        id: "s1".to_string(),
                        vector: vec![1.0, 0.0, 0.0, 0.0],
                        metadata: None,
                    },
                    StreamingVector {
                        id: "s2".to_string(),
                        vector: vec![0.0, 1.0, 0.0, 0.0],
                        metadata: Some(serde_json::json!({"key": "val"})),
                    },
                ],
                sequence_id: None,
                flush: false,
            }),
        ).await;
        let db = state.db.read().await;
        let coll = db.collection("test")?;
        assert!(coll.get("s1").is_some());
        assert!(coll.get("s2").is_some());
        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_insert_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let _response = streaming_insert_handler(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            Json(StreamingInsertRequest {
                vectors: vec![StreamingVector {
                    id: "s1".to_string(),
                    vector: vec![1.0],
                    metadata: None,
                }],
                sequence_id: None,
                flush: false,
            }),
        ).await;
        // Returns impl IntoResponse; verify no panic
        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_insert_partial_failure() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let _response = streaming_insert_handler(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(StreamingInsertRequest {
                vectors: vec![
                    StreamingVector {
                        id: "ok".to_string(),
                        vector: vec![1.0, 0.0, 0.0, 0.0],
                        metadata: None,
                    },
                    StreamingVector {
                        id: "bad".to_string(),
                        vector: vec![1.0], // wrong dims
                        metadata: None,
                    },
                ],
                sequence_id: None,
                flush: false,
            }),
        ).await;
        Ok(())
    }

    // ── list_vectors ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_list_vectors_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
            coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
        }
        let result = list_vectors(
            State(state),
            axum::extract::Path("test".to_string()),
            axum::extract::Query(QueryParams { offset: None, limit: None }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_list_vectors_collection_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = list_vectors(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
            axum::extract::Query(QueryParams { offset: None, limit: None }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── export_collection with data ──────────────────────────────────────

    #[tokio::test]
    async fn test_export_collection_with_data() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(serde_json::json!({"k": "v"})))?;
        }
        let result = export_collection(
            State(state),
            axum::extract::Path("test".to_string()),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── alias handlers ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_create_alias() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = create_alias_handler(
            State(state),
            Json(CreateAliasRequest {
                alias: "my_alias".to_string(),
                collection: "test".to_string(),
            }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    #[tokio::test]
    async fn test_create_alias_nonexistent_collection() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = create_alias_handler(
            State(state),
            Json(CreateAliasRequest {
                alias: "my_alias".to_string(),
                collection: "nonexistent".to_string(),
            }),
        ).await;
        assert!(result.is_err());
        Ok(())
    }

    #[tokio::test]
    async fn test_get_alias_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = get_alias_handler(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_delete_alias_not_found() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = delete_alias_handler(
            State(state),
            axum::extract::Path("nonexistent".to_string()),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── get_vector: success ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_vector_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(serde_json::json!({"key": "val"})))?;
        }
        let result = get_vector(
            State(state),
            axum::extract::Path(("test".to_string(), "v1".to_string())),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── compact_collection: success ──────────────────────────────────────

    #[tokio::test]
    async fn test_compact_collection_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
            coll.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
            coll.delete("v1")?;
        }
        let result = compact_collection(
            State(state),
            axum::extract::Path("test".to_string()),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── get_info endpoint ────────────────────────────────────────────────

    #[tokio::test]
    async fn test_get_info() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let _response = get_info(State(state)).await;
        Ok(())
    }

    // ── create_collection: with custom distance ──────────────────────────

    #[tokio::test]
    async fn test_create_collection_with_distance() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = create_collection(
            State(state),
            Json(CreateCollectionRequest {
                name: "test".to_string(),
                dimensions: 128,
                distance: Some("euclidean".to_string()),
                m: Some(32),
                ef_construction: Some(400),
            }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── create_collection: dimensions too large ──────────────────────────

    #[tokio::test]
    async fn test_create_collection_dimensions_too_large() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = create_collection(
            State(state),
            Json(CreateCollectionRequest {
                name: "test".to_string(),
                dimensions: MAX_DIMENSIONS + 1,
                distance: None,
                m: None,
                ef_construction: None,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::BAD_REQUEST),
            Ok(_) => return Err("Expected error for dimensions too large".into()),
        }
        Ok(())
    }

    // ── update_metadata: success ─────────────────────────────────────────

    #[tokio::test]
    async fn test_update_metadata_success() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        {
            let db = state.db.write().await;
            let coll = db.collection("test")?;
            coll.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
        }
        let result = update_metadata(
            State(state),
            axum::extract::Path(("test".to_string(), "v1".to_string())),
            Json(UpdateMetadataRequest {
                metadata: Some(serde_json::json!({"new_key": "new_val"})),
            }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── search: k=0 ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_search_k_zero() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 0,
                filter: None,
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: false,
                distance: None,
                explain: false,
            }),
        ).await;
        // k=0 should either be rejected or return empty
        let _ = result;
        Ok(())
    }

    // ── search: k too large ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_search_k_too_large() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: MAX_SEARCH_K + 1,
                filter: None,
                post_filter: None,
                post_filter_factor: 3,
                include_vectors: false,
                distance: None,
                explain: false,
            }),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::BAD_REQUEST),
            Ok(_) => return Err("Expected error for k too large".into()),
        }
        Ok(())
    }

    // ── delete_vector: from nonexistent collection ───────────────────────

    #[tokio::test]
    async fn test_delete_vector_nonexistent_collection() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state();
        let result = delete_vector(
            State(state),
            axum::extract::Path(("nonexistent".to_string(), "v1".to_string())),
        ).await;
        match result {
            Err((status, _)) => assert_eq!(status, StatusCode::NOT_FOUND),
            Ok(_) => return Err("Expected error".into()),
        }
        Ok(())
    }

    // ── batch_insert: empty batch ────────────────────────────────────────

    #[tokio::test]
    async fn test_batch_insert_empty() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = batch_insert(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(BatchInsertRequest { vectors: vec![] }),
        ).await;
        assert!(result.is_ok());
        Ok(())
    }

    // ── validate_vector_id ──────────────────────────────────────────────

    #[test]
    fn test_validate_vector_id_empty() {
        assert!(validate_vector_id("").is_err());
    }

    #[test]
    fn test_validate_vector_id_too_long() {
        let id = "a".repeat(1025);
        assert!(validate_vector_id(&id).is_err());
    }

    #[test]
    fn test_validate_vector_id_control_chars() {
        assert!(validate_vector_id("bad\x00id").is_err());
        assert!(validate_vector_id("bad\nid").is_err());
    }

    #[test]
    fn test_validate_vector_id_valid() {
        assert!(validate_vector_id("valid-id_123").is_ok());
        assert!(validate_vector_id("a").is_ok());
    }

    // ── validate_collection_name ────────────────────────────────────────

    #[test]
    fn test_validate_collection_name_special_chars() {
        assert!(validate_collection_name("has spaces").is_err());
        assert!(validate_collection_name("has.dots").is_err());
        assert!(validate_collection_name("has/slash").is_err());
    }

    #[test]
    fn test_validate_collection_name_max_boundary() {
        let name = "a".repeat(256);
        assert!(validate_collection_name(&name).is_ok());
    }

    // ── validate_metadata ───────────────────────────────────────────────

    #[test]
    fn test_validate_metadata_none() {
        assert!(validate_metadata(&None).is_ok());
    }

    #[test]
    fn test_validate_metadata_small() {
        let meta = Some(serde_json::json!({"key": "value"}));
        assert!(validate_metadata(&meta).is_ok());
    }

    #[test]
    fn test_validate_metadata_too_large() {
        let big = "x".repeat(MAX_METADATA_BYTES + 1);
        let meta = Some(serde_json::json!({"data": big}));
        assert!(validate_metadata(&meta).is_err());
    }

    #[test]
    fn test_validate_metadata_too_deep() {
        // Build deeply nested JSON exceeding MAX_METADATA_DEPTH
        let mut value = serde_json::json!("leaf");
        for _ in 0..MAX_METADATA_DEPTH + 1 {
            value = serde_json::json!({"nested": value});
        }
        let meta = Some(value);
        assert!(validate_metadata(&meta).is_err());
    }

    // ── json_depth ──────────────────────────────────────────────────────

    #[test]
    fn test_json_depth_scalar() {
        assert_eq!(json_depth(&serde_json::json!(42)), 0);
        assert_eq!(json_depth(&serde_json::json!("hello")), 0);
    }

    #[test]
    fn test_json_depth_flat_object() {
        assert_eq!(json_depth(&serde_json::json!({"a": 1, "b": 2})), 1);
    }

    #[test]
    fn test_json_depth_nested() {
        assert_eq!(json_depth(&serde_json::json!({"a": {"b": {"c": 1}}})), 3);
    }

    #[test]
    fn test_json_depth_array() {
        assert_eq!(json_depth(&serde_json::json!([1, [2, [3]]])), 3);
    }

    // ── insert_vector: 400 for NaN vector ───────────────────────────────

    #[tokio::test]
    async fn test_insert_vector_nan() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = insert_vector(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(InsertRequest {
                id: "v1".to_string(),
                vector: vec![f32::NAN, 0.0, 0.0, 0.0],
                metadata: None,
                ttl_seconds: None,
            }),
        ).await;
        match result {
            Err((status, err)) => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(err.code, "INVALID_VECTOR");
            }
            Ok(_) => return Err("Expected error for NaN vector".into()),
        }
        Ok(())
    }

    // ── search: k=0 should return 400 ──────────────────────────────────

    #[tokio::test]
    async fn test_search_k_zero_validation() -> Result<(), Box<dyn std::error::Error>> {
        let state = make_state_with_collection("test", 4).await;
        let result = search(
            State(state),
            axum::extract::Path("test".to_string()),
            Json(SearchRequest {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                k: 0,
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
                assert_eq!(err.code, "INVALID_K");
            }
            Ok(_) => return Err("Expected error for k=0".into()),
        }
        Ok(())
    }
}
