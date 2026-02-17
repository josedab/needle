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
#[allow(clippy::unwrap_used)]
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
            db.create_collection(name, dims).unwrap();
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
}
