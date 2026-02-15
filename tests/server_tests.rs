//! Integration tests for the HTTP REST API server.
//!
//! Tests cover all REST endpoints, error handling, rate limiting, and CORS configuration.
//! Uses axum-test for HTTP-level testing without starting a real server.
//!
//! Run with: cargo test --test server_tests --features server

#![cfg(feature = "server")]

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use needle::database::Database;
use needle::server::{
    create_router_with_config, AppState, CorsConfig, RateLimitConfig, ServerConfig,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower::ServiceExt;

/// Create test app state with in-memory database
fn create_test_state() -> Arc<AppState> {
    Arc::new(AppState::new(Database::in_memory()))
}

/// Create test router with default config
fn create_test_router(state: Arc<AppState>) -> axum::Router {
    let config = ServerConfig::default();
    create_router_with_config(state, &config)
}

/// Helper to make JSON POST request
async fn post_json(router: &axum::Router, path: &str, body: Value) -> axum::response::Response {
    let request = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    router.clone().oneshot(request).await.unwrap()
}

/// Helper to make GET request
async fn get(router: &axum::Router, path: &str) -> axum::response::Response {
    let request = Request::builder()
        .method(Method::GET)
        .uri(path)
        .body(Body::empty())
        .unwrap();

    router.clone().oneshot(request).await.unwrap()
}

/// Helper to make DELETE request
async fn delete(router: &axum::Router, path: &str) -> axum::response::Response {
    let request = Request::builder()
        .method(Method::DELETE)
        .uri(path)
        .body(Body::empty())
        .unwrap();

    router.clone().oneshot(request).await.unwrap()
}

/// Helper to extract JSON body from response
async fn body_json(response: axum::response::Response) -> Value {
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap_or(json!(null))
}

// ============================================================================
// Health and Info Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_health_endpoint() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = get(&router, "/health").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert_eq!(body["status"], "healthy");
}

#[tokio::test]
async fn test_info_endpoint() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = get(&router, "/info").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    // /info returns collections count and total_vectors (version is on /health)
    assert!(body["collections"].is_number());
    assert!(body["total_vectors"].is_number());
}

#[tokio::test]
async fn test_root_endpoint() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = get(&router, "/").await;
    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Collection Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_list_collections_empty() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = get(&router, "/collections").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert!(body["collections"].is_array());
    assert_eq!(body["collections"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn test_create_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = post_json(
        &router,
        "/collections",
        json!({
            "name": "test_collection",
            "dimensions": 128
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = body_json(response).await;
    // Response is {"created": "collection_name"}
    assert_eq!(body["created"], "test_collection");
}

#[tokio::test]
async fn test_create_collection_with_options() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = post_json(
        &router,
        "/collections",
        json!({
            "name": "custom_collection",
            "dimensions": 256,
            "distance": "euclidean",
            "m": 32,
            "ef_construction": 400
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);

    let body = body_json(response).await;
    // Response is {"created": "collection_name"}
    assert_eq!(body["created"], "custom_collection");
}

#[tokio::test]
async fn test_create_duplicate_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    // Create first
    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "duplicate",
            "dimensions": 64
        }),
    )
    .await;

    // Try to create duplicate
    let response = post_json(
        &router,
        "/collections",
        json!({
            "name": "duplicate",
            "dimensions": 64
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CONFLICT);

    let body = body_json(response).await;
    assert_eq!(body["code"], "COLLECTION_EXISTS");
}

#[tokio::test]
async fn test_get_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    // Create collection
    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "getme",
            "dimensions": 64
        }),
    )
    .await;

    // Get collection info
    let response = get(&router, "/collections/getme").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert_eq!(body["name"], "getme");
    assert_eq!(body["dimensions"], 64);
    assert_eq!(body["count"], 0);
}

#[tokio::test]
async fn test_get_nonexistent_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = get(&router, "/collections/ghost").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);

    let body = body_json(response).await;
    assert_eq!(body["code"], "COLLECTION_NOT_FOUND");
}

#[tokio::test]
async fn test_delete_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    // Create collection
    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "deleteme",
            "dimensions": 64
        }),
    )
    .await;

    // Delete it
    let response = delete(&router, "/collections/deleteme").await;
    assert_eq!(response.status(), StatusCode::OK);

    // Verify it's gone
    let response = get(&router, "/collections/deleteme").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_delete_nonexistent_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = delete(&router, "/collections/ghost").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// Vector Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_insert_vector() {
    let state = create_test_state();
    let router = create_test_router(state);

    // Create collection
    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "vectors",
            "dimensions": 4
        }),
    )
    .await;

    // Insert vector
    let response = post_json(
        &router,
        "/collections/vectors/vectors",
        json!({
            "id": "vec1",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"key": "value"}
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn test_insert_vector_without_metadata() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    let response = post_json(
        &router,
        "/collections/test/vectors",
        json!({
            "id": "simple",
            "vector": [1.0, 2.0, 3.0, 4.0]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CREATED);
}

#[tokio::test]
async fn test_insert_duplicate_vector() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    // First insert
    let _ = post_json(
        &router,
        "/collections/test/vectors",
        json!({
            "id": "dup",
            "vector": [1.0, 2.0, 3.0, 4.0]
        }),
    )
    .await;

    // Duplicate insert
    let response = post_json(
        &router,
        "/collections/test/vectors",
        json!({
            "id": "dup",
            "vector": [5.0, 6.0, 7.0, 8.0]
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::CONFLICT);
}

#[tokio::test]
async fn test_insert_wrong_dimensions() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    let response = post_json(
        &router,
        "/collections/test/vectors",
        json!({
            "id": "wrong",
            "vector": [1.0, 2.0] // Only 2 dimensions, expected 4
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    let body = body_json(response).await;
    assert_eq!(body["code"], "DIMENSION_MISMATCH");
}

#[tokio::test]
async fn test_get_vector() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    let _ = post_json(
        &router,
        "/collections/test/vectors",
        json!({
            "id": "getme",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "metadata": {"tag": "test"}
        }),
    )
    .await;

    let response = get(&router, "/collections/test/vectors/getme").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert_eq!(body["id"], "getme");
    assert!(body["vector"].is_array());
    assert_eq!(body["metadata"]["tag"], "test");
}

#[tokio::test]
async fn test_get_nonexistent_vector() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    let response = get(&router, "/collections/test/vectors/ghost").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
    let body = body_json(response).await;
    assert_eq!(body["code"], "VECTOR_NOT_FOUND");
}

#[tokio::test]
async fn test_delete_vector() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    let _ = post_json(
        &router,
        "/collections/test/vectors",
        json!({
            "id": "deleteme",
            "vector": [0.1, 0.2, 0.3, 0.4]
        }),
    )
    .await;

    let response = delete(&router, "/collections/test/vectors/deleteme").await;
    assert_eq!(response.status(), StatusCode::OK);

    // Verify deletion
    let response = get(&router, "/collections/test/vectors/deleteme").await;
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_list_vectors() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "test",
            "dimensions": 4
        }),
    )
    .await;

    // Insert some vectors
    for i in 0..5 {
        let _ = post_json(
            &router,
            "/collections/test/vectors",
            json!({
                "id": format!("vec{}", i),
                "vector": [i as f32, 0.0, 0.0, 0.0]
            }),
        )
        .await;
    }

    let response = get(&router, "/collections/test/vectors?limit=10").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    // Response has "ids" array, not "vectors"
    assert!(body["ids"].is_array());
    assert_eq!(body["ids"].as_array().unwrap().len(), 5);
}

// ============================================================================
// Search Endpoint Tests
// ============================================================================

#[tokio::test]
async fn test_basic_search() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "search_test",
            "dimensions": 4
        }),
    )
    .await;

    // Insert vectors
    for i in 0..10 {
        let _ = post_json(
            &router,
            "/collections/search_test/vectors",
            json!({
                "id": format!("vec{}", i),
                "vector": [i as f32 / 10.0, 0.0, 0.0, 0.0]
            }),
        )
        .await;
    }

    // Search
    let response = post_json(
        &router,
        "/collections/search_test/search",
        json!({
            "vector": [0.5, 0.0, 0.0, 0.0],
            "k": 3
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert!(body["results"].is_array());
    assert_eq!(body["results"].as_array().unwrap().len(), 3);
}

#[tokio::test]
async fn test_search_with_filter() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "filtered",
            "dimensions": 4
        }),
    )
    .await;

    // Insert with metadata
    for i in 0..10 {
        let category = if i % 2 == 0 { "even" } else { "odd" };
        let _ = post_json(
            &router,
            "/collections/filtered/vectors",
            json!({
                "id": format!("vec{}", i),
                "vector": [i as f32 / 10.0, 0.0, 0.0, 0.0],
                "metadata": {"category": category}
            }),
        )
        .await;
    }

    // Search with filter
    let response = post_json(
        &router,
        "/collections/filtered/search",
        json!({
            "vector": [0.5, 0.0, 0.0, 0.0],
            "k": 10,
            "filter": {"category": "even"}
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    let results = body["results"].as_array().unwrap();

    // All results should be "even"
    for result in results {
        assert_eq!(result["metadata"]["category"], "even");
    }
}

#[tokio::test]
async fn test_search_empty_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "empty",
            "dimensions": 4
        }),
    )
    .await;

    let response = post_json(
        &router,
        "/collections/empty/search",
        json!({
            "vector": [0.1, 0.2, 0.3, 0.4],
            "k": 5
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert!(body["results"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_search_nonexistent_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let response = post_json(
        &router,
        "/collections/ghost/search",
        json!({
            "vector": [0.1, 0.2, 0.3, 0.4],
            "k": 5
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

// ============================================================================
// Batch Operation Tests
// ============================================================================

#[tokio::test]
async fn test_batch_insert() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "batch",
            "dimensions": 4
        }),
    )
    .await;

    let vectors: Vec<Value> = (0..10)
        .map(|i| {
            json!({
                "id": format!("batch{}", i),
                "vector": [i as f32, 0.0, 0.0, 0.0]
            })
        })
        .collect();

    let response = post_json(
        &router,
        "/collections/batch/vectors/batch",
        json!({ "vectors": vectors }),
    )
    .await;

    // Batch insert returns 200 OK (not 201 CREATED)
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert_eq!(body["inserted"], 10);
}

#[tokio::test]
async fn test_batch_search() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "batch_search",
            "dimensions": 4
        }),
    )
    .await;

    // Insert vectors
    for i in 0..20 {
        let _ = post_json(
            &router,
            "/collections/batch_search/vectors",
            json!({
                "id": format!("vec{}", i),
                "vector": [i as f32 / 20.0, 0.0, 0.0, 0.0]
            }),
        )
        .await;
    }

    // Batch search
    let response = post_json(
        &router,
        "/collections/batch_search/search/batch",
        json!({
            "vectors": [
                [0.1, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.9, 0.0, 0.0, 0.0]
            ],
            "k": 3
        }),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    assert!(body["results"].is_array());
    assert_eq!(body["results"].as_array().unwrap().len(), 3);
}

// ============================================================================
// Upsert Tests
// ============================================================================

#[tokio::test]
async fn test_upsert_insert_new() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "upsert",
            "dimensions": 4
        }),
    )
    .await;

    let response = post_json(
        &router,
        "/collections/upsert/vectors/upsert",
        json!({
            "id": "new_vec",
            "vector": [1.0, 2.0, 3.0, 4.0]
        }),
    )
    .await;

    // Upsert returns 200 OK with {"id": ..., "updated": bool}
    assert_eq!(response.status(), StatusCode::OK);
    let body = body_json(response).await;
    assert_eq!(body["id"], "new_vec");
    assert_eq!(body["updated"], false); // New insert, not an update
}

#[tokio::test]
async fn test_upsert_update_existing() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "upsert",
            "dimensions": 4
        }),
    )
    .await;

    // Insert first
    let _ = post_json(
        &router,
        "/collections/upsert/vectors",
        json!({
            "id": "existing",
            "vector": [1.0, 1.0, 1.0, 1.0]
        }),
    )
    .await;

    // Upsert (should update)
    let response = post_json(
        &router,
        "/collections/upsert/vectors/upsert",
        json!({
            "id": "existing",
            "vector": [2.0, 2.0, 2.0, 2.0]
        }),
    )
    .await;

    // Note: might return CREATED or OK depending on implementation
    assert!(response.status().is_success());
}

// ============================================================================
// Export/Compact Tests
// ============================================================================

#[tokio::test]
async fn test_export_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "export",
            "dimensions": 4
        }),
    )
    .await;

    for i in 0..5 {
        let _ = post_json(
            &router,
            "/collections/export/vectors",
            json!({
                "id": format!("vec{}", i),
                "vector": [i as f32, 0.0, 0.0, 0.0]
            }),
        )
        .await;
    }

    let response = get(&router, "/collections/export/export").await;
    assert_eq!(response.status(), StatusCode::OK);

    let body = body_json(response).await;
    // Response has "vectors" array, not "data"
    assert!(body["vectors"].is_array());
    assert_eq!(body["vectors"].as_array().unwrap().len(), 5);
}

#[tokio::test]
async fn test_compact_collection() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "compact",
            "dimensions": 4
        }),
    )
    .await;

    // Insert and delete to create gaps
    for i in 0..10 {
        let _ = post_json(
            &router,
            "/collections/compact/vectors",
            json!({
                "id": format!("vec{}", i),
                "vector": [i as f32, 0.0, 0.0, 0.0]
            }),
        )
        .await;
    }

    for i in 0..5 {
        let _ = delete(&router, &format!("/collections/compact/vectors/vec{}", i)).await;
    }

    // Compact
    let response = post_json(&router, "/collections/compact/compact", json!({})).await;
    assert_eq!(response.status(), StatusCode::OK);
}

// ============================================================================
// Save Database Test
// ============================================================================

#[tokio::test]
async fn test_save_database() {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("test.needle");

    let db = Database::open(&db_path).unwrap();
    let state = Arc::new(AppState::new(db));

    let config = ServerConfig::default().with_db_path(db_path.to_str().unwrap());
    let router = create_router_with_config(state, &config);

    // Create collection
    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "persist",
            "dimensions": 4
        }),
    )
    .await;

    // Save
    let response = post_json(&router, "/save", json!({})).await;
    assert_eq!(response.status(), StatusCode::OK);

    // Verify file exists
    assert!(db_path.exists());
}

// ============================================================================
// Metadata Update Tests
// ============================================================================

#[tokio::test]
async fn test_update_metadata() {
    let state = create_test_state();
    let router = create_test_router(state);

    let _ = post_json(
        &router,
        "/collections",
        json!({
            "name": "meta",
            "dimensions": 4
        }),
    )
    .await;

    let _ = post_json(
        &router,
        "/collections/meta/vectors",
        json!({
            "id": "doc1",
            "vector": [1.0, 2.0, 3.0, 4.0],
            "metadata": {"version": 1}
        }),
    )
    .await;

    // Update metadata - request body must wrap metadata in "metadata" key
    let response = post_json(
        &router,
        "/collections/meta/vectors/doc1/metadata",
        json!({"metadata": {"version": 2, "updated": true}}),
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);

    // Verify update
    let response = get(&router, "/collections/meta/vectors/doc1").await;
    let body = body_json(response).await;
    assert_eq!(body["metadata"]["version"], 2);
    assert_eq!(body["metadata"]["updated"], true);
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[tokio::test]
async fn test_cors_config_default() {
    let config = CorsConfig::default();
    assert!(config.enabled);
    assert!(config.allowed_origins.is_some());
}

#[tokio::test]
async fn test_cors_config_permissive() {
    let config = CorsConfig::permissive();
    assert!(config.enabled);
    assert!(config.allowed_origins.is_none());
    // Note: allow_credentials must be false when using wildcard origins (CORS security requirement)
    assert!(!config.allow_credentials);
}

#[tokio::test]
async fn test_cors_config_restrictive() {
    let config = CorsConfig::restrictive();
    assert!(config.enabled);
    assert_eq!(config.allowed_origins, Some(vec![]));
    assert!(!config.allow_credentials);
}

#[tokio::test]
async fn test_rate_limit_config() {
    let config = RateLimitConfig::default();
    assert!(config.enabled);

    let disabled = RateLimitConfig::disabled();
    assert!(!disabled.enabled);
}

#[tokio::test]
async fn test_server_config_builder() {
    let config = ServerConfig::new("127.0.0.1:3000")
        .unwrap()
        .with_db_path("/tmp/test.needle")
        .with_cors(CorsConfig::permissive())
        .with_rate_limit(RateLimitConfig::disabled())
        .with_max_body_size(50 * 1024 * 1024);

    assert_eq!(config.addr.port(), 3000);
    assert_eq!(config.db_path, Some("/tmp/test.needle".to_string()));
    assert_eq!(config.max_body_size, 50 * 1024 * 1024);
}
