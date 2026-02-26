//! Search handlers: ANN search, batch search, radius search, and specialized search variants.

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
use tracing::warn;

use super::{MAX_SEARCH_K, MAX_DIMENSIONS, validate_vector_dimensions};

/// Maximum allowed post-filter factor to prevent memory exhaustion.
const MAX_POST_FILTER_FACTOR: usize = 100;

/// Maximum number of vectors allowed for graph search export.
const MAX_GRAPH_VECTORS: usize = 50_000;

/// Maximum allowed hops for graph traversal.
const MAX_GRAPH_HOPS: usize = 10;

/// Maximum allowed oversample factor for matryoshka search.
const MAX_OVERSAMPLE: usize = 20;

// ============ Search Handlers ============

/// Search for similar vectors using approximate nearest neighbor search.
///
/// `POST /collections/:name/search` — accepts a query vector, k, optional
/// filter, and optional explain flag. Returns ranked results with distances
/// and similarity scores.
#[allow(clippy::too_many_lines)]
pub(in crate::server) async fn search(
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

    if req.post_filter_factor == 0 || req.post_filter_factor > MAX_POST_FILTER_FACTOR {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!(
                    "post_filter_factor must be between 1 and {MAX_POST_FILTER_FACTOR}"
                ),
                "INVALID_POST_FILTER_FACTOR",
            )),
        ));
    }

    validate_vector_dimensions(&req.vector)?;

    let db = state.db.read().await;
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
            .map_or_else(|| "cosine".to_string(), |p| p.distance_function.clone());

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

/// Batch search: run multiple queries in a single request.
///
/// `POST /collections/:name/search/batch` — accepts an array of query vectors
/// with shared k and optional filter. Returns per-query result arrays.
pub(in crate::server) async fn batch_search(
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

    for query in &req.vectors {
        validate_vector_dimensions(query)?;
    }

    let db = state.db.read().await;
    let coll = db
        .collection(&collection)
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;

    if let Some(expected) = coll.dimensions() {
        for (i, query) in req.vectors.iter().enumerate() {
            if query.len() != expected {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiError::new(
                        format!(
                            "Query vector [{}] dimension mismatch: expected {}, got {}",
                            i, expected, query.len()
                        ),
                        "DIMENSION_MISMATCH",
                    )),
                ));
            }
        }
    }

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

/// Radius-based (range) search.
///
/// `POST /collections/:name/search/radius` — returns all vectors within
/// `max_distance` of the query vector, up to `limit` results.
/// Supports optional metadata filter and `include_vectors` flag.
pub(in crate::server) async fn radius_search(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(req): Json<RadiusSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    if req.limit == 0 || req.limit > MAX_SEARCH_K {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ApiError::new(
                format!("limit must be between 1 and {MAX_SEARCH_K}"),
                "INVALID_LIMIT",
            )),
        ));
    }

    validate_vector_dimensions(&req.vector)?;

    let db = state.db.read().await;
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

/// Matryoshka (adaptive-dimension) search.
///
/// `POST /collections/:name/matryoshka-search` — performs a coarse search at
/// reduced dimensions, then re-ranks candidates at full resolution.
pub(in crate::server) async fn matryoshka_search_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<MatryoshkaSearchRequest>,
) -> impl IntoResponse {
    if body.k == 0 || body.k > MAX_SEARCH_K {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": format!("k must be between 1 and {MAX_SEARCH_K}")
        })));
    }
    if body.coarse_dims == 0 || body.coarse_dims > MAX_DIMENSIONS {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": format!("coarse_dims must be between 1 and {MAX_DIMENSIONS}")
        })));
    }
    if body.oversample == 0 || body.oversample > MAX_OVERSAMPLE {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": format!("oversample must be between 1 and {MAX_OVERSAMPLE}")
        })));
    }

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

/// Time-travel search against a named snapshot.
///
/// `POST /collections/:name/time-travel` — searches the collection as it
/// existed at the given snapshot point. Currently searches current state
/// and annotates with snapshot metadata.
pub(in crate::server) async fn time_travel_search_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
    Json(body): Json<TimeTravelSearchRequest>,
) -> impl IntoResponse {
    use crate::persistence::time_travel::{MvccConfig, TimeExpression, TimeTravelIndex};
    use std::sync::Arc as StdArc;

    let db = state.db.read().await;

    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    // Determine query mode: timestamp, expression, or snapshot name
    if let Some(timestamp) = body.as_of_timestamp {
        let time_expr = TimeExpression::Timestamp(timestamp);
        let db_arc = StdArc::new(crate::database::Database::in_memory());
        let index = TimeTravelIndex::new(db_arc, &collection, MvccConfig::default());

        let results = match index.search_at(&body.vector, body.k, time_expr) {
            Ok(r) => r,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
        };

        let response: Vec<Value> = results.iter().map(|r| {
            json!({
                "id": r.result.id,
                "distance": r.result.distance,
                "version": r.version,
                "query_timestamp": r.query_timestamp,
                "metadata": r.result.metadata,
            })
        }).collect();

        return (StatusCode::OK, Json(json!({
            "results": response,
            "count": response.len(),
            "as_of_timestamp": timestamp,
        })));
    }

    if let Some(ref expr) = body.as_of_expression {
        let time_expr = match TimeExpression::parse(expr) {
            Ok(t) => t,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
        };
        let resolved_ts = match time_expr.resolve() {
            Ok(ts) => ts,
            Err(e) => return (StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() }))),
        };

        // Search current state with time annotation
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

        return (StatusCode::OK, Json(json!({
            "results": response,
            "count": response.len(),
            "as_of_expression": expr,
            "resolved_timestamp": resolved_ts,
        })));
    }

    // Fallback: snapshot-based search
    if body.snapshot.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": "Must provide one of: snapshot, as_of_timestamp, as_of_version, or as_of_expression"
        })));
    }

    let current_snapshots = coll.list_snapshots();
    if !current_snapshots.contains(&body.snapshot) {
        return (StatusCode::NOT_FOUND, Json(json!({
            "error": format!("Snapshot '{}' not found", body.snapshot),
            "available_snapshots": current_snapshots,
        })));
    }

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
    })))
}

/// Graph-augmented vector search (GraphRAG).
///
/// `POST /collections/:name/graph-search` — builds a knowledge graph from
/// collection vectors, then performs a combined vector + graph traversal
/// search up to `max_hops` deep.
pub(in crate::server) async fn graph_search_handler(
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

    let count = coll.len();
    if count > MAX_GRAPH_VECTORS {
        return (StatusCode::PAYLOAD_TOO_LARGE, Json(json!({
            "error": format!(
                "Collection has {} vectors, exceeding graph search limit of {}. Use standard search for large collections.",
                count, MAX_GRAPH_VECTORS
            )
        })));
    }

    // Build a GraphRAG index from the collection's vectors and metadata
    let dims = coll.dimensions().unwrap_or(0);
    let max_hops = body.max_hops.min(MAX_GRAPH_HOPS);
    let config = GraphRAGConfig {
        dimensions: dims,
        max_hops,
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
        if let Err(e) = graph.add_entity(entity) {
            warn!(id = %id, error = %e, "Failed to index entity in graph");
        }
    }

    let results = match graph.search(&body.vector, body.k, Some(max_hops)) {
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

// ============ Cache Handlers ============

/// Semantic cache lookup.
///
/// `POST /collections/:name/cache/lookup` — checks whether a semantically
/// similar query has been cached. Requires the `experimental` feature.
#[cfg(feature = "experimental")]
pub(in crate::server) async fn cache_lookup_handler(
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

/// Semantic cache lookup stub (experimental feature not enabled).
#[cfg(not(feature = "experimental"))]
pub(in crate::server) async fn cache_lookup_handler(
    Path(_collection): Path<String>,
    Json(_body): Json<CacheLookupRequest>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({ "error": "Requires 'experimental' feature" })))
}

/// Store a response in the semantic cache.
///
/// `POST /collections/:name/cache/store` — caches a query-response pair
/// so future similar queries can be served from cache.
pub(in crate::server) async fn cache_store_handler(
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

#[cfg(test)]
mod tests {
    use super::*;
}
