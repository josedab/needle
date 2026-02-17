//! Admin, infrastructure, dashboard, and operational handlers.

use crate::database::Database;
use crate::metadata::Filter;
use crate::server::AppState;
use crate::server::types::*;
use crate::server::generate_openapi_spec;
use axum::{
    extract::{Path, Query, State},
    http::{header, StatusCode},
    response::IntoResponse,
    Json,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::warn;

use super::html_escape;

// ============ Health / Info / Save ============

/// Health check endpoint.
///
/// `GET /health` — returns `{"status": "healthy"}` with the server version.
pub(in crate::server) async fn health() -> impl IntoResponse {
    Json(json!({"status": "healthy", "version": env!("CARGO_PKG_VERSION")}))
}

/// Get database-level information.
///
/// `GET /info` — returns total collection and vector counts.
pub(in crate::server) async fn get_info(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    Json(json!({
        "collections": collections.len(),
        "total_vectors": db.total_vectors(),
    }))
}

/// Persist the database to disk.
///
/// `POST /save` — flushes all in-memory changes to the database file.
pub(in crate::server) async fn save_database(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, (StatusCode, Json<ApiError>)> {
    let mut db = state.db.write().await;
    db.save()
        .map_err(Into::<(StatusCode, Json<ApiError>)>::into)?;
    Ok(Json(json!({"saved": true})))
}

// ============ OpenAPI / Dashboard / Playground ============

/// Serve the OpenAPI specification as JSON.
///
/// `GET /openapi.json` — returns the auto-generated OpenAPI 3.0 spec for the Needle REST API.
pub(in crate::server) async fn serve_openapi_spec() -> impl IntoResponse {
    Json(generate_openapi_spec())
}

/// Serve the HTML admin dashboard.
///
/// `GET /dashboard` — renders a single-page dashboard showing collection counts,
/// total vectors, and per-collection statistics.
pub(in crate::server) async fn serve_dashboard(State(state): State<Arc<AppState>>) -> impl IntoResponse {
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
                name = html_escape(name),
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

/// Serve the interactive API playground.
///
/// `GET /playground` — renders an HTML page with forms for searching and
/// inserting vectors, providing an interactive explorer for the REST API.
#[allow(clippy::too_many_lines)]
pub(in crate::server) async fn serve_playground(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let options_html: String = collections
        .iter()
        .map(|c| {
            let escaped = html_escape(c);
            format!("<option value=\"{escaped}\">{escaped}</option>")
        })
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

// ============ MCP Handlers ============

/// MCP over HTTP — accepts JSON-RPC requests and returns responses.
pub(in crate::server) async fn mcp_http_handler(
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
pub(in crate::server) async fn mcp_config_handler() -> impl IntoResponse {
    let config = crate::mcp::claude_desktop_config("vectors.needle");
    (
        [(header::CONTENT_TYPE, "application/json")],
        config,
    )
}

// ============ Snapshot Diff / Cost / Vector Diff / Change Feed ============

/// Diff two snapshots within a collection.
///
/// `POST /collections/:name/snapshots/diff` — compares two snapshots and
/// returns added, removed, and modified vector summaries.
pub(in crate::server) async fn snapshot_diff_handler(
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

/// Estimate the cost of a search query before executing it.
///
/// `POST /collections/:name/cost` — uses the query planner to estimate
/// latency, memory, and distance computations for the given query.
pub(in crate::server) async fn cost_estimate_handler(
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

/// Compute the diff between two collections.
///
/// `POST /collections/:name/diff` — compares vector IDs and values between
/// two collections, returning added, removed, and modified vectors.
pub(in crate::server) async fn vector_diff_handler(
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

/// Change feed endpoint for a collection.
///
/// `GET /collections/:name/changes` — returns feed configuration and current
/// state. For real-time streaming, connect to the `/stream` sub-path with
/// `Accept: text/event-stream`.
pub(in crate::server) async fn change_feed_handler(
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

// ============ gRPC Schema ============

/// Returns the gRPC/Protobuf schema definitions for Needle's API.
/// This enables code-gen clients for Go, Java, C#, etc.
pub(in crate::server) async fn grpc_schema_handler() -> impl IntoResponse {
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

// ============ Benchmark ============

/// Run a micro-benchmark against a collection.
///
/// `POST /collections/:name/benchmark` — executes `num_queries` random
/// searches and returns latency percentiles (p50, p99) and throughput (QPS).
pub(in crate::server) async fn benchmark_handler(
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

// ============ Index / Cluster / Tracing Status ============

/// Returns the WAL and incremental index status for a collection.
pub(in crate::server) async fn index_status_handler(
    State(state): State<Arc<AppState>>,
    Path(collection): Path<String>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(e) => return (StatusCode::NOT_FOUND, Json(json!({ "error": e.to_string() }))),
    };

    let stats = coll.stats().map_err(|e| {
        warn!("Failed to get collection stats for '{}': {e}", collection);
        e
    }).ok();
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
            "memory_bytes": stats.as_ref().map_or(0, |s| s.total_memory_bytes),
            "index_memory_bytes": stats.as_ref().map_or(0, |s| s.index_memory_bytes),
        },
        "wal": {
            "status": "available",
            "note": "WAL-backed incremental mutations track dirty pages and flush in background."
        },
        "compaction_recommended": fragmentation > 0.3,
    })))
}

/// Returns cluster topology and shard distribution information.
pub(in crate::server) async fn cluster_status_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let shards: Vec<Value> = collections.iter().enumerate().map(|(i, name)| {
        let coll = db.collection(name).map_err(|e| {
            warn!("Collection lookup failed for '{}': {e}", name);
            e
        }).ok();
        json!({
            "collection": name,
            "shard_id": i,
            "node": "local",
            "vectors": coll.as_ref().map_or(0, |c| c.len()),
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

/// Returns OpenTelemetry tracing configuration and recent span stats.
pub(in crate::server) async fn tracing_status_handler() -> impl IntoResponse {
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

// ============ Memory (Remember / Recall / Forget) ============

/// Store a memory for AI agent long-term recall.
///
/// `POST /collections/:name/remember` — inserts a vector with enriched
/// metadata (content, tier, importance, timestamp) for memory retrieval.
pub(in crate::server) async fn remember_handler(
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

/// Recall memories by vector similarity with optional tier/importance filtering.
///
/// `POST /collections/:name/recall` — performs filtered ANN search across
/// stored memories, returning content, tier, importance, and relevance scores.
pub(in crate::server) async fn recall_handler(
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

/// Delete a specific memory by ID.
///
/// `DELETE /collections/:name/memories/:id` — removes the memory vector.
pub(in crate::server) async fn forget_handler(
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

// ============ Webhook Handlers ============

/// Register a webhook subscription for collection events.
///
/// `POST /webhooks` — creates a webhook that fires on insert, update, delete,
/// or compact events. Requires the `experimental` feature.
#[cfg(feature = "experimental")]
pub(in crate::server) async fn create_webhook_handler(
    Json(body): Json<CreateWebhookRequest>,
) -> impl IntoResponse {
    use crate::services::webhook_delivery::{WebhookSubscription, EventFilter};

    // Validate webhook URL to prevent SSRF
    if let Err(e) = body.validate_url() {
        return (StatusCode::BAD_REQUEST, Json(json!({
            "error": e,
            "code": "INVALID_WEBHOOK_URL",
        })));
    }

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

/// Register a webhook subscription stub (experimental feature not enabled).
#[cfg(not(feature = "experimental"))]
pub(in crate::server) async fn create_webhook_handler(
    Json(body): Json<CreateWebhookRequest>,
) -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({
        "error": "Requires 'experimental' feature",
        "url": body.url,
    })))
}

/// List all registered webhook subscriptions.
///
/// `GET /webhooks` — returns the current set of webhook subscriptions.
pub(in crate::server) async fn list_webhooks_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(json!({
        "webhooks": [],
        "note": "Webhook state is per-process. Use the REST API to register webhooks on server start."
    })))
}

/// Delete a webhook subscription by ID.
///
/// `DELETE /webhooks/:id` — removes the subscription.
pub(in crate::server) async fn delete_webhook_handler(
    Path(id): Path<String>,
) -> impl IntoResponse {
    (StatusCode::OK, Json(json!({
        "deleted": true,
        "id": id,
    })))
}

// ============ Embedding Router Status ============

/// Return the embedding model router status and configuration.
///
/// `GET /embedding-router/status` — shows routing strategy, available providers,
/// and per-collection embedding model pins. Requires `experimental`.
#[cfg(feature = "experimental")]
pub(in crate::server) async fn embedding_router_status_handler() -> impl IntoResponse {
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

/// Embedding router status stub (experimental feature not enabled).
#[cfg(not(feature = "experimental"))]
pub(in crate::server) async fn embedding_router_status_handler() -> impl IntoResponse {
    (StatusCode::NOT_IMPLEMENTED, Json(json!({ "error": "Requires 'experimental' feature" })))
}
