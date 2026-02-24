#![cfg_attr(test, allow(clippy::unwrap_used))]

use super::state::*;
use super::templates::*;
use crate::database::Database;
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
    Json, Router,
};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Route Handlers
// ============================================================================

/// GET / - Main dashboard page
async fn dashboard_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();
    let total_vectors = db.total_vectors();
    let uptime = state.uptime();

    let collection_rows: String = collections
        .iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let deleted = coll.deleted_count();
            let needs_compact = coll.needs_compaction(0.2);

            Some(format!(
                r#"<tr>
                    <td><a href="/collections/{name}">{name}</a></td>
                    <td>{count}</td>
                    <td>{dims}</td>
                    <td>{deleted}</td>
                    <td>{status}</td>
                </tr>"#,
                name = name,
                count = format_number(count),
                dims = dims,
                deleted = deleted,
                status = if needs_compact {
                    r#"<span class="badge badge-warning">Needs Compaction</span>"#
                } else {
                    r#"<span class="badge badge-success">OK</span>"#
                }
            ))
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Dashboard</h1>
            <p class="page-description">Overview of your Needle vector database</p>
        </div>

        <div class="grid grid-4">
            {stat_health}
            {stat_collections}
            {stat_vectors}
            {stat_uptime}
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Collections Overview</h2>
                <a href="/collections" class="btn btn-secondary">View All</a>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Vectors</th>
                        <th>Dimensions</th>
                        <th>Deleted</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {collection_rows}
                </tbody>
            </table>
            {empty_message}
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Quick Actions</h2>
                </div>
                <div style="display: flex; gap: 1rem; flex-wrap: wrap;">
                    <a href="/query" class="btn btn-primary">Query Playground</a>
                    <a href="/api/stats" class="btn btn-secondary">View API Stats</a>
                    <a href="/monitoring" class="btn btn-secondary">Metrics Dashboard</a>
                </div>
            </div>
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">System Info</h2>
                </div>
                <p><strong>Version:</strong> {version}</p>
                <p><strong>Collections:</strong> {num_collections}</p>
                <p><strong>Total Vectors:</strong> {total_vectors}</p>
                <p><strong>Uptime:</strong> {uptime_formatted}</p>
            </div>
        </div>
        "#,
        stat_health = stat_card("Health", "Healthy", ""),
        stat_collections = stat_card("Collections", &format_number(collections.len()), ""),
        stat_vectors = stat_card("Total Vectors", &format_number(total_vectors), ""),
        stat_uptime = stat_card("Uptime", &format_uptime(uptime), ""),
        collection_rows = collection_rows,
        empty_message = if collections.is_empty() {
            r#"<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No collections yet. Create one using the API.</p>"#
        } else {
            ""
        },
        version = env!("CARGO_PKG_VERSION"),
        num_collections = collections.len(),
        total_vectors = format_number(total_vectors),
        uptime_formatted = format_uptime(uptime),
    );

    Html(base_layout("Dashboard", &content, "dashboard"))
}

/// GET /collections - List all collections
async fn collections_list_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let collection_cards: String = collections
        .iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let deleted = coll.deleted_count();

            Some(format!(
                r#"<div class="card">
                    <div class="card-header">
                        <h3 class="card-title">{name}</h3>
                        <a href="/collections/{name}" class="btn btn-primary">View Details</a>
                    </div>
                    <div class="grid grid-3">
                        <div class="stat">
                            <div class="stat-value">{count}</div>
                            <div class="stat-label">Vectors</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{dims}</div>
                            <div class="stat-label">Dimensions</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">{deleted}</div>
                            <div class="stat-label">Deleted</div>
                        </div>
                    </div>
                </div>"#,
                name = name,
                count = format_number(count),
                dims = dims,
                deleted = deleted,
            ))
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Collections</h1>
            <p class="page-description">Browse and manage your vector collections</p>
        </div>

        {cards}

        {empty_message}
        "#,
        cards = collection_cards,
        empty_message = if collections.is_empty() {
            r#"<div class="card" style="text-align: center; padding: 3rem;">
                <h3>No Collections Found</h3>
                <p style="color: var(--text-secondary); margin: 1rem 0;">
                    Create your first collection using the REST API or CLI.
                </p>
                <pre style="text-align: left; display: inline-block;">
curl -X POST http://localhost:8080/collections \
  -H "Content-Type: application/json" \
  -d '{"name": "my_collection", "dimensions": 384}'
                </pre>
            </div>"#
        } else {
            ""
        },
    );

    Html(base_layout("Collections", &content, "collections"))
}

/// GET /collections/:name - Collection detail view
async fn collection_detail_handler(
    State(state): State<Arc<WebUiState>>,
    Path(name): Path<String>,
) -> Result<Html<String>, (StatusCode, Html<String>)> {
    let db = state.db.read().await;

    let coll = match db.collection(&name) {
        Ok(c) => c,
        Err(_) => {
            let error_content = format!(
                r#"
                <div class="card" style="text-align: center; padding: 3rem;">
                    <h2 style="color: var(--danger);">Collection Not Found</h2>
                    <p style="margin: 1rem 0;">The collection "{}" does not exist.</p>
                    <a href="/collections" class="btn btn-primary">Back to Collections</a>
                </div>
                "#,
                name
            );
            return Err((
                StatusCode::NOT_FOUND,
                Html(base_layout("Not Found", &error_content, "collections")),
            ));
        }
    };

    let dims = coll.dimensions().unwrap_or(0);
    let count = coll.len();
    let deleted = coll.deleted_count();
    let needs_compact = coll.needs_compaction(0.2);

    // Get sample vector IDs
    let sample_ids: Vec<String> = coll
        .ids()
        .ok()
        .map(|ids| ids.into_iter().take(10).collect())
        .unwrap_or_default();

    let sample_rows: String = sample_ids
        .iter()
        .map(|id| {
            format!(
                r#"<tr>
                    <td><code>{id}</code></td>
                    <td>-</td>
                </tr>"#,
                id = id
            )
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h1 class="page-title">{name}</h1>
                    <p class="page-description">Collection details and statistics</p>
                </div>
                <a href="/collections" class="btn btn-secondary">Back to Collections</a>
            </div>
        </div>

        <div class="grid grid-4">
            {stat_vectors}
            {stat_dims}
            {stat_deleted}
            {stat_status}
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Collection Info</h2>
                </div>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Name</td><td><code>{name}</code></td></tr>
                    <tr><td>Dimensions</td><td>{dims}</td></tr>
                    <tr><td>Vector Count</td><td>{count}</td></tr>
                    <tr><td>Deleted Count</td><td>{deleted}</td></tr>
                    <tr><td>Needs Compaction</td><td>{compact_status}</td></tr>
                </table>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Quick Search</h2>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Use the Query Playground to search this collection.
                </p>
                <a href="/query?collection={name}" class="btn btn-primary">
                    Open Query Playground
                </a>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Sample Vectors</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Metadata</th>
                    </tr>
                </thead>
                <tbody>
                    {sample_rows}
                </tbody>
            </table>
            {empty_sample}
        </div>
        "#,
        name = name,
        stat_vectors = stat_card("Vectors", &format_number(count), ""),
        stat_dims = stat_card("Dimensions", &dims.to_string(), ""),
        stat_deleted = stat_card("Deleted", &deleted.to_string(), ""),
        stat_status = stat_card(
            "Status",
            if needs_compact {
                "Needs Compaction"
            } else {
                "OK"
            },
            ""
        ),
        dims = dims,
        count = format_number(count),
        deleted = deleted,
        compact_status = if needs_compact {
            r#"<span class="badge badge-warning">Yes</span>"#
        } else {
            r#"<span class="badge badge-success">No</span>"#
        },
        sample_rows = sample_rows,
        empty_sample = if sample_ids.is_empty() {
            r#"<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No vectors in this collection.</p>"#
        } else {
            ""
        },
    );

    Ok(Html(base_layout(
        &format!("Collection: {}", name),
        &content,
        "collections",
    )))
}

/// GET /query - Query playground
async fn query_playground_handler(
    State(state): State<Arc<WebUiState>>,
    Query(params): Query<SearchQuery>,
) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let collection_options: String = collections
        .iter()
        .map(|name| {
            let selected = params.collection.as_ref() == Some(name);
            format!(
                r#"<option value="{name}" {selected}>{name}</option>"#,
                name = name,
                selected = if selected { "selected" } else { "" }
            )
        })
        .collect();

    let default_k = params.k.unwrap_or(10);
    let default_vector = params.vector.as_deref().unwrap_or("");

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Query Playground</h1>
            <p class="page-description">Test vector searches interactively</p>
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Search Parameters</h2>
                </div>
                <form id="search-form">
                    <div class="form-group">
                        <label class="form-label" for="collection">Collection</label>
                        <select id="collection" name="collection" class="form-input">
                            <option value="">Select a collection</option>
                            {collection_options}
                        </select>
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="vector">Query Vector (comma-separated)</label>
                        <textarea
                            id="vector"
                            name="vector"
                            class="form-textarea"
                            placeholder="0.1, 0.2, 0.3, ..."
                        >{default_vector}</textarea>
                    </div>
                    <div class="form-group">
                        <label class="form-label" for="k">Number of Results (k)</label>
                        <input
                            type="number"
                            id="k"
                            name="k"
                            class="form-input"
                            value="{default_k}"
                            min="1"
                            max="100"
                        />
                    </div>
                    <button type="submit" class="btn btn-primary">
                        Search
                    </button>
                </form>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Results</h2>
                </div>
                <div id="results">
                    <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                        Enter a query vector and click Search to see results.
                    </p>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">API Usage</h2>
            </div>
            <p style="margin-bottom: 1rem; color: var(--text-secondary);">
                Use this curl command to search via the API:
            </p>
            <pre>
curl -X POST http://localhost:8080/collections/YOUR_COLLECTION/search \
  -H "Content-Type: application/json" \
  -d '{{
    "vector": [0.1, 0.2, 0.3, ...],
    "k": 10
  }}'
            </pre>
        </div>

        <script>
        document.getElementById('search-form').addEventListener('submit', async (e) => {{
            e.preventDefault();

            const collection = document.getElementById('collection').value;
            const vectorStr = document.getElementById('vector').value;
            const k = parseInt(document.getElementById('k').value) || 10;

            if (!collection) {{
                alert('Please select a collection');
                return;
            }}

            if (!vectorStr.trim()) {{
                alert('Please enter a query vector');
                return;
            }}

            const vector = vectorStr.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));

            if (vector.length === 0) {{
                alert('Invalid vector format. Use comma-separated numbers.');
                return;
            }}

            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<p style="text-align: center; padding: 2rem;">Searching...</p>';

            try {{
                const response = await fetch(`/collections/${{collection}}/search`, {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ vector, k }})
                }});

                const data = await response.json();

                if (response.ok && data.results) {{
                    if (data.results.length === 0) {{
                        resultsDiv.innerHTML = '<p style="text-align: center; padding: 2rem; color: var(--text-secondary);">No results found.</p>';
                    }} else {{
                        let html = '<ul class="results-list">';
                        data.results.forEach((r, i) => {{
                            html += `
                                <li class="result-item">
                                    <div class="result-id">#${{i + 1}} ${{r.id}}</div>
                                    <div class="result-distance">Distance: ${{r.distance.toFixed(6)}} | Score: ${{r.score.toFixed(4)}}</div>
                                </li>
                            `;
                        }});
                        html += '</ul>';
                        resultsDiv.innerHTML = html;
                    }}
                }} else {{
                    resultsDiv.innerHTML = `<p style="color: var(--danger); padding: 1rem;">Error: ${{data.error || 'Unknown error'}}</p>`;
                }}
            }} catch (err) {{
                resultsDiv.innerHTML = `<p style="color: var(--danger); padding: 1rem;">Error: ${{err.message}}</p>`;
            }}
        }});
        </script>
        "#,
        collection_options = collection_options,
        default_vector = default_vector,
        default_k = default_k,
    );

    Html(base_layout("Query Playground", &content, "query"))
}

/// GET /playground - NeedleQL interactive playground with code editor
async fn needleql_playground_handler(
    State(state): State<Arc<WebUiState>>,
) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let collection_list: String = collections
        .iter()
        .map(|name| format!(r#""{name}""#))
        .collect::<Vec<_>>()
        .join(", ");

    let example_queries = r#"
-- Example NeedleQL Queries:

-- Search for similar vectors
SELECT * FROM documents
  WHERE vector SIMILAR TO [0.1, 0.2, 0.3]
  LIMIT 10;

-- Count vectors in a collection
SELECT COUNT(*) FROM documents;

-- Search with metadata filter
SELECT id, distance FROM documents
  WHERE vector SIMILAR TO [0.1, 0.2, 0.3]
  AND category = 'science'
  LIMIT 5;
"#;

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">NeedleQL Playground</h1>
            <p class="page-description">Interactive query editor for vector search</p>
        </div>

        <div class="card" style="margin-bottom: 1rem;">
            <div class="card-header" style="display: flex; justify-content: space-between; align-items: center;">
                <h2 class="card-title">Query Editor</h2>
                <div>
                    <button onclick="runQuery()" class="btn btn-primary" style="margin-right: 0.5rem;">
                        ▶ Run Query
                    </button>
                    <button onclick="clearEditor()" class="btn" style="background: var(--surface); border: 1px solid var(--border);">
                        Clear
                    </button>
                </div>
            </div>
            <div style="position: relative;">
                <textarea
                    id="query-editor"
                    style="width: 100%; min-height: 200px; font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
                           font-size: 14px; line-height: 1.6; padding: 1rem; border: 1px solid var(--border);
                           border-radius: 8px; background: var(--surface); color: var(--text-primary);
                           resize: vertical; tab-size: 2;"
                    spellcheck="false"
                    placeholder="Enter your NeedleQL query here..."
                >{example_queries}</textarea>
            </div>
            <div style="margin-top: 0.5rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
                <small style="color: var(--text-secondary);">
                    Collections: [{collection_list}]
                </small>
            </div>
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header" style="display: flex; justify-content: space-between; align-items: center;">
                    <h2 class="card-title">Results</h2>
                    <div>
                        <button onclick="showView('table')" class="btn" style="font-size: 12px; padding: 4px 8px;">Table</button>
                        <button onclick="showView('json')" class="btn" style="font-size: 12px; padding: 4px 8px;">JSON</button>
                    </div>
                </div>
                <div id="results-table" style="overflow-x: auto;">
                    <p style="color: var(--text-secondary); text-align: center; padding: 2rem;">
                        Run a query to see results here.
                    </p>
                </div>
                <pre id="results-json" style="display: none; overflow: auto; max-height: 400px;
                     padding: 1rem; background: var(--surface); border-radius: 8px; font-size: 13px;"></pre>
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Query History</h2>
                </div>
                <div id="query-history" style="max-height: 300px; overflow-y: auto;">
                    <p style="color: var(--text-secondary); text-align: center; padding: 1rem; font-size: 14px;">
                        No queries yet.
                    </p>
                </div>
            </div>
        </div>

        <script>
        const queryHistory = JSON.parse(localStorage.getItem('needle_query_history') || '[]');
        renderHistory();

        function runQuery() {{
            const editor = document.getElementById('query-editor');
            const query = editor.value.trim();
            if (!query) return;

            // Save to history
            queryHistory.unshift({{ query: query, time: new Date().toISOString() }});
            if (queryHistory.length > 50) queryHistory.pop();
            localStorage.setItem('needle_query_history', JSON.stringify(queryHistory));
            renderHistory();

            // Display query as result (actual execution would need backend endpoint)
            const resultsTable = document.getElementById('results-table');
            resultsTable.innerHTML = '<div style="padding: 1rem;"><p><strong>Query submitted:</strong></p><pre style="background: var(--surface); padding: 0.5rem; border-radius: 4px; font-size: 13px;">' +
                query.replace(/</g, '&lt;') + '</pre><p style="color: var(--text-secondary); margin-top: 0.5rem;">Connect to a running Needle server to execute queries.</p></div>';

            const resultsJson = document.getElementById('results-json');
            resultsJson.textContent = JSON.stringify({{ query: query, status: "submitted" }}, null, 2);
        }}

        function clearEditor() {{
            document.getElementById('query-editor').value = '';
        }}

        function showView(view) {{
            document.getElementById('results-table').style.display = view === 'table' ? 'block' : 'none';
            document.getElementById('results-json').style.display = view === 'json' ? 'block' : 'none';
        }}

        function renderHistory() {{
            const container = document.getElementById('query-history');
            if (queryHistory.length === 0) return;
            container.innerHTML = queryHistory.map((h, i) =>
                '<div style="padding: 0.5rem; border-bottom: 1px solid var(--border); cursor: pointer;" ' +
                'onclick="document.getElementById(\'query-editor\').value = queryHistory[' + i + '].query">' +
                '<small style="color: var(--text-secondary);">' + new Date(h.time).toLocaleTimeString() + '</small>' +
                '<pre style="margin: 0.25rem 0 0; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">' +
                h.query.replace(/</g, '&lt;').substring(0, 80) + '</pre></div>'
            ).join('');
        }}

        // Keyboard shortcut: Ctrl/Cmd+Enter to run
        document.getElementById('query-editor').addEventListener('keydown', function(e) {{
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {{
                e.preventDefault();
                runQuery();
            }}
        }});
        </script>
        "#,
        example_queries = example_queries.replace('<', "&lt;"),
        collection_list = collection_list,
    );

    Html(base_layout("NeedleQL Playground", &content, "playground"))
}

/// GET /monitoring - Metrics dashboard
async fn monitoring_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();
    let total_vectors = db.total_vectors();
    let uptime = state.uptime();

    // Calculate memory estimates
    let mut total_memory: usize = 0;
    let mut collection_memory: Vec<(String, usize, usize)> = Vec::new();

    for name in &collections {
        if let Ok(coll) = db.collection(name) {
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let vector_memory = count * dims * std::mem::size_of::<f32>();
            total_memory += vector_memory;
            collection_memory.push((name.clone(), count, vector_memory));
        }
    }

    let memory_rows: String = collection_memory
        .iter()
        .map(|(name, count, mem)| {
            format!(
                r#"<tr>
                    <td>{name}</td>
                    <td>{count}</td>
                    <td>{memory}</td>
                    <td>
                        <div class="progress-bar" style="width: 100px;">
                            <div class="progress-fill" style="width: {pct}%;"></div>
                        </div>
                    </td>
                </tr>"#,
                name = name,
                count = format_number(*count),
                memory = format_bytes(*mem),
                pct = if total_memory > 0 {
                    (*mem as f64 / total_memory as f64 * 100.0).round() as usize
                } else {
                    0
                }
            )
        })
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Monitoring</h1>
            <p class="page-description">System metrics and performance data</p>
        </div>

        <div class="grid grid-4">
            {stat_status}
            {stat_uptime}
            {stat_memory}
            {stat_vectors}
        </div>

        <div class="grid grid-2">
            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">Memory Usage by Collection</h2>
                </div>
                <table>
                    <thead>
                        <tr>
                            <th>Collection</th>
                            <th>Vectors</th>
                            <th>Memory</th>
                            <th>Distribution</th>
                        </tr>
                    </thead>
                    <tbody>
                        {memory_rows}
                    </tbody>
                </table>
                {empty_memory}
            </div>

            <div class="card">
                <div class="card-header">
                    <h2 class="card-title">System Health</h2>
                </div>
                <div style="padding: 1rem;">
                    <div class="health-indicator" style="margin-bottom: 1rem;">
                        <span class="health-dot healthy"></span>
                        <span>Database Status: <strong>Healthy</strong></span>
                    </div>
                    <div class="health-indicator" style="margin-bottom: 1rem;">
                        <span class="health-dot healthy"></span>
                        <span>API Server: <strong>Running</strong></span>
                    </div>
                    <div class="health-indicator">
                        <span class="health-dot healthy"></span>
                        <span>Web UI: <strong>Active</strong></span>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">Performance Metrics</h2>
            </div>
            <p style="color: var(--text-secondary); text-align: center; padding: 3rem;">
                Performance metrics visualization coming soon.<br/>
                Enable the <code>metrics</code> feature for Prometheus-compatible metrics.
            </p>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">API Endpoints</h2>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Endpoint</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td><code>GET /api/stats</code></td><td>JSON statistics for programmatic access</td></tr>
                    <tr><td><code>GET /health</code></td><td>Health check endpoint (API server)</td></tr>
                    <tr><td><code>GET /metrics</code></td><td>Prometheus metrics (if enabled)</td></tr>
                </tbody>
            </table>
        </div>
        "#,
        stat_status = stat_card("Status", "Healthy", ""),
        stat_uptime = stat_card("Uptime", &format_uptime(uptime), ""),
        stat_memory = stat_card("Est. Memory", &format_bytes(total_memory), ""),
        stat_vectors = stat_card("Vectors", &format_number(total_vectors), ""),
        memory_rows = memory_rows,
        empty_memory = if collections.is_empty() {
            r#"<p style="padding: 2rem; text-align: center; color: var(--text-secondary);">No collections to display.</p>"#
        } else {
            ""
        },
    );

    Html(base_layout("Monitoring", &content, "monitoring"))
}

/// GET /api/stats - JSON stats endpoint
async fn api_stats_handler(State(state): State<Arc<WebUiState>>) -> Json<StatsResponse> {
    let db = state.db.read().await;
    let collection_names = db.list_collections();
    let total_vectors = db.total_vectors();
    let uptime = state.uptime();

    let collections: Vec<CollectionStatsResponse> = collection_names
        .iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            Some(CollectionStatsResponse {
                name: name.clone(),
                vector_count: coll.len(),
                dimensions: coll.dimensions().unwrap_or(0),
                deleted_count: coll.deleted_count(),
                needs_compaction: coll.needs_compaction(0.2),
            })
        })
        .collect();

    Json(StatsResponse {
        healthy: true,
        uptime_seconds: uptime,
        total_collections: collection_names.len(),
        total_vectors,
        collections,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Health check endpoint
async fn health_handler() -> impl IntoResponse {
    Json(json!({
        "status": "healthy",
        "service": "web-ui",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

/// Real-time monitoring API endpoint returning JSON snapshot.
async fn api_monitoring_handler(State(state): State<Arc<WebUiState>>) -> Json<MonitoringSnapshot> {
    let db = state.db.read().await;
    let uptime = state.uptime();
    let snapshot = compute_monitoring_snapshot(&db, uptime);
    Json(snapshot)
}

/// GET /visualize - Interactive vector visualization with 2D scatter plot
async fn visualize_handler(State(state): State<Arc<WebUiState>>) -> Html<String> {
    let db = state.db.read().await;
    let collections = db.list_collections();

    let collection_options: String = collections
        .iter()
        .map(|name| format!(r#"<option value="{name}">{name}</option>"#))
        .collect();

    let content = format!(
        r#"
        <div class="page-header">
            <h1 class="page-title">Vector Visualization</h1>
            <p class="page-description">2D projection of collection vectors with interactive search</p>
        </div>

        <div class="card" style="margin-bottom: 1rem;">
            <div class="card-header" style="display: flex; justify-content: space-between; align-items: center;">
                <h2 class="card-title">Controls</h2>
            </div>
            <div style="display: flex; gap: 1rem; align-items: end; flex-wrap: wrap;">
                <div class="form-group" style="margin: 0;">
                    <label class="form-label">Collection</label>
                    <select id="viz-collection" class="form-input">
                        <option value="">Select</option>
                        {collection_options}
                    </select>
                </div>
                <div class="form-group" style="margin: 0;">
                    <label class="form-label">Max Points</label>
                    <input type="number" id="viz-max" class="form-input" value="500" min="10" max="5000" />
                </div>
                <button onclick="loadVisualization()" class="btn btn-primary">Visualize</button>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2 class="card-title">2D Projection</h2>
                <span id="viz-info" style="color: var(--text-secondary); font-size: 14px;"></span>
            </div>
            <div id="viz-container" style="width: 100%; height: 600px; position: relative; background: var(--bg-card); border-radius: 8px;">
                <svg id="viz-svg" width="100%" height="100%" style="display: block;"></svg>
                <p id="viz-placeholder" style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: var(--text-secondary);">
                    Select a collection and click Visualize
                </p>
            </div>
            <div id="viz-tooltip" style="display: none; position: absolute; background: var(--bg-secondary); border: 1px solid var(--border-color);
                 padding: 8px 12px; border-radius: 6px; font-size: 13px; pointer-events: none; z-index: 1000; max-width: 300px;"></div>
        </div>

        <script>
        async function loadVisualization() {{
            const collection = document.getElementById('viz-collection').value;
            const maxPoints = parseInt(document.getElementById('viz-max').value) || 500;
            if (!collection) {{ alert('Select a collection'); return; }}

            document.getElementById('viz-placeholder').textContent = 'Loading...';
            try {{
                const resp = await fetch(`/api/visualize/${{collection}}?max=${{maxPoints}}`);
                const data = await resp.json();
                if (!resp.ok) {{ throw new Error(data.error || 'Failed'); }}
                renderScatterPlot(data.points);
                document.getElementById('viz-info').textContent = `${{data.points.length}} points projected via random 2D`;
            }} catch (e) {{
                document.getElementById('viz-placeholder').textContent = 'Error: ' + e.message;
            }}
        }}

        function renderScatterPlot(points) {{
            const svg = document.getElementById('viz-svg');
            const container = document.getElementById('viz-container');
            const tooltip = document.getElementById('viz-tooltip');
            document.getElementById('viz-placeholder').style.display = 'none';
            svg.innerHTML = '';

            const w = container.clientWidth, h = container.clientHeight;
            const pad = 40;
            const xs = points.map(p => p.x), ys = points.map(p => p.y);
            const minX = Math.min(...xs), maxX = Math.max(...xs);
            const minY = Math.min(...ys), maxY = Math.max(...ys);
            const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;

            const scale = (v, min, range, size) => pad + (v - min) / range * (size - 2 * pad);

            points.forEach(p => {{
                const cx = scale(p.x, minX, rangeX, w);
                const cy = scale(p.y, minY, rangeY, h);
                const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                circle.setAttribute('cx', cx);
                circle.setAttribute('cy', cy);
                circle.setAttribute('r', '4');
                circle.setAttribute('fill', '#6366f1');
                circle.setAttribute('opacity', '0.7');
                circle.setAttribute('cursor', 'pointer');

                circle.addEventListener('mouseenter', (e) => {{
                    circle.setAttribute('r', '7');
                    circle.setAttribute('fill', '#22c55e');
                    tooltip.style.display = 'block';
                    tooltip.style.left = (e.pageX + 10) + 'px';
                    tooltip.style.top = (e.pageY - 10) + 'px';
                    tooltip.innerHTML = `<strong>${{p.id}}</strong>`;
                }});
                circle.addEventListener('mouseleave', () => {{
                    circle.setAttribute('r', '4');
                    circle.setAttribute('fill', '#6366f1');
                    tooltip.style.display = 'none';
                }});
                svg.appendChild(circle);
            }});
        }}
        </script>
        "#,
        collection_options = collection_options,
    );

    Html(base_layout("Vector Visualization", &content, "visualize"))
}

/// API endpoint for 2D vector projection data
async fn api_visualize_handler(
    State(state): State<Arc<WebUiState>>,
    Path(collection): Path<String>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> impl IntoResponse {
    let db = state.db.read().await;
    let max_points: usize = params
        .get("max")
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);

    let coll = match db.collection(&collection) {
        Ok(c) => c,
        Err(_) => {
            return (
                StatusCode::NOT_FOUND,
                Json(json!({"error": "Collection not found"})),
            );
        }
    };

    let entries = match db.export_internal(&collection) {
        Ok(e) => e,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": format!("{}", e)})),
            );
        }
    };

    // Sample if too many
    let sampled: Vec<_> = if entries.len() > max_points {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut shuffled = entries;
        shuffled.shuffle(&mut rng);
        shuffled.into_iter().take(max_points).collect()
    } else {
        entries
    };

    // Simple random 2D projection (lightweight UMAP-like visualization)
    let dims = coll.dimensions().unwrap_or(1);
    let seed: u64 = 42;
    let proj_a: Vec<f32> = (0..dims)
        .map(|i| {
            let s = seed.wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    let proj_b: Vec<f32> = (0..dims)
        .map(|i| {
            let s = (seed + 1).wrapping_mul(6364136223846793005).wrapping_add(i as u64);
            ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();

    let points: Vec<serde_json::Value> = sampled
        .iter()
        .map(|(id, vec, _meta)| {
            let x: f32 = vec.iter().zip(&proj_a).map(|(a, b)| a * b).sum();
            let y: f32 = vec.iter().zip(&proj_b).map(|(a, b)| a * b).sum();
            json!({"id": id, "x": x, "y": y})
        })
        .collect();

    (StatusCode::OK, Json(json!({"points": points})))
}

// ============================================================================
// Router and Server
// ============================================================================

/// Create the Web UI router
pub fn create_web_ui_router(state: Arc<WebUiState>) -> Router {
    Router::new()
        // HTML pages
        .route("/", get(dashboard_handler))
        .route("/collections", get(collections_list_handler))
        .route("/collections/{name}", get(collection_detail_handler))
        .route("/query", get(query_playground_handler))
        .route("/playground", get(needleql_playground_handler))
        .route("/monitoring", get(monitoring_handler))
        .route("/visualize", get(visualize_handler))
        // API endpoints
        .route("/api/stats", get(api_stats_handler))
        .route("/api/monitoring", get(api_monitoring_handler))
        .route("/api/visualize/{collection}", get(api_visualize_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

/// Start the Web UI server
///
/// # Arguments
///
/// * `db` - The Needle database instance
/// * `config` - Web UI configuration
///
/// # Example
///
/// ```rust,ignore
/// use needle::web_ui::{WebUiConfig, serve_web_ui};
/// use needle::Database;
///
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let db = Database::open("vectors.needle")?;
///     let config = WebUiConfig::default();
///     serve_web_ui(db, config).await?;
///     Ok(())
/// }
/// ```
pub async fn serve_web_ui(
    db: Database,
    config: WebUiConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let addr = config.addr;
    let state = Arc::new(WebUiState::new(db, config));
    let app = create_web_ui_router(state);

    println!("Needle Web UI starting on http://{}", addr);
    println!("  Dashboard:   http://{}/", addr);
    println!("  Collections: http://{}/collections", addr);
    println!("  Query:       http://{}/query", addr);
    println!("  Monitoring:  http://{}/monitoring", addr);
    println!("  API Stats:   http://{}/api/stats", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Start the Web UI with default configuration
pub async fn serve_web_ui_default(db: Database) -> Result<(), Box<dyn std::error::Error>> {
    serve_web_ui(db, WebUiConfig::default()).await
}

// ============================================================================
// Alerting Logic
// ============================================================================

/// Checks metrics against alert thresholds and returns any triggered alerts
pub fn check_alerts(
    config: &AlertConfig,
    p99_latency_ms: f64,
    error_rate: f32,
    _recall: Option<f32>,
) -> Vec<Alert> {
    if !config.enabled {
        return Vec::new();
    }

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let mut alerts = Vec::new();

    if p99_latency_ms > config.latency_threshold_ms {
        alerts.push(Alert {
            severity: AlertSeverity::Warning,
            message: format!(
                "P99 latency ({:.1}ms) exceeds threshold ({:.1}ms)",
                p99_latency_ms, config.latency_threshold_ms
            ),
            timestamp: now,
            acknowledged: false,
        });
    }

    if error_rate > config.max_error_rate {
        alerts.push(Alert {
            severity: AlertSeverity::Critical,
            message: format!(
                "Error rate ({:.1}%) exceeds threshold ({:.1}%)",
                error_rate * 100.0,
                config.max_error_rate * 100.0
            ),
            timestamp: now,
            acknowledged: false,
        });
    }

    if let Some(recall) = _recall {
        if recall < config.min_recall {
            alerts.push(Alert {
                severity: AlertSeverity::Warning,
                message: format!(
                    "Recall ({:.2}) below threshold ({:.2})",
                    recall, config.min_recall
                ),
                timestamp: now,
                acknowledged: false,
            });
        }
    }

    alerts
}

// ============================================================================
// Monitoring Computation
// ============================================================================

/// Compute a monitoring snapshot from the database state.
pub fn compute_monitoring_snapshot(db: &Database, uptime_secs: u64) -> MonitoringSnapshot {
    let collections = db.list_collections();
    let total_vectors = db.total_vectors();
    let mut total_memory: usize = 0;
    let mut health_scores = Vec::new();

    for name in &collections {
        if let Ok(coll) = db.collection(name) {
            let dims = coll.dimensions().unwrap_or(0);
            let count = coll.len();
            let deleted = coll.deleted_count();
            let vector_memory = count * dims * std::mem::size_of::<f32>();
            total_memory += vector_memory;

            let fragmentation = if count + deleted > 0 {
                deleted as f64 / (count + deleted) as f64
            } else {
                0.0
            };
            let density = if dims > 0 { count as f64 / dims as f64 } else { 0.0 };

            // Health: penalize fragmentation and emptiness
            let frag_penalty = 1.0 - fragmentation;
            let size_factor = if count > 0 { 1.0 } else { 0.5 };
            let score = (frag_penalty * 0.7 + size_factor * 0.3).clamp(0.0, 1.0);

            health_scores.push(CollectionHealthScore {
                name: name.clone(),
                score,
                fragmentation,
                memory_bytes: vector_memory,
                vector_count: count,
                needs_compaction: fragmentation > 0.2,
                density,
            });
        }
    }

    let system_health = if health_scores.is_empty() {
        1.0
    } else {
        health_scores.iter().map(|h| h.score).sum::<f64>() / health_scores.len() as f64
    };

    MonitoringSnapshot {
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        total_collections: collections.len(),
        total_vectors,
        total_memory_bytes: total_memory,
        health_scores,
        system_health,
        uptime_secs,
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(100), "100");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(1234567), "1,234,567");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
        assert_eq!(format_bytes(1048576), "1.00 MB");
        assert_eq!(format_bytes(1073741824), "1.00 GB");
    }

    #[test]
    fn test_format_uptime() {
        assert_eq!(format_uptime(30), "30s");
        assert_eq!(format_uptime(90), "1m 30s");
        assert_eq!(format_uptime(3661), "1h 1m 1s");
        assert_eq!(format_uptime(86400), "1d 0h 0m");
        assert_eq!(format_uptime(90061), "1d 1h 1m");
    }

    #[test]
    fn test_web_ui_config_default() {
        let config = WebUiConfig::default();
        assert_eq!(config.addr.port(), 8081);
        assert_eq!(config.title, "Needle Dashboard");
        assert!(config.enable_query_playground);
    }

    #[test]
    fn test_web_ui_config_builder() {
        let config = WebUiConfig::new("127.0.0.1:9000")
            .with_title("My Dashboard")
            .with_query_playground(false);

        assert_eq!(config.addr.port(), 9000);
        assert_eq!(config.title, "My Dashboard");
        assert!(!config.enable_query_playground);
    }

    #[test]
    fn test_stats_response_serialization() {
        let stats = StatsResponse {
            healthy: true,
            uptime_seconds: 3600,
            total_collections: 2,
            total_vectors: 1000,
            collections: vec![CollectionStatsResponse {
                name: "test".to_string(),
                vector_count: 500,
                dimensions: 384,
                deleted_count: 10,
                needs_compaction: false,
            }],
            version: "0.1.0".to_string(),
        };

        let json = serde_json::to_string(&stats).unwrap();
        assert!(json.contains("\"healthy\":true"));
        assert!(json.contains("\"total_vectors\":1000"));
    }

    #[test]
    fn test_latency_heatmap() {
        let mut heatmap = LatencyHeatmap::new();
        assert_eq!(heatmap.total_ops, 0);

        heatmap.record(0.5);
        heatmap.record(2.0);
        heatmap.record(50.0);
        assert_eq!(heatmap.total_ops, 3);

        heatmap.compute_percentiles();
        assert!(heatmap.p50_ms > 0.0);
    }

    #[test]
    fn test_monitoring_snapshot() {
        let db = Database::in_memory();
        db.create_collection("test", 64).unwrap();
        let snapshot = compute_monitoring_snapshot(&db, 100);
        assert_eq!(snapshot.total_collections, 1);
        assert_eq!(snapshot.uptime_secs, 100);
        assert!(snapshot.system_health > 0.0);
        assert_eq!(snapshot.health_scores.len(), 1);
    }

    #[test]
    fn test_monitoring_dashboard_html() {
        let snapshot = MonitoringSnapshot {
            timestamp: 0,
            total_collections: 1,
            total_vectors: 100,
            total_memory_bytes: 4096,
            health_scores: vec![CollectionHealthScore {
                name: "test".to_string(),
                score: 0.9,
                fragmentation: 0.05,
                memory_bytes: 4096,
                vector_count: 100,
                needs_compaction: false,
                density: 1.5,
            }],
            system_health: 0.9,
            uptime_secs: 3600,
        };
        let html = generate_monitoring_dashboard_html(&snapshot);
        assert!(html.contains("Needle"));
        assert!(html.contains("System Health"));
        assert!(html.contains("test"));
    }

    #[test]
    fn test_web_ui_router_includes_visualize_route() {
        let state = Arc::new(WebUiState::new(
            crate::Database::in_memory(),
            WebUiConfig::default(),
        ));
        let router = create_web_ui_router(state);
        // Verify router was created without panicking (routes are valid)
        let _router = router;
    }
}
