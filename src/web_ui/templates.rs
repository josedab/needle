use super::state::{
    ActionParam, AdminAction, AdminSection, MonitoringSnapshot, VisualQueryState,
};

// ============================================================================
// CSS Styles
// ============================================================================

/// Inline CSS styles for the dashboard
pub(crate) const CSS_STYLES: &str = r#"
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --secondary: #64748b;
    --success: #22c55e;
    --warning: #f59e0b;
    --danger: #ef4444;
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-color: #475569;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 1rem;
}

/* Navigation */
nav {
    background-color: var(--bg-secondary);
    border-bottom: 1px solid var(--border-color);
    padding: 1rem 0;
    position: sticky;
    top: 0;
    z-index: 100;
}

nav .container {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary);
    text-decoration: none;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.nav-links {
    display: flex;
    gap: 1.5rem;
    list-style: none;
}

.nav-links a {
    color: var(--text-secondary);
    text-decoration: none;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    transition: all 0.2s;
}

.nav-links a:hover,
.nav-links a.active {
    color: var(--text-primary);
    background-color: var(--bg-card);
}

/* Cards */
.card {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 0.5rem;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}

.card-title {
    font-size: 1.125rem;
    font-weight: 600;
}

/* Grid layouts */
.grid {
    display: grid;
    gap: 1rem;
}

.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }

@media (max-width: 768px) {
    .grid-2, .grid-3, .grid-4 {
        grid-template-columns: 1fr;
    }
}

/* Stats */
.stat {
    text-align: center;
    padding: 1rem;
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

.stat-label {
    font-size: 0.875rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
}

th, td {
    padding: 0.75rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border-color);
}

th {
    background-color: var(--bg-card);
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}

tr:hover {
    background-color: var(--bg-card);
}

/* Badges */
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-success { background-color: var(--success); color: white; }
.badge-warning { background-color: var(--warning); color: black; }
.badge-danger { background-color: var(--danger); color: white; }
.badge-primary { background-color: var(--primary); color: white; }

/* Buttons */
.btn {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    font-size: 0.875rem;
    font-weight: 500;
    text-decoration: none;
    cursor: pointer;
    border: none;
    transition: all 0.2s;
}

.btn-primary {
    background-color: var(--primary);
    color: white;
}

.btn-primary:hover {
    background-color: var(--primary-dark);
}

.btn-secondary {
    background-color: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
}

.btn-secondary:hover {
    background-color: var(--border-color);
}

/* Forms */
.form-group {
    margin-bottom: 1rem;
}

.form-label {
    display: block;
    margin-bottom: 0.5rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.form-input, .form-textarea {
    width: 100%;
    padding: 0.75rem;
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    background-color: var(--bg-card);
    color: var(--text-primary);
    font-size: 1rem;
}

.form-input:focus, .form-textarea:focus {
    outline: none;
    border-color: var(--primary);
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
}

.form-textarea {
    min-height: 120px;
    font-family: monospace;
    resize: vertical;
}

/* Health indicator */
.health-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.health-dot {
    width: 0.75rem;
    height: 0.75rem;
    border-radius: 50%;
    animation: pulse 2s infinite;
}

.health-dot.healthy { background-color: var(--success); }
.health-dot.warning { background-color: var(--warning); }
.health-dot.error { background-color: var(--danger); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Main content */
main {
    padding: 2rem 0;
}

.page-header {
    margin-bottom: 2rem;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.page-description {
    color: var(--text-secondary);
}

/* Code blocks */
pre, code {
    font-family: 'Fira Code', 'Monaco', 'Consolas', monospace;
    background-color: var(--bg-card);
    border-radius: 0.375rem;
}

pre {
    padding: 1rem;
    overflow-x: auto;
}

code {
    padding: 0.125rem 0.375rem;
    font-size: 0.875em;
}

/* Results */
.results-list {
    list-style: none;
}

.result-item {
    padding: 1rem;
    border: 1px solid var(--border-color);
    border-radius: 0.375rem;
    margin-bottom: 0.5rem;
    background-color: var(--bg-card);
}

.result-item:hover {
    border-color: var(--primary);
}

.result-id {
    font-weight: 600;
    color: var(--primary);
}

.result-distance {
    color: var(--text-secondary);
    font-size: 0.875rem;
}

/* Progress bar */
.progress-bar {
    height: 0.5rem;
    background-color: var(--bg-card);
    border-radius: 9999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background-color: var(--primary);
    transition: width 0.3s ease;
}

/* Footer */
footer {
    padding: 2rem 0;
    margin-top: 2rem;
    border-top: 1px solid var(--border-color);
    text-align: center;
    color: var(--text-secondary);
    font-size: 0.875rem;
}
"#;

// ============================================================================
// Template Helpers
// ============================================================================

/// Generate the base HTML layout with navigation
pub(crate) fn base_layout(title: &str, content: &str, active_page: &str) -> String {
    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Needle Dashboard</title>
    <style>{CSS_STYLES}</style>
</head>
<body>
    <nav>
        <div class="container">
            <a href="/" class="logo">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
                Needle
            </a>
            <ul class="nav-links">
                <li><a href="/" class="{dashboard_active}">Dashboard</a></li>
                <li><a href="/collections" class="{collections_active}">Collections</a></li>
                <li><a href="/query" class="{query_active}">Query</a></li>
                <li><a href="/monitoring" class="{monitoring_active}">Monitoring</a></li>
                <li><a href="/playground" class="{playground_active}">Playground</a></li>
            </ul>
        </div>
    </nav>
    <main>
        <div class="container">
            {content}
        </div>
    </main>
    <footer>
        <div class="container">
            <p>Needle Vector Database &copy; 2024 - v{version}</p>
        </div>
    </footer>
</body>
</html>"#,
        title = title,
        content = content,
        version = env!("CARGO_PKG_VERSION"),
        dashboard_active = if active_page == "dashboard" {
            "active"
        } else {
            ""
        },
        collections_active = if active_page == "collections" {
            "active"
        } else {
            ""
        },
        query_active = if active_page == "query" { "active" } else { "" },
        monitoring_active = if active_page == "monitoring" {
            "active"
        } else {
            ""
        },
        playground_active = if active_page == "playground" {
            "active"
        } else {
            ""
        },
    )
}

/// Format a number with thousands separators
pub(crate) fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    result
}

/// Format bytes into human-readable size
pub(crate) fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = KB * 1024;
    const GB: usize = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// Format uptime into human-readable duration
pub(crate) fn format_uptime(seconds: u64) -> String {
    let days = seconds / 86400;
    let hours = (seconds % 86400) / 3600;
    let minutes = (seconds % 3600) / 60;
    let secs = seconds % 60;

    if days > 0 {
        format!("{}d {}h {}m", days, hours, minutes)
    } else if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

/// Generate a stat card HTML component
pub(crate) fn stat_card(label: &str, value: &str, icon: &str) -> String {
    format!(
        r#"<div class="card stat">
            <div class="stat-icon">{icon}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-label">{label}</div>
        </div>"#,
        icon = icon,
        value = value,
        label = label
    )
}

// ============================================================================
// Visual Query Builder HTML
// ============================================================================

/// Generate the visual query builder HTML
pub fn generate_query_builder_html(state: &VisualQueryState, collections: &[String]) -> String {
    let collection_options: String = collections
        .iter()
        .map(|c| {
            let selected = state.collection.as_ref() == Some(c);
            format!(
                r#"<option value="{}" {}>{}</option>"#,
                c,
                if selected { "selected" } else { "" },
                c
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let filters_html: String = state
        .filters
        .iter()
        .enumerate()
        .map(|(i, f)| {
            format!(
                r#"
                <div class="filter-row" data-index="{}">
                    <input type="text" name="filter_field_{}" value="{}" placeholder="Field" class="field-input">
                    <select name="filter_op_{}">
                        <option value="eq" {}>equals</option>
                        <option value="ne" {}>not equals</option>
                        <option value="gt" {}>greater than</option>
                        <option value="gte" {}>greater or equal</option>
                        <option value="lt" {}>less than</option>
                        <option value="lte" {}>less or equal</option>
                        <option value="in" {}>in list</option>
                    </select>
                    <input type="text" name="filter_value_{}" value="{}" placeholder="Value" class="value-input">
                    <button type="button" class="remove-filter" onclick="removeFilter({})">✕</button>
                </div>
                "#,
                i,
                i, f.field,
                i,
                if f.operator == "eq" { "selected" } else { "" },
                if f.operator == "ne" { "selected" } else { "" },
                if f.operator == "gt" { "selected" } else { "" },
                if f.operator == "gte" { "selected" } else { "" },
                if f.operator == "lt" { "selected" } else { "" },
                if f.operator == "lte" { "selected" } else { "" },
                if f.operator == "in" { "selected" } else { "" },
                i, serde_json::to_string(&f.value).unwrap_or_default(),
                i
            )
        })
        .collect();

    format!(
        r#"
        <div class="query-builder">
            <h2>Visual Query Builder</h2>
            
            <div class="builder-section">
                <label>Collection</label>
                <select id="collection-select" name="collection">
                    <option value="">Select collection...</option>
                    {}
                </select>
            </div>

            <div class="builder-section">
                <label>Query Vector</label>
                <div class="vector-input-group">
                    <textarea id="query-vector" name="query_vector" 
                        placeholder="Paste vector as JSON array, e.g., [0.1, 0.2, ...]"
                        rows="3">{}</textarea>
                    <button type="button" onclick="generateRandomVector()">Generate Random</button>
                </div>
            </div>

            <div class="builder-section">
                <label>Filters</label>
                <div id="filters-container">
                    {}
                </div>
                <button type="button" onclick="addFilter()">+ Add Filter</button>
            </div>

            <div class="builder-section options">
                <div class="option-group">
                    <label>Results Limit</label>
                    <input type="number" name="limit" value="{}" min="1" max="1000">
                </div>
                <div class="option-group">
                    <label>Distance Function</label>
                    <select name="distance">
                        <option value="cosine" {}>Cosine</option>
                        <option value="euclidean" {}>Euclidean</option>
                        <option value="dot" {}>Dot Product</option>
                    </select>
                </div>
                <div class="option-group">
                    <label>
                        <input type="checkbox" name="include_metadata" {}> Include Metadata
                    </label>
                </div>
                <div class="option-group">
                    <label>
                        <input type="checkbox" name="use_index" {}> Use HNSW Index
                    </label>
                </div>
            </div>

            <div class="builder-actions">
                <button type="submit" class="primary">Execute Query</button>
                <button type="button" onclick="resetBuilder()">Reset</button>
                <button type="button" onclick="generateCurl()">Generate cURL</button>
            </div>
        </div>

        <script>
        let filterIndex = {};
        
        function addFilter() {{
            const container = document.getElementById('filters-container');
            const html = `
                <div class="filter-row" data-index="${{filterIndex}}">
                    <input type="text" name="filter_field_${{filterIndex}}" placeholder="Field" class="field-input">
                    <select name="filter_op_${{filterIndex}}">
                        <option value="eq">equals</option>
                        <option value="ne">not equals</option>
                        <option value="gt">greater than</option>
                        <option value="gte">greater or equal</option>
                        <option value="lt">less than</option>
                        <option value="lte">less or equal</option>
                        <option value="in">in list</option>
                    </select>
                    <input type="text" name="filter_value_${{filterIndex}}" placeholder="Value" class="value-input">
                    <button type="button" class="remove-filter" onclick="removeFilter(${{filterIndex}})">✕</button>
                </div>
            `;
            container.insertAdjacentHTML('beforeend', html);
            filterIndex++;
        }}

        function removeFilter(index) {{
            const row = document.querySelector(`.filter-row[data-index="${{index}}"]`);
            if (row) row.remove();
        }}

        function generateRandomVector() {{
            const collection = document.getElementById('collection-select').value;
            // Default to 384 dimensions
            const dims = 384;
            const vector = Array.from({{length: dims}}, () => Math.random() * 2 - 1);
            document.getElementById('query-vector').value = JSON.stringify(vector);
        }}

        function resetBuilder() {{
            document.getElementById('collection-select').value = '';
            document.getElementById('query-vector').value = '';
            document.getElementById('filters-container').innerHTML = '';
            filterIndex = 0;
        }}

        function generateCurl() {{
            const collection = document.getElementById('collection-select').value;
            const vector = document.getElementById('query-vector').value;
            const limit = document.querySelector('input[name="limit"]').value;
            
            if (!collection || !vector) {{
                alert('Please select a collection and enter a query vector');
                return;
            }}

            const curl = `curl -X POST http://localhost:8080/collections/${{collection}}/search \\
  -H "Content-Type: application/json" \\
  -d '{{"vector": ${{vector}}, "k": ${{limit}}}}'`;
            
            navigator.clipboard.writeText(curl);
            alert('cURL command copied to clipboard!');
        }}
        </script>
        "#,
        collection_options,
        state
            .query_vector
            .as_ref()
            .map(|v| serde_json::to_string(v).unwrap_or_default())
            .unwrap_or_default(),
        filters_html,
        state.limit,
        if state.distance == "cosine" {
            "selected"
        } else {
            ""
        },
        if state.distance == "euclidean" {
            "selected"
        } else {
            ""
        },
        if state.distance == "dot" {
            "selected"
        } else {
            ""
        },
        if state.include_metadata {
            "checked"
        } else {
            ""
        },
        if state.use_index { "checked" } else { "" },
        state.filters.len()
    )
}

// ============================================================================
// Admin Dashboard HTML
// ============================================================================

/// Generate admin sections
pub fn get_admin_sections() -> Vec<AdminSection> {
    vec![
        AdminSection {
            name: "Collection Management".to_string(),
            description: "Create, modify, and delete collections".to_string(),
            actions: vec![
                AdminAction {
                    id: "create_collection".to_string(),
                    name: "Create Collection".to_string(),
                    description: "Create a new vector collection".to_string(),
                    method: "POST".to_string(),
                    endpoint: "/api/collections".to_string(),
                    params: vec![
                        ActionParam {
                            name: "name".to_string(),
                            param_type: "string".to_string(),
                            required: true,
                            default: None,
                            description: "Collection name".to_string(),
                        },
                        ActionParam {
                            name: "dimensions".to_string(),
                            param_type: "number".to_string(),
                            required: true,
                            default: Some(serde_json::json!(384)),
                            description: "Vector dimensions".to_string(),
                        },
                        ActionParam {
                            name: "distance".to_string(),
                            param_type: "string".to_string(),
                            required: false,
                            default: Some(serde_json::json!("cosine")),
                            description: "Distance function".to_string(),
                        },
                    ],
                    dangerous: false,
                },
                AdminAction {
                    id: "delete_collection".to_string(),
                    name: "Delete Collection".to_string(),
                    description: "Delete a collection and all its data".to_string(),
                    method: "DELETE".to_string(),
                    endpoint: "/api/collections/{name}".to_string(),
                    params: vec![ActionParam {
                        name: "name".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        default: None,
                        description: "Collection to delete".to_string(),
                    }],
                    dangerous: true,
                },
                AdminAction {
                    id: "compact_collection".to_string(),
                    name: "Compact Collection".to_string(),
                    description: "Remove deleted vectors and reclaim space".to_string(),
                    method: "POST".to_string(),
                    endpoint: "/api/collections/{name}/compact".to_string(),
                    params: vec![ActionParam {
                        name: "name".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        default: None,
                        description: "Collection to compact".to_string(),
                    }],
                    dangerous: false,
                },
            ],
        },
        AdminSection {
            name: "Data Management".to_string(),
            description: "Import, export, and backup data".to_string(),
            actions: vec![
                AdminAction {
                    id: "export_collection".to_string(),
                    name: "Export Collection".to_string(),
                    description: "Export collection data to JSON".to_string(),
                    method: "GET".to_string(),
                    endpoint: "/api/collections/{name}/export".to_string(),
                    params: vec![ActionParam {
                        name: "name".to_string(),
                        param_type: "string".to_string(),
                        required: true,
                        default: None,
                        description: "Collection to export".to_string(),
                    }],
                    dangerous: false,
                },
                AdminAction {
                    id: "save_database".to_string(),
                    name: "Save Database".to_string(),
                    description: "Persist all changes to disk".to_string(),
                    method: "POST".to_string(),
                    endpoint: "/api/save".to_string(),
                    params: vec![],
                    dangerous: false,
                },
            ],
        },
        AdminSection {
            name: "Monitoring".to_string(),
            description: "View metrics and health status".to_string(),
            actions: vec![
                AdminAction {
                    id: "health_check".to_string(),
                    name: "Health Check".to_string(),
                    description: "Check database health status".to_string(),
                    method: "GET".to_string(),
                    endpoint: "/health".to_string(),
                    params: vec![],
                    dangerous: false,
                },
                AdminAction {
                    id: "get_stats".to_string(),
                    name: "Get Statistics".to_string(),
                    description: "Get database statistics".to_string(),
                    method: "GET".to_string(),
                    endpoint: "/api/stats".to_string(),
                    params: vec![],
                    dangerous: false,
                },
            ],
        },
    ]
}

/// Generate the admin dashboard HTML
pub fn generate_admin_dashboard_html() -> String {
    let sections = get_admin_sections();

    let sections_html: String = sections
        .iter()
        .map(|section| {
            let actions_html: String = section
                .actions
                .iter()
                .map(|action| {
                    let params_html: String = action
                        .params
                        .iter()
                        .map(|p| {
                            let input_type = match p.param_type.as_str() {
                                "number" => "number",
                                "boolean" => "checkbox",
                                _ => "text",
                            };
                            format!(
                                r#"
                                <div class="param-row">
                                    <label>{}{}</label>
                                    <input type="{}" name="{}" value="{}" placeholder="{}">
                                </div>
                                "#,
                                p.name,
                                if p.required { " *" } else { "" },
                                input_type,
                                p.name,
                                p.default
                                    .as_ref()
                                    .map(|v| v.to_string())
                                    .unwrap_or_default(),
                                p.description
                            )
                        })
                        .collect();

                    let danger_class = if action.dangerous { "danger" } else { "" };

                    format!(
                        r#"
                        <div class="action-card {}">
                            <h4>{}</h4>
                            <p class="description">{}</p>
                            <div class="action-details">
                                <span class="method method-{}">{}</span>
                                <code>{}</code>
                            </div>
                            <form class="action-form" data-endpoint="{}" data-method="{}">
                                {}
                                <button type="submit" class="{}">{}</button>
                            </form>
                        </div>
                        "#,
                        danger_class,
                        action.name,
                        action.description,
                        action.method.to_lowercase(),
                        action.method,
                        action.endpoint,
                        action.endpoint,
                        action.method,
                        params_html,
                        danger_class,
                        if action.dangerous {
                            "⚠ Execute"
                        } else {
                            "Execute"
                        }
                    )
                })
                .collect();

            format!(
                r#"
                <div class="admin-section">
                    <h3>{}</h3>
                    <p class="section-desc">{}</p>
                    <div class="actions-grid">
                        {}
                    </div>
                </div>
                "#,
                section.name, section.description, actions_html
            )
        })
        .collect();

    format!(
        r#"
        <div class="admin-dashboard">
            <h2>Admin Dashboard</h2>
            <p class="warning">⚠️ Actions here can modify or delete data. Use with caution.</p>
            {}
        </div>

        <style>
        .admin-dashboard {{ padding: 20px; max-width: 1200px; margin: 0 auto; }}
        .warning {{ color: #f59e0b; background: #fef3c7; padding: 10px; border-radius: 4px; }}
        .admin-section {{ margin: 30px 0; padding: 20px; background: #f8fafc; border-radius: 8px; }}
        .section-desc {{ color: #64748b; margin-bottom: 20px; }}
        .actions-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(350px, 1fr)); gap: 20px; }}
        .action-card {{ background: white; padding: 20px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .action-card.danger {{ border-color: #ef4444; }}
        .action-card h4 {{ margin: 0 0 10px 0; }}
        .action-card .description {{ color: #64748b; font-size: 14px; }}
        .action-details {{ margin: 15px 0; }}
        .method {{ padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .method-get {{ background: #22c55e; color: white; }}
        .method-post {{ background: #3b82f6; color: white; }}
        .method-delete {{ background: #ef4444; color: white; }}
        .action-form {{ margin-top: 15px; }}
        .param-row {{ margin: 10px 0; }}
        .param-row label {{ display: block; font-size: 14px; margin-bottom: 4px; }}
        .param-row input {{ width: 100%; padding: 8px; border: 1px solid #d1d5db; border-radius: 4px; }}
        .action-form button {{ margin-top: 10px; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }}
        .action-form button {{ background: #3b82f6; color: white; }}
        .action-form button.danger {{ background: #ef4444; }}
        </style>

        <script>
        document.querySelectorAll('.action-form').forEach(form => {{
            form.addEventListener('submit', async (e) => {{
                e.preventDefault();
                const endpoint = form.dataset.endpoint;
                const method = form.dataset.method;
                
                // Confirm dangerous actions
                if (form.querySelector('button.danger')) {{
                    if (!confirm('Are you sure you want to perform this action?')) {{
                        return;
                    }}
                }}

                // Build the request
                const formData = new FormData(form);
                let url = endpoint;
                let body = null;

                // Replace path params
                formData.forEach((value, key) => {{
                    if (url.includes(`{{${{key}}}}`)) {{
                        url = url.replace(`{{${{key}}}}`, value);
                    }}
                }});

                if (method !== 'GET') {{
                    const data = {{}};
                    formData.forEach((value, key) => {{
                        if (!endpoint.includes(`{{${{key}}}}`)) {{
                            data[key] = value;
                        }}
                    }});
                    body = JSON.stringify(data);
                }}

                try {{
                    const response = await fetch(url, {{
                        method: method,
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: body
                    }});
                    const result = await response.json();
                    alert(response.ok ? 'Success!' : 'Error: ' + JSON.stringify(result));
                }} catch (err) {{
                    alert('Error: ' + err.message);
                }}
            }});
        }});
        </script>
        "#,
        sections_html
    )
}

// ============================================================================
// Real-Time Monitoring Dashboard HTML
// ============================================================================

/// Generate the real-time monitoring dashboard HTML page.
pub fn generate_monitoring_dashboard_html(snapshot: &MonitoringSnapshot) -> String {
    let health_class = if snapshot.system_health >= 0.8 {
        "success"
    } else if snapshot.system_health >= 0.5 {
        "warning"
    } else {
        "danger"
    };

    let collection_rows: String = snapshot
        .health_scores
        .iter()
        .map(|h| {
            let status = if h.score >= 0.8 { "&#x2705;" } else if h.score >= 0.5 { "&#x26A0;&#xFE0F;" } else { "&#x274C;" };
            format!(
                "<tr><td>{name}</td><td>{count}</td><td>{mem}</td><td>{frag:.1}%</td><td>{score:.0}%</td><td>{status}</td></tr>",
                name = h.name,
                count = format_number(h.vector_count),
                mem = format_bytes(h.memory_bytes),
                frag = h.fragmentation * 100.0,
                score = h.score * 100.0,
                status = status,
            )
        })
        .collect();

    format!(
        r#"<!DOCTYPE html>
<html><head><title>Needle — Real-Time Monitoring</title>
<meta http-equiv="refresh" content="5">
<style>{css}</style></head>
<body>
<nav><div class="container"><span class="logo">&#x1F50D; Needle Monitoring</span></div></nav>
<div class="container" style="padding:2rem 1rem;">
<div class="grid grid-4">
  <div class="card"><div class="stat-value" style="color:var(--{health_class})">{health:.0}%</div><div class="stat-label">System Health</div></div>
  <div class="card"><div class="stat-value">{collections}</div><div class="stat-label">Collections</div></div>
  <div class="card"><div class="stat-value">{vectors}</div><div class="stat-label">Total Vectors</div></div>
  <div class="card"><div class="stat-value">{memory}</div><div class="stat-label">Memory Usage</div></div>
</div>
<div class="card"><div class="card-header"><span class="card-title">Collection Health</span></div>
<table class="data-table" style="width:100%;"><thead><tr><th>Collection</th><th>Vectors</th><th>Memory</th><th>Fragmentation</th><th>Health</th><th>Status</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<div class="card"><div class="card-header"><span class="card-title">Uptime</span></div><p>{uptime}</p></div>
</div></body></html>"#,
        css = CSS_STYLES,
        health_class = health_class,
        health = snapshot.system_health * 100.0,
        collections = format_number(snapshot.total_collections),
        vectors = format_number(snapshot.total_vectors),
        memory = format_bytes(snapshot.total_memory_bytes),
        rows = collection_rows,
        uptime = format_uptime(snapshot.uptime_secs),
    )
}
