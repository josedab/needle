//! Plugin management REST API handlers.

use crate::experimental::plugin_registry::{PluginRegistry, RegistryConfig};
use axum::{
    extract::Path,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;

/// `GET /v1/plugins` — list all registered plugins.
pub(in crate::server) async fn list_plugins() -> impl IntoResponse {
    let registry = PluginRegistry::new(RegistryConfig::default());
    let plugins = registry.list_all();
    let items: Vec<_> = plugins
        .iter()
        .map(|p| {
            json!({
                "id": p.manifest.id,
                "name": p.manifest.name,
                "version": p.manifest.version,
                "type": p.manifest.plugin_type.to_string(),
                "description": p.manifest.description,
                "verified": p.verified,
                "deprecated": p.deprecated,
            })
        })
        .collect();
    Json(json!({ "plugins": items, "count": items.len() }))
}

/// `GET /v1/plugins/:name` — get details for a single plugin.
pub(in crate::server) async fn get_plugin(
    Path(name): Path<String>,
) -> impl IntoResponse {
    let registry = PluginRegistry::new(RegistryConfig::default());
    match registry.get(&name) {
        Some(plugin) => Json(json!({
            "id": plugin.manifest.id,
            "name": plugin.manifest.name,
            "version": plugin.manifest.version,
            "type": plugin.manifest.plugin_type.to_string(),
            "description": plugin.manifest.description,
            "author": plugin.manifest.author,
            "license": plugin.manifest.license,
            "size_bytes": plugin.manifest.size_bytes,
            "capabilities": plugin.manifest.capabilities,
            "dependencies": plugin.manifest.dependencies,
            "min_needle_version": plugin.manifest.min_needle_version,
            "verified": plugin.verified,
            "deprecated": plugin.deprecated,
            "downloads": plugin.downloads,
        }))
        .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": format!("Plugin '{}' not found", name) })),
        )
            .into_response(),
    }
}
