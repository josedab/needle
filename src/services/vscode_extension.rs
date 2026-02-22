//! VS Code Extension Protocol
//!
//! Extension manifest, semantic code search protocol, and LSP integration
//! types for building a Needle-powered VS Code extension.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::vscode_extension::{
//!     ExtensionManifest, SearchRequest, SearchResponse, CodeChunk,
//! };
//!
//! let manifest = ExtensionManifest::default();
//! println!("Extension: {} v{}", manifest.name, manifest.version);
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// VS Code extension manifest (package.json equivalent).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionManifest {
    pub name: String,
    pub display_name: String,
    pub description: String,
    pub version: String,
    pub publisher: String,
    pub categories: Vec<String>,
    pub activation_events: Vec<String>,
    pub contributes: ExtensionContributions,
}

impl Default for ExtensionManifest {
    fn default() -> Self {
        Self {
            name: "needle-search".into(), display_name: "Needle Semantic Search".into(),
            description: "Semantic code search powered by Needle vector database".into(),
            version: "0.1.0".into(), publisher: "anthropics".into(),
            categories: vec!["Search".into(), "Other".into()],
            activation_events: vec!["onCommand:needle.search".into(), "onCommand:needle.index".into()],
            contributes: ExtensionContributions::default(),
        }
    }
}

/// Extension contributions (commands, configuration).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionContributions {
    pub commands: Vec<ExtensionCommand>,
    pub configuration: Value,
}

impl Default for ExtensionContributions {
    fn default() -> Self {
        Self {
            commands: vec![
                ExtensionCommand { command: "needle.search".into(), title: "Needle: Semantic Search".into() },
                ExtensionCommand { command: "needle.index".into(), title: "Needle: Index Workspace".into() },
                ExtensionCommand { command: "needle.status".into(), title: "Needle: Show Index Status".into() },
            ],
            configuration: serde_json::json!({
                "type": "object",
                "title": "Needle Semantic Search",
                "properties": {
                    "needle.dimensions": { "type": "number", "default": 384, "description": "Embedding dimensions" },
                    "needle.excludePatterns": { "type": "array", "default": ["**/node_modules/**", "**/target/**"], "description": "Glob patterns to exclude" },
                    "needle.maxFileSize": { "type": "number", "default": 100000, "description": "Max file size in bytes" }
                }
            }),
        }
    }
}

/// An extension command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionCommand { pub command: String, pub title: String }

/// Code chunk for indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub content: String,
    pub kind: ChunkKind,
}

/// Kind of code chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChunkKind { Function, Struct, Enum, Trait, Impl, Module, Comment, Other }

/// Search request from VS Code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub max_results: usize,
    pub file_filter: Option<String>,
    pub language_filter: Option<String>,
}

/// Search response to VS Code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub query_time_ms: f64,
    pub total_indexed: usize,
}

/// A single search result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub preview: String,
    pub score: f32,
    pub language: String,
    pub kind: ChunkKind,
}

/// Index status report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStatus {
    pub indexed_files: usize,
    pub indexed_chunks: usize,
    pub index_size_bytes: usize,
    pub last_indexed: Option<String>,
    pub languages: Vec<(String, usize)>,
}

/// Generate the extension's package.json.
pub fn generate_package_json(manifest: &ExtensionManifest) -> String {
    serde_json::to_string_pretty(manifest).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_manifest() {
        let m = ExtensionManifest::default();
        assert_eq!(m.name, "needle-search");
        assert!(m.contributes.commands.len() >= 3);
    }

    #[test]
    fn test_package_json() {
        let json = generate_package_json(&ExtensionManifest::default());
        assert!(json.contains("needle-search"));
        assert!(json.contains("needle.search"));
    }

    #[test]
    fn test_search_request() {
        let req = SearchRequest { query: "auth handler".into(), max_results: 10, file_filter: Some("*.rs".into()), language_filter: None };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("auth handler"));
    }

    #[test]
    fn test_code_chunk() {
        let chunk = CodeChunk {
            file_path: "src/main.rs".into(), start_line: 1, end_line: 10,
            language: "rust".into(), content: "fn main() {}".into(), kind: ChunkKind::Function,
        };
        assert_eq!(chunk.language, "rust");
    }

    #[test]
    fn test_index_status() {
        let status = IndexStatus {
            indexed_files: 100, indexed_chunks: 500, index_size_bytes: 1_000_000,
            last_indexed: Some("2026-02-22T12:00:00Z".into()),
            languages: vec![("rust".into(), 400), ("python".into(), 100)],
        };
        assert_eq!(status.indexed_files, 100);
    }
}
