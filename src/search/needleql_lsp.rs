//! NeedleQL Language Server Protocol (LSP)
//!
//! IDE integration providing autocomplete, syntax validation, inline query results,
//! and EXPLAIN visualization for NeedleQL queries in VS Code, JetBrains, and Neovim.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::needleql_lsp::{
//!     LspServer, LspConfig, CompletionRequest, DiagnosticResult,
//! };
//!
//! let mut server = LspServer::new(LspConfig::default());
//!
//! // Register known collections for autocomplete
//! server.register_collection("documents", 384, vec!["title".into(), "category".into()]);
//!
//! // Get completions
//! let completions = server.complete(&CompletionRequest {
//!     text: "SELECT * FROM ".into(),
//!     cursor_pos: 14,
//! });
//! assert!(completions.items.iter().any(|c| c.label == "documents"));
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── LSP Types ───────────────────────────────────────────────────────────────

/// LSP position in a document.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    pub line: usize,
    pub character: usize,
}

/// LSP range.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Range {
    pub start: Position,
    pub end: Position,
}

/// Diagnostic severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Error,
    Warning,
    Information,
    Hint,
}

/// A diagnostic message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub range: Range,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub source: String,
    pub code: Option<String>,
}

/// Diagnostic result for a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticResult {
    pub diagnostics: Vec<Diagnostic>,
    pub is_valid: bool,
}

// ── Completion ──────────────────────────────────────────────────────────────

/// Completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Full text content.
    pub text: String,
    /// Cursor position (byte offset).
    pub cursor_pos: usize,
}

/// Completion kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompletionKind {
    Keyword,
    Collection,
    Field,
    Function,
    Operator,
    Snippet,
}

/// A completion item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    pub label: String,
    pub kind: CompletionKind,
    pub detail: Option<String>,
    pub insert_text: Option<String>,
    pub documentation: Option<String>,
    pub sort_priority: u32,
}

/// Completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub items: Vec<CompletionItem>,
    pub is_incomplete: bool,
}

// ── Hover ───────────────────────────────────────────────────────────────────

/// Hover information at a position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoverInfo {
    pub contents: String,
    pub range: Option<Range>,
}

// ── Signature Help ──────────────────────────────────────────────────────────

/// Signature help for a function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignatureHelp {
    pub label: String,
    pub documentation: Option<String>,
    pub parameters: Vec<ParameterInfo>,
    pub active_parameter: usize,
}

/// Parameter info for a function signature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterInfo {
    pub label: String,
    pub documentation: Option<String>,
}

// ── Collection Metadata ─────────────────────────────────────────────────────

/// Metadata about a known collection for IDE intelligence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMetadata {
    pub name: String,
    pub dimensions: usize,
    pub fields: Vec<String>,
    pub vector_count: Option<u64>,
}

// ── LSP Config ──────────────────────────────────────────────────────────────

/// LSP server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspConfig {
    /// Enable auto-complete.
    pub enable_completion: bool,
    /// Enable diagnostics.
    pub enable_diagnostics: bool,
    /// Enable hover information.
    pub enable_hover: bool,
    /// Enable signature help.
    pub enable_signature_help: bool,
    /// Maximum completions to return.
    pub max_completions: usize,
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            enable_completion: true,
            enable_diagnostics: true,
            enable_hover: true,
            enable_signature_help: true,
            max_completions: 50,
        }
    }
}

// ── NeedleQL Keywords ───────────────────────────────────────────────────────

const KEYWORDS: &[&str] = &[
    "SELECT", "FROM", "WHERE", "LIMIT", "OFFSET", "ORDER", "BY",
    "ASC", "DESC", "AND", "OR", "NOT", "IN", "LIKE",
    "NEAREST_TO", "HYBRID_SEARCH", "EXPLAIN", "WITH",
    "INSERT", "INTO", "VALUES", "DELETE", "CREATE", "DROP",
    "COLLECTION", "DIMENSIONS", "IF", "EXISTS", "SHOW", "COLLECTIONS",
];

const FUNCTIONS: &[(&str, &str, &str)] = &[
    ("NEAREST_TO", "NEAREST_TO([vector])", "Vector similarity search using HNSW index"),
    ("HYBRID_SEARCH", "HYBRID_SEARCH('text query')", "Combined BM25 text + vector search with RRF fusion"),
    ("COUNT", "COUNT(*)", "Count matching results"),
];

const OPERATORS: &[(&str, &str)] = &[
    ("=", "Equal"),
    ("!=", "Not equal"),
    (">", "Greater than"),
    (">=", "Greater than or equal"),
    ("<", "Less than"),
    ("<=", "Less than or equal"),
    ("IN", "Value in list"),
    ("LIKE", "Pattern match"),
];

// ── LSP Server ──────────────────────────────────────────────────────────────

/// NeedleQL LSP server.
pub struct LspServer {
    config: LspConfig,
    collections: HashMap<String, CollectionMetadata>,
    open_documents: HashMap<String, String>,
}

impl LspServer {
    /// Create a new LSP server.
    pub fn new(config: LspConfig) -> Self {
        Self {
            config,
            collections: HashMap::new(),
            open_documents: HashMap::new(),
        }
    }

    /// Register a known collection for intelligence.
    pub fn register_collection(
        &mut self,
        name: &str,
        dimensions: usize,
        fields: Vec<String>,
    ) {
        self.collections.insert(
            name.to_string(),
            CollectionMetadata {
                name: name.to_string(),
                dimensions,
                fields,
                vector_count: None,
            },
        );
    }

    /// Update collection metadata.
    pub fn update_collection_metadata(
        &mut self,
        name: &str,
        vector_count: u64,
    ) -> Result<()> {
        let meta = self.collections.get_mut(name).ok_or_else(|| {
            NeedleError::NotFound(format!("Collection '{name}'"))
        })?;
        meta.vector_count = Some(vector_count);
        Ok(())
    }

    /// Open a document for tracking.
    pub fn open_document(&mut self, uri: &str, content: &str) {
        self.open_documents
            .insert(uri.to_string(), content.to_string());
    }

    /// Update a document's content.
    pub fn update_document(&mut self, uri: &str, content: &str) {
        self.open_documents
            .insert(uri.to_string(), content.to_string());
    }

    /// Close a document.
    pub fn close_document(&mut self, uri: &str) {
        self.open_documents.remove(uri);
    }

    // ── Completion ──────────────────────────────────────────────────────

    /// Get completions at the cursor position.
    pub fn complete(&self, request: &CompletionRequest) -> CompletionResponse {
        if !self.config.enable_completion {
            return CompletionResponse {
                items: vec![],
                is_incomplete: false,
            };
        }

        let text = &request.text;
        let cursor = request.cursor_pos.min(text.len());
        let before_cursor = &text[..cursor];
        let context = Self::analyze_context(before_cursor);

        let mut items = Vec::new();

        match context {
            CompletionContext::AfterFrom | CompletionContext::AfterInto | CompletionContext::AfterCollection => {
                // Suggest collection names
                for coll in self.collections.values() {
                    let detail = format!(
                        "{}D, {} vectors",
                        coll.dimensions,
                        coll.vector_count.unwrap_or(0)
                    );
                    items.push(CompletionItem {
                        label: coll.name.clone(),
                        kind: CompletionKind::Collection,
                        detail: Some(detail),
                        insert_text: Some(coll.name.clone()),
                        documentation: None,
                        sort_priority: 1,
                    });
                }
            }
            CompletionContext::AfterWhere | CompletionContext::AfterAnd | CompletionContext::AfterOr => {
                // Suggest metadata field names from the current collection
                if let Some(coll_name) = Self::extract_collection_name(before_cursor) {
                    if let Some(coll) = self.collections.get(&coll_name) {
                        for field in &coll.fields {
                            items.push(CompletionItem {
                                label: field.clone(),
                                kind: CompletionKind::Field,
                                detail: Some("metadata field".into()),
                                insert_text: Some(field.clone()),
                                documentation: None,
                                sort_priority: 1,
                            });
                        }
                    }
                }
                // Also suggest operators
                for (op, desc) in OPERATORS {
                    items.push(CompletionItem {
                        label: op.to_string(),
                        kind: CompletionKind::Operator,
                        detail: Some(desc.to_string()),
                        insert_text: Some(op.to_string()),
                        documentation: None,
                        sort_priority: 3,
                    });
                }
            }
            CompletionContext::AfterSelect => {
                items.push(CompletionItem {
                    label: "*".into(),
                    kind: CompletionKind::Keyword,
                    detail: Some("All columns".into()),
                    insert_text: Some("* ".into()),
                    documentation: None,
                    sort_priority: 1,
                });
                items.push(CompletionItem {
                    label: "id".into(),
                    kind: CompletionKind::Field,
                    detail: Some("Vector ID".into()),
                    insert_text: Some("id".into()),
                    documentation: None,
                    sort_priority: 2,
                });
                items.push(CompletionItem {
                    label: "distance".into(),
                    kind: CompletionKind::Field,
                    detail: Some("Search distance".into()),
                    insert_text: Some("distance".into()),
                    documentation: None,
                    sort_priority: 2,
                });
            }
            CompletionContext::Start | CompletionContext::General => {
                // Suggest top-level keywords
                for kw in KEYWORDS {
                    items.push(CompletionItem {
                        label: kw.to_string(),
                        kind: CompletionKind::Keyword,
                        detail: Some("keyword".into()),
                        insert_text: Some(kw.to_string()),
                        documentation: None,
                        sort_priority: 5,
                    });
                }
                // Suggest functions
                for (name, insert, doc) in FUNCTIONS {
                    items.push(CompletionItem {
                        label: name.to_string(),
                        kind: CompletionKind::Function,
                        detail: Some(doc.to_string()),
                        insert_text: Some(insert.to_string()),
                        documentation: Some(doc.to_string()),
                        sort_priority: 3,
                    });
                }
                // Suggest snippets
                items.push(CompletionItem {
                    label: "SELECT...FROM...NEAREST_TO".into(),
                    kind: CompletionKind::Snippet,
                    detail: Some("Vector search query template".into()),
                    insert_text: Some(
                        "SELECT * FROM ${1:collection} NEAREST_TO([${2:vector}]) LIMIT ${3:10}"
                            .into(),
                    ),
                    documentation: Some("Search for similar vectors in a collection".into()),
                    sort_priority: 0,
                });
                items.push(CompletionItem {
                    label: "SELECT...WHERE...LIMIT".into(),
                    kind: CompletionKind::Snippet,
                    detail: Some("Filtered query template".into()),
                    insert_text: Some(
                        "SELECT * FROM ${1:collection} WHERE ${2:field} = '${3:value}' LIMIT ${4:10}"
                            .into(),
                    ),
                    documentation: Some("Search with metadata filter".into()),
                    sort_priority: 0,
                });
            }
        }

        items.sort_by_key(|i| i.sort_priority);
        items.truncate(self.config.max_completions);

        CompletionResponse {
            items,
            is_incomplete: false,
        }
    }

    // ── Diagnostics ─────────────────────────────────────────────────────

    /// Validate a NeedleQL document and return diagnostics.
    pub fn diagnose(&self, text: &str) -> DiagnosticResult {
        if !self.config.enable_diagnostics {
            return DiagnosticResult {
                diagnostics: vec![],
                is_valid: true,
            };
        }

        let mut diagnostics = Vec::new();

        // Check for basic syntax issues
        let upper = text.to_uppercase();
        let trimmed = text.trim();

        if trimmed.is_empty() {
            return DiagnosticResult {
                diagnostics: vec![],
                is_valid: true,
            };
        }

        // Check statement starts with a valid keyword
        let valid_starts = [
            "SELECT", "INSERT", "DELETE", "CREATE", "DROP", "SHOW", "EXPLAIN",
        ];
        if !valid_starts.iter().any(|s| upper.trim_start().starts_with(s)) {
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position { line: 0, character: 0 },
                    end: Position {
                        line: 0,
                        character: trimmed.len().min(20),
                    },
                },
                severity: DiagnosticSeverity::Error,
                message: "Statement must start with SELECT, INSERT, DELETE, CREATE, DROP, SHOW, or EXPLAIN".into(),
                source: "needleql".into(),
                code: Some("E001".into()),
            });
        }

        // Check for FROM clause in SELECT
        if upper.starts_with("SELECT") && !upper.contains("FROM") {
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position { line: 0, character: 0 },
                    end: Position {
                        line: 0,
                        character: trimmed.len(),
                    },
                },
                severity: DiagnosticSeverity::Error,
                message: "SELECT statement requires a FROM clause".into(),
                source: "needleql".into(),
                code: Some("E002".into()),
            });
        }

        // Check for unknown collection references
        if let Some(coll_name) = Self::extract_collection_name(text) {
            if !self.collections.contains_key(&coll_name) && !self.collections.is_empty() {
                let from_pos = upper.find("FROM ").unwrap_or(0);
                diagnostics.push(Diagnostic {
                    range: Range {
                        start: Position {
                            line: 0,
                            character: from_pos + 5,
                        },
                        end: Position {
                            line: 0,
                            character: from_pos + 5 + coll_name.len(),
                        },
                    },
                    severity: DiagnosticSeverity::Warning,
                    message: format!("Unknown collection '{coll_name}'"),
                    source: "needleql".into(),
                    code: Some("W001".into()),
                });
            }
        }

        // Check for NEAREST_TO without a vector
        if upper.contains("NEAREST_TO") && !text.contains('[') {
            let pos = upper.find("NEAREST_TO").unwrap_or(0);
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position {
                        line: 0,
                        character: pos,
                    },
                    end: Position {
                        line: 0,
                        character: pos + 10,
                    },
                },
                severity: DiagnosticSeverity::Error,
                message: "NEAREST_TO requires a vector argument: NEAREST_TO([...])".into(),
                source: "needleql".into(),
                code: Some("E003".into()),
            });
        }

        // Warn about missing LIMIT
        if (upper.starts_with("SELECT") || upper.starts_with("EXPLAIN SELECT"))
            && !upper.contains("LIMIT")
        {
            diagnostics.push(Diagnostic {
                range: Range {
                    start: Position {
                        line: 0,
                        character: trimmed.len().saturating_sub(1),
                    },
                    end: Position {
                        line: 0,
                        character: trimmed.len(),
                    },
                },
                severity: DiagnosticSeverity::Warning,
                message: "Consider adding a LIMIT clause to avoid returning too many results"
                    .into(),
                source: "needleql".into(),
                code: Some("W002".into()),
            });
        }

        let is_valid = !diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Error);

        DiagnosticResult {
            diagnostics,
            is_valid,
        }
    }

    // ── Hover ───────────────────────────────────────────────────────────

    /// Get hover information for a position.
    pub fn hover(&self, text: &str, position: usize) -> Option<HoverInfo> {
        if !self.config.enable_hover {
            return None;
        }

        let word = Self::word_at_position(text, position)?;
        let upper = word.to_uppercase();

        // Check if it's a keyword
        if KEYWORDS.contains(&upper.as_str()) {
            let doc = keyword_documentation(&upper);
            return Some(HoverInfo {
                contents: format!("**{upper}** (keyword)\n\n{doc}"),
                range: None,
            });
        }

        // Check if it's a function
        for (name, _, doc) in FUNCTIONS {
            if upper == *name {
                return Some(HoverInfo {
                    contents: format!("**{name}** (function)\n\n{doc}"),
                    range: None,
                });
            }
        }

        // Check if it's a collection
        if let Some(coll) = self.collections.get(&word) {
            return Some(HoverInfo {
                contents: format!(
                    "**{}** (collection)\n\nDimensions: {}\nFields: {}\nVectors: {}",
                    coll.name,
                    coll.dimensions,
                    coll.fields.join(", "),
                    coll.vector_count.unwrap_or(0)
                ),
                range: None,
            });
        }

        None
    }

    // ── Signature Help ──────────────────────────────────────────────────

    /// Get signature help for functions.
    pub fn signature_help(&self, text: &str, position: usize) -> Option<SignatureHelp> {
        if !self.config.enable_signature_help {
            return None;
        }

        let before = &text[..position.min(text.len())];
        let upper = before.to_uppercase();

        if upper.contains("NEAREST_TO(") {
            return Some(SignatureHelp {
                label: "NEAREST_TO(vector: [f32])".into(),
                documentation: Some(
                    "Search for vectors nearest to the given query vector using HNSW index."
                        .into(),
                ),
                parameters: vec![ParameterInfo {
                    label: "vector".into(),
                    documentation: Some("Query vector as array of floats, e.g., [0.1, 0.2, 0.3]".into()),
                }],
                active_parameter: 0,
            });
        }

        if upper.contains("HYBRID_SEARCH(") {
            let active = if before.matches(',').count() >= 1 {
                1
            } else {
                0
            };
            return Some(SignatureHelp {
                label: "HYBRID_SEARCH(text_query: string, vector?: [f32])".into(),
                documentation: Some(
                    "Combined BM25 text search and vector similarity search with RRF fusion."
                        .into(),
                ),
                parameters: vec![
                    ParameterInfo {
                        label: "text_query".into(),
                        documentation: Some("Text search query for BM25".into()),
                    },
                    ParameterInfo {
                        label: "vector".into(),
                        documentation: Some("Optional query vector for similarity search".into()),
                    },
                ],
                active_parameter: active,
            });
        }

        None
    }

    /// Get registered collection count.
    pub fn collection_count(&self) -> usize {
        self.collections.len()
    }

    /// Get config.
    pub fn config(&self) -> &LspConfig {
        &self.config
    }

    // ── Internals ───────────────────────────────────────────────────────

    fn analyze_context(before_cursor: &str) -> CompletionContext {
        let upper = before_cursor.trim_end().to_uppercase();
        if upper.is_empty() {
            return CompletionContext::Start;
        }
        if upper.ends_with("FROM ") || upper.ends_with("FROM") {
            return CompletionContext::AfterFrom;
        }
        if upper.ends_with("INTO ") || upper.ends_with("INTO") {
            return CompletionContext::AfterInto;
        }
        if upper.ends_with("COLLECTION ") {
            return CompletionContext::AfterCollection;
        }
        if upper.ends_with("WHERE ") || upper.ends_with("WHERE") {
            return CompletionContext::AfterWhere;
        }
        if upper.ends_with("AND ") || upper.ends_with("AND") {
            return CompletionContext::AfterAnd;
        }
        if upper.ends_with("OR ") || upper.ends_with("OR") {
            return CompletionContext::AfterOr;
        }
        if upper.ends_with("SELECT ") || upper.ends_with("SELECT") {
            return CompletionContext::AfterSelect;
        }
        CompletionContext::General
    }

    fn extract_collection_name(text: &str) -> Option<String> {
        let upper = text.to_uppercase();
        let from_pos = upper.find("FROM ")?;
        let after = &text[from_pos + 5..].trim_start();
        let end = after
            .find(|c: char| c.is_whitespace() || c == ';')
            .unwrap_or(after.len());
        let name = after[..end].to_string();
        if name.is_empty() {
            None
        } else {
            Some(name)
        }
    }

    fn word_at_position(text: &str, position: usize) -> Option<String> {
        if position > text.len() {
            return None;
        }
        let before = &text[..position];
        let after = &text[position..];
        let start = before.rfind(|c: char| !c.is_alphanumeric() && c != '_').map_or(0, |p| p + 1);
        let end = after.find(|c: char| !c.is_alphanumeric() && c != '_').unwrap_or(after.len());
        let word = format!("{}{}", &before[start..], &after[..end]);
        if word.is_empty() {
            None
        } else {
            Some(word)
        }
    }
}

/// Completion context based on cursor position.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionContext {
    Start,
    AfterSelect,
    AfterFrom,
    AfterInto,
    AfterCollection,
    AfterWhere,
    AfterAnd,
    AfterOr,
    General,
}

fn keyword_documentation(keyword: &str) -> &'static str {
    match keyword {
        "SELECT" => "Retrieve vectors and metadata from a collection.",
        "FROM" => "Specify the collection to query.",
        "WHERE" => "Filter results by metadata field conditions.",
        "LIMIT" => "Maximum number of results to return.",
        "OFFSET" => "Number of results to skip (pagination).",
        "ORDER" => "Sort results by a field.",
        "BY" => "Used with ORDER to specify the sort field.",
        "ASC" => "Sort in ascending order.",
        "DESC" => "Sort in descending order.",
        "AND" => "Logical AND — both conditions must be true.",
        "OR" => "Logical OR — either condition can be true.",
        "NOT" => "Logical NOT — negate a condition.",
        "IN" => "Check if a value is in a list.",
        "NEAREST_TO" => "Find vectors nearest to a query vector using HNSW index.",
        "HYBRID_SEARCH" => "Combined BM25 text + vector search with RRF fusion.",
        "EXPLAIN" => "Show the query execution plan without running the query.",
        "WITH" => "Specify query execution options (e.g., ef_search, timeout).",
        "INSERT" => "Insert a new vector with metadata into a collection.",
        "DELETE" => "Remove vectors from a collection.",
        "CREATE" => "Create a new collection.",
        "DROP" => "Delete a collection and all its data.",
        "SHOW" => "List available collections.",
        _ => "NeedleQL keyword.",
    }
}

impl Default for LspServer {
    fn default() -> Self {
        Self::new(LspConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_server() -> LspServer {
        let mut server = LspServer::new(LspConfig::default());
        server.register_collection("documents", 384, vec!["title".into(), "category".into()]);
        server.register_collection("images", 512, vec!["label".into(), "source".into()]);
        server
    }

    #[test]
    fn test_complete_after_from() {
        let server = make_server();
        let completions = server.complete(&CompletionRequest {
            text: "SELECT * FROM ".into(),
            cursor_pos: 14,
        });
        assert!(completions.items.iter().any(|c| c.label == "documents"));
        assert!(completions.items.iter().any(|c| c.label == "images"));
    }

    #[test]
    fn test_complete_after_where() {
        let server = make_server();
        let completions = server.complete(&CompletionRequest {
            text: "SELECT * FROM documents WHERE ".into(),
            cursor_pos: 29,
        });
        assert!(completions.items.iter().any(|c| c.label == "title"));
        assert!(completions.items.iter().any(|c| c.label == "category"));
    }

    #[test]
    fn test_complete_start() {
        let server = make_server();
        let completions = server.complete(&CompletionRequest {
            text: "".into(),
            cursor_pos: 0,
        });
        assert!(completions.items.iter().any(|c| c.label == "SELECT"));
        assert!(completions
            .items
            .iter()
            .any(|c| c.kind == CompletionKind::Snippet));
    }

    #[test]
    fn test_complete_after_select() {
        let server = make_server();
        let completions = server.complete(&CompletionRequest {
            text: "SELECT ".into(),
            cursor_pos: 7,
        });
        assert!(completions.items.iter().any(|c| c.label == "*"));
        assert!(completions.items.iter().any(|c| c.label == "id"));
        assert!(completions.items.iter().any(|c| c.label == "distance"));
    }

    #[test]
    fn test_diagnose_valid() {
        let server = make_server();
        let result = server.diagnose("SELECT * FROM documents LIMIT 10");
        assert!(result.is_valid);
    }

    #[test]
    fn test_diagnose_missing_from() {
        let server = make_server();
        let result = server.diagnose("SELECT *");
        assert!(!result.is_valid);
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == Some("E002".into())));
    }

    #[test]
    fn test_diagnose_invalid_start() {
        let server = make_server();
        let result = server.diagnose("INVALID STUFF");
        assert!(!result.is_valid);
    }

    #[test]
    fn test_diagnose_unknown_collection() {
        let server = make_server();
        let result = server.diagnose("SELECT * FROM unknown_coll LIMIT 10");
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.severity == DiagnosticSeverity::Warning));
    }

    #[test]
    fn test_diagnose_missing_limit_warning() {
        let server = make_server();
        let result = server.diagnose("SELECT * FROM documents");
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == Some("W002".into())));
    }

    #[test]
    fn test_diagnose_nearest_to_no_vector() {
        let server = make_server();
        let result = server.diagnose("SELECT * FROM documents NEAREST_TO() LIMIT 10");
        assert!(result
            .diagnostics
            .iter()
            .any(|d| d.code == Some("E003".into())));
    }

    #[test]
    fn test_hover_keyword() {
        let server = make_server();
        let hover = server.hover("SELECT * FROM docs", 0); // hover on SELECT
        assert!(hover.is_some());
        assert!(hover.unwrap().contents.contains("SELECT"));
    }

    #[test]
    fn test_hover_collection() {
        let server = make_server();
        let hover = server.hover("SELECT * FROM documents", 14); // hover on documents
        assert!(hover.is_some());
        assert!(hover.unwrap().contents.contains("384"));
    }

    #[test]
    fn test_hover_function() {
        let server = make_server();
        let hover = server.hover("SELECT * FROM docs NEAREST_TO([0.1])", 20);
        assert!(hover.is_some());
    }

    #[test]
    fn test_signature_help_nearest_to() {
        let server = make_server();
        let help = server.signature_help("SELECT * FROM docs NEAREST_TO(", 30);
        assert!(help.is_some());
        let sig = help.unwrap();
        assert!(sig.label.contains("NEAREST_TO"));
        assert_eq!(sig.parameters.len(), 1);
    }

    #[test]
    fn test_signature_help_hybrid() {
        let server = make_server();
        let help = server.signature_help("SELECT * FROM docs HYBRID_SEARCH(", 33);
        assert!(help.is_some());
        assert_eq!(help.unwrap().parameters.len(), 2);
    }

    #[test]
    fn test_document_lifecycle() {
        let mut server = make_server();
        server.open_document("file://test.nql", "SELECT * FROM docs");
        server.update_document("file://test.nql", "SELECT * FROM docs LIMIT 10");
        server.close_document("file://test.nql");
    }

    #[test]
    fn test_disabled_features() {
        let server = LspServer::new(LspConfig {
            enable_completion: false,
            enable_diagnostics: false,
            enable_hover: false,
            enable_signature_help: false,
            ..Default::default()
        });
        assert!(server
            .complete(&CompletionRequest {
                text: "SELECT ".into(),
                cursor_pos: 7,
            })
            .items
            .is_empty());
        assert!(server.diagnose("INVALID").diagnostics.is_empty());
        assert!(server.hover("SELECT", 0).is_none());
        assert!(server.signature_help("NEAREST_TO(", 11).is_none());
    }

    #[test]
    fn test_update_collection_metadata() {
        let mut server = make_server();
        server
            .update_collection_metadata("documents", 50_000)
            .unwrap();
        // Now completions should show updated count
        let completions = server.complete(&CompletionRequest {
            text: "SELECT * FROM ".into(),
            cursor_pos: 14,
        });
        let doc_item = completions
            .items
            .iter()
            .find(|c| c.label == "documents")
            .unwrap();
        assert!(doc_item.detail.as_ref().unwrap().contains("50000"));
    }

    #[test]
    fn test_diagnose_empty() {
        let server = make_server();
        let result = server.diagnose("");
        assert!(result.is_valid);
    }
}
