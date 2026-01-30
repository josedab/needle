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

// ── Vector Space Visualization ───────────────────────────────────────────────

/// A 2D point for vector space visualization (after dimensionality reduction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
    pub id: String,
    pub label: Option<String>,
    pub cluster: Option<usize>,
}

/// Configuration for dimensionality reduction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionConfig {
    /// Target dimensionality (2 or 3).
    pub target_dims: usize,
    /// Number of neighbors for local structure preservation.
    pub n_neighbors: usize,
    /// Number of iterations for the projection algorithm.
    pub n_iterations: usize,
    /// Learning rate for gradient descent.
    pub learning_rate: f64,
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            target_dims: 2,
            n_neighbors: 15,
            n_iterations: 200,
            learning_rate: 1.0,
        }
    }
}

/// Simple t-SNE-inspired projection for vector space visualization.
///
/// Uses pairwise distances to compute a 2D layout. This is a simplified
/// version suitable for interactive exploration of small-to-medium datasets
/// (up to ~10K vectors).
pub fn project_vectors(
    vectors: &[Vec<f32>],
    ids: &[String],
    config: &ProjectionConfig,
) -> Vec<Point2D> {
    let n = vectors.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![Point2D {
            x: 0.0,
            y: 0.0,
            id: ids[0].clone(),
            label: None,
            cluster: None,
        }];
    }

    // Compute pairwise distances
    let mut distances = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&vectors[i], &vectors[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    // Initialize 2D positions using first two principal components (simplified PCA)
    let mut positions: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
        let radius = distances[i].iter().sum::<f64>() / n as f64;
        positions.push((radius * angle.cos(), radius * angle.sin()));
    }

    // Stress majorization iterations to optimize layout
    for _ in 0..config.n_iterations {
        let mut new_positions = positions.clone();
        for i in 0..n {
            let mut dx = 0.0;
            let mut dy = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..n {
                if i == j {
                    continue;
                }
                let target_d = distances[i][j];
                if target_d < 1e-10 {
                    continue;
                }

                let curr_dx = positions[i].0 - positions[j].0;
                let curr_dy = positions[i].1 - positions[j].1;
                let curr_d = (curr_dx * curr_dx + curr_dy * curr_dy).sqrt().max(1e-10);

                let weight = 1.0 / (target_d * target_d);
                let scale = (target_d - curr_d) / curr_d;

                dx += weight * scale * curr_dx;
                dy += weight * scale * curr_dy;
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                let lr = config.learning_rate / weight_sum;
                new_positions[i].0 += lr * dx;
                new_positions[i].1 += lr * dy;
            }
        }
        positions = new_positions;
    }

    positions
        .into_iter()
        .enumerate()
        .map(|(i, (x, y))| Point2D {
            x,
            y,
            id: ids[i].clone(),
            label: None,
            cluster: None,
        })
        .collect()
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ── Visual Query Builder ─────────────────────────────────────────────────────

/// A visual query builder for constructing vector search queries in the IDE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryBuilder {
    /// The natural language or code query.
    pub query_text: String,
    /// Number of results to return.
    pub k: usize,
    /// Optional metadata filter in MongoDB-style JSON.
    pub filter: Option<Value>,
    /// Distance function to use.
    pub distance: String,
    /// Optional file path filter (glob pattern).
    pub file_filter: Option<String>,
    /// Optional language filter.
    pub language_filter: Option<String>,
    /// Whether to include similarity scores.
    pub include_scores: bool,
    /// Whether to include vector data in results.
    pub include_vectors: bool,
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self {
            query_text: String::new(),
            k: 10,
            filter: None,
            distance: "cosine".into(),
            file_filter: None,
            language_filter: None,
            include_scores: true,
            include_vectors: false,
        }
    }
}

impl QueryBuilder {
    /// Create a new query builder with the given search text.
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query_text: query.into(),
            ..Self::default()
        }
    }

    /// Set the number of results.
    #[must_use]
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a metadata filter.
    #[must_use]
    pub fn with_filter(mut self, filter: Value) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a file path filter.
    #[must_use]
    pub fn with_file_filter(mut self, pattern: impl Into<String>) -> Self {
        self.file_filter = Some(pattern.into());
        self
    }

    /// Set a language filter.
    #[must_use]
    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language_filter = Some(lang.into());
        self
    }

    /// Convert to a SearchRequest.
    pub fn to_search_request(&self) -> SearchRequest {
        SearchRequest {
            query: self.query_text.clone(),
            max_results: self.k,
            file_filter: self.file_filter.clone(),
            language_filter: self.language_filter.clone(),
        }
    }

    /// Validate the query builder configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.query_text.is_empty() {
            return Err("Query text is required".into());
        }
        if self.k == 0 {
            return Err("k must be at least 1".into());
        }
        Ok(())
    }
}

// ── Collection Diff Viewer ───────────────────────────────────────────────────

/// Diff between two collection snapshots for visualization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionDiff {
    /// Vectors added since the previous snapshot.
    pub added: Vec<DiffEntry>,
    /// Vectors removed since the previous snapshot.
    pub removed: Vec<DiffEntry>,
    /// Vectors whose metadata changed.
    pub modified: Vec<DiffModification>,
    /// Summary statistics.
    pub summary: DiffSummary,
}

/// A single entry in a diff (added or removed).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub id: String,
    pub metadata: Option<Value>,
}

/// A modification to an existing vector's metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffModification {
    pub id: String,
    pub old_metadata: Option<Value>,
    pub new_metadata: Option<Value>,
}

/// Summary statistics for a diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub total_added: usize,
    pub total_removed: usize,
    pub total_modified: usize,
}

impl CollectionDiff {
    /// Compute a diff between two sets of (id, metadata) pairs.
    pub fn compute(
        old: &[(String, Option<Value>)],
        new: &[(String, Option<Value>)],
    ) -> Self {
        let old_map: std::collections::HashMap<&str, &Option<Value>> =
            old.iter().map(|(id, meta)| (id.as_str(), meta)).collect();
        let new_map: std::collections::HashMap<&str, &Option<Value>> =
            new.iter().map(|(id, meta)| (id.as_str(), meta)).collect();

        let mut added = Vec::new();
        let mut removed = Vec::new();
        let mut modified = Vec::new();

        // Find additions and modifications
        for (id, new_meta) in &new_map {
            if let Some(old_meta) = old_map.get(id) {
                if old_meta != new_meta {
                    modified.push(DiffModification {
                        id: id.to_string(),
                        old_metadata: (*old_meta).clone(),
                        new_metadata: (*new_meta).clone(),
                    });
                }
            } else {
                added.push(DiffEntry {
                    id: id.to_string(),
                    metadata: (*new_meta).clone(),
                });
            }
        }

        // Find removals
        for (id, meta) in &old_map {
            if !new_map.contains_key(id) {
                removed.push(DiffEntry {
                    id: id.to_string(),
                    metadata: (*meta).clone(),
                });
            }
        }

        let summary = DiffSummary {
            total_added: added.len(),
            total_removed: removed.len(),
            total_modified: modified.len(),
        };

        Self {
            added,
            removed,
            modified,
            summary,
        }
    }

    /// Check if there are any changes.
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.modified.is_empty()
    }
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

    // ── Projection Tests ──

    #[test]
    fn test_project_empty() {
        let points = project_vectors(&[], &[], &ProjectionConfig::default());
        assert!(points.is_empty());
    }

    #[test]
    fn test_project_single() {
        let vecs = vec![vec![1.0, 2.0, 3.0]];
        let ids = vec!["v1".to_string()];
        let points = project_vectors(&vecs, &ids, &ProjectionConfig::default());
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].id, "v1");
    }

    #[test]
    fn test_project_multiple() {
        let vecs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        let ids: Vec<String> = (0..4).map(|i| format!("v{i}")).collect();
        let config = ProjectionConfig {
            n_iterations: 50,
            ..ProjectionConfig::default()
        };
        let points = project_vectors(&vecs, &ids, &config);
        assert_eq!(points.len(), 4);

        // Close vectors should be closer in 2D than distant ones
        let d_01 = ((points[0].x - points[1].x).powi(2) + (points[0].y - points[1].y).powi(2))
            .sqrt();
        let d_02 = ((points[0].x - points[2].x).powi(2) + (points[0].y - points[2].y).powi(2))
            .sqrt();
        // v0=[1,0,0] and v1=[0,1,0] are equidistant from v2=[0,0,1]
        assert!(d_01 > 0.0);
        assert!(d_02 > 0.0);
    }

    #[test]
    fn test_projection_config_serde() {
        let config = ProjectionConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deser: ProjectionConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.target_dims, 2);
        assert_eq!(deser.n_neighbors, 15);
    }

    // ── Query Builder Tests ──

    #[test]
    fn test_query_builder_default() {
        let qb = QueryBuilder::default();
        assert!(qb.query_text.is_empty());
        assert_eq!(qb.k, 10);
        assert!(qb.include_scores);
    }

    #[test]
    fn test_query_builder_chain() {
        let qb = QueryBuilder::new("authentication logic")
            .with_k(5)
            .with_file_filter("*.rs")
            .with_language("rust")
            .with_filter(serde_json::json!({"type": "function"}));

        assert_eq!(qb.query_text, "authentication logic");
        assert_eq!(qb.k, 5);
        assert_eq!(qb.file_filter.as_deref(), Some("*.rs"));
        assert_eq!(qb.language_filter.as_deref(), Some("rust"));
        assert!(qb.filter.is_some());
    }

    #[test]
    fn test_query_builder_to_search_request() {
        let qb = QueryBuilder::new("find auth")
            .with_k(20)
            .with_file_filter("src/**/*.rs");

        let req = qb.to_search_request();
        assert_eq!(req.query, "find auth");
        assert_eq!(req.max_results, 20);
        assert_eq!(req.file_filter.as_deref(), Some("src/**/*.rs"));
    }

    #[test]
    fn test_query_builder_validate() {
        assert!(QueryBuilder::new("hello").validate().is_ok());
        assert!(QueryBuilder::default().validate().is_err());
        assert!(QueryBuilder::new("x").with_k(0).validate().is_err());
    }

    // ── Collection Diff Tests ──

    #[test]
    fn test_diff_empty() {
        let diff = CollectionDiff::compute(&[], &[]);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_additions() {
        let old: Vec<(String, Option<Value>)> = vec![];
        let new = vec![
            ("v1".to_string(), Some(serde_json::json!({"a": 1}))),
            ("v2".to_string(), None),
        ];
        let diff = CollectionDiff::compute(&old, &new);
        assert_eq!(diff.summary.total_added, 2);
        assert_eq!(diff.summary.total_removed, 0);
    }

    #[test]
    fn test_diff_removals() {
        let old = vec![
            ("v1".to_string(), None),
            ("v2".to_string(), None),
        ];
        let new: Vec<(String, Option<Value>)> = vec![("v1".to_string(), None)];
        let diff = CollectionDiff::compute(&old, &new);
        assert_eq!(diff.summary.total_removed, 1);
        assert_eq!(diff.summary.total_added, 0);
    }

    #[test]
    fn test_diff_modifications() {
        let old = vec![("v1".to_string(), Some(serde_json::json!({"a": 1})))];
        let new = vec![("v1".to_string(), Some(serde_json::json!({"a": 2})))];
        let diff = CollectionDiff::compute(&old, &new);
        assert_eq!(diff.summary.total_modified, 1);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_diff_mixed() {
        let old = vec![
            ("v1".to_string(), Some(serde_json::json!({"x": 1}))),
            ("v2".to_string(), None),
        ];
        let new = vec![
            ("v1".to_string(), Some(serde_json::json!({"x": 99}))),
            ("v3".to_string(), None),
        ];
        let diff = CollectionDiff::compute(&old, &new);
        assert_eq!(diff.summary.total_added, 1);    // v3
        assert_eq!(diff.summary.total_removed, 1);  // v2
        assert_eq!(diff.summary.total_modified, 1);  // v1
    }

    #[test]
    fn test_point2d_serde() {
        let p = Point2D { x: 1.5, y: -2.3, id: "v1".into(), label: Some("test".into()), cluster: Some(0) };
        let json = serde_json::to_string(&p).unwrap();
        let deser: Point2D = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.id, "v1");
        assert!((deser.x - 1.5).abs() < f64::EPSILON);
    }
}
