//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! WASM-Native Playground
//!
//! Browser-based interactive environment for exploring Needle vector database
//! with embedded sample datasets and guided tutorials.
//!
//! # Features
//!
//! - **Interactive Shell**: Monaco-based code editor with Needle API
//! - **Sample Datasets**: Pre-loaded Wikipedia, product catalog, code snippets
//! - **Visualizations**: 2D/3D vector space projections
//! - **Guided Tutorials**: Step-by-step walkthroughs
//! - **Shareable**: Export/import playground state
//!
//! # Architecture
//!
//! The playground runs entirely in the browser using WebAssembly:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                    Browser                               │
//! │  ┌─────────────────────────────────────────────────────┐ │
//! │  │              Playground UI (JS/HTML)                 │ │
//! │  │  ┌───────────┐ ┌───────────┐ ┌───────────────────┐  │ │
//! │  │  │Code Editor│ │Visualizer │ │Tutorial Panel     │  │ │
//! │  │  └─────┬─────┘ └─────┬─────┘ └─────────────────────┘  │ │
//! │  │        │             │                               │ │
//! │  │  ┌─────▼─────────────▼─────────────────────────────┐ │ │
//! │  │  │           WASM Needle Runtime                   │ │ │
//! │  │  │  ┌────────────┐ ┌────────────┐ ┌────────────┐  │ │ │
//! │  │  │  │ Collection │ │ Index      │ │ Datasets   │  │ │ │
//! │  │  │  └────────────┘ └────────────┘ └────────────┘  │ │ │
//! │  │  └─────────────────────────────────────────────────┘ │ │
//! │  └─────────────────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::playground::{Playground, PlaygroundConfig, Dataset};
//!
//! let config = PlaygroundConfig::default()
//!     .with_dataset(Dataset::WikipediaSubset)
//!     .with_tutorial(Tutorial::GettingStarted);
//!
//! let playground = Playground::new(config);
//! playground.execute("collection.search([0.1, 0.2, ...], 5)")?;
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// Available sample datasets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Dataset {
    /// Wikipedia article subset (1K articles, 384d embeddings)
    WikipediaSubset,
    /// E-commerce product catalog (5K products, 256d)
    ProductCatalog,
    /// Code snippets from GitHub (2K snippets, 768d)
    CodeSnippets,
    /// Movie descriptions (10K movies, 384d)
    MovieDescriptions,
    /// Scientific papers abstracts (3K papers, 768d)
    ScientificPapers,
    /// News articles (5K articles, 384d)
    NewsArticles,
    /// Empty dataset for user data
    Empty,
}

impl Dataset {
    /// Get dataset info
    pub fn info(&self) -> DatasetInfo {
        match self {
            Dataset::WikipediaSubset => DatasetInfo {
                name: "Wikipedia Subset".to_string(),
                description: "1,000 Wikipedia articles with embeddings".to_string(),
                count: 1000,
                dimensions: 384,
                size_bytes: 1_500_000,
                categories: vec!["Science".to_string(), "History".to_string(), "Technology".to_string()],
            },
            Dataset::ProductCatalog => DatasetInfo {
                name: "Product Catalog".to_string(),
                description: "5,000 e-commerce products".to_string(),
                count: 5000,
                dimensions: 256,
                size_bytes: 5_200_000,
                categories: vec!["Electronics".to_string(), "Clothing".to_string(), "Home".to_string()],
            },
            Dataset::CodeSnippets => DatasetInfo {
                name: "Code Snippets".to_string(),
                description: "2,000 code snippets from popular repos".to_string(),
                count: 2000,
                dimensions: 768,
                size_bytes: 6_200_000,
                categories: vec!["Python".to_string(), "JavaScript".to_string(), "Rust".to_string()],
            },
            Dataset::MovieDescriptions => DatasetInfo {
                name: "Movie Descriptions".to_string(),
                description: "10,000 movie plot summaries".to_string(),
                count: 10000,
                dimensions: 384,
                size_bytes: 15_600_000,
                categories: vec!["Action".to_string(), "Comedy".to_string(), "Drama".to_string()],
            },
            Dataset::ScientificPapers => DatasetInfo {
                name: "Scientific Papers".to_string(),
                description: "3,000 paper abstracts from arXiv".to_string(),
                count: 3000,
                dimensions: 768,
                size_bytes: 9_400_000,
                categories: vec!["CS".to_string(), "Physics".to_string(), "Math".to_string()],
            },
            Dataset::NewsArticles => DatasetInfo {
                name: "News Articles".to_string(),
                description: "5,000 news article summaries".to_string(),
                count: 5000,
                dimensions: 384,
                size_bytes: 7_800_000,
                categories: vec!["Politics".to_string(), "Business".to_string(), "Sports".to_string()],
            },
            Dataset::Empty => DatasetInfo {
                name: "Empty".to_string(),
                description: "Empty dataset for your own data".to_string(),
                count: 0,
                dimensions: 0,
                size_bytes: 0,
                categories: vec![],
            },
        }
    }

    /// Get all available datasets
    pub fn all() -> Vec<Dataset> {
        vec![
            Dataset::WikipediaSubset,
            Dataset::ProductCatalog,
            Dataset::CodeSnippets,
            Dataset::MovieDescriptions,
            Dataset::ScientificPapers,
            Dataset::NewsArticles,
            Dataset::Empty,
        ]
    }
}

/// Dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Description
    pub description: String,
    /// Number of vectors
    pub count: usize,
    /// Embedding dimensions
    pub dimensions: usize,
    /// Approximate size in bytes
    pub size_bytes: usize,
    /// Available categories
    pub categories: Vec<String>,
}

/// Available tutorials
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tutorial {
    /// Getting started with Needle
    GettingStarted,
    /// Building a semantic search
    SemanticSearch,
    /// Filtering and metadata
    MetadataFiltering,
    /// Hybrid search (BM25 + vector)
    HybridSearch,
    /// Building a RAG pipeline
    RagPipeline,
    /// Performance optimization
    PerformanceTuning,
    /// Multi-collection search
    CrossCollection,
}

impl Tutorial {
    /// Get tutorial info
    pub fn info(&self) -> TutorialInfo {
        match self {
            Tutorial::GettingStarted => TutorialInfo {
                title: "Getting Started".to_string(),
                description: "Learn the basics of Needle".to_string(),
                steps: 5,
                estimated_minutes: 10,
                difficulty: Difficulty::Beginner,
            },
            Tutorial::SemanticSearch => TutorialInfo {
                title: "Semantic Search".to_string(),
                description: "Build a semantic search engine".to_string(),
                steps: 8,
                estimated_minutes: 15,
                difficulty: Difficulty::Beginner,
            },
            Tutorial::MetadataFiltering => TutorialInfo {
                title: "Metadata Filtering".to_string(),
                description: "Filter results with metadata queries".to_string(),
                steps: 6,
                estimated_minutes: 12,
                difficulty: Difficulty::Intermediate,
            },
            Tutorial::HybridSearch => TutorialInfo {
                title: "Hybrid Search".to_string(),
                description: "Combine BM25 and vector search".to_string(),
                steps: 7,
                estimated_minutes: 15,
                difficulty: Difficulty::Intermediate,
            },
            Tutorial::RagPipeline => TutorialInfo {
                title: "RAG Pipeline".to_string(),
                description: "Build a retrieval-augmented generation system".to_string(),
                steps: 10,
                estimated_minutes: 25,
                difficulty: Difficulty::Advanced,
            },
            Tutorial::PerformanceTuning => TutorialInfo {
                title: "Performance Tuning".to_string(),
                description: "Optimize search performance".to_string(),
                steps: 8,
                estimated_minutes: 20,
                difficulty: Difficulty::Advanced,
            },
            Tutorial::CrossCollection => TutorialInfo {
                title: "Cross-Collection Search".to_string(),
                description: "Search across multiple collections".to_string(),
                steps: 6,
                estimated_minutes: 15,
                difficulty: Difficulty::Intermediate,
            },
        }
    }

    /// Get all tutorials
    pub fn all() -> Vec<Tutorial> {
        vec![
            Tutorial::GettingStarted,
            Tutorial::SemanticSearch,
            Tutorial::MetadataFiltering,
            Tutorial::HybridSearch,
            Tutorial::RagPipeline,
            Tutorial::PerformanceTuning,
            Tutorial::CrossCollection,
        ]
    }
}

/// Tutorial information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialInfo {
    /// Tutorial title
    pub title: String,
    /// Description
    pub description: String,
    /// Number of steps
    pub steps: usize,
    /// Estimated time in minutes
    pub estimated_minutes: usize,
    /// Difficulty level
    pub difficulty: Difficulty,
}

/// Difficulty level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Difficulty {
    Beginner,
    Intermediate,
    Advanced,
}

/// A tutorial step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TutorialStep {
    /// Step number
    pub number: usize,
    /// Step title
    pub title: String,
    /// Description/instructions
    pub description: String,
    /// Example code
    pub code: String,
    /// Expected output (for validation)
    pub expected_output: Option<String>,
    /// Hints
    pub hints: Vec<String>,
}

/// Playground configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundConfig {
    /// Initial dataset to load
    pub initial_dataset: Option<Dataset>,
    /// Active tutorial
    pub active_tutorial: Option<Tutorial>,
    /// Enable visualizations
    pub enable_visualizations: bool,
    /// Enable code completion
    pub enable_autocomplete: bool,
    /// Theme
    pub theme: Theme,
    /// Maximum vectors to visualize
    pub max_visualization_vectors: usize,
    /// Auto-save interval in seconds
    pub auto_save_interval_secs: u64,
}

impl Default for PlaygroundConfig {
    fn default() -> Self {
        Self {
            initial_dataset: Some(Dataset::WikipediaSubset),
            active_tutorial: Some(Tutorial::GettingStarted),
            enable_visualizations: true,
            enable_autocomplete: true,
            theme: Theme::Dark,
            max_visualization_vectors: 1000,
            auto_save_interval_secs: 60,
        }
    }
}

impl PlaygroundConfig {
    /// Set initial dataset
    pub fn with_dataset(mut self, dataset: Dataset) -> Self {
        self.initial_dataset = Some(dataset);
        self
    }

    /// Set active tutorial
    pub fn with_tutorial(mut self, tutorial: Tutorial) -> Self {
        self.active_tutorial = Some(tutorial);
        self
    }

    /// Set theme
    pub fn with_theme(mut self, theme: Theme) -> Self {
        self.theme = theme;
        self
    }

    /// Disable visualizations
    pub fn without_visualizations(mut self) -> Self {
        self.enable_visualizations = false;
        self
    }
}

/// UI theme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Theme {
    Light,
    Dark,
    HighContrast,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Whether execution succeeded
    pub success: bool,
    /// Output (JSON or text)
    pub output: Value,
    /// Error message if failed
    pub error: Option<String>,
    /// Execution time in microseconds
    pub execution_time_us: u64,
    /// Vectors affected/returned
    pub vector_count: usize,
}

/// Visualization data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationPoint {
    /// Vector ID
    pub id: String,
    /// 2D projected coordinates
    pub x: f32,
    pub y: f32,
    /// Optional 3D z-coordinate
    pub z: Option<f32>,
    /// Category/label
    pub label: Option<String>,
    /// Color (hex)
    pub color: Option<String>,
    /// Whether this is a query point
    pub is_query: bool,
    /// Whether this is a result
    pub is_result: bool,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    /// Projection method
    pub projection: ProjectionMethod,
    /// Show labels
    pub show_labels: bool,
    /// Color by category
    pub color_by_category: bool,
    /// Point size
    pub point_size: f32,
    /// Show connections for search results
    pub show_connections: bool,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            projection: ProjectionMethod::UMAP,
            show_labels: false,
            color_by_category: true,
            point_size: 3.0,
            show_connections: true,
        }
    }
}

/// Projection method for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionMethod {
    /// UMAP (default)
    UMAP,
    /// t-SNE
    TSNE,
    /// PCA
    PCA,
    /// Random projection
    Random,
}

/// Playground state that can be saved/shared
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaygroundState {
    /// State version
    pub version: String,
    /// Config
    pub config: PlaygroundConfig,
    /// Current code in editor
    pub editor_code: String,
    /// Execution history
    pub history: Vec<HistoryEntry>,
    /// Custom dataset (if any)
    pub custom_data: Option<CustomDataset>,
    /// Bookmarks
    pub bookmarks: Vec<Bookmark>,
    /// Tutorial progress
    pub tutorial_progress: HashMap<Tutorial, usize>,
}

impl Default for PlaygroundState {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            config: PlaygroundConfig::default(),
            editor_code: String::new(),
            history: Vec::new(),
            custom_data: None,
            bookmarks: Vec::new(),
            tutorial_progress: HashMap::new(),
        }
    }
}

/// History entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    /// Timestamp (Unix ms)
    pub timestamp: u64,
    /// Code executed
    pub code: String,
    /// Result summary
    pub result_summary: String,
    /// Success
    pub success: bool,
}

/// Custom dataset uploaded by user
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomDataset {
    /// Name
    pub name: String,
    /// Dimensions
    pub dimensions: usize,
    /// Vector count
    pub count: usize,
    /// Vectors (flattened)
    pub vectors: Vec<f32>,
    /// IDs
    pub ids: Vec<String>,
    /// Metadata
    pub metadata: Vec<Option<Value>>,
}

/// Bookmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bookmark {
    /// Name
    pub name: String,
    /// Code
    pub code: String,
    /// Description
    pub description: Option<String>,
    /// Created timestamp
    pub created_at: u64,
}

/// Playground runtime
pub struct Playground {
    #[allow(dead_code)]
    config: PlaygroundConfig,
    state: Arc<RwLock<PlaygroundState>>,
    tutorials: HashMap<Tutorial, Vec<TutorialStep>>,
}

impl Playground {
    /// Create a new playground
    pub fn new(config: PlaygroundConfig) -> Self {
        let mut playground = Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(PlaygroundState {
                config,
                ..Default::default()
            })),
            tutorials: HashMap::new(),
        };

        // Load tutorial content
        playground.load_tutorials();
        playground
    }

    /// Execute code in the playground
    pub fn execute(&self, code: &str) -> ExecutionResult {
        let start = std::time::Instant::now();

        // Parse and execute code
        // In WASM, this would call into the Needle WASM bindings
        let result = self.execute_internal(code);

        let elapsed = start.elapsed();

        // Record in history
        let entry = HistoryEntry {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            code: code.to_string(),
            result_summary: if result.success {
                format!("Success: {} vectors", result.vector_count)
            } else {
                result.error.clone().unwrap_or_default()
            },
            success: result.success,
        };

        self.state.write().history.push(entry);

        ExecutionResult {
            execution_time_us: elapsed.as_micros() as u64,
            ..result
        }
    }

    /// Get current tutorial step
    pub fn get_tutorial_step(&self, tutorial: Tutorial, step: usize) -> Option<TutorialStep> {
        self.tutorials
            .get(&tutorial)
            .and_then(|steps| steps.get(step).cloned())
    }

    /// Advance tutorial progress
    pub fn advance_tutorial(&self, tutorial: Tutorial) -> Option<TutorialStep> {
        let mut state = self.state.write();
        let current = state.tutorial_progress.entry(tutorial).or_insert(0);
        *current += 1;

        drop(state);
        self.get_tutorial_step(tutorial, *self.state.read().tutorial_progress.get(&tutorial).unwrap_or(&0))
    }

    /// Get tutorial progress
    pub fn get_tutorial_progress(&self, tutorial: Tutorial) -> (usize, usize) {
        let state = self.state.read();
        let current = *state.tutorial_progress.get(&tutorial).unwrap_or(&0);
        let total = self.tutorials.get(&tutorial).map(|s| s.len()).unwrap_or(0);
        (current, total)
    }

    /// Get visualization data
    pub fn get_visualization(&self, _config: &VisualizationConfig) -> Vec<VisualizationPoint> {
        // In real implementation, this would project vectors to 2D/3D
        // For now, return empty
        Vec::new()
    }

    /// Save state
    pub fn save_state(&self) -> PlaygroundState {
        self.state.read().clone()
    }

    /// Load state
    pub fn load_state(&self, state: PlaygroundState) {
        *self.state.write() = state;
    }

    /// Export state as JSON
    pub fn export_json(&self) -> String {
        serde_json::to_string_pretty(&*self.state.read()).unwrap_or_default()
    }

    /// Import state from JSON
    pub fn import_json(&self, json: &str) -> Result<(), String> {
        let state: PlaygroundState = serde_json::from_str(json)
            .map_err(|e| format!("Failed to parse state: {}", e))?;
        self.load_state(state);
        Ok(())
    }

    /// Add bookmark
    pub fn add_bookmark(&self, name: &str, code: &str, description: Option<String>) {
        let bookmark = Bookmark {
            name: name.to_string(),
            code: code.to_string(),
            description,
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };
        self.state.write().bookmarks.push(bookmark);
    }

    /// Get bookmarks
    pub fn get_bookmarks(&self) -> Vec<Bookmark> {
        self.state.read().bookmarks.clone()
    }

    /// Get available datasets
    pub fn available_datasets(&self) -> Vec<DatasetInfo> {
        Dataset::all().into_iter().map(|d| d.info()).collect()
    }

    /// Get available tutorials
    pub fn available_tutorials(&self) -> Vec<TutorialInfo> {
        Tutorial::all().into_iter().map(|t| t.info()).collect()
    }

    // === Private methods ===

    fn execute_internal(&self, code: &str) -> ExecutionResult {
        // Simplified execution - in real WASM, this calls Needle bindings
        
        // Check for basic commands
        if code.trim().starts_with("help") {
            return ExecutionResult {
                success: true,
                output: serde_json::json!({
                    "commands": [
                        "collection.insert(id, vector, metadata)",
                        "collection.search(query, k)",
                        "collection.search_with_filter(query, k, filter)",
                        "collection.delete(id)",
                        "collection.get(id)",
                        "collection.len()",
                    ]
                }),
                error: None,
                execution_time_us: 0,
                vector_count: 0,
            };
        }

        if code.trim().starts_with("info") {
            let state = self.state.read();
            return ExecutionResult {
                success: true,
                output: serde_json::json!({
                    "dataset": state.config.initial_dataset.map(|d| d.info()),
                    "tutorial": state.config.active_tutorial.map(|t| t.info()),
                }),
                error: None,
                execution_time_us: 0,
                vector_count: 0,
            };
        }

        // Default: return simulated success
        ExecutionResult {
            success: true,
            output: serde_json::json!({
                "message": "Execution simulated",
                "code": code,
            }),
            error: None,
            execution_time_us: 100,
            vector_count: 0,
        }
    }

    fn load_tutorials(&mut self) {
        // Load Getting Started tutorial
        self.tutorials.insert(Tutorial::GettingStarted, vec![
            TutorialStep {
                number: 1,
                title: "Welcome to Needle!".to_string(),
                description: "Needle is an embedded vector database. Let's start by exploring the pre-loaded dataset.".to_string(),
                code: "info".to_string(),
                expected_output: None,
                hints: vec!["Type 'info' to see current dataset info".to_string()],
            },
            TutorialStep {
                number: 2,
                title: "Your First Search".to_string(),
                description: "Let's perform a semantic search. We'll search for documents similar to a query.".to_string(),
                code: "collection.search([0.1, 0.2, ...], 5)".to_string(),
                expected_output: None,
                hints: vec!["The search returns the 5 most similar vectors".to_string()],
            },
            TutorialStep {
                number: 3,
                title: "Adding Vectors".to_string(),
                description: "Insert a new vector with metadata.".to_string(),
                code: r#"collection.insert("my_doc", [0.1, 0.2, ...], {"title": "Hello"})"#.to_string(),
                expected_output: None,
                hints: vec!["Metadata is optional but useful for filtering".to_string()],
            },
            TutorialStep {
                number: 4,
                title: "Filtering Results".to_string(),
                description: "Use metadata filters to narrow down search results.".to_string(),
                code: r#"collection.search_with_filter([0.1, ...], 5, {"category": "science"})"#.to_string(),
                expected_output: None,
                hints: vec!["Filters use MongoDB-style query syntax".to_string()],
            },
            TutorialStep {
                number: 5,
                title: "Congratulations!".to_string(),
                description: "You've learned the basics! Try the other tutorials to learn more.".to_string(),
                code: "help".to_string(),
                expected_output: None,
                hints: vec![],
            },
        ]);

        // Load Semantic Search tutorial
        self.tutorials.insert(Tutorial::SemanticSearch, vec![
            TutorialStep {
                number: 1,
                title: "What is Semantic Search?".to_string(),
                description: "Semantic search finds documents by meaning, not just keywords.".to_string(),
                code: "// Semantic search uses vector similarity".to_string(),
                expected_output: None,
                hints: vec![],
            },
            // ... more steps
        ]);
    }
}

impl Default for Playground {
    fn default() -> Self {
        Self::new(PlaygroundConfig::default())
    }
}

// ---------------------------------------------------------------------------
// Projection Engine: dimensionality reduction (PCA, random, simple UMAP/t-SNE)
// ---------------------------------------------------------------------------

/// Input vector for projection.
#[derive(Debug, Clone)]
pub struct ProjectionInput {
    pub id: String,
    pub vector: Vec<f32>,
    pub label: Option<String>,
    pub is_query: bool,
    pub is_result: bool,
}

/// Projected 2D/3D scene ready for rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionScene {
    pub points: Vec<VisualizationPoint>,
    pub clusters: Vec<ClusterInfo>,
    pub bounds: SceneBounds,
}

/// Axis-aligned bounding box of the projected scene.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneBounds {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
}

/// Cluster detected in the projected space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterInfo {
    pub id: usize,
    pub centroid_x: f32,
    pub centroid_y: f32,
    pub point_count: usize,
    pub color: String,
}

const CLUSTER_COLORS: &[&str] = &[
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
];

/// Projects high-dimensional vectors into 2D for visualization.
pub struct ProjectionEngine;

impl ProjectionEngine {
    /// Project vectors using the specified method.
    pub fn project(
        inputs: &[ProjectionInput],
        config: &VisualizationConfig,
    ) -> ProjectionScene {
        if inputs.is_empty() {
            return ProjectionScene {
                points: Vec::new(),
                clusters: Vec::new(),
                bounds: SceneBounds {
                    min_x: 0.0, max_x: 1.0, min_y: 0.0, max_y: 1.0,
                },
            };
        }

        let projected: Vec<(f32, f32)> = match config.projection {
            ProjectionMethod::PCA => Self::pca_project(inputs),
            ProjectionMethod::Random => Self::random_project(inputs),
            ProjectionMethod::UMAP | ProjectionMethod::TSNE => Self::simple_umap(inputs),
        };

        let clusters = Self::detect_clusters(&projected, 5);

        // Assign cluster colors
        let cluster_assignments = Self::assign_clusters(&projected, &clusters);

        let mut points = Vec::with_capacity(inputs.len());
        for (i, input) in inputs.iter().enumerate() {
            let (x, y) = projected[i];
            let color = if input.is_query {
                Some("#ff0000".to_string())
            } else if input.is_result {
                Some("#00ff00".to_string())
            } else if config.color_by_category {
                let c = cluster_assignments[i];
                Some(CLUSTER_COLORS[c % CLUSTER_COLORS.len()].to_string())
            } else {
                None
            };

            points.push(VisualizationPoint {
                id: input.id.clone(),
                x,
                y,
                z: None,
                label: input.label.clone(),
                color,
                is_query: input.is_query,
                is_result: input.is_result,
            });
        }

        let bounds = Self::compute_bounds(&projected);

        let cluster_infos: Vec<ClusterInfo> = clusters
            .iter()
            .enumerate()
            .map(|(i, (cx, cy, count))| ClusterInfo {
                id: i,
                centroid_x: *cx,
                centroid_y: *cy,
                point_count: *count,
                color: CLUSTER_COLORS[i % CLUSTER_COLORS.len()].to_string(),
            })
            .collect();

        ProjectionScene {
            points,
            clusters: cluster_infos,
            bounds,
        }
    }

    /// Simple PCA: project onto the two directions of maximum variance.
    fn pca_project(inputs: &[ProjectionInput]) -> Vec<(f32, f32)> {
        let n = inputs.len();
        let dim = inputs[0].vector.len();
        if dim < 2 || n == 0 {
            return vec![(0.0, 0.0); n];
        }

        // Compute mean
        let mut mean = vec![0.0f32; dim];
        for inp in inputs {
            for (j, v) in inp.vector.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f32;
        }

        // Center data
        let centered: Vec<Vec<f32>> = inputs
            .iter()
            .map(|inp| inp.vector.iter().zip(&mean).map(|(a, b)| a - b).collect())
            .collect();

        // Power iteration for top 2 principal components
        let pc1 = Self::power_iteration(&centered, dim, None);
        let pc2 = Self::power_iteration(&centered, dim, Some(&pc1));

        centered
            .iter()
            .map(|v| {
                let x: f32 = v.iter().zip(&pc1).map(|(a, b)| a * b).sum();
                let y: f32 = v.iter().zip(&pc2).map(|(a, b)| a * b).sum();
                (x, y)
            })
            .collect()
    }

    /// Power iteration to find a principal component direction.
    fn power_iteration(data: &[Vec<f32>], dim: usize, deflect: Option<&[f32]>) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim).map(|i| ((i + 1) as f32).sin()).collect();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in v.iter_mut() { *x /= norm; }

        for _ in 0..50 {
            // Multiply: Av = Σ (data_i · v) * data_i
            let mut new_v = vec![0.0f32; dim];
            for row in data {
                let dot: f32 = row.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (j, r) in row.iter().enumerate() {
                    new_v[j] += dot * r;
                }
            }

            // Deflect away from previous component
            if let Some(prev) = deflect {
                let dot: f32 = new_v.iter().zip(prev).map(|(a, b)| a * b).sum();
                for (j, p) in prev.iter().enumerate() {
                    new_v[j] -= dot * p;
                }
            }

            let norm = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                break;
            }
            for x in new_v.iter_mut() { *x /= norm; }
            v = new_v;
        }
        v
    }

    /// Random projection using a fixed seed for reproducibility.
    fn random_project(inputs: &[ProjectionInput]) -> Vec<(f32, f32)> {
        let dim = inputs[0].vector.len();
        // Deterministic pseudo-random projection vectors
        let r1: Vec<f32> = (0..dim).map(|i| ((i as f32 * 0.7 + 0.3).sin())).collect();
        let r2: Vec<f32> = (0..dim).map(|i| ((i as f32 * 1.3 + 0.7).cos())).collect();

        inputs
            .iter()
            .map(|inp| {
                let x: f32 = inp.vector.iter().zip(&r1).map(|(a, b)| a * b).sum();
                let y: f32 = inp.vector.iter().zip(&r2).map(|(a, b)| a * b).sum();
                (x, y)
            })
            .collect()
    }

    /// Simplified UMAP-like projection: uses PCA initialization then force-directed
    /// neighbor attraction/repulsion for a fixed number of iterations.
    fn simple_umap(inputs: &[ProjectionInput]) -> Vec<(f32, f32)> {
        let mut positions = Self::pca_project(inputs);
        let n = positions.len();
        if n < 3 {
            return positions;
        }

        // Build k-nearest neighbor graph (k=min(15,n-1))
        let k = 15.min(n - 1);
        let neighbors: Vec<Vec<usize>> = (0..n)
            .map(|i| {
                let mut dists: Vec<(usize, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let d: f32 = inputs[i]
                            .vector
                            .iter()
                            .zip(&inputs[j].vector)
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f32>();
                        (j, d)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(k).map(|(j, _)| *j).collect()
            })
            .collect();

        // Force-directed layout: 100 iterations
        let lr = 0.1f32;
        for iteration in 0..100 {
            let decay = 1.0 - (iteration as f32 / 100.0);
            let step = lr * decay;
            let mut dx = vec![0.0f32; n];
            let mut dy = vec![0.0f32; n];

            // Attractive forces from neighbors
            for i in 0..n {
                for &j in &neighbors[i] {
                    let fx = positions[j].0 - positions[i].0;
                    let fy = positions[j].1 - positions[i].1;
                    dx[i] += fx * 0.1;
                    dy[i] += fy * 0.1;
                }
            }

            // Repulsive forces (sample to avoid O(n²))
            let sample = 30.min(n);
            for i in 0..n {
                for s in 0..sample {
                    let j = (i * 7 + s * 13 + iteration) % n;
                    if j == i { continue; }
                    let fx = positions[i].0 - positions[j].0;
                    let fy = positions[i].1 - positions[j].1;
                    let dist_sq = fx * fx + fy * fy + 1e-4;
                    dx[i] += fx / dist_sq * 0.01;
                    dy[i] += fy / dist_sq * 0.01;
                }
            }

            for i in 0..n {
                positions[i].0 += dx[i] * step;
                positions[i].1 += dy[i] * step;
            }
        }

        positions
    }

    /// Simple grid-based cluster detection.
    fn detect_clusters(points: &[(f32, f32)], max_clusters: usize) -> Vec<(f32, f32, usize)> {
        if points.is_empty() {
            return Vec::new();
        }

        // K-means clustering
        let k = max_clusters.min(points.len());
        // Initialize centroids from first k points
        let mut centroids: Vec<(f32, f32)> = points.iter().take(k).copied().collect();
        let mut assignments = vec![0usize; points.len()];

        for _ in 0..20 {
            // Assign
            for (i, p) in points.iter().enumerate() {
                let mut best = 0;
                let mut best_d = f32::MAX;
                for (c, cent) in centroids.iter().enumerate() {
                    let d = (p.0 - cent.0).powi(2) + (p.1 - cent.1).powi(2);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                assignments[i] = best;
            }

            // Update centroids
            let mut sums = vec![(0.0f32, 0.0f32, 0usize); k];
            for (i, p) in points.iter().enumerate() {
                let c = assignments[i];
                sums[c].0 += p.0;
                sums[c].1 += p.1;
                sums[c].2 += 1;
            }
            for (c, s) in sums.iter().enumerate() {
                if s.2 > 0 {
                    centroids[c] = (s.0 / s.2 as f32, s.1 / s.2 as f32);
                }
            }
        }

        // Count final sizes
        let mut counts = vec![0usize; k];
        for &a in &assignments {
            counts[a] += 1;
        }

        centroids
            .into_iter()
            .zip(counts)
            .filter(|(_, count)| *count > 0)
            .map(|((x, y), count)| (x, y, count))
            .collect()
    }

    /// Assign each point to the nearest cluster.
    fn assign_clusters(points: &[(f32, f32)], clusters: &[(f32, f32, usize)]) -> Vec<usize> {
        if clusters.is_empty() {
            return vec![0; points.len()];
        }
        points
            .iter()
            .map(|(px, py)| {
                let mut best = 0;
                let mut best_d = f32::MAX;
                for (i, (cx, cy, _)) in clusters.iter().enumerate() {
                    let d = (px - cx).powi(2) + (py - cy).powi(2);
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                best
            })
            .collect()
    }

    fn compute_bounds(points: &[(f32, f32)]) -> SceneBounds {
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        for (x, y) in points {
            if *x < min_x { min_x = *x; }
            if *x > max_x { max_x = *x; }
            if *y < min_y { min_y = *y; }
            if *y > max_y { max_y = *y; }
        }
        SceneBounds { min_x, max_x, min_y, max_y }
    }
}

/// Generate HTML for playground embedding
pub fn generate_playground_html(config: &PlaygroundConfig) -> String {
    format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Needle Playground</title>
    <style>
        :root {{
            --bg-color: {};
            --text-color: {};
            --editor-bg: {};
        }}
        body {{
            margin: 0;
            font-family: system-ui, -apple-system, sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
        }}
        .playground {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto 1fr auto;
            height: 100vh;
            gap: 1px;
        }}
        .editor-panel {{
            grid-row: 2;
            background: var(--editor-bg);
            padding: 1rem;
        }}
        .output-panel {{
            grid-row: 2;
            background: var(--editor-bg);
            padding: 1rem;
            overflow: auto;
        }}
        .toolbar {{
            grid-column: 1 / -1;
            padding: 0.5rem 1rem;
            background: var(--editor-bg);
            display: flex;
            gap: 1rem;
            align-items: center;
        }}
        .status-bar {{
            grid-column: 1 / -1;
            padding: 0.25rem 1rem;
            background: var(--editor-bg);
            font-size: 0.875rem;
        }}
        .btn {{
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-weight: 500;
        }}
        .btn-primary {{
            background: #3b82f6;
            color: white;
        }}
    </style>
</head>
<body>
    <div class="playground">
        <div class="toolbar">
            <button class="btn btn-primary" onclick="runCode()">▶ Run</button>
            <select id="dataset-select">
                <option value="wikipedia">Wikipedia</option>
                <option value="products">Products</option>
                <option value="code">Code Snippets</option>
            </select>
            <select id="tutorial-select">
                <option value="getting-started">Getting Started</option>
                <option value="semantic-search">Semantic Search</option>
            </select>
        </div>
        <div class="editor-panel">
            <div id="editor"></div>
        </div>
        <div class="output-panel">
            <div id="output"></div>
        </div>
        <div class="status-bar">
            Ready | Dataset: Wikipedia (1K vectors)
        </div>
    </div>
    <script type="module">
        import init, {{ WasmCollection }} from './needle_wasm.js';
        
        async function main() {{
            await init();
            window.collection = new WasmCollection("playground", 384, "cosine");
            console.log("Needle playground ready!");
        }}
        
        window.runCode = function() {{
            const code = document.getElementById('editor').innerText;
            try {{
                const result = eval(code);
                document.getElementById('output').innerText = JSON.stringify(result, null, 2);
            }} catch (e) {{
                document.getElementById('output').innerText = 'Error: ' + e.message;
            }}
        }};
        
        main();
    </script>
</body>
</html>"#,
        match config.theme {
            Theme::Dark => "#1e1e1e",
            Theme::Light => "#ffffff",
            Theme::HighContrast => "#000000",
        },
        match config.theme {
            Theme::Dark => "#e0e0e0",
            Theme::Light => "#1e1e1e",
            Theme::HighContrast => "#ffffff",
        },
        match config.theme {
            Theme::Dark => "#252526",
            Theme::Light => "#f5f5f5",
            Theme::HighContrast => "#1e1e1e",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_info() {
        let info = Dataset::WikipediaSubset.info();
        assert_eq!(info.count, 1000);
        assert_eq!(info.dimensions, 384);
    }

    #[test]
    fn test_tutorial_info() {
        let info = Tutorial::GettingStarted.info();
        assert_eq!(info.steps, 5);
        assert_eq!(info.difficulty, Difficulty::Beginner);
    }

    #[test]
    fn test_playground_creation() {
        let playground = Playground::new(PlaygroundConfig::default());
        let datasets = playground.available_datasets();
        assert!(!datasets.is_empty());
    }

    #[test]
    fn test_execute_help() {
        let playground = Playground::default();
        let result = playground.execute("help");
        assert!(result.success);
    }

    #[test]
    fn test_execute_info() {
        let playground = Playground::default();
        let result = playground.execute("info");
        assert!(result.success);
    }

    #[test]
    fn test_tutorial_steps() {
        let playground = Playground::default();
        let step = playground.get_tutorial_step(Tutorial::GettingStarted, 0);
        assert!(step.is_some());
        assert_eq!(step.unwrap().number, 1);
    }

    #[test]
    fn test_bookmarks() {
        let playground = Playground::default();
        playground.add_bookmark("test", "collection.search()", Some("Test bookmark".to_string()));
        
        let bookmarks = playground.get_bookmarks();
        assert_eq!(bookmarks.len(), 1);
        assert_eq!(bookmarks[0].name, "test");
    }

    #[test]
    fn test_state_export_import() {
        let playground = Playground::default();
        playground.add_bookmark("export_test", "code", None);
        
        let json = playground.export_json();
        assert!(json.contains("export_test"));
        
        let new_playground = Playground::default();
        new_playground.import_json(&json).unwrap();
        
        let bookmarks = new_playground.get_bookmarks();
        assert_eq!(bookmarks.len(), 1);
    }

    #[test]
    fn test_config_builder() {
        let config = PlaygroundConfig::default()
            .with_dataset(Dataset::CodeSnippets)
            .with_tutorial(Tutorial::SemanticSearch)
            .with_theme(Theme::Light)
            .without_visualizations();
        
        assert_eq!(config.initial_dataset, Some(Dataset::CodeSnippets));
        assert_eq!(config.active_tutorial, Some(Tutorial::SemanticSearch));
        assert_eq!(config.theme, Theme::Light);
        assert!(!config.enable_visualizations);
    }

    #[test]
    fn test_generate_html() {
        let html = generate_playground_html(&PlaygroundConfig::default());
        assert!(html.contains("Needle Playground"));
        assert!(html.contains("WasmCollection"));
    }

    // ---- Projection Engine tests ----

    #[test]
    fn test_projection_pca() {
        let inputs: Vec<ProjectionInput> = (0..20)
            .map(|i| ProjectionInput {
                id: format!("v{}", i),
                vector: vec![(i as f32).sin(), (i as f32).cos(), i as f32 * 0.1],
                label: None,
                is_query: false,
                is_result: false,
            })
            .collect();

        let config = VisualizationConfig {
            projection: ProjectionMethod::PCA,
            ..Default::default()
        };
        let scene = ProjectionEngine::project(&inputs, &config);
        assert_eq!(scene.points.len(), 20);
        assert!(!scene.clusters.is_empty());
    }

    #[test]
    fn test_projection_random() {
        let inputs = vec![
            ProjectionInput {
                id: "a".into(),
                vector: vec![1.0, 0.0, 0.0],
                label: Some("cat".into()),
                is_query: true,
                is_result: false,
            },
            ProjectionInput {
                id: "b".into(),
                vector: vec![0.0, 1.0, 0.0],
                label: None,
                is_query: false,
                is_result: true,
            },
        ];

        let config = VisualizationConfig {
            projection: ProjectionMethod::Random,
            ..Default::default()
        };
        let scene = ProjectionEngine::project(&inputs, &config);
        assert_eq!(scene.points.len(), 2);
        assert_eq!(scene.points[0].color, Some("#ff0000".to_string())); // query
        assert_eq!(scene.points[1].color, Some("#00ff00".to_string())); // result
    }

    #[test]
    fn test_projection_umap() {
        let inputs: Vec<ProjectionInput> = (0..30)
            .map(|i| ProjectionInput {
                id: format!("v{}", i),
                vector: vec![
                    (i as f32 * 0.5).sin(),
                    (i as f32 * 0.3).cos(),
                    i as f32 / 30.0,
                    (i as f32).sqrt(),
                ],
                label: None,
                is_query: false,
                is_result: false,
            })
            .collect();

        let config = VisualizationConfig {
            projection: ProjectionMethod::UMAP,
            ..Default::default()
        };
        let scene = ProjectionEngine::project(&inputs, &config);
        assert_eq!(scene.points.len(), 30);
        // Bounds should be valid
        assert!(scene.bounds.min_x <= scene.bounds.max_x);
        assert!(scene.bounds.min_y <= scene.bounds.max_y);
    }

    #[test]
    fn test_projection_empty() {
        let scene = ProjectionEngine::project(&[], &VisualizationConfig::default());
        assert!(scene.points.is_empty());
        assert!(scene.clusters.is_empty());
    }

    #[test]
    fn test_cluster_info() {
        let inputs: Vec<ProjectionInput> = (0..50)
            .map(|i| {
                let cluster = i % 3;
                ProjectionInput {
                    id: format!("v{}", i),
                    vector: vec![
                        cluster as f32 * 10.0 + (i as f32 * 0.01),
                        0.0,
                        0.0,
                    ],
                    label: None,
                    is_query: false,
                    is_result: false,
                }
            })
            .collect();

        let config = VisualizationConfig {
            projection: ProjectionMethod::PCA,
            ..Default::default()
        };
        let scene = ProjectionEngine::project(&inputs, &config);
        // Should detect some clusters
        assert!(!scene.clusters.is_empty());
        let total: usize = scene.clusters.iter().map(|c| c.point_count).sum();
        assert_eq!(total, 50);
    }
}
