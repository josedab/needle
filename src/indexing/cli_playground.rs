//! CLI Developer Playground
//!
//! Interactive REPL shell with sample datasets, HNSW visualization, and
//! parameter tuning for rapid prototyping and exploration.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::indexing::cli_playground::{Repl, ReplConfig, SampleDataset};
//!
//! let config = ReplConfig::default();
//! let mut repl = Repl::new(config);
//! repl.load_sample_dataset(SampleDataset::Sift10K)?;
//! let output = repl.execute("search [0.1, 0.2, ...] k=5")?;
//! ```

use crate::error::{NeedleError, Result};
use crate::Database;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Sample Datasets ─────────────────────────────────────────────────────────

/// Available sample datasets that ship with the playground.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SampleDataset {
    /// 10K SIFT descriptors (128-dim).
    Sift10K,
    /// 10K Wikipedia article embeddings (64-dim).
    Wikipedia10K,
    /// Movie recommendation embeddings (32-dim).
    MovieRecommendations,
}

impl SampleDataset {
    /// Get dataset metadata.
    pub fn metadata(&self) -> DatasetMetadata {
        match self {
            SampleDataset::Sift10K => DatasetMetadata {
                name: "SIFT-10K".into(),
                description: "10,000 SIFT visual descriptors for image similarity".into(),
                dimensions: 128,
                count: 10_000,
                distance: "euclidean".into(),
            },
            SampleDataset::Wikipedia10K => DatasetMetadata {
                name: "Wikipedia-10K".into(),
                description: "10,000 Wikipedia article embeddings for semantic search".into(),
                dimensions: 64,
                count: 10_000,
                distance: "cosine".into(),
            },
            SampleDataset::MovieRecommendations => DatasetMetadata {
                name: "Movies-5K".into(),
                description: "5,000 movie embeddings for recommendation".into(),
                dimensions: 32,
                count: 5_000,
                distance: "cosine".into(),
            },
        }
    }

    /// Generate synthetic vectors matching the dataset characteristics.
    pub fn generate(&self) -> Vec<(String, Vec<f32>)> {
        let meta = self.metadata();
        let mut vectors = Vec::with_capacity(meta.count);

        for i in 0..meta.count {
            let id = format!("{}_{}", meta.name.to_lowercase().replace('-', "_"), i);
            let vec: Vec<f32> = (0..meta.dimensions)
                .map(|j| {
                    let seed = (i * meta.dimensions + j) as f32;
                    (seed * 0.0137).sin() * 0.5 + 0.5
                })
                .collect();
            vectors.push((id, vec));
        }

        vectors
    }
}

/// Metadata about a sample dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Human-readable name.
    pub name: String,
    /// Description.
    pub description: String,
    /// Vector dimensions.
    pub dimensions: usize,
    /// Number of vectors.
    pub count: usize,
    /// Default distance function.
    pub distance: String,
}

// ── ASCII HNSW Visualization ────────────────────────────────────────────────

/// ASCII visualization of HNSW graph structure.
#[derive(Debug, Clone)]
pub struct HnswVisualization {
    /// Text lines of the visualization.
    pub lines: Vec<String>,
    /// Number of layers shown.
    pub layers: usize,
    /// Number of nodes shown.
    pub nodes: usize,
}

/// Generate an ASCII visualization of HNSW graph layers.
pub fn visualize_hnsw(
    num_vectors: usize,
    m: usize,
    ef_search: usize,
    max_nodes: usize,
    max_layers: usize,
) -> HnswVisualization {
    let nodes_to_show = max_nodes.min(num_vectors).min(30);

    let mut lines = Vec::new();
    lines.push(format!(
        "╔══════════════════════════════════════════════╗"
    ));
    lines.push(format!(
        "║  HNSW Graph Visualization ({} vectors)     ║",
        num_vectors
    ));
    lines.push(format!(
        "╚══════════════════════════════════════════════╝"
    ));
    lines.push(String::new());

    let effective_layers = max_layers.min(4);
    for layer in (0..effective_layers).rev() {
        let nodes_in_layer = if layer == 0 {
            nodes_to_show
        } else {
            (nodes_to_show as f64 / (2.0_f64.powi(layer as i32))) as usize
        }
        .max(1);

        lines.push(format!("  Layer {} ({} nodes):", layer, nodes_in_layer));

        let mut node_line = String::from("    ");
        for i in 0..nodes_in_layer.min(20) {
            node_line.push_str(&format!("[{:2}]", i));
            if i < nodes_in_layer.min(20) - 1 {
                node_line.push_str("──");
            }
        }
        if nodes_in_layer > 20 {
            node_line.push_str(" ...");
        }
        lines.push(node_line);

        if layer > 0 {
            let mut conn_line = String::from("    ");
            for _ in 0..nodes_in_layer.min(20) {
                conn_line.push_str("  │   ");
            }
            lines.push(conn_line);
            lines.push(format!("    {:─>width$}", "", width = nodes_in_layer.min(20) * 6));
        }
        lines.push(String::new());
    }

    lines.push("  Stats:".to_string());
    lines.push(format!("    Total vectors: {}", num_vectors));
    lines.push(format!("    Layers shown:  {}", effective_layers));
    lines.push(format!(
        "    Config: M={}, ef_search={}",
        m, ef_search
    ));

    HnswVisualization {
        lines,
        layers: effective_layers,
        nodes: nodes_to_show,
    }
}

// ── REPL Shell ──────────────────────────────────────────────────────────────

/// REPL configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplConfig {
    /// Default vector dimensions for new collections.
    pub default_dimensions: usize,
    /// Default distance function.
    pub default_distance: String,
    /// Command history size.
    pub history_size: usize,
    /// Enable colored output.
    pub color: bool,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            default_dimensions: 128,
            default_distance: "cosine".into(),
            history_size: 1000,
            color: true,
        }
    }
}

/// REPL command types.
#[derive(Debug, Clone)]
pub enum ReplCommand {
    /// Create a collection.
    CreateCollection { name: String, dimensions: usize },
    /// List collections.
    ListCollections,
    /// Insert a vector.
    Insert { collection: String, id: String, vector: Vec<f32> },
    /// Search a collection with a random query.
    Search { collection: String, k: usize },
    /// Show collection info.
    Info { collection: String },
    /// Load a sample dataset.
    LoadDataset { dataset: SampleDataset },
    /// Visualize HNSW graph.
    Visualize { collection: String },
    /// Show help.
    Help,
    /// Exit.
    Exit,
    /// Unknown command.
    Unknown(String),
}

/// Parse a REPL command from user input.
pub fn parse_command(input: &str) -> ReplCommand {
    let input = input.trim();
    let parts: Vec<&str> = input.splitn(3, ' ').collect();

    match parts.first().map(|s| s.to_lowercase()).as_deref() {
        Some("create") if parts.len() >= 3 => {
            let name = parts[1].to_string();
            let dims = parts[2].parse().unwrap_or(128);
            ReplCommand::CreateCollection {
                name,
                dimensions: dims,
            }
        }
        Some("collections") | Some("list") => ReplCommand::ListCollections,
        Some("info") if parts.len() >= 2 => ReplCommand::Info {
            collection: parts[1].to_string(),
        },
        Some("load") => {
            let dataset = match parts.get(1).map(|s| s.to_lowercase()).as_deref() {
                Some("sift") | Some("sift10k") => SampleDataset::Sift10K,
                Some("wiki") | Some("wikipedia") => SampleDataset::Wikipedia10K,
                Some("movies") | Some("movie") => SampleDataset::MovieRecommendations,
                _ => SampleDataset::Sift10K,
            };
            ReplCommand::LoadDataset { dataset }
        }
        Some("visualize") | Some("viz") if parts.len() >= 2 => ReplCommand::Visualize {
            collection: parts[1].to_string(),
        },
        Some("search") if parts.len() >= 2 => {
            let collection = parts[1].to_string();
            let k = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
            ReplCommand::Search { collection, k }
        },
        Some("help") | Some("?") => ReplCommand::Help,
        Some("exit") | Some("quit") | Some("q") => ReplCommand::Exit,
        _ => ReplCommand::Unknown(input.to_string()),
    }
}

/// Generate help text for the REPL.
pub fn help_text() -> String {
    let mut text = String::new();
    text.push_str("╔══════════════════════════════════════════════════╗\n");
    text.push_str("║           Needle Playground Commands             ║\n");
    text.push_str("╠══════════════════════════════════════════════════╣\n");
    text.push_str("║ create <name> <dims>  Create a collection       ║\n");
    text.push_str("║ collections           List all collections      ║\n");
    text.push_str("║ info <name>           Show collection details   ║\n");
    text.push_str("║ load <dataset>        Load sample dataset       ║\n");
    text.push_str("║   sift10k             - SIFT 10K (128-dim)     ║\n");
    text.push_str("║   wikipedia           - Wikipedia 10K (64-dim)  ║\n");
    text.push_str("║   movies              - Movies 5K (32-dim)     ║\n");
    text.push_str("║ visualize <name>      ASCII HNSW visualization  ║\n");
    text.push_str("║ help                  Show this help            ║\n");
    text.push_str("║ exit                  Exit playground           ║\n");
    text.push_str("╚══════════════════════════════════════════════════╝\n");
    text
}

/// Interactive REPL engine (non-blocking, command-by-command).
pub struct Repl {
    db: Database,
    config: ReplConfig,
    history: Vec<String>,
}

impl Repl {
    /// Create a new REPL with an in-memory database.
    pub fn new(config: ReplConfig) -> Self {
        Self {
            db: Database::in_memory(),
            config,
            history: Vec::new(),
        }
    }

    /// Create a REPL with an existing database.
    pub fn with_database(db: Database, config: ReplConfig) -> Self {
        Self {
            db,
            config,
            history: Vec::new(),
        }
    }

    /// Execute a command and return the output string.
    pub fn execute(&mut self, input: &str) -> Result<String> {
        self.history.push(input.to_string());
        if self.history.len() > self.config.history_size {
            self.history.remove(0);
        }

        let cmd = parse_command(input);
        match cmd {
            ReplCommand::CreateCollection { name, dimensions } => {
                self.db.create_collection(&name, dimensions)?;
                Ok(format!("✓ Created collection '{}' ({}-dim)", name, dimensions))
            }
            ReplCommand::ListCollections => {
                let collections = self.db.list_collections();
                if collections.is_empty() {
                    Ok("No collections. Use 'load sift10k' to get started.".into())
                } else {
                    let mut out = String::from("Collections:\n");
                    for name in collections {
                        if let Ok(coll) = self.db.collection(&name) {
                            out.push_str(&format!(
                                "  {} ({} vectors, {}-dim)\n",
                                name,
                                coll.len(),
                                coll.dimensions().unwrap_or(0)
                            ));
                        }
                    }
                    Ok(out)
                }
            }
            ReplCommand::Info { collection } => {
                let coll = self.db.collection(&collection)?;
                let stats = coll.stats()?;
                Ok(format!(
                    "Collection: {}\n  Vectors: {}\n  Dimensions: {}\n  Config: M={}, ef_search={}",
                    collection,
                    coll.len(),
                    coll.dimensions().unwrap_or(0),
                    stats.index_stats.m,
                    stats.index_stats.ef_search,
                ))
            }
            ReplCommand::LoadDataset { dataset } => {
                let meta = dataset.metadata();
                self.db.create_collection(&meta.name, meta.dimensions)?;
                let coll = self.db.collection(&meta.name)?;
                let vectors = dataset.generate();
                let count = vectors.len();
                for (id, vec) in vectors.into_iter().take(1000) {
                    // Load first 1000 for quick startup
                    coll.insert(&id, &vec, None)?;
                }
                Ok(format!(
                    "✓ Loaded {} vectors into '{}' ({}-dim, {})\n  {} — {}",
                    count.min(1000),
                    meta.name,
                    meta.dimensions,
                    meta.distance,
                    meta.name,
                    meta.description
                ))
            }
            ReplCommand::Visualize { collection } => {
                let coll = self.db.collection(&collection)?;
                let stats = coll.stats()?;
                let viz = visualize_hnsw(
                    coll.len(),
                    stats.index_stats.m,
                    stats.index_stats.ef_search,
                    20,
                    3,
                );
                Ok(viz.lines.join("\n"))
            }
            ReplCommand::Search { collection, k } => {
                let coll = self.db.collection(&collection)?;
                let dims = coll.dimensions().unwrap_or(self.config.default_dimensions);
                // Generate a random query vector
                let query: Vec<f32> = (0..dims)
                    .map(|i| ((i as f32) * 0.137).sin() * 0.5 + 0.5)
                    .collect();
                let start = std::time::Instant::now();
                let results = coll.search(&query, k)?;
                let elapsed = start.elapsed();

                let mut out = format!(
                    "Search results ({} results in {:.1}ms):\n",
                    results.len(),
                    elapsed.as_secs_f64() * 1000.0
                );
                for (i, r) in results.iter().enumerate() {
                    out.push_str(&format!(
                        "  {}. {} (distance: {:.4})\n",
                        i + 1, r.id, r.distance
                    ));
                }
                Ok(out)
            }
            ReplCommand::Help => Ok(help_text()),
            ReplCommand::Exit => Ok("Goodbye!".into()),
            ReplCommand::Unknown(cmd) => {
                Ok(format!("Unknown command: '{}'. Type 'help' for available commands.", cmd))
            }
            _ => Ok("Command not yet implemented in REPL mode.".into()),
        }
    }

    /// Get command history.
    pub fn history(&self) -> &[String] {
        &self.history
    }

    /// Get the database reference.
    pub fn database(&self) -> &Database {
        &self.db
    }

    /// Tab completion for a partial command.
    pub fn complete(&self, partial: &str) -> Vec<String> {
        let commands = vec![
            "create", "collections", "info", "load", "visualize", "help", "exit",
        ];
        let datasets = vec!["sift10k", "wikipedia", "movies"];

        let partial = partial.to_lowercase();
        let parts: Vec<&str> = partial.split_whitespace().collect();

        match parts.len() {
            0 | 1 => commands
                .iter()
                .filter(|c| c.starts_with(&partial))
                .map(|c| c.to_string())
                .collect(),
            2 if parts[0] == "load" => datasets
                .iter()
                .filter(|d| d.starts_with(parts[1]))
                .map(|d| format!("load {}", d))
                .collect(),
            2 if parts[0] == "info" || parts[0] == "visualize" => {
                self.db
                    .list_collections()
                    .into_iter()
                    .filter(|c| c.starts_with(parts[1]))
                    .map(|c| format!("{} {}", parts[0], c))
                    .collect()
            }
            _ => vec![],
        }
    }
}

// ── Parameter Tuning Wizard ─────────────────────────────────────────────────

/// Parameter tuning recommendation based on workload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningRecommendation {
    /// Recommended M parameter.
    pub m: usize,
    /// Recommended ef_construction.
    pub ef_construction: usize,
    /// Recommended ef_search.
    pub ef_search: usize,
    /// Estimated recall at this configuration.
    pub estimated_recall: f64,
    /// Estimated memory per vector (bytes).
    pub memory_per_vector: usize,
    /// Explanation.
    pub explanation: String,
}

/// Workload profile for parameter tuning.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum WorkloadProfile {
    /// Prioritize search speed.
    LatencySensitive,
    /// Prioritize recall accuracy.
    RecallSensitive,
    /// Balance between speed and recall.
    Balanced,
    /// Minimize memory usage.
    MemoryConstrained,
}

/// Generate parameter tuning recommendations based on workload profile.
pub fn tune_parameters(
    num_vectors: usize,
    dimensions: usize,
    profile: WorkloadProfile,
) -> TuningRecommendation {
    match profile {
        WorkloadProfile::LatencySensitive => TuningRecommendation {
            m: 12,
            ef_construction: 100,
            ef_search: 30,
            estimated_recall: 0.85,
            memory_per_vector: dimensions * 4 + 12 * 8,
            explanation: "Low M and ef_search for fast queries. Good for real-time applications.".into(),
        },
        WorkloadProfile::RecallSensitive => TuningRecommendation {
            m: 32,
            ef_construction: 400,
            ef_search: 200,
            estimated_recall: 0.99,
            memory_per_vector: dimensions * 4 + 32 * 8,
            explanation: "High M and ef for maximum recall. Best for precision-critical search.".into(),
        },
        WorkloadProfile::Balanced => {
            let m = if num_vectors > 1_000_000 { 24 } else { 16 };
            let ef_c = if num_vectors > 1_000_000 { 300 } else { 200 };
            let ef_s = if num_vectors > 1_000_000 { 100 } else { 50 };
            TuningRecommendation {
                m,
                ef_construction: ef_c,
                ef_search: ef_s,
                estimated_recall: 0.95,
                memory_per_vector: dimensions * 4 + m * 8,
                explanation: format!(
                    "Balanced config for {} vectors. Good recall with reasonable latency.",
                    num_vectors
                ),
            }
        }
        WorkloadProfile::MemoryConstrained => TuningRecommendation {
            m: 8,
            ef_construction: 100,
            ef_search: 30,
            estimated_recall: 0.80,
            memory_per_vector: dimensions * 4 + 8 * 8,
            explanation: "Minimal M to reduce memory. Consider quantization for further savings.".into(),
        },
    }
}

/// Format dataset metadata as a display string.
pub fn format_dataset_stats(meta: &DatasetMetadata) -> String {
    format!(
        "Dataset: {}\n  Description: {}\n  Vectors: {}\n  Dimensions: {}\n  Distance: {}",
        meta.name, meta.description, meta.count, meta.dimensions, meta.distance
    )
}

/// Generate synthetic recall vs latency scatter data for visualization.
/// Models the recall-latency tradeoff curve: higher ef_search → more recall, more latency.
pub fn generate_scatter_data(num_points: usize, base_recall: f64, base_latency: f64) -> Vec<(f64, f64)> {
    (0..num_points)
        .map(|i| {
            let t = i as f64 / num_points as f64;
            let recall = base_recall * (1.0 - (-3.0 * t).exp());
            let latency = base_latency * (1.0 + 5.0 * t.powi(2));
            (recall.clamp(0.0, 1.0), latency)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_dataset_metadata() {
        let meta = SampleDataset::Sift10K.metadata();
        assert_eq!(meta.dimensions, 128);
        assert_eq!(meta.count, 10_000);

        let meta = SampleDataset::Wikipedia10K.metadata();
        assert_eq!(meta.dimensions, 64);

        let meta = SampleDataset::MovieRecommendations.metadata();
        assert_eq!(meta.dimensions, 32);
    }

    #[test]
    fn test_sample_dataset_generate() {
        let vectors = SampleDataset::MovieRecommendations.generate();
        assert_eq!(vectors.len(), 5_000);
        assert_eq!(vectors[0].1.len(), 32);
    }

    #[test]
    fn test_parse_commands() {
        assert!(matches!(
            parse_command("create docs 384"),
            ReplCommand::CreateCollection { name, dimensions } if name == "docs" && dimensions == 384
        ));
        assert!(matches!(parse_command("collections"), ReplCommand::ListCollections));
        assert!(matches!(parse_command("list"), ReplCommand::ListCollections));
        assert!(matches!(
            parse_command("info mydb"),
            ReplCommand::Info { collection } if collection == "mydb"
        ));
        assert!(matches!(
            parse_command("load sift10k"),
            ReplCommand::LoadDataset { dataset: SampleDataset::Sift10K }
        ));
        assert!(matches!(parse_command("help"), ReplCommand::Help));
        assert!(matches!(parse_command("exit"), ReplCommand::Exit));
        assert!(matches!(parse_command("unknown"), ReplCommand::Unknown(_)));
        assert!(matches!(
            parse_command("search mydb 5"),
            ReplCommand::Search { collection, k } if collection == "mydb" && k == 5
        ));
    }

    #[test]
    fn test_repl_create_and_list() {
        let mut repl = Repl::new(ReplConfig::default());
        let output = repl.execute("create test 64").unwrap();
        assert!(output.contains("Created collection"));

        let output = repl.execute("collections").unwrap();
        assert!(output.contains("test"));
    }

    #[test]
    fn test_repl_load_dataset() {
        let mut repl = Repl::new(ReplConfig::default());
        let output = repl.execute("load movies").unwrap();
        assert!(output.contains("Loaded"));
        assert!(output.contains("Movies"));

        let output = repl.execute("collections").unwrap();
        assert!(output.contains("Movies-5K"));
    }

    #[test]
    fn test_repl_help() {
        let mut repl = Repl::new(ReplConfig::default());
        let output = repl.execute("help").unwrap();
        assert!(output.contains("create"));
        assert!(output.contains("load"));
    }

    #[test]
    fn test_repl_history() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.execute("help").unwrap();
        repl.execute("collections").unwrap();
        assert_eq!(repl.history().len(), 2);
    }

    #[test]
    fn test_repl_completion() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.execute("create test 64").unwrap();

        let completions = repl.complete("cr");
        assert!(completions.contains(&"create".to_string()));

        let completions = repl.complete("load s");
        assert!(completions.contains(&"load sift10k".to_string()));

        let completions = repl.complete("info t");
        assert!(completions.contains(&"info test".to_string()));
    }

    #[test]
    fn test_tune_parameters() {
        let rec = tune_parameters(100_000, 128, WorkloadProfile::Balanced);
        assert_eq!(rec.m, 16);
        assert!(rec.estimated_recall > 0.9);

        let rec = tune_parameters(100_000, 128, WorkloadProfile::LatencySensitive);
        assert!(rec.ef_search < 50);

        let rec = tune_parameters(100_000, 128, WorkloadProfile::RecallSensitive);
        assert!(rec.estimated_recall > 0.98);

        let rec = tune_parameters(100_000, 128, WorkloadProfile::MemoryConstrained);
        assert!(rec.m <= 8);
    }

    #[test]
    fn test_help_text() {
        let text = help_text();
        assert!(text.contains("Needle Playground"));
        assert!(text.contains("create"));
    }

    #[test]
    fn test_search_command() {
        let mut repl = Repl::new(ReplConfig::default());
        repl.execute("load movies").unwrap();

        // Search should work after loading data
        let output = repl.execute("search Movies-5K 5").unwrap();
        assert!(output.contains("result") || output.contains("Search"));
    }

    #[test]
    fn test_dataset_stats() {
        let meta = SampleDataset::Sift10K.metadata();
        let stats = format_dataset_stats(&meta);
        assert!(stats.contains("SIFT-10K"));
        assert!(stats.contains("128"));
        assert!(stats.contains("10000") || stats.contains("10,000"));
    }

    #[test]
    fn test_scatter_data_generation() {
        let data = generate_scatter_data(10, 0.9, 100.0);
        assert_eq!(data.len(), 10);
        for (recall, latency) in &data {
            assert!(*recall >= 0.0 && *recall <= 1.0);
            assert!(*latency > 0.0);
        }
    }

    #[test]
    fn test_visualize_hnsw_output() {
        let viz = visualize_hnsw(100, 16, 50, 15, 3);
        let output = viz.lines.join("\n");
        assert!(output.contains("HNSW Graph"));
        assert!(output.contains("Layer 0"));
        assert!(output.contains("100"));
    }

    #[test]
    fn test_unknown_command() {
        let mut repl = Repl::new(ReplConfig::default());
        let output = repl.execute("foobar baz").unwrap();
        assert!(output.contains("Unknown command"));
    }
}
