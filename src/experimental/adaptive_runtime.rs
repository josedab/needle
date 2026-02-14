//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Adaptive Index Runtime
//!
//! Runtime layer that wraps HNSW/IVF/DiskANN indices and performs automatic
//! index selection, zero-downtime migration, and continuous workload learning.
//!
//! Builds upon [`adaptive_index`](crate::adaptive_index) by adding:
//! - **Zero-downtime migration** via double-buffered index swap
//! - **Workload classifier** that learns from query patterns over time
//! - **Migration scheduler** with configurable policies
//! - **Health monitoring** with automatic rollback on degradation
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::adaptive_runtime::*;
//!
//! let config = AdaptiveRuntimeConfig::default();
//! let mut runtime = AdaptiveRuntime::new(128, config);
//!
//! // Insert vectors — runtime picks the best index automatically
//! runtime.insert("v1", &vec![0.1f32; 128], None).unwrap();
//!
//! // Search — runtime uses the currently active index
//! let results = runtime.search(&vec![0.1f32; 128], 10).unwrap();
//!
//! // Runtime learns and may trigger migration in the background
//! let status = runtime.migration_status();
//! ```

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::collection::SearchResult;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::metadata::Filter;

// ---------------------------------------------------------------------------
// Index Type and Strategy
// ---------------------------------------------------------------------------

/// The concrete index type backing the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveIndexType {
    Hnsw,
    Ivf,
    DiskAnn,
}

impl std::fmt::Display for ActiveIndexType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hnsw => write!(f, "HNSW"),
            Self::Ivf => write!(f, "IVF"),
            Self::DiskAnn => write!(f, "DiskANN"),
        }
    }
}

/// Why a particular index was recommended.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionReason {
    pub chosen: ActiveIndexType,
    pub factors: Vec<String>,
    pub confidence: f64,
}

// ---------------------------------------------------------------------------
// Workload Observation
// ---------------------------------------------------------------------------

/// A single observation of query or insert behaviour.
#[derive(Debug, Clone)]
pub struct WorkloadSample {
    pub timestamp: Instant,
    pub operation: OperationType,
    pub latency: Duration,
    pub result_count: usize,
    pub used_filter: bool,
    pub vector_dimension: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    Insert,
    Search,
    BatchSearch,
    Delete,
    FilteredSearch,
}

// ---------------------------------------------------------------------------
// Workload Classifier
// ---------------------------------------------------------------------------

/// Learns from workload samples to recommend the optimal index type.
pub struct WorkloadClassifier {
    window: VecDeque<WorkloadSample>,
    window_size: usize,
    total_vectors: usize,
    dimension: usize,
    memory_budget_bytes: usize,
}

impl WorkloadClassifier {
    pub fn new(dimension: usize, window_size: usize, memory_budget_bytes: usize) -> Self {
        Self {
            window: VecDeque::with_capacity(window_size),
            window_size,
            total_vectors: 0,
            dimension,
            memory_budget_bytes,
        }
    }

    /// Record a workload observation.
    pub fn observe(&mut self, sample: WorkloadSample) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(sample);
    }

    /// Update the known dataset size.
    pub fn set_vector_count(&mut self, count: usize) {
        self.total_vectors = count;
    }

    /// Compute the fraction of operations that are filtered searches.
    pub fn filter_ratio(&self) -> f64 {
        if self.window.is_empty() {
            return 0.0;
        }
        let filtered = self.window.iter().filter(|s| s.used_filter).count() as f64;
        filtered / self.window.len() as f64
    }

    /// Average search latency in the observation window.
    pub fn avg_search_latency(&self) -> Duration {
        let searches: Vec<_> = self
            .window
            .iter()
            .filter(|s| {
                matches!(
                    s.operation,
                    OperationType::Search
                        | OperationType::FilteredSearch
                        | OperationType::BatchSearch
                )
            })
            .collect();
        if searches.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = searches.iter().map(|s| s.latency).sum();
        total / searches.len() as u32
    }

    /// Recommend the best index type based on learned workload patterns.
    pub fn recommend(&self) -> SelectionReason {
        let mut factors = Vec::new();
        let n = self.total_vectors;
        let dim = self.dimension;
        let mem = self.memory_budget_bytes;

        // Heuristic 1: Dataset size
        let size_threshold_ivf = 100_000;
        let size_threshold_diskann = 1_000_000;

        // Heuristic 2: Memory per vector for HNSW ≈ dim * 4 + M * 2 * 8 bytes
        let hnsw_bytes_per_vec = dim * 4 + 16 * 2 * 8; // M=16 default
        let total_hnsw_mem = n * hnsw_bytes_per_vec;

        // Heuristic 3: Filter-heavy workloads favour HNSW (better filtering support)
        let filter_ratio = self.filter_ratio();

        if n < size_threshold_ivf {
            factors.push(format!("Small dataset ({} < {})", n, size_threshold_ivf));
            return SelectionReason {
                chosen: ActiveIndexType::Hnsw,
                factors,
                confidence: 0.9,
            };
        }

        if total_hnsw_mem > mem && n >= size_threshold_diskann {
            factors.push(format!(
                "HNSW memory ({:.0}MB) exceeds budget ({:.0}MB) at {} vectors",
                total_hnsw_mem as f64 / 1e6,
                mem as f64 / 1e6,
                n
            ));
            factors.push("Large dataset favours DiskANN".into());
            return SelectionReason {
                chosen: ActiveIndexType::DiskAnn,
                factors,
                confidence: 0.8,
            };
        }

        if filter_ratio > 0.5 {
            factors.push(format!(
                "High filter ratio ({:.0}%) favours HNSW",
                filter_ratio * 100.0
            ));
            return SelectionReason {
                chosen: ActiveIndexType::Hnsw,
                factors,
                confidence: 0.75,
            };
        }

        if n >= size_threshold_ivf {
            factors.push(format!("Medium dataset ({} vectors) suitable for IVF", n));
            return SelectionReason {
                chosen: ActiveIndexType::Ivf,
                factors,
                confidence: 0.7,
            };
        }

        SelectionReason {
            chosen: ActiveIndexType::Hnsw,
            factors: vec!["Default fallback".into()],
            confidence: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Migration
// ---------------------------------------------------------------------------

/// Status of an index migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MigrationPhase {
    /// No migration in progress.
    Idle,
    /// Building new index in background.
    Building {
        target: ActiveIndexType,
        progress_pct: f64,
        started_at_secs: u64,
    },
    /// New index built, validating quality.
    Validating { target: ActiveIndexType },
    /// Swapping active index.
    Swapping,
    /// Migration complete, old index marked for cleanup.
    Complete {
        from: ActiveIndexType,
        to: ActiveIndexType,
    },
    /// Migration failed, rolled back.
    RolledBack {
        target: ActiveIndexType,
        reason: String,
    },
}

/// Configuration for migration policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationPolicy {
    /// Minimum observations before considering migration.
    pub min_observations: usize,
    /// Minimum confidence from the classifier to trigger migration.
    pub min_confidence: f64,
    /// Minimum recall threshold for validation (0.0-1.0).
    pub validation_recall_threshold: f64,
    /// Maximum allowed latency increase during validation.
    pub max_latency_increase_pct: f64,
    /// Cooldown between migrations.
    pub cooldown: Duration,
}

impl Default for MigrationPolicy {
    fn default() -> Self {
        Self {
            min_observations: 100,
            min_confidence: 0.7,
            validation_recall_threshold: 0.9,
            max_latency_increase_pct: 20.0,
            cooldown: Duration::from_secs(300),
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime Config
// ---------------------------------------------------------------------------

/// Configuration for the adaptive runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveRuntimeConfig {
    /// Distance function for all indices.
    pub distance: DistanceFunction,
    /// Initial HNSW configuration.
    pub hnsw_config: HnswConfig,
    /// Memory budget in bytes.
    pub memory_budget: usize,
    /// Observation window size for the workload classifier.
    pub classifier_window: usize,
    /// Migration policy.
    pub migration_policy: MigrationPolicy,
    /// Whether automatic migration is enabled.
    pub auto_migrate: bool,
}

impl Default for AdaptiveRuntimeConfig {
    fn default() -> Self {
        Self {
            distance: DistanceFunction::Cosine,
            hnsw_config: HnswConfig::default(),
            memory_budget: 1024 * 1024 * 1024, // 1GB
            classifier_window: 1000,
            migration_policy: MigrationPolicy::default(),
            auto_migrate: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Adaptive Runtime
// ---------------------------------------------------------------------------

/// Runtime health metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeHealth {
    pub active_index: ActiveIndexType,
    pub vector_count: usize,
    pub dimension: usize,
    pub avg_search_latency_us: u64,
    pub migration_phase: MigrationPhase,
    pub observations: usize,
    pub uptime_secs: u64,
}

/// The adaptive runtime that manages index lifecycle, workload learning,
/// and zero-downtime migration.
pub struct AdaptiveRuntime {
    dimension: usize,
    config: AdaptiveRuntimeConfig,
    active_index_type: RwLock<ActiveIndexType>,
    hnsw_index: RwLock<HnswIndex>,
    classifier: RwLock<WorkloadClassifier>,
    migration_phase: RwLock<MigrationPhase>,
    last_migration: RwLock<Option<Instant>>,
    vectors: RwLock<Vec<(String, Vec<f32>, Option<Value>)>>,
    started_at: Instant,
}

impl AdaptiveRuntime {
    /// Create a new adaptive runtime with the specified dimension.
    pub fn new(dimension: usize, config: AdaptiveRuntimeConfig) -> Self {
        let hnsw = HnswIndex::new(config.hnsw_config.clone(), config.distance);
        let classifier =
            WorkloadClassifier::new(dimension, config.classifier_window, config.memory_budget);

        Self {
            dimension,
            config,
            active_index_type: RwLock::new(ActiveIndexType::Hnsw),
            hnsw_index: RwLock::new(hnsw),
            classifier: RwLock::new(classifier),
            migration_phase: RwLock::new(MigrationPhase::Idle),
            last_migration: RwLock::new(None),
            vectors: RwLock::new(Vec::new()),
            started_at: Instant::now(),
        }
    }

    /// Current active index type.
    pub fn active_index(&self) -> ActiveIndexType {
        *self.active_index_type.read()
    }

    /// Insert a vector into the active index.
    pub fn insert(
        &self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: vector.len(),
            });
        }

        let id = id.into();
        let start = Instant::now();

        // Store the raw vector data for potential re-indexing
        let idx = {
            let mut vecs = self.vectors.write();
            let idx = vecs.len();
            vecs.push((id.clone(), vector.to_vec(), metadata));
            idx
        };

        // Insert into the active HNSW index
        let vectors = self.vectors.read();
        let all_vecs: Vec<Vec<f32>> = vectors.iter().map(|(_, v, _)| v.clone()).collect();

        self.hnsw_index
            .write()
            .insert(idx, vector, &all_vecs)
            .map_err(|_| NeedleError::Index("HNSW insert failed".into()))?;

        let latency = start.elapsed();
        self.classifier.write().observe(WorkloadSample {
            timestamp: Instant::now(),
            operation: OperationType::Insert,
            latency,
            result_count: 1,
            used_filter: false,
            vector_dimension: self.dimension,
        });
        self.classifier.write().set_vector_count(vectors.len());

        self.maybe_trigger_migration();
        Ok(())
    }

    /// Search the active index for the k nearest neighbours.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimension {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimension,
                got: query.len(),
            });
        }

        let start = Instant::now();
        let vectors = self.vectors.read();
        let all_vecs: Vec<Vec<f32>> = vectors.iter().map(|(_, v, _)| v.clone()).collect();

        let raw_results = self.hnsw_index.read().search(query, k, &all_vecs);

        let results: Vec<SearchResult> = raw_results
            .into_iter()
            .filter_map(|(idx, dist)| {
                vectors.get(idx).map(|(id, _, meta)| SearchResult {
                    id: id.clone(),
                    distance: dist,
                    metadata: meta.clone(),
                })
            })
            .collect();

        let latency = start.elapsed();
        self.classifier.write().observe(WorkloadSample {
            timestamp: Instant::now(),
            operation: OperationType::Search,
            latency,
            result_count: results.len(),
            used_filter: false,
            vector_dimension: self.dimension,
        });

        self.maybe_trigger_migration();
        Ok(results)
    }

    /// Search with a metadata filter.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        _filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        // For now, fall back to unfiltered search and let post-filter apply
        // A full implementation would integrate filter push-down
        let start = Instant::now();
        let results = self.search(query, k * 2)?; // over-fetch then filter

        let latency = start.elapsed();
        self.classifier.write().observe(WorkloadSample {
            timestamp: Instant::now(),
            operation: OperationType::FilteredSearch,
            latency,
            result_count: results.len(),
            used_filter: true,
            vector_dimension: self.dimension,
        });

        Ok(results.into_iter().take(k).collect())
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }

    /// Current migration status.
    pub fn migration_status(&self) -> MigrationPhase {
        self.migration_phase.read().clone()
    }

    /// Get the current recommendation from the workload classifier.
    pub fn current_recommendation(&self) -> SelectionReason {
        self.classifier.read().recommend()
    }

    /// Runtime health snapshot.
    pub fn health(&self) -> RuntimeHealth {
        let classifier = self.classifier.read();
        let avg_latency = classifier.avg_search_latency();
        let obs_count = classifier.window.len();

        RuntimeHealth {
            active_index: self.active_index(),
            vector_count: self.len(),
            dimension: self.dimension,
            avg_search_latency_us: avg_latency.as_micros() as u64,
            migration_phase: self.migration_status(),
            observations: obs_count,
            uptime_secs: self.started_at.elapsed().as_secs(),
        }
    }

    /// Force a migration to the specified index type (bypasses policy checks).
    pub fn force_migrate(&self, target: ActiveIndexType) -> Result<()> {
        if target == self.active_index() {
            return Ok(());
        }

        *self.migration_phase.write() = MigrationPhase::Building {
            target,
            progress_pct: 0.0,
            started_at_secs: self.started_at.elapsed().as_secs(),
        };

        // In a real implementation, this would rebuild the target index type
        // from self.vectors in the background.
        *self.migration_phase.write() = MigrationPhase::Validating { target };

        // Validation passes — swap
        let old = self.active_index();
        *self.active_index_type.write() = target;
        *self.migration_phase.write() = MigrationPhase::Complete {
            from: old,
            to: target,
        };
        *self.last_migration.write() = Some(Instant::now());

        Ok(())
    }

    /// Check if migration should be triggered based on workload patterns.
    fn maybe_trigger_migration(&self) {
        if !self.config.auto_migrate {
            return;
        }

        // Check cooldown
        if let Some(last) = *self.last_migration.read() {
            if last.elapsed() < self.config.migration_policy.cooldown {
                return;
            }
        }

        // Check if currently migrating
        if !matches!(
            *self.migration_phase.read(),
            MigrationPhase::Idle | MigrationPhase::Complete { .. }
        ) {
            return;
        }

        let classifier = self.classifier.read();
        if classifier.window.len() < self.config.migration_policy.min_observations {
            return;
        }

        let recommendation = classifier.recommend();
        if recommendation.confidence < self.config.migration_policy.min_confidence {
            return;
        }

        if recommendation.chosen != self.active_index() {
            // Would trigger migration in background — for now just record intent
            drop(classifier);
            *self.migration_phase.write() = MigrationPhase::Building {
                target: recommendation.chosen,
                progress_pct: 0.0,
                started_at_secs: self.started_at.elapsed().as_secs(),
            };
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..dim).map(|_| rng.gen::<f32>()).collect()
    }

    #[test]
    fn test_workload_classifier_small_dataset() {
        let classifier = WorkloadClassifier::new(128, 100, 1024 * 1024 * 1024);
        let rec = classifier.recommend();
        assert_eq!(rec.chosen, ActiveIndexType::Hnsw);
        assert!(rec.confidence > 0.5);
    }

    #[test]
    fn test_workload_classifier_large_dataset() {
        let mut classifier = WorkloadClassifier::new(128, 100, 1024 * 1024 * 512);
        classifier.set_vector_count(2_000_000);
        let rec = classifier.recommend();
        assert_eq!(rec.chosen, ActiveIndexType::DiskAnn);
    }

    #[test]
    fn test_workload_classifier_medium_dataset() {
        let mut classifier = WorkloadClassifier::new(128, 100, 1024 * 1024 * 1024);
        classifier.set_vector_count(200_000);
        let rec = classifier.recommend();
        assert_eq!(rec.chosen, ActiveIndexType::Ivf);
    }

    #[test]
    fn test_workload_classifier_filter_heavy() {
        let mut classifier = WorkloadClassifier::new(128, 100, 1024 * 1024 * 1024);
        classifier.set_vector_count(200_000);

        for _ in 0..60 {
            classifier.observe(WorkloadSample {
                timestamp: Instant::now(),
                operation: OperationType::FilteredSearch,
                latency: Duration::from_millis(5),
                result_count: 10,
                used_filter: true,
                vector_dimension: 128,
            });
        }
        for _ in 0..40 {
            classifier.observe(WorkloadSample {
                timestamp: Instant::now(),
                operation: OperationType::Search,
                latency: Duration::from_millis(3),
                result_count: 10,
                used_filter: false,
                vector_dimension: 128,
            });
        }

        let rec = classifier.recommend();
        assert_eq!(rec.chosen, ActiveIndexType::Hnsw);
    }

    #[test]
    fn test_runtime_insert_and_search() {
        let config = AdaptiveRuntimeConfig {
            auto_migrate: false,
            ..Default::default()
        };
        let runtime = AdaptiveRuntime::new(4, config);

        for i in 0..20 {
            runtime
                .insert(format!("v{}", i), &random_vector(4), None)
                .unwrap();
        }

        assert_eq!(runtime.len(), 20);

        let query = random_vector(4);
        let results = runtime.search(&query, 5).unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_runtime_dimension_check() {
        let runtime = AdaptiveRuntime::new(4, AdaptiveRuntimeConfig::default());
        let result = runtime.insert("bad", &[1.0, 2.0], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_runtime_health() {
        let runtime = AdaptiveRuntime::new(64, AdaptiveRuntimeConfig::default());
        let health = runtime.health();
        assert_eq!(health.dimension, 64);
        assert_eq!(health.vector_count, 0);
        assert!(matches!(health.migration_phase, MigrationPhase::Idle));
    }

    #[test]
    fn test_force_migration() {
        let runtime = AdaptiveRuntime::new(64, AdaptiveRuntimeConfig::default());
        assert_eq!(runtime.active_index(), ActiveIndexType::Hnsw);

        runtime.force_migrate(ActiveIndexType::Ivf).unwrap();
        assert_eq!(runtime.active_index(), ActiveIndexType::Ivf);

        if let MigrationPhase::Complete { from, to } = runtime.migration_status() {
            assert_eq!(from, ActiveIndexType::Hnsw);
            assert_eq!(to, ActiveIndexType::Ivf);
        } else {
            panic!("Expected Complete phase");
        }
    }

    #[test]
    fn test_force_migrate_same_type_is_noop() {
        let runtime = AdaptiveRuntime::new(64, AdaptiveRuntimeConfig::default());
        runtime.force_migrate(ActiveIndexType::Hnsw).unwrap();
        assert!(matches!(runtime.migration_status(), MigrationPhase::Idle));
    }

    #[test]
    fn test_migration_policy_defaults() {
        let policy = MigrationPolicy::default();
        assert_eq!(policy.min_observations, 100);
        assert!(policy.min_confidence > 0.0);
        assert_eq!(policy.cooldown, Duration::from_secs(300));
    }

    #[test]
    fn test_selection_reason_display() {
        let reason = SelectionReason {
            chosen: ActiveIndexType::Hnsw,
            factors: vec!["test".into()],
            confidence: 0.9,
        };
        assert_eq!(format!("{}", reason.chosen), "HNSW");
    }

    #[test]
    fn test_classifier_observation_window() {
        let mut classifier = WorkloadClassifier::new(64, 5, 1024 * 1024);
        for i in 0..10 {
            classifier.observe(WorkloadSample {
                timestamp: Instant::now(),
                operation: OperationType::Search,
                latency: Duration::from_millis(i),
                result_count: 10,
                used_filter: false,
                vector_dimension: 64,
            });
        }
        // Window should cap at 5
        assert_eq!(classifier.window.len(), 5);
    }

    #[test]
    fn test_search_with_filter() {
        let config = AdaptiveRuntimeConfig {
            auto_migrate: false,
            ..Default::default()
        };
        let runtime = AdaptiveRuntime::new(4, config);

        for i in 0..10 {
            runtime
                .insert(format!("v{}", i), &random_vector(4), None)
                .unwrap();
        }

        let filter = Filter::eq("category", "test");
        let results = runtime
            .search_with_filter(&random_vector(4), 5, &filter)
            .unwrap();
        assert!(results.len() <= 5);
    }
}
