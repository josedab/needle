//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Adaptive Index Selection for Needle Vector Database
//!
//! Automatically selects and switches between HNSW, IVF, DiskANN, and exact
//! brute-force search based on data characteristics and query patterns.
//!
//! **Selection heuristics:**
//! - **BruteForce**: Best for tiny datasets (<1K vectors). Provides perfect
//!   recall with minimal overhead.
//! - **HNSW**: Best for datasets under ~1M vectors with sufficient RAM. Provides
//!   the highest recall and lowest latency for in-memory workloads.
//! - **IVF**: Suited for 1M–50M vectors or memory-constrained environments.
//!   Partitions the space into clusters for coarse-to-fine search.
//! - **DiskANN**: Recommended when the memory budget is very tight relative to
//!   dataset size, enabling disk-backed approximate search.
//!
//! The [`AdaptiveIndex`] struct wraps the underlying indexes, monitors a
//! [`WorkloadProfile`], and exposes [`analyze_workload`](AdaptiveIndex::analyze_workload)
//! to produce an [`IndexRecommendation`] that callers can act on.
//!
//! ## Explain API
//!
//! Use [`AdaptiveIndex::explain`] to understand *why* a particular strategy was
//! chosen without executing a search:
//!
//! ```rust,ignore
//! let explanation = index.explain(&query, 10);
//! println!("{}", explanation);
//! ```

use std::collections::VecDeque;
use std::fmt;
use std::time::Instant;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::collection::SearchResult;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::{HnswConfig, HnswIndex};
use crate::ivf::{IvfConfig, IvfIndex};

// ---------------------------------------------------------------------------
// IndexStrategy
// ---------------------------------------------------------------------------

/// The family of index algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexStrategy {
    /// Exact brute-force scan – perfect recall, best for tiny datasets.
    BruteForce,
    /// Hierarchical Navigable Small World graph – high recall, in-memory.
    Hnsw,
    /// Inverted File index – partition-based, scales to large datasets.
    Ivf,
    /// Disk-backed ANN index for memory-constrained scenarios.
    DiskAnn,
    /// Let the adaptive layer choose based on workload analysis.
    Auto,
}

impl Default for IndexStrategy {
    fn default() -> Self {
        Self::Auto
    }
}

impl fmt::Display for IndexStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BruteForce => write!(f, "BruteForce"),
            Self::Hnsw => write!(f, "HNSW"),
            Self::Ivf => write!(f, "IVF"),
            Self::DiskAnn => write!(f, "DiskANN"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

// ---------------------------------------------------------------------------
// AdaptiveIndexConfig
// ---------------------------------------------------------------------------

/// Thresholds that govern when the adaptive layer recommends switching indexes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveIndexConfig {
    /// Maximum number of vectors for which brute-force is acceptable.
    pub brute_force_max_vectors: usize,
    /// Maximum number of vectors for which HNSW is preferred.
    pub hnsw_max_vectors: usize,
    /// Minimum number of vectors before IVF becomes worthwhile.
    pub ivf_min_vectors: usize,
    /// Maximum number of vectors for IVF before DiskANN is recommended.
    pub ivf_max_vectors: usize,
    /// Available memory budget in bytes (0 = unlimited).
    pub memory_budget_bytes: usize,
    /// Target recall in [0.0, 1.0]. Higher values bias toward HNSW.
    pub target_recall: f32,
    /// Maximum acceptable query latency in microseconds.
    pub max_latency_us: u64,
    /// Bytes-per-vector estimate used for memory projections.
    pub bytes_per_vector_estimate: usize,
    /// Threshold for filter-heavy workloads (0.0–1.0).
    pub high_filter_pct_threshold: f32,
    /// Number of recent queries to keep in the latency window.
    pub latency_window_size: usize,
    /// If true, automatically migrate when a recommendation changes.
    pub auto_migrate: bool,
}

impl Default for AdaptiveIndexConfig {
    fn default() -> Self {
        Self {
            brute_force_max_vectors: 1_000,
            hnsw_max_vectors: 1_000_000,
            ivf_min_vectors: 100_000,
            ivf_max_vectors: 50_000_000,
            memory_budget_bytes: 0, // unlimited
            target_recall: 0.95,
            max_latency_us: 10_000,
            bytes_per_vector_estimate: 1536, // 384-dim × 4 bytes
            high_filter_pct_threshold: 0.5,
            latency_window_size: 100,
            auto_migrate: false,
        }
    }
}

// ---------------------------------------------------------------------------
// WorkloadProfile
// ---------------------------------------------------------------------------

/// Runtime statistics that the adaptive layer uses to decide on index strategy.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkloadProfile {
    pub query_count: u64,
    pub avg_latency_us: f64,
    pub avg_recall_estimate: f64,
    pub vector_count: usize,
    pub dimensions: usize,
    pub memory_usage_bytes: usize,
    pub queries_with_filters_pct: f64,

    // Internal accumulators (not part of the public averages).
    #[serde(skip)]
    total_latency_us: u64,
    #[serde(skip)]
    filter_query_count: u64,
    #[serde(skip)]
    recent_latencies_us: VecDeque<u64>,
}

impl WorkloadProfile {
    fn record_query(&mut self, latency_us: u64, used_filter: bool, window_size: usize) {
        self.query_count += 1;
        self.total_latency_us += latency_us;
        self.avg_latency_us = self.total_latency_us as f64 / self.query_count as f64;

        self.recent_latencies_us.push_back(latency_us);
        while self.recent_latencies_us.len() > window_size {
            self.recent_latencies_us.pop_front();
        }

        if used_filter {
            self.filter_query_count += 1;
        }
        self.queries_with_filters_pct = self.filter_query_count as f64 / self.query_count as f64;
    }

    fn update_vector_stats(&mut self, count: usize, dims: usize) {
        self.vector_count = count;
        self.dimensions = dims;
        self.memory_usage_bytes = count * dims * std::mem::size_of::<f32>();
    }

    /// p50 latency over the recent window.
    pub fn p50_latency_us(&self) -> u64 {
        percentile_latency(&self.recent_latencies_us, 50)
    }

    /// p99 latency over the recent window.
    pub fn p99_latency_us(&self) -> u64 {
        percentile_latency(&self.recent_latencies_us, 99)
    }
}

fn percentile_latency(window: &VecDeque<u64>, pct: u8) -> u64 {
    if window.is_empty() {
        return 0;
    }
    let mut sorted: Vec<u64> = window.iter().copied().collect();
    sorted.sort_unstable();
    let idx = ((pct as usize) * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

// ---------------------------------------------------------------------------
// IndexRecommendation
// ---------------------------------------------------------------------------

/// The result of workload analysis — describes which strategy should be active.
#[derive(Debug, Clone)]
pub struct IndexRecommendation {
    pub recommended: IndexStrategy,
    pub reason: String,
    pub estimated_recall: f32,
    pub estimated_latency_us: u64,
}

// ---------------------------------------------------------------------------
// SearchExplanation
// ---------------------------------------------------------------------------

/// Human-readable explanation of why the adaptive engine chose a particular
/// index strategy for a given query, without actually executing the search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchExplanation {
    /// The strategy that would be used.
    pub chosen_strategy: IndexStrategy,
    /// Ordered list of decision factors that led to the choice.
    pub factors: Vec<String>,
    /// Estimated cost in microseconds.
    pub estimated_cost_us: u64,
    /// Whether a migration is recommended.
    pub migration_recommended: bool,
    /// If migration is recommended, what to migrate to.
    pub migration_target: Option<IndexStrategy>,
}

impl fmt::Display for SearchExplanation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Strategy: {}", self.chosen_strategy)?;
        writeln!(f, "Estimated cost: {} µs", self.estimated_cost_us)?;
        writeln!(f, "Decision factors:")?;
        for (i, factor) in self.factors.iter().enumerate() {
            writeln!(f, "  {}. {}", i + 1, factor)?;
        }
        if self.migration_recommended {
            if let Some(target) = &self.migration_target {
                writeln!(f, "⚠ Migration recommended → {}", target)?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MigrationEvent
// ---------------------------------------------------------------------------

/// Record of a completed index migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationEvent {
    pub from: IndexStrategy,
    pub to: IndexStrategy,
    pub duration_us: u64,
    pub vector_count: usize,
    pub reason: String,
}

// ---------------------------------------------------------------------------
// AdaptiveIndex
// ---------------------------------------------------------------------------

/// An index wrapper that adaptively selects and delegates to the best
/// underlying index based on data characteristics and query patterns.
pub struct AdaptiveIndex {
    hnsw: Option<HnswIndex>,
    ivf: Option<IvfIndex>,
    active_strategy: IndexStrategy,
    config: AdaptiveIndexConfig,
    profile: RwLock<WorkloadProfile>,
    distance: DistanceFunction,
    dimensions: usize,
    vectors: Vec<Vec<f32>>,
    id_map: Vec<String>,
    migration_history: Vec<MigrationEvent>,
}

impl AdaptiveIndex {
    /// Create a new adaptive index.
    ///
    /// If `strategy` is [`IndexStrategy::Auto`] the index starts with
    /// brute-force (for small datasets) or HNSW and may recommend migration
    /// later.
    pub fn new(
        dimensions: usize,
        distance: DistanceFunction,
        strategy: IndexStrategy,
        config: AdaptiveIndexConfig,
    ) -> Self {
        let effective = match strategy {
            IndexStrategy::Auto | IndexStrategy::Hnsw => IndexStrategy::Hnsw,
            IndexStrategy::BruteForce => IndexStrategy::BruteForce,
            other => other,
        };

        let hnsw = if effective == IndexStrategy::Hnsw {
            Some(HnswIndex::new(HnswConfig::default(), distance))
        } else {
            None
        };

        let ivf = if effective == IndexStrategy::Ivf {
            Some(IvfIndex::new(dimensions, IvfConfig::new(256)))
        } else {
            None
        };

        Self {
            hnsw,
            ivf,
            active_strategy: effective,
            config,
            profile: RwLock::new(WorkloadProfile::default()),
            distance,
            dimensions,
            vectors: Vec::new(),
            id_map: Vec::new(),
            migration_history: Vec::new(),
        }
    }

    // -- Mutation ---------------------------------------------------------

    /// Insert a vector with an associated string identifier.
    pub fn insert(&mut self, id: impl Into<String>, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: vector.len(),
            });
        }

        let internal_id = self.vectors.len();
        self.vectors.push(vector.to_vec());
        self.id_map.push(id.into());

        // Delegate to the active index.
        if let Some(ref mut hnsw) = self.hnsw {
            hnsw.insert(internal_id, vector, &self.vectors)?;
        }
        // IVF requires trained centroids; skip insertion if untrained.
        // BruteForce needs no index structure.

        // Update profile.
        {
            let mut profile = self.profile.write();
            profile.update_vector_stats(self.vectors.len(), self.dimensions);
        }

        // Auto-migrate if enabled and the recommendation has changed.
        if self.config.auto_migrate {
            let rec = self.analyze_workload();
            if rec.recommended != self.active_strategy && rec.recommended != IndexStrategy::DiskAnn
            {
                let _ = self.migrate_index(rec.recommended);
            }
        }

        Ok(())
    }

    // -- Search -----------------------------------------------------------

    /// Search for the `k` nearest neighbors of `query`.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let start = Instant::now();

        let results = match self.active_strategy {
            IndexStrategy::BruteForce => self.brute_force_search(query, k),
            IndexStrategy::Hnsw | IndexStrategy::Auto => {
                if let Some(ref hnsw) = self.hnsw {
                    let raw = hnsw.search(query, k, &self.vectors);
                    self.to_search_results(&raw)
                } else {
                    self.brute_force_search(query, k)
                }
            }
            IndexStrategy::Ivf => {
                if let Some(ref ivf) = self.ivf {
                    let raw = ivf.search(query, k).unwrap_or_default();
                    self.to_search_results(&raw)
                } else {
                    self.brute_force_search(query, k)
                }
            }
            IndexStrategy::DiskAnn => {
                if let Some(ref hnsw) = self.hnsw {
                    let raw = hnsw.search(query, k, &self.vectors);
                    self.to_search_results(&raw)
                } else {
                    self.brute_force_search(query, k)
                }
            }
        };

        let elapsed_us = start.elapsed().as_micros() as u64;
        {
            let mut profile = self.profile.write();
            profile.record_query(elapsed_us, false, self.config.latency_window_size);
        }

        Ok(results)
    }

    /// Search with a pre-filter function applied to candidate IDs.
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: impl Fn(&str) -> bool,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.dimensions,
                got: query.len(),
            });
        }

        let start = Instant::now();

        let over_k = k * 4;
        let candidates = match self.active_strategy {
            IndexStrategy::BruteForce => self.brute_force_search(query, over_k),
            IndexStrategy::Hnsw | IndexStrategy::Auto | IndexStrategy::DiskAnn => {
                if let Some(ref hnsw) = self.hnsw {
                    let raw = hnsw.search(query, over_k, &self.vectors);
                    self.to_search_results(&raw)
                } else {
                    self.brute_force_search(query, over_k)
                }
            }
            IndexStrategy::Ivf => {
                if let Some(ref ivf) = self.ivf {
                    let raw = ivf.search(query, over_k).unwrap_or_default();
                    self.to_search_results(&raw)
                } else {
                    self.brute_force_search(query, over_k)
                }
            }
        };

        let results: Vec<SearchResult> = candidates
            .into_iter()
            .filter(|r| filter(&r.id))
            .take(k)
            .collect();

        let elapsed_us = start.elapsed().as_micros() as u64;
        {
            let mut profile = self.profile.write();
            profile.record_query(elapsed_us, true, self.config.latency_window_size);
        }

        Ok(results)
    }

    // -- Explain API ------------------------------------------------------

    /// Explain which strategy *would* be used for a search without executing it.
    pub fn explain(&self, query: &[f32], k: usize) -> SearchExplanation {
        let profile = self.profile.read();
        let mut factors = Vec::new();

        factors.push(format!(
            "Active strategy: {}, vectors: {}, dims: {}",
            self.active_strategy, profile.vector_count, profile.dimensions,
        ));

        if profile.vector_count <= self.config.brute_force_max_vectors {
            factors.push(format!(
                "Dataset ({} vectors) below brute-force threshold ({})",
                profile.vector_count, self.config.brute_force_max_vectors,
            ));
        }

        if profile.query_count > 0 {
            factors.push(format!(
                "p50={} µs, p99={} µs over {} recent queries",
                profile.p50_latency_us(),
                profile.p99_latency_us(),
                profile.recent_latencies_us.len(),
            ));
        }

        if profile.queries_with_filters_pct > self.config.high_filter_pct_threshold as f64 {
            factors.push(format!(
                "High filter workload ({:.0}%); over-fetch strategy active",
                profile.queries_with_filters_pct * 100.0,
            ));
        }

        let _ = query.len(); // acknowledge query
        let _ = k;

        let rec = self.analyze_workload_inner(&profile);
        let migration_recommended = rec.recommended != self.active_strategy;

        SearchExplanation {
            chosen_strategy: self.active_strategy,
            factors,
            estimated_cost_us: rec.estimated_latency_us,
            migration_recommended,
            migration_target: if migration_recommended {
                Some(rec.recommended)
            } else {
                None
            },
        }
    }

    // -- Analysis & migration --------------------------------------------

    /// Analyze the current workload and recommend an [`IndexStrategy`].
    pub fn analyze_workload(&self) -> IndexRecommendation {
        let profile = self.profile.read();
        self.analyze_workload_inner(&profile)
    }

    fn analyze_workload_inner(&self, profile: &WorkloadProfile) -> IndexRecommendation {
        let cfg = &self.config;

        // Tiny dataset → brute-force is fastest.
        if profile.vector_count <= cfg.brute_force_max_vectors {
            return IndexRecommendation {
                recommended: IndexStrategy::BruteForce,
                reason: format!(
                    "Dataset ({} vectors) fits within brute-force threshold ({}); exact search is optimal",
                    profile.vector_count, cfg.brute_force_max_vectors,
                ),
                estimated_recall: 1.0,
                estimated_latency_us: 100,
            };
        }

        let projected_memory = profile.vector_count * cfg.bytes_per_vector_estimate;
        let budget_constrained =
            cfg.memory_budget_bytes > 0 && projected_memory > cfg.memory_budget_bytes;
        let severely_constrained =
            cfg.memory_budget_bytes > 0 && projected_memory > cfg.memory_budget_bytes * 2;

        // DiskANN: severely memory-constrained.
        if severely_constrained {
            return IndexRecommendation {
                recommended: IndexStrategy::DiskAnn,
                reason: format!(
                    "Projected memory ({} MB) far exceeds budget ({} MB); DiskANN recommended",
                    projected_memory / (1024 * 1024),
                    cfg.memory_budget_bytes / (1024 * 1024),
                ),
                estimated_recall: 0.90,
                estimated_latency_us: 5_000,
            };
        }

        // IVF: large dataset or moderate memory pressure.
        if profile.vector_count >= cfg.ivf_min_vectors
            && (profile.vector_count > cfg.hnsw_max_vectors || budget_constrained)
        {
            let recall = if budget_constrained { 0.92 } else { 0.94 };
            return IndexRecommendation {
                recommended: IndexStrategy::Ivf,
                reason: format!(
                    "Vector count ({}) exceeds HNSW sweet-spot or memory is constrained; IVF recommended",
                    profile.vector_count,
                ),
                estimated_recall: recall,
                estimated_latency_us: 2_000,
            };
        }

        // High-filter workloads: HNSW handles post-filtering better in-memory.
        if profile.queries_with_filters_pct > cfg.high_filter_pct_threshold as f64 {
            return IndexRecommendation {
                recommended: IndexStrategy::Hnsw,
                reason: format!(
                    "High filter workload ({:.0}%); HNSW with over-fetch is best",
                    profile.queries_with_filters_pct * 100.0,
                ),
                estimated_recall: 0.96,
                estimated_latency_us: 1_000,
            };
        }

        // Default: HNSW.
        IndexRecommendation {
            recommended: IndexStrategy::Hnsw,
            reason: "Dataset fits comfortably in memory; HNSW provides best recall/latency".into(),
            estimated_recall: 0.98,
            estimated_latency_us: 500,
        }
    }

    /// Migrate the active index to `target` strategy.
    ///
    /// Rebuilds the target index from the stored vectors. DiskANN is
    /// recommendation-only so migration to it is a no-op (returns an error).
    pub fn migrate_index(&mut self, target: IndexStrategy) -> Result<()> {
        if target == self.active_strategy {
            return Ok(());
        }

        let start = Instant::now();
        let from = self.active_strategy;

        match target {
            IndexStrategy::BruteForce => {
                // No index structure needed.
                self.active_strategy = IndexStrategy::BruteForce;
            }
            IndexStrategy::Hnsw | IndexStrategy::Auto => {
                let mut hnsw = HnswIndex::new(HnswConfig::default(), self.distance);
                for (i, vec) in self.vectors.iter().enumerate() {
                    hnsw.insert(i, vec, &self.vectors)?;
                }
                self.hnsw = Some(hnsw);
                self.active_strategy = IndexStrategy::Hnsw;
            }
            IndexStrategy::Ivf => {
                let n_clusters = (self.vectors.len() as f64).sqrt().ceil() as usize;
                let n_clusters = n_clusters.max(1);
                let ivf = IvfIndex::new(self.dimensions, IvfConfig::new(n_clusters));
                self.ivf = Some(ivf);
                self.active_strategy = IndexStrategy::Ivf;
            }
            IndexStrategy::DiskAnn => {
                return Err(NeedleError::InvalidOperation(
                    "DiskANN migration is not yet implemented".into(),
                ));
            }
        }

        let duration_us = start.elapsed().as_micros() as u64;
        self.migration_history.push(MigrationEvent {
            from,
            to: self.active_strategy,
            duration_us,
            vector_count: self.vectors.len(),
            reason: format!("Migrated from {} to {}", from, self.active_strategy),
        });

        Ok(())
    }

    /// Return a snapshot of the current workload profile.
    pub fn profile(&self) -> WorkloadProfile {
        self.profile.read().clone()
    }

    /// Return the currently active strategy.
    pub fn active_strategy(&self) -> IndexStrategy {
        self.active_strategy
    }

    /// Return the history of index migrations.
    pub fn migration_history(&self) -> &[MigrationEvent] {
        &self.migration_history
    }

    /// Return total number of stored vectors.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    // -- Helpers ----------------------------------------------------------

    fn brute_force_search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        use ordered_float::OrderedFloat;
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let mut heap: BinaryHeap<Reverse<(OrderedFloat<f32>, usize)>> = BinaryHeap::new();

        for (i, vec) in self.vectors.iter().enumerate() {
            let dist = self.distance.compute(query, vec);
            heap.push(Reverse((OrderedFloat(dist), i)));
        }

        let mut results = Vec::with_capacity(k.min(self.vectors.len()));
        for _ in 0..k {
            if let Some(Reverse((OrderedFloat(dist), idx))) = heap.pop() {
                if let Some(id) = self.id_map.get(idx) {
                    results.push(SearchResult::new(id.clone(), dist, None));
                }
            } else {
                break;
            }
        }
        results
    }

    fn to_search_results(&self, raw: &[(usize, f32)]) -> Vec<SearchResult> {
        raw.iter()
            .filter_map(|(vid, dist)| {
                let id = self.id_map.get(*vid)?;
                Some(SearchResult::new(id.clone(), *dist, None))
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_index(dims: usize) -> AdaptiveIndex {
        AdaptiveIndex::new(
            dims,
            DistanceFunction::Cosine,
            IndexStrategy::Auto,
            AdaptiveIndexConfig::default(),
        )
    }

    fn random_vec(dims: usize, seed: u64) -> Vec<f32> {
        let mut v = Vec::with_capacity(dims);
        let mut s = seed;
        for _ in 0..dims {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            v.push(((s >> 33) as f32) / (u32::MAX as f32));
        }
        v
    }

    #[test]
    fn test_default_hnsw_for_small_data() {
        let idx = make_index(128);
        assert_eq!(idx.active_strategy(), IndexStrategy::Hnsw);
        let rec = idx.analyze_workload();
        // With 0 vectors, brute-force is recommended.
        assert_eq!(rec.recommended, IndexStrategy::BruteForce);
        assert!((rec.estimated_recall - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_brute_force_for_tiny_dataset() {
        let mut idx = AdaptiveIndex::new(
            4,
            DistanceFunction::Cosine,
            IndexStrategy::BruteForce,
            AdaptiveIndexConfig::default(),
        );
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a");
    }

    #[test]
    fn test_recommendation_switches_to_ivf() {
        let idx = AdaptiveIndex::new(
            128,
            DistanceFunction::Cosine,
            IndexStrategy::Auto,
            AdaptiveIndexConfig {
                brute_force_max_vectors: 100,
                hnsw_max_vectors: 1_000,
                ivf_min_vectors: 500,
                ..Default::default()
            },
        );

        {
            let mut p = idx.profile.write();
            p.update_vector_stats(2_000, 128);
        }

        let rec = idx.analyze_workload();
        assert_eq!(rec.recommended, IndexStrategy::Ivf);
        assert!(rec.reason.contains("IVF"));
    }

    #[test]
    fn test_workload_profile_tracking() {
        let mut idx = make_index(4);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let _ = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        let _ = idx
            .search_with_filter(&[0.0, 1.0, 0.0, 0.0], 1, |_| true)
            .unwrap();

        let p = idx.profile();
        assert_eq!(p.query_count, 2);
        assert_eq!(p.vector_count, 2);
        assert_eq!(p.dimensions, 4);
        assert!(p.queries_with_filters_pct > 0.0);
    }

    #[test]
    fn test_insert_and_search() {
        let mut idx = make_index(4);
        idx.insert("v1", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("v2", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.insert("v3", &[0.0, 0.0, 1.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_migration_recommendation() {
        let mut idx = make_index(4);
        for i in 0..10 {
            idx.insert(format!("v{i}"), &random_vec(4, i as u64))
                .unwrap();
        }

        // Migrate to BruteForce.
        idx.migrate_index(IndexStrategy::BruteForce).unwrap();
        assert_eq!(idx.active_strategy(), IndexStrategy::BruteForce);
        assert_eq!(idx.migration_history().len(), 1);

        // Migrate to IVF (creates untrained index).
        idx.migrate_index(IndexStrategy::Ivf).unwrap();
        assert_eq!(idx.active_strategy(), IndexStrategy::Ivf);

        // Migrate back to HNSW.
        idx.migrate_index(IndexStrategy::Hnsw).unwrap();
        assert_eq!(idx.active_strategy(), IndexStrategy::Hnsw);
        assert_eq!(idx.migration_history().len(), 3);

        // DiskANN migration should fail.
        let err = idx.migrate_index(IndexStrategy::DiskAnn);
        assert!(err.is_err());
    }

    #[test]
    fn test_memory_budget_constraint() {
        let cfg = AdaptiveIndexConfig {
            memory_budget_bytes: 1_000,
            bytes_per_vector_estimate: 512,
            brute_force_max_vectors: 0,
            hnsw_max_vectors: 1_000_000,
            ivf_min_vectors: 100_000,
            ..Default::default()
        };
        let idx = AdaptiveIndex::new(128, DistanceFunction::Cosine, IndexStrategy::Auto, cfg);

        {
            let mut p = idx.profile.write();
            p.update_vector_stats(10_000, 128);
        }

        let rec = idx.analyze_workload();
        assert_eq!(rec.recommended, IndexStrategy::DiskAnn);
        assert!(rec.reason.contains("DiskANN"));
    }

    #[test]
    fn test_config_defaults() {
        let cfg = AdaptiveIndexConfig::default();
        assert_eq!(cfg.brute_force_max_vectors, 1_000);
        assert_eq!(cfg.hnsw_max_vectors, 1_000_000);
        assert_eq!(cfg.ivf_min_vectors, 100_000);
        assert_eq!(cfg.ivf_max_vectors, 50_000_000);
        assert_eq!(cfg.memory_budget_bytes, 0);
        assert!((cfg.target_recall - 0.95).abs() < f32::EPSILON);
        assert_eq!(cfg.max_latency_us, 10_000);
        assert!((cfg.high_filter_pct_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.latency_window_size, 100);
        assert!(!cfg.auto_migrate);
    }

    #[test]
    fn test_high_filter_workload() {
        let mut idx = AdaptiveIndex::new(
            4,
            DistanceFunction::Cosine,
            IndexStrategy::Auto,
            AdaptiveIndexConfig {
                brute_force_max_vectors: 0, // disable brute-force
                ..Default::default()
            },
        );
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.insert("b", &[0.0, 1.0, 0.0, 0.0]).unwrap();

        for _ in 0..10 {
            let _ = idx
                .search_with_filter(&[1.0, 0.0, 0.0, 0.0], 1, |_| true)
                .unwrap();
        }

        let rec = idx.analyze_workload();
        assert_eq!(rec.recommended, IndexStrategy::Hnsw);
        assert!(rec.reason.contains("filter"));
    }

    #[test]
    fn test_dimension_mismatch_insert() {
        let mut idx = make_index(4);
        let err = idx.insert("bad", &[1.0, 2.0]);
        assert!(err.is_err());
    }

    #[test]
    fn test_dimension_mismatch_search() {
        let idx = make_index(4);
        let err = idx.search(&[1.0], 1);
        assert!(err.is_err());
    }

    #[test]
    fn test_explain_api() {
        let mut idx = make_index(4);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let _ = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();

        let explanation = idx.explain(&[1.0, 0.0, 0.0, 0.0], 1);
        assert_eq!(explanation.chosen_strategy, IndexStrategy::Hnsw);
        assert!(!explanation.factors.is_empty());

        // The display impl should produce readable output.
        let display = format!("{}", explanation);
        assert!(display.contains("Strategy:"));
    }

    #[test]
    fn test_latency_percentiles() {
        let mut idx = make_index(4);
        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();

        for _ in 0..20 {
            let _ = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        }

        let p = idx.profile();
        assert!(p.p50_latency_us() <= p.p99_latency_us());
        assert_eq!(p.query_count, 20);
    }

    #[test]
    fn test_auto_migrate() {
        let cfg = AdaptiveIndexConfig {
            brute_force_max_vectors: 5,
            auto_migrate: true,
            ..Default::default()
        };
        let mut idx =
            AdaptiveIndex::new(4, DistanceFunction::Cosine, IndexStrategy::BruteForce, cfg);
        // Insert enough vectors to trigger migration from brute-force.
        for i in 0..10 {
            idx.insert(format!("v{i}"), &random_vec(4, i as u64))
                .unwrap();
        }
        // Should have auto-migrated to HNSW since brute_force_max_vectors=5.
        assert_eq!(idx.active_strategy(), IndexStrategy::Hnsw);
        assert!(!idx.migration_history().is_empty());
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut idx = make_index(4);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);

        idx.insert("a", &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(!idx.is_empty());
        assert_eq!(idx.len(), 1);
    }
}
