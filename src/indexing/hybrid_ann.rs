//! # ANN + Exact Hybrid Search
//!
//! This module provides a hybrid search strategy that combines Approximate
//! Nearest Neighbor (ANN) search with exact brute-force search for guaranteed
//! accuracy when needed.
//!
//! ## Features
//!
//! - **Adaptive Switching**: Automatically switches between ANN and exact search
//! - **Recall Estimation**: Estimates ANN recall based on distance distributions
//! - **Quality Verification**: Optionally verifies ANN results with exact search
//! - **Configurable Thresholds**: Tune the tradeoff between speed and accuracy
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use needle::hybrid_ann::{HybridSearch, HybridConfig, SearchStrategy};
//!
//! let config = HybridConfig::default()
//!     .with_min_recall(0.99)
//!     .with_verification_rate(0.1);
//!
//! let hybrid = HybridSearch::new(collection, config);
//!
//! // Search with automatic strategy selection
//! let results = hybrid.search(&query, 10)?;
//!
//! // Force exact search for critical queries
//! let exact_results = hybrid.exact_search(&query, 10)?;
//! ```

use std::collections::{HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::collection::{Collection, SearchResult};
use crate::error::Result;
use crate::metadata::Filter;

/// Search strategy to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Use ANN (HNSW) only
    AnnOnly,
    /// Use exact brute-force only
    ExactOnly,
    /// Adaptive: choose based on collection size and recall requirements
    Adaptive,
    /// ANN with verification: run ANN, then verify with exact on a sample
    AnnWithVerification,
    /// Cascade: try ANN first, fall back to exact if recall seems low
    Cascade,
}

/// Configuration for hybrid search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridConfig {
    /// Minimum acceptable recall (0.0-1.0)
    pub min_recall: f32,
    /// Collection size threshold below which exact search is preferred
    pub exact_threshold: usize,
    /// Search strategy to use
    pub strategy: SearchStrategy,
    /// Fraction of queries to verify with exact search (0.0-1.0)
    pub verification_rate: f32,
    /// Distance ratio threshold for cascade fallback
    pub cascade_threshold: f32,
    /// Number of recall samples to maintain
    pub recall_sample_size: usize,
    /// ef_search override for ANN (if set)
    pub ef_search_override: Option<usize>,
    /// Enable recall estimation
    pub enable_recall_estimation: bool,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            min_recall: 0.95,
            exact_threshold: 1000,
            strategy: SearchStrategy::Adaptive,
            verification_rate: 0.05, // 5% verification
            cascade_threshold: 1.5,  // If worst ANN distance > 1.5x best, cascade
            recall_sample_size: 100,
            ef_search_override: None,
            enable_recall_estimation: true,
        }
    }
}

impl HybridConfig {
    /// Set minimum recall requirement
    #[must_use]
    pub fn with_min_recall(mut self, recall: f32) -> Self {
        self.min_recall = recall.clamp(0.0, 1.0);
        self
    }

    /// Set exact search threshold
    #[must_use]
    pub fn with_exact_threshold(mut self, threshold: usize) -> Self {
        self.exact_threshold = threshold;
        self
    }

    /// Set search strategy
    #[must_use]
    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set verification rate
    #[must_use]
    pub fn with_verification_rate(mut self, rate: f32) -> Self {
        self.verification_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set ef_search override
    #[must_use]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search_override = Some(ef);
        self
    }
}

/// Result of a hybrid search operation
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Search results
    pub results: Vec<SearchResult>,
    /// Strategy that was used
    pub strategy_used: SearchStrategy,
    /// Estimated recall (if available)
    pub estimated_recall: Option<f32>,
    /// Whether verification was performed
    pub verified: bool,
    /// Actual recall (if verified)
    pub actual_recall: Option<f32>,
    /// Search latency in milliseconds
    pub latency_ms: f32,
    /// Number of distance computations
    pub distance_computations: usize,
}

/// Statistics for hybrid search
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HybridSearchStats {
    /// Total searches performed
    pub total_searches: u64,
    /// ANN-only searches
    pub ann_searches: u64,
    /// Exact-only searches
    pub exact_searches: u64,
    /// Cascade searches
    pub cascade_searches: u64,
    /// Verified searches
    pub verified_searches: u64,
    /// Average estimated recall
    pub avg_estimated_recall: f32,
    /// Average actual recall (when verified)
    pub avg_actual_recall: f32,
    /// Average ANN latency (ms)
    pub avg_ann_latency_ms: f32,
    /// Average exact latency (ms)
    pub avg_exact_latency_ms: f32,
}

/// Recall estimation sample
#[derive(Debug, Clone)]
struct RecallSample {
    /// Estimated recall
    estimated: f32,
    /// Actual recall (if verified)
    actual: Option<f32>,
    /// Query characteristics hash
    _query_hash: u64,
}

/// Hybrid search combining ANN and exact search
pub struct HybridSearch<'a> {
    collection: &'a Collection,
    config: HybridConfig,
    /// Recent recall samples
    recall_samples: RwLock<VecDeque<RecallSample>>,
    /// Statistics
    stats: RwLock<HybridSearchStats>,
    /// Query counter for verification sampling
    query_counter: AtomicU64,
}

impl<'a> HybridSearch<'a> {
    /// Create a new hybrid search instance
    pub fn new(collection: &'a Collection, config: HybridConfig) -> Self {
        let sample_size = config.recall_sample_size;
        Self {
            collection,
            config,
            recall_samples: RwLock::new(VecDeque::with_capacity(sample_size)),
            stats: RwLock::new(HybridSearchStats::default()),
            query_counter: AtomicU64::new(0),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(collection: &'a Collection) -> Self {
        Self::new(collection, HybridConfig::default())
    }

    /// Perform a search with automatic strategy selection
    pub fn search(&self, query: &[f32], k: usize) -> Result<HybridSearchResult> {
        self.search_with_filter(query, k, None)
    }

    /// Perform a search with filter
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<HybridSearchResult> {
        let start = std::time::Instant::now();
        let query_num = self.query_counter.fetch_add(1, Ordering::Relaxed);

        // Determine strategy
        let strategy = self.select_strategy(k);

        let result = match strategy {
            SearchStrategy::AnnOnly => self.ann_search(query, k, filter)?,
            SearchStrategy::ExactOnly => self.exact_search_internal(query, k, filter)?,
            SearchStrategy::Adaptive => self.adaptive_search(query, k, filter)?,
            SearchStrategy::AnnWithVerification => {
                self.ann_with_verification(query, k, filter, query_num)?
            }
            SearchStrategy::Cascade => self.cascade_search(query, k, filter)?,
        };

        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        // Update statistics
        self.update_stats(&result, elapsed_ms);

        Ok(HybridSearchResult {
            results: result.results,
            strategy_used: strategy,
            estimated_recall: result.estimated_recall,
            verified: result.verified,
            actual_recall: result.actual_recall,
            latency_ms: elapsed_ms,
            distance_computations: result.distance_computations,
        })
    }

    /// Force ANN search (internal)
    fn ann_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<InternalSearchResult> {
        let results = if let Some(f) = filter {
            self.collection.search_with_filter(query, k, f)?
        } else {
            self.collection.search(query, k)?
        };

        let estimated_recall = if self.config.enable_recall_estimation {
            Some(self.estimate_recall(&results, k))
        } else {
            None
        };

        Ok(InternalSearchResult {
            results,
            estimated_recall,
            verified: false,
            actual_recall: None,
            distance_computations: k * 10, // Approximate
        })
    }

    /// Force exact search
    pub fn exact_search(&self, query: &[f32], k: usize) -> Result<HybridSearchResult> {
        let start = std::time::Instant::now();

        let internal = self.exact_search_internal(query, k, None)?;

        let elapsed_ms = start.elapsed().as_secs_f32() * 1000.0;

        Ok(HybridSearchResult {
            results: internal.results,
            strategy_used: SearchStrategy::ExactOnly,
            estimated_recall: Some(1.0), // Exact search has perfect recall
            verified: true,
            actual_recall: Some(1.0),
            latency_ms: elapsed_ms,
            distance_computations: internal.distance_computations,
        })
    }

    /// Internal exact search implementation
    fn exact_search_internal(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<InternalSearchResult> {
        // Use collection's iterator to scan all vectors
        let distance_fn = self.collection.config().distance;

        let mut all_distances: Vec<(String, f32)> = Vec::new();

        for (id, vector, metadata) in self.collection.iter() {
            // Apply filter if present
            if let Some(f) = filter {
                if let Some(ref meta) = metadata {
                    if !f.matches(Some(meta)) {
                        continue;
                    }
                } else {
                    continue; // No metadata to match
                }
            }

            let distance = distance_fn.compute(query, &vector)?;
            all_distances.push((id.to_string(), distance));
        }

        // Sort by distance
        all_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        let results: Vec<SearchResult> = all_distances
            .into_iter()
            .take(k)
            .map(|(id, distance)| {
                let metadata = self.collection.get(&id).and_then(|(_, m)| m.cloned());
                SearchResult {
                    id,
                    distance,
                    metadata,
                }
            })
            .collect();

        let count = self.collection.len();

        Ok(InternalSearchResult {
            results,
            estimated_recall: Some(1.0),
            verified: true,
            actual_recall: Some(1.0),
            distance_computations: count,
        })
    }

    /// Adaptive search based on collection size and requirements
    fn adaptive_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<InternalSearchResult> {
        let collection_size = self.collection.len();

        // Use exact for small collections
        if collection_size <= self.config.exact_threshold {
            debug!(
                size = collection_size,
                threshold = self.config.exact_threshold,
                "Using exact search for small collection"
            );
            return self.exact_search_internal(query, k, filter);
        }

        // Check recall estimation from recent samples
        let avg_recall = self.average_recall();
        if avg_recall < self.config.min_recall {
            debug!(
                avg_recall = avg_recall,
                min_recall = self.config.min_recall,
                "Recall too low, using exact search"
            );
            return self.exact_search_internal(query, k, filter);
        }

        // Default to ANN
        self.ann_search(query, k, filter)
    }

    /// ANN search with periodic verification
    fn ann_with_verification(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
        query_num: u64,
    ) -> Result<InternalSearchResult> {
        let mut result = self.ann_search(query, k, filter)?;

        // Verify based on rate
        let should_verify = (query_num as f32 * self.config.verification_rate) as u64
            != ((query_num.saturating_sub(1)) as f32 * self.config.verification_rate) as u64;

        if should_verify && !result.results.is_empty() {
            let exact = self.exact_search_internal(query, k, filter)?;
            let actual_recall = self.calculate_recall(&result.results, &exact.results);

            result.verified = true;
            result.actual_recall = Some(actual_recall);

            // Record sample
            self.record_recall_sample(
                result.estimated_recall.unwrap_or(0.0),
                Some(actual_recall),
                self.hash_query(query),
            );
        }

        Ok(result)
    }

    /// Cascade search: ANN first, exact if quality seems low
    fn cascade_search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<InternalSearchResult> {
        let ann_result = self.ann_search(query, k, filter)?;

        if ann_result.results.is_empty() {
            return self.exact_search_internal(query, k, filter);
        }

        // Check distance spread
        let min_dist = ann_result
            .results
            .first()
            .map(|r| r.distance)
            .unwrap_or(0.0);
        let max_dist = ann_result.results.last().map(|r| r.distance).unwrap_or(0.0);

        if min_dist > 0.0 && max_dist / min_dist > self.config.cascade_threshold {
            debug!(
                min_dist = min_dist,
                max_dist = max_dist,
                threshold = self.config.cascade_threshold,
                "Distance spread too high, cascading to exact search"
            );

            let mut stats = self.stats.write();
            stats.cascade_searches += 1;

            return self.exact_search_internal(query, k, filter);
        }

        Ok(ann_result)
    }

    /// Select strategy based on configuration and conditions
    fn select_strategy(&self, _k: usize) -> SearchStrategy {
        match self.config.strategy {
            SearchStrategy::Adaptive => {
                if self.collection.len() <= self.config.exact_threshold {
                    SearchStrategy::ExactOnly
                } else {
                    SearchStrategy::AnnOnly
                }
            }
            other => other,
        }
    }

    /// Estimate recall based on result distances
    fn estimate_recall(&self, results: &[SearchResult], _k: usize) -> f32 {
        if results.is_empty() {
            return 0.0;
        }

        // Heuristic: estimate recall based on distance distribution
        // Lower variance in distances suggests higher recall
        let distances: Vec<f32> = results.iter().map(|r| r.distance).collect();

        let mean = distances.iter().sum::<f32>() / distances.len() as f32;
        let variance =
            distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
        let std_dev = variance.sqrt();

        // Coefficient of variation - lower is better
        let cv = if mean > 1e-6 { std_dev / mean } else { 0.0 };

        // Map CV to recall estimate (heuristic)
        // CV < 0.1 -> high recall (~0.99)
        // CV > 0.5 -> lower recall (~0.85)
        let estimated_recall = 1.0 - (cv * 0.3).min(0.15);

        estimated_recall.clamp(0.8, 0.99)
    }

    /// Calculate actual recall between ANN and exact results
    fn calculate_recall(
        &self,
        ann_results: &[SearchResult],
        exact_results: &[SearchResult],
    ) -> f32 {
        if exact_results.is_empty() {
            return if ann_results.is_empty() { 1.0 } else { 0.0 };
        }

        let exact_ids: HashSet<_> = exact_results.iter().map(|r| &r.id).collect();
        let ann_ids: HashSet<_> = ann_results.iter().map(|r| &r.id).collect();

        let overlap = exact_ids.intersection(&ann_ids).count();
        overlap as f32 / exact_results.len() as f32
    }

    /// Record a recall sample
    fn record_recall_sample(&self, estimated: f32, actual: Option<f32>, query_hash: u64) {
        let mut samples = self.recall_samples.write();

        if samples.len() >= self.config.recall_sample_size {
            samples.pop_front();
        }

        samples.push_back(RecallSample {
            estimated,
            actual,
            _query_hash: query_hash,
        });
    }

    /// Get average recall from recent samples
    fn average_recall(&self) -> f32 {
        let samples = self.recall_samples.read();

        if samples.is_empty() {
            return 1.0; // Assume good recall if no data
        }

        // Prefer actual recall if available, otherwise use estimated
        let sum: f32 = samples
            .iter()
            .map(|s| s.actual.unwrap_or(s.estimated))
            .sum();

        sum / samples.len() as f32
    }

    /// Hash query vector for tracking
    fn hash_query(&self, query: &[f32]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for &v in query {
            v.to_bits().hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Update statistics
    fn update_stats(&self, result: &InternalSearchResult, latency_ms: f32) {
        let mut stats = self.stats.write();

        stats.total_searches += 1;

        if result.verified {
            stats.verified_searches += 1;
            if let Some(recall) = result.actual_recall {
                let n = stats.verified_searches as f32;
                stats.avg_actual_recall = stats.avg_actual_recall * (n - 1.0) / n + recall / n;
            }
        }

        if let Some(recall) = result.estimated_recall {
            let n = stats.total_searches as f32;
            stats.avg_estimated_recall = stats.avg_estimated_recall * (n - 1.0) / n + recall / n;
        }

        // Update latency averages (approximate based on whether exact was used)
        if result.distance_computations > 1000 {
            let n = (stats.exact_searches + 1) as f32;
            stats.avg_exact_latency_ms =
                stats.avg_exact_latency_ms * (n - 1.0) / n + latency_ms / n;
            stats.exact_searches += 1;
        } else {
            let n = (stats.ann_searches + 1) as f32;
            stats.avg_ann_latency_ms = stats.avg_ann_latency_ms * (n - 1.0) / n + latency_ms / n;
            stats.ann_searches += 1;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> HybridSearchStats {
        self.stats.read().clone()
    }

    /// Get recall statistics
    pub fn recall_stats(&self) -> RecallStats {
        let samples = self.recall_samples.read();

        let verified_samples: Vec<_> = samples.iter().filter(|s| s.actual.is_some()).collect();

        let avg_estimated = if samples.is_empty() {
            0.0
        } else {
            samples.iter().map(|s| s.estimated).sum::<f32>() / samples.len() as f32
        };

        let avg_actual = if verified_samples.is_empty() {
            0.0
        } else {
            verified_samples
                .iter()
                .map(|s| s.actual.unwrap_or(0.0))
                .sum::<f32>()
                / verified_samples.len() as f32
        };

        let estimation_error = if !verified_samples.is_empty() {
            verified_samples
                .iter()
                .map(|s| (s.estimated - s.actual.unwrap_or(0.0)).abs())
                .sum::<f32>()
                / verified_samples.len() as f32
        } else {
            0.0
        };

        RecallStats {
            sample_count: samples.len(),
            verified_count: verified_samples.len(),
            avg_estimated_recall: avg_estimated,
            avg_actual_recall: avg_actual,
            estimation_error,
        }
    }
}

/// Internal search result
struct InternalSearchResult {
    results: Vec<SearchResult>,
    estimated_recall: Option<f32>,
    verified: bool,
    actual_recall: Option<f32>,
    distance_computations: usize,
}

/// Recall statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallStats {
    /// Number of samples collected
    pub sample_count: usize,
    /// Number of verified samples
    pub verified_count: usize,
    /// Average estimated recall
    pub avg_estimated_recall: f32,
    /// Average actual recall (from verification)
    pub avg_actual_recall: f32,
    /// Average estimation error
    pub estimation_error: f32,
}

/// Quality-aware search builder
pub struct QualityAwareSearch<'a> {
    collection: &'a Collection,
    min_recall: f32,
    max_latency_ms: Option<f32>,
    strategy_preference: Option<SearchStrategy>,
}

impl<'a> QualityAwareSearch<'a> {
    /// Create a new quality-aware search
    pub fn new(collection: &'a Collection) -> Self {
        Self {
            collection,
            min_recall: 0.95,
            max_latency_ms: None,
            strategy_preference: None,
        }
    }

    /// Set minimum recall requirement
    #[must_use]
    pub fn with_min_recall(mut self, recall: f32) -> Self {
        self.min_recall = recall.clamp(0.0, 1.0);
        self
    }

    /// Set maximum latency constraint
    #[must_use]
    pub fn with_max_latency(mut self, ms: f32) -> Self {
        self.max_latency_ms = Some(ms);
        self
    }

    /// Set strategy preference
    #[must_use]
    pub fn with_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.strategy_preference = Some(strategy);
        self
    }

    /// Execute the search
    pub fn search(&self, query: &[f32], k: usize) -> Result<HybridSearchResult> {
        let config = HybridConfig::default()
            .with_min_recall(self.min_recall)
            .with_strategy(self.strategy_preference.unwrap_or(SearchStrategy::Adaptive));

        let hybrid = HybridSearch::new(self.collection, config);
        hybrid.search(query, k)
    }
}

// ── Sparse-Dense Fusion Index ───────────────────────────────────────────────

/// Configuration for the sparse-dense fusion index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseDenseFusionConfig {
    /// Weight for dense (vector) similarity in [0, 1].
    pub dense_weight: f32,
    /// Weight for sparse (term) similarity in [0, 1].
    pub sparse_weight: f32,
    /// Whether to dynamically tune alpha based on query characteristics.
    pub dynamic_alpha: bool,
    /// Minimum number of sparse matches before including sparse scores.
    pub min_sparse_matches: usize,
    /// RRF k parameter for rank fusion.
    pub rrf_k: f32,
}

impl Default for SparseDenseFusionConfig {
    fn default() -> Self {
        Self {
            dense_weight: 0.6,
            sparse_weight: 0.4,
            dynamic_alpha: true,
            min_sparse_matches: 1,
            rrf_k: 60.0,
        }
    }
}

impl SparseDenseFusionConfig {
    /// Set dense weight (automatically adjusts sparse weight).
    #[must_use]
    pub fn with_dense_weight(mut self, weight: f32) -> Self {
        self.dense_weight = weight.clamp(0.0, 1.0);
        self.sparse_weight = 1.0 - self.dense_weight;
        self
    }

    /// Enable or disable dynamic alpha tuning.
    #[must_use]
    pub fn with_dynamic_alpha(mut self, enabled: bool) -> Self {
        self.dynamic_alpha = enabled;
        self
    }
}

/// Result from a fused sparse-dense search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionSearchResult {
    /// Vector ID.
    pub id: String,
    /// Combined fused score (lower is better).
    pub fused_score: f32,
    /// Dense (vector) score component.
    pub dense_score: f32,
    /// Sparse (term) score component.
    pub sparse_score: f32,
    /// The alpha (dense weight) used for this result.
    pub alpha_used: f32,
}

/// In-memory sparse-dense fusion index that scores both dense similarity
/// and sparse term overlap simultaneously during traversal.
pub struct SparseDenseFusionIndex {
    config: SparseDenseFusionConfig,
    /// Dense vectors: id -> vector.
    dense_vectors: HashMap<String, Vec<f32>>,
    /// Sparse inverted index: term_index -> [(id, weight)].
    sparse_postings: HashMap<u32, Vec<(String, f32)>>,
    /// Sparse vectors: id -> sparse representation.
    sparse_vectors: HashMap<String, Vec<(u32, f32)>>,
}

impl SparseDenseFusionIndex {
    /// Create a new fusion index.
    pub fn new(config: SparseDenseFusionConfig) -> Self {
        Self {
            config,
            dense_vectors: HashMap::new(),
            sparse_postings: HashMap::new(),
            sparse_vectors: HashMap::new(),
        }
    }

    /// Insert a document with both dense and sparse representations.
    pub fn insert(
        &mut self,
        id: &str,
        dense_vector: Vec<f32>,
        sparse_terms: Vec<(u32, f32)>,
    ) {
        self.dense_vectors.insert(id.to_string(), dense_vector);

        for &(term_idx, weight) in &sparse_terms {
            self.sparse_postings
                .entry(term_idx)
                .or_default()
                .push((id.to_string(), weight));
        }
        self.sparse_vectors.insert(id.to_string(), sparse_terms);
    }

    /// Search with interleaved dense + sparse scoring.
    pub fn search(
        &self,
        dense_query: &[f32],
        sparse_query: &[(u32, f32)],
        k: usize,
    ) -> Vec<FusionSearchResult> {
        let alpha = if self.config.dynamic_alpha {
            self.compute_dynamic_alpha(sparse_query)
        } else {
            self.config.dense_weight
        };

        // Compute dense scores for all documents
        let mut scores: HashMap<String, (f32, f32)> = HashMap::new(); // (dense, sparse)

        for (id, vec) in &self.dense_vectors {
            let dense_score = cosine_dist(dense_query, vec);
            scores.insert(id.clone(), (dense_score, 0.0));
        }

        // Compute sparse scores via inverted index lookup
        for &(term_idx, query_weight) in sparse_query {
            if let Some(postings) = self.sparse_postings.get(&term_idx) {
                for (id, doc_weight) in postings {
                    let sparse_contribution = query_weight * doc_weight;
                    if let Some(entry) = scores.get_mut(id) {
                        entry.1 += sparse_contribution;
                    }
                }
            }
        }

        // Normalize sparse scores to [0, 1] range
        let max_sparse = scores.values().map(|(_, s)| *s).fold(0.0_f32, f32::max);
        if max_sparse > 0.0 {
            for entry in scores.values_mut() {
                entry.1 = 1.0 - (entry.1 / max_sparse); // Convert to distance (lower = better)
            }
        }

        // Combine scores: fused = alpha * dense + (1 - alpha) * sparse
        let mut results: Vec<FusionSearchResult> = scores
            .into_iter()
            .map(|(id, (dense, sparse))| {
                let fused = alpha * dense + (1.0 - alpha) * sparse;
                FusionSearchResult {
                    id,
                    fused_score: fused,
                    dense_score: dense,
                    sparse_score: sparse,
                    alpha_used: alpha,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            a.fused_score
                .partial_cmp(&b.fused_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Dynamically compute alpha based on query sparsity.
    /// More sparse terms → lower alpha (more weight on sparse).
    fn compute_dynamic_alpha(&self, sparse_query: &[(u32, f32)]) -> f32 {
        if sparse_query.is_empty() {
            return 1.0; // No sparse terms → pure dense search
        }

        // Count how many documents have sparse term matches
        let mut matched_docs = HashSet::new();
        for &(term_idx, _) in sparse_query {
            if let Some(postings) = self.sparse_postings.get(&term_idx) {
                for (id, _) in postings {
                    matched_docs.insert(id.clone());
                }
            }
        }

        let coverage = if self.dense_vectors.is_empty() {
            0.0
        } else {
            matched_docs.len() as f32 / self.dense_vectors.len() as f32
        };

        // High coverage → more trust in sparse signal → lower alpha
        // Low coverage → sparse is too selective → higher alpha
        let base = self.config.dense_weight;
        if coverage > 0.5 {
            (base - 0.1).max(0.3)
        } else if coverage < 0.1 {
            (base + 0.1).min(0.9)
        } else {
            base
        }
    }

    /// Number of indexed documents.
    pub fn len(&self) -> usize {
        self.dense_vectors.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.dense_vectors.is_empty()
    }
}

/// Simple cosine distance computation.
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < f32::EPSILON || nb < f32::EPSILON {
        return 1.0;
    }
    1.0 - (dot / (na * nb))
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collection::{Collection, CollectionConfig};

    fn create_test_collection() -> Collection {
        let mut collection = Collection::new(CollectionConfig::new("test", 4));

        // Insert test vectors
        for i in 0..100 {
            let vec = vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32];
            collection.insert(format!("vec_{}", i), &vec, None).unwrap();
        }

        collection
    }

    #[test]
    fn test_hybrid_config() {
        let config = HybridConfig::default()
            .with_min_recall(0.99)
            .with_exact_threshold(500)
            .with_strategy(SearchStrategy::Cascade);

        assert!((config.min_recall - 0.99).abs() < 0.001);
        assert_eq!(config.exact_threshold, 500);
        assert_eq!(config.strategy, SearchStrategy::Cascade);
    }

    #[test]
    fn test_exact_search() {
        let collection = create_test_collection();

        let config = HybridConfig::default().with_strategy(SearchStrategy::ExactOnly);
        let hybrid = HybridSearch::new(&collection, config);

        let query = vec![10.0, 20.0, 30.0, 40.0];
        let result = hybrid.search(&query, 5).unwrap();

        assert_eq!(result.results.len(), 5);
        assert_eq!(result.strategy_used, SearchStrategy::ExactOnly);
        assert!(result.estimated_recall.is_some());
    }

    #[test]
    fn test_adaptive_strategy() {
        let collection = create_test_collection();

        // With threshold higher than collection size, should use exact
        let config = HybridConfig::default()
            .with_strategy(SearchStrategy::Adaptive)
            .with_exact_threshold(200);

        let hybrid = HybridSearch::new(&collection, config);

        let query = vec![10.0, 20.0, 30.0, 40.0];
        let result = hybrid.search(&query, 5).unwrap();

        // Should choose exact since collection size (100) < threshold (200)
        assert_eq!(result.strategy_used, SearchStrategy::ExactOnly);
    }

    #[test]
    fn test_recall_calculation() {
        let collection = create_test_collection();

        let config = HybridConfig::default();
        let hybrid = HybridSearch::new(&collection, config);

        // Create two result sets with some overlap
        let ann_results = vec![
            SearchResult {
                id: "vec_1".to_string(),
                distance: 0.1,
                metadata: None,
            },
            SearchResult {
                id: "vec_2".to_string(),
                distance: 0.2,
                metadata: None,
            },
            SearchResult {
                id: "vec_3".to_string(),
                distance: 0.3,
                metadata: None,
            },
        ];

        let exact_results = vec![
            SearchResult {
                id: "vec_1".to_string(),
                distance: 0.1,
                metadata: None,
            },
            SearchResult {
                id: "vec_2".to_string(),
                distance: 0.2,
                metadata: None,
            },
            SearchResult {
                id: "vec_4".to_string(),
                distance: 0.25,
                metadata: None,
            },
        ];

        let recall = hybrid.calculate_recall(&ann_results, &exact_results);
        assert!((recall - 0.666).abs() < 0.01); // 2/3 overlap
    }

    #[test]
    fn test_statistics_tracking() {
        let collection = create_test_collection();

        let config = HybridConfig::default().with_strategy(SearchStrategy::ExactOnly);
        let hybrid = HybridSearch::new(&collection, config);

        let query = vec![10.0, 20.0, 30.0, 40.0];

        for _ in 0..10 {
            hybrid.search(&query, 5).unwrap();
        }

        let stats = hybrid.stats();
        assert_eq!(stats.total_searches, 10);
    }

    #[test]
    fn test_quality_aware_search() {
        let collection = create_test_collection();

        let query = vec![10.0, 20.0, 30.0, 40.0];

        let result = QualityAwareSearch::new(&collection)
            .with_min_recall(0.99)
            .with_strategy(SearchStrategy::ExactOnly)
            .search(&query, 5)
            .unwrap();

        assert_eq!(result.results.len(), 5);
    }

    #[test]
    fn test_sparse_dense_fusion_basic() {
        let mut index = SparseDenseFusionIndex::new(SparseDenseFusionConfig::default());

        index.insert("doc1", vec![1.0, 0.0, 0.0, 0.0], vec![(0, 1.0), (1, 0.5)]);
        index.insert("doc2", vec![0.0, 1.0, 0.0, 0.0], vec![(1, 1.0), (2, 0.5)]);
        index.insert("doc3", vec![0.0, 0.0, 1.0, 0.0], vec![(2, 1.0), (3, 0.5)]);

        let results = index.search(
            &[1.0, 0.0, 0.0, 0.0],
            &[(0, 1.0)],
            2,
        );

        assert_eq!(results.len(), 2);
        // doc1 should be the best match (closest in both dense and sparse)
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_sparse_dense_fusion_pure_dense() {
        let mut index = SparseDenseFusionIndex::new(SparseDenseFusionConfig::default());

        index.insert("doc1", vec![1.0, 0.0], vec![]);
        index.insert("doc2", vec![0.0, 1.0], vec![]);

        // No sparse query → pure dense search
        let results = index.search(&[1.0, 0.0], &[], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_sparse_dense_fusion_dynamic_alpha() {
        let config = SparseDenseFusionConfig::default().with_dynamic_alpha(true);
        let mut index = SparseDenseFusionIndex::new(config);

        for i in 0..10 {
            index.insert(
                &format!("doc{}", i),
                vec![i as f32 * 0.1, 1.0 - i as f32 * 0.1],
                vec![(i as u32, 1.0)],
            );
        }

        let results = index.search(&[0.5, 0.5], &[(0, 1.0)], 3);
        assert_eq!(results.len(), 3);
        // Dynamic alpha should be adjusted based on coverage
        assert!(results[0].alpha_used > 0.0 && results[0].alpha_used < 1.0);
    }
}
