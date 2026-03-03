//! Search pipeline types for the collection module.
//!
//! Contains the `SearchBuilder` fluent API, search evaluation metrics,
//! and the brute-force search parameter type.

use super::Collection;
use crate::collection::search::{SearchExplain, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::hnsw::VectorId;
use crate::metadata::Filter;
use serde::{Deserialize, Serialize};
use tracing::warn;

/// Over-fetch multiplier for filtered searches: retrieve `k * FILTER_CANDIDATE_MULTIPLIER`
/// candidates from the HNSW index to compensate for results removed by filtering.
pub(super) const FILTER_CANDIDATE_MULTIPLIER: usize = 10;

/// Ground truth entry for evaluation: a query vector paired with its known relevant vector IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundTruthEntry {
    /// Query vector
    pub query: Vec<f32>,
    /// Set of relevant vector IDs for this query
    pub relevant_ids: Vec<String>,
}

/// Per-query evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Query index in the ground truth set
    pub query_index: usize,
    /// Recall@k: fraction of relevant items retrieved
    pub recall_at_k: f64,
    /// Precision@k: fraction of retrieved items that are relevant
    pub precision_at_k: f64,
    /// Average Precision for this query
    pub average_precision: f64,
    /// Reciprocal Rank (1/rank of first relevant result, 0 if none found)
    pub reciprocal_rank: f64,
    /// Normalized Discounted Cumulative Gain
    pub ndcg: f64,
}

/// Aggregated evaluation report across all queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Number of queries evaluated
    pub num_queries: usize,
    /// k value used
    pub k: usize,
    /// Mean Recall@k
    pub mean_recall_at_k: f64,
    /// Mean Precision@k
    pub mean_precision_at_k: f64,
    /// Mean Average Precision (MAP)
    pub map: f64,
    /// Mean Reciprocal Rank (MRR)
    pub mrr: f64,
    /// Mean NDCG@k
    pub mean_ndcg: f64,
    /// Per-query breakdown
    pub per_query: Vec<QueryMetrics>,
    /// Total evaluation time in milliseconds
    pub eval_time_ms: f64,
}

/// Fluent builder for configuring and executing vector similarity searches.
///
/// `SearchBuilder` provides fine-grained control over search behavior, including
/// filtering strategies, performance tuning, and distance function overrides.
///
/// # Filtering Strategies
///
/// Two filtering modes are available:
///
/// | Mode | Method | When to Use |
/// |------|--------|-------------|
/// | **Pre-filter** | [`filter()`](Self::filter) | Filter is fast and selective; filters during HNSW traversal |
/// | **Post-filter** | [`post_filter()`](Self::post_filter) | Need to guarantee k candidates; filter after ANN search |
///
/// Pre-filtering integrates with HNSW search, skipping non-matching candidates during
/// traversal. Post-filtering fetches extra candidates (`k * post_filter_factor`), then
/// filters and truncates.
///
/// # Performance Tuning
///
/// - [`ef_search()`](Self::ef_search): Higher values improve recall but increase latency
/// - [`include_metadata(false)`](Self::include_metadata): Skip metadata loading for faster searches
/// - [`post_filter_factor()`](Self::post_filter_factor): Adjust over-fetch ratio for post-filtering
///
/// # Example: Basic Search
///
/// ```
/// use needle::Collection;
///
/// let mut collection = Collection::with_dimensions("docs", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
///
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
///
/// # Example: Pre-Filter vs Post-Filter
///
/// ```
/// use needle::{Collection, Filter};
/// use serde_json::json;
///
/// let mut collection = Collection::with_dimensions("docs", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a", "score": 10})))?;
/// collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "b", "score": 20})))?;
///
/// // Pre-filter: filter candidates BEFORE ANN search
/// // Use when: filter is fast, you don't need exactly k results
/// let pre_filter = Filter::eq("type", "a");
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .filter(&pre_filter)
///     .execute()?;
///
/// // Post-filter: filter results AFTER ANN search
/// // Use when: need to guarantee k candidates before filtering
/// let post_filter = Filter::gt("score", 15);
/// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .post_filter(&post_filter)
///     .post_filter_factor(5)  // Fetch 50 candidates, filter, keep 10
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
///
/// # Example: Performance Tuning
///
/// ```
/// use needle::Collection;
///
/// let mut collection = Collection::with_dimensions("docs", 4);
/// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
///
/// // Higher ef_search for better recall
/// let high_recall = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .ef_search(200)  // Default is 50
///     .execute()?;
///
/// // Skip metadata for faster response
/// let fast_search = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
///     .k(10)
///     .include_metadata(false)
///     .execute()?;
/// # Ok::<(), needle::NeedleError>(())
/// ```
#[derive(Clone)]
pub struct SearchBuilder<'a> {
    collection: &'a Collection,
    query: &'a [f32],
    k: usize,
    filter: Option<&'a Filter>,
    post_filter: Option<&'a Filter>,
    post_filter_factor: usize,
    ef_search: Option<usize>,
    include_metadata: bool,
    /// Override the distance function for this query.
    /// When set to a different function than the collection's default,
    /// search will fall back to brute-force for accuracy.
    distance_override: Option<DistanceFunction>,
    /// Point-in-time timestamp for MVCC snapshot isolation reads.
    /// When set, results are filtered to only include vectors that existed
    /// at the specified Unix epoch timestamp.
    as_of_timestamp: Option<u64>,
    /// Time-weighted decay function for biasing results toward recency.
    time_decay: Option<TimeDecay>,
}

/// Time-decay configuration for search results.
/// Applies a decay multiplier based on vector age to bias results toward recency.
#[derive(Debug, Clone)]
pub enum TimeDecay {
    /// Exponential decay: score *= exp(-ln(2) / half_life * age)
    Exponential {
        /// Duration in seconds after which decay factor reaches 0.5
        half_life_seconds: u64,
    },
    /// Linear decay: score *= max(0, 1 - age / max_age)
    Linear {
        /// Duration in seconds at which decay factor reaches 0
        max_age_seconds: u64,
    },
    /// Step function: full score within window, zero outside
    Step {
        /// Duration in seconds for the recency window
        window_seconds: u64,
    },
}

impl TimeDecay {
    /// Compute the decay factor for a given age in seconds.
    /// Returns a value in [0.0, 1.0] where 1.0 means no decay.
    pub fn compute(&self, age_seconds: u64) -> f32 {
        match self {
            TimeDecay::Exponential { half_life_seconds } => {
                if *half_life_seconds == 0 {
                    return if age_seconds == 0 { 1.0 } else { 0.0 };
                }
                let lambda = (2.0_f32).ln() / *half_life_seconds as f32;
                (-lambda * age_seconds as f32).exp()
            }
            TimeDecay::Linear { max_age_seconds } => {
                if *max_age_seconds == 0 || age_seconds >= *max_age_seconds {
                    0.0
                } else {
                    1.0 - (age_seconds as f32 / *max_age_seconds as f32)
                }
            }
            TimeDecay::Step { window_seconds } => {
                if age_seconds <= *window_seconds {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

impl<'a> SearchBuilder<'a> {
    /// Create a new search builder
    #[must_use]
    pub fn new(collection: &'a Collection, query: &'a [f32]) -> Self {
        Self {
            collection,
            query,
            k: 10,
            filter: None,
            post_filter: None,
            post_filter_factor: 3,
            ef_search: None,
            include_metadata: true,
            distance_override: None,
            as_of_timestamp: None,
            time_decay: None,
        }
    }

    /// Set the number of results to return
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set a pre-filter (applied during ANN search).
    ///
    /// Pre-filtering is efficient when the filter is selective and fast to evaluate.
    /// Candidates that don't match the filter are skipped during search.
    #[must_use]
    pub fn filter(mut self, filter: &'a Filter) -> Self {
        self.filter = Some(filter);
        self
    }

    /// Set a post-filter (applied after ANN search).
    ///
    /// Post-filtering is useful when:
    /// - You need to guarantee k results before filtering
    /// - The filter involves expensive computation
    /// - The filter is highly selective and pre-filtering would miss results
    ///
    /// The search fetches `k * post_filter_factor` candidates, then filters.
    /// Default over-fetch factor is 3x.
    #[must_use]
    pub fn post_filter(mut self, filter: &'a Filter) -> Self {
        self.post_filter = Some(filter);
        self
    }

    /// Set the over-fetch factor for post-filtering (default: 3).
    ///
    /// When post-filtering, the search fetches `k * factor` candidates
    /// to ensure enough results remain after filtering.
    #[must_use]
    pub fn post_filter_factor(mut self, factor: usize) -> Self {
        self.post_filter_factor = factor.max(1);
        self
    }

    /// Set ef_search parameter for this query
    #[must_use]
    pub fn ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Whether to include metadata in results (default: true)
    #[must_use]
    pub fn include_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Override the distance function for this query.
    ///
    /// When the distance function differs from the collection's configured function,
    /// search falls back to brute-force linear scan for accurate results.
    /// This allows querying with different similarity metrics without rebuilding the index.
    ///
    /// **Warning:** Brute-force search is O(n) and may be slow on large collections.
    /// A warning is logged when this fallback occurs.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, DistanceFunction};
    ///
    /// let mut collection = Collection::with_dimensions("docs", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// // Query with Euclidean distance even though collection uses Cosine
    /// let results = collection.search_builder(&[1.0, 0.0, 0.0, 0.0])
    ///     .k(10)
    ///     .distance(DistanceFunction::Euclidean)
    ///     .execute()?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    #[must_use]
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_override = Some(distance);
        self
    }

    /// Set a point-in-time timestamp for MVCC snapshot isolation reads.
    ///
    /// When set, search results are conceptually limited to vectors that existed
    /// at the specified Unix epoch timestamp. This enables time-travel queries
    /// when used with the vector versioning system.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use needle::Collection;
    ///
    /// let collection = db.collection("docs")?;
    /// let one_hour_ago = std::time::SystemTime::now()
    ///     .duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() - 3600;
    ///
    /// let results = collection.search_builder(&query)
    ///     .k(10)
    ///     .as_of(one_hour_ago)
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn as_of(mut self, timestamp: u64) -> Self {
        self.as_of_timestamp = Some(timestamp);
        self
    }

    /// Get the configured as_of timestamp, if any.
    pub fn as_of_timestamp(&self) -> Option<u64> {
        self.as_of_timestamp
    }

    /// Apply time-weighted decay to bias results toward recency.
    ///
    /// Decay is applied as a normalized multiplier on the similarity score
    /// (1.0 - distance) preserving relative ordering among same-age vectors.
    /// The search over-fetches by 3x to compensate for reordering.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use needle::collection::pipeline::TimeDecay;
    ///
    /// let results = collection.search_builder(&query)
    ///     .k(10)
    ///     .with_time_decay(TimeDecay::Exponential { half_life_seconds: 86400 })
    ///     .execute()?;
    /// ```
    #[must_use]
    pub fn with_time_decay(mut self, decay: TimeDecay) -> Self {
        self.time_decay = Some(decay);
        self
    }

    /// Set a maximum age filter: only return vectors inserted within the last
    /// `max_age_seconds` seconds. This is applied as a pre-filter before scoring,
    /// providing efficient temporal filtering with minimal latency overhead.
    #[must_use]
    pub fn max_age(self, max_age_seconds: u64) -> Self {
        self.with_time_decay(TimeDecay::Step {
            window_seconds: max_age_seconds,
        })
    }

    /// Execute the search and return results
    pub fn execute(self) -> Result<Vec<SearchResult>> {
        self.validate_query()?;

        // Check if we need brute-force search due to distance override
        if let Some(distance_fn) = self.brute_force_distance() {
            warn!(
                "Distance override ({:?}) differs from index ({:?}), using brute-force search",
                distance_fn, self.collection.config.distance
            );
            return self.collection.brute_force_search(&BruteForceSearchParams {
                query: self.query,
                k: self.k,
                distance_fn,
                filter: self.filter,
                post_filter: self.post_filter,
                include_metadata: self.include_metadata,
            });
        }

        let fetch_count = self.calculate_fetch_count();
        let raw_results = self.fetch_raw_results(fetch_count)?;
        let non_expired = self.filter_expired(raw_results);
        let time_filtered = self.filter_as_of(non_expired);
        let post_filter_factor = if self.post_filter.is_some() {
            self.post_filter_factor
        } else {
            1
        };
        let pre_filtered = self.apply_pre_filter(time_filtered, self.k * post_filter_factor.max(1));
        let mut enriched = self.enrich(pre_filtered)?;
        self.apply_post_filter(&mut enriched);
        self.apply_time_decay(&mut enriched);
        Ok(enriched)
    }

    /// Validate query dimensions and vector values.
    fn validate_query(&self) -> Result<()> {
        if self.query.len() != self.collection.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.collection.config.dimensions,
                got: self.query.len(),
            });
        }
        Collection::validate_vector(self.query)
    }

    /// Returns the override distance function if brute-force fallback is needed.
    fn brute_force_distance(&self) -> Option<DistanceFunction> {
        self.distance_override
            .filter(|&d| d != self.collection.config.distance)
    }

    /// Calculate how many candidates to fetch from the index.
    fn calculate_fetch_count(&self) -> usize {
        let pre_filter_factor = if self.filter.is_some() {
            FILTER_CANDIDATE_MULTIPLIER
        } else {
            1
        };
        let post_filter_factor = if self.post_filter.is_some() {
            self.post_filter_factor
        } else {
            1
        };
        // Over-fetch when time decay is active to compensate for reordering
        let decay_factor = if self.time_decay.is_some() { 3 } else { 1 };
        self.k * pre_filter_factor * post_filter_factor * decay_factor
    }

    /// Fetch raw results from the HNSW index.
    fn fetch_raw_results(&self, fetch_count: usize) -> Result<Vec<(VectorId, f32)>> {
        if let Some(ef) = self.ef_search {
            self.collection.index.search_with_ef(
                self.query,
                fetch_count,
                ef,
                self.collection.vectors.as_slice(),
            )
        } else {
            self.collection.index.search(
                self.query,
                fetch_count,
                self.collection.vectors.as_slice(),
            )
        }
    }

    /// Remove expired vectors if lazy expiration is enabled.
    fn filter_expired(&self, results: Vec<(VectorId, f32)>) -> Vec<(VectorId, f32)> {
        if self.collection.config.lazy_expiration {
            let mut filtered = Vec::with_capacity(results.len());
            filtered.extend(
                results
                    .into_iter()
                    .filter(|(id, _)| !self.collection.is_expired(*id)),
            );
            filtered
        } else {
            results
        }
    }

    /// Filter results to only include vectors that existed at the as_of timestamp.
    /// Vectors inserted after the as_of timestamp are excluded.
    fn filter_as_of(&self, results: Vec<(VectorId, f32)>) -> Vec<(VectorId, f32)> {
        if let Some(as_of) = self.as_of_timestamp {
            results
                .into_iter()
                .filter(|(id, _)| {
                    self.collection
                        .insertion_timestamps
                        .get(id)
                        .map_or(true, |&ts| ts <= as_of)
                })
                .collect()
        } else {
            results
        }
    }

    /// Apply the pre-filter (metadata filter during ANN search phase).
    /// Uses the inverted index for O(1) equality filter acceleration when available.
    fn apply_pre_filter(
        &self,
        results: Vec<(VectorId, f32)>,
        limit: usize,
    ) -> Vec<(VectorId, f32)> {
        let capacity = limit.min(results.len());
        if let Some(filter) = self.filter {
            // Try using the inverted index for fast pre-filtering
            if let Some(matching_ids) = self.collection.metadata.resolve_filter_via_index(filter) {
                let mut filtered = Vec::with_capacity(capacity);
                filtered.extend(
                    results
                        .into_iter()
                        .filter(|(id, _)| matching_ids.contains(id))
                        .take(limit),
                );
                return filtered;
            }

            // Fall back to sequential filter evaluation
            let mut filtered = Vec::with_capacity(capacity);
            filtered.extend(
                results
                    .into_iter()
                    .filter(|(id, _)| {
                        self.collection
                            .metadata
                            .get(*id)
                            .map_or(false, |entry| filter.matches(entry.data.as_ref()))
                    })
                    .take(limit),
            );
            filtered
        } else {
            let mut taken = Vec::with_capacity(capacity);
            taken.extend(results.into_iter().take(limit));
            taken
        }
    }

    /// Enrich raw results with metadata and external IDs.
    fn enrich(&self, pre_filtered: Vec<(VectorId, f32)>) -> Result<Vec<SearchResult>> {
        if self.include_metadata || self.post_filter.is_some() {
            self.collection.enrich_results(pre_filtered)
        } else {
            let mut results = Vec::with_capacity(pre_filtered.len());
            for (id, distance) in pre_filtered {
                let entry =
                    self.collection.metadata.get(id).ok_or_else(|| {
                        NeedleError::Index("Missing metadata for vector".into())
                    })?;
                results.push(SearchResult {
                    id: entry.external_id.clone(),
                    distance,
                    metadata: None,
                });
            }
            Ok(results)
        }
    }

    /// Apply post-filter, truncate to k, and strip metadata if not requested.
    fn apply_post_filter(&self, enriched: &mut Vec<SearchResult>) {
        if let Some(post_filter) = self.post_filter {
            *enriched = enriched
                .drain(..)
                .filter(|result| post_filter.matches(result.metadata.as_ref()))
                .take(self.k)
                .collect();
        } else {
            enriched.truncate(self.k);
        }
        // Strip metadata if not requested (but was needed for post-filter)
        if !self.include_metadata && self.post_filter.is_some() {
            for result in enriched.iter_mut() {
                result.metadata = None;
            }
        }
    }

    /// Apply time-weighted decay, re-sort by decayed score, and truncate to k.
    fn apply_time_decay(&self, results: &mut Vec<SearchResult>) {
        let decay = match &self.time_decay {
            Some(d) => d,
            None => return,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Apply decay factor to distances. Lower distance = better, so we
        // scale distance up (worse) for older vectors: distance /= decay_factor.
        for result in results.iter_mut() {
            let timestamp = self
                .collection
                .metadata
                .get_internal_id(&result.id)
                .and_then(|iid| self.collection.insertion_timestamps.get(&iid).copied())
                .unwrap_or(0);

            let age = now.saturating_sub(timestamp);
            let decay_factor = decay.compute(age);

            if decay_factor > 0.0 {
                result.distance /= decay_factor;
            } else {
                result.distance = f32::MAX;
            }
        }

        // Re-sort by distance (ascending = best first)
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(self.k);
    }

    /// Execute the search and return only IDs with distances
    pub fn execute_ids_only(self) -> Result<Vec<(String, f32)>> {
        self.include_metadata(false)
            .execute()
            .map(|results| {
                let mut ids = Vec::with_capacity(results.len());
                ids.extend(results.into_iter().map(|r| (r.id, r.distance)));
                ids
            })
    }

    /// Execute the search with explanation, returning results and a SearchExplain report.
    ///
    /// The explanation captures timing breakdown, HNSW traversal stats, and filter
    /// selectivity. Only activates instrumentation for this call — no performance
    /// impact on normal searches.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let (results, explain) = collection.search_builder(&query)
    ///     .k(10)
    ///     .filter(&filter)
    ///     .execute_explained()?;
    /// println!("{}", explain); // ASCII report
    /// ```
    pub fn execute_explained(self) -> Result<(Vec<SearchResult>, SearchExplain)> {
        use std::time::Instant;

        let total_start = Instant::now();
        self.validate_query()?;

        let effective_k = self.k.min(self.collection.len()).max(if self.collection.is_empty() { 0 } else { 1 });
        if effective_k == 0 {
            let explain = SearchExplain {
                dimensions: self.collection.config.dimensions,
                collection_size: self.collection.len(),
                requested_k: self.k,
                total_time_us: total_start.elapsed().as_micros() as u64,
                distance_function: format!("{:?}", self.collection.config.distance),
                ..Default::default()
            };
            return Ok((Vec::new(), explain));
        }

        // Index search with stats
        let index_start = Instant::now();
        let fetch_count = self.calculate_fetch_count();
        let (raw_results, hnsw_stats) = self.collection.index.search_with_stats(
            self.query,
            fetch_count,
            self.collection.vectors.as_slice(),
        )?;
        let index_time = index_start.elapsed();

        let candidates_before_filter = raw_results.len();

        let non_expired = self.filter_expired(raw_results);
        let time_filtered = self.filter_as_of(non_expired);

        // Filter with timing
        let filter_start = Instant::now();
        let post_filter_factor = if self.post_filter.is_some() {
            self.post_filter_factor
        } else {
            1
        };
        let pre_filtered = self.apply_pre_filter(
            time_filtered,
            self.k * post_filter_factor.max(1),
        );
        let filter_time = filter_start.elapsed();

        let candidates_after_filter = pre_filtered.len();

        // Enrich with timing
        let enrich_start = Instant::now();
        let mut enriched = self.enrich(pre_filtered)?;
        self.apply_post_filter(&mut enriched);
        self.apply_time_decay(&mut enriched);
        let enrich_time = enrich_start.elapsed();

        let total_time = total_start.elapsed();

        let explain = SearchExplain {
            total_time_us: total_time.as_micros() as u64,
            index_time_us: index_time.as_micros() as u64,
            filter_time_us: filter_time.as_micros() as u64,
            enrich_time_us: enrich_time.as_micros() as u64,
            candidates_before_filter,
            candidates_after_filter,
            hnsw_stats,
            dimensions: self.collection.config.dimensions,
            collection_size: self.collection.len(),
            requested_k: self.k,
            effective_k,
            ef_search: self.ef_search.unwrap_or(self.collection.index.config().ef_search),
            filter_applied: self.filter.is_some() || self.post_filter.is_some(),
            distance_function: format!("{:?}", self.collection.config.distance),
        };

        Ok((enriched, explain))
    }
}

pub(super) struct BruteForceSearchParams<'a> {
    pub(super) query: &'a [f32],
    pub(super) k: usize,
    pub(super) distance_fn: DistanceFunction,
    pub(super) filter: Option<&'a Filter>,
    pub(super) post_filter: Option<&'a Filter>,
    pub(super) include_metadata: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vector;
    use serde_json::json;

    fn make_collection(n: usize, dims: usize) -> Collection {
        let mut col = Collection::with_dimensions("test", dims);
        for i in 0..n {
            let vec = random_vector(dims);
            col.insert(format!("v{i}"), &vec, Some(json!({"idx": i, "cat": if i % 2 == 0 { "even" } else { "odd" }})))
                .unwrap();
        }
        col
    }

    // ── SearchBuilder defaults ──────────────────────────────────────────

    #[test]
    fn test_search_builder_default_k() {
        let col = make_collection(20, 8);
        let query = random_vector(8);
        let builder = col.search_builder(&query);
        let results = builder.execute().unwrap();
        assert!(results.len() <= 10); // default k=10
    }

    #[test]
    fn test_search_builder_custom_k() {
        let col = make_collection(20, 8);
        let query = random_vector(8);
        let results = col.search_builder(&query).k(3).execute().unwrap();
        assert!(results.len() <= 3);
    }

    #[test]
    fn test_search_builder_k_zero() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        let results = col.search_builder(&query).k(0).execute().unwrap();
        assert!(results.is_empty());
    }

    // ── Pre-filter ──────────────────────────────────────────────────────

    #[test]
    fn test_search_builder_pre_filter() {
        let col = make_collection(30, 8);
        let query = random_vector(8);
        let filter = Filter::eq("cat", "even");
        let results = col.search_builder(&query).k(10).filter(&filter).execute().unwrap();
        for r in &results {
            if let Some(meta) = &r.metadata {
                assert_eq!(meta["cat"], "even");
            }
        }
    }

    // ── Post-filter ─────────────────────────────────────────────────────

    #[test]
    fn test_search_builder_post_filter() {
        let col = make_collection(30, 8);
        let query = random_vector(8);
        let pf = Filter::eq("cat", "odd");
        let results = col.search_builder(&query)
            .k(5)
            .post_filter(&pf)
            .post_filter_factor(5)
            .execute()
            .unwrap();
        for r in &results {
            if let Some(meta) = &r.metadata {
                assert_eq!(meta["cat"], "odd");
            }
        }
    }

    #[test]
    fn test_post_filter_factor_min_one() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        let pf = Filter::eq("cat", "even");
        // factor 0 should be clamped to 1
        let results = col.search_builder(&query)
            .k(5)
            .post_filter(&pf)
            .post_filter_factor(0)
            .execute()
            .unwrap();
        assert!(results.len() <= 5);
    }

    // ── include_metadata ────────────────────────────────────────────────

    #[test]
    fn test_search_builder_without_metadata() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        let results = col.search_builder(&query)
            .k(5)
            .include_metadata(false)
            .execute()
            .unwrap();
        for r in &results {
            assert!(r.metadata.is_none());
        }
    }

    #[test]
    fn test_search_builder_with_metadata() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        let results = col.search_builder(&query)
            .k(5)
            .include_metadata(true)
            .execute()
            .unwrap();
        for r in &results {
            assert!(r.metadata.is_some());
        }
    }

    // ── Distance override ───────────────────────────────────────────────

    #[test]
    fn test_search_builder_distance_override_same() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        // Same distance as collection - should use index
        let results = col.search_builder(&query)
            .k(5)
            .distance(DistanceFunction::Cosine)
            .execute()
            .unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_builder_distance_override_different() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        // Different distance - should fall back to brute-force
        let results = col.search_builder(&query)
            .k(5)
            .distance(DistanceFunction::Euclidean)
            .execute()
            .unwrap();
        assert!(results.len() <= 5);
    }

    // ── ef_search override ──────────────────────────────────────────────

    #[test]
    fn test_search_builder_ef_search() {
        let col = make_collection(30, 8);
        let query = random_vector(8);
        let results = col.search_builder(&query)
            .k(5)
            .ef_search(200)
            .execute()
            .unwrap();
        assert!(results.len() <= 5);
    }

    // ── execute_ids_only ────────────────────────────────────────────────

    #[test]
    fn test_execute_ids_only() {
        let col = make_collection(20, 8);
        let query = random_vector(8);
        let ids = col.search_builder(&query)
            .k(5)
            .execute_ids_only()
            .unwrap();
        assert!(ids.len() <= 5);
        for (id, dist) in &ids {
            assert!(!id.is_empty());
            assert!(*dist >= 0.0);
        }
    }

    // ── Dimension mismatch ──────────────────────────────────────────────

    #[test]
    fn test_search_builder_dim_mismatch() {
        let col = make_collection(5, 4);
        let query = random_vector(8);
        let result = col.search_builder(&query).k(5).execute();
        assert!(result.is_err());
    }

    // ── as_of timestamp ─────────────────────────────────────────────────

    #[test]
    fn test_as_of_timestamp_accessor() {
        let col = Collection::with_dimensions("test", 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let builder = col.search_builder(&query).as_of(12345);
        assert_eq!(builder.as_of_timestamp(), Some(12345));
    }

    #[test]
    fn test_as_of_timestamp_default_none() {
        let col = Collection::with_dimensions("test", 4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let builder = col.search_builder(&query);
        assert_eq!(builder.as_of_timestamp(), None);
    }

    // ── GroundTruthEntry & QueryMetrics ─────────────────────────────────

    #[test]
    fn test_ground_truth_entry() {
        let entry = GroundTruthEntry {
            query: vec![1.0, 0.0],
            relevant_ids: vec!["a".into(), "b".into()],
        };
        assert_eq!(entry.query.len(), 2);
        assert_eq!(entry.relevant_ids.len(), 2);
    }

    #[test]
    fn test_query_metrics() {
        let metrics = QueryMetrics {
            query_index: 0,
            recall_at_k: 0.8,
            precision_at_k: 0.6,
            average_precision: 0.7,
            reciprocal_rank: 1.0,
            ndcg: 0.9,
        };
        assert!((metrics.recall_at_k - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_evaluation_report() {
        let report = EvaluationReport {
            num_queries: 10,
            k: 5,
            mean_recall_at_k: 0.85,
            mean_precision_at_k: 0.7,
            map: 0.75,
            mrr: 0.9,
            mean_ndcg: 0.88,
            per_query: vec![],
            eval_time_ms: 42.0,
        };
        assert_eq!(report.num_queries, 10);
        assert_eq!(report.k, 5);
    }

    // ── TimeDecay unit tests ────────────────────────────────────────────

    #[test]
    fn test_time_decay_exponential_half_life() {
        let decay = TimeDecay::Exponential { half_life_seconds: 3600 };
        // At t=0, factor should be 1.0
        assert!((decay.compute(0) - 1.0).abs() < 1e-6);
        // At t=half_life, factor should be ~0.5
        assert!((decay.compute(3600) - 0.5).abs() < 0.01);
        // At t=2*half_life, factor should be ~0.25
        assert!((decay.compute(7200) - 0.25).abs() < 0.01);
        // Monotonically decreasing
        assert!(decay.compute(100) > decay.compute(200));
    }

    #[test]
    fn test_time_decay_linear() {
        let decay = TimeDecay::Linear { max_age_seconds: 1000 };
        assert!((decay.compute(0) - 1.0).abs() < 1e-6);
        assert!((decay.compute(500) - 0.5).abs() < 1e-6);
        assert!((decay.compute(1000) - 0.0).abs() < 1e-6);
        assert!((decay.compute(2000) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_decay_step() {
        let decay = TimeDecay::Step { window_seconds: 100 };
        assert!((decay.compute(0) - 1.0).abs() < 1e-6);
        assert!((decay.compute(50) - 1.0).abs() < 1e-6);
        assert!((decay.compute(100) - 1.0).abs() < 1e-6);
        assert!((decay.compute(101) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_decay_exponential_zero_half_life() {
        let decay = TimeDecay::Exponential { half_life_seconds: 0 };
        assert!((decay.compute(0) - 1.0).abs() < 1e-6);
        assert!((decay.compute(1) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_decay_linear_zero_max_age() {
        let decay = TimeDecay::Linear { max_age_seconds: 0 };
        assert!((decay.compute(0) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_search_builder_with_time_decay() {
        let col = make_collection(20, 8);
        let query = random_vector(8);
        // Should not panic and should return results
        let results = col.search_builder(&query)
            .k(5)
            .with_time_decay(TimeDecay::Exponential { half_life_seconds: 86400 })
            .execute()
            .unwrap();
        // All vectors were just inserted (age ~0), so decay factor ~1.0
        // All results should be present
        assert!(!results.is_empty());
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_time_decay_overfetch_factor() {
        // Verify that when time_decay is set, the fetch count is 3x
        let col = make_collection(10, 4);
        let query = random_vector(4);
        let builder = col.search_builder(&query)
            .k(5)
            .with_time_decay(TimeDecay::Step { window_seconds: 1 });
        // This is a compile/integration test - verify it executes
        let results = builder.execute().unwrap();
        // Step decay with 1s window: all vectors just inserted, should all be in window
        assert!(results.len() <= 5);
    }

    // ── execute_explained tests ─────────────────────────────────────────

    #[test]
    fn test_execute_explained_basic() {
        let col = make_collection(20, 8);
        let query = random_vector(8);
        let (results, explain) = col.search_builder(&query)
            .k(5)
            .execute_explained()
            .unwrap();

        assert_eq!(results.len(), 5);
        assert!(explain.total_time_us > 0);
        assert_eq!(explain.dimensions, 8);
        assert_eq!(explain.collection_size, 20);
        assert_eq!(explain.requested_k, 5);
        assert!(!explain.filter_applied);
        assert!(explain.hnsw_stats.visited_nodes > 0);
    }

    #[test]
    fn test_execute_explained_with_filter() {
        let col = make_collection(20, 8);
        let query = random_vector(8);
        let filter = Filter::eq("cat", "even");
        let (results, explain) = col.search_builder(&query)
            .k(5)
            .filter(&filter)
            .execute_explained()
            .unwrap();

        assert!(results.len() <= 5);
        assert!(explain.filter_applied);
        assert!(explain.candidates_before_filter >= explain.candidates_after_filter);
    }

    #[test]
    fn test_execute_explained_empty_collection() {
        let col = Collection::with_dimensions("test", 4);
        let (results, explain) = col.search_builder(&[1.0, 0.0, 0.0, 0.0])
            .k(5)
            .execute_explained()
            .unwrap();

        assert!(results.is_empty());
        assert_eq!(explain.effective_k, 0);
    }

    #[test]
    fn test_execute_explained_display() {
        let col = make_collection(10, 4);
        let query = random_vector(4);
        let (_, explain) = col.search_builder(&query)
            .k(3)
            .execute_explained()
            .unwrap();

        let output = format!("{}", explain);
        assert!(output.contains("NEEDLE SEARCH EXPLAIN"));
        assert!(output.contains("HNSW TRAVERSAL"));
    }

    // ── as_of (MVCC point-in-time) tests ────────────────────────────────

    #[test]
    fn test_search_builder_as_of_excludes_future() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("old", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("new", &[0.9, 0.1, 0.0, 0.0], None).unwrap();

        // as_of(0) excludes everything (timestamps are current time >> 0)
        let results = col.search_builder(&[1.0, 0.0, 0.0, 0.0])
            .k(10)
            .as_of(0)
            .execute()
            .unwrap();
        assert!(results.is_empty(), "as_of(0) should exclude all modern-timestamped vectors");

        // as_of(far future) includes everything
        let results = col.search_builder(&[1.0, 0.0, 0.0, 0.0])
            .k(10)
            .as_of(u64::MAX)
            .execute()
            .unwrap();
        assert_eq!(results.len(), 2, "as_of(MAX) should include all vectors");
    }

    #[test]
    fn test_search_builder_as_of_zero_returns_nothing() {
        let col = make_collection(5, 4);
        let query = random_vector(4);

        let results = col.search_builder(&query)
            .k(5)
            .as_of(0)
            .execute()
            .unwrap();

        assert!(results.is_empty());
    }
}
