//! Search pipeline types for the collection module.
//!
//! Contains the `SearchBuilder` fluent API, search evaluation metrics,
//! and the brute-force search parameter type.

use super::Collection;
use crate::collection::search::SearchResult;
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
        self.k * pre_filter_factor * post_filter_factor
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
    fn apply_pre_filter(
        &self,
        results: Vec<(VectorId, f32)>,
        limit: usize,
    ) -> Vec<(VectorId, f32)> {
        let capacity = limit.min(results.len());
        if let Some(filter) = self.filter {
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
}

pub(super) struct BruteForceSearchParams<'a> {
    pub(super) query: &'a [f32],
    pub(super) k: usize,
    pub(super) distance_fn: DistanceFunction,
    pub(super) filter: Option<&'a Filter>,
    pub(super) post_filter: Option<&'a Filter>,
    pub(super) include_metadata: bool,
}
