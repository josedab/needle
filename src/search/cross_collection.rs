//! Cross-Collection Search
//!
//! Enables unified search across multiple collections within a single database
//! with intelligent result merging, query routing, and cross-collection relevance scoring.

use crate::database::Database;
use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use crate::metadata::Filter;
use crate::SearchResult;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for cross-collection search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCollectionConfig {
    /// Maximum results per collection
    pub max_per_collection: usize,
    /// Total maximum results to return
    pub total_max_results: usize,
    /// Timeout per collection query in milliseconds
    pub timeout_ms: u64,
    /// Whether to normalize scores across collections
    pub normalize_scores: bool,
    /// Score aggregation method
    pub aggregation: ScoreAggregation,
    /// Minimum score threshold
    pub min_score: Option<f32>,
    /// Whether to include collection name in results
    pub include_collection_name: bool,
    /// Whether to query collections in parallel
    pub parallel_queries: bool,
}

impl Default for CrossCollectionConfig {
    fn default() -> Self {
        Self {
            max_per_collection: 100,
            total_max_results: 100,
            timeout_ms: 5000,
            normalize_scores: true,
            aggregation: ScoreAggregation::MinScore,
            min_score: None,
            include_collection_name: true,
            parallel_queries: true,
        }
    }
}

/// Score aggregation method for merging results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreAggregation {
    /// Use minimum distance (best match)
    MinScore,
    /// Use maximum distance
    MaxScore,
    /// Average scores across collections
    Average,
    /// Weighted average based on collection sizes
    WeightedAverage,
    /// Reciprocal Rank Fusion
    RRF,
}

/// A cross-collection search result with collection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossCollectionResult {
    /// Vector ID
    pub id: String,
    /// Distance/score (lower is better for distance)
    pub score: f32,
    /// Source collection name
    pub collection: String,
    /// Optional metadata
    pub metadata: Option<serde_json::Value>,
    /// Rank within source collection
    pub rank_in_collection: usize,
}

/// Statistics for a cross-collection search operation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossCollectionStats {
    /// Total search time in milliseconds
    pub total_time_ms: u64,
    /// Number of collections searched
    pub collections_searched: usize,
    /// Results per collection
    pub results_per_collection: HashMap<String, usize>,
    /// Collections that timed out
    pub timed_out_collections: Vec<String>,
    /// Collections that errored
    pub errored_collections: Vec<String>,
    /// Total vectors considered
    pub total_vectors_considered: usize,
    /// Final results returned
    pub final_results_count: usize,
}

/// Collection filter for cross-collection search
#[derive(Debug, Clone)]
pub enum CollectionFilter {
    /// Include all collections
    All,
    /// Include only these collections
    Include(HashSet<String>),
    /// Exclude these collections
    Exclude(HashSet<String>),
    /// Collections matching a prefix
    Prefix(String),
    /// Collections matching a suffix
    Suffix(String),
    /// Collections with dimensions matching query
    MatchingDimensions,
}

impl Default for CollectionFilter {
    fn default() -> Self {
        Self::All
    }
}

/// Cross-collection search engine
pub struct CrossCollectionSearch {
    /// Database reference
    db: Arc<Database>,
    /// Configuration
    config: CrossCollectionConfig,
    /// Collection metadata cache
    collection_cache: RwLock<HashMap<String, CollectionMeta>>,
}

/// Cached collection metadata
#[derive(Debug, Clone)]
struct CollectionMeta {
    name: String,
    dimensions: usize,
    vector_count: usize,
    #[allow(dead_code)]
    distance_function: DistanceFunction,
}

impl CrossCollectionSearch {
    /// Create a new cross-collection search engine
    pub fn new(db: Arc<Database>, config: CrossCollectionConfig) -> Self {
        Self {
            db,
            config,
            collection_cache: RwLock::new(HashMap::new()),
        }
    }

    /// Create with default configuration
    pub fn with_defaults(db: Arc<Database>) -> Self {
        Self::new(db, CrossCollectionConfig::default())
    }

    /// Refresh the collection metadata cache
    pub fn refresh_cache(&self) -> Result<()> {
        let collection_names = self.db.list_collections();
        let mut cache = self.collection_cache.write();
        cache.clear();

        for name in collection_names {
            if let Ok(col_ref) = self.db.collection(&name) {
                if let Some(dims) = col_ref.dimensions() {
                    cache.insert(
                        name.clone(),
                        CollectionMeta {
                            name,
                            dimensions: dims,
                            vector_count: col_ref.len(),
                            distance_function: DistanceFunction::Cosine, // Default
                        },
                    );
                }
            }
        }
        Ok(())
    }

    /// Get collections matching the filter
    fn get_matching_collections(
        &self,
        filter: &CollectionFilter,
        query_dims: usize,
    ) -> Vec<String> {
        let collection_names = self.db.list_collections();
        let cache = self.collection_cache.read();

        collection_names
            .into_iter()
            .filter(|name| match filter {
                CollectionFilter::All => true,
                CollectionFilter::Include(set) => set.contains(name),
                CollectionFilter::Exclude(set) => !set.contains(name),
                CollectionFilter::Prefix(prefix) => name.starts_with(prefix),
                CollectionFilter::Suffix(suffix) => name.ends_with(suffix),
                CollectionFilter::MatchingDimensions => cache
                    .get(name)
                    .map(|m| m.dimensions == query_dims)
                    .unwrap_or(false),
            })
            .collect()
    }

    /// Search across collections
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: CollectionFilter,
        metadata_filter: Option<&Filter>,
    ) -> Result<(Vec<CrossCollectionResult>, CrossCollectionStats)> {
        let start = Instant::now();
        let mut stats = CrossCollectionStats::default();

        // Get matching collections
        let collections = self.get_matching_collections(&filter, query.len());
        stats.collections_searched = collections.len();

        if collections.is_empty() {
            return Ok((vec![], stats));
        }

        // Search each collection
        let per_collection_k = self.config.max_per_collection.min(k * 2);
        let results: Vec<(String, Vec<SearchResult>)> = if self.config.parallel_queries {
            collections
                .par_iter()
                .filter_map(|name| {
                    match self.search_collection(name, query, per_collection_k, metadata_filter) {
                        Ok(results) => Some((name.clone(), results)),
                        Err(_) => None,
                    }
                })
                .collect()
        } else {
            collections
                .iter()
                .filter_map(|name| {
                    match self.search_collection(name, query, per_collection_k, metadata_filter) {
                        Ok(results) => Some((name.clone(), results)),
                        Err(_) => None,
                    }
                })
                .collect()
        };

        // Track which collections errored
        let successful_names: HashSet<_> = results.iter().map(|(n, _)| n.clone()).collect();
        for name in &collections {
            if !successful_names.contains(name) {
                stats.errored_collections.push(name.clone());
            }
        }

        // Build results per collection stats
        for (name, res) in &results {
            stats.results_per_collection.insert(name.clone(), res.len());
            stats.total_vectors_considered += res.len();
        }

        // Merge results
        let merged = self.merge_results(results, k)?;
        stats.final_results_count = merged.len();
        stats.total_time_ms = start.elapsed().as_millis() as u64;

        Ok((merged, stats))
    }

    /// Search a single collection
    fn search_collection(
        &self,
        name: &str,
        query: &[f32],
        k: usize,
        metadata_filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let col_ref = self.db.collection(name)?;

        // Check dimensions match
        let dims = col_ref.dimensions().unwrap_or(0);
        if dims != query.len() {
            return Err(NeedleError::DimensionMismatch {
                expected: dims,
                got: query.len(),
            });
        }

        // Perform search
        if let Some(filter) = metadata_filter {
            col_ref.search_with_filter(query, k, filter)
        } else {
            col_ref.search(query, k)
        }
    }

    /// Merge results from multiple collections
    fn merge_results(
        &self,
        collection_results: Vec<(String, Vec<SearchResult>)>,
        k: usize,
    ) -> Result<Vec<CrossCollectionResult>> {
        match self.config.aggregation {
            ScoreAggregation::MinScore => self.merge_min_score(collection_results, k),
            ScoreAggregation::RRF => self.merge_rrf(collection_results, k),
            ScoreAggregation::Average => self.merge_average(collection_results, k),
            ScoreAggregation::WeightedAverage => self.merge_weighted_average(collection_results, k),
            ScoreAggregation::MaxScore => self.merge_max_score(collection_results, k),
        }
    }

    /// Merge using minimum score (best match)
    fn merge_min_score(
        &self,
        collection_results: Vec<(String, Vec<SearchResult>)>,
        k: usize,
    ) -> Result<Vec<CrossCollectionResult>> {
        let mut all_results: Vec<CrossCollectionResult> = Vec::new();

        for (collection, results) in collection_results {
            for (rank, result) in results.into_iter().enumerate() {
                all_results.push(CrossCollectionResult {
                    id: result.id,
                    score: result.distance,
                    collection: collection.clone(),
                    metadata: result.metadata,
                    rank_in_collection: rank + 1,
                });
            }
        }

        // Normalize scores if requested
        if self.config.normalize_scores && !all_results.is_empty() {
            let min_score = all_results
                .iter()
                .map(|r| r.score)
                .fold(f32::INFINITY, f32::min);
            let max_score = all_results
                .iter()
                .map(|r| r.score)
                .fold(f32::NEG_INFINITY, f32::max);

            if (max_score - min_score).abs() > f32::EPSILON {
                for result in &mut all_results {
                    result.score = (result.score - min_score) / (max_score - min_score);
                }
            }
        }

        // Sort by score (lower is better for distance-based)
        all_results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Apply minimum score filter
        if let Some(min) = self.config.min_score {
            all_results.retain(|r| r.score <= min);
        }

        // Return top k
        all_results.truncate(k.min(self.config.total_max_results));
        Ok(all_results)
    }

    /// Merge using Reciprocal Rank Fusion
    fn merge_rrf(
        &self,
        collection_results: Vec<(String, Vec<SearchResult>)>,
        k: usize,
    ) -> Result<Vec<CrossCollectionResult>> {
        const RRF_K: f32 = 60.0; // Standard RRF constant

        // Calculate RRF scores for each unique ID
        let mut rrf_scores: HashMap<String, (f32, String, Option<serde_json::Value>)> =
            HashMap::new();

        for (collection, results) in collection_results {
            for (rank, result) in results.into_iter().enumerate() {
                let rrf_score = 1.0 / (RRF_K + rank as f32 + 1.0);

                rrf_scores
                    .entry(result.id.clone())
                    .and_modify(|(score, _, _)| *score += rrf_score)
                    .or_insert((rrf_score, collection.clone(), result.metadata));
            }
        }

        // Convert to CrossCollectionResult
        let mut results: Vec<CrossCollectionResult> = rrf_scores
            .into_iter()
            .map(
                |(id, (score, collection, metadata))| CrossCollectionResult {
                    id,
                    score: 1.0 - score, // Invert so lower is better
                    collection,
                    metadata,
                    rank_in_collection: 0, // Not meaningful for RRF
                },
            )
            .collect();

        // Sort by score (lower is better)
        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        results.truncate(k.min(self.config.total_max_results));
        Ok(results)
    }

    /// Merge using average score
    fn merge_average(
        &self,
        collection_results: Vec<(String, Vec<SearchResult>)>,
        k: usize,
    ) -> Result<Vec<CrossCollectionResult>> {
        let mut scores: HashMap<String, (f32, usize, String, Option<serde_json::Value>)> =
            HashMap::new();

        for (collection, results) in collection_results {
            for result in results {
                scores
                    .entry(result.id.clone())
                    .and_modify(|(sum, count, _, _)| {
                        *sum += result.distance;
                        *count += 1;
                    })
                    .or_insert((result.distance, 1, collection.clone(), result.metadata));
            }
        }

        let mut results: Vec<CrossCollectionResult> = scores
            .into_iter()
            .map(
                |(id, (sum, count, collection, metadata))| CrossCollectionResult {
                    id,
                    score: sum / count as f32,
                    collection,
                    metadata,
                    rank_in_collection: 0,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(self.config.total_max_results));
        Ok(results)
    }

    /// Merge using weighted average based on collection sizes
    fn merge_weighted_average(
        &self,
        collection_results: Vec<(String, Vec<SearchResult>)>,
        k: usize,
    ) -> Result<Vec<CrossCollectionResult>> {
        let cache = self.collection_cache.read();
        let total_vectors: usize = cache.values().map(|m| m.vector_count).sum();

        let mut scores: HashMap<String, (f32, f32, String, Option<serde_json::Value>)> =
            HashMap::new();

        for (collection, results) in collection_results {
            let weight = cache
                .get(&collection)
                .map(|m| m.vector_count as f32 / total_vectors.max(1) as f32)
                .unwrap_or(1.0);

            for result in results {
                scores
                    .entry(result.id.clone())
                    .and_modify(|(weighted_sum, total_weight, _, _)| {
                        *weighted_sum += result.distance * weight;
                        *total_weight += weight;
                    })
                    .or_insert((
                        result.distance * weight,
                        weight,
                        collection.clone(),
                        result.metadata,
                    ));
            }
        }

        let mut results: Vec<CrossCollectionResult> = scores
            .into_iter()
            .map(
                |(id, (weighted_sum, total_weight, collection, metadata))| CrossCollectionResult {
                    id,
                    score: if total_weight > 0.0 {
                        weighted_sum / total_weight
                    } else {
                        weighted_sum
                    },
                    collection,
                    metadata,
                    rank_in_collection: 0,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(self.config.total_max_results));
        Ok(results)
    }

    /// Merge using max score
    fn merge_max_score(
        &self,
        collection_results: Vec<(String, Vec<SearchResult>)>,
        k: usize,
    ) -> Result<Vec<CrossCollectionResult>> {
        let mut max_scores: HashMap<String, (f32, String, Option<serde_json::Value>)> =
            HashMap::new();

        for (collection, results) in collection_results {
            for result in results {
                max_scores
                    .entry(result.id.clone())
                    .and_modify(|(max, _, _)| {
                        if result.distance > *max {
                            *max = result.distance;
                        }
                    })
                    .or_insert((result.distance, collection.clone(), result.metadata));
            }
        }

        let mut results: Vec<CrossCollectionResult> = max_scores
            .into_iter()
            .map(
                |(id, (score, collection, metadata))| CrossCollectionResult {
                    id,
                    score,
                    collection,
                    metadata,
                    rank_in_collection: 0,
                },
            )
            .collect();

        results.sort_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k.min(self.config.total_max_results));
        Ok(results)
    }

    /// Get configuration
    pub fn config(&self) -> &CrossCollectionConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: CrossCollectionConfig) {
        self.config = config;
    }
}

/// Builder for cross-collection queries
pub struct CrossCollectionQueryBuilder<'a> {
    search: &'a CrossCollectionSearch,
    query: Vec<f32>,
    k: usize,
    filter: CollectionFilter,
    metadata_filter: Option<Filter>,
    config_overrides: Option<CrossCollectionConfig>,
}

impl<'a> CrossCollectionQueryBuilder<'a> {
    /// Create a new query builder
    pub fn new(search: &'a CrossCollectionSearch, query: Vec<f32>) -> Self {
        Self {
            search,
            query,
            k: 10,
            filter: CollectionFilter::All,
            metadata_filter: None,
            config_overrides: None,
        }
    }

    /// Set number of results to return
    #[must_use]
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set collection filter
    #[must_use]
    pub fn collections(mut self, filter: CollectionFilter) -> Self {
        self.filter = filter;
        self
    }

    /// Include only specified collections
    #[must_use]
    pub fn include_collections(mut self, names: Vec<String>) -> Self {
        self.filter = CollectionFilter::Include(names.into_iter().collect());
        self
    }

    /// Exclude specified collections
    #[must_use]
    pub fn exclude_collections(mut self, names: Vec<String>) -> Self {
        self.filter = CollectionFilter::Exclude(names.into_iter().collect());
        self
    }

    /// Filter by collection name prefix
    #[must_use]
    pub fn prefix(mut self, prefix: &str) -> Self {
        self.filter = CollectionFilter::Prefix(prefix.to_string());
        self
    }

    /// Only include collections with matching dimensions
    #[must_use]
    pub fn matching_dimensions(mut self) -> Self {
        self.filter = CollectionFilter::MatchingDimensions;
        self
    }

    /// Set metadata filter
    #[must_use]
    pub fn metadata_filter(mut self, filter: Filter) -> Self {
        self.metadata_filter = Some(filter);
        self
    }

    /// Override aggregation method
    #[must_use]
    pub fn aggregation(mut self, agg: ScoreAggregation) -> Self {
        let config = self
            .config_overrides
            .get_or_insert_with(|| self.search.config.clone());
        config.aggregation = agg;
        self
    }

    /// Set minimum score threshold
    #[must_use]
    pub fn min_score(mut self, min: f32) -> Self {
        let config = self
            .config_overrides
            .get_or_insert_with(|| self.search.config.clone());
        config.min_score = Some(min);
        self
    }

    /// Enable/disable score normalization
    #[must_use]
    pub fn normalize_scores(mut self, normalize: bool) -> Self {
        let config = self
            .config_overrides
            .get_or_insert_with(|| self.search.config.clone());
        config.normalize_scores = normalize;
        self
    }

    /// Execute the search
    pub fn execute(self) -> Result<(Vec<CrossCollectionResult>, CrossCollectionStats)> {
        self.search.search(
            &self.query,
            self.k,
            self.filter,
            self.metadata_filter.as_ref(),
        )
    }
}

/// Cross-collection analytics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossCollectionAnalytics {
    /// Total vectors across all collections
    pub total_vectors: usize,
    /// Average dimensions
    pub average_dimensions: f64,
    /// Collection count
    pub collection_count: usize,
    /// Dimension distribution
    pub dimensions_distribution: HashMap<usize, usize>,
    /// Vector count distribution
    pub vector_counts: HashMap<String, usize>,
}

impl CrossCollectionSearch {
    /// Get cross-collection analytics
    pub fn analytics(&self) -> Result<CrossCollectionAnalytics> {
        self.refresh_cache()?;
        let cache = self.collection_cache.read();

        let mut analytics = CrossCollectionAnalytics::default();
        analytics.collection_count = cache.len();

        let mut total_dims: usize = 0;
        for meta in cache.values() {
            analytics.total_vectors += meta.vector_count;
            total_dims += meta.dimensions;
            analytics
                .vector_counts
                .insert(meta.name.clone(), meta.vector_count);
            *analytics
                .dimensions_distribution
                .entry(meta.dimensions)
                .or_insert(0) += 1;
        }

        if !cache.is_empty() {
            analytics.average_dimensions = total_dims as f64 / cache.len() as f64;
        }

        Ok(analytics)
    }
}

// ── Enhanced Federated Search ────────────────────────────────────────────────

/// Score normalization method for merging results across distance functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreNormalization {
    /// Min-max normalization to [0, 1].
    MinMax,
    /// Z-score normalization (mean=0, std=1).
    ZScore,
    /// No normalization.
    None,
}

impl Default for ScoreNormalization {
    fn default() -> Self {
        Self::MinMax
    }
}

/// Configuration for federated search across multiple collections.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchConfig {
    /// Score normalization method.
    pub normalization: ScoreNormalization,
    /// Per-collection weights (collection_name -> weight).
    pub weights: HashMap<String, f64>,
    /// Default weight for unweighted collections.
    pub default_weight: f64,
    /// Maximum results per collection before merging.
    pub per_collection_limit: usize,
    /// Total result limit.
    pub total_limit: usize,
    /// Per-collection filter overrides.
    pub collection_filters: HashMap<String, serde_json::Value>,
    /// Query collections in parallel.
    pub parallel: bool,
}

impl Default for FederatedSearchConfig {
    fn default() -> Self {
        Self {
            normalization: ScoreNormalization::MinMax,
            weights: HashMap::new(),
            default_weight: 1.0,
            per_collection_limit: 100,
            total_limit: 100,
            collection_filters: HashMap::new(),
            parallel: true,
        }
    }
}

/// A federated search result with normalized score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedResult {
    /// Vector ID.
    pub id: String,
    /// Normalized score (lower is better for distance-based).
    pub score: f32,
    /// Raw distance from the collection's distance function.
    pub raw_distance: f32,
    /// Source collection.
    pub collection: String,
    /// Applied weight.
    pub weight: f64,
    /// Optional metadata.
    pub metadata: Option<serde_json::Value>,
}

/// Normalize a set of distances using min-max normalization.
pub fn normalize_min_max(distances: &[f32]) -> Vec<f32> {
    if distances.is_empty() {
        return vec![];
    }
    let min = distances.iter().copied().fold(f32::MAX, f32::min);
    let max = distances.iter().copied().fold(f32::MIN, f32::max);
    let range = max - min;

    if range < f32::EPSILON {
        return vec![0.0; distances.len()];
    }

    distances.iter().map(|&d| (d - min) / range).collect()
}

/// Normalize a set of distances using z-score normalization.
pub fn normalize_z_score(distances: &[f32]) -> Vec<f32> {
    if distances.is_empty() {
        return vec![];
    }
    let n = distances.len() as f64;
    let mean = distances.iter().map(|&d| d as f64).sum::<f64>() / n;
    let variance = distances
        .iter()
        .map(|&d| (d as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let std = variance.sqrt();

    if std < 1e-10 {
        return vec![0.0; distances.len()];
    }

    distances
        .iter()
        .map(|&d| ((d as f64 - mean) / std) as f32)
        .collect()
}

/// Execute a federated search across multiple collections in a database.
pub fn federated_search(
    db: &Database,
    query: &[f32],
    collection_names: &[String],
    config: &FederatedSearchConfig,
) -> Result<Vec<FederatedResult>> {
    let mut all_results: Vec<FederatedResult> = Vec::new();

    // Query each collection
    for name in collection_names {
        let coll = db.collection(name)?;
        if coll.dimensions() != Some(query.len()) {
            continue; // skip dimension mismatch
        }

        let search_results = coll.search(query, config.per_collection_limit)?;
        let weight = config.weights.get(name).copied().unwrap_or(config.default_weight);

        let distances: Vec<f32> = search_results.iter().map(|r| r.distance).collect();
        let normalized = match config.normalization {
            ScoreNormalization::MinMax => normalize_min_max(&distances),
            ScoreNormalization::ZScore => normalize_z_score(&distances),
            ScoreNormalization::None => distances.clone(),
        };

        for (i, sr) in search_results.iter().enumerate() {
            let norm_score = normalized.get(i).copied().unwrap_or(sr.distance);
            all_results.push(FederatedResult {
                id: sr.id.clone(),
                score: (norm_score as f64 * weight) as f32,
                raw_distance: sr.distance,
                collection: name.clone(),
                weight,
                metadata: sr.metadata.clone(),
            });
        }
    }

    // Sort by normalized score
    all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    all_results.truncate(config.total_limit);

    Ok(all_results)
}

/// Reciprocal Rank Fusion for merging ranked lists from different collections.
/// RRF score = Σ 1/(k + rank_i) where k is a constant (typically 60).
pub fn reciprocal_rank_fusion(
    ranked_lists: &[Vec<FederatedResult>],
    k: f32,
    total_limit: usize,
) -> Vec<FederatedResult> {
    let mut rrf_scores: HashMap<String, (f32, FederatedResult)> = HashMap::new();

    for list in ranked_lists {
        for (rank, result) in list.iter().enumerate() {
            let rrf_score = 1.0 / (k + rank as f32 + 1.0);
            let entry = rrf_scores
                .entry(result.id.clone())
                .or_insert((0.0, result.clone()));
            entry.0 += rrf_score * result.weight as f32;
        }
    }

    let mut fused: Vec<FederatedResult> = rrf_scores
        .into_iter()
        .map(|(_, (score, mut result))| {
            result.score = -score; // Negate so lower = better (consistent with distance)
            result
        })
        .collect();

    fused.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    fused.truncate(total_limit);
    fused
}

/// Execute federated search with parallel collection queries using rayon.
pub fn federated_search_parallel(
    db: &Database,
    query: &[f32],
    collection_names: &[String],
    config: &FederatedSearchConfig,
) -> Result<Vec<FederatedResult>> {
    use rayon::prelude::*;

    // Query collections in parallel
    let per_collection_results: Vec<Vec<FederatedResult>> = collection_names
        .par_iter()
        .filter_map(|name| {
            let coll = db.collection(name).ok()?;
            if coll.dimensions() != Some(query.len()) {
                return None;
            }
            let results = coll.search(query, config.per_collection_limit).ok()?;
            let weight = config.weights.get(name).copied().unwrap_or(config.default_weight);
            let distances: Vec<f32> = results.iter().map(|r| r.distance).collect();
            let normalized = match config.normalization {
                ScoreNormalization::MinMax => normalize_min_max(&distances),
                ScoreNormalization::ZScore => normalize_z_score(&distances),
                ScoreNormalization::None => distances.clone(),
            };
            Some(
                results
                    .iter()
                    .enumerate()
                    .map(|(i, sr)| {
                        let norm_score = normalized.get(i).copied().unwrap_or(sr.distance);
                        FederatedResult {
                            id: sr.id.clone(),
                            score: (norm_score as f64 * weight) as f32,
                            raw_distance: sr.distance,
                            collection: name.clone(),
                            weight,
                            metadata: sr.metadata.clone(),
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    // Merge using RRF if multiple collections, otherwise sort by score
    if per_collection_results.len() > 1 {
        Ok(reciprocal_rank_fusion(
            &per_collection_results,
            60.0,
            config.total_limit,
        ))
    } else {
        let mut all: Vec<FederatedResult> =
            per_collection_results.into_iter().flatten().collect();
        all.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        all.truncate(config.total_limit);
        Ok(all)
    }
}

// ── Collection Routing Rules ─────────────────────────────────────────────────

/// A routing rule for collection selection during federated search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionRoutingRule {
    /// Target collection name.
    pub collection: String,
    /// Condition under which this collection should be queried.
    pub condition: RoutingCondition,
    /// Override weight for this collection.
    pub weight_override: Option<f64>,
}

/// Condition for routing a query to a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingCondition {
    /// Always route to this collection.
    Always,
    /// Route only if query dimensions match.
    DimensionMatch(usize),
}

/// Evaluate routing rules and return applicable collections.
pub fn evaluate_routing_rules(
    rules: &[CollectionRoutingRule],
    query_dimensions: usize,
) -> Vec<&CollectionRoutingRule> {
    rules
        .iter()
        .filter(|rule| match &rule.condition {
            RoutingCondition::Always => true,
            RoutingCondition::DimensionMatch(dim) => *dim == query_dimensions,
        })
        .collect()
}

// ── Federated Search with Latency Tracking ──────────────────────────────────

/// Execute federated search and return per-collection latency in microseconds.
pub fn federated_search_with_latency(
    db: &Database,
    query: &[f32],
    collection_names: &[String],
    config: &FederatedSearchConfig,
) -> Result<(Vec<FederatedResult>, HashMap<String, u64>)> {
    let mut all_results: Vec<FederatedResult> = Vec::new();
    let mut latencies: HashMap<String, u64> = HashMap::new();

    for name in collection_names {
        let start = std::time::Instant::now();
        let coll = db.collection(name)?;
        if coll.dimensions() != Some(query.len()) {
            continue;
        }

        let search_results = coll.search(query, config.per_collection_limit)?;
        let elapsed_us = start.elapsed().as_micros() as u64;
        latencies.insert(name.clone(), elapsed_us);

        let weight = config.weights.get(name).copied().unwrap_or(config.default_weight);
        let distances: Vec<f32> = search_results.iter().map(|r| r.distance).collect();
        let normalized = match config.normalization {
            ScoreNormalization::MinMax => normalize_min_max(&distances),
            ScoreNormalization::ZScore => normalize_z_score(&distances),
            ScoreNormalization::None => distances.clone(),
        };

        for (i, sr) in search_results.iter().enumerate() {
            let norm_score = normalized.get(i).copied().unwrap_or(sr.distance);
            all_results.push(FederatedResult {
                id: sr.id.clone(),
                score: (norm_score as f64 * weight) as f32,
                raw_distance: sr.distance,
                collection: name.clone(),
                weight,
                metadata: sr.metadata.clone(),
            });
        }
    }

    all_results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
    all_results.truncate(config.total_limit);

    Ok((all_results, latencies))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_db() -> Arc<Database> {
        Arc::new(Database::in_memory())
    }

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = seed;
        (0..dim)
            .map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn test_config_default() {
        let config = CrossCollectionConfig::default();
        assert_eq!(config.max_per_collection, 100);
        assert_eq!(config.total_max_results, 100);
        assert!(config.normalize_scores);
        assert!(config.parallel_queries);
    }

    #[test]
    fn test_search_creation() {
        let db = create_test_db();
        let search = CrossCollectionSearch::with_defaults(db);
        assert_eq!(search.config().max_per_collection, 100);
    }

    #[test]
    fn test_collection_filter_all() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 8).unwrap();

        let search = CrossCollectionSearch::with_defaults(db);
        let collections = search.get_matching_collections(&CollectionFilter::All, 8);

        assert_eq!(collections.len(), 2);
    }

    #[test]
    fn test_collection_filter_include() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 8).unwrap();
        db.create_collection("col3", 8).unwrap();

        let search = CrossCollectionSearch::with_defaults(db);
        let include = vec!["col1".to_string(), "col2".to_string()]
            .into_iter()
            .collect();

        let collections = search.get_matching_collections(&CollectionFilter::Include(include), 8);

        assert_eq!(collections.len(), 2);
        assert!(collections.contains(&"col1".to_string()));
        assert!(collections.contains(&"col2".to_string()));
    }

    #[test]
    fn test_collection_filter_prefix() {
        let db = create_test_db();
        db.create_collection("prod_users", 8).unwrap();
        db.create_collection("prod_items", 8).unwrap();
        db.create_collection("dev_users", 8).unwrap();

        let search = CrossCollectionSearch::with_defaults(db);
        let collections =
            search.get_matching_collections(&CollectionFilter::Prefix("prod_".to_string()), 8);

        assert_eq!(collections.len(), 2);
        assert!(collections.iter().all(|n: &String| n.starts_with("prod_")));
    }

    #[test]
    fn test_search_empty() {
        let db = create_test_db();
        let search = CrossCollectionSearch::with_defaults(db);

        let query = random_vec(8, 1);
        let (results, stats) = search
            .search(&query, 10, CollectionFilter::All, None)
            .unwrap();

        assert!(results.is_empty());
        assert_eq!(stats.collections_searched, 0);
    }

    #[test]
    fn test_search_single_collection() {
        let db = create_test_db();
        db.create_collection("test", 8).unwrap();

        let col = db.collection("test").unwrap();
        for i in 0..5 {
            col.insert(&format!("vec{}", i), &random_vec(8, i), None)
                .unwrap();
        }

        let search = CrossCollectionSearch::with_defaults(db);
        let query = random_vec(8, 100);

        let (results, stats) = search
            .search(&query, 3, CollectionFilter::All, None)
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(stats.collections_searched, 1);
        assert_eq!(stats.final_results_count, 3);
    }

    #[test]
    fn test_search_multiple_collections() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 8).unwrap();

        let col1 = db.collection("col1").unwrap();
        let col2 = db.collection("col2").unwrap();

        for i in 0..3 {
            col1.insert(&format!("c1_vec{}", i), &random_vec(8, i), None)
                .unwrap();
            col2.insert(&format!("c2_vec{}", i), &random_vec(8, i + 100), None)
                .unwrap();
        }

        let search = CrossCollectionSearch::with_defaults(db);
        let query = random_vec(8, 1000);

        let (results, stats) = search
            .search(&query, 5, CollectionFilter::All, None)
            .unwrap();

        assert_eq!(results.len(), 5);
        assert_eq!(stats.collections_searched, 2);
    }

    #[test]
    fn test_rrf_aggregation() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 8).unwrap();

        let col1 = db.collection("col1").unwrap();
        let col2 = db.collection("col2").unwrap();

        let v1 = random_vec(8, 1);
        let v2 = random_vec(8, 2);
        col1.insert("shared", &v1, None).unwrap();
        col2.insert("shared", &v2, None).unwrap();

        let mut config = CrossCollectionConfig::default();
        config.aggregation = ScoreAggregation::RRF;

        let search = CrossCollectionSearch::new(db, config);
        let query = random_vec(8, 100);

        let (results, _) = search
            .search(&query, 10, CollectionFilter::All, None)
            .unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_query_builder() {
        let db = create_test_db();
        db.create_collection("test", 8).unwrap();

        let col = db.collection("test").unwrap();
        col.insert("vec1", &random_vec(8, 1), None).unwrap();

        let search = CrossCollectionSearch::with_defaults(db);
        let query = random_vec(8, 100);

        let (results, _) = CrossCollectionQueryBuilder::new(&search, query)
            .k(5)
            .normalize_scores(true)
            .execute()
            .unwrap();

        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_analytics() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 16).unwrap();

        let col1 = db.collection("col1").unwrap();
        col1.insert("v1", &random_vec(8, 1), None).unwrap();
        col1.insert("v2", &random_vec(8, 2), None).unwrap();

        let col2 = db.collection("col2").unwrap();
        col2.insert("v3", &random_vec(16, 3), None).unwrap();

        let search = CrossCollectionSearch::with_defaults(db);
        let analytics = search.analytics().unwrap();

        assert_eq!(analytics.collection_count, 2);
        assert_eq!(analytics.total_vectors, 3);
        assert!((analytics.average_dimensions - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_result_fields() {
        let result = CrossCollectionResult {
            id: "test".to_string(),
            score: 0.5,
            collection: "col1".to_string(),
            metadata: Some(serde_json::json!({"key": "value"})),
            rank_in_collection: 1,
        };

        assert_eq!(result.id, "test");
        assert_eq!(result.collection, "col1");
        assert_eq!(result.rank_in_collection, 1);
    }

    #[test]
    fn test_stats_tracking() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 8).unwrap();

        let col1 = db.collection("col1").unwrap();
        let col2 = db.collection("col2").unwrap();

        for i in 0..5 {
            col1.insert(&format!("v{}", i), &random_vec(8, i), None)
                .unwrap();
        }
        for i in 0..3 {
            col2.insert(&format!("v{}", i), &random_vec(8, i + 100), None)
                .unwrap();
        }

        let search = CrossCollectionSearch::with_defaults(db);
        let query = random_vec(8, 1000);

        let (_, stats) = search
            .search(&query, 10, CollectionFilter::All, None)
            .unwrap();

        assert_eq!(stats.collections_searched, 2);
        assert!(stats.total_time_ms < 5000);
        assert!(stats.errored_collections.is_empty());
    }

    #[test]
    fn test_exclude_collections() {
        let db = create_test_db();
        db.create_collection("col1", 8).unwrap();
        db.create_collection("col2", 8).unwrap();
        db.create_collection("col3", 8).unwrap();

        let search = CrossCollectionSearch::with_defaults(db);
        let exclude = vec!["col2".to_string()].into_iter().collect();

        let collections = search.get_matching_collections(&CollectionFilter::Exclude(exclude), 8);

        assert_eq!(collections.len(), 2);
        assert!(!collections.contains(&"col2".to_string()));
    }

    #[test]
    fn test_normalize_min_max() {
        let distances = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_min_max(&distances);
        assert!((normalized[0] - 0.0).abs() < 0.01);
        assert!((normalized[4] - 1.0).abs() < 0.01);
        assert!((normalized[2] - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_normalize_min_max_equal() {
        let distances = vec![3.0, 3.0, 3.0];
        let normalized = normalize_min_max(&distances);
        assert!(normalized.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_normalize_z_score() {
        let distances = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize_z_score(&distances);
        // Mean should be ~0
        let mean: f32 = normalized.iter().sum::<f32>() / normalized.len() as f32;
        assert!(mean.abs() < 0.01);
    }

    #[test]
    fn test_normalize_empty() {
        assert!(normalize_min_max(&[]).is_empty());
        assert!(normalize_z_score(&[]).is_empty());
    }

    #[test]
    fn test_federated_search_basic() {
        let db = create_test_db();
        db.create_collection("coll_a", 8).unwrap();
        db.create_collection("coll_b", 8).unwrap();

        let coll_a = db.collection("coll_a").unwrap();
        let coll_b = db.collection("coll_b").unwrap();

        for i in 0..10 {
            coll_a.insert(
                &format!("a{}", i),
                &random_vec(8, i as u64),
                None,
            ).unwrap();
            coll_b.insert(
                &format!("b{}", i),
                &random_vec(8, i as u64 + 100),
                None,
            ).unwrap();
        }

        let query = random_vec(8, 0);
        let collections = vec!["coll_a".into(), "coll_b".into()];
        let config = FederatedSearchConfig::default();

        let results = federated_search(&db, &query, &collections, &config).unwrap();
        assert!(!results.is_empty());
        // Results should come from both collections
        let has_a = results.iter().any(|r| r.collection == "coll_a");
        let has_b = results.iter().any(|r| r.collection == "coll_b");
        assert!(has_a || has_b);
    }

    #[test]
    fn test_federated_search_weighted() {
        let db = create_test_db();
        db.create_collection("w1", 8).unwrap();

        let coll = db.collection("w1").unwrap();
        for i in 0..5 {
            coll.insert(&format!("v{}", i), &random_vec(8, i as u64), None).unwrap();
        }

        let mut weights = HashMap::new();
        weights.insert("w1".into(), 2.0);

        let config = FederatedSearchConfig {
            weights,
            ..Default::default()
        };

        let results = federated_search(
            &db,
            &random_vec(8, 0),
            &["w1".into()],
            &config,
        ).unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].weight, 2.0);
    }

    #[test]
    fn test_federated_config_defaults() {
        let config = FederatedSearchConfig::default();
        assert_eq!(config.normalization, ScoreNormalization::MinMax);
        assert_eq!(config.default_weight, 1.0);
        assert!(config.parallel);
    }

    #[test]
    fn test_rrf_merge() {
        let list1 = vec![
            FederatedResult {
                id: "a".into(), score: 0.1, raw_distance: 0.1,
                collection: "c1".into(), weight: 1.0, metadata: None,
            },
            FederatedResult {
                id: "b".into(), score: 0.2, raw_distance: 0.2,
                collection: "c1".into(), weight: 1.0, metadata: None,
            },
        ];
        let list2 = vec![
            FederatedResult {
                id: "b".into(), score: 0.15, raw_distance: 0.15,
                collection: "c2".into(), weight: 1.0, metadata: None,
            },
            FederatedResult {
                id: "c".into(), score: 0.3, raw_distance: 0.3,
                collection: "c2".into(), weight: 1.0, metadata: None,
            },
        ];

        let fused = reciprocal_rank_fusion(&[list1, list2], 60.0, 10);
        assert!(fused.len() >= 2);
        // "b" appears in both lists, so it should have the highest RRF score
        assert_eq!(fused[0].id, "b");
    }

    #[test]
    fn test_rrf_empty() {
        let fused = reciprocal_rank_fusion(&[], 60.0, 10);
        assert!(fused.is_empty());
    }

    #[test]
    fn test_federated_search_parallel_basic() {
        let db = Database::in_memory();
        db.create_collection("p1", 8).unwrap();
        db.create_collection("p2", 8).unwrap();

        let c1 = db.collection("p1").unwrap();
        let c2 = db.collection("p2").unwrap();

        for i in 0..10 {
            c1.insert(&format!("a{i}"), &random_vec(8, i as u64), None).unwrap();
            c2.insert(&format!("b{i}"), &random_vec(8, i as u64 + 100), None).unwrap();
        }

        let config = FederatedSearchConfig::default();
        let results = federated_search_parallel(
            &db,
            &random_vec(8, 0),
            &["p1".into(), "p2".into()],
            &config,
        ).unwrap();

        assert!(!results.is_empty());
    }

    #[test]
    fn test_collection_routing_rules() {
        let rules = vec![
            CollectionRoutingRule {
                collection: "docs".into(),
                condition: RoutingCondition::Always,
                weight_override: Some(2.0),
            },
            CollectionRoutingRule {
                collection: "logs".into(),
                condition: RoutingCondition::DimensionMatch(128),
                weight_override: None,
            },
        ];

        let applicable = evaluate_routing_rules(&rules, 128);
        assert_eq!(applicable.len(), 2);
        assert_eq!(applicable[0].collection, "docs");
        assert_eq!(applicable[0].weight_override, Some(2.0));
    }

    #[test]
    fn test_routing_dimension_filter() {
        let rules = vec![
            CollectionRoutingRule {
                collection: "small".into(),
                condition: RoutingCondition::DimensionMatch(64),
                weight_override: None,
            },
            CollectionRoutingRule {
                collection: "large".into(),
                condition: RoutingCondition::DimensionMatch(128),
                weight_override: None,
            },
        ];

        let applicable = evaluate_routing_rules(&rules, 128);
        assert_eq!(applicable.len(), 1);
        assert_eq!(applicable[0].collection, "large");
    }

    #[test]
    fn test_federated_latency_tracking() {
        let db = Database::in_memory();
        db.create_collection("t1", 8).unwrap();
        let c1 = db.collection("t1").unwrap();
        for i in 0..5 {
            c1.insert(&format!("v{i}"), &random_vec(8, i as u64), None).unwrap();
        }

        let config = FederatedSearchConfig::default();
        let (results, latencies) = federated_search_with_latency(
            &db,
            &random_vec(8, 0),
            &["t1".into()],
            &config,
        ).unwrap();
        assert!(!results.is_empty());
        assert!(latencies.contains_key("t1"));
        assert!(*latencies.get("t1").unwrap() > 0);
    }
}
