//! Matryoshka Embedding Support
//!
//! Support for Matryoshka Representation Learning (MRL) embeddings that allow
//! variable-dimension embeddings with graceful quality degradation.
//!
//! # Overview
//!
//! Matryoshka embeddings are trained to be useful at multiple dimension prefixes.
//! For example, a 768-dim embedding can be truncated to 256 or 384 dimensions
//! while maintaining reasonable quality, enabling:
//!
//! - **Memory savings**: Store truncated embeddings (4x smaller at 256 vs 1024)
//! - **Coarse-to-fine search**: Fast candidate retrieval at low dims, rerank at full dims
//! - **Adaptive quality**: Choose dimension based on accuracy/latency tradeoff
//!
//! # Supported Dimension Tiers
//!
//! Common Matryoshka dimension tiers:
//! - 64 (ultrafast, ~6% of full quality)
//! - 128 (fast, ~25% storage)
//! - 256 (balanced, ~50% storage)
//! - 384 (high quality)
//! - 512 (very high quality)
//! - 768/1024 (full quality)
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::matryoshka::{MatryoshkaIndex, MatryoshkaConfig, DimensionTier};
//!
//! let config = MatryoshkaConfig::new(768)
//!     .with_tiers(vec![64, 128, 256, 384, 768])
//!     .with_search_strategy(SearchStrategy::CoarseToFine);
//!
//! let index = MatryoshkaIndex::new(config);
//!
//! // Insert full-dimension embedding
//! index.insert("doc1", &embedding_768d, metadata)?;
//!
//! // Search with adaptive dimensions
//! let results = index.search(&query, 10)?;  // Auto-selects best tier
//!
//! // Search at specific dimension
//! let fast_results = index.search_at_dimension(&query, 10, 128)?;
//! ```

use crate::distance::DistanceFunction;
use crate::error::{NeedleError, Result};
use ordered_float::OrderedFloat;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

/// Standard Matryoshka dimension tiers
pub const STANDARD_TIERS: &[usize] = &[64, 128, 256, 384, 512, 768, 1024];

/// Search strategy for Matryoshka embeddings
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Use full dimensions for all searches
    FullDimensions,
    /// Use lowest dimension for speed
    FastSearch,
    /// Coarse-to-fine: search at low dim, rerank at high dim
    CoarseToFine {
        /// Dimension for initial coarse search
        coarse_dim: usize,
        /// Dimension for fine reranking
        fine_dim: usize,
        /// Candidate multiplier (fetch this many more candidates for reranking)
        candidate_multiplier: usize,
    },
    /// Adaptive based on query latency budget
    Adaptive {
        /// Target latency in milliseconds
        target_latency_ms: u32,
    },
    /// Use specific dimension
    Fixed { dimension: usize },
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::CoarseToFine {
            coarse_dim: 128,
            fine_dim: 768,
            candidate_multiplier: 4,
        }
    }
}

/// Configuration for Matryoshka index
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaConfig {
    /// Full embedding dimensions
    pub full_dimensions: usize,
    /// Available dimension tiers (sorted ascending)
    pub dimension_tiers: Vec<usize>,
    /// Search strategy
    pub search_strategy: SearchStrategy,
    /// Distance function
    pub distance_function: DistanceFunction,
    /// Store truncated versions for faster search
    pub store_truncated: bool,
    /// Tiers to store (subset of dimension_tiers)
    pub stored_tiers: Vec<usize>,
    /// Quality threshold for tier selection (0-1)
    pub quality_threshold: f32,
}

impl MatryoshkaConfig {
    /// Create config with full dimensions
    pub fn new(full_dimensions: usize) -> Self {
        let tiers: Vec<usize> = STANDARD_TIERS
            .iter()
            .copied()
            .filter(|&d| d <= full_dimensions)
            .collect();

        let stored_tiers = if full_dimensions >= 768 {
            vec![128, 384, full_dimensions]
        } else {
            vec![full_dimensions]
        };

        Self {
            full_dimensions,
            dimension_tiers: tiers,
            search_strategy: SearchStrategy::default(),
            distance_function: DistanceFunction::Cosine,
            store_truncated: true,
            stored_tiers,
            quality_threshold: 0.9,
        }
    }

    /// Set dimension tiers
    pub fn with_tiers(mut self, tiers: Vec<usize>) -> Self {
        let mut tiers = tiers;
        tiers.sort();
        tiers.retain(|&d| d <= self.full_dimensions);
        self.dimension_tiers = tiers;
        self
    }

    /// Set search strategy
    pub fn with_search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.search_strategy = strategy;
        self
    }

    /// Set stored tiers
    pub fn with_stored_tiers(mut self, tiers: Vec<usize>) -> Self {
        self.stored_tiers = tiers;
        self
    }

    /// Set distance function
    pub fn with_distance(mut self, distance: DistanceFunction) -> Self {
        self.distance_function = distance;
        self
    }

    /// Get the closest available tier for a dimension
    pub fn closest_tier(&self, dim: usize) -> usize {
        self.dimension_tiers
            .iter()
            .copied()
            .filter(|&d| d <= dim)
            .max()
            .unwrap_or(self.dimension_tiers[0])
    }

    /// Get recommended tier for given quality requirement
    pub fn tier_for_quality(&self, min_quality: f32) -> usize {
        // Empirical quality vs dimension mapping
        for &tier in &self.dimension_tiers {
            let quality = self.estimated_quality(tier);
            if quality >= min_quality {
                return tier;
            }
        }
        self.full_dimensions
    }

    /// Estimate quality retention for a dimension tier
    pub fn estimated_quality(&self, dim: usize) -> f32 {
        // Empirical formula based on Matryoshka paper
        let ratio = dim as f32 / self.full_dimensions as f32;
        // Quality follows roughly: q = ratio^0.3 for well-trained MRL models
        ratio.powf(0.3)
    }
}

/// A stored Matryoshka embedding with multiple tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatryoshkaEmbedding {
    /// Full embedding
    pub full: Vec<f32>,
    /// Truncated embeddings by dimension
    pub truncated: HashMap<usize, Vec<f32>>,
}

impl MatryoshkaEmbedding {
    /// Create from full embedding
    pub fn new(full: Vec<f32>, stored_tiers: &[usize]) -> Self {
        let full_dim = full.len();
        let mut truncated = HashMap::new();

        for &tier in stored_tiers {
            if tier < full_dim {
                truncated.insert(tier, Self::truncate(&full, tier));
            }
        }

        Self { full, truncated }
    }

    /// Get embedding at specific dimension
    pub fn at_dimension(&self, dim: usize) -> Vec<f32> {
        if dim >= self.full.len() {
            return self.full.clone();
        }

        // Check cached truncated versions
        if let Some(cached) = self.truncated.get(&dim) {
            return cached.clone();
        }

        // Find closest cached tier
        let closest = self.truncated.keys().copied().filter(|&d| d >= dim).min();

        match closest {
            Some(cached_dim) => Self::truncate(&self.truncated[&cached_dim], dim),
            None => Self::truncate(&self.full, dim),
        }
    }

    /// Truncate embedding to dimension
    pub fn truncate(embedding: &[f32], dim: usize) -> Vec<f32> {
        let truncated: Vec<f32> = embedding.iter().take(dim).copied().collect();
        // Re-normalize after truncation
        Self::normalize(&truncated)
    }

    /// Normalize vector to unit length
    fn normalize(vec: &[f32]) -> Vec<f32> {
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            vec.iter().map(|x| x / norm).collect()
        } else {
            vec.to_vec()
        }
    }

    /// Get full dimensions
    pub fn dimensions(&self) -> usize {
        self.full.len()
    }

    /// Get available tiers
    pub fn available_tiers(&self) -> Vec<usize> {
        let mut tiers: Vec<_> = self.truncated.keys().copied().collect();
        tiers.push(self.full.len());
        tiers.sort();
        tiers
    }
}

/// Search result from Matryoshka index
#[derive(Debug, Clone)]
pub struct MatryoshkaSearchResult {
    /// Vector ID
    pub id: String,
    /// Distance at search dimension
    pub coarse_distance: f32,
    /// Distance at full dimension (if reranked)
    pub fine_distance: Option<f32>,
    /// Final distance used for ranking
    pub final_distance: f32,
    /// Metadata
    pub metadata: Option<Value>,
    /// Dimension used for search
    pub search_dimension: usize,
    /// Whether result was reranked
    pub reranked: bool,
}

/// Statistics for Matryoshka index
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MatryoshkaStats {
    /// Total vectors
    pub total_vectors: usize,
    /// Searches performed
    pub searches: u64,
    /// Searches by dimension tier
    pub searches_by_tier: HashMap<usize, u64>,
    /// Average search time in microseconds
    pub avg_search_time_us: u64,
    /// Rerank count
    pub reranks: u64,
    /// Memory usage estimate in bytes
    pub memory_bytes: u64,
}

/// Matryoshka embedding index
pub struct MatryoshkaIndex {
    config: MatryoshkaConfig,
    /// Vectors by ID
    vectors: RwLock<HashMap<String, MatryoshkaEmbedding>>,
    /// Metadata by ID
    metadata: RwLock<HashMap<String, Value>>,
    /// Statistics
    stats: RwLock<MatryoshkaStats>,
    /// Search counter
    #[allow(dead_code)]
    search_count: AtomicU64,
}

impl MatryoshkaIndex {
    /// Create a new Matryoshka index
    pub fn new(config: MatryoshkaConfig) -> Self {
        Self {
            config,
            vectors: RwLock::new(HashMap::new()),
            metadata: RwLock::new(HashMap::new()),
            stats: RwLock::new(MatryoshkaStats::default()),
            search_count: AtomicU64::new(0),
        }
    }

    /// Insert a vector
    pub fn insert(
        &self,
        id: impl Into<String>,
        embedding: &[f32],
        metadata: Option<Value>,
    ) -> Result<()> {
        if embedding.len() != self.config.full_dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.full_dimensions,
                got: embedding.len(),
            });
        }

        let id = id.into();
        let mrl = MatryoshkaEmbedding::new(embedding.to_vec(), &self.config.stored_tiers);

        self.vectors.write().insert(id.clone(), mrl);
        if let Some(meta) = metadata {
            self.metadata.write().insert(id, meta);
        }

        self.stats.write().total_vectors += 1;
        Ok(())
    }

    /// Search with default strategy
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<MatryoshkaSearchResult>> {
        match self.config.search_strategy {
            SearchStrategy::FullDimensions => {
                self.search_at_dimension(query, k, self.config.full_dimensions)
            }
            SearchStrategy::FastSearch => {
                let min_tier = *self
                    .config
                    .dimension_tiers
                    .first()
                    .unwrap_or(&self.config.full_dimensions);
                self.search_at_dimension(query, k, min_tier)
            }
            SearchStrategy::CoarseToFine {
                coarse_dim,
                fine_dim,
                candidate_multiplier,
            } => self.search_coarse_to_fine(query, k, coarse_dim, fine_dim, candidate_multiplier),
            SearchStrategy::Adaptive { target_latency_ms } => {
                self.search_adaptive(query, k, target_latency_ms)
            }
            SearchStrategy::Fixed { dimension } => self.search_at_dimension(query, k, dimension),
        }
    }

    /// Search at specific dimension
    pub fn search_at_dimension(
        &self,
        query: &[f32],
        k: usize,
        dim: usize,
    ) -> Result<Vec<MatryoshkaSearchResult>> {
        let start = Instant::now();

        // Truncate query if needed
        let search_query = if query.len() > dim {
            MatryoshkaEmbedding::truncate(query, dim)
        } else if query.len() < dim {
            return Err(NeedleError::DimensionMismatch {
                expected: dim,
                got: query.len(),
            });
        } else {
            query.to_vec()
        };

        let vectors = self.vectors.read();
        let metadata = self.metadata.read();

        let mut results: Vec<_> = vectors
            .iter()
            .map(|(id, mrl)| {
                let embedding = mrl.at_dimension(dim);
                let distance = self
                    .config
                    .distance_function
                    .compute(&search_query, &embedding);
                (id.clone(), distance)
            })
            .collect();

        // Sort by distance
        results.sort_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)));
        results.truncate(k);

        let search_results: Vec<_> = results
            .into_iter()
            .map(|(id, distance)| MatryoshkaSearchResult {
                id: id.clone(),
                coarse_distance: distance,
                fine_distance: None,
                final_distance: distance,
                metadata: metadata.get(&id).cloned(),
                search_dimension: dim,
                reranked: false,
            })
            .collect();

        // Update stats
        let elapsed = start.elapsed();
        self.update_stats(dim, elapsed.as_micros() as u64, false);

        Ok(search_results)
    }

    /// Coarse-to-fine search
    pub fn search_coarse_to_fine(
        &self,
        query: &[f32],
        k: usize,
        coarse_dim: usize,
        fine_dim: usize,
        candidate_multiplier: usize,
    ) -> Result<Vec<MatryoshkaSearchResult>> {
        let start = Instant::now();

        // Phase 1: Coarse search
        let candidates = k * candidate_multiplier;
        let coarse_query = MatryoshkaEmbedding::truncate(query, coarse_dim);
        let fine_query = if query.len() >= fine_dim {
            MatryoshkaEmbedding::truncate(query, fine_dim)
        } else {
            query.to_vec()
        };

        let vectors = self.vectors.read();
        let metadata_map = self.metadata.read();

        let mut coarse_results: Vec<_> = vectors
            .iter()
            .map(|(id, mrl)| {
                let embedding = mrl.at_dimension(coarse_dim);
                let distance = self
                    .config
                    .distance_function
                    .compute(&coarse_query, &embedding);
                (id.clone(), distance, mrl.clone())
            })
            .collect();

        coarse_results.sort_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)));
        coarse_results.truncate(candidates);

        // Phase 2: Fine rerank
        let mut fine_results: Vec<_> = coarse_results
            .into_iter()
            .map(|(id, coarse_dist, mrl)| {
                let fine_embedding = mrl.at_dimension(fine_dim);
                let fine_dist = self
                    .config
                    .distance_function
                    .compute(&fine_query, &fine_embedding);
                (id, coarse_dist, fine_dist)
            })
            .collect();

        fine_results.sort_by(|a, b| OrderedFloat(a.2).cmp(&OrderedFloat(b.2)));
        fine_results.truncate(k);

        let search_results: Vec<_> = fine_results
            .into_iter()
            .map(|(id, coarse_dist, fine_dist)| MatryoshkaSearchResult {
                id: id.clone(),
                coarse_distance: coarse_dist,
                fine_distance: Some(fine_dist),
                final_distance: fine_dist,
                metadata: metadata_map.get(&id).cloned(),
                search_dimension: fine_dim,
                reranked: true,
            })
            .collect();

        // Update stats
        let elapsed = start.elapsed();
        self.update_stats(coarse_dim, elapsed.as_micros() as u64, true);

        Ok(search_results)
    }

    /// Adaptive search based on latency budget
    pub fn search_adaptive(
        &self,
        query: &[f32],
        k: usize,
        target_latency_ms: u32,
    ) -> Result<Vec<MatryoshkaSearchResult>> {
        // Estimate dimension based on collection size and target latency
        let vector_count = self.vectors.read().len();

        // Simple heuristic: lower dims for larger collections or tighter latency
        let dim = if vector_count > 100000 || target_latency_ms < 10 {
            self.config.dimension_tiers.first().copied().unwrap_or(128)
        } else if vector_count > 10000 || target_latency_ms < 50 {
            self.config
                .dimension_tiers
                .get(1)
                .copied()
                .unwrap_or(self.config.dimension_tiers[0])
        } else {
            self.config.full_dimensions
        };

        self.search_at_dimension(query, k, dim)
    }

    /// Get vector by ID
    pub fn get(&self, id: &str) -> Option<MatryoshkaEmbedding> {
        self.vectors.read().get(id).cloned()
    }

    /// Get metadata by ID
    pub fn get_metadata(&self, id: &str) -> Option<Value> {
        self.metadata.read().get(id).cloned()
    }

    /// Delete vector by ID
    pub fn delete(&self, id: &str) -> bool {
        let removed = self.vectors.write().remove(id).is_some();
        if removed {
            self.metadata.write().remove(id);
            self.stats.write().total_vectors -= 1;
        }
        removed
    }

    /// Get statistics
    pub fn stats(&self) -> MatryoshkaStats {
        self.stats.read().clone()
    }

    /// Get configuration
    pub fn config(&self) -> &MatryoshkaConfig {
        &self.config
    }

    /// Get vector count
    pub fn len(&self) -> usize {
        self.vectors.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.read().is_empty()
    }

    /// Estimate memory usage
    pub fn memory_usage_bytes(&self) -> u64 {
        let vectors = self.vectors.read();
        let mut bytes: u64 = 0;

        for (id, mrl) in vectors.iter() {
            // ID string
            bytes += id.len() as u64;
            // Full embedding
            bytes += (mrl.full.len() * 4) as u64;
            // Truncated embeddings
            for (_, truncated) in &mrl.truncated {
                bytes += (truncated.len() * 4) as u64;
            }
        }

        bytes
    }

    /// Get available dimension tiers
    pub fn dimension_tiers(&self) -> &[usize] {
        &self.config.dimension_tiers
    }

    // === Private helpers ===

    fn update_stats(&self, dimension: usize, time_us: u64, reranked: bool) {
        let mut stats = self.stats.write();
        stats.searches += 1;
        *stats.searches_by_tier.entry(dimension).or_insert(0) += 1;
        stats.avg_search_time_us =
            (stats.avg_search_time_us * (stats.searches - 1) + time_us) / stats.searches;
        if reranked {
            stats.reranks += 1;
        }
    }
}

/// Builder for MatryoshkaIndex
pub struct MatryoshkaIndexBuilder {
    config: MatryoshkaConfig,
}

impl MatryoshkaIndexBuilder {
    /// Create builder with full dimensions
    pub fn new(full_dimensions: usize) -> Self {
        Self {
            config: MatryoshkaConfig::new(full_dimensions),
        }
    }

    /// Set dimension tiers
    pub fn tiers(mut self, tiers: Vec<usize>) -> Self {
        self.config = self.config.with_tiers(tiers);
        self
    }

    /// Set search strategy
    pub fn search_strategy(mut self, strategy: SearchStrategy) -> Self {
        self.config.search_strategy = strategy;
        self
    }

    /// Use coarse-to-fine search
    pub fn coarse_to_fine(mut self, coarse_dim: usize, fine_dim: usize, multiplier: usize) -> Self {
        self.config.search_strategy = SearchStrategy::CoarseToFine {
            coarse_dim,
            fine_dim,
            candidate_multiplier: multiplier,
        };
        self
    }

    /// Use fast search (lowest dimension)
    pub fn fast_search(mut self) -> Self {
        self.config.search_strategy = SearchStrategy::FastSearch;
        self
    }

    /// Set stored tiers
    pub fn store_tiers(mut self, tiers: Vec<usize>) -> Self {
        self.config.stored_tiers = tiers;
        self
    }

    /// Set distance function
    pub fn distance(mut self, distance: DistanceFunction) -> Self {
        self.config.distance_function = distance;
        self
    }

    /// Build the index
    pub fn build(self) -> MatryoshkaIndex {
        MatryoshkaIndex::new(self.config)
    }
}

/// Quality estimator for Matryoshka embeddings
pub struct QualityEstimator {
    /// Full dimensions
    full_dim: usize,
    /// Calibration samples (dim -> quality)
    calibration: HashMap<usize, f32>,
}

impl QualityEstimator {
    /// Create with default calibration
    pub fn new(full_dim: usize) -> Self {
        let mut calibration = HashMap::new();

        // Default quality estimates based on empirical data
        for &dim in STANDARD_TIERS {
            if dim <= full_dim {
                let ratio = dim as f32 / full_dim as f32;
                let quality = ratio.powf(0.3); // Empirical formula
                calibration.insert(dim, quality);
            }
        }

        Self {
            full_dim,
            calibration,
        }
    }

    /// Estimate quality at dimension
    pub fn estimate_quality(&self, dim: usize) -> f32 {
        if let Some(&quality) = self.calibration.get(&dim) {
            return quality;
        }

        // Interpolate
        let ratio = dim as f32 / self.full_dim as f32;
        ratio.powf(0.3).min(1.0)
    }

    /// Recommend dimension for target quality
    pub fn recommend_dimension(&self, target_quality: f32) -> usize {
        // q = ratio^0.3 => ratio = q^(1/0.3)
        let ratio = target_quality.powf(1.0 / 0.3);
        let dim = (ratio * self.full_dim as f32).ceil() as usize;

        // Round to nearest standard tier
        STANDARD_TIERS
            .iter()
            .copied()
            .filter(|&d| d <= self.full_dim)
            .min_by_key(|&d| (d as i32 - dim as i32).abs())
            .unwrap_or(self.full_dim)
    }

    /// Add calibration point from actual measurements
    pub fn add_calibration(&mut self, dim: usize, quality: f32) {
        self.calibration.insert(dim, quality);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_embedding(dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        let mut emb = Vec::with_capacity(dim);
        for _ in 0..dim {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            emb.push(((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0);
        }
        // Normalize
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut emb {
            *v /= norm;
        }
        emb
    }

    #[test]
    fn test_matryoshka_embedding() {
        let full = random_embedding(768, 42);
        let mrl = MatryoshkaEmbedding::new(full.clone(), &[128, 256, 384]);

        assert_eq!(mrl.dimensions(), 768);
        assert_eq!(mrl.available_tiers(), vec![128, 256, 384, 768]);

        // Test truncation
        let at_128 = mrl.at_dimension(128);
        assert_eq!(at_128.len(), 128);

        // Verify normalization
        let norm: f32 = at_128.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_config_tiers() {
        let config = MatryoshkaConfig::new(768);
        assert!(config.dimension_tiers.contains(&64));
        assert!(config.dimension_tiers.contains(&128));
        assert!(config.dimension_tiers.contains(&768));
        assert!(!config.dimension_tiers.contains(&1024));
    }

    #[test]
    fn test_estimated_quality() {
        let config = MatryoshkaConfig::new(768);

        let q_768 = config.estimated_quality(768);
        let q_384 = config.estimated_quality(384);
        let q_128 = config.estimated_quality(128);

        assert!(q_768 > q_384);
        assert!(q_384 > q_128);
        assert!((q_768 - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_index_insert_and_search() {
        let config =
            MatryoshkaConfig::new(384).with_search_strategy(SearchStrategy::FullDimensions);
        let index = MatryoshkaIndex::new(config);

        // Insert vectors
        for i in 0..10 {
            let emb = random_embedding(384, i as u64);
            index.insert(format!("vec_{}", i), &emb, None).unwrap();
        }

        assert_eq!(index.len(), 10);

        // Search
        let query = random_embedding(384, 0);
        let results = index.search(&query, 5).unwrap();

        assert_eq!(results.len(), 5);
        // First result should be vec_0 (same embedding)
        assert_eq!(results[0].id, "vec_0");
        assert!(results[0].final_distance < 0.01);
    }

    #[test]
    fn test_coarse_to_fine_search() {
        let config =
            MatryoshkaConfig::new(384).with_search_strategy(SearchStrategy::CoarseToFine {
                coarse_dim: 64,
                fine_dim: 384,
                candidate_multiplier: 3,
            });
        let index = MatryoshkaIndex::new(config);

        for i in 0..20 {
            let emb = random_embedding(384, i as u64);
            index.insert(format!("vec_{}", i), &emb, None).unwrap();
        }

        let query = random_embedding(384, 5);
        let results = index.search(&query, 3).unwrap();

        assert_eq!(results.len(), 3);
        assert!(results[0].reranked);
        assert!(results[0].fine_distance.is_some());
    }

    #[test]
    fn test_search_at_dimension() {
        let index = MatryoshkaIndex::new(MatryoshkaConfig::new(768));

        for i in 0..10 {
            let emb = random_embedding(768, i as u64);
            index.insert(format!("vec_{}", i), &emb, None).unwrap();
        }

        let query = random_embedding(768, 0);

        // Search at different dimensions
        let full_results = index.search_at_dimension(&query, 5, 768).unwrap();
        let truncated_results = index.search_at_dimension(&query, 5, 128).unwrap();

        assert_eq!(full_results.len(), 5);
        assert_eq!(truncated_results.len(), 5);

        // Both should find vec_0 first
        assert_eq!(full_results[0].id, "vec_0");
    }

    #[test]
    fn test_quality_estimator() {
        let estimator = QualityEstimator::new(768);

        let q_768 = estimator.estimate_quality(768);
        let q_256 = estimator.estimate_quality(256);

        assert!(q_768 > q_256);

        // Test recommendation
        let recommended = estimator.recommend_dimension(0.8);
        assert!(recommended <= 768);
        assert!(estimator.estimate_quality(recommended) >= 0.75);
    }

    #[test]
    fn test_builder() {
        let index = MatryoshkaIndexBuilder::new(512)
            .tiers(vec![64, 128, 256, 512])
            .coarse_to_fine(64, 512, 4)
            .store_tiers(vec![64, 256, 512])
            .build();

        assert_eq!(index.config().full_dimensions, 512);
        assert!(matches!(
            index.config().search_strategy,
            SearchStrategy::CoarseToFine { .. }
        ));
    }

    #[test]
    fn test_delete() {
        let index = MatryoshkaIndex::new(MatryoshkaConfig::new(128));

        let emb = random_embedding(128, 1);
        index.insert("test", &emb, None).unwrap();
        assert_eq!(index.len(), 1);

        assert!(index.delete("test"));
        assert_eq!(index.len(), 0);
        assert!(!index.delete("test")); // Already deleted
    }

    #[test]
    fn test_stats() {
        let index = MatryoshkaIndex::new(MatryoshkaConfig::new(128));

        for i in 0..5 {
            let emb = random_embedding(128, i);
            index.insert(format!("v{}", i), &emb, None).unwrap();
        }

        let query = random_embedding(128, 0);
        index.search(&query, 3).unwrap();
        index.search(&query, 3).unwrap();

        let stats = index.stats();
        assert_eq!(stats.total_vectors, 5);
        assert_eq!(stats.searches, 2);
    }
}
