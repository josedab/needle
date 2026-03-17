//! Shared Framework Adapter Internals
//!
//! Common collection wrapper and utilities used by both the LangChain and
//! LlamaIndex integration modules. This module is `pub(crate)` — external
//! consumers should use `langchain` or `llamaindex` directly.

use crate::collection::{Collection, CollectionConfig, CollectionStats, SearchResult};
use crate::distance::DistanceFunction;
use crate::error::Result;
use crate::metadata::Filter;
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::Arc;

/// Shared collection wrapper that encapsulates the `Arc<RwLock<Collection>>`
/// pattern and common operations used by both framework adapters.
pub(crate) struct FrameworkCollection {
    collection: Arc<RwLock<Collection>>,
}

impl FrameworkCollection {
    /// Create a new framework collection with the given parameters.
    pub fn new(name: &str, dimension: usize, distance: DistanceFunction) -> Self {
        let config = CollectionConfig::new(name, dimension).with_distance(distance);
        let collection = Collection::new(config);
        Self {
            collection: Arc::new(RwLock::new(collection)),
        }
    }

    /// Wrap an existing `Collection`.
    pub fn from_collection(collection: Collection) -> Self {
        Self {
            collection: Arc::new(RwLock::new(collection)),
        }
    }

    /// Number of vectors in the collection.
    pub fn len(&self) -> usize {
        self.collection.read().len()
    }

    /// Whether the collection is empty.
    pub fn is_empty(&self) -> bool {
        self.collection.read().is_empty()
    }

    /// Search with an optional metadata filter.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<SearchResult>> {
        let collection = self.collection.read();
        if let Some(f) = filter {
            collection.search_with_filter(query, k, f)
        } else {
            collection.search(query, k)
        }
    }

    /// Update metadata for a vector.
    pub fn update_metadata(&self, id: &str, metadata: Option<Value>) -> Result<()> {
        self.collection.write().update_metadata(id, metadata)
    }

    /// Compact the collection, removing deleted vectors.
    pub fn compact(&self) -> Result<usize> {
        self.collection.write().compact()
    }

    /// Get collection statistics.
    pub fn stats(&self) -> CollectionStats {
        self.collection.read().stats()
    }

    /// Serialize the collection to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        self.collection.read().to_bytes()
    }

    /// Deserialize a collection from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let collection = Collection::from_bytes(bytes)?;
        Ok(Self::from_collection(collection))
    }

    /// Acquire a read lock on the underlying collection.
    pub fn read(&self) -> parking_lot::RwLockReadGuard<'_, Collection> {
        self.collection.read()
    }

    /// Acquire a write lock on the underlying collection.
    pub fn write(&self) -> parking_lot::RwLockWriteGuard<'_, Collection> {
        self.collection.write()
    }
}

/// Convert a distance value to a similarity score based on the distance function.
///
/// Used by both LangChain and LlamaIndex adapters for score normalization.
pub(crate) fn distance_to_score(distance: f32, distance_fn: DistanceFunction) -> f32 {
    match distance_fn {
        DistanceFunction::Cosine | DistanceFunction::CosineNormalized => 1.0 - distance,
        DistanceFunction::Euclidean | DistanceFunction::Manhattan | DistanceFunction::Hamming | DistanceFunction::Chebyshev => 1.0 / (1.0 + distance),
        DistanceFunction::DotProduct => (distance + 1.0) / 2.0,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // ---- distance_to_score tests ----

    #[test]
    fn test_cosine_distance_to_score() {
        // Cosine: score = 1.0 - distance
        assert!((distance_to_score(0.0, DistanceFunction::Cosine) - 1.0).abs() < f32::EPSILON);
        assert!((distance_to_score(0.5, DistanceFunction::Cosine) - 0.5).abs() < f32::EPSILON);
        assert!((distance_to_score(1.0, DistanceFunction::Cosine) - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_normalized_distance_to_score() {
        // Same formula as Cosine
        assert!(
            (distance_to_score(0.0, DistanceFunction::CosineNormalized) - 1.0).abs()
                < f32::EPSILON
        );
        assert!(
            (distance_to_score(0.3, DistanceFunction::CosineNormalized) - 0.7).abs()
                < f32::EPSILON
        );
    }

    #[test]
    fn test_euclidean_distance_to_score() {
        // Euclidean: score = 1.0 / (1.0 + distance)
        assert!(
            (distance_to_score(0.0, DistanceFunction::Euclidean) - 1.0).abs() < f32::EPSILON
        );
        assert!(
            (distance_to_score(1.0, DistanceFunction::Euclidean) - 0.5).abs() < f32::EPSILON
        );
        // As distance grows, score approaches 0
        let large = distance_to_score(1000.0, DistanceFunction::Euclidean);
        assert!(large > 0.0 && large < 0.01);
    }

    #[test]
    fn test_manhattan_distance_to_score() {
        // Same formula as Euclidean
        assert!(
            (distance_to_score(0.0, DistanceFunction::Manhattan) - 1.0).abs() < f32::EPSILON
        );
        assert!(
            (distance_to_score(1.0, DistanceFunction::Manhattan) - 0.5).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_hamming_distance_to_score() {
        // Same formula as Euclidean
        assert!((distance_to_score(0.0, DistanceFunction::Hamming) - 1.0).abs() < f32::EPSILON);
        assert!((distance_to_score(3.0, DistanceFunction::Hamming) - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_chebyshev_distance_to_score() {
        // Same formula as Euclidean
        assert!(
            (distance_to_score(0.0, DistanceFunction::Chebyshev) - 1.0).abs() < f32::EPSILON
        );
        assert!(
            (distance_to_score(1.0, DistanceFunction::Chebyshev) - 0.5).abs() < f32::EPSILON
        );
    }

    #[test]
    fn test_dot_product_distance_to_score() {
        // DotProduct: score = (distance + 1.0) / 2.0
        assert!(
            (distance_to_score(1.0, DistanceFunction::DotProduct) - 1.0).abs() < f32::EPSILON
        );
        assert!(
            (distance_to_score(0.0, DistanceFunction::DotProduct) - 0.5).abs() < f32::EPSILON
        );
        assert!(
            (distance_to_score(-1.0, DistanceFunction::DotProduct) - 0.0).abs() < f32::EPSILON
        );
    }

    // ---- FrameworkCollection tests ----

    #[test]
    fn test_new_collection() {
        let fc = FrameworkCollection::new("test", 4, DistanceFunction::Cosine);
        assert_eq!(fc.len(), 0);
        assert!(fc.is_empty());
    }

    #[test]
    fn test_from_collection() {
        let config = CollectionConfig::new("test", 4);
        let collection = Collection::new(config);
        let fc = FrameworkCollection::from_collection(collection);
        assert!(fc.is_empty());
    }

    #[test]
    fn test_collection_insert_and_search() {
        let fc = FrameworkCollection::new("test", 3, DistanceFunction::Cosine);

        {
            let mut coll = fc.write();
            coll.insert("v1", &[1.0, 0.0, 0.0], None).unwrap();
            coll.insert("v2", &[0.0, 1.0, 0.0], None).unwrap();
        }

        assert_eq!(fc.len(), 2);
        assert!(!fc.is_empty());

        let results = fc.search(&[1.0, 0.0, 0.0], 1, None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "v1");
    }

    #[test]
    fn test_collection_stats() {
        let fc = FrameworkCollection::new("test", 3, DistanceFunction::Cosine);
        let stats = fc.stats();
        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.dimensions, 3);
    }

    #[test]
    fn test_collection_serialization_roundtrip() {
        let fc = FrameworkCollection::new("test", 3, DistanceFunction::Cosine);
        {
            let mut coll = fc.write();
            coll.insert("v1", &[1.0, 2.0, 3.0], None).unwrap();
        }

        let bytes = fc.to_bytes().unwrap();
        let fc2 = FrameworkCollection::from_bytes(&bytes).unwrap();
        assert_eq!(fc2.len(), 1);
    }
}
