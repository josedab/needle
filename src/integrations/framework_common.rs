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
        DistanceFunction::Euclidean | DistanceFunction::Manhattan => 1.0 / (1.0 + distance),
        DistanceFunction::DotProduct => (distance + 1.0) / 2.0,
    }
}
