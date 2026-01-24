//! Vector index implementations.

use crate::error::Result;

/// Common trait for pluggable vector index backends.
///
/// All index implementations share these operations, enabling the engine
/// to swap backends without changing higher-level code.
pub trait VectorIndex: Send + Sync {
    /// Search for the `k` nearest neighbors to `query`.
    ///
    /// Returns a list of `(id, distance)` pairs sorted by distance (closest first).
    fn search(
        &self,
        query: &[f32],
        k: usize,
        vectors: &[Vec<f32>],
    ) -> Result<Vec<(usize, f32)>>;

    /// Insert a vector with the given `id` into the index.
    fn insert(&mut self, id: usize, vector: &[f32], vectors: &[Vec<f32>]) -> Result<()>;

    /// Delete a vector by `id`. Returns `true` if the vector was found and deleted.
    fn delete(&mut self, id: usize) -> Result<bool>;

    /// Return the number of (non-deleted) vectors in the index.
    fn len(&self) -> usize;

    /// Return `true` if the index contains no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(feature = "diskann")]
pub mod diskann;
pub mod float16;
pub mod graph_vector_index;
pub mod hnsw;
pub mod hybrid_ann;
pub mod incremental;
pub mod ivf;
pub mod multimodal_index;
pub mod multivec;
pub mod quantization;
pub mod sparse;
pub mod graph_vector_fusion;
pub mod multimodal_fusion;
#[cfg(feature = "diskann")]
pub mod tiered_ann;
