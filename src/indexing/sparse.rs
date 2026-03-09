//! Sparse Vector Support
//!
//! Provides efficient storage and operations for sparse vectors,
//! useful for lexical features (TF-IDF, BM25, SPLADE) and hybrid search.
//!
//! # Overview
//!
//! Sparse vectors store only non-zero values, making them memory-efficient
//! for high-dimensional data where most values are zero. Common use cases:
//!
//! - **TF-IDF vectors**: Term frequency-inverse document frequency
//! - **BM25 vectors**: Best Match 25 scoring
//! - **SPLADE vectors**: Sparse neural retrieval
//!
//! # Example: Creating Sparse Vectors
//!
//! ```
//! use needle::SparseVector;
//!
//! // Create from indices and values
//! let sparse = SparseVector::new(
//!     vec![0, 5, 10],       // Non-zero indices
//!     vec![0.5, 1.2, 0.8],  // Corresponding values
//! );
//! assert_eq!(sparse.len(), 3);
//!
//! // Create from a dense vector (values below threshold become zero)
//! let dense = vec![0.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.7];
//! let sparse = SparseVector::from_dense(&dense, 0.1);
//! assert_eq!(sparse.len(), 3); // Only values > 0.1 are kept
//! ```
//!
//! # Example: Using the Sparse Index
//!
//! ```
//! use needle::{SparseIndex, SparseVector, SparseDistance};
//!
//! // Create a sparse index
//! let mut index = SparseIndex::new();
//!
//! // Insert sparse vectors (returns internal ID)
//! let v1 = SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
//! let v2 = SparseVector::new(vec![1, 2, 3], vec![2.0, 1.0, 1.0]);
//! let v3 = SparseVector::new(vec![0, 3], vec![1.0, 2.0]);
//!
//! let id1 = index.insert(v1);
//! let id2 = index.insert(v2);
//! let id3 = index.insert(v3);
//!
//! assert_eq!(index.len(), 3);
//!
//! // Search for similar vectors
//! let query = SparseVector::new(vec![0, 1], vec![1.0, 1.0]);
//! let results = index.search(&query, 2);
//!
//! assert_eq!(results.len(), 2);
//! ```
//!
//! # Example: Computing Sparse Distance
//!
//! ```
//! use needle::{SparseVector, SparseDistance};
//!
//! let v1 = SparseVector::new(vec![0, 1, 2], vec![1.0, 0.0, 1.0]);
//! let v2 = SparseVector::new(vec![0, 1, 2], vec![1.0, 1.0, 0.0]);
//!
//! // Compute cosine similarity
//! let similarity = SparseDistance::cosine_similarity(&v1, &v2);
//! println!("Cosine similarity: {}", similarity);
//!
//! // Compute dot product
//! let dot = SparseDistance::dot_product(&v1, &v2);
//! assert!((dot - 1.0).abs() < 0.001);
//! ```

use crate::error::{NeedleError, Result};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// A sparse vector represented as index-value pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero indices
    pub indices: Vec<u32>,
    /// Corresponding values
    pub values: Vec<f32>,
}

impl SparseVector {
    /// Create a new sparse vector from indices and values
    ///
    /// # Panics
    /// Panics if indices and values have different lengths, or if any value is NaN or infinite.
    /// Use [`try_new`](Self::try_new) for a non-panicking alternative.
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        Self::try_new(indices, values).expect("SparseVector::new: invalid input")
    }

    /// Create a new sparse vector, returning an error on invalid input.
    pub fn try_new(indices: Vec<u32>, values: Vec<f32>) -> Result<Self> {
        if indices.len() != values.len() {
            return Err(NeedleError::InvalidVector(format!(
                "Indices length ({}) != values length ({})",
                indices.len(),
                values.len()
            )));
        }
        for (i, &v) in values.iter().enumerate() {
            if !v.is_finite() {
                return Err(NeedleError::InvalidVector(format!(
                    "Value at index {i} is not finite: {v}"
                )));
            }
        }
        Ok(Self { indices, values })
    }

    /// Create a sparse vector from a HashMap
    ///
    /// # Panics
    /// Panics if any value is NaN or infinite.
    /// Use [`try_from_hashmap`](Self::try_from_hashmap) for a non-panicking alternative.
    pub fn from_hashmap(map: &HashMap<u32, f32>) -> Self {
        Self::try_from_hashmap(map).expect("SparseVector::from_hashmap: invalid input")
    }

    /// Create a sparse vector from a HashMap, returning an error on invalid input.
    pub fn try_from_hashmap(map: &HashMap<u32, f32>) -> Result<Self> {
        for (&idx, &v) in map.iter() {
            if !v.is_finite() {
                return Err(NeedleError::InvalidVector(format!(
                    "Value at index {idx} is not finite: {v}"
                )));
            }
        }
        let mut pairs: Vec<(u32, f32)> = map.iter().map(|(&i, &v)| (i, v)).collect();
        pairs.sort_by_key(|(i, _)| *i);
        Ok(Self {
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
        })
    }

    /// Create a sparse vector from a dense vector (only non-zero values)
    ///
    /// # Panics
    /// Panics if any value is NaN or infinite.
    /// Use [`try_from_dense`](Self::try_from_dense) for a non-panicking alternative.
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        Self::try_from_dense(dense, threshold).expect("SparseVector::from_dense: invalid input")
    }

    /// Create a sparse vector from a dense vector, returning an error on invalid input.
    ///
    /// Returns an error if any value is non-finite or if the dense vector exceeds
    /// `u32::MAX` elements.
    pub fn try_from_dense(dense: &[f32], threshold: f32) -> Result<Self> {
        if dense.len() > u32::MAX as usize {
            return Err(NeedleError::InvalidVector(format!(
                "Dense vector length ({}) exceeds u32::MAX",
                dense.len()
            )));
        }
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in dense.iter().enumerate() {
            if !v.is_finite() {
                return Err(NeedleError::InvalidVector(format!(
                    "Value at index {i} is not finite: {v}"
                )));
            }
            if v.abs() > threshold {
                indices.push(i as u32);
                values.push(v);
            }
        }

        Ok(Self { indices, values })
    }

    /// Convert to a dense vector
    pub fn to_dense(&self, dimensions: usize) -> Vec<f32> {
        let mut dense = vec![0.0; dimensions];
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if (idx as usize) < dimensions {
                dense[idx as usize] = val;
            }
        }
        dense
    }

    /// Number of non-zero elements
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Get the maximum index
    pub fn max_dimension(&self) -> Option<u32> {
        self.indices.iter().max().copied()
    }

    /// Get value at index, returns 0.0 if not present
    pub fn get(&self, index: u32) -> f32 {
        match self.indices.binary_search(&index) {
            Ok(pos) => self.values[pos],
            Err(_) => 0.0,
        }
    }

    /// Compute L2 norm
    pub fn l2_norm(&self) -> f32 {
        self.values.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length (in-place)
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 0.0 {
            for v in &mut self.values {
                *v /= norm;
            }
        }
    }

    /// Return a normalized copy
    pub fn normalized(&self) -> Self {
        let mut copy = self.clone();
        copy.normalize();
        copy
    }
}

/// Distance functions for sparse vectors
pub struct SparseDistance;

impl SparseDistance {
    /// Dot product between two sparse vectors
    pub fn dot_product(a: &SparseVector, b: &SparseVector) -> f32 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < a.indices.len() && j < b.indices.len() {
            match a.indices[i].cmp(&b.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += a.values[i] * b.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1;
                }
            }
        }

        result
    }

    /// Cosine similarity between two sparse vectors
    pub fn cosine_similarity(a: &SparseVector, b: &SparseVector) -> f32 {
        let dot = Self::dot_product(a, b);
        let norm_a = a.l2_norm();
        let norm_b = b.l2_norm();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Cosine distance (1 - similarity)
    pub fn cosine_distance(a: &SparseVector, b: &SparseVector) -> f32 {
        1.0 - Self::cosine_similarity(a, b)
    }

    /// Euclidean distance between two sparse vectors
    pub fn euclidean_distance(a: &SparseVector, b: &SparseVector) -> f32 {
        let mut sum_sq = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < a.indices.len() || j < b.indices.len() {
            let idx_a = a.indices.get(i).copied().unwrap_or(u32::MAX);
            let idx_b = b.indices.get(j).copied().unwrap_or(u32::MAX);

            match idx_a.cmp(&idx_b) {
                std::cmp::Ordering::Equal => {
                    let diff = a.values[i] - b.values[j];
                    sum_sq += diff * diff;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    sum_sq += a.values[i] * a.values[i];
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    sum_sq += b.values[j] * b.values[j];
                    j += 1;
                }
            }
        }

        sum_sq.sqrt()
    }
}

/// Statistics about a [`SparseIndex`] instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseIndexStats {
    /// Number of vectors stored.
    pub num_vectors: usize,
    /// Number of distinct dimensions with at least one posting.
    pub num_posting_lists: usize,
    /// Total number of postings across all lists.
    pub total_postings: usize,
    /// Length of the longest posting list.
    pub max_posting_len: usize,
    /// Average posting list length.
    pub avg_posting_len: f64,
    /// Total non-zero entries across all stored vectors.
    pub total_nnz: usize,
    /// Average non-zero entries per vector.
    pub avg_nnz: f64,
    /// Estimated heap memory usage in bytes.
    pub estimated_memory_bytes: usize,
}

/// Sparse vector index for efficient search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseIndex {
    /// Inverted index: dimension -> list of (vector_id, value)
    inverted_index: HashMap<u32, Vec<(usize, f32)>>,
    /// Vector ID to sparse vector mapping
    vectors: HashMap<usize, SparseVector>,
    /// Next vector ID
    next_id: usize,
}

impl Default for SparseIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseIndex {
    /// Create a new sparse index
    pub fn new() -> Self {
        Self {
            inverted_index: HashMap::new(),
            vectors: HashMap::new(),
            next_id: 0,
        }
    }

    /// Insert a sparse vector and return its ID
    pub fn insert(&mut self, vector: SparseVector) -> usize {
        let id = self.next_id;
        self.next_id += 1;

        // Add to inverted index (skip zero values)
        for (&idx, &val) in vector.indices.iter().zip(vector.values.iter()) {
            if val != 0.0 {
                self.inverted_index.entry(idx).or_default().push((id, val));
            }
        }

        self.vectors.insert(id, vector);
        id
    }

    /// Insert with a specific ID
    pub fn insert_with_id(&mut self, id: usize, vector: SparseVector) {
        // Remove old vector if exists
        self.remove(id);

        // Add to inverted index (skip zero values)
        for (&idx, &val) in vector.indices.iter().zip(vector.values.iter()) {
            if val != 0.0 {
                self.inverted_index.entry(idx).or_default().push((id, val));
            }
        }

        self.vectors.insert(id, vector);
        self.next_id = self.next_id.max(id + 1);
    }

    /// Remove a vector by ID
    pub fn remove(&mut self, id: usize) -> bool {
        if let Some(vector) = self.vectors.remove(&id) {
            // Remove from inverted index
            for &idx in &vector.indices {
                if let Some(postings) = self.inverted_index.get_mut(&idx) {
                    postings.retain(|(vid, _)| *vid != id);
                }
            }
            true
        } else {
            false
        }
    }

    /// Get a vector by ID
    pub fn get(&self, id: usize) -> Option<&SparseVector> {
        self.vectors.get(&id)
    }

    /// Number of vectors
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Check whether the index contains a vector with the given ID.
    pub fn contains(&self, id: usize) -> bool {
        self.vectors.contains_key(&id)
    }

    /// Return all vector IDs in the index.
    pub fn ids(&self) -> Vec<usize> {
        self.vectors.keys().copied().collect()
    }

    /// Insert multiple sparse vectors, returning their assigned IDs.
    pub fn batch_insert(&mut self, vectors: Vec<SparseVector>) -> Vec<usize> {
        vectors.into_iter().map(|v| self.insert(v)).collect()
    }

    /// Remove multiple vectors by ID. Returns the number of vectors actually removed.
    pub fn batch_delete(&mut self, ids: &[usize]) -> usize {
        ids.iter().filter(|&&id| self.remove(id)).count()
    }

    /// Remove all vectors from the index.
    pub fn clear(&mut self) {
        self.inverted_index.clear();
        self.vectors.clear();
    }

    /// Compute statistics about the sparse index.
    pub fn stats(&self) -> SparseIndexStats {
        let num_posting_lists = self.inverted_index.len();
        let total_postings: usize = self.inverted_index.values().map(|p| p.len()).sum();
        let max_posting_len = self.inverted_index.values().map(|p| p.len()).max().unwrap_or(0);
        let avg_posting_len = if num_posting_lists > 0 {
            total_postings as f64 / num_posting_lists as f64
        } else {
            0.0
        };
        let total_nnz: usize = self.vectors.values().map(|v| v.len()).sum();
        let avg_nnz = if self.vectors.is_empty() {
            0.0
        } else {
            total_nnz as f64 / self.vectors.len() as f64
        };

        SparseIndexStats {
            num_vectors: self.vectors.len(),
            num_posting_lists,
            total_postings,
            max_posting_len,
            avg_posting_len,
            total_nnz,
            avg_nnz,
            estimated_memory_bytes: self.estimated_memory(),
        }
    }

    /// Estimate heap memory usage in bytes.
    pub fn estimated_memory(&self) -> usize {
        let posting_mem: usize = self
            .inverted_index
            .values()
            .map(|p| p.capacity() * std::mem::size_of::<(usize, f32)>())
            .sum();
        let hashmap_overhead =
            self.inverted_index.capacity() * std::mem::size_of::<(u32, Vec<(usize, f32)>)>();
        let vector_mem: usize = self
            .vectors
            .values()
            .map(|v| {
                v.indices.capacity() * std::mem::size_of::<u32>()
                    + v.values.capacity() * std::mem::size_of::<f32>()
            })
            .sum();
        let vector_hashmap_overhead =
            self.vectors.capacity() * std::mem::size_of::<(usize, SparseVector)>();
        posting_mem + hashmap_overhead + vector_mem + vector_hashmap_overhead
    }

    /// Search for similar vectors using dot product
    /// Returns vector IDs and scores, sorted by score descending
    ///
    /// Uses a min-heap for O(n log k) complexity instead of O(n log n) sorting.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(usize, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        // Accumulate scores using inverted index
        for (&idx, &val) in query.indices.iter().zip(query.values.iter()) {
            if let Some(postings) = self.inverted_index.get(&idx) {
                for &(vec_id, vec_val) in postings {
                    *scores.entry(vec_id).or_default() += val * vec_val;
                }
            }
        }

        // Use min-heap to efficiently find top-k results
        // Heap contains (Reverse(score), id) so smallest score is at top
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        for (id, score) in scores {
            if heap.len() < k {
                heap.push((Reverse(OrderedFloat(score)), id));
            } else if let Some(&(Reverse(OrderedFloat(min_score)), _)) = heap.peek() {
                if score > min_score {
                    heap.pop();
                    heap.push((Reverse(OrderedFloat(score)), id));
                }
            }
        }

        // Extract results sorted by score descending
        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|(Reverse(OrderedFloat(score)), id)| (id, score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search with cosine similarity (normalizes scores)
    ///
    /// Uses a min-heap for O(n log k) complexity instead of O(n log n) sorting.
    pub fn search_cosine(&self, query: &SparseVector, k: usize) -> Vec<(usize, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let query_norm = query.l2_norm();
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        // Accumulate dot products using inverted index
        for (&idx, &val) in query.indices.iter().zip(query.values.iter()) {
            if let Some(postings) = self.inverted_index.get(&idx) {
                for &(vec_id, vec_val) in postings {
                    *scores.entry(vec_id).or_default() += val * vec_val;
                }
            }
        }

        // Use min-heap to efficiently find top-k results
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        for (id, dot) in scores {
            if let Some(v) = self.vectors.get(&id) {
                let vec_norm = v.l2_norm();
                let similarity = if vec_norm > 0.0 {
                    dot / (query_norm * vec_norm)
                } else {
                    0.0
                };

                if heap.len() < k {
                    heap.push((Reverse(OrderedFloat(similarity)), id));
                } else if let Some(&(Reverse(OrderedFloat(min_score)), _)) = heap.peek() {
                    if similarity > min_score {
                        heap.pop();
                        heap.push((Reverse(OrderedFloat(similarity)), id));
                    }
                }
            }
        }

        // Extract results sorted by score descending
        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|(Reverse(OrderedFloat(score)), id)| (id, score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search using dot product with a minimum score threshold.
    ///
    /// Only returns results with a score >= `min_score`, which avoids
    /// accumulating and ranking irrelevant candidates.
    pub fn search_with_threshold(
        &self,
        query: &SparseVector,
        k: usize,
        min_score: f32,
    ) -> Vec<(usize, f32)> {
        if k == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<usize, f32> = HashMap::new();

        for (&idx, &val) in query.indices.iter().zip(query.values.iter()) {
            if let Some(postings) = self.inverted_index.get(&idx) {
                for &(vec_id, vec_val) in postings {
                    *scores.entry(vec_id).or_default() += val * vec_val;
                }
            }
        }

        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        for (id, score) in scores {
            if score < min_score {
                continue;
            }
            if heap.len() < k {
                heap.push((Reverse(OrderedFloat(score)), id));
            } else if let Some(&(Reverse(OrderedFloat(heap_min)), _)) = heap.peek() {
                if score > heap_min {
                    heap.pop();
                    heap.push((Reverse(OrderedFloat(score)), id));
                }
            }
        }

        let mut results: Vec<(usize, f32)> = heap
            .into_iter()
            .map(|(Reverse(OrderedFloat(score)), id)| (id, score))
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_creation() {
        let sv = SparseVector::new(vec![0, 5, 10], vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.get(0), 1.0);
        assert_eq!(sv.get(5), 2.0);
        assert_eq!(sv.get(10), 3.0);
        assert_eq!(sv.get(3), 0.0);
    }

    #[test]
    fn test_sparse_from_dense() {
        let dense = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let sv = SparseVector::from_dense(&dense, 0.0);

        assert_eq!(sv.len(), 3);
        assert_eq!(sv.get(1), 1.0);
        assert_eq!(sv.get(3), 2.0);
        assert_eq!(sv.get(5), 3.0);
    }

    #[test]
    fn test_sparse_to_dense() {
        let sv = SparseVector::new(vec![1, 3, 5], vec![1.0, 2.0, 3.0]);
        let dense = sv.to_dense(6);

        assert_eq!(dense, vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn test_dot_product() {
        let a = SparseVector::new(vec![0, 2, 4], vec![1.0, 2.0, 3.0]);
        let b = SparseVector::new(vec![1, 2, 4], vec![1.0, 2.0, 2.0]);

        // Only indices 2 and 4 overlap: 2*2 + 3*2 = 10
        let dot = SparseDistance::dot_product(&a, &b);
        assert!((dot - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 0.0]);
        let b = SparseVector::new(vec![0, 1], vec![0.0, 1.0]);

        // Orthogonal vectors
        let sim = SparseDistance::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);

        // Same direction
        let c = SparseVector::new(vec![0], vec![2.0]);
        let d = SparseVector::new(vec![0], vec![3.0]);
        let sim2 = SparseDistance::cosine_similarity(&c, &d);
        assert!((sim2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_index_search() {
        let mut index = SparseIndex::new();

        // Insert some vectors
        index.insert(SparseVector::new(vec![0, 1], vec![1.0, 0.0]));
        index.insert(SparseVector::new(vec![0, 1], vec![0.5, 0.5]));
        index.insert(SparseVector::new(vec![0, 1], vec![0.0, 1.0]));

        // Search for vector similar to [1, 0]
        let query = SparseVector::new(vec![0], vec![1.0]);
        let results = index.search(&query, 3);

        // First vector should be most similar
        assert_eq!(results.len(), 2); // Only 2 have non-zero overlap
        assert_eq!(results[0].0, 0); // ID 0 has highest score
    }

    #[test]
    fn test_normalize() {
        let mut sv = SparseVector::new(vec![0, 1], vec![3.0, 4.0]);
        sv.normalize();

        let norm = sv.l2_norm();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((sv.get(0) - 0.6).abs() < 1e-6);
        assert!((sv.get(1) - 0.8).abs() < 1e-6);
    }

    // ========================================================================
    // SparseVector edge cases
    // ========================================================================

    #[test]
    fn test_sparse_empty_vector() {
        let sv = SparseVector::new(vec![], vec![]);
        assert!(sv.is_empty());
        assert_eq!(sv.len(), 0);
        assert_eq!(sv.l2_norm(), 0.0);
        assert_eq!(sv.get(0), 0.0);
        assert_eq!(sv.max_dimension(), None);
    }

    #[test]
    fn test_sparse_normalized_empty() {
        let sv = SparseVector::new(vec![], vec![]);
        let norm = sv.normalized();
        assert!(norm.is_empty());
    }

    #[test]
    fn test_sparse_from_hashmap() {
        let mut map = HashMap::new();
        map.insert(5u32, 2.0f32);
        map.insert(1, 1.0);
        map.insert(10, 3.0);

        let sv = SparseVector::from_hashmap(&map);
        assert_eq!(sv.len(), 3);
        assert_eq!(sv.get(1), 1.0);
        assert_eq!(sv.get(5), 2.0);
        assert_eq!(sv.get(10), 3.0);
        // Should be sorted by index
        assert!(sv.indices.windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn test_sparse_from_hashmap_empty() {
        let map: HashMap<u32, f32> = HashMap::new();
        let sv = SparseVector::from_hashmap(&map);
        assert!(sv.is_empty());
    }

    #[test]
    fn test_sparse_max_dimension() {
        let sv = SparseVector::new(vec![2, 100, 50], vec![1.0, 2.0, 3.0]);
        assert_eq!(sv.max_dimension(), Some(100));
    }

    #[test]
    fn test_sparse_to_dense_out_of_range() {
        let sv = SparseVector::new(vec![0, 10], vec![1.0, 2.0]);
        let dense = sv.to_dense(5); // dimension 10 is out of range
        assert_eq!(dense.len(), 5);
        assert_eq!(dense[0], 1.0);
        // Index 10 should be ignored since dim is 5
    }

    #[test]
    #[should_panic(expected = "invalid input")]
    fn test_sparse_mismatched_lengths() {
        SparseVector::new(vec![0, 1, 2], vec![1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "invalid input")]
    fn test_sparse_nan_value() {
        SparseVector::new(vec![0], vec![f32::NAN]);
    }

    #[test]
    #[should_panic(expected = "invalid input")]
    fn test_sparse_infinity_value() {
        SparseVector::new(vec![0], vec![f32::INFINITY]);
    }

    #[test]
    fn test_try_new_mismatched_lengths() {
        let result = SparseVector::try_new(vec![0, 1, 2], vec![1.0, 2.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("length"));
    }

    #[test]
    fn test_try_new_nan() {
        let result = SparseVector::try_new(vec![0], vec![f32::NAN]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not finite"));
    }

    #[test]
    fn test_try_from_dense_valid() {
        let sv = SparseVector::try_from_dense(&[0.0, 0.5, 0.0, 0.8], 0.1).unwrap();
        assert_eq!(sv.len(), 2);
    }

    // ========================================================================
    // SparseDistance edge cases
    // ========================================================================

    #[test]
    fn test_sparse_cosine_distance() {
        let a = SparseVector::new(vec![0, 1], vec![1.0, 0.0]);
        let b = SparseVector::new(vec![0, 1], vec![0.0, 1.0]);
        let dist = SparseDistance::cosine_distance(&a, &b);
        assert!((dist - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_euclidean_distance() {
        let a = SparseVector::new(vec![0, 1], vec![3.0, 0.0]);
        let b = SparseVector::new(vec![0, 1], vec![0.0, 4.0]);
        let dist = SparseDistance::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_euclidean_no_overlap() {
        let a = SparseVector::new(vec![0], vec![3.0]);
        let b = SparseVector::new(vec![1], vec![4.0]);
        let dist = SparseDistance::euclidean_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_sparse_cosine_with_zero_vector() {
        let a = SparseVector::new(vec![], vec![]);
        let b = SparseVector::new(vec![0], vec![1.0]);
        let sim = SparseDistance::cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    // ========================================================================
    // SparseIndex CRUD and search tests
    // ========================================================================

    #[test]
    fn test_sparse_index_insert_with_id() {
        let mut index = SparseIndex::new();
        let v = SparseVector::new(vec![0, 1], vec![1.0, 2.0]);
        index.insert_with_id(42, v);

        assert_eq!(index.len(), 1);
        assert!(index.get(42).is_some());
        assert_eq!(index.get(42).unwrap().get(0), 1.0);
    }

    #[test]
    fn test_sparse_index_insert_with_id_replace() {
        let mut index = SparseIndex::new();
        index.insert_with_id(0, SparseVector::new(vec![0], vec![1.0]));
        index.insert_with_id(0, SparseVector::new(vec![0], vec![2.0]));

        assert_eq!(index.len(), 1);
        assert_eq!(index.get(0).unwrap().get(0), 2.0);
    }

    #[test]
    fn test_sparse_index_remove() {
        let mut index = SparseIndex::new();
        let id = index.insert(SparseVector::new(vec![0], vec![1.0]));
        assert_eq!(index.len(), 1);

        assert!(index.remove(id));
        assert_eq!(index.len(), 0);
        assert!(index.get(id).is_none());
    }

    #[test]
    fn test_sparse_index_remove_nonexistent() {
        let mut index = SparseIndex::new();
        assert!(!index.remove(999));
    }

    #[test]
    fn test_sparse_index_search_k_zero() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![1.0]));

        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_index_search_k_greater_than_len() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![1.0]));
        index.insert(SparseVector::new(vec![0], vec![2.0]));

        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 100);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_sparse_index_search_cosine() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0, 1], vec![1.0, 0.0]));
        index.insert(SparseVector::new(vec![0, 1], vec![0.5, 0.5]));
        index.insert(SparseVector::new(vec![0, 1], vec![0.0, 1.0]));

        let query = SparseVector::new(vec![0], vec![1.0]);
        let results = index.search_cosine(&query, 3);
        assert!(!results.is_empty());
        // First result should be most similar (vector at [1, 0])
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_sparse_index_search_cosine_zero_query() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![1.0]));

        let results = index.search_cosine(&SparseVector::new(vec![], vec![]), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_index_empty_search() {
        let index = SparseIndex::new();
        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_index_is_empty() {
        let mut index = SparseIndex::new();
        assert!(index.is_empty());
        index.insert(SparseVector::new(vec![0], vec![1.0]));
        assert!(!index.is_empty());
    }

    // ========================================================================
    // SparseIndex batch, contains, clear, and ids
    // ========================================================================

    #[test]
    fn test_sparse_index_contains() {
        let mut index = SparseIndex::new();
        let id = index.insert(SparseVector::new(vec![0], vec![1.0]));
        assert!(index.contains(id));
        assert!(!index.contains(id + 1));
    }

    #[test]
    fn test_sparse_index_ids() {
        let mut index = SparseIndex::new();
        let id1 = index.insert(SparseVector::new(vec![0], vec![1.0]));
        let id2 = index.insert(SparseVector::new(vec![1], vec![2.0]));
        let mut ids = index.ids();
        ids.sort();
        assert_eq!(ids, vec![id1, id2]);
    }

    #[test]
    fn test_sparse_index_batch_insert() {
        let mut index = SparseIndex::new();
        let vecs = vec![
            SparseVector::new(vec![0], vec![1.0]),
            SparseVector::new(vec![1], vec![2.0]),
            SparseVector::new(vec![2], vec![3.0]),
        ];
        let ids = index.batch_insert(vecs);
        assert_eq!(ids.len(), 3);
        assert_eq!(index.len(), 3);
        for &id in &ids {
            assert!(index.contains(id));
        }
    }

    #[test]
    fn test_sparse_index_batch_delete() {
        let mut index = SparseIndex::new();
        let ids = index.batch_insert(vec![
            SparseVector::new(vec![0], vec![1.0]),
            SparseVector::new(vec![1], vec![2.0]),
            SparseVector::new(vec![2], vec![3.0]),
        ]);
        let removed = index.batch_delete(&[ids[0], ids[2], 999]);
        assert_eq!(removed, 2);
        assert_eq!(index.len(), 1);
        assert!(index.contains(ids[1]));
    }

    #[test]
    fn test_sparse_index_clear() {
        let mut index = SparseIndex::new();
        index.batch_insert(vec![
            SparseVector::new(vec![0, 1], vec![1.0, 2.0]),
            SparseVector::new(vec![2, 3], vec![3.0, 4.0]),
        ]);
        assert_eq!(index.len(), 2);
        index.clear();
        assert!(index.is_empty());
        // Search should return nothing after clear
        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sparse_index_batch_delete_empty() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![1.0]));
        let removed = index.batch_delete(&[]);
        assert_eq!(removed, 0);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_sparse_index_search_after_batch_delete() {
        let mut index = SparseIndex::new();
        let ids = index.batch_insert(vec![
            SparseVector::new(vec![0], vec![1.0]),
            SparseVector::new(vec![0], vec![2.0]),
            SparseVector::new(vec![0], vec![3.0]),
        ]);
        // Delete the top scorer
        index.batch_delete(&[ids[2]]);
        let results = index.search(&SparseVector::new(vec![0], vec![1.0]), 3);
        assert_eq!(results.len(), 2);
        // Highest remaining score is id1 with val 2.0
        assert_eq!(results[0].0, ids[1]);
    }

    // ========================================================================
    // SparseIndex threshold search
    // ========================================================================

    #[test]
    fn test_search_with_threshold_filters_low_scores() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![0.1])); // score 0.1
        index.insert(SparseVector::new(vec![0], vec![0.5])); // score 0.5
        index.insert(SparseVector::new(vec![0], vec![1.0])); // score 1.0

        let query = SparseVector::new(vec![0], vec![1.0]);
        let results = index.search_with_threshold(&query, 10, 0.4);
        // Only vectors with score >= 0.4 should appear
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(_, s)| *s >= 0.4));
    }

    #[test]
    fn test_search_with_threshold_zero_returns_all() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![0.01]));
        index.insert(SparseVector::new(vec![0], vec![1.0]));

        let query = SparseVector::new(vec![0], vec![1.0]);
        let all = index.search(&query, 10);
        let thresholded = index.search_with_threshold(&query, 10, 0.0);
        assert_eq!(all.len(), thresholded.len());
    }

    #[test]
    fn test_search_with_threshold_high_filters_everything() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![1.0]));

        let query = SparseVector::new(vec![0], vec![1.0]);
        let results = index.search_with_threshold(&query, 10, 100.0);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_threshold_k_zero() {
        let mut index = SparseIndex::new();
        index.insert(SparseVector::new(vec![0], vec![1.0]));
        let results = index.search_with_threshold(&SparseVector::new(vec![0], vec![1.0]), 0, 0.0);
        assert!(results.is_empty());
    }

    // ========================================================================
    // SparseIndex stats and memory estimation
    // ========================================================================

    #[test]
    fn test_sparse_index_stats_empty() {
        let index = SparseIndex::new();
        let stats = index.stats();
        assert_eq!(stats.num_vectors, 0);
        assert_eq!(stats.num_posting_lists, 0);
        assert_eq!(stats.total_postings, 0);
        assert_eq!(stats.max_posting_len, 0);
        assert_eq!(stats.avg_posting_len, 0.0);
        assert_eq!(stats.total_nnz, 0);
        assert_eq!(stats.avg_nnz, 0.0);
    }

    #[test]
    fn test_sparse_index_stats_populated() {
        let mut index = SparseIndex::new();
        // v0: dims [0, 1] → 2 nnz
        index.insert(SparseVector::new(vec![0, 1], vec![1.0, 2.0]));
        // v1: dims [1, 2] → 2 nnz
        index.insert(SparseVector::new(vec![1, 2], vec![3.0, 4.0]));

        let stats = index.stats();
        assert_eq!(stats.num_vectors, 2);
        // Posting lists for dims 0, 1, 2
        assert_eq!(stats.num_posting_lists, 3);
        // dim0: [v0], dim1: [v0, v1], dim2: [v1] → total 4
        assert_eq!(stats.total_postings, 4);
        assert_eq!(stats.max_posting_len, 2); // dim 1 has 2 entries
        assert!((stats.avg_posting_len - 4.0 / 3.0).abs() < 1e-6);
        assert_eq!(stats.total_nnz, 4);
        assert!((stats.avg_nnz - 2.0).abs() < 1e-6);
        assert!(stats.estimated_memory_bytes > 0);
    }

    #[test]
    fn test_sparse_index_memory_grows_with_data() {
        let mut index = SparseIndex::new();
        let mem_empty = index.estimated_memory();
        index.insert(SparseVector::new(vec![0, 1, 2, 3], vec![1.0, 2.0, 3.0, 4.0]));
        let mem_one = index.estimated_memory();
        assert!(mem_one > mem_empty);
        for i in 1..100 {
            index.insert(SparseVector::new(vec![0, 1, 2, 3], vec![i as f32; 4]));
        }
        let mem_hundred = index.estimated_memory();
        assert!(mem_hundred > mem_one);
    }
}
