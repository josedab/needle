//! Sparse Vector Support
//!
//! Provides efficient storage and operations for sparse vectors,
//! useful for lexical features (TF-IDF, BM25, SPLADE) and hybrid search.

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
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        assert_eq!(
            indices.len(),
            values.len(),
            "Indices and values must have the same length"
        );
        for (i, &v) in values.iter().enumerate() {
            assert!(v.is_finite(), "Value at index {} is not finite: {}", i, v);
        }
        Self { indices, values }
    }

    /// Create a sparse vector from a HashMap
    ///
    /// # Panics
    /// Panics if any value is NaN or infinite.
    pub fn from_hashmap(map: &HashMap<u32, f32>) -> Self {
        for (&idx, &v) in map.iter() {
            assert!(v.is_finite(), "Value at index {} is not finite: {}", idx, v);
        }
        let mut pairs: Vec<(u32, f32)> = map.iter().map(|(&i, &v)| (i, v)).collect();
        pairs.sort_by_key(|(i, _)| *i);
        Self {
            indices: pairs.iter().map(|(i, _)| *i).collect(),
            values: pairs.iter().map(|(_, v)| *v).collect(),
        }
    }

    /// Create a sparse vector from a dense vector (only non-zero values)
    ///
    /// # Panics
    /// Panics if any value is NaN or infinite.
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &v) in dense.iter().enumerate() {
            assert!(v.is_finite(), "Value at index {} is not finite: {}", i, v);
            if v.abs() > threshold {
                indices.push(i as u32);
                values.push(v);
            }
        }

        Self { indices, values }
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
                self.inverted_index
                    .entry(idx)
                    .or_default()
                    .push((id, val));
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
                self.inverted_index
                    .entry(idx)
                    .or_default()
                    .push((id, val));
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
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> = BinaryHeap::with_capacity(k + 1);

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
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> = BinaryHeap::with_capacity(k + 1);

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
}
