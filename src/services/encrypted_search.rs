//! Encrypted Search
//!
//! Locality-sensitive hashing (LSH) for approximate vector search on encrypted
//! data. Vectors are hashed into binary codes for comparison without decryption.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::encrypted_search::{
//!     EncryptedIndex, EncryptionConfig, EncryptedVector,
//! };
//!
//! let mut index = EncryptedIndex::new(EncryptionConfig::new(128, 64));
//!
//! // Encrypt and index a vector
//! let encrypted = index.encrypt_and_insert("v1", &vec![0.5f32; 128]);
//!
//! // Search on encrypted data (no decryption needed)
//! let results = index.encrypted_search(&vec![0.5f32; 128], 5);
//! assert_eq!(results[0].id, "v1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Encryption configuration.
#[derive(Debug, Clone)]
pub struct EncryptionConfig {
    /// Original vector dimensions.
    pub dimensions: usize,
    /// Number of hash bits for LSH.
    pub hash_bits: usize,
    /// Number of hash tables (more = better recall, more memory).
    pub num_tables: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl EncryptionConfig {
    pub fn new(dimensions: usize, hash_bits: usize) -> Self {
        Self { dimensions, hash_bits, num_tables: 4, seed: 42 }
    }

    #[must_use]
    pub fn with_tables(mut self, tables: usize) -> Self { self.num_tables = tables; self }
}

/// An encrypted vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedVector {
    pub id: String,
    pub hash_codes: Vec<u64>,
    pub original_norm: f32,
}

/// Search result from encrypted search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedSearchResult {
    pub id: String,
    pub hamming_distance: u32,
    pub estimated_similarity: f32,
}

/// LSH-based encrypted vector index.
pub struct EncryptedIndex {
    config: EncryptionConfig,
    vectors: HashMap<String, EncryptedVector>,
    /// Random hyperplanes for LSH (precomputed).
    hyperplanes: Vec<Vec<f32>>,
}

impl EncryptedIndex {
    /// Create a new encrypted index.
    pub fn new(config: EncryptionConfig) -> Self {
        let hyperplanes = Self::generate_hyperplanes(config.dimensions, config.hash_bits * config.num_tables, config.seed);
        Self { config, vectors: HashMap::new(), hyperplanes }
    }

    /// Encrypt a vector and insert it.
    pub fn encrypt_and_insert(&mut self, id: &str, vector: &[f32]) -> EncryptedVector {
        let hash_codes = self.compute_lsh_codes(vector);
        let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let enc = EncryptedVector { id: id.into(), hash_codes, original_norm: norm };
        self.vectors.insert(id.into(), enc.clone());
        enc
    }

    /// Search using encrypted comparison (Hamming distance on LSH codes).
    pub fn encrypted_search(&self, query: &[f32], k: usize) -> Vec<EncryptedSearchResult> {
        let query_codes = self.compute_lsh_codes(query);
        let mut scored: Vec<EncryptedSearchResult> = self.vectors.values()
            .map(|v| {
                let hamming = Self::hamming_distance_multi(&query_codes, &v.hash_codes);
                let max_bits = (self.config.hash_bits * self.config.num_tables) as f32;
                let similarity = 1.0 - hamming as f32 / max_bits;
                EncryptedSearchResult { id: v.id.clone(), hamming_distance: hamming, estimated_similarity: similarity }
            })
            .collect();
        scored.sort_by(|a, b| a.hamming_distance.cmp(&b.hamming_distance));
        scored.truncate(k);
        scored
    }

    /// Remove a vector.
    pub fn remove(&mut self, id: &str) -> bool { self.vectors.remove(id).is_some() }

    /// Number of indexed vectors.
    pub fn len(&self) -> usize { self.vectors.len() }
    pub fn is_empty(&self) -> bool { self.vectors.is_empty() }

    /// Configuration.
    pub fn config(&self) -> &EncryptionConfig { &self.config }

    fn compute_lsh_codes(&self, vector: &[f32]) -> Vec<u64> {
        let bits_per_code = 64;
        let total_bits = self.config.hash_bits * self.config.num_tables;
        let num_codes = (total_bits + bits_per_code - 1) / bits_per_code;
        let mut codes = vec![0u64; num_codes];

        for (bit_idx, hyperplane) in self.hyperplanes.iter().enumerate().take(total_bits) {
            let dot: f32 = vector.iter().zip(hyperplane.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                let code_idx = bit_idx / bits_per_code;
                let bit_pos = bit_idx % bits_per_code;
                codes[code_idx] |= 1 << bit_pos;
            }
        }
        codes
    }

    fn hamming_distance_multi(a: &[u64], b: &[u64]) -> u32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x ^ y).count_ones()).sum()
    }

    fn generate_hyperplanes(dims: usize, count: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut planes = Vec::with_capacity(count);
        let mut state = seed;
        for _ in 0..count {
            let mut plane = Vec::with_capacity(dims);
            for _ in 0..dims {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let val = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
                plane.push(val);
            }
            planes.push(plane);
        }
        planes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_and_search() {
        let mut idx = EncryptedIndex::new(EncryptionConfig::new(32, 16));
        idx.encrypt_and_insert("v1", &vec![1.0; 32]);
        idx.encrypt_and_insert("v2", &vec![-1.0; 32]);
        let results = idx.encrypted_search(&vec![1.0; 32], 2);
        assert_eq!(results[0].id, "v1");
        assert!(results[0].estimated_similarity > results[1].estimated_similarity);
    }

    #[test]
    fn test_hamming_distance() {
        assert_eq!(EncryptedIndex::hamming_distance_multi(&[0b1010], &[0b1001]), 2);
    }

    #[test]
    fn test_remove() {
        let mut idx = EncryptedIndex::new(EncryptionConfig::new(16, 8));
        idx.encrypt_and_insert("v1", &vec![1.0; 16]);
        assert!(idx.remove("v1"));
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn test_lsh_deterministic() {
        let idx = EncryptedIndex::new(EncryptionConfig::new(16, 8));
        let c1 = idx.compute_lsh_codes(&vec![0.5; 16]);
        let c2 = idx.compute_lsh_codes(&vec![0.5; 16]);
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_similarity_range() {
        let mut idx = EncryptedIndex::new(EncryptionConfig::new(32, 32));
        idx.encrypt_and_insert("v1", &vec![1.0; 32]);
        let results = idx.encrypted_search(&vec![1.0; 32], 1);
        assert!(results[0].estimated_similarity >= 0.0 && results[0].estimated_similarity <= 1.0);
    }
}
