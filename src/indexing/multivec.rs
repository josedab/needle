//! Multi-Vector Support (ColBERT-style)
//!
//! Provides late interaction retrieval where documents are represented as
//! multiple vectors (e.g., one per token). Uses MaxSim scoring for accurate
//! semantic matching while maintaining efficiency.

use crate::distance::DistanceFunction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A multi-vector document representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVector {
    /// ID of the document
    pub id: String,
    /// Token-level vectors
    pub vectors: Vec<Vec<f32>>,
    /// Optional token text for debugging
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<String>>,
}

impl MultiVector {
    /// Create a new multi-vector document
    pub fn new(id: impl Into<String>, vectors: Vec<Vec<f32>>) -> Self {
        Self {
            id: id.into(),
            vectors,
            tokens: None,
        }
    }

    /// Create with token labels
    pub fn with_tokens(id: impl Into<String>, vectors: Vec<Vec<f32>>, tokens: Vec<String>) -> Self {
        Self {
            id: id.into(),
            vectors,
            tokens: Some(tokens),
        }
    }

    /// Number of vectors (tokens)
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get dimensions of vectors
    pub fn dimensions(&self) -> Option<usize> {
        self.vectors.first().map(|v| v.len())
    }
}

/// Multi-vector index configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorConfig {
    /// Vector dimensions
    pub dimensions: usize,
    /// Distance function for similarity
    pub distance: DistanceFunction,
    /// Whether to normalize vectors
    pub normalize: bool,
    /// Centroid pooling for initial filtering
    pub use_centroid: bool,
    /// Default candidate multiplier for two-stage search.
    /// Controls the ratio of candidates to final `k` in the first stage.
    /// Higher values improve recall at the cost of latency.
    pub default_candidate_multiplier: usize,
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            distance: DistanceFunction::Cosine,
            normalize: true,
            use_centroid: true,
            default_candidate_multiplier: 4,
        }
    }
}

impl MultiVectorConfig {
    /// Create a new config with specific dimensions
    pub fn new(dimensions: usize) -> Self {
        Self {
            dimensions,
            ..Default::default()
        }
    }

    /// Use dot product distance
    #[must_use]
    pub fn with_dot_product(mut self) -> Self {
        self.distance = DistanceFunction::DotProduct;
        self
    }

    /// Set the default candidate multiplier for two-stage search.
    #[must_use]
    pub fn with_candidate_multiplier(mut self, multiplier: usize) -> Self {
        self.default_candidate_multiplier = multiplier.max(1);
        self
    }
}

/// Multi-vector index for ColBERT-style retrieval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiVectorIndex {
    /// Configuration
    config: MultiVectorConfig,
    /// Documents indexed by ID
    documents: HashMap<String, MultiVector>,
    /// Centroid vectors for fast filtering (mean of all token vectors)
    centroids: HashMap<String, Vec<f32>>,
}

impl MultiVectorIndex {
    /// Create a new multi-vector index
    pub fn new(config: MultiVectorConfig) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            centroids: HashMap::new(),
        }
    }

    /// Create with default config
    pub fn with_dimensions(dimensions: usize) -> Self {
        Self::new(MultiVectorConfig::new(dimensions))
    }

    /// Insert a multi-vector document
    pub fn insert(&mut self, doc: MultiVector) -> Result<(), String> {
        // Validate dimensions
        for (i, vec) in doc.vectors.iter().enumerate() {
            if vec.len() != self.config.dimensions {
                return Err(format!(
                    "Vector {} has {} dimensions, expected {}",
                    i,
                    vec.len(),
                    self.config.dimensions
                ));
            }
        }

        // Compute centroid if enabled
        if self.config.use_centroid && !doc.vectors.is_empty() {
            let centroid = Self::compute_centroid(&doc.vectors);
            self.centroids.insert(doc.id.clone(), centroid);
        }

        self.documents.insert(doc.id.clone(), doc);
        Ok(())
    }

    /// Remove a document by ID
    pub fn remove(&mut self, id: &str) -> bool {
        self.centroids.remove(id);
        self.documents.remove(id).is_some()
    }

    /// Get a document by ID
    pub fn get(&self, id: &str) -> Option<&MultiVector> {
        self.documents.get(id)
    }

    /// Number of documents
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Compute centroid (mean) of vectors
    fn compute_centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
        if vectors.is_empty() {
            return Vec::new();
        }

        let dims = vectors[0].len();
        let mut centroid = vec![0.0; dims];
        let n = vectors.len() as f32;

        for vec in vectors {
            for (i, &v) in vec.iter().enumerate() {
                centroid[i] += v;
            }
        }

        for v in &mut centroid {
            *v /= n;
        }

        centroid
    }

    /// Compute MaxSim score between query and document
    /// For each query vector, find max similarity to any document vector,
    /// then sum all max similarities
    fn max_sim_score(&self, query: &[Vec<f32>], doc: &MultiVector) -> f32 {
        let mut total_score = 0.0;

        for q_vec in query {
            let mut max_sim = f32::NEG_INFINITY;

            for d_vec in &doc.vectors {
                let sim = self.similarity(q_vec, d_vec);
                if sim > max_sim {
                    max_sim = sim;
                }
            }

            if max_sim > f32::NEG_INFINITY {
                total_score += max_sim;
            }
        }

        total_score
    }

    /// Compute similarity based on distance function
    fn similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        match self.config.distance {
            DistanceFunction::Cosine => {
                // Cosine similarity = 1 - cosine_distance
                1.0 - self.config.distance.compute(a, b).unwrap_or(0.0)
            }
            DistanceFunction::DotProduct => {
                // Dot product (negated distance)
                -self.config.distance.compute(a, b).unwrap_or(0.0)
            }
            _ => {
                // For Euclidean/Manhattan, convert to similarity
                1.0 / (1.0 + self.config.distance.compute(a, b).unwrap_or(0.0))
            }
        }
    }

    /// Search for similar documents using MaxSim
    pub fn search(&self, query: &[Vec<f32>], k: usize) -> Vec<MultiVectorSearchResult> {
        // Validate query dimensions
        for q_vec in query {
            if q_vec.len() != self.config.dimensions {
                return Vec::new();
            }
        }

        let mut results: Vec<MultiVectorSearchResult> = self
            .documents
            .values()
            .map(|doc| {
                let score = self.max_sim_score(query, doc);
                MultiVectorSearchResult {
                    id: doc.id.clone(),
                    score,
                    num_tokens: doc.len(),
                }
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Two-stage search: filter by centroid, then refine with MaxSim
    pub fn search_two_stage(
        &self,
        query: &[Vec<f32>],
        k: usize,
        candidate_multiplier: usize,
    ) -> Vec<MultiVectorSearchResult> {
        if !self.config.use_centroid || query.is_empty() {
            return self.search(query, k);
        }

        // Compute query centroid
        let query_centroid = Self::compute_centroid(query);

        // First stage: find top candidates by centroid similarity
        let num_candidates = (k * candidate_multiplier).min(self.documents.len());
        let mut candidates: Vec<(&String, f32)> = self
            .centroids
            .iter()
            .map(|(id, centroid)| (id, self.similarity(&query_centroid, centroid)))
            .collect();

        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(num_candidates);

        // Second stage: rerank with MaxSim
        let mut results: Vec<MultiVectorSearchResult> = candidates
            .iter()
            .filter_map(|(id, _)| {
                self.documents.get(*id).map(|doc| {
                    let score = self.max_sim_score(query, doc);
                    MultiVectorSearchResult {
                        id: doc.id.clone(),
                        score,
                        num_tokens: doc.len(),
                    }
                })
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        results
    }

    /// Two-stage search using the configured default candidate multiplier.
    ///
    /// Equivalent to calling
    /// `search_two_stage(query, k, config.default_candidate_multiplier)`.
    pub fn search_auto(&self, query: &[Vec<f32>], k: usize) -> Vec<MultiVectorSearchResult> {
        self.search_two_stage(query, k, self.config.default_candidate_multiplier)
    }

    /// Get detailed matching info between query and document
    pub fn explain_match(&self, query: &[Vec<f32>], doc_id: &str) -> Option<MatchExplanation> {
        let doc = self.documents.get(doc_id)?;

        let mut token_matches = Vec::new();
        let mut total_score = 0.0;

        for (q_idx, q_vec) in query.iter().enumerate() {
            let mut best_match = TokenMatch {
                query_index: q_idx,
                doc_index: 0,
                similarity: f32::NEG_INFINITY,
                query_token: None,
                doc_token: None,
            };

            for (d_idx, d_vec) in doc.vectors.iter().enumerate() {
                let sim = self.similarity(q_vec, d_vec);
                if sim > best_match.similarity {
                    best_match.similarity = sim;
                    best_match.doc_index = d_idx;
                }
            }

            if best_match.similarity > f32::NEG_INFINITY {
                if let Some(tokens) = &doc.tokens {
                    best_match.doc_token = tokens.get(best_match.doc_index).cloned();
                }
                total_score += best_match.similarity;
                token_matches.push(best_match);
            }
        }

        Some(MatchExplanation {
            doc_id: doc_id.to_string(),
            total_score,
            token_matches,
        })
    }
}

/// Result from multi-vector search
#[derive(Debug, Clone)]
pub struct MultiVectorSearchResult {
    /// Document ID
    pub id: String,
    /// MaxSim score
    pub score: f32,
    /// Number of tokens in document
    pub num_tokens: usize,
}

/// Token-level match information
#[derive(Debug, Clone)]
pub struct TokenMatch {
    /// Index of query token
    pub query_index: usize,
    /// Index of best matching document token
    pub doc_index: usize,
    /// Similarity score
    pub similarity: f32,
    /// Query token text (if available)
    pub query_token: Option<String>,
    /// Document token text (if available)
    pub doc_token: Option<String>,
}

/// Explanation of how query matched a document
#[derive(Debug, Clone)]
pub struct MatchExplanation {
    /// Document ID
    pub doc_id: String,
    /// Total MaxSim score
    pub total_score: f32,
    /// Token-level matches
    pub token_matches: Vec<TokenMatch>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vector(dim: usize) -> Vec<f32> {
        crate::test_utils::normalized_vector(dim)
    }

    #[test]
    fn test_multi_vector_creation() {
        let doc = MultiVector::new(
            "doc1",
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        );

        assert_eq!(doc.id, "doc1");
        assert_eq!(doc.len(), 3);
        assert_eq!(doc.dimensions(), Some(3));
    }

    #[test]
    fn test_multi_vector_index() {
        let mut index = MultiVectorIndex::with_dimensions(3);

        index
            .insert(MultiVector::new(
                "doc1",
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            ))
            .unwrap();

        index
            .insert(MultiVector::new(
                "doc2",
                vec![vec![0.0, 0.0, 1.0], vec![0.5, 0.5, 0.0]],
            ))
            .unwrap();

        assert_eq!(index.len(), 2);
        assert!(index.get("doc1").is_some());
    }

    #[test]
    fn test_maxsim_search() {
        let mut index = MultiVectorIndex::with_dimensions(3);

        // Document 1: vectors in x and y directions
        index
            .insert(MultiVector::new(
                "doc1",
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            ))
            .unwrap();

        // Document 2: vector in z direction
        index
            .insert(MultiVector::new("doc2", vec![vec![0.0, 0.0, 1.0]]))
            .unwrap();

        // Query with x direction
        let query = vec![vec![1.0, 0.0, 0.0]];
        let results = index.search(&query, 2);

        // doc1 should score higher (has x-direction vector)
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_two_stage_search() {
        let mut index = MultiVectorIndex::with_dimensions(32);

        // Add some documents
        for i in 0..20 {
            let vectors: Vec<Vec<f32>> = (0..5).map(|_| random_vector(32)).collect();
            index
                .insert(MultiVector::new(format!("doc{}", i), vectors))
                .unwrap();
        }

        let query: Vec<Vec<f32>> = (0..3).map(|_| random_vector(32)).collect();

        // Two-stage should return same top results as full search
        let full_results = index.search(&query, 5);
        let two_stage_results = index.search_two_stage(&query, 5, 3);

        // Top result should likely be the same
        assert!(!full_results.is_empty());
        assert!(!two_stage_results.is_empty());
    }

    #[test]
    fn test_explain_match() {
        let mut index = MultiVectorIndex::with_dimensions(3);

        let doc = MultiVector::with_tokens(
            "doc1",
            vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            vec!["hello".to_string(), "world".to_string()],
        );
        index.insert(doc).unwrap();

        let query = vec![vec![0.9, 0.1, 0.0]]; // Similar to "hello" vector
        let explanation = index.explain_match(&query, "doc1").unwrap();

        assert_eq!(explanation.doc_id, "doc1");
        assert_eq!(explanation.token_matches.len(), 1);
        // Should match "hello" (index 0) best
        assert_eq!(explanation.token_matches[0].doc_index, 0);
        assert_eq!(
            explanation.token_matches[0].doc_token,
            Some("hello".to_string())
        );
    }

    #[test]
    fn test_centroid_computation() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let centroid = MultiVectorIndex::compute_centroid(&vectors);

        // Centroid should be roughly [1/3, 1/3, 1/3]
        assert!((centroid[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((centroid[1] - 1.0 / 3.0).abs() < 1e-6);
        assert!((centroid[2] - 1.0 / 3.0).abs() < 1e-6);
    }

    // ========================================================================
    // Extended multi-vector tests
    // ========================================================================

    #[test]
    fn test_with_tokens_constructor() {
        let doc = MultiVector::with_tokens(
            "doc1",
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec!["hello".to_string(), "world".to_string()],
        );

        assert_eq!(doc.len(), 2);
        assert_eq!(doc.dimensions(), Some(2));
        assert_eq!(doc.tokens.as_ref().unwrap().len(), 2);
        assert!(!doc.is_empty());
    }

    #[test]
    fn test_empty_multi_vector() {
        let doc = MultiVector::new("empty", vec![]);
        assert!(doc.is_empty());
        assert_eq!(doc.len(), 0);
        assert_eq!(doc.dimensions(), None);
    }

    #[test]
    fn test_with_dot_product_config() {
        let config = MultiVectorConfig::new(3).with_dot_product();
        assert_eq!(config.distance, DistanceFunction::DotProduct);
        assert_eq!(config.dimensions, 3);
    }

    #[test]
    fn test_insert_dimension_mismatch() {
        let mut index = MultiVectorIndex::with_dimensions(3);
        let doc = MultiVector::new("bad", vec![vec![1.0, 2.0]]); // 2 dims, expected 3
        let result = index.insert(doc);
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut index = MultiVectorIndex::with_dimensions(3);
        assert!(!index.remove("nonexistent"));
    }

    #[test]
    fn test_remove_existing() {
        let mut index = MultiVectorIndex::with_dimensions(3);
        index
            .insert(MultiVector::new("doc1", vec![vec![1.0, 0.0, 0.0]]))
            .unwrap();
        assert_eq!(index.len(), 1);

        assert!(index.remove("doc1"));
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
        assert!(index.get("doc1").is_none());
    }

    #[test]
    fn test_variable_token_counts() {
        let mut index = MultiVectorIndex::with_dimensions(3);

        // Document with 1 token
        index
            .insert(MultiVector::new("doc1", vec![vec![1.0, 0.0, 0.0]]))
            .unwrap();
        // Document with 5 tokens
        let vecs: Vec<Vec<f32>> = (0..5).map(|_| random_vector(3)).collect();
        index
            .insert(MultiVector::new("doc5", vecs))
            .unwrap();

        assert_eq!(index.len(), 2);
        let results = index.search(&vec![vec![1.0, 0.0, 0.0]], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_empty_query() {
        let mut index = MultiVectorIndex::with_dimensions(3);
        index
            .insert(MultiVector::new(
                "doc1",
                vec![vec![1.0, 0.0, 0.0]],
            ))
            .unwrap();

        // Empty query falls through to two_stage's search call
        let results = index.search(&[], 5);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_wrong_query_dims() {
        let mut index = MultiVectorIndex::with_dimensions(3);
        index
            .insert(MultiVector::new("doc1", vec![vec![1.0, 0.0, 0.0]]))
            .unwrap();

        // Query with wrong dimensions
        let results = index.search(&vec![vec![1.0, 0.0]], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_explain_match_missing_doc() {
        let index = MultiVectorIndex::with_dimensions(3);
        let result = index.explain_match(&vec![vec![1.0, 0.0, 0.0]], "nonexistent");
        assert!(result.is_none());
    }

    #[test]
    fn test_search_two_stage_empty_query() {
        let mut index = MultiVectorIndex::with_dimensions(3);
        index
            .insert(MultiVector::new("doc1", vec![vec![1.0, 0.0, 0.0]]))
            .unwrap();

        let results = index.search_two_stage(&[], 5, 3);
        // Empty query with centroid enabled falls back to search
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_centroid_empty_vectors() {
        let centroid = MultiVectorIndex::compute_centroid(&[]);
        assert!(centroid.is_empty());
    }

    #[test]
    fn test_dot_product_index_search() {
        let config = MultiVectorConfig::new(3).with_dot_product();
        let mut index = MultiVectorIndex::new(config);

        index
            .insert(MultiVector::new(
                "doc1",
                vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
            ))
            .unwrap();
        index
            .insert(MultiVector::new("doc2", vec![vec![0.0, 0.0, 1.0]]))
            .unwrap();

        let query = vec![vec![1.0, 0.0, 0.0]];
        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_index_overwrite_doc() {
        let mut index = MultiVectorIndex::with_dimensions(3);

        index
            .insert(MultiVector::new("doc1", vec![vec![1.0, 0.0, 0.0]]))
            .unwrap();
        // Insert again with same ID - should overwrite
        index
            .insert(MultiVector::new("doc1", vec![vec![0.0, 1.0, 0.0]]))
            .unwrap();

        assert_eq!(index.len(), 1);
        let doc = index.get("doc1").unwrap();
        assert_eq!(doc.vectors[0], vec![0.0, 1.0, 0.0]);
    }

    // ========================================================================
    // Two-stage search quality validation
    // ========================================================================

    /// Measure recall: fraction of brute-force top-k results found by two-stage.
    fn recall(brute: &[MultiVectorSearchResult], two_stage: &[MultiVectorSearchResult]) -> f64 {
        let brute_ids: std::collections::HashSet<_> = brute.iter().map(|r| &r.id).collect();
        let hits = two_stage.iter().filter(|r| brute_ids.contains(&r.id)).count();
        if brute_ids.is_empty() {
            1.0
        } else {
            hits as f64 / brute_ids.len() as f64
        }
    }

    #[test]
    fn test_two_stage_recall_small() {
        let dim = 16;
        let mut index = MultiVectorIndex::with_dimensions(dim);

        // Insert 50 docs with 3-5 token vectors each
        for i in 0..50 {
            let ntokens = 3 + (i % 3);
            let vecs: Vec<Vec<f32>> = (0..ntokens).map(|_| random_vector(dim)).collect();
            index.insert(MultiVector::new(format!("d{i}"), vecs)).unwrap();
        }

        let k = 5;
        let query: Vec<Vec<f32>> = (0..3).map(|_| random_vector(dim)).collect();

        let brute_results = index.search(&query, k);
        let two_stage_results = index.search_two_stage(&query, k, 4);

        // Verify two-stage returns k results
        assert_eq!(two_stage_results.len(), k);
        // Verify results are a subset of all documents
        for r in &two_stage_results {
            assert!(index.get(&r.id).is_some());
        }
        // With 4× candidates, recall is typically good (not perfect on random data)
        let r = recall(&brute_results, &two_stage_results);
        assert!(
            r >= 0.4,
            "Two-stage recall {r:.2} is unexpectedly low (expected >= 0.4)"
        );
    }

    #[test]
    fn test_two_stage_recall_medium() {
        let dim = 32;
        let mut index = MultiVectorIndex::with_dimensions(dim);

        // Insert 200 docs
        for i in 0..200 {
            let ntokens = 2 + (i % 5);
            let vecs: Vec<Vec<f32>> = (0..ntokens).map(|_| random_vector(dim)).collect();
            index.insert(MultiVector::new(format!("d{i}"), vecs)).unwrap();
        }

        let k = 10;
        let query: Vec<Vec<f32>> = (0..4).map(|_| random_vector(dim)).collect();

        let brute_results = index.search(&query, k);

        // With a large multiplier, recall should improve
        let large_results = index.search_two_stage(&query, k, 10);
        let r = recall(&brute_results, &large_results);
        assert!(
            r >= 0.4,
            "Two-stage recall {r:.2} with mult=10 too low (expected >= 0.4)"
        );
    }

    #[test]
    fn test_two_stage_score_ordering() {
        let dim = 8;
        let mut index = MultiVectorIndex::with_dimensions(dim);

        for i in 0..30 {
            let vecs: Vec<Vec<f32>> = (0..3).map(|_| random_vector(dim)).collect();
            index.insert(MultiVector::new(format!("d{i}"), vecs)).unwrap();
        }

        let query: Vec<Vec<f32>> = (0..2).map(|_| random_vector(dim)).collect();
        let results = index.search_two_stage(&query, 10, 3);

        // Verify scores are in descending order
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Results not sorted: {} > {}",
                w[0].score,
                w[1].score
            );
        }
    }

    #[test]
    fn test_two_stage_full_candidate_set_equals_brute_force() {
        let dim = 8;
        let n = 20;
        let mut index = MultiVectorIndex::with_dimensions(dim);

        for i in 0..n {
            let vecs = vec![random_vector(dim), random_vector(dim)];
            index.insert(MultiVector::new(format!("d{i}"), vecs)).unwrap();
        }

        let query = vec![random_vector(dim)];
        let k = 5;

        // When candidate_multiplier is large enough to cover all docs, recall is 100%
        let brute = index.search(&query, k);
        let two_stage = index.search_two_stage(&query, k, n);

        let r = recall(&brute, &two_stage);
        assert!(
            (r - 1.0).abs() < f64::EPSILON,
            "Full candidate set should give perfect recall, got {r:.2}"
        );
    }

    // ========================================================================
    // Config builder and search_auto
    // ========================================================================

    #[test]
    fn test_config_default_candidate_multiplier() {
        let config = MultiVectorConfig::new(128);
        assert_eq!(config.default_candidate_multiplier, 4);
    }

    #[test]
    fn test_config_with_candidate_multiplier() {
        let config = MultiVectorConfig::new(128).with_candidate_multiplier(8);
        assert_eq!(config.default_candidate_multiplier, 8);
    }

    #[test]
    fn test_config_candidate_multiplier_clamps_to_one() {
        let config = MultiVectorConfig::new(128).with_candidate_multiplier(0);
        assert_eq!(config.default_candidate_multiplier, 1);
    }

    #[test]
    fn test_search_auto_uses_config() {
        let dim = 8;
        let config = MultiVectorConfig::new(dim).with_candidate_multiplier(10);
        let mut index = MultiVectorIndex::new(config);

        for i in 0..30 {
            let vecs = vec![random_vector(dim), random_vector(dim)];
            index.insert(MultiVector::new(format!("d{i}"), vecs)).unwrap();
        }

        let query = vec![random_vector(dim)];
        let auto_results = index.search_auto(&query, 5);
        let manual_results = index.search_two_stage(&query, 5, 10);

        // search_auto with mult=10 should give same results as manual two-stage with mult=10
        assert_eq!(auto_results.len(), manual_results.len());
        for (a, m) in auto_results.iter().zip(manual_results.iter()) {
            assert_eq!(a.id, m.id);
            assert!((a.score - m.score).abs() < 1e-6);
        }
    }
}
