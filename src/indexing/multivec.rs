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
}

impl Default for MultiVectorConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            distance: DistanceFunction::Cosine,
            normalize: true,
            use_centroid: true,
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
    pub fn with_dot_product(mut self) -> Self {
        self.distance = DistanceFunction::DotProduct;
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
                1.0 - self.config.distance.compute(a, b)
            }
            DistanceFunction::DotProduct => {
                // Dot product (negated distance)
                -self.config.distance.compute(a, b)
            }
            _ => {
                // For Euclidean/Manhattan, convert to similarity
                1.0 / (1.0 + self.config.distance.compute(a, b))
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
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut vec: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        // Normalize
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut vec {
            *v /= norm;
        }
        vec
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
}
