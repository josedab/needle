//! Hybrid Search - Combining BM25 text search with vector similarity
//!
//! This module provides hybrid search capabilities that combine:
//! - BM25 (Best Match 25) for keyword/lexical search
//! - Vector similarity for semantic search
//! - Reciprocal Rank Fusion (RRF) for combining results

use rust_stemmers::{Algorithm, Stemmer};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// BM25 parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Config {
    /// Term frequency saturation parameter (typically 1.2-2.0)
    pub k1: f32,
    /// Length normalization parameter (typically 0.75)
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.5, b: 0.75 }
    }
}

/// RRF (Reciprocal Rank Fusion) parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RrfConfig {
    /// Constant to prevent division by zero and control rank importance
    /// Higher values give more weight to lower-ranked results
    pub k: f32,
    /// Weight for vector search results (0.0-1.0)
    pub vector_weight: f32,
    /// Weight for BM25 results (0.0-1.0)
    pub bm25_weight: f32,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self {
            k: 60.0,
            vector_weight: 0.5,
            bm25_weight: 0.5,
        }
    }
}

impl RrfConfig {
    /// Create config favoring semantic search
    pub fn semantic_focused() -> Self {
        Self {
            k: 60.0,
            vector_weight: 0.7,
            bm25_weight: 0.3,
        }
    }

    /// Create config favoring keyword search
    pub fn keyword_focused() -> Self {
        Self {
            k: 60.0,
            vector_weight: 0.3,
            bm25_weight: 0.7,
        }
    }
}

/// A document in the BM25 index
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Document {
    /// Document ID (matches vector ID)
    id: String,
    /// Term frequencies in this document
    term_freqs: HashMap<String, u32>,
    /// Total number of terms in document
    length: u32,
}

/// BM25 text index
#[derive(Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    /// Configuration
    config: Bm25Config,
    /// Documents indexed by ID
    documents: HashMap<String, Document>,
    /// Document frequency for each term (number of docs containing term)
    doc_freqs: HashMap<String, u32>,
    /// Total number of documents
    doc_count: u32,
    /// Average document length
    avg_doc_length: f32,
    /// Stop words
    #[serde(skip)]
    stop_words: HashSet<String>,
}

impl std::fmt::Debug for Bm25Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25Index")
            .field("config", &self.config)
            .field("doc_count", &self.doc_count)
            .field("avg_doc_length", &self.avg_doc_length)
            .finish()
    }
}

impl Default for Bm25Index {
    fn default() -> Self {
        Self::new(Bm25Config::default())
    }
}

impl Bm25Index {
    /// Create a new BM25 index
    pub fn new(config: Bm25Config) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            doc_freqs: HashMap::new(),
            doc_count: 0,
            avg_doc_length: 0.0,
            stop_words: Self::default_stop_words(),
        }
    }

    fn default_stop_words() -> HashSet<String> {
        [
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with", "the", "this", "but", "they",
            "have", "had", "what", "when", "where", "who", "which", "why", "how",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect()
    }

    /// Tokenize and stem text
    fn tokenize(&self, text: &str) -> Vec<String> {
        let stemmer = Stemmer::create(Algorithm::English);

        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty() && s.len() > 1)
            .filter(|s| !self.stop_words.contains(*s))
            .map(|s| stemmer.stem(s).to_string())
            .collect()
    }

    /// Add or update a document in the index
    pub fn index_document(&mut self, id: impl Into<String>, text: &str) {
        let id = id.into();
        let tokens = self.tokenize(text);

        // Remove old document if exists
        if let Some(old_doc) = self.documents.remove(&id) {
            for term in old_doc.term_freqs.keys() {
                if let Some(freq) = self.doc_freqs.get_mut(term) {
                    *freq = freq.saturating_sub(1);
                }
            }
            self.doc_count -= 1;
        }

        // Build term frequencies
        let mut term_freqs: HashMap<String, u32> = HashMap::new();
        for token in &tokens {
            *term_freqs.entry(token.clone()).or_insert(0) += 1;
        }

        // Update document frequencies
        for term in term_freqs.keys() {
            *self.doc_freqs.entry(term.clone()).or_insert(0) += 1;
        }

        let doc = Document {
            id: id.clone(),
            term_freqs,
            length: tokens.len() as u32,
        };

        self.documents.insert(id, doc);
        self.doc_count += 1;

        // Update average document length
        let total_length: u32 = self.documents.values().map(|d| d.length).sum();
        self.avg_doc_length = total_length as f32 / self.doc_count as f32;
    }

    /// Remove a document from the index
    pub fn remove_document(&mut self, id: &str) -> bool {
        if let Some(doc) = self.documents.remove(id) {
            for term in doc.term_freqs.keys() {
                if let Some(freq) = self.doc_freqs.get_mut(term) {
                    *freq = freq.saturating_sub(1);
                }
            }
            self.doc_count -= 1;

            if self.doc_count > 0 {
                let total_length: u32 = self.documents.values().map(|d| d.length).sum();
                self.avg_doc_length = total_length as f32 / self.doc_count as f32;
            } else {
                self.avg_doc_length = 0.0;
            }

            true
        } else {
            false
        }
    }

    /// Calculate BM25 score for a document given a query
    fn score_document(&self, doc: &Document, query_terms: &[String]) -> f32 {
        let k1 = self.config.k1;
        let b = self.config.b;
        let n = self.doc_count as f32;

        let mut score = 0.0;

        for term in query_terms {
            let tf = *doc.term_freqs.get(term).unwrap_or(&0) as f32;
            let df = *self.doc_freqs.get(term).unwrap_or(&0) as f32;

            if df == 0.0 || tf == 0.0 {
                continue;
            }

            // IDF component
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

            // TF component with length normalization
            let length_norm = 1.0 - b + b * (doc.length as f32 / self.avg_doc_length);
            let tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * length_norm);

            score += idf * tf_norm;
        }

        score
    }

    /// Search the index and return scored results
    pub fn search(&self, query: &str, limit: usize) -> Vec<(String, f32)> {
        let query_terms = self.tokenize(query);

        if query_terms.is_empty() {
            return Vec::new();
        }

        let mut scores: Vec<(String, f32)> = self
            .documents
            .values()
            .map(|doc| (doc.id.clone(), self.score_document(doc, &query_terms)))
            .filter(|(_, score)| *score > 0.0)
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(limit);
        scores
    }

    /// Get number of indexed documents
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Clear the index
    pub fn clear(&mut self) {
        self.documents.clear();
        self.doc_freqs.clear();
        self.doc_count = 0;
        self.avg_doc_length = 0.0;
    }
}

/// Result of hybrid search with component scores
#[derive(Debug, Clone)]
pub struct HybridSearchResult {
    /// Document ID
    pub id: String,
    /// Combined RRF score
    pub score: f32,
    /// Vector similarity score (if available)
    pub vector_score: Option<f32>,
    /// BM25 score (if available)
    pub bm25_score: Option<f32>,
    /// Rank in vector results
    pub vector_rank: Option<usize>,
    /// Rank in BM25 results
    pub bm25_rank: Option<usize>,
}

/// Combine vector and BM25 results using Reciprocal Rank Fusion
pub fn reciprocal_rank_fusion(
    vector_results: &[(String, f32)],
    bm25_results: &[(String, f32)],
    config: &RrfConfig,
    limit: usize,
) -> Vec<HybridSearchResult> {
    let mut scores: HashMap<String, HybridSearchResult> = HashMap::new();

    // Process vector results
    for (rank, (id, distance)) in vector_results.iter().enumerate() {
        let rrf_score = config.vector_weight / (config.k + rank as f32 + 1.0);
        let similarity = 1.0 / (1.0 + distance); // Convert distance to similarity

        scores
            .entry(id.clone())
            .and_modify(|r| {
                r.score += rrf_score;
                r.vector_score = Some(similarity);
                r.vector_rank = Some(rank + 1);
            })
            .or_insert(HybridSearchResult {
                id: id.clone(),
                score: rrf_score,
                vector_score: Some(similarity),
                bm25_score: None,
                vector_rank: Some(rank + 1),
                bm25_rank: None,
            });
    }

    // Process BM25 results
    for (rank, (id, bm25_score)) in bm25_results.iter().enumerate() {
        let rrf_score = config.bm25_weight / (config.k + rank as f32 + 1.0);

        scores
            .entry(id.clone())
            .and_modify(|r| {
                r.score += rrf_score;
                r.bm25_score = Some(*bm25_score);
                r.bm25_rank = Some(rank + 1);
            })
            .or_insert(HybridSearchResult {
                id: id.clone(),
                score: rrf_score,
                vector_score: None,
                bm25_score: Some(*bm25_score),
                vector_rank: None,
                bm25_rank: Some(rank + 1),
            });
    }

    // Sort by combined score
    let mut results: Vec<HybridSearchResult> = scores.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(limit);
    results
}

/// Hybrid search configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct HybridConfig {
    pub bm25: Bm25Config,
    pub rrf: RrfConfig,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bm25_basic() {
        let mut index = Bm25Index::default();

        index.index_document("doc1", "the quick brown fox jumps over the lazy dog");
        index.index_document("doc2", "machine learning and artificial intelligence");
        index.index_document("doc3", "the lazy cat sleeps all day");

        let results = index.search("lazy", 10);
        assert!(!results.is_empty());

        // Both doc1 and doc3 contain "lazy"
        let ids: Vec<&str> = results.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&"doc1"));
        assert!(ids.contains(&"doc3"));
    }

    #[test]
    fn test_bm25_relevance() {
        let mut index = Bm25Index::default();

        index.index_document("doc1", "rust programming language systems");
        index.index_document("doc2", "rust rust rust programming");
        index.index_document("doc3", "python programming language");

        let results = index.search("rust programming", 10);

        // doc2 should rank higher due to more "rust" occurrences
        assert_eq!(results[0].0, "doc2");
    }

    #[test]
    fn test_rrf_fusion() {
        let vector_results = vec![
            ("doc1".to_string(), 0.1), // closest
            ("doc2".to_string(), 0.2),
            ("doc3".to_string(), 0.3),
        ];

        let bm25_results = vec![
            ("doc2".to_string(), 5.0), // highest BM25
            ("doc1".to_string(), 3.0),
            ("doc4".to_string(), 2.0),
        ];

        let config = RrfConfig::default();
        let results = reciprocal_rank_fusion(&vector_results, &bm25_results, &config, 10);

        // doc1 and doc2 should be top since they appear in both
        let top_ids: Vec<&str> = results.iter().take(2).map(|r| r.id.as_str()).collect();
        assert!(top_ids.contains(&"doc1"));
        assert!(top_ids.contains(&"doc2"));

        // Check that scores are combined
        let doc1 = results.iter().find(|r| r.id == "doc1").unwrap();
        assert!(doc1.vector_score.is_some());
        assert!(doc1.bm25_score.is_some());
    }

    #[test]
    fn test_remove_document() {
        let mut index = Bm25Index::default();

        index.index_document("doc1", "hello world");
        index.index_document("doc2", "hello there");

        assert_eq!(index.len(), 2);

        index.remove_document("doc1");
        assert_eq!(index.len(), 1);

        let results = index.search("hello", 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc2");
    }
}
