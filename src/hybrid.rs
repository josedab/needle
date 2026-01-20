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

// ============================================================================
// Adaptive Hybrid Fusion
// ============================================================================

/// Query type classification for adaptive fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryType {
    /// Keyword-heavy query (specific terms, entities, codes)
    Keyword,
    /// Semantic query (natural language questions, concepts)
    Semantic,
    /// Mixed query (combination of keywords and semantic content)
    Mixed,
}

/// Features extracted from a query for classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryFeatures {
    /// Average word length
    pub avg_word_length: f32,
    /// Number of words
    pub word_count: usize,
    /// Ratio of stopwords
    pub stopword_ratio: f32,
    /// Contains question words (who, what, why, how, etc.)
    pub is_question: bool,
    /// Contains quoted phrases
    pub has_quotes: bool,
    /// Contains special characters or codes
    pub has_special_chars: bool,
    /// Number of capitalized words (potential entities/acronyms)
    pub capitalized_count: usize,
}

impl QueryFeatures {
    /// Extract features from a query string
    pub fn extract(query: &str) -> Self {
        let words: Vec<&str> = query.split_whitespace().collect();
        let word_count = words.len();

        let avg_word_length = if word_count > 0 {
            words.iter().map(|w| w.len()).sum::<usize>() as f32 / word_count as f32
        } else {
            0.0
        };

        let stopwords: HashSet<&str> = [
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "under", "again", "further", "then",
            "once", "and", "but", "or", "nor", "so", "yet", "both", "either",
            "neither", "not", "only", "own", "same", "than", "too", "very",
        ].into_iter().collect();

        let stopword_count = words
            .iter()
            .filter(|w| stopwords.contains(w.to_lowercase().as_str()))
            .count();
        let stopword_ratio = if word_count > 0 {
            stopword_count as f32 / word_count as f32
        } else {
            0.0
        };

        let question_words: HashSet<&str> = [
            "who", "what", "where", "when", "why", "how", "which", "whom",
            "whose", "whether", "can", "could", "would", "should", "is", "are",
            "do", "does", "did", "will",
        ].into_iter().collect();

        let is_question = words
            .first()
            .map(|w| question_words.contains(w.to_lowercase().as_str()))
            .unwrap_or(false)
            || query.trim().ends_with('?');

        let has_quotes = query.contains('"') || query.contains('\'');

        let has_special_chars = query.chars().any(|c| {
            !c.is_alphanumeric() && !c.is_whitespace() && c != '\'' && c != '"' && c != '?'
        });

        let capitalized_count = words
            .iter()
            .filter(|w| {
                w.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && w.len() > 1
            })
            .count();

        Self {
            avg_word_length,
            word_count,
            stopword_ratio,
            is_question,
            has_quotes,
            has_special_chars,
            capitalized_count,
        }
    }

    /// Classify the query type based on features
    pub fn classify(&self) -> QueryType {
        // Scoring heuristics
        let mut keyword_score: f32 = 0.0;
        let mut semantic_score: f32 = 0.0;

        // Questions tend to be semantic
        if self.is_question {
            semantic_score += 2.0;
        }

        // Quoted phrases suggest keyword search
        if self.has_quotes {
            keyword_score += 2.0;
        }

        // Special characters suggest keyword/code search
        if self.has_special_chars {
            keyword_score += 1.5;
        }

        // Short queries tend to be keyword-based
        if self.word_count <= 2 {
            keyword_score += 1.0;
        } else if self.word_count >= 5 {
            semantic_score += 1.0;
        }

        // Low stopword ratio suggests keyword search
        if self.stopword_ratio < 0.2 {
            keyword_score += 1.0;
        } else if self.stopword_ratio > 0.4 {
            semantic_score += 1.0;
        }

        // Many capitalized words suggest entities (keyword)
        if self.capitalized_count >= 2 {
            keyword_score += 1.0;
        }

        // Long average word length might indicate technical terms
        if self.avg_word_length > 7.0 {
            keyword_score += 0.5;
        }

        // Classify based on scores
        let diff = (keyword_score - semantic_score).abs();
        if diff < 1.0 {
            QueryType::Mixed
        } else if keyword_score > semantic_score {
            QueryType::Keyword
        } else {
            QueryType::Semantic
        }
    }
}

/// User feedback for a search result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchFeedback {
    /// Query that was searched
    pub query: String,
    /// Result ID that received feedback
    pub result_id: String,
    /// Whether the result was relevant (clicked, selected, etc.)
    pub relevant: bool,
    /// Position in result list where it was shown
    pub position: usize,
    /// Vector weight used for this search
    pub vector_weight: f32,
    /// BM25 weight used for this search
    pub bm25_weight: f32,
}

/// Learned weights from feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LearnedWeights {
    /// Total feedback samples
    samples: usize,
    /// Accumulated vector weight contribution
    vector_weight_sum: f32,
    /// Accumulated BM25 weight contribution
    bm25_weight_sum: f32,
}

impl Default for LearnedWeights {
    fn default() -> Self {
        Self {
            samples: 0,
            vector_weight_sum: 0.0,
            bm25_weight_sum: 0.0,
        }
    }
}

/// Adaptive Hybrid Fusion engine
///
/// Dynamically adjusts fusion weights based on:
/// 1. Query characteristics (keyword vs semantic)
/// 2. User feedback (click-through, relevance signals)
/// 3. Historical performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveFusion {
    /// Base RRF configuration
    base_config: RrfConfig,
    /// Learned weights per query type
    learned_weights: HashMap<String, LearnedWeights>,
    /// Learning rate for weight updates
    learning_rate: f32,
    /// Minimum samples before using learned weights
    min_samples: usize,
    /// Enable adaptive mode
    adaptive_enabled: bool,
}

impl Default for AdaptiveFusion {
    fn default() -> Self {
        Self {
            base_config: RrfConfig::default(),
            learned_weights: HashMap::new(),
            learning_rate: 0.1,
            min_samples: 10,
            adaptive_enabled: true,
        }
    }
}

impl AdaptiveFusion {
    /// Create a new adaptive fusion engine
    pub fn new(base_config: RrfConfig) -> Self {
        Self {
            base_config,
            ..Default::default()
        }
    }

    /// Create with custom learning parameters
    pub fn with_learning_params(mut self, learning_rate: f32, min_samples: usize) -> Self {
        self.learning_rate = learning_rate;
        self.min_samples = min_samples;
        self
    }

    /// Enable or disable adaptive mode
    pub fn set_adaptive(&mut self, enabled: bool) {
        self.adaptive_enabled = enabled;
    }

    /// Get optimal weights for a query
    pub fn get_weights(&self, query: &str) -> RrfConfig {
        if !self.adaptive_enabled {
            return self.base_config.clone();
        }

        let features = QueryFeatures::extract(query);
        let query_type = features.classify();

        // Get base weights for query type
        let (base_vector, base_bm25) = match query_type {
            QueryType::Keyword => (0.3, 0.7),
            QueryType::Semantic => (0.7, 0.3),
            QueryType::Mixed => (0.5, 0.5),
        };

        // Check for learned weights
        let query_type_key = format!("{:?}", query_type);
        let (vector_weight, bm25_weight) = if let Some(learned) = self.learned_weights.get(&query_type_key) {
            if learned.samples >= self.min_samples {
                // Use learned weights
                let avg_vector = learned.vector_weight_sum / learned.samples as f32;
                let avg_bm25 = learned.bm25_weight_sum / learned.samples as f32;

                // Blend learned with base weights
                let blend_factor = (learned.samples as f32 / (learned.samples + self.min_samples) as f32).min(0.8);
                (
                    base_vector * (1.0 - blend_factor) + avg_vector * blend_factor,
                    base_bm25 * (1.0 - blend_factor) + avg_bm25 * blend_factor,
                )
            } else {
                (base_vector, base_bm25)
            }
        } else {
            (base_vector, base_bm25)
        };

        // Normalize weights
        let total = vector_weight + bm25_weight;
        RrfConfig {
            k: self.base_config.k,
            vector_weight: vector_weight / total,
            bm25_weight: bm25_weight / total,
        }
    }

    /// Record feedback and update learned weights
    pub fn record_feedback(&mut self, feedback: SearchFeedback) {
        let features = QueryFeatures::extract(&feedback.query);
        let query_type = features.classify();
        let query_type_key = format!("{:?}", query_type);

        let learned = self.learned_weights.entry(query_type_key).or_default();

        if feedback.relevant {
            // Positive feedback - reinforce weights used
            // Apply position-based weighting (higher positions are more valuable)
            let position_weight = 1.0 / (feedback.position as f32 + 1.0);

            learned.vector_weight_sum += feedback.vector_weight * position_weight;
            learned.bm25_weight_sum += feedback.bm25_weight * position_weight;
            learned.samples += 1;
        } else {
            // Negative feedback - adjust away from weights used
            // Only apply if we have enough samples
            if learned.samples > 0 {
                let adjustment = self.learning_rate;

                // Slightly reduce the contribution of the weights that led to bad results
                learned.vector_weight_sum -= feedback.vector_weight * adjustment;
                learned.bm25_weight_sum -= feedback.bm25_weight * adjustment;

                // Ensure we don't go negative
                learned.vector_weight_sum = learned.vector_weight_sum.max(0.0);
                learned.bm25_weight_sum = learned.bm25_weight_sum.max(0.0);
            }
        }
    }

    /// Perform adaptive hybrid search
    pub fn search(
        &self,
        query: &str,
        vector_results: &[(String, f32)],
        bm25_results: &[(String, f32)],
        limit: usize,
    ) -> (Vec<HybridSearchResult>, RrfConfig) {
        let config = self.get_weights(query);
        let results = reciprocal_rank_fusion(vector_results, bm25_results, &config, limit);
        (results, config)
    }

    /// Get statistics about learned weights
    pub fn stats(&self) -> AdaptiveFusionStats {
        let mut stats = AdaptiveFusionStats {
            total_feedback: 0,
            weights_by_type: HashMap::new(),
        };

        for (query_type, learned) in &self.learned_weights {
            stats.total_feedback += learned.samples;

            if learned.samples > 0 {
                let avg_vector = learned.vector_weight_sum / learned.samples as f32;
                let avg_bm25 = learned.bm25_weight_sum / learned.samples as f32;
                let total = avg_vector + avg_bm25;

                stats.weights_by_type.insert(
                    query_type.clone(),
                    LearnedWeightStats {
                        samples: learned.samples,
                        avg_vector_weight: if total > 0.0 { avg_vector / total } else { 0.5 },
                        avg_bm25_weight: if total > 0.0 { avg_bm25 / total } else { 0.5 },
                    },
                );
            }
        }

        stats
    }

    /// Reset learned weights
    pub fn reset(&mut self) {
        self.learned_weights.clear();
    }

    /// Export learned weights for persistence
    pub fn export_weights(&self) -> String {
        serde_json::to_string(&self.learned_weights).unwrap_or_default()
    }

    /// Import learned weights from persistence
    pub fn import_weights(&mut self, data: &str) -> Result<(), serde_json::Error> {
        self.learned_weights = serde_json::from_str(data)?;
        Ok(())
    }
}

/// Statistics about learned weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveFusionStats {
    /// Total feedback samples received
    pub total_feedback: usize,
    /// Learned weights by query type
    pub weights_by_type: HashMap<String, LearnedWeightStats>,
}

/// Learned weight statistics for a query type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnedWeightStats {
    /// Number of feedback samples
    pub samples: usize,
    /// Average vector weight
    pub avg_vector_weight: f32,
    /// Average BM25 weight
    pub avg_bm25_weight: f32,
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

    // ========================================================================
    // Adaptive Hybrid Fusion Tests
    // ========================================================================

    #[test]
    fn test_query_features_extraction() {
        // Test keyword-style query
        let features = QueryFeatures::extract("API_KEY config.json");
        assert!(features.has_special_chars);
        assert_eq!(features.word_count, 2);

        // Test semantic-style query
        let features = QueryFeatures::extract("How do I implement authentication in my application?");
        assert!(features.is_question);
        assert!(features.word_count > 5);
        assert!(features.stopword_ratio > 0.2); // "How", "do", "I", "in", "my" are stopwords

        // Test quoted query
        let features = QueryFeatures::extract("\"exact phrase\" search");
        assert!(features.has_quotes);
    }

    #[test]
    fn test_query_type_classification() {
        // Keyword queries
        assert_eq!(
            QueryFeatures::extract("API_KEY").classify(),
            QueryType::Keyword
        );
        assert_eq!(
            QueryFeatures::extract("\"exact match\"").classify(),
            QueryType::Keyword
        );

        // Semantic queries
        assert_eq!(
            QueryFeatures::extract("How does machine learning work in production?").classify(),
            QueryType::Semantic
        );
        assert_eq!(
            QueryFeatures::extract("What is the best approach for error handling?").classify(),
            QueryType::Semantic
        );

        // Short ambiguous queries might be classified as keyword or mixed
        let short_query_type = QueryFeatures::extract("rust").classify();
        assert!(short_query_type == QueryType::Keyword || short_query_type == QueryType::Mixed);
    }

    #[test]
    fn test_adaptive_fusion_weights() {
        let fusion = AdaptiveFusion::default();

        // Keyword query should favor BM25
        let keyword_weights = fusion.get_weights("API_KEY config.json");
        assert!(keyword_weights.bm25_weight > keyword_weights.vector_weight);

        // Semantic query should favor vectors
        let semantic_weights = fusion.get_weights("How do I implement authentication?");
        assert!(semantic_weights.vector_weight > semantic_weights.bm25_weight);
    }

    #[test]
    fn test_adaptive_fusion_search() {
        let fusion = AdaptiveFusion::default();

        let vector_results = vec![
            ("doc1".to_string(), 0.1),
            ("doc2".to_string(), 0.2),
        ];

        let bm25_results = vec![
            ("doc2".to_string(), 5.0),
            ("doc1".to_string(), 3.0),
        ];

        let (results, config) = fusion.search(
            "How to configure settings?",
            &vector_results,
            &bm25_results,
            10,
        );

        assert!(!results.is_empty());
        assert!(config.vector_weight + config.bm25_weight > 0.99); // Normalized
    }

    #[test]
    fn test_adaptive_fusion_feedback_learning() {
        let mut fusion = AdaptiveFusion::default()
            .with_learning_params(0.1, 5); // Lower min_samples for testing

        // Record positive feedback for keyword queries
        for i in 0..10 {
            fusion.record_feedback(SearchFeedback {
                query: "API_KEY".to_string(),
                result_id: format!("doc{}", i),
                relevant: true,
                position: i,
                vector_weight: 0.3,
                bm25_weight: 0.7,
            });
        }

        let stats = fusion.stats();
        assert!(stats.total_feedback >= 10);

        // Should have learned weights for keyword type
        assert!(stats.weights_by_type.contains_key("Keyword"));
    }

    #[test]
    fn test_adaptive_fusion_weight_persistence() {
        let mut fusion = AdaptiveFusion::default();

        // Record some feedback
        fusion.record_feedback(SearchFeedback {
            query: "test query".to_string(),
            result_id: "doc1".to_string(),
            relevant: true,
            position: 0,
            vector_weight: 0.5,
            bm25_weight: 0.5,
        });

        // Export weights
        let exported = fusion.export_weights();
        assert!(!exported.is_empty());

        // Create new fusion and import
        let mut new_fusion = AdaptiveFusion::default();
        new_fusion.import_weights(&exported).unwrap();

        assert_eq!(fusion.stats().total_feedback, new_fusion.stats().total_feedback);
    }

    #[test]
    fn test_adaptive_fusion_disabled() {
        let mut fusion = AdaptiveFusion::default();
        fusion.set_adaptive(false);

        // Should return base config regardless of query
        let weights1 = fusion.get_weights("API_KEY");
        let weights2 = fusion.get_weights("How does this work?");

        assert_eq!(weights1.vector_weight, weights2.vector_weight);
        assert_eq!(weights1.bm25_weight, weights2.bm25_weight);
    }

    #[test]
    fn test_adaptive_fusion_negative_feedback() {
        let mut fusion = AdaptiveFusion::default()
            .with_learning_params(0.2, 3);

        // First add positive feedback
        for _ in 0..5 {
            fusion.record_feedback(SearchFeedback {
                query: "test query".to_string(),
                result_id: "doc1".to_string(),
                relevant: true,
                position: 0,
                vector_weight: 0.5,
                bm25_weight: 0.5,
            });
        }

        let stats_before = fusion.stats();

        // Now add negative feedback
        fusion.record_feedback(SearchFeedback {
            query: "test query".to_string(),
            result_id: "doc2".to_string(),
            relevant: false,
            position: 5,
            vector_weight: 0.5,
            bm25_weight: 0.5,
        });

        // Negative feedback doesn't increase sample count
        let stats_after = fusion.stats();
        assert_eq!(stats_before.total_feedback, stats_after.total_feedback);
    }

    #[test]
    fn test_adaptive_fusion_reset() {
        let mut fusion = AdaptiveFusion::default();

        fusion.record_feedback(SearchFeedback {
            query: "test".to_string(),
            result_id: "doc1".to_string(),
            relevant: true,
            position: 0,
            vector_weight: 0.5,
            bm25_weight: 0.5,
        });

        assert!(fusion.stats().total_feedback > 0);

        fusion.reset();
        assert_eq!(fusion.stats().total_feedback, 0);
    }
}
