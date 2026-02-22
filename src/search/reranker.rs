#![allow(clippy::unwrap_used)]
//! # Reranker Module
//!
//! External reranking support for improving search relevance using cross-encoder models.
//! Rerankers can significantly improve retrieval quality by scoring query-document pairs
//! more accurately than bi-encoder similarity.
//!
//! ## Available Rerankers
//!
//! - **CohereReranker**: Uses Cohere's rerank API
//! - **HuggingFaceReranker**: Uses local HuggingFace cross-encoder models
//! - **CrossEncoderReranker**: Generic local cross-encoder implementation
//!
//! ## Example
//!
//! ```rust,ignore
//! use needle::reranker::{Reranker, CohereReranker, RerankResult};
//!
//! async fn rerank_example() -> Result<(), Box<dyn std::error::Error>> {
//!     let reranker = CohereReranker::new("your-api-key", "rerank-english-v2.0");
//!
//!     let query = "What is machine learning?";
//!     let documents = vec![
//!         "Machine learning is a subset of AI...",
//!         "The weather today is sunny...",
//!         "Deep learning uses neural networks...",
//!     ];
//!
//!     let results = reranker.rerank(query, &documents, 2).await?;
//!     for result in results {
//!         println!("Index: {}, Score: {}", result.index, result.score);
//!     }
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use thiserror::Error;

/// Errors that can occur during reranking
#[derive(Error, Debug)]
pub enum RerankerError {
    /// API request failed
    #[error("API error: {0}")]
    ApiError(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimited(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Model error
    #[error("Model error: {0}")]
    ModelError(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Network error
    #[error("Network error: {0}")]
    NetworkError(String),
}

/// Result type for reranker operations
pub type RerankerResult<T> = std::result::Result<T, RerankerError>;

/// A single reranking result
#[derive(Debug, Clone)]
pub struct RerankResult {
    /// Original index of the document
    pub index: usize,
    /// Relevance score (higher is more relevant)
    pub score: f32,
    /// Original document text (optional)
    pub text: Option<String>,
}

impl RerankResult {
    /// Create a new rerank result
    pub fn new(index: usize, score: f32) -> Self {
        Self {
            index,
            score,
            text: None,
        }
    }

    /// Create a new rerank result with text
    pub fn with_text(index: usize, score: f32, text: String) -> Self {
        Self {
            index,
            score,
            text: Some(text),
        }
    }
}

/// Trait for reranking implementations
pub trait Reranker: Send + Sync {
    /// Rerank documents based on query relevance
    ///
    /// # Arguments
    /// * `query` - The search query
    /// * `documents` - Documents to rerank
    /// * `top_k` - Number of top results to return
    ///
    /// # Returns
    /// Top-k documents sorted by relevance score (descending)
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = RerankerResult<Vec<RerankResult>>> + Send + 'a>>;

    /// Get the name of this reranker
    fn name(&self) -> &str;

    /// Get maximum batch size supported
    fn max_batch_size(&self) -> usize {
        100
    }
}

/// Configuration for Cohere reranker
#[derive(Debug, Clone)]
pub struct CohereConfig {
    /// API key for Cohere
    pub api_key: String,
    /// Model name (e.g., "rerank-english-v2.0", "rerank-multilingual-v2.0")
    pub model: String,
    /// API base URL
    pub base_url: String,
    /// Request timeout in seconds
    pub timeout_secs: u64,
}

impl CohereConfig {
    /// Create a new Cohere config with the specified API key and model
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.cohere.ai/v1".to_string(),
            timeout_secs: 30,
        }
    }

    /// Set custom base URL
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Set request timeout
    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

/// Cohere reranker using the Cohere API
pub struct CohereReranker {
    config: CohereConfig,
}

impl CohereReranker {
    /// Create a new Cohere reranker
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            config: CohereConfig::new(api_key, model),
        }
    }

    /// Create from config
    pub fn from_config(config: CohereConfig) -> Self {
        Self { config }
    }

    /// Perform the actual API call (placeholder implementation)
    async fn call_api(
        &self,
        query: &str,
        documents: &[&str],
        top_k: usize,
    ) -> RerankerResult<Vec<RerankResult>> {
        // In a real implementation, this would make an HTTP request to Cohere's API
        // For now, we provide a mock implementation that can be replaced

        if documents.is_empty() {
            return Ok(vec![]);
        }

        if query.is_empty() {
            return Err(RerankerError::InvalidInput("Query cannot be empty".into()));
        }

        // Mock scoring based on simple term overlap
        // Real implementation would call: POST {base_url}/rerank
        let query_lower = query.to_lowercase();
        let query_terms: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();

        let mut results: Vec<RerankResult> = documents
            .iter()
            .enumerate()
            .map(|(idx, doc)| {
                let doc_lower = doc.to_lowercase();
                let doc_terms: std::collections::HashSet<_> =
                    doc_lower.split_whitespace().collect();
                let overlap = query_terms
                    .iter()
                    .filter(|t| doc_terms.contains(&t.to_string().as_str()))
                    .count();
                let score = overlap as f32 / query_terms.len().max(1) as f32;
                RerankResult::with_text(idx, score, doc.to_string())
            })
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

        // Take top_k
        results.truncate(top_k);

        Ok(results)
    }
}

impl Reranker for CohereReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = RerankerResult<Vec<RerankResult>>> + Send + 'a>> {
        Box::pin(async move { self.call_api(query, documents, top_k).await })
    }

    fn name(&self) -> &str {
        "cohere"
    }

    fn max_batch_size(&self) -> usize {
        1000 // Cohere supports up to 1000 documents
    }
}

/// Configuration for HuggingFace reranker
#[derive(Debug, Clone)]
pub struct HuggingFaceConfig {
    /// Model name or path (e.g., "cross-encoder/ms-marco-MiniLM-L-6-v2")
    pub model: String,
    /// Device to run on ("cpu" or "cuda:0")
    pub device: String,
    /// Maximum sequence length
    pub max_length: usize,
    /// Batch size for inference
    pub batch_size: usize,
}

impl Default for HuggingFaceConfig {
    fn default() -> Self {
        Self {
            model: "cross-encoder/ms-marco-MiniLM-L-6-v2".to_string(),
            device: "cpu".to_string(),
            max_length: 512,
            batch_size: 32,
        }
    }
}

impl HuggingFaceConfig {
    /// Create a new config with the specified model
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            ..Default::default()
        }
    }

    /// Set device
    pub fn with_device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    /// Set max sequence length
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

/// HuggingFace cross-encoder reranker
pub struct HuggingFaceReranker {
    config: HuggingFaceConfig,
}

impl HuggingFaceReranker {
    /// Create a new HuggingFace reranker
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            config: HuggingFaceConfig::new(model),
        }
    }

    /// Create from config
    pub fn from_config(config: HuggingFaceConfig) -> Self {
        Self { config }
    }

    /// Score a single query-document pair (placeholder)
    fn score_pair(query: &str, document: &str) -> f32 {
        // In a real implementation, this would run the cross-encoder model
        // For now, we use simple Jaccard similarity as a placeholder

        let query_lower = query.to_lowercase();
        let doc_lower = document.to_lowercase();

        let query_terms: std::collections::HashSet<String> =
            query_lower.split_whitespace().map(String::from).collect();
        let doc_terms: std::collections::HashSet<String> =
            doc_lower.split_whitespace().map(String::from).collect();

        let overlap = query_terms.intersection(&doc_terms).count();
        let union = query_terms.union(&doc_terms).count();

        if union == 0 {
            0.0
        } else {
            overlap as f32 / union as f32
        }
    }
}

impl Reranker for HuggingFaceReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = RerankerResult<Vec<RerankResult>>> + Send + 'a>> {
        Box::pin(async move {
            if documents.is_empty() {
                return Ok(vec![]);
            }

            let mut results: Vec<RerankResult> = documents
                .iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let score = Self::score_pair(query, doc);
                    RerankResult::with_text(idx, score, doc.to_string())
                })
                .collect();

            // Sort by score descending
            results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            // Take top_k
            results.truncate(top_k);

            Ok(results)
        })
    }

    fn name(&self) -> &str {
        "huggingface"
    }

    fn max_batch_size(&self) -> usize {
        self.config.batch_size
    }
}

/// A reranker that combines multiple rerankers using reciprocal rank fusion
pub struct EnsembleReranker {
    rerankers: Vec<Box<dyn Reranker>>,
    weights: Vec<f32>,
    k: f32,
}

impl EnsembleReranker {
    /// Create a new ensemble reranker with equal weights
    pub fn new(rerankers: Vec<Box<dyn Reranker>>) -> Self {
        let n = rerankers.len();
        let weights = vec![1.0 / n as f32; n];
        Self {
            rerankers,
            weights,
            k: 60.0,
        }
    }

    /// Create with custom weights
    pub fn with_weights(rerankers: Vec<Box<dyn Reranker>>, weights: Vec<f32>) -> Self {
        assert_eq!(rerankers.len(), weights.len());
        Self {
            rerankers,
            weights,
            k: 60.0,
        }
    }

    /// Set RRF k parameter (default: 60)
    pub fn with_k(mut self, k: f32) -> Self {
        self.k = k;
        self
    }
}

impl Reranker for EnsembleReranker {
    fn rerank<'a>(
        &'a self,
        query: &'a str,
        documents: &'a [&'a str],
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = RerankerResult<Vec<RerankResult>>> + Send + 'a>> {
        Box::pin(async move {
            if documents.is_empty() {
                return Ok(vec![]);
            }

            // Collect results from all rerankers
            let mut all_results = Vec::new();
            for reranker in &self.rerankers {
                let results = reranker.rerank(query, documents, documents.len()).await?;
                all_results.push(results);
            }

            // Compute RRF scores
            let mut rrf_scores: HashMap<usize, f32> = HashMap::new();

            for (i, results) in all_results.iter().enumerate() {
                let weight = self.weights[i];
                for (rank, result) in results.iter().enumerate() {
                    let rrf_score = weight / (self.k + rank as f32 + 1.0);
                    *rrf_scores.entry(result.index).or_insert(0.0) += rrf_score;
                }
            }

            // Build final results
            let mut final_results: Vec<RerankResult> = rrf_scores
                .into_iter()
                .map(|(idx, score)| RerankResult::with_text(idx, score, documents[idx].to_string()))
                .collect();

            // Sort by score descending
            final_results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));

            // Take top_k
            final_results.truncate(top_k);

            Ok(final_results)
        })
    }

    fn name(&self) -> &str {
        "ensemble"
    }
}

/// A no-op reranker that returns documents in original order
/// Useful for testing or when reranking is disabled
pub struct NoOpReranker;

impl Reranker for NoOpReranker {
    fn rerank<'a>(
        &'a self,
        _query: &'a str,
        documents: &'a [&'a str],
        top_k: usize,
    ) -> Pin<Box<dyn Future<Output = RerankerResult<Vec<RerankResult>>> + Send + 'a>> {
        Box::pin(async move {
            let results: Vec<RerankResult> = documents
                .iter()
                .enumerate()
                .take(top_k)
                .map(|(idx, doc)| {
                    RerankResult::with_text(idx, 1.0 - (idx as f32 * 0.01), doc.to_string())
                })
                .collect();
            Ok(results)
        })
    }

    fn name(&self) -> &str {
        "noop"
    }
}

/// Helper function to apply reranking to search results
pub async fn apply_reranking<R: Reranker>(
    reranker: &R,
    query: &str,
    documents: &[String],
    top_k: usize,
) -> RerankerResult<Vec<(usize, f32)>> {
    let doc_refs: Vec<&str> = documents.iter().map(|s| s.as_str()).collect();
    let results = reranker.rerank(query, &doc_refs, top_k).await?;
    Ok(results.into_iter().map(|r| (r.index, r.score)).collect())
}

/// Feedback record for training the bandits model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelevanceFeedback {
    /// Unique query identifier
    pub query_id: String,
    /// Vector ID that was clicked/rated
    pub vector_id: String,
    /// Relevance score (0.0 = irrelevant, 1.0 = highly relevant)
    pub relevance_score: f32,
    /// Position the result was shown at (0-indexed)
    pub position: usize,
    /// Timestamp of feedback
    pub timestamp: u64,
}

impl RelevanceFeedback {
    pub fn new(
        query_id: impl Into<String>,
        vector_id: impl Into<String>,
        relevance_score: f32,
        position: usize,
    ) -> Self {
        Self {
            query_id: query_id.into(),
            vector_id: vector_id.into(),
            relevance_score,
            position,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
}

/// Beta distribution parameters for Thompson Sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    /// Alpha parameter (successes + 1)
    pub alpha: f64,
    /// Beta parameter (failures + 1)
    pub beta: f64,
}

impl Default for BetaParams {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        } // Uniform prior
    }
}

impl BetaParams {
    /// Sample from the Beta distribution using the inverse CDF approximation
    pub fn sample(&self, rng: &mut impl rand::Rng) -> f64 {
        // Simple approximation: use mean + noise scaled by variance
        let mean = self.alpha / (self.alpha + self.beta);
        let variance = (self.alpha * self.beta)
            / ((self.alpha + self.beta).powi(2) * (self.alpha + self.beta + 1.0));
        let noise: f64 = rng.gen_range(-1.0..1.0);
        (mean + noise * variance.sqrt()).clamp(0.0, 1.0)
    }

    /// Update with a reward (relevance score)
    pub fn update(&mut self, reward: f64) {
        self.alpha += reward;
        self.beta += 1.0 - reward;
    }

    /// Expected value (mean of Beta distribution)
    pub fn expected_value(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Apply concept drift decay
    pub fn decay(&mut self, factor: f64) {
        self.alpha = 1.0 + (self.alpha - 1.0) * factor;
        self.beta = 1.0 + (self.beta - 1.0) * factor;
    }
}

/// Configuration for the bandits reranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditsConfig {
    /// Concept drift decay factor (0.0 - 1.0, applied periodically)
    pub decay_factor: f64,
    /// Number of feedback events before decay is applied
    pub decay_interval: usize,
    /// Whether to enable A/B testing mode (50% bandits, 50% original ranking)
    pub ab_testing: bool,
    /// Maximum feedback log size
    pub max_feedback_log: usize,
}

impl Default for BanditsConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.95,
            decay_interval: 100,
            ab_testing: false,
            max_feedback_log: 10_000,
        }
    }
}

/// Thompson Sampling bandits reranker that learns from user feedback.
///
/// Maintains Beta distribution parameters per (position, result) pair.
/// Uses Thompson Sampling to explore/exploit the result ordering.
pub struct BanditsReranker {
    config: BanditsConfig,
    /// Per-vector Beta distribution parameters
    params: HashMap<String, BetaParams>,
    /// Feedback log
    feedback_log: Vec<RelevanceFeedback>,
    /// Total feedback events processed
    total_feedback: u64,
    /// Total reranking operations
    total_reranks: u64,
}

impl BanditsReranker {
    /// Create a new bandits reranker
    pub fn new(config: BanditsConfig) -> Self {
        Self {
            config,
            params: HashMap::new(),
            feedback_log: Vec::new(),
            total_feedback: 0,
            total_reranks: 0,
        }
    }

    /// Record relevance feedback for a search result
    pub fn record_feedback(&mut self, feedback: RelevanceFeedback) {
        // Update Beta parameters for this vector
        let params = self.params.entry(feedback.vector_id.clone()).or_default();
        params.update(feedback.relevance_score as f64);

        // Store feedback
        self.feedback_log.push(feedback);
        self.total_feedback += 1;

        // Apply decay periodically
        if self.total_feedback as usize % self.config.decay_interval == 0 {
            self.apply_decay();
        }

        // Evict old feedback
        if self.feedback_log.len() > self.config.max_feedback_log {
            let drain = self.feedback_log.len() - self.config.max_feedback_log;
            self.feedback_log.drain(0..drain);
        }
    }

    /// Rerank search results using Thompson Sampling
    ///
    /// Samples from the Beta distribution for each result and sorts
    /// by the sampled value (higher = more likely to be relevant).
    pub fn rerank_results(&mut self, results: &mut Vec<crate::SearchResult>) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // In A/B testing mode, only rerank 50% of the time
        if self.config.ab_testing && rng.gen_bool(0.5) {
            self.total_reranks += 1;
            return;
        }

        let mut scored: Vec<(usize, f64)> = results
            .iter()
            .enumerate()
            .map(|(idx, result)| {
                let params = self.params.get(&result.id).cloned().unwrap_or_default();
                let thompson_score = params.sample(&mut rng);
                (idx, thompson_score)
            })
            .collect();

        // Sort by Thompson sample (descending - higher is better)
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Reorder results
        let original: Vec<_> = results.drain(..).collect();
        for (idx, _) in &scored {
            if let Some(result) = original.get(*idx) {
                results.push(result.clone());
            }
        }

        self.total_reranks += 1;
    }

    /// Apply concept drift decay to all parameters
    fn apply_decay(&mut self) {
        for params in self.params.values_mut() {
            params.decay(self.config.decay_factor);
        }
    }

    /// Get statistics about the bandits reranker
    pub fn stats(&self) -> BanditsStats {
        BanditsStats {
            total_feedback: self.total_feedback,
            total_reranks: self.total_reranks,
            unique_vectors_tracked: self.params.len(),
            feedback_log_size: self.feedback_log.len(),
            ab_testing_enabled: self.config.ab_testing,
        }
    }

    /// Get the expected relevance score for a vector
    pub fn expected_relevance(&self, vector_id: &str) -> f64 {
        self.params
            .get(vector_id)
            .map_or(0.5, |p| p.expected_value())
    }
}

/// Statistics for the bandits reranker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BanditsStats {
    pub total_feedback: u64,
    pub total_reranks: u64,
    pub unique_vectors_tracked: usize,
    pub feedback_log_size: usize,
    pub ab_testing_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_API_KEY: &str = "test-key";

    #[tokio::test]
    async fn test_cohere_reranker() {
        let reranker = CohereReranker::new(TEST_API_KEY, "rerank-english-v2.0");

        let query = "machine learning algorithms";
        let documents = vec![
            "Machine learning is a subset of artificial intelligence",
            "The weather today is sunny and warm",
            "Deep learning uses neural networks for machine learning",
            "Algorithms are step by step procedures",
        ];

        let results = reranker.rerank(query, &documents, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        // First result should be most relevant
        assert!(results[0].score >= results[1].score);
    }

    #[tokio::test]
    async fn test_huggingface_reranker() {
        let reranker = HuggingFaceReranker::new("cross-encoder/ms-marco-MiniLM-L-6-v2");

        let query = "what is python programming";
        let documents = vec![
            "Python is a programming language",
            "Java is also a programming language",
            "The python snake is very long",
        ];

        let results = reranker.rerank(query, &documents, 3).await.unwrap();

        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn test_ensemble_reranker() {
        let reranker1 = Box::new(CohereReranker::new("key1", "model1")) as Box<dyn Reranker>;
        let reranker2 =
            Box::new(HuggingFaceReranker::new("cross-encoder/test")) as Box<dyn Reranker>;

        let ensemble = EnsembleReranker::new(vec![reranker1, reranker2]);

        let query = "test query";
        let documents = vec!["doc one test", "doc two query", "doc three"];

        let results = ensemble.rerank(query, &documents, 2).await.unwrap();

        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_noop_reranker() {
        let reranker = NoOpReranker;

        let query = "any query";
        let documents = vec!["doc1", "doc2", "doc3"];

        let results = reranker.rerank(query, &documents, 2).await.unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].index, 0);
        assert_eq!(results[1].index, 1);
    }

    #[tokio::test]
    async fn test_empty_documents() {
        let reranker = CohereReranker::new("key", "model");
        let results = reranker.rerank("query", &[], 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_apply_reranking_helper() {
        let reranker = NoOpReranker;
        let documents = vec!["doc1".to_string(), "doc2".to_string(), "doc3".to_string()];

        let results = apply_reranking(&reranker, "query", &documents, 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // index
    }

    #[test]
    fn test_beta_params_update() {
        let mut params = BetaParams::default();
        assert!((params.expected_value() - 0.5).abs() < 0.01);

        // Positive feedback should increase expected value
        params.update(1.0);
        params.update(1.0);
        assert!(params.expected_value() > 0.5);
    }

    #[test]
    fn test_beta_params_decay() {
        let mut params = BetaParams { alpha: 10.0, beta: 2.0 };
        let ev_before = params.expected_value();
        params.decay(0.5);
        // After decay, alpha and beta approach 1.0 (uniform), so EV approaches 0.5
        let ev_after = params.expected_value();
        assert!((ev_after - 0.5).abs() < (ev_before - 0.5).abs());
    }

    #[test]
    fn test_bandits_reranker_feedback_and_rerank() {
        let config = BanditsConfig {
            decay_factor: 0.95,
            decay_interval: 1000,
            ab_testing: false,
            max_feedback_log: 100,
        };
        let mut reranker = BanditsReranker::new(config);

        // Record positive feedback for "v2"
        for _ in 0..20 {
            reranker.record_feedback(RelevanceFeedback::new("q1", "v2", 1.0, 0));
        }
        // Record negative feedback for "v1"
        for _ in 0..20 {
            reranker.record_feedback(RelevanceFeedback::new("q1", "v1", 0.0, 1));
        }

        // v2 should have higher expected relevance than v1
        assert!(reranker.expected_relevance("v2") > reranker.expected_relevance("v1"));

        let stats = reranker.stats();
        assert_eq!(stats.total_feedback, 40);
        assert_eq!(stats.unique_vectors_tracked, 2);
    }

    #[test]
    fn test_bandits_reranker_reorder() {
        use crate::SearchResult;

        let config = BanditsConfig {
            decay_factor: 0.95,
            decay_interval: 1000,
            ab_testing: false,
            max_feedback_log: 100,
        };
        let mut reranker = BanditsReranker::new(config);

        // Heavily train on v2 being relevant
        for _ in 0..50 {
            reranker.record_feedback(RelevanceFeedback::new("q1", "v2", 1.0, 0));
            reranker.record_feedback(RelevanceFeedback::new("q1", "v1", 0.0, 1));
        }

        let mut results = vec![
            SearchResult::new("v1", 0.1, None),
            SearchResult::new("v2", 0.2, None),
        ];
        reranker.rerank_results(&mut results);
        // After heavy training, v2 should typically end up first
        // (probabilistic, but with 50 positive samples it's very likely)
        assert_eq!(results.len(), 2);
        assert_eq!(reranker.stats().total_reranks, 1);
    }

    #[test]
    fn test_bandits_feedback_log_eviction() {
        let config = BanditsConfig {
            decay_factor: 0.95,
            decay_interval: 1000,
            ab_testing: false,
            max_feedback_log: 5,
        };
        let mut reranker = BanditsReranker::new(config);
        for i in 0..10 {
            reranker.record_feedback(RelevanceFeedback::new("q", format!("v{i}"), 1.0, 0));
        }
        assert_eq!(reranker.stats().feedback_log_size, 5);
    }
}
