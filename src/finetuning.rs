//! Embedded Model Fine-Tuning Module
//!
//! This module provides in-database contrastive learning capabilities for
//! automatically improving embeddings based on user interactions and feedback.
//!
//! # Features
//!
//! - **Contrastive Learning**: Triplet loss, InfoNCE, and NT-Xent loss functions
//! - **Interaction Tracking**: Automatic collection of positive/negative pairs from user behavior
//! - **Online Learning**: Incremental model updates without full retraining
//! - **Embedding Adaptation**: Fine-tune embeddings for domain-specific use cases
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::finetuning::{FineTuner, FineTuneConfig, InteractionType};
//!
//! // Create a fine-tuner
//! let config = FineTuneConfig::default();
//! let mut tuner = FineTuner::new(384, config);
//!
//! // Record user interactions
//! tuner.record_interaction("query1", "doc1", InteractionType::Click);
//! tuner.record_interaction("query1", "doc2", InteractionType::Skip);
//!
//! // Train on collected interactions
//! let result = tuner.train(100)?;
//!
//! // Apply learned transformation to new embeddings
//! let adapted = tuner.transform(&original_embedding);
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

/// Configuration for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTuneConfig {
    /// Learning rate for gradient descent
    pub learning_rate: f32,
    /// Momentum for SGD optimizer
    pub momentum: f32,
    /// L2 regularization weight
    pub weight_decay: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Margin for triplet loss
    pub triplet_margin: f32,
    /// Temperature for InfoNCE/NT-Xent loss
    pub temperature: f32,
    /// Maximum number of interactions to store
    pub max_interactions: usize,
    /// Minimum interactions before training
    pub min_interactions_for_training: usize,
    /// Loss function to use
    pub loss_function: LossFunction,
    /// Whether to use hard negative mining
    pub hard_negative_mining: bool,
    /// Number of hard negatives to mine per positive
    pub num_hard_negatives: usize,
    /// Interaction decay half-life in seconds
    pub interaction_decay_seconds: u64,
    /// Whether to enable automatic background training
    pub auto_train: bool,
    /// Training interval for auto-train (in interactions)
    pub auto_train_interval: usize,
}

impl Default for FineTuneConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0001,
            batch_size: 32,
            triplet_margin: 0.2,
            temperature: 0.07,
            max_interactions: 100_000,
            min_interactions_for_training: 100,
            loss_function: LossFunction::TripletLoss,
            hard_negative_mining: true,
            num_hard_negatives: 5,
            interaction_decay_seconds: 86400 * 7, // 1 week
            auto_train: false,
            auto_train_interval: 1000,
        }
    }
}

/// Loss function types for contrastive learning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Triplet loss with margin
    TripletLoss,
    /// InfoNCE (Noise Contrastive Estimation)
    InfoNCE,
    /// NT-Xent (Normalized Temperature-scaled Cross Entropy)
    NTXent,
    /// Multiple Negatives Ranking Loss
    MultipleNegativesRanking,
    /// Cosine Embedding Loss
    CosineEmbedding,
}

/// Types of user interactions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractionType {
    /// User clicked on a result
    Click,
    /// User viewed result details
    View,
    /// User bookmarked/saved result
    Bookmark,
    /// User purchased/converted
    Purchase,
    /// User skipped/ignored result
    Skip,
    /// User explicitly marked as irrelevant
    Irrelevant,
    /// User explicitly marked as relevant
    Relevant,
    /// User spent significant time on result
    Dwell,
    /// User shared the result
    Share,
}

impl InteractionType {
    /// Get the relevance signal strength (-1.0 to 1.0)
    pub fn signal_strength(&self) -> f32 {
        match self {
            InteractionType::Purchase => 1.0,
            InteractionType::Share => 0.95,
            InteractionType::Bookmark => 0.9,
            InteractionType::Relevant => 0.85,
            InteractionType::Dwell => 0.7,
            InteractionType::Click => 0.5,
            InteractionType::View => 0.3,
            InteractionType::Skip => -0.3,
            InteractionType::Irrelevant => -1.0,
        }
    }

    /// Check if this is a positive signal
    pub fn is_positive(&self) -> bool {
        self.signal_strength() > 0.0
    }
}

/// A recorded user interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interaction {
    /// Query ID or embedding
    pub query_id: String,
    /// Query embedding (optional, for efficiency)
    pub query_embedding: Option<Vec<f32>>,
    /// Document/result ID
    pub document_id: String,
    /// Document embedding (optional)
    pub document_embedding: Option<Vec<f32>>,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Timestamp of interaction
    pub timestamp: u64,
    /// Position in results list (0-indexed)
    pub position: Option<usize>,
    /// Session ID for grouping interactions
    pub session_id: Option<String>,
    /// Additional context metadata
    pub metadata: Option<serde_json::Value>,
}

/// A training triplet (anchor, positive, negative)
#[derive(Debug, Clone)]
pub struct Triplet {
    pub anchor: Vec<f32>,
    pub positive: Vec<f32>,
    pub negative: Vec<f32>,
    pub weight: f32,
}

/// A contrastive pair (query, document, label)
#[derive(Debug, Clone)]
pub struct ContrastivePair {
    pub query: Vec<f32>,
    pub document: Vec<f32>,
    pub label: f32, // 1.0 for positive, -1.0 for negative
    pub weight: f32,
}

/// Training batch for contrastive learning
#[derive(Debug, Clone)]
pub struct TrainingBatch {
    pub triplets: Vec<Triplet>,
    pub pairs: Vec<ContrastivePair>,
}

/// Linear transformation layer for embedding adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearTransform {
    /// Weight matrix (output_dim x input_dim)
    pub weights: Vec<Vec<f32>>,
    /// Bias vector
    pub bias: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl LinearTransform {
    /// Create a new identity-initialized linear transform
    pub fn new(dim: usize) -> Self {
        let mut weights = vec![vec![0.0; dim]; dim];
        for (i, row) in weights.iter_mut().enumerate() {
            row[i] = 1.0; // Identity initialization
        }
        Self {
            weights,
            bias: vec![0.0; dim],
            input_dim: dim,
            output_dim: dim,
        }
    }

    /// Create with random initialization
    pub fn new_random(input_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng_state = seed;
        let scale = (2.0 / (input_dim + output_dim) as f32).sqrt(); // Xavier initialization

        let weights: Vec<Vec<f32>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| {
                        rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                        let u = ((rng_state >> 16) as f32 / 32768.0) - 1.0;
                        u * scale
                    })
                    .collect()
            })
            .collect();

        Self {
            weights,
            bias: vec![0.0; output_dim],
            input_dim,
            output_dim,
        }
    }

    /// Apply the transformation
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        if input.len() != self.input_dim {
            return input.to_vec(); // Return unchanged if dimensions don't match
        }

        let mut output = vec![0.0; self.output_dim];
        for (i, row) in self.weights.iter().enumerate() {
            let mut sum = self.bias[i];
            for (j, &w) in row.iter().enumerate() {
                sum += w * input[j];
            }
            output[i] = sum;
        }
        output
    }

    /// Compute gradient and update weights
    pub fn backward(
        &mut self,
        input: &[f32],
        output_grad: &[f32],
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        velocity: &mut LinearTransformVelocity,
    ) -> Vec<f32> {
        // Compute input gradient
        let mut input_grad = vec![0.0; self.input_dim];
        for (j, ig) in input_grad.iter_mut().enumerate() {
            for (i, &og) in output_grad.iter().enumerate() {
                *ig += self.weights[i][j] * og;
            }
        }

        // Update weights with momentum and weight decay
        for (i, row) in self.weights.iter_mut().enumerate() {
            for (j, w) in row.iter_mut().enumerate() {
                let grad = output_grad[i] * input[j] + weight_decay * *w;
                velocity.weights[i][j] = momentum * velocity.weights[i][j] - learning_rate * grad;
                *w += velocity.weights[i][j];
            }
        }

        // Update bias
        for (i, b) in self.bias.iter_mut().enumerate() {
            let grad = output_grad[i];
            velocity.bias[i] = momentum * velocity.bias[i] - learning_rate * grad;
            *b += velocity.bias[i];
        }

        input_grad
    }
}

/// Velocity state for momentum-based optimization
#[derive(Debug, Clone)]
pub struct LinearTransformVelocity {
    pub weights: Vec<Vec<f32>>,
    pub bias: Vec<f32>,
}

impl LinearTransformVelocity {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![vec![0.0; input_dim]; output_dim],
            bias: vec![0.0; output_dim],
        }
    }
}

/// Result of a training step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingResult {
    /// Number of training steps completed
    pub steps: usize,
    /// Final loss value
    pub final_loss: f32,
    /// Loss history
    pub loss_history: Vec<f32>,
    /// Number of triplets/pairs processed
    pub samples_processed: usize,
    /// Training duration in milliseconds
    pub duration_ms: u64,
    /// Average loss
    pub average_loss: f32,
    /// Improvement from initial loss
    pub improvement: f32,
}

/// Embedding storage for fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingStore {
    /// Query embeddings by ID
    pub queries: HashMap<String, Vec<f32>>,
    /// Document embeddings by ID
    pub documents: HashMap<String, Vec<f32>>,
}

impl Default for EmbeddingStore {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingStore {
    pub fn new() -> Self {
        Self {
            queries: HashMap::new(),
            documents: HashMap::new(),
        }
    }

    pub fn add_query(&mut self, id: &str, embedding: Vec<f32>) {
        self.queries.insert(id.to_string(), embedding);
    }

    pub fn add_document(&mut self, id: &str, embedding: Vec<f32>) {
        self.documents.insert(id.to_string(), embedding);
    }

    pub fn get_query(&self, id: &str) -> Option<&Vec<f32>> {
        self.queries.get(id)
    }

    pub fn get_document(&self, id: &str) -> Option<&Vec<f32>> {
        self.documents.get(id)
    }
}

/// Main fine-tuning engine
pub struct FineTuner {
    /// Embedding dimension
    dim: usize,
    /// Configuration
    config: FineTuneConfig,
    /// Linear transformation layer
    transform: LinearTransform,
    /// Velocity for momentum
    velocity: LinearTransformVelocity,
    /// Recorded interactions
    interactions: VecDeque<Interaction>,
    /// Embedding store
    embedding_store: EmbeddingStore,
    /// Training statistics
    stats: FineTunerStats,
    /// Interactions since last auto-train
    interactions_since_train: usize,
}

/// Statistics for the fine-tuner
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FineTunerStats {
    /// Total interactions recorded
    pub total_interactions: usize,
    /// Total training steps
    pub total_training_steps: usize,
    /// Total training time in milliseconds
    pub total_training_time_ms: u64,
    /// Number of training sessions
    pub training_sessions: usize,
    /// Best loss achieved
    pub best_loss: Option<f32>,
    /// Last training loss
    pub last_loss: Option<f32>,
    /// Positive interactions count
    pub positive_interactions: usize,
    /// Negative interactions count
    pub negative_interactions: usize,
}

impl FineTuner {
    /// Create a new fine-tuner
    pub fn new(dim: usize, config: FineTuneConfig) -> Self {
        Self {
            dim,
            config: config.clone(),
            transform: LinearTransform::new(dim),
            velocity: LinearTransformVelocity::new(dim, dim),
            interactions: VecDeque::new(),
            embedding_store: EmbeddingStore::new(),
            stats: FineTunerStats::default(),
            interactions_since_train: 0,
        }
    }

    /// Get the embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Get the configuration
    pub fn config(&self) -> &FineTuneConfig {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &FineTunerStats {
        &self.stats
    }

    /// Record a user interaction
    pub fn record_interaction(
        &mut self,
        query_id: &str,
        document_id: &str,
        interaction_type: InteractionType,
    ) {
        self.record_interaction_with_details(
            query_id,
            None,
            document_id,
            None,
            interaction_type,
            None,
            None,
            None,
        );
    }

    /// Record an interaction with full details
    #[allow(clippy::too_many_arguments)]
    pub fn record_interaction_with_details(
        &mut self,
        query_id: &str,
        query_embedding: Option<Vec<f32>>,
        document_id: &str,
        document_embedding: Option<Vec<f32>>,
        interaction_type: InteractionType,
        position: Option<usize>,
        session_id: Option<String>,
        metadata: Option<serde_json::Value>,
    ) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Store embeddings if provided
        if let Some(ref emb) = query_embedding {
            self.embedding_store.add_query(query_id, emb.clone());
        }
        if let Some(ref emb) = document_embedding {
            self.embedding_store.add_document(document_id, emb.clone());
        }

        let interaction = Interaction {
            query_id: query_id.to_string(),
            query_embedding,
            document_id: document_id.to_string(),
            document_embedding,
            interaction_type,
            timestamp,
            position,
            session_id,
            metadata,
        };

        // Update stats
        self.stats.total_interactions += 1;
        if interaction_type.is_positive() {
            self.stats.positive_interactions += 1;
        } else {
            self.stats.negative_interactions += 1;
        }

        // Add to queue, removing old if necessary
        if self.interactions.len() >= self.config.max_interactions {
            self.interactions.pop_front();
        }
        self.interactions.push_back(interaction);
        self.interactions_since_train += 1;

        // Auto-train if configured
        if self.config.auto_train
            && self.interactions_since_train >= self.config.auto_train_interval
            && self.interactions.len() >= self.config.min_interactions_for_training
        {
            let _ = self.train(10); // Quick training session
        }
    }

    /// Add a query embedding to the store
    pub fn add_query_embedding(&mut self, id: &str, embedding: Vec<f32>) {
        self.embedding_store.add_query(id, embedding);
    }

    /// Add a document embedding to the store
    pub fn add_document_embedding(&mut self, id: &str, embedding: Vec<f32>) {
        self.embedding_store.add_document(id, embedding);
    }

    /// Build training triplets from interactions
    pub fn build_triplets(&self) -> Vec<Triplet> {
        let mut triplets = Vec::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Group interactions by query
        let mut by_query: HashMap<&str, Vec<&Interaction>> = HashMap::new();
        for interaction in &self.interactions {
            by_query
                .entry(&interaction.query_id)
                .or_default()
                .push(interaction);
        }

        for (query_id, interactions) in by_query {
            // Get query embedding
            let query_emb = if let Some(emb) = self.embedding_store.get_query(query_id) {
                emb.clone()
            } else if let Some(int) = interactions.first() {
                if let Some(ref emb) = int.query_embedding {
                    emb.clone()
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // Separate positives and negatives
            let positives: Vec<_> = interactions
                .iter()
                .filter(|i| i.interaction_type.is_positive())
                .collect();
            let negatives: Vec<_> = interactions
                .iter()
                .filter(|i| !i.interaction_type.is_positive())
                .collect();

            // Create triplets
            for positive in &positives {
                let pos_emb = if let Some(ref emb) = positive.document_embedding {
                    emb.clone()
                } else if let Some(emb) = self.embedding_store.get_document(&positive.document_id) {
                    emb.clone()
                } else {
                    continue;
                };

                // Calculate weight based on interaction strength and recency
                let age_seconds = now.saturating_sub(positive.timestamp);
                let decay = (-((age_seconds as f64) / (self.config.interaction_decay_seconds as f64)))
                    .exp() as f32;
                let base_weight = positive.interaction_type.signal_strength().abs();

                for negative in &negatives {
                    let neg_emb = if let Some(ref emb) = negative.document_embedding {
                        emb.clone()
                    } else if let Some(emb) = self.embedding_store.get_document(&negative.document_id)
                    {
                        emb.clone()
                    } else {
                        continue;
                    };

                    let neg_weight = negative.interaction_type.signal_strength().abs();
                    let weight = base_weight * neg_weight * decay;

                    triplets.push(Triplet {
                        anchor: query_emb.clone(),
                        positive: pos_emb.clone(),
                        negative: neg_emb.clone(),
                        weight,
                    });
                }
            }
        }

        // Hard negative mining if enabled
        if self.config.hard_negative_mining && !triplets.is_empty() {
            triplets = self.mine_hard_negatives(triplets);
        }

        triplets
    }

    /// Mine hard negatives (closest negatives to anchor)
    fn mine_hard_negatives(&self, mut triplets: Vec<Triplet>) -> Vec<Triplet> {
        // Sort by negative distance (ascending = harder negatives first)
        triplets.sort_by(|a, b| {
            let dist_a = cosine_distance(&a.anchor, &a.negative);
            let dist_b = cosine_distance(&b.anchor, &b.negative);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Keep top hard negatives per anchor
        let mut result = Vec::new();
        let mut anchor_counts: HashMap<String, usize> = HashMap::new();

        for triplet in triplets {
            let anchor_key = format!("{:?}", &triplet.anchor[..5.min(triplet.anchor.len())]);
            let count = anchor_counts.entry(anchor_key.clone()).or_insert(0);
            if *count < self.config.num_hard_negatives {
                result.push(triplet);
                *count += 1;
            }
        }

        result
    }

    /// Build contrastive pairs from interactions
    pub fn build_pairs(&self) -> Vec<ContrastivePair> {
        let mut pairs = Vec::new();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        for interaction in &self.interactions {
            let query_emb = if let Some(ref emb) = interaction.query_embedding {
                emb.clone()
            } else if let Some(emb) = self.embedding_store.get_query(&interaction.query_id) {
                emb.clone()
            } else {
                continue;
            };

            let doc_emb = if let Some(ref emb) = interaction.document_embedding {
                emb.clone()
            } else if let Some(emb) = self.embedding_store.get_document(&interaction.document_id) {
                emb.clone()
            } else {
                continue;
            };

            let age_seconds = now.saturating_sub(interaction.timestamp);
            let decay =
                (-((age_seconds as f64) / (self.config.interaction_decay_seconds as f64))).exp()
                    as f32;
            let signal = interaction.interaction_type.signal_strength();

            pairs.push(ContrastivePair {
                query: query_emb,
                document: doc_emb,
                label: if signal > 0.0 { 1.0 } else { -1.0 },
                weight: signal.abs() * decay,
            });
        }

        pairs
    }

    /// Train the transformation on collected interactions
    pub fn train(&mut self, epochs: usize) -> Result<TrainingResult> {
        if self.interactions.len() < self.config.min_interactions_for_training {
            return Err(NeedleError::InvalidInput(format!(
                "Not enough interactions for training: {} < {}",
                self.interactions.len(),
                self.config.min_interactions_for_training
            )));
        }

        let start = Instant::now();
        let mut loss_history = Vec::new();
        let mut samples_processed = 0;

        let triplets = self.build_triplets();
        if triplets.is_empty() {
            return Err(NeedleError::InvalidInput(
                "No valid triplets could be generated from interactions".to_string(),
            ));
        }

        let initial_loss = self.compute_loss(&triplets);
        loss_history.push(initial_loss);

        for _epoch in 0..epochs {
            // Shuffle triplets (simple deterministic shuffle)
            let mut batch_indices: Vec<usize> = (0..triplets.len()).collect();
            for i in (1..batch_indices.len()).rev() {
                let j = i % (i + 1);
                batch_indices.swap(i, j);
            }

            let mut epoch_loss = 0.0;
            let mut batch_count = 0;

            for batch_start in (0..triplets.len()).step_by(self.config.batch_size) {
                let batch_end = (batch_start + self.config.batch_size).min(triplets.len());
                let batch: Vec<_> = batch_indices[batch_start..batch_end]
                    .iter()
                    .map(|&i| &triplets[i])
                    .collect();

                let loss = self.train_batch(&batch)?;
                epoch_loss += loss;
                batch_count += 1;
                samples_processed += batch.len();
            }

            let avg_epoch_loss = if batch_count > 0 {
                epoch_loss / batch_count as f32
            } else {
                0.0
            };
            loss_history.push(avg_epoch_loss);
        }

        let duration = start.elapsed();
        let final_loss = *loss_history.last().unwrap_or(&0.0);

        // Update stats
        self.stats.total_training_steps += epochs;
        self.stats.total_training_time_ms += duration.as_millis() as u64;
        self.stats.training_sessions += 1;
        self.stats.last_loss = Some(final_loss);
        if self.stats.best_loss.is_none() || final_loss < self.stats.best_loss.unwrap() {
            self.stats.best_loss = Some(final_loss);
        }
        self.interactions_since_train = 0;

        let improvement = initial_loss - final_loss;
        let average_loss = loss_history.iter().sum::<f32>() / loss_history.len() as f32;

        Ok(TrainingResult {
            steps: epochs,
            final_loss,
            loss_history,
            samples_processed,
            duration_ms: duration.as_millis() as u64,
            average_loss,
            improvement,
        })
    }

    /// Train on a single batch
    fn train_batch(&mut self, batch: &[&Triplet]) -> Result<f32> {
        let mut total_loss = 0.0;

        for triplet in batch {
            // Forward pass through transform
            let anchor_t = self.transform.forward(&triplet.anchor);
            let positive_t = self.transform.forward(&triplet.positive);
            let negative_t = self.transform.forward(&triplet.negative);

            // Compute loss and gradients based on loss function
            let (loss, anchor_grad, positive_grad, negative_grad) = match self.config.loss_function
            {
                LossFunction::TripletLoss => {
                    self.triplet_loss_backward(&anchor_t, &positive_t, &negative_t, triplet.weight)
                }
                LossFunction::InfoNCE => {
                    self.infonce_loss_backward(&anchor_t, &positive_t, &negative_t, triplet.weight)
                }
                LossFunction::NTXent => {
                    self.ntxent_loss_backward(&anchor_t, &positive_t, &negative_t, triplet.weight)
                }
                LossFunction::MultipleNegativesRanking => self.mnr_loss_backward(
                    &anchor_t,
                    &positive_t,
                    &negative_t,
                    triplet.weight,
                ),
                LossFunction::CosineEmbedding => self.cosine_loss_backward(
                    &anchor_t,
                    &positive_t,
                    &negative_t,
                    triplet.weight,
                ),
            };

            total_loss += loss;

            // Backward pass - update transform weights
            self.transform.backward(
                &triplet.anchor,
                &anchor_grad,
                self.config.learning_rate,
                self.config.momentum,
                self.config.weight_decay,
                &mut self.velocity,
            );

            self.transform.backward(
                &triplet.positive,
                &positive_grad,
                self.config.learning_rate,
                self.config.momentum,
                self.config.weight_decay,
                &mut self.velocity,
            );

            self.transform.backward(
                &triplet.negative,
                &negative_grad,
                self.config.learning_rate,
                self.config.momentum,
                self.config.weight_decay,
                &mut self.velocity,
            );
        }

        Ok(total_loss / batch.len() as f32)
    }

    /// Compute triplet loss with margin
    fn triplet_loss_backward(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        weight: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pos_dist = cosine_distance(anchor, positive);
        let neg_dist = cosine_distance(anchor, negative);

        let loss = (pos_dist - neg_dist + self.config.triplet_margin).max(0.0) * weight;

        if loss > 0.0 {
            // Compute gradients
            let anchor_grad = gradient_cosine_distance(anchor, positive, negative);
            let positive_grad = gradient_cosine_distance_positive(anchor, positive);
            let negative_grad = gradient_cosine_distance_negative(anchor, negative);

            (
                loss,
                scale_vec(&anchor_grad, weight),
                scale_vec(&positive_grad, weight),
                scale_vec(&negative_grad, -weight),
            )
        } else {
            (
                0.0,
                vec![0.0; anchor.len()],
                vec![0.0; positive.len()],
                vec![0.0; negative.len()],
            )
        }
    }

    /// InfoNCE loss
    fn infonce_loss_backward(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        weight: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pos_sim = cosine_similarity(anchor, positive) / self.config.temperature;
        let neg_sim = cosine_similarity(anchor, negative) / self.config.temperature;

        let max_sim = pos_sim.max(neg_sim);
        let exp_pos = (pos_sim - max_sim).exp();
        let exp_neg = (neg_sim - max_sim).exp();
        let sum_exp = exp_pos + exp_neg;

        let loss = -(pos_sim - max_sim - sum_exp.ln()) * weight;

        // Simplified gradients
        let softmax_pos = exp_pos / sum_exp;
        let softmax_neg = exp_neg / sum_exp;

        let anchor_grad = scale_vec(
            &sub_vecs(
                &scale_vec(&gradient_cosine_similarity(anchor, positive), softmax_pos - 1.0),
                &scale_vec(&gradient_cosine_similarity(anchor, negative), softmax_neg),
            ),
            weight / self.config.temperature,
        );

        let positive_grad = scale_vec(
            &gradient_cosine_similarity_other(anchor, positive),
            weight * (softmax_pos - 1.0) / self.config.temperature,
        );

        let negative_grad = scale_vec(
            &gradient_cosine_similarity_other(anchor, negative),
            weight * softmax_neg / self.config.temperature,
        );

        (loss, anchor_grad, positive_grad, negative_grad)
    }

    /// NT-Xent loss
    fn ntxent_loss_backward(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        weight: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        // NT-Xent is similar to InfoNCE but with specific normalization
        self.infonce_loss_backward(anchor, positive, negative, weight)
    }

    /// Multiple Negatives Ranking loss
    fn mnr_loss_backward(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        weight: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pos_sim = cosine_similarity(anchor, positive);
        let neg_sim = cosine_similarity(anchor, negative);

        // Softmax cross-entropy style loss
        let max_sim = pos_sim.max(neg_sim);
        let exp_pos = (pos_sim - max_sim).exp();
        let exp_neg = (neg_sim - max_sim).exp();
        let loss = -((exp_pos / (exp_pos + exp_neg)).ln()) * weight;

        // Gradients similar to InfoNCE
        let softmax_neg = exp_neg / (exp_pos + exp_neg);

        let anchor_grad = scale_vec(
            &sub_vecs(
                &gradient_cosine_similarity(anchor, negative),
                &gradient_cosine_similarity(anchor, positive),
            ),
            weight * softmax_neg,
        );

        let positive_grad = scale_vec(
            &gradient_cosine_similarity_other(anchor, positive),
            -weight * softmax_neg,
        );

        let negative_grad = scale_vec(
            &gradient_cosine_similarity_other(anchor, negative),
            weight * softmax_neg,
        );

        (loss, anchor_grad, positive_grad, negative_grad)
    }

    /// Cosine embedding loss
    fn cosine_loss_backward(
        &self,
        anchor: &[f32],
        positive: &[f32],
        negative: &[f32],
        weight: f32,
    ) -> (f32, Vec<f32>, Vec<f32>, Vec<f32>) {
        let pos_sim = cosine_similarity(anchor, positive);
        let neg_sim = cosine_similarity(anchor, negative);

        // Loss: (1 - pos_sim) + max(0, neg_sim - margin)
        let pos_loss = (1.0 - pos_sim) * weight;
        let neg_loss = (neg_sim - self.config.triplet_margin).max(0.0) * weight;
        let loss = pos_loss + neg_loss;

        let anchor_grad = if neg_sim > self.config.triplet_margin {
            sub_vecs(
                &scale_vec(&gradient_cosine_similarity(anchor, negative), weight),
                &scale_vec(&gradient_cosine_similarity(anchor, positive), weight),
            )
        } else {
            scale_vec(&gradient_cosine_similarity(anchor, positive), -weight)
        };

        let positive_grad =
            scale_vec(&gradient_cosine_similarity_other(anchor, positive), -weight);

        let negative_grad = if neg_sim > self.config.triplet_margin {
            scale_vec(&gradient_cosine_similarity_other(anchor, negative), weight)
        } else {
            vec![0.0; negative.len()]
        };

        (loss, anchor_grad, positive_grad, negative_grad)
    }

    /// Compute total loss over triplets
    fn compute_loss(&self, triplets: &[Triplet]) -> f32 {
        let mut total_loss = 0.0;

        for triplet in triplets {
            let anchor_t = self.transform.forward(&triplet.anchor);
            let positive_t = self.transform.forward(&triplet.positive);
            let negative_t = self.transform.forward(&triplet.negative);

            let loss = match self.config.loss_function {
                LossFunction::TripletLoss => {
                    let pos_dist = cosine_distance(&anchor_t, &positive_t);
                    let neg_dist = cosine_distance(&anchor_t, &negative_t);
                    (pos_dist - neg_dist + self.config.triplet_margin).max(0.0) * triplet.weight
                }
                LossFunction::InfoNCE | LossFunction::NTXent => {
                    let pos_sim = cosine_similarity(&anchor_t, &positive_t) / self.config.temperature;
                    let neg_sim = cosine_similarity(&anchor_t, &negative_t) / self.config.temperature;
                    let max_sim = pos_sim.max(neg_sim);
                    let exp_pos = (pos_sim - max_sim).exp();
                    let exp_neg = (neg_sim - max_sim).exp();
                    -(pos_sim - max_sim - (exp_pos + exp_neg).ln()) * triplet.weight
                }
                LossFunction::MultipleNegativesRanking => {
                    let pos_sim = cosine_similarity(&anchor_t, &positive_t);
                    let neg_sim = cosine_similarity(&anchor_t, &negative_t);
                    let max_sim = pos_sim.max(neg_sim);
                    let exp_pos = (pos_sim - max_sim).exp();
                    let exp_neg = (neg_sim - max_sim).exp();
                    -((exp_pos / (exp_pos + exp_neg)).ln()) * triplet.weight
                }
                LossFunction::CosineEmbedding => {
                    let pos_sim = cosine_similarity(&anchor_t, &positive_t);
                    let neg_sim = cosine_similarity(&anchor_t, &negative_t);
                    (1.0 - pos_sim + (neg_sim - self.config.triplet_margin).max(0.0)) * triplet.weight
                }
            };

            total_loss += loss;
        }

        total_loss / triplets.len() as f32
    }

    /// Apply the learned transformation to an embedding
    pub fn transform(&self, embedding: &[f32]) -> Vec<f32> {
        let transformed = self.transform.forward(embedding);
        normalize(&transformed)
    }

    /// Apply transformation to multiple embeddings
    pub fn transform_batch(&self, embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
        embeddings.iter().map(|e| self.transform(e)).collect()
    }

    /// Reset the transformation to identity
    pub fn reset(&mut self) {
        self.transform = LinearTransform::new(self.dim);
        self.velocity = LinearTransformVelocity::new(self.dim, self.dim);
    }

    /// Clear all recorded interactions
    pub fn clear_interactions(&mut self) {
        self.interactions.clear();
        self.interactions_since_train = 0;
    }

    /// Get the number of recorded interactions
    pub fn num_interactions(&self) -> usize {
        self.interactions.len()
    }

    /// Export the model state
    pub fn export(&self) -> FineTunerState {
        FineTunerState {
            dim: self.dim,
            config: self.config.clone(),
            transform: self.transform.clone(),
            stats: self.stats.clone(),
        }
    }

    /// Import a model state
    pub fn import(&mut self, state: FineTunerState) -> Result<()> {
        if state.dim != self.dim {
            return Err(NeedleError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dim, state.dim
            )));
        }
        self.config = state.config;
        self.transform = state.transform;
        self.stats = state.stats;
        self.velocity = LinearTransformVelocity::new(self.dim, self.dim);
        Ok(())
    }
}

/// Serializable state for FineTuner
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FineTunerState {
    pub dim: usize,
    pub config: FineTuneConfig,
    pub transform: LinearTransform,
    pub stats: FineTunerStats,
}

/// Thread-safe wrapper for FineTuner
pub struct SharedFineTuner {
    inner: Arc<RwLock<FineTuner>>,
}

impl SharedFineTuner {
    pub fn new(dim: usize, config: FineTuneConfig) -> Self {
        Self {
            inner: Arc::new(RwLock::new(FineTuner::new(dim, config))),
        }
    }

    pub fn record_interaction(
        &self,
        query_id: &str,
        document_id: &str,
        interaction_type: InteractionType,
    ) {
        if let Ok(mut tuner) = self.inner.write() {
            tuner.record_interaction(query_id, document_id, interaction_type);
        }
    }

    pub fn transform(&self, embedding: &[f32]) -> Vec<f32> {
        if let Ok(tuner) = self.inner.read() {
            tuner.transform(embedding)
        } else {
            embedding.to_vec()
        }
    }

    pub fn train(&self, epochs: usize) -> Result<TrainingResult> {
        if let Ok(mut tuner) = self.inner.write() {
            tuner.train(epochs)
        } else {
            Err(NeedleError::LockError)
        }
    }

    pub fn stats(&self) -> Option<FineTunerStats> {
        self.inner.read().ok().map(|t| t.stats().clone())
    }
}

impl Clone for SharedFineTuner {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

// Helper functions for vector operations

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn scale_vec(v: &[f32], scale: f32) -> Vec<f32> {
    v.iter().map(|x| x * scale).collect()
}

fn sub_vecs(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn gradient_cosine_distance(anchor: &[f32], positive: &[f32], negative: &[f32]) -> Vec<f32> {
    let grad_pos = gradient_cosine_similarity(anchor, positive);
    let grad_neg = gradient_cosine_similarity(anchor, negative);
    sub_vecs(&scale_vec(&grad_neg, -1.0), &grad_pos)
}

fn gradient_cosine_similarity(a: &[f32], b: &[f32]) -> Vec<f32> {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        let scale = 1.0 / (norm_a * norm_b);
        let dot_scale = dot / (norm_a * norm_a * norm_a * norm_b);
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| bi * scale - ai * dot_scale)
            .collect()
    } else {
        vec![0.0; a.len()]
    }
}

fn gradient_cosine_similarity_other(anchor: &[f32], other: &[f32]) -> Vec<f32> {
    let dot: f32 = anchor.iter().zip(other.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = anchor.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_o: f32 = other.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_o > 0.0 {
        let scale = 1.0 / (norm_a * norm_o);
        let dot_scale = dot / (norm_a * norm_o * norm_o * norm_o);
        anchor
            .iter()
            .zip(other.iter())
            .map(|(ai, oi)| ai * scale - oi * dot_scale)
            .collect()
    } else {
        vec![0.0; other.len()]
    }
}

fn gradient_cosine_distance_positive(anchor: &[f32], positive: &[f32]) -> Vec<f32> {
    scale_vec(&gradient_cosine_similarity_other(anchor, positive), -1.0)
}

fn gradient_cosine_distance_negative(anchor: &[f32], negative: &[f32]) -> Vec<f32> {
    gradient_cosine_similarity_other(anchor, negative)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut rng_state = seed;
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng_state >> 16) as f32 / 32768.0) - 1.0
            })
            .collect();
        normalize(&v)
    }

    #[test]
    fn test_linear_transform_identity() {
        let transform = LinearTransform::new(128);
        let input = random_vec(128, 42);
        let output = transform.forward(&input);

        // Identity transform should preserve input
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn test_fine_tuner_record_interaction() {
        let config = FineTuneConfig::default();
        let mut tuner = FineTuner::new(128, config);

        tuner.record_interaction("q1", "d1", InteractionType::Click);
        tuner.record_interaction("q1", "d2", InteractionType::Skip);

        assert_eq!(tuner.num_interactions(), 2);
        assert_eq!(tuner.stats().positive_interactions, 1);
        assert_eq!(tuner.stats().negative_interactions, 1);
    }

    #[test]
    fn test_fine_tuner_with_embeddings() {
        let config = FineTuneConfig {
            min_interactions_for_training: 2,
            ..Default::default()
        };
        let mut tuner = FineTuner::new(64, config);

        // Add embeddings
        let query = random_vec(64, 1);
        let pos_doc = random_vec(64, 2);
        let neg_doc = random_vec(64, 3);

        tuner.add_query_embedding("q1", query.clone());
        tuner.add_document_embedding("d1", pos_doc.clone());
        tuner.add_document_embedding("d2", neg_doc.clone());

        // Record interactions
        tuner.record_interaction("q1", "d1", InteractionType::Click);
        tuner.record_interaction("q1", "d2", InteractionType::Skip);

        // Build triplets
        let triplets = tuner.build_triplets();
        assert!(!triplets.is_empty());
    }

    #[test]
    fn test_fine_tuner_training() {
        let config = FineTuneConfig {
            min_interactions_for_training: 1,
            learning_rate: 0.01,
            ..Default::default()
        };
        let mut tuner = FineTuner::new(32, config);

        // Add embeddings and interactions
        for i in 0..10 {
            let query = random_vec(32, i * 100);
            let pos = random_vec(32, i * 100 + 1);
            let neg = random_vec(32, i * 100 + 2);

            tuner.add_query_embedding(&format!("q{}", i), query);
            tuner.add_document_embedding(&format!("pos{}", i), pos);
            tuner.add_document_embedding(&format!("neg{}", i), neg);

            tuner.record_interaction(&format!("q{}", i), &format!("pos{}", i), InteractionType::Click);
            tuner.record_interaction(&format!("q{}", i), &format!("neg{}", i), InteractionType::Skip);
        }

        // Train
        let result = tuner.train(5).unwrap();
        assert!(result.steps == 5);
        assert!(result.samples_processed > 0);
    }

    #[test]
    fn test_interaction_type_signal_strength() {
        assert!(InteractionType::Purchase.signal_strength() > InteractionType::Click.signal_strength());
        assert!(InteractionType::Click.is_positive());
        assert!(!InteractionType::Skip.is_positive());
        assert!(!InteractionType::Irrelevant.is_positive());
    }

    #[test]
    fn test_fine_tuner_export_import() {
        let config = FineTuneConfig::default();
        let tuner = FineTuner::new(64, config.clone());

        let state = tuner.export();
        assert_eq!(state.dim, 64);

        let mut new_tuner = FineTuner::new(64, config);
        new_tuner.import(state).unwrap();
    }

    #[test]
    fn test_shared_fine_tuner() {
        let config = FineTuneConfig::default();
        let tuner = SharedFineTuner::new(64, config);

        tuner.record_interaction("q1", "d1", InteractionType::Click);

        let embedding = random_vec(64, 42);
        let transformed = tuner.transform(&embedding);
        assert_eq!(transformed.len(), 64);

        let stats = tuner.stats().unwrap();
        assert_eq!(stats.total_interactions, 1);
    }
}
