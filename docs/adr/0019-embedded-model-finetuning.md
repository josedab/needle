# ADR-0019: Embedded Model Fine-Tuning Framework

## Status

Accepted

## Context

Pre-trained embedding models provide good general-purpose vectors, but domain-specific applications often need better performance:

1. **Domain vocabulary** — Medical, legal, or technical terms may not be well-represented
2. **User feedback** — Click-through data indicates what users actually find relevant
3. **Semantic drift** — Language and concepts evolve over time
4. **Cold start** — New domains have no existing fine-tuned models
5. **Operational complexity** — External fine-tuning pipelines are complex to maintain

Traditional approaches require exporting data, running separate training jobs, and reimporting embeddings — a process that takes days and introduces consistency challenges.

## Decision

Needle implements **in-database contrastive learning** for embedding fine-tuning:

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FineTuner                              │
├─────────────────────────────────────────────────────────────┤
│  Interaction      │  Records clicks, views, relevance      │
│  Collector        │  feedback from user actions            │
├───────────────────┼─────────────────────────────────────────┤
│  Triplet          │  Generates (anchor, positive, negative) │
│  Sampler          │  training examples from interactions   │
├───────────────────┼─────────────────────────────────────────┤
│  Linear           │  Learnable transformation matrix       │
│  Transform        │  applied to base embeddings            │
├───────────────────┼─────────────────────────────────────────┤
│  Trainer          │  Mini-batch SGD with momentum          │
│                   │  and learning rate scheduling          │
└───────────────────┴─────────────────────────────────────────┘
```

### Fine-Tuning Configuration

```rust
pub struct FineTuneConfig {
    /// Embedding dimension
    pub dimension: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum for SGD
    pub momentum: f64,
    /// Margin for triplet loss
    pub margin: f64,
    /// Batch size for training
    pub batch_size: usize,
    /// L2 regularization
    pub weight_decay: f64,
    /// Negative sampling ratio
    pub negative_ratio: usize,
}
```

### Interaction Types

```rust
pub struct Interaction {
    /// Query that produced these results
    pub query_id: String,
    /// User ID (for personalization)
    pub user_id: Option<String>,
    /// Vector ID that was interacted with
    pub vector_id: String,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

pub enum InteractionType {
    Click,           // User clicked the result
    View,            // Result was viewed (impression)
    Purchase,        // Conversion event
    Bookmark,        // Explicit positive signal
    Skip,            // Viewed but not clicked (implicit negative)
    Explicit(f32),   // Explicit relevance rating (0-1)
}
```

### Training Data Generation

```rust
pub struct Triplet {
    /// Anchor embedding (the query)
    pub anchor: Vec<f32>,
    /// Positive example (clicked/relevant result)
    pub positive: Vec<f32>,
    /// Negative example (not clicked or low relevance)
    pub negative: Vec<f32>,
}

pub struct ContrastivePair {
    /// First embedding
    pub a: Vec<f32>,
    /// Second embedding
    pub b: Vec<f32>,
    /// Similarity label (1.0 = similar, 0.0 = dissimilar)
    pub label: f32,
}
```

### Linear Transform Model

Rather than fine-tuning the entire embedding model, Needle learns a lightweight transformation:

```rust
pub struct LinearTransform {
    /// Transformation matrix (dimension x dimension)
    weights: Array2<f64>,
    /// Bias vector (dimension)
    bias: Array1<f64>,
}

impl LinearTransform {
    pub fn transform(&self, embedding: &[f32]) -> Vec<f32> {
        // Apply learned transformation: y = Wx + b
        let x: Array1<f64> = embedding.iter().map(|&v| v as f64).collect();
        let result = self.weights.dot(&x) + &self.bias;
        result.iter().map(|&v| v as f32).collect()
    }
}
```

### Training Loop

```rust
impl FineTuner {
    pub fn train_step(&mut self, batch: &TrainingBatch) -> f64 {
        let mut total_loss = 0.0;

        for triplet in &batch.triplets {
            // Forward pass
            let anchor_t = self.transform.transform(&triplet.anchor);
            let positive_t = self.transform.transform(&triplet.positive);
            let negative_t = self.transform.transform(&triplet.negative);

            // Triplet loss: max(0, d(a,p) - d(a,n) + margin)
            let d_pos = cosine_distance(&anchor_t, &positive_t);
            let d_neg = cosine_distance(&anchor_t, &negative_t);
            let loss = (d_pos - d_neg + self.config.margin).max(0.0);

            // Backward pass (gradient computation)
            if loss > 0.0 {
                self.backward(triplet, d_pos, d_neg);
                total_loss += loss;
            }
        }

        // Apply gradients with momentum
        self.optimizer_step();

        total_loss / batch.triplets.len() as f64
    }
}
```

### Code References

- `src/finetuning.rs:43-156` — `FineTuneConfig` configuration
- `src/finetuning.rs:157-200` — `Interaction`, `Triplet`, `ContrastivePair` structures
- `src/finetuning.rs:205-350` — `LinearTransform` implementation
- `src/finetuning.rs:387-500` — `FineTuner` main implementation
- `src/finetuning.rs:1124-1200` — `SharedFineTuner` thread-safe wrapper

## Consequences

### Benefits

1. **Automatic improvement** — Search quality improves from user interactions
2. **Low latency** — Linear transform adds <1ms to queries
3. **No external dependencies** — Fine-tuning happens inside the database
4. **Continuous learning** — Model updates incrementally, not in batch jobs
5. **Lightweight** — Only transforms embeddings, doesn't retrain base model
6. **Reversible** — Original embeddings preserved, transform can be reverted

### Tradeoffs

1. **Limited expressiveness** — Linear transform can't fix fundamentally bad embeddings
2. **Feedback requirements** — Needs sufficient interaction data to be effective
3. **Cold start** — New deployments start with identity transform
4. **Computation overhead** — Training consumes CPU/GPU resources
5. **Potential overfitting** — Can memorize patterns in small datasets

### What This Enabled

- Self-improving search without ML pipeline maintenance
- Personalized search based on user interaction history
- Domain adaptation without external training infrastructure
- A/B testing of fine-tuned vs. base embeddings
- Continuous model improvement from production feedback

### What This Prevented

- Full model fine-tuning (only learns linear transforms)
- Architectural changes to embedding model
- Training on raw text (requires pre-computed embeddings)
- Offline-only fine-tuning workflows

### Online Learning Strategy

```rust
// Continuous learning with periodic checkpoints
async fn online_learning_loop(&mut self) {
    let mut batch = Vec::new();

    loop {
        // Collect interactions
        match self.interaction_receiver.recv().await {
            Some(interaction) => {
                batch.push(interaction);

                // Train when batch is full
                if batch.len() >= self.config.batch_size {
                    let triplets = self.generate_triplets(&batch);
                    let loss = self.train_step(&TrainingBatch { triplets });

                    // Log metrics
                    self.stats.update(loss, batch.len());

                    // Checkpoint periodically
                    if self.should_checkpoint() {
                        self.save_checkpoint().await?;
                    }

                    batch.clear();
                }
            }
            None => break,
        }
    }
}
```

### Evaluation Metrics

```rust
pub struct FineTunerStats {
    /// Total triplets trained on
    pub triplets_trained: u64,
    /// Average loss over recent batches
    pub avg_loss: f64,
    /// MRR improvement (if validation set available)
    pub mrr_improvement: Option<f64>,
    /// Training throughput (triplets/sec)
    pub throughput: f64,
}
```
