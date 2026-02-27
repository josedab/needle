use crate::error::Result;

/// Trait for embedding text
pub trait Embedder: Send + Sync {
    /// Generate embedding for text
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get embedding dimensions
    fn dimensions(&self) -> usize;
}

/// Simple mock embedder for testing
pub struct MockEmbedder {
    dimensions: usize,
}

impl MockEmbedder {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        let seed = hasher.finish();

        // Generate deterministic pseudo-random embedding
        let mut rng = SimpleRng::new(seed);
        let embedding: Vec<f32> = (0..self.dimensions)
            .map(|_| rng.next_f32() * 2.0 - 1.0)
            .collect();

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        Ok(embedding.into_iter().map(|x| x / norm).collect())
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}

// Simple RNG for deterministic embeddings
pub(crate) struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    pub(crate) fn next_f32(&mut self) -> f32 {
        (self.next_u64() as f32) / (u64::MAX as f32)
    }
}

#[cfg(test)]
mod tests {}
