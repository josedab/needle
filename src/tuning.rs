//! Auto-tuning for HNSW parameters
//!
//! Automatically configures index parameters based on:
//! - Expected data size
//! - Available memory
//! - Performance requirements (latency vs recall)

use crate::hnsw::HnswConfig;
use serde::{Deserialize, Serialize};

/// Performance profile for auto-tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum PerformanceProfile {
    /// Optimize for lowest latency (may sacrifice some recall)
    LowLatency,
    /// Balance between latency and recall (default)
    #[default]
    Balanced,
    /// Optimize for highest recall (may have higher latency)
    HighRecall,
    /// Optimize for minimum memory usage
    LowMemory,
}


/// Constraints for auto-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningConstraints {
    /// Expected number of vectors
    pub expected_vectors: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Maximum memory budget in bytes (None = unlimited)
    pub max_memory_bytes: Option<usize>,
    /// Target query latency in milliseconds (None = no target)
    pub target_latency_ms: Option<f32>,
    /// Minimum acceptable recall (0.0-1.0)
    pub min_recall: f32,
    /// Performance profile
    pub profile: PerformanceProfile,
}

impl Default for TuningConstraints {
    fn default() -> Self {
        Self {
            expected_vectors: 10_000,
            dimensions: 128,
            max_memory_bytes: None,
            target_latency_ms: None,
            min_recall: 0.9,
            profile: PerformanceProfile::Balanced,
        }
    }
}

impl TuningConstraints {
    pub fn new(expected_vectors: usize, dimensions: usize) -> Self {
        Self {
            expected_vectors,
            dimensions,
            ..Default::default()
        }
    }

    pub fn with_profile(mut self, profile: PerformanceProfile) -> Self {
        self.profile = profile;
        self
    }

    pub fn with_memory_budget(mut self, bytes: usize) -> Self {
        self.max_memory_bytes = Some(bytes);
        self
    }

    pub fn with_target_latency(mut self, ms: f32) -> Self {
        self.target_latency_ms = Some(ms);
        self
    }

    pub fn with_min_recall(mut self, recall: f32) -> Self {
        self.min_recall = recall.clamp(0.0, 1.0);
        self
    }
}

/// Result of auto-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    /// Recommended HNSW config
    pub config: HnswConfig,
    /// Recommended ef_search for queries
    pub ef_search: usize,
    /// Estimated memory usage per vector (bytes)
    pub estimated_memory_per_vector: usize,
    /// Estimated total memory usage (bytes)
    pub estimated_total_memory: usize,
    /// Estimated recall at recommended ef_search
    pub estimated_recall: f32,
    /// Estimated query latency (ms)
    pub estimated_latency_ms: f32,
    /// Explanation of choices
    pub explanation: Vec<String>,
}

/// Auto-tune HNSW parameters based on constraints
pub fn auto_tune(constraints: &TuningConstraints) -> TuningResult {
    let mut explanation = Vec::new();

    // Base parameters from profile
    let (base_m, base_ef_construction, base_ef_search) = match constraints.profile {
        PerformanceProfile::LowLatency => (8, 100, 20),
        PerformanceProfile::Balanced => (16, 200, 50),
        PerformanceProfile::HighRecall => (32, 400, 100),
        PerformanceProfile::LowMemory => (8, 100, 30),
    };

    explanation.push(format!(
        "Base parameters from {:?} profile: M={}, ef_construction={}",
        constraints.profile, base_m, base_ef_construction
    ));

    // Adjust M based on data size
    let m = if constraints.expected_vectors < 1_000 {
        // Small dataset: lower M is fine
        (base_m / 2).max(4)
    } else if constraints.expected_vectors < 100_000 {
        // Medium dataset: use base
        base_m
    } else if constraints.expected_vectors < 1_000_000 {
        // Large dataset: slightly higher M
        (base_m * 3 / 2).min(48)
    } else {
        // Very large: higher M for better connectivity
        (base_m * 2).min(64)
    };

    if m != base_m {
        explanation.push(format!(
            "Adjusted M from {} to {} based on {} expected vectors",
            base_m, m, constraints.expected_vectors
        ));
    }

    // Adjust ef_construction based on recall requirement
    let ef_construction = if constraints.min_recall > 0.98 {
        (base_ef_construction * 2).min(800)
    } else if constraints.min_recall > 0.95 {
        (base_ef_construction * 3 / 2).min(500)
    } else if constraints.min_recall < 0.85 {
        base_ef_construction / 2
    } else {
        base_ef_construction
    };

    if ef_construction != base_ef_construction {
        explanation.push(format!(
            "Adjusted ef_construction from {} to {} for {:.0}% min recall",
            base_ef_construction,
            ef_construction,
            constraints.min_recall * 100.0
        ));
    }

    // Calculate memory per vector
    // Each vector has:
    // - Vector data: dimensions * 4 bytes
    // - HNSW connections: ~M * 2 * 8 bytes (on average, including upper layers)
    // - Metadata overhead: ~100 bytes average
    let vector_bytes = constraints.dimensions * 4;
    let connection_bytes = m * 2 * 8; // Approximate average
    let metadata_overhead = 100;
    let memory_per_vector = vector_bytes + connection_bytes + metadata_overhead;

    let total_memory = memory_per_vector * constraints.expected_vectors;

    // Adjust for memory constraints
    let (final_m, final_ef_construction) = if let Some(max_memory) = constraints.max_memory_bytes {
        if total_memory > max_memory {
            // Need to reduce memory usage
            let reduction_ratio = max_memory as f32 / total_memory as f32;
            let reduced_m = ((m as f32 * reduction_ratio.sqrt()) as usize).max(4);
            let reduced_ef = ((ef_construction as f32 * reduction_ratio) as usize).max(50);

            explanation.push(format!(
                "Reduced M from {} to {} and ef_construction from {} to {} to fit memory budget",
                m, reduced_m, ef_construction, reduced_ef
            ));

            (reduced_m, reduced_ef)
        } else {
            (m, ef_construction)
        }
    } else {
        (m, ef_construction)
    };

    // Calculate ef_search based on target latency and recall
    let ef_search = if let Some(target_ms) = constraints.target_latency_ms {
        // Lower ef_search for lower latency
        // Rough heuristic: ef_search ~= base_ef_search * (target_ms / 10)
        let scaled = (base_ef_search as f32 * target_ms / 10.0) as usize;
        scaled.clamp(10, 500)
    } else {
        // Use profile default, adjusted for recall
        if constraints.min_recall > 0.95 {
            base_ef_search * 2
        } else {
            base_ef_search
        }
    };

    // Estimate recall based on parameters
    // This is a rough approximation based on empirical observations
    let estimated_recall = estimate_recall(final_m, final_ef_construction, ef_search, constraints.expected_vectors);

    // Estimate latency based on parameters
    // Rough model: latency ~= log(N) * ef_search * 0.001 ms
    let estimated_latency = (constraints.expected_vectors as f32).ln() * ef_search as f32 * 0.001;

    explanation.push(format!(
        "Final parameters: M={}, ef_construction={}, ef_search={}",
        final_m, final_ef_construction, ef_search
    ));
    explanation.push(format!(
        "Estimated recall: {:.1}%, latency: {:.2}ms",
        estimated_recall * 100.0,
        estimated_latency
    ));

    let final_memory_per_vector = constraints.dimensions * 4 + final_m * 2 * 8 + metadata_overhead;
    let final_total_memory = final_memory_per_vector * constraints.expected_vectors;

    TuningResult {
        config: HnswConfig {
            m: final_m,
            m_max_0: final_m * 2,
            ef_construction: final_ef_construction,
            ef_search,
            ml: 1.0 / (final_m as f64).ln(),
        },
        ef_search,
        estimated_memory_per_vector: final_memory_per_vector,
        estimated_total_memory: final_total_memory,
        estimated_recall,
        estimated_latency_ms: estimated_latency,
        explanation,
    }
}

/// Estimate recall based on parameters (rough approximation)
fn estimate_recall(m: usize, ef_construction: usize, ef_search: usize, n: usize) -> f32 {
    // This is a simplified model based on empirical observations
    // Real recall depends on data distribution

    let base_recall = 0.7;

    // M contribution (diminishing returns)
    let m_factor = 1.0 - (1.0 / (1.0 + m as f32 / 16.0));

    // ef_construction contribution
    let ef_c_factor = 1.0 - (1.0 / (1.0 + ef_construction as f32 / 200.0));

    // ef_search contribution (most important at query time)
    let ef_s_factor = 1.0 - (1.0 / (1.0 + ef_search as f32 / 50.0));

    // Scale factor (recall decreases with more vectors)
    let scale_factor = 1.0 - 0.05 * (n as f32 / 1_000_000.0).min(1.0);

    let recall = base_recall + 0.1 * m_factor + 0.05 * ef_c_factor + 0.15 * ef_s_factor;
    (recall * scale_factor).min(0.999)
}

/// Quick function to get recommended config for common scenarios
pub fn quick_config(vectors: usize, dimensions: usize) -> HnswConfig {
    auto_tune(&TuningConstraints::new(vectors, dimensions)).config
}

/// Get recommended ef_search for runtime queries
pub fn recommended_ef_search(vectors: usize, target_recall: f32) -> usize {
    if vectors < 1_000 {
        20
    } else if vectors < 100_000 {
        if target_recall > 0.95 {
            100
        } else {
            50
        }
    } else if vectors < 1_000_000 {
        if target_recall > 0.95 {
            200
        } else {
            100
        }
    } else if target_recall > 0.95 {
        400
    } else {
        200
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_auto_tune_small_dataset() {
        let constraints = TuningConstraints::new(1000, 128);
        let result = auto_tune(&constraints);

        assert!(result.config.m >= 4);
        assert!(result.config.m <= 32);
        assert!(result.estimated_recall > 0.8);
    }

    #[test]
    fn test_auto_tune_large_dataset() {
        let constraints = TuningConstraints::new(10_000_000, 768);
        let result = auto_tune(&constraints);

        assert!(result.config.m >= 16);
        println!("Large dataset config: {:?}", result);
    }

    #[test]
    fn test_auto_tune_memory_constrained() {
        let constraints = TuningConstraints::new(100_000, 384)
            .with_memory_budget(200 * 1024 * 1024); // 200MB

        let result = auto_tune(&constraints);
        println!("Memory constrained: {:?}", result);
        assert!(result.estimated_total_memory <= 200 * 1024 * 1024);
    }

    #[test]
    fn test_auto_tune_high_recall() {
        let constraints = TuningConstraints::new(50_000, 256)
            .with_min_recall(0.99)
            .with_profile(PerformanceProfile::HighRecall);

        let result = auto_tune(&constraints);
        // High recall profile should produce higher ef values
        assert!(result.estimated_recall > 0.85);
        assert!(result.config.ef_construction >= 200);
    }

    #[test]
    fn test_auto_tune_low_latency() {
        let constraints = TuningConstraints::new(50_000, 256)
            .with_target_latency(5.0)
            .with_profile(PerformanceProfile::LowLatency);

        let result = auto_tune(&constraints);
        assert!(result.ef_search <= 50);
    }
}
