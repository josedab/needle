//! Auto-tuning for index parameters and index type selection.
//!
//! This module provides automatic configuration for HNSW parameters and
//! intelligent index type selection based on data characteristics.
//!
//! # Index Type Selection
//!
//! Needle supports three index types, each optimized for different use cases:
//!
//! | Index | Best For | Memory | Latency | Recall |
//! |-------|----------|--------|---------|--------|
//! | **HNSW** | < 10M vectors, real-time search | High | Low | High |
//! | **IVF** | 5M-100M vectors, memory-constrained | Medium | Medium | Medium-High |
//! | **DiskANN** | > 10M vectors, disk-based search | Low | Medium-High | High |
//!
//! ## HNSW (Default)
//!
//! HNSW is the default index and best choice for most use cases:
//! - **Strengths**: Fast queries (<1ms), high recall (>95%), good for updates
//! - **Weaknesses**: High memory usage (~1KB per vector for M=16)
//! - **When to use**: Real-time applications, datasets < 10M vectors, when memory is available
//!
//! ## IVF (Inverted File Index)
//!
//! IVF uses clustering to reduce search space:
//! - **Strengths**: Lower memory than HNSW, scales to large datasets
//! - **Weaknesses**: Requires training, slower updates, lower recall at same latency
//! - **When to use**: Large datasets (5M-100M), memory-constrained environments
//!
//! ## DiskANN
//!
//! DiskANN stores vectors on disk with minimal RAM:
//! - **Strengths**: Can handle billions of vectors, minimal RAM
//! - **Weaknesses**: Higher latency due to disk I/O, requires SSD
//! - **When to use**: Massive datasets (>10M), when vectors don't fit in RAM
//!
//! # Quick Index Selection
//!
//! ```
//! use needle::tuning::quick_recommend_index;
//!
//! // Get a quick recommendation based on size
//! let index = quick_recommend_index(1_000_000, 384);
//! println!("Recommended index: {}", index);
//! ```
//!
//! # Detailed Index Recommendation
//!
//! ```
//! use needle::tuning::{IndexSelectionConstraints, recommend_index};
//!
//! // Detailed recommendation with constraints
//! let constraints = IndexSelectionConstraints::new(5_000_000, 768)
//!     .with_available_memory(16 * 1024 * 1024 * 1024) // 16GB RAM
//!     .with_low_latency_critical(true)
//!     .with_target_recall(0.95);
//!
//! let recommendation = recommend_index(&constraints);
//! println!("Recommended: {}", recommendation.recommended);
//! println!("Estimated memory: {} bytes", recommendation.estimated_memory_bytes);
//! for reason in &recommendation.explanation {
//!     println!("  - {}", reason);
//! }
//! ```
//!
//! # HNSW Parameter Tuning
//!
//! For HNSW, the key parameters are:
//! - **M**: Connections per node (default: 16). Higher = better recall, more memory
//! - **ef_construction**: Build-time search depth (default: 200). Higher = better index quality
//! - **ef_search**: Query-time search depth (default: 50). Higher = better recall, slower queries
//!
//! ```
//! use needle::tuning::{TuningConstraints, PerformanceProfile, auto_tune};
//!
//! let constraints = TuningConstraints::new(100_000, 384)
//!     .with_profile(PerformanceProfile::HighRecall)
//!     .with_min_recall(0.98);
//!
//! let result = auto_tune(&constraints);
//! println!("Recommended M: {}", result.config.m);
//! println!("Recommended ef_construction: {}", result.config.ef_construction);
//! println!("Estimated recall: {:.1}%", result.estimated_recall * 100.0);
//! ```

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

// ============ Index Type Recommendation ============

/// Recommended index type based on data characteristics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedIndex {
    /// HNSW - Hierarchical Navigable Small World graph
    /// Best for: most use cases, especially < 10M vectors with good memory availability
    Hnsw,
    /// IVF - Inverted File Index with clustering
    /// Best for: large datasets (1M-100M) where memory is constrained
    Ivf,
    /// DiskANN - Disk-based approximate nearest neighbor
    /// Best for: massive datasets (>10M) that don't fit in memory
    DiskAnn,
}

impl std::fmt::Display for RecommendedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecommendedIndex::Hnsw => write!(f, "HNSW"),
            RecommendedIndex::Ivf => write!(f, "IVF"),
            RecommendedIndex::DiskAnn => write!(f, "DiskANN"),
        }
    }
}

/// Result of index type recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexRecommendation {
    /// Primary recommended index type
    pub recommended: RecommendedIndex,
    /// Alternative index types that could also work
    pub alternatives: Vec<RecommendedIndex>,
    /// Estimated memory usage for the recommended index (bytes)
    pub estimated_memory_bytes: usize,
    /// Whether data fits in available memory
    pub fits_in_memory: bool,
    /// Explanation for the recommendation
    pub explanation: Vec<String>,
}

/// Constraints for index type recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexSelectionConstraints {
    /// Expected number of vectors
    pub expected_vectors: usize,
    /// Vector dimensions
    pub dimensions: usize,
    /// Available memory in bytes (None = unlimited/system memory)
    pub available_memory_bytes: Option<usize>,
    /// Available disk space in bytes (for DiskANN)
    pub available_disk_bytes: Option<usize>,
    /// Whether low latency is critical (favors in-memory indexes)
    pub low_latency_critical: bool,
    /// Target recall requirement (higher = may need better index)
    pub target_recall: f32,
    /// Whether data is expected to be frequently updated
    pub frequent_updates: bool,
}

impl Default for IndexSelectionConstraints {
    fn default() -> Self {
        Self {
            expected_vectors: 100_000,
            dimensions: 128,
            available_memory_bytes: None,
            available_disk_bytes: None,
            low_latency_critical: false,
            target_recall: 0.9,
            frequent_updates: false,
        }
    }
}

impl IndexSelectionConstraints {
    /// Create constraints with vector count and dimensions
    pub fn new(expected_vectors: usize, dimensions: usize) -> Self {
        Self {
            expected_vectors,
            dimensions,
            ..Default::default()
        }
    }

    /// Set available memory in bytes
    pub fn with_available_memory(mut self, bytes: usize) -> Self {
        self.available_memory_bytes = Some(bytes);
        self
    }

    /// Set available disk space in bytes
    pub fn with_available_disk(mut self, bytes: usize) -> Self {
        self.available_disk_bytes = Some(bytes);
        self
    }

    /// Mark low latency as critical
    pub fn with_low_latency_critical(mut self, critical: bool) -> Self {
        self.low_latency_critical = critical;
        self
    }

    /// Set target recall
    pub fn with_target_recall(mut self, recall: f32) -> Self {
        self.target_recall = recall.clamp(0.0, 1.0);
        self
    }

    /// Set whether updates are frequent
    pub fn with_frequent_updates(mut self, frequent: bool) -> Self {
        self.frequent_updates = frequent;
        self
    }
}

/// Recommend the best index type based on data characteristics and constraints
///
/// # Arguments
/// * `constraints` - Index selection constraints
///
/// # Returns
/// * `IndexRecommendation` with the recommended index and explanation
///
/// # Example
/// ```
/// use needle::tuning::{IndexSelectionConstraints, recommend_index, RecommendedIndex};
///
/// // Small dataset - HNSW recommended
/// let small = IndexSelectionConstraints::new(10_000, 384);
/// let rec = recommend_index(&small);
/// assert_eq!(rec.recommended, RecommendedIndex::Hnsw);
///
/// // Large dataset with limited memory - IVF might be recommended
/// let large = IndexSelectionConstraints::new(50_000_000, 768)
///     .with_available_memory(8 * 1024 * 1024 * 1024); // 8GB
/// let rec = recommend_index(&large);
/// println!("Recommended: {}", rec.recommended);
/// ```
pub fn recommend_index(constraints: &IndexSelectionConstraints) -> IndexRecommendation {
    let mut explanation = Vec::new();
    let mut alternatives = Vec::new();

    // Calculate memory requirements
    // HNSW: vector_data + M*2*8 per vector + overhead
    // IVF: vector_data + cluster_overhead (much lower than HNSW)
    // DiskANN: minimal RAM, data on disk

    let vector_bytes = constraints.dimensions * 4; // f32 = 4 bytes
    let hnsw_memory_per_vector = vector_bytes + 16 * 2 * 8 + 100; // M=16 default, 100 byte overhead
    let ivf_memory_per_vector = vector_bytes + 50; // Much lower overhead
    let diskann_memory_per_vector = 100; // Minimal - just index structures

    let hnsw_total_memory = hnsw_memory_per_vector * constraints.expected_vectors;
    let ivf_total_memory = ivf_memory_per_vector * constraints.expected_vectors;
    let diskann_total_memory = diskann_memory_per_vector * constraints.expected_vectors;

    explanation.push(format!(
        "Dataset: {} vectors × {} dimensions",
        constraints.expected_vectors, constraints.dimensions
    ));
    explanation.push(format!(
        "Estimated memory: HNSW={:.1}GB, IVF={:.1}GB, DiskANN={:.1}GB (index only)",
        hnsw_total_memory as f64 / 1e9,
        ivf_total_memory as f64 / 1e9,
        diskann_total_memory as f64 / 1e9
    ));

    // Determine if data fits in memory
    let fits_in_memory = match constraints.available_memory_bytes {
        Some(avail) => hnsw_total_memory <= avail,
        None => {
            // Assume 16GB available if not specified, be conservative
            hnsw_total_memory <= 16 * 1024 * 1024 * 1024
        }
    };

    // Decision logic
    let recommended = if constraints.expected_vectors < 100_000 {
        // Small dataset: HNSW is always best
        explanation.push("Small dataset (<100K): HNSW provides best recall and latency".to_string());
        alternatives.push(RecommendedIndex::Ivf);
        RecommendedIndex::Hnsw
    } else if constraints.expected_vectors < 1_000_000 {
        // Medium dataset: HNSW if fits, otherwise IVF
        if fits_in_memory {
            explanation.push("Medium dataset (<1M) fits in memory: HNSW recommended".to_string());
            alternatives.push(RecommendedIndex::Ivf);
            RecommendedIndex::Hnsw
        } else {
            explanation.push("Medium dataset (<1M) with memory constraints: IVF recommended".to_string());
            alternatives.push(RecommendedIndex::Hnsw);
            alternatives.push(RecommendedIndex::DiskAnn);
            RecommendedIndex::Ivf
        }
    } else if constraints.expected_vectors < 10_000_000 {
        // Large dataset: IVF unless memory is abundant
        if fits_in_memory && constraints.low_latency_critical {
            explanation.push("Large dataset (1-10M) with abundant memory and latency critical: HNSW recommended".to_string());
            alternatives.push(RecommendedIndex::Ivf);
            RecommendedIndex::Hnsw
        } else if fits_in_memory || ivf_total_memory <= constraints.available_memory_bytes.unwrap_or(usize::MAX) {
            explanation.push("Large dataset (1-10M): IVF recommended for memory efficiency".to_string());
            alternatives.push(RecommendedIndex::Hnsw);
            alternatives.push(RecommendedIndex::DiskAnn);
            RecommendedIndex::Ivf
        } else {
            explanation.push("Large dataset (1-10M) exceeds memory: DiskANN recommended".to_string());
            alternatives.push(RecommendedIndex::Ivf);
            RecommendedIndex::DiskAnn
        }
    } else {
        // Massive dataset (>10M): DiskANN unless very high memory available
        let has_abundant_memory = constraints
            .available_memory_bytes
            .map(|m| ivf_total_memory < m)
            .unwrap_or(false);

        if has_abundant_memory && constraints.low_latency_critical {
            explanation.push("Massive dataset (>10M) with abundant memory: IVF recommended".to_string());
            alternatives.push(RecommendedIndex::DiskAnn);
            RecommendedIndex::Ivf
        } else {
            explanation.push("Massive dataset (>10M): DiskANN recommended for scalability".to_string());
            alternatives.push(RecommendedIndex::Ivf);
            RecommendedIndex::DiskAnn
        }
    };

    // Add notes about specific constraints
    if constraints.frequent_updates {
        if recommended == RecommendedIndex::DiskAnn {
            explanation.push("Note: DiskANN has higher update overhead; consider IVF if updates are very frequent".to_string());
        }
        if recommended == RecommendedIndex::Ivf {
            explanation.push("Note: IVF may need periodic re-clustering for optimal performance after many updates".to_string());
        }
    }

    if constraints.target_recall > 0.95 {
        explanation.push(format!(
            "High recall target ({:.0}%): increase ef_search (HNSW) or nprobe (IVF)",
            constraints.target_recall * 100.0
        ));
    }

    let estimated_memory = match recommended {
        RecommendedIndex::Hnsw => hnsw_total_memory,
        RecommendedIndex::Ivf => ivf_total_memory,
        RecommendedIndex::DiskAnn => diskann_total_memory,
    };

    IndexRecommendation {
        recommended,
        alternatives,
        estimated_memory_bytes: estimated_memory,
        fits_in_memory,
        explanation,
    }
}

/// Quick function to get recommended index for common scenarios
pub fn quick_recommend_index(vectors: usize, dimensions: usize) -> RecommendedIndex {
    recommend_index(&IndexSelectionConstraints::new(vectors, dimensions)).recommended
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

    // Index recommendation tests

    #[test]
    fn test_recommend_index_small_dataset() {
        let constraints = IndexSelectionConstraints::new(10_000, 384);
        let rec = recommend_index(&constraints);

        assert_eq!(rec.recommended, RecommendedIndex::Hnsw);
        assert!(rec.fits_in_memory);
        assert!(!rec.explanation.is_empty());
    }

    #[test]
    fn test_recommend_index_medium_dataset() {
        let constraints = IndexSelectionConstraints::new(500_000, 512);
        let rec = recommend_index(&constraints);

        // Medium dataset with default assumptions should recommend HNSW
        assert_eq!(rec.recommended, RecommendedIndex::Hnsw);
    }

    #[test]
    fn test_recommend_index_large_dataset() {
        let constraints = IndexSelectionConstraints::new(5_000_000, 768);
        let rec = recommend_index(&constraints);

        // Large dataset should recommend IVF
        assert_eq!(rec.recommended, RecommendedIndex::Ivf);
        assert!(rec.alternatives.contains(&RecommendedIndex::Hnsw));
    }

    #[test]
    fn test_recommend_index_massive_dataset() {
        let constraints = IndexSelectionConstraints::new(100_000_000, 384);
        let rec = recommend_index(&constraints);

        // Massive dataset should recommend DiskANN
        assert_eq!(rec.recommended, RecommendedIndex::DiskAnn);
    }

    #[test]
    fn test_recommend_index_memory_constrained() {
        let constraints = IndexSelectionConstraints::new(1_000_000, 768)
            .with_available_memory(1024 * 1024 * 1024); // 1GB only

        let rec = recommend_index(&constraints);

        // Should not fit HNSW, recommend IVF or DiskANN
        assert!(rec.recommended == RecommendedIndex::Ivf || rec.recommended == RecommendedIndex::DiskAnn);
    }

    #[test]
    fn test_recommend_index_low_latency_critical() {
        let constraints = IndexSelectionConstraints::new(5_000_000, 384)
            .with_available_memory(64 * 1024 * 1024 * 1024) // 64GB
            .with_low_latency_critical(true);

        let rec = recommend_index(&constraints);

        // With abundant memory and latency critical, should prefer HNSW
        assert_eq!(rec.recommended, RecommendedIndex::Hnsw);
    }

    #[test]
    fn test_recommend_index_frequent_updates() {
        let constraints = IndexSelectionConstraints::new(50_000_000, 128)
            .with_frequent_updates(true);

        let rec = recommend_index(&constraints);

        // Should include note about updates
        let has_update_note = rec.explanation.iter().any(|e| e.contains("update"));
        assert!(has_update_note);
    }

    #[test]
    fn test_quick_recommend_index() {
        assert_eq!(quick_recommend_index(1_000, 128), RecommendedIndex::Hnsw);
        assert_eq!(quick_recommend_index(50_000_000, 768), RecommendedIndex::DiskAnn);
    }

    #[test]
    fn test_recommended_index_display() {
        assert_eq!(format!("{}", RecommendedIndex::Hnsw), "HNSW");
        assert_eq!(format!("{}", RecommendedIndex::Ivf), "IVF");
        assert_eq!(format!("{}", RecommendedIndex::DiskAnn), "DiskANN");
    }
}
