#![allow(clippy::unwrap_used)]
#![allow(dead_code)]

//! Adaptive Vector Compression Pipeline
//!
//! Composable compression pipeline using existing quantization methods (SQ, PQ, BQ)
//! and Matryoshka dimension reduction. Provides a `CompressionPipeline` trait with
//! composable stages, auto-selection via sample benchmarking, and Matryoshka integration.
//!
//! # Example
//!
//! ```rust,ignore
//! use needle::indexing::compression_pipeline::{
//!     CompressionPipelineBuilder, CompressionStage, AutoSelector,
//! };
//!
//! // Build a custom pipeline: dimension reduction + scalar quantization
//! let pipeline = CompressionPipelineBuilder::new()
//!     .add_stage(CompressionStage::DimensionReduction { target_dims: 128 })
//!     .add_stage(CompressionStage::ScalarQuantization)
//!     .build(&training_vectors)?;
//!
//! // Or use auto-selection
//! let auto = AutoSelector::new(target_compression_ratio: 8.0, max_recall_loss: 0.05);
//! let pipeline = auto.select(&training_vectors, dimensions)?;
//! ```

use crate::error::{NeedleError, Result};
use crate::indexing::quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};
use serde::{Deserialize, Serialize};

/// Trait for a composable compression stage.
pub trait CompressionStageOp: Send + Sync {
    /// Compress a vector, returning the compressed representation.
    fn compress(&self, vector: &[f32]) -> Vec<u8>;
    /// Decompress back to f32 vector.
    fn decompress(&self, data: &[u8]) -> Vec<f32>;
    /// Get the compression ratio (original_size / compressed_size).
    fn compression_ratio(&self) -> f64;
    /// Name of this stage.
    fn name(&self) -> &str;
    /// Output dimensions (for chaining).
    fn output_dimensions(&self) -> usize;
}

/// Enumeration of available compression stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionStage {
    /// 8-bit scalar quantization (4x compression).
    ScalarQuantization,
    /// Product quantization with configurable subvectors.
    ProductQuantization { num_subvectors: usize },
    /// Binary quantization (32x compression).
    BinaryQuantization,
    /// Matryoshka dimension reduction to target dimensions.
    DimensionReduction { target_dims: usize },
    /// Passthrough (no compression).
    Identity,
}

/// Scalar quantization stage wrapper.
struct SqStage {
    quantizer: ScalarQuantizer,
    dimensions: usize,
}

impl CompressionStageOp for SqStage {
    fn compress(&self, vector: &[f32]) -> Vec<u8> {
        self.quantizer.quantize(vector)
    }

    fn decompress(&self, data: &[u8]) -> Vec<f32> {
        self.quantizer.dequantize(data)
    }

    fn compression_ratio(&self) -> f64 {
        4.0 // f32 (4 bytes) -> u8 (1 byte)
    }

    fn name(&self) -> &str {
        "scalar_quantization"
    }

    fn output_dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Product quantization stage wrapper.
struct PqStage {
    quantizer: ProductQuantizer,
    dimensions: usize,
    num_subvectors: usize,
}

impl CompressionStageOp for PqStage {
    fn compress(&self, vector: &[f32]) -> Vec<u8> {
        self.quantizer.encode(vector)
    }

    fn decompress(&self, data: &[u8]) -> Vec<f32> {
        self.quantizer.decode(data)
    }

    fn compression_ratio(&self) -> f64 {
        // PQ: D*4 bytes -> num_subvectors bytes
        (self.dimensions as f64 * 4.0) / self.num_subvectors.max(1) as f64
    }

    fn name(&self) -> &str {
        "product_quantization"
    }

    fn output_dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Binary quantization stage wrapper.
struct BqStage {
    quantizer: BinaryQuantizer,
    dimensions: usize,
}

impl CompressionStageOp for BqStage {
    fn compress(&self, vector: &[f32]) -> Vec<u8> {
        self.quantizer.quantize(vector)
    }

    fn decompress(&self, data: &[u8]) -> Vec<f32> {
        // Binary quantization doesn't support lossless decompression,
        // return threshold-based reconstruction
        let mut result = Vec::with_capacity(self.dimensions);
        for byte_idx in 0..data.len() {
            for bit in 0..8 {
                let dim = byte_idx * 8 + bit;
                if dim >= self.dimensions {
                    break;
                }
                result.push(if (data[byte_idx] >> bit) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                });
            }
        }
        result
    }

    fn compression_ratio(&self) -> f64 {
        32.0 // f32 (32 bits) -> 1 bit
    }

    fn name(&self) -> &str {
        "binary_quantization"
    }

    fn output_dimensions(&self) -> usize {
        self.dimensions
    }
}

/// Matryoshka dimension reduction stage.
struct DimReductionStage {
    input_dims: usize,
    target_dims: usize,
}

impl CompressionStageOp for DimReductionStage {
    fn compress(&self, vector: &[f32]) -> Vec<u8> {
        // Truncate to target dimensions (Matryoshka-style)
        let truncated: Vec<f32> = vector.iter().take(self.target_dims).copied().collect();
        // Convert f32 to bytes
        truncated
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect()
    }

    fn decompress(&self, data: &[u8]) -> Vec<f32> {
        // Reconstruct truncated dimensions, pad with zeros
        let mut result = Vec::with_capacity(self.input_dims);
        for chunk in data.chunks(4) {
            if chunk.len() == 4 {
                let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                result.push(f32::from_le_bytes(bytes));
            }
        }
        // Pad remaining dimensions with 0
        result.resize(self.input_dims, 0.0);
        result
    }

    fn compression_ratio(&self) -> f64 {
        self.input_dims as f64 / self.target_dims.max(1) as f64
    }

    fn name(&self) -> &str {
        "dimension_reduction"
    }

    fn output_dimensions(&self) -> usize {
        self.target_dims
    }
}

/// Identity (passthrough) stage.
struct IdentityStage {
    dimensions: usize,
}

impl CompressionStageOp for IdentityStage {
    fn compress(&self, vector: &[f32]) -> Vec<u8> {
        vector.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    fn decompress(&self, data: &[u8]) -> Vec<f32> {
        data.chunks(4)
            .filter_map(|chunk| {
                if chunk.len() == 4 {
                    Some(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                } else {
                    None
                }
            })
            .collect()
    }

    fn compression_ratio(&self) -> f64 {
        1.0
    }

    fn name(&self) -> &str {
        "identity"
    }

    fn output_dimensions(&self) -> usize {
        self.dimensions
    }
}

/// A composed compression pipeline with multiple stages.
pub struct CompressionPipeline {
    stages: Vec<Box<dyn CompressionStageOp>>,
    input_dimensions: usize,
    description: String,
}

impl CompressionPipeline {
    /// Compress a vector through all stages.
    pub fn compress(&self, vector: &[f32]) -> Vec<u8> {
        if self.stages.is_empty() {
            return vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        }

        // First stage compresses from f32
        let mut data = self.stages[0].compress(vector);

        // Subsequent stages: decompress previous output, then re-compress
        for stage in self.stages.iter().skip(1) {
            let intermediate = self.stages[0].decompress(&data);
            data = stage.compress(&intermediate);
        }

        data
    }

    /// Decompress data back to f32 vector (approximate).
    pub fn decompress(&self, data: &[u8]) -> Vec<f32> {
        if self.stages.is_empty() {
            return data
                .chunks(4)
                .filter_map(|c| {
                    if c.len() == 4 {
                        Some(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    } else {
                        None
                    }
                })
                .collect();
        }

        // Decompress from last stage back
        let last = self.stages.last().expect("stages is non-empty");
        last.decompress(data)
    }

    /// Get the overall compression ratio.
    pub fn compression_ratio(&self) -> f64 {
        self.stages
            .iter()
            .map(|s| s.compression_ratio())
            .product::<f64>()
            .max(1.0)
    }

    /// Get a description of the pipeline stages.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Get the number of stages.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Get the stage names.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name()).collect()
    }
}

/// Builder for composing compression pipelines.
pub struct CompressionPipelineBuilder {
    stages: Vec<CompressionStage>,
}

impl CompressionPipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    /// Add a compression stage to the pipeline.
    #[must_use]
    pub fn add_stage(mut self, stage: CompressionStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the pipeline by training stages on the provided vectors.
    pub fn build(self, vectors: &[&[f32]]) -> Result<CompressionPipeline> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput(
                "No training vectors provided".to_string(),
            ));
        }

        let input_dimensions = vectors[0].len();
        let mut built_stages: Vec<Box<dyn CompressionStageOp>> = Vec::new();
        let mut current_dims = input_dimensions;
        let mut description_parts: Vec<String> = Vec::new();

        for stage_spec in &self.stages {
            let stage: Box<dyn CompressionStageOp> = match stage_spec {
                CompressionStage::ScalarQuantization => {
                    let sq = ScalarQuantizer::train(vectors);
                    description_parts.push(format!("SQ({}d)", current_dims));
                    Box::new(SqStage {
                        quantizer: sq,
                        dimensions: current_dims,
                    })
                }
                CompressionStage::ProductQuantization { num_subvectors } => {
                    let pq = ProductQuantizer::train(vectors, *num_subvectors);
                    description_parts.push(format!("PQ({}d, {}sv)", current_dims, num_subvectors));
                    Box::new(PqStage {
                        quantizer: pq,
                        dimensions: current_dims,
                        num_subvectors: *num_subvectors,
                    })
                }
                CompressionStage::BinaryQuantization => {
                    let bq = BinaryQuantizer::train(vectors);
                    description_parts.push(format!("BQ({}d)", current_dims));
                    Box::new(BqStage {
                        quantizer: bq,
                        dimensions: current_dims,
                    })
                }
                CompressionStage::DimensionReduction { target_dims } => {
                    let target = (*target_dims).min(current_dims);
                    description_parts.push(format!("DR({}→{}d)", current_dims, target));
                    let stage = Box::new(DimReductionStage {
                        input_dims: current_dims,
                        target_dims: target,
                    });
                    current_dims = target;
                    stage
                }
                CompressionStage::Identity => {
                    description_parts.push("ID".to_string());
                    Box::new(IdentityStage {
                        dimensions: current_dims,
                    })
                }
            };
            built_stages.push(stage);
        }

        let description = if description_parts.is_empty() {
            "identity".to_string()
        } else {
            description_parts.join(" → ")
        };

        Ok(CompressionPipeline {
            stages: built_stages,
            input_dimensions,
            description,
        })
    }
}

impl Default for CompressionPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of evaluating a compression option.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionEvaluation {
    /// Description of the pipeline.
    pub description: String,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
    /// Estimated recall loss (0.0 = no loss, 1.0 = complete loss).
    pub recall_loss: f64,
    /// Compressed size per vector in bytes.
    pub bytes_per_vector: usize,
    /// Time to compress one vector (microseconds).
    pub compress_time_us: u64,
    /// Time to decompress one vector (microseconds).
    pub decompress_time_us: u64,
}

/// Auto-selector that benchmarks compression options and picks the best.
pub struct AutoSelector {
    /// Target compression ratio (e.g., 8.0 = 8x compression).
    pub target_compression_ratio: f64,
    /// Maximum tolerable recall loss (e.g., 0.05 = 5%).
    pub max_recall_loss: f64,
    /// Number of sample vectors to benchmark.
    pub sample_size: usize,
}

impl AutoSelector {
    /// Create a new auto-selector.
    pub fn new(target_compression_ratio: f64, max_recall_loss: f64) -> Self {
        Self {
            target_compression_ratio,
            max_recall_loss,
            sample_size: 1000,
        }
    }

    /// Evaluate multiple compression options and select the best.
    pub fn select(
        &self,
        vectors: &[&[f32]],
        dimensions: usize,
    ) -> Result<(CompressionPipeline, CompressionEvaluation)> {
        if vectors.is_empty() {
            return Err(NeedleError::InvalidInput(
                "No vectors for auto-selection".to_string(),
            ));
        }

        // Sample vectors for benchmarking
        let sample: Vec<&[f32]> = if vectors.len() > self.sample_size {
            let step = vectors.len() / self.sample_size;
            vectors.iter().step_by(step).copied().take(self.sample_size).collect()
        } else {
            vectors.to_vec()
        };

        // Define candidate pipelines
        let candidates: Vec<Vec<CompressionStage>> = vec![
            // SQ only (4x)
            vec![CompressionStage::ScalarQuantization],
            // PQ with 4 subvectors
            vec![CompressionStage::ProductQuantization {
                num_subvectors: (dimensions / 4).max(1),
            }],
            // BQ (32x)
            vec![CompressionStage::BinaryQuantization],
            // Dim reduction + SQ (high compression)
            vec![
                CompressionStage::DimensionReduction {
                    target_dims: dimensions / 2,
                },
                CompressionStage::ScalarQuantization,
            ],
            // Dim reduction + PQ (very high compression)
            vec![
                CompressionStage::DimensionReduction {
                    target_dims: dimensions / 4,
                },
                CompressionStage::ProductQuantization {
                    num_subvectors: (dimensions / 16).max(1),
                },
            ],
        ];

        let mut best_pipeline = None;
        let mut best_eval = None;
        let mut best_score = f64::MAX;

        for candidate_stages in candidates {
            let mut builder = CompressionPipelineBuilder::new();
            for stage in &candidate_stages {
                builder = builder.add_stage(stage.clone());
            }

            let pipeline = match builder.build(&sample) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let eval = Self::evaluate_pipeline(&pipeline, &sample);

            // Score: prefer pipelines close to target ratio with low recall loss
            let ratio_distance =
                (pipeline.compression_ratio() - self.target_compression_ratio).abs()
                    / self.target_compression_ratio;
            let score = ratio_distance + eval.recall_loss * 10.0;

            if eval.recall_loss <= self.max_recall_loss && score < best_score {
                best_score = score;
                best_eval = Some(eval);
                best_pipeline = Some(pipeline);
            }
        }

        match (best_pipeline, best_eval) {
            (Some(pipeline), Some(eval)) => Ok((pipeline, eval)),
            _ => {
                // Fallback to SQ
                let pipeline = CompressionPipelineBuilder::new()
                    .add_stage(CompressionStage::ScalarQuantization)
                    .build(&sample)?;
                let eval = Self::evaluate_pipeline(&pipeline, &sample);
                Ok((pipeline, eval))
            }
        }
    }

    /// Evaluate a pipeline on sample vectors.
    fn evaluate_pipeline(
        pipeline: &CompressionPipeline,
        samples: &[&[f32]],
    ) -> CompressionEvaluation {
        let mut total_recall_loss = 0.0;
        let mut total_bytes = 0;
        let mut compress_time = std::time::Duration::ZERO;
        let mut decompress_time = std::time::Duration::ZERO;

        for &vec in samples.iter().take(100) {
            // Compress
            let start = std::time::Instant::now();
            let compressed = pipeline.compress(vec);
            compress_time += start.elapsed();

            total_bytes += compressed.len();

            // Decompress
            let start = std::time::Instant::now();
            let decompressed = pipeline.decompress(&compressed);
            decompress_time += start.elapsed();

            // Compute recall loss as normalized MSE
            let mse: f32 = vec
                .iter()
                .zip(decompressed.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f32>()
                / vec.len().max(1) as f32;

            let magnitude: f32 = vec.iter().map(|x| x.powi(2)).sum::<f32>() / vec.len().max(1) as f32;
            let relative_error = if magnitude > 0.0 {
                (mse / magnitude) as f64
            } else {
                0.0
            };
            total_recall_loss += relative_error;
        }

        let n = samples.len().min(100).max(1);
        let avg_bytes = total_bytes / n;

        CompressionEvaluation {
            description: pipeline.description().to_string(),
            compression_ratio: pipeline.compression_ratio(),
            recall_loss: total_recall_loss / n as f64,
            bytes_per_vector: avg_bytes,
            compress_time_us: compress_time.as_micros() as u64 / n as u64,
            decompress_time_us: decompress_time.as_micros() as u64 / n as u64,
        }
    }
}

/// Recommend a compression pipeline based on collection characteristics.
///
/// Takes vector count, dimensions, and target compression ratio and returns
/// a recommended pipeline configuration.
pub fn recommend_compression(
    vector_count: usize,
    dimensions: usize,
    target_ratio: f64,
) -> Vec<CompressionStage> {
    if target_ratio <= 1.5 {
        // Minimal compression
        return vec![CompressionStage::Identity];
    }

    if target_ratio <= 4.5 {
        // 4x: scalar quantization
        return vec![CompressionStage::ScalarQuantization];
    }

    if target_ratio <= 10.0 {
        // 8x: dimension reduction (half) + SQ
        return vec![
            CompressionStage::DimensionReduction {
                target_dims: dimensions / 2,
            },
            CompressionStage::ScalarQuantization,
        ];
    }

    if target_ratio <= 20.0 {
        // 16x: dimension reduction (quarter) + PQ
        let num_subvectors = (dimensions / 4 / 4).max(1); // each subvector ~ 4 dims
        return vec![
            CompressionStage::DimensionReduction {
                target_dims: dimensions / 4,
            },
            CompressionStage::ProductQuantization { num_subvectors },
        ];
    }

    // 32x+: binary quantization
    vec![CompressionStage::BinaryQuantization]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vectors;

    #[test]
    fn test_sq_pipeline() {
        let vecs = random_vectors(100, 64);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let pipeline = CompressionPipelineBuilder::new()
            .add_stage(CompressionStage::ScalarQuantization)
            .build(&refs)
            .unwrap();

        assert!(pipeline.compression_ratio() >= 3.0);
        assert_eq!(pipeline.stage_count(), 1);

        let compressed = pipeline.compress(&vecs[0]);
        let decompressed = pipeline.decompress(&compressed);
        assert_eq!(decompressed.len(), 64);
    }

    #[test]
    fn test_dim_reduction_pipeline() {
        let vecs = random_vectors(100, 128);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let pipeline = CompressionPipelineBuilder::new()
            .add_stage(CompressionStage::DimensionReduction { target_dims: 32 })
            .build(&refs)
            .unwrap();

        assert!(pipeline.compression_ratio() >= 3.0);

        let compressed = pipeline.compress(&vecs[0]);
        assert_eq!(compressed.len(), 32 * 4); // 32 dims * 4 bytes
    }

    #[test]
    fn test_auto_selector() {
        let vecs = random_vectors(200, 64);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let selector = AutoSelector::new(4.0, 0.5); // 4x compression, 50% max recall loss
        let (pipeline, eval) = selector.select(&refs, 64).unwrap();

        assert!(pipeline.compression_ratio() >= 1.0);
        assert!(eval.bytes_per_vector > 0);
    }

    #[test]
    fn test_identity_pipeline() {
        let vecs = random_vectors(10, 32);
        let refs: Vec<&[f32]> = vecs.iter().map(|v| v.as_slice()).collect();

        let pipeline = CompressionPipelineBuilder::new()
            .add_stage(CompressionStage::Identity)
            .build(&refs)
            .unwrap();

        assert!((pipeline.compression_ratio() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_empty_vectors_error() {
        let refs: Vec<&[f32]> = Vec::new();
        let result = CompressionPipelineBuilder::new()
            .add_stage(CompressionStage::ScalarQuantization)
            .build(&refs);
        assert!(result.is_err());
    }

    #[test]
    fn test_recommend_compression() {
        let stages_4x = recommend_compression(10_000, 384, 4.0);
        assert!(matches!(stages_4x[0], CompressionStage::ScalarQuantization));

        let stages_8x = recommend_compression(10_000, 384, 8.0);
        assert!(matches!(stages_8x[0], CompressionStage::DimensionReduction { .. }));

        let stages_32x = recommend_compression(10_000, 384, 32.0);
        assert!(matches!(stages_32x[0], CompressionStage::BinaryQuantization));

        let stages_1x = recommend_compression(10_000, 384, 1.0);
        assert!(matches!(stages_1x[0], CompressionStage::Identity));
    }
}
