//! Quantization example for Needle vector database
//!
//! Run with: cargo run --example quantization

use needle::quantization::{BinaryQuantizer, ProductQuantizer, ScalarQuantizer};

fn main() {
    println!("=== Needle Quantization Example ===\n");

    // Generate sample vectors for training
    let vectors: Vec<Vec<f32>> = (0..1000)
        .map(|i| {
            (0..128)
                .map(|j| ((i * 128 + j) as f32 / 128000.0).sin() * 2.0)
                .collect()
        })
        .collect();

    let refs: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();

    // Scalar Quantization (4x compression)
    println!("--- Scalar Quantization ---");
    let sq = ScalarQuantizer::train(&refs);
    let test_vec = &vectors[500];
    let quantized = sq.quantize(test_vec);
    let dequantized = sq.dequantize(&quantized);

    let sq_error = compute_error(test_vec, &dequantized);
    println!("Original size: {} bytes", test_vec.len() * 4);
    println!("Quantized size: {} bytes (4x compression)", quantized.len());
    println!("Reconstruction error: {:.6}\n", sq_error);

    // Product Quantization (higher compression)
    println!("--- Product Quantization ---");
    let pq = ProductQuantizer::train(&refs, 8); // 8 subspaces
    let pq_codes = pq.encode(test_vec);
    let pq_dequantized = pq.decode(&pq_codes);

    let pq_error = compute_error(test_vec, &pq_dequantized);
    println!("Original size: {} bytes", test_vec.len() * 4);
    println!(
        "Quantized size: {} bytes ({}x compression)",
        pq_codes.len(),
        test_vec.len() * 4 / pq_codes.len()
    );
    println!("Reconstruction error: {:.6}\n", pq_error);

    // Binary Quantization (32x compression)
    println!("--- Binary Quantization ---");
    let bq = BinaryQuantizer::train(&refs);
    let binary = bq.quantize(test_vec);

    // Binary quantization is lossy and doesn't support full reconstruction
    // Just measure the size reduction
    println!("Original size: {} bytes", test_vec.len() * 4);
    println!(
        "Quantized size: {} bytes ({}x compression)",
        binary.len(),
        test_vec.len() * 4 / binary.len()
    );
    println!("Note: Binary quantization uses Hamming distance, not reconstruction\n");

    // Compare all methods
    println!("--- Comparison ---");
    println!("{:<20} {:>15} {:>20}", "Method", "Compression", "Error");
    println!("{:-<55}", "");
    println!("{:<20} {:>15} {:>20.6}", "Scalar (8-bit)", "4x", sq_error);
    println!("{:<20} {:>15} {:>20.6}", "Product (8 sub)", "64x", pq_error);
    println!(
        "{:<20} {:>15} {:>20}",
        "Binary (1-bit)", "32x", "N/A (Hamming)"
    );

    println!("\n=== Quantization Example Complete ===");
}

fn compute_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}
