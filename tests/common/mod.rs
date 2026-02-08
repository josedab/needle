//! Shared test utilities for Needle integration tests.

use needle::Database;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::SystemTime;

/// Generate a random vector of the given dimension using a time-based seed.
pub fn random_vector(dim: usize) -> Vec<f32> {
    let seed = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_nanos() as u64;
    seeded_vector(dim, seed)
}

/// Generate a deterministic vector of the given dimension from a fixed seed.
pub fn seeded_vector(dim: usize, seed: u64) -> Vec<f32> {
    (0..dim)
        .map(|i| {
            let mut hasher = DefaultHasher::new();
            (seed.wrapping_add(i as u64)).hash(&mut hasher);
            (hasher.finish() as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Create an in-memory `Database` wrapped in an `Arc`, with a single collection
/// of the given name and dimensions already created.
pub fn create_test_db(collection_name: &str, dimensions: usize) -> Arc<Database> {
    let db = Arc::new(Database::in_memory());
    db.create_collection(collection_name, dimensions).unwrap();
    db
}
