//! Shared test utilities for unit tests across the crate.
//!
//! Provides common helpers to avoid duplicating random vector generation
//! and test fixture setup in every module's `#[cfg(test)]` block.

use rand::Rng;

/// Shared placeholder API key for tests that require an API key value.
pub const TEST_API_KEY: &str = "test-api-key";

/// Generate a random vector of the given dimensionality with values in [0, 1).
pub fn random_vector(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..dim).map(|_| rng.gen::<f32>()).collect()
}

/// Generate a normalized (unit-length) random vector.
pub fn normalized_vector(dim: usize) -> Vec<f32> {
    let mut v = random_vector(dim);
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Generate `n` random vectors of the given dimensionality.
pub fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    (0..n).map(|_| random_vector(dim)).collect()
}

/// Create an in-memory [`Database`](crate::Database) with one collection of the
/// given name and dimensionality, returning the database handle.
pub fn create_test_collection(name: &str, dimensions: usize) -> crate::Database {
    let db = crate::Database::in_memory();
    db.create_collection(name, dimensions)
        .expect("failed to create test collection");
    db
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_vector_length() {
        assert_eq!(random_vector(128).len(), 128);
        assert_eq!(random_vector(0).len(), 0);
    }

    #[test]
    fn test_normalized_vector_unit_length() {
        let v = normalized_vector(64);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn test_random_vectors_count() {
        let vecs = random_vectors(10, 32);
        assert_eq!(vecs.len(), 10);
        assert!(vecs.iter().all(|v| v.len() == 32));
    }

    #[test]
    fn test_create_test_collection() {
        let db = create_test_collection("test_coll", 16);
        let coll = db.collection("test_coll").unwrap();
        assert_eq!(coll.len(), 0);
    }
}
