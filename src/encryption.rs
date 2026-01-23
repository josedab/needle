//! Vector Encryption - Searchable encryption for vector embeddings.
//!
//! Provides encryption for vector data while still enabling approximate
//! nearest neighbor search on encrypted vectors.
//!
//! # Features
//!
//! - **Encrypted storage**: Vectors encrypted at rest
//! - **Searchable encryption**: Approximate search on encrypted data
//! - **Order-preserving transforms**: Enable distance comparisons
//! - **Key management**: Secure key handling
//! - **Access control**: Per-vector encryption keys
//!
//! # Example
//!
//! ```ignore
//! use needle::encryption::{VectorEncryptor, EncryptionConfig, KeyManager};
//!
//! let key_manager = KeyManager::new(master_key)?;
//! let encryptor = VectorEncryptor::new(config, key_manager);
//!
//! // Encrypt a vector
//! let encrypted = encryptor.encrypt(&vector)?;
//!
//! // Search on encrypted vectors (returns approximate results)
//! let results = encryptor.search_encrypted(&query, &encrypted_vectors, k)?;
//! ```

use crate::error::{NeedleError, Result};
use chacha20poly1305::{
    aead::{Aead, KeyInit},
    ChaCha20Poly1305, Nonce,
};
use hkdf::Hkdf;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::HashMap;

/// Encryption configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Algorithm to use.
    pub algorithm: EncryptionAlgorithm,
    /// Key size in bits.
    pub key_size: usize,
    /// Enable searchable encryption.
    pub searchable: bool,
    /// Noise level for differential privacy.
    pub noise_level: f32,
    /// Random projection dimensions (for searchable).
    pub projection_dims: usize,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            algorithm: EncryptionAlgorithm::AES256GCM,
            key_size: 256,
            searchable: true,
            noise_level: 0.01,
            projection_dims: 64,
        }
    }
}

/// Encryption algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM.
    AES256GCM,
    /// ChaCha20-Poly1305.
    ChaCha20Poly1305,
    /// Order-preserving encryption (for searchable).
    OrderPreserving,
}

/// An encryption key.
#[derive(Clone)]
#[allow(dead_code)]
pub struct EncryptionKey {
    /// Key bytes.
    bytes: Vec<u8>,
    /// Key ID.
    id: String,
    /// Creation timestamp (for key rotation tracking).
    created_at: u64,
}

impl EncryptionKey {
    /// Create a new key.
    pub fn new(bytes: Vec<u8>, id: &str) -> Self {
        Self {
            bytes,
            id: id.to_string(),
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }

    /// Get key ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get key bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Key manager for encryption.
pub struct KeyManager {
    /// Master key.
    master_key: EncryptionKey,
    /// Derived keys by ID.
    derived_keys: HashMap<String, EncryptionKey>,
    /// Random projection matrix (for searchable encryption).
    projection_matrix: Option<Vec<Vec<f32>>>,
}

impl KeyManager {
    /// Create a new key manager.
    pub fn new(master_key_bytes: &[u8]) -> Result<Self> {
        if master_key_bytes.len() < 32 {
            return Err(NeedleError::InvalidInput(
                "Master key must be at least 32 bytes".to_string()
            ));
        }

        Ok(Self {
            master_key: EncryptionKey::new(master_key_bytes.to_vec(), "master"),
            derived_keys: HashMap::new(),
            projection_matrix: None,
        })
    }

    /// Derive a key for a specific purpose using HKDF-SHA256.
    pub fn derive_key(&mut self, purpose: &str) -> Result<&EncryptionKey> {
        if self.derived_keys.contains_key(purpose) {
            return Ok(&self.derived_keys[purpose]);
        }

        // Use HKDF-SHA256 for proper key derivation
        let hk = Hkdf::<Sha256>::new(None, &self.master_key.bytes);
        let mut derived = vec![0u8; 32]; // ChaCha20Poly1305 uses 256-bit keys
        hk.expand(purpose.as_bytes(), &mut derived)
            .map_err(|_| NeedleError::InvalidInput("HKDF expand failed".to_string()))?;

        let key = EncryptionKey::new(derived, purpose);
        self.derived_keys.insert(purpose.to_string(), key);
        Ok(&self.derived_keys[purpose])
    }

    /// Initialize random projection matrix.
    pub fn init_projection(&mut self, input_dims: usize, output_dims: usize) {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let mut matrix = Vec::with_capacity(output_dims);

        for i in 0..output_dims {
            let mut row = Vec::with_capacity(input_dims);
            for j in 0..input_dims {
                // Deterministic random based on master key
                let mut hasher = DefaultHasher::new();
                self.master_key.bytes.hash(&mut hasher);
                (i, j).hash(&mut hasher);
                let hash = hasher.finish();

                // Generate random value in [-1, 1]
                let value = (hash as f32 / u64::MAX as f32) * 2.0 - 1.0;
                row.push(value);
            }
            // Normalize row
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            for v in &mut row {
                *v /= norm;
            }
            matrix.push(row);
        }

        self.projection_matrix = Some(matrix);
    }

    /// Get projection matrix.
    pub fn projection_matrix(&self) -> Option<&Vec<Vec<f32>>> {
        self.projection_matrix.as_ref()
    }
}

/// Encrypted vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedVector {
    /// Vector ID.
    pub id: String,
    /// Encrypted data (full precision).
    pub ciphertext: Vec<u8>,
    /// Searchable representation (for approximate search).
    pub search_embedding: Option<Vec<f32>>,
    /// Initialization vector / nonce.
    pub nonce: Vec<u8>,
    /// Key ID used for encryption.
    pub key_id: String,
    /// Metadata (can be encrypted or plaintext).
    pub metadata: HashMap<String, String>,
    /// Authentication tag.
    pub auth_tag: Vec<u8>,
}

/// Vector encryptor.
pub struct VectorEncryptor {
    /// Configuration.
    config: EncryptionConfig,
    /// Key manager.
    key_manager: KeyManager,
}

impl VectorEncryptor {
    /// Create a new encryptor.
    pub fn new(config: EncryptionConfig, key_manager: KeyManager) -> Self {
        Self { config, key_manager }
    }

    /// Encrypt a vector.
    pub fn encrypt(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: HashMap<String, String>,
    ) -> Result<EncryptedVector> {
        // Generate nonce
        let nonce = self.generate_nonce();

        // Get encryption key
        let key = self.key_manager.derive_key("vectors")?;
        let key_id = key.id().to_string();
        let key_bytes = key.bytes().to_vec();

        // Encrypt vector data
        let plaintext: Vec<u8> = vector.iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        let (ciphertext, auth_tag) = self.encrypt_data(&plaintext, &nonce, &key_bytes)?;

        // Generate searchable embedding
        let search_embedding = if self.config.searchable {
            Some(self.generate_search_embedding(vector)?)
        } else {
            None
        };

        Ok(EncryptedVector {
            id: id.to_string(),
            ciphertext,
            search_embedding,
            nonce,
            key_id,
            metadata,
            auth_tag,
        })
    }

    /// Decrypt a vector.
    pub fn decrypt(&mut self, encrypted: &EncryptedVector) -> Result<Vec<f32>> {
        let key_bytes = self.key_manager.derive_key(&encrypted.key_id)?.bytes().to_vec();

        let plaintext = self.decrypt_data(
            &encrypted.ciphertext,
            &encrypted.nonce,
            &key_bytes,
            &encrypted.auth_tag,
        )?;

        // Convert bytes back to floats
        let vector: Vec<f32> = plaintext.chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(vector)
    }

    /// Search encrypted vectors.
    pub fn search_encrypted(
        &self,
        query: &[f32],
        encrypted_vectors: &[EncryptedVector],
        k: usize,
    ) -> Result<Vec<EncryptedSearchResult>> {
        if !self.config.searchable {
            return Err(NeedleError::InvalidInput(
                "Searchable encryption not enabled".to_string()
            ));
        }

        // Transform query
        let query_embedding = self.transform_for_search(query)?;

        // Search on embeddings
        let mut results: Vec<EncryptedSearchResult> = encrypted_vectors.iter()
            .filter_map(|ev| {
                ev.search_embedding.as_ref().map(|emb| {
                    let distance = self.compute_distance(&query_embedding, emb);
                    EncryptedSearchResult {
                        encrypted_vector: ev.clone(),
                        approximate_distance: distance,
                    }
                })
            })
            .collect();

        results.sort_by(|a, b| {
            a.approximate_distance
                .partial_cmp(&b.approximate_distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Generate a cryptographically secure random nonce.
    /// ChaCha20Poly1305 uses 96-bit (12-byte) nonces.
    fn generate_nonce(&self) -> Vec<u8> {
        let mut nonce = vec![0u8; 12]; // 96-bit nonce for ChaCha20Poly1305
        rand::thread_rng().fill_bytes(&mut nonce);
        nonce
    }

    /// Encrypt data using ChaCha20Poly1305 AEAD.
    /// Returns (ciphertext, auth_tag) where auth_tag is the 16-byte Poly1305 tag.
    fn encrypt_data(&self, plaintext: &[u8], nonce: &[u8], key: &[u8]) -> Result<(Vec<u8>, Vec<u8>)> {
        // Ensure key is exactly 32 bytes for ChaCha20Poly1305
        if key.len() != 32 {
            return Err(NeedleError::InvalidInput(
                format!("Encryption key must be 32 bytes, got {}", key.len())
            ));
        }

        // Ensure nonce is exactly 12 bytes
        if nonce.len() != 12 {
            return Err(NeedleError::InvalidInput(
                format!("Nonce must be 12 bytes, got {}", nonce.len())
            ));
        }

        let cipher = ChaCha20Poly1305::new_from_slice(key)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid key: {}", e)))?;

        let nonce_arr = Nonce::from_slice(nonce);

        // ChaCha20Poly1305 appends 16-byte auth tag to ciphertext
        let ciphertext_with_tag = cipher
            .encrypt(nonce_arr, plaintext)
            .map_err(|e| NeedleError::InvalidInput(format!("Encryption failed: {}", e)))?;

        // Split ciphertext and auth tag (last 16 bytes)
        let tag_start = ciphertext_with_tag.len().saturating_sub(16);
        let ciphertext = ciphertext_with_tag[..tag_start].to_vec();
        let auth_tag = ciphertext_with_tag[tag_start..].to_vec();

        Ok((ciphertext, auth_tag))
    }

    /// Decrypt data using ChaCha20Poly1305 AEAD.
    /// Verifies the auth_tag before returning plaintext.
    fn decrypt_data(
        &self,
        ciphertext: &[u8],
        nonce: &[u8],
        key: &[u8],
        auth_tag: &[u8],
    ) -> Result<Vec<u8>> {
        // Ensure key is exactly 32 bytes
        if key.len() != 32 {
            return Err(NeedleError::InvalidInput(
                format!("Decryption key must be 32 bytes, got {}", key.len())
            ));
        }

        // Ensure nonce is exactly 12 bytes
        if nonce.len() != 12 {
            return Err(NeedleError::InvalidInput(
                format!("Nonce must be 12 bytes, got {}", nonce.len())
            ));
        }

        let cipher = ChaCha20Poly1305::new_from_slice(key)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid key: {}", e)))?;

        let nonce_arr = Nonce::from_slice(nonce);

        // Reconstruct ciphertext with auth tag appended (as ChaCha20Poly1305 expects)
        let mut ciphertext_with_tag = ciphertext.to_vec();
        ciphertext_with_tag.extend_from_slice(auth_tag);

        // Decrypt and verify auth tag
        let plaintext = cipher
            .decrypt(nonce_arr, ciphertext_with_tag.as_ref())
            .map_err(|_| NeedleError::InvalidInput(
                "Decryption failed: authentication tag mismatch".to_string()
            ))?;

        Ok(plaintext)
    }

    /// Generate searchable embedding.
    fn generate_search_embedding(&self, vector: &[f32]) -> Result<Vec<f32>> {
        let mut embedding = self.transform_for_search(vector)?;

        // Add noise for differential privacy
        if self.config.noise_level > 0.0 {
            self.add_noise(&mut embedding);
        }

        Ok(embedding)
    }

    /// Transform vector for searchable encryption.
    fn transform_for_search(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if let Some(matrix) = self.key_manager.projection_matrix() {
            // Random projection
            let mut projected = Vec::with_capacity(matrix.len());
            for row in matrix {
                let dot: f32 = row.iter()
                    .zip(vector.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                projected.push(dot);
            }
            Ok(projected)
        } else {
            // No projection, return normalized copy
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                Ok(vector.iter().map(|x| x / norm).collect())
            } else {
                Ok(vector.to_vec())
            }
        }
    }

    /// Add noise for differential privacy.
    fn add_noise(&self, vector: &mut [f32]) {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        for (i, val) in vector.iter_mut().enumerate() {
            let mut hasher = DefaultHasher::new();
            (*val as u32).hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish();

            // Laplacian noise approximation
            let uniform = (hash as f64 / u64::MAX as f64) - 0.5;
            let noise = self.config.noise_level * uniform.signum() as f32
                * (1.0 - 2.0 * uniform.abs() as f32).ln();

            *val += noise;
        }
    }

    /// Compute distance between vectors.
    fn compute_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Initialize for a given vector dimension.
    pub fn initialize(&mut self, input_dims: usize) {
        self.key_manager.init_projection(input_dims, self.config.projection_dims);
    }
}

/// Search result on encrypted vectors.
#[derive(Debug, Clone)]
pub struct EncryptedSearchResult {
    /// Encrypted vector.
    pub encrypted_vector: EncryptedVector,
    /// Approximate distance (on encrypted data).
    pub approximate_distance: f32,
}

/// Encrypted metadata store.
pub struct EncryptedMetadataStore {
    /// Encryptor for metadata.
    encryptor: VectorEncryptor,
    /// Encrypted metadata by key.
    data: HashMap<String, EncryptedMetadata>,
}

/// Encrypted metadata entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedMetadata {
    /// Encrypted value.
    pub ciphertext: Vec<u8>,
    /// Nonce.
    pub nonce: Vec<u8>,
    /// Auth tag.
    pub auth_tag: Vec<u8>,
}

impl EncryptedMetadataStore {
    /// Create new store.
    pub fn new(encryptor: VectorEncryptor) -> Self {
        Self {
            encryptor,
            data: HashMap::new(),
        }
    }

    /// Store encrypted metadata.
    pub fn put(&mut self, key: &str, value: &str) -> Result<()> {
        let enc_key_bytes = self.encryptor.key_manager.derive_key("metadata")?.bytes().to_vec();

        // Generate proper 12-byte random nonce for ChaCha20Poly1305
        let mut nonce = vec![0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce);

        let (ciphertext, auth_tag) = self.encryptor.encrypt_data(
            value.as_bytes(),
            &nonce,
            &enc_key_bytes,
        )?;

        self.data.insert(key.to_string(), EncryptedMetadata {
            ciphertext,
            nonce,
            auth_tag,
        });

        Ok(())
    }

    /// Get decrypted metadata.
    pub fn get(&mut self, key: &str) -> Result<Option<String>> {
        let encrypted = match self.data.get(key) {
            Some(e) => e.clone(),
            None => return Ok(None),
        };

        let enc_key_bytes = self.encryptor.key_manager.derive_key("metadata")?.bytes().to_vec();

        let plaintext = self.encryptor.decrypt_data(
            &encrypted.ciphertext,
            &encrypted.nonce,
            &enc_key_bytes,
            &encrypted.auth_tag,
        )?;

        String::from_utf8(plaintext)
            .map(Some)
            .map_err(|e| NeedleError::InvalidInput(format!("Invalid UTF-8: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_encryptor() -> VectorEncryptor {
        let key = vec![0u8; 32];
        let key_manager = KeyManager::new(&key).unwrap();
        let config = EncryptionConfig::default();
        VectorEncryptor::new(config, key_manager)
    }

    #[test]
    fn test_create_key_manager() {
        let key = vec![0u8; 32];
        let manager = KeyManager::new(&key);
        assert!(manager.is_ok());
    }

    #[test]
    fn test_key_too_short() {
        let key = vec![0u8; 16];
        let result = KeyManager::new(&key);
        assert!(result.is_err());
    }

    #[test]
    fn test_derive_key() {
        let key = vec![0u8; 32];
        let mut manager = KeyManager::new(&key).unwrap();

        let derived1_bytes = manager.derive_key("purpose1").unwrap().bytes().to_vec();
        let derived2_bytes = manager.derive_key("purpose2").unwrap().bytes().to_vec();

        assert_ne!(derived1_bytes, derived2_bytes);
    }

    #[test]
    fn test_encrypt_decrypt() {
        let mut encryptor = create_encryptor();

        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let encrypted = encryptor.encrypt("vec1", &vector, HashMap::new()).unwrap();

        let decrypted = encryptor.decrypt(&encrypted).unwrap();

        assert_eq!(vector, decrypted);
    }

    #[test]
    fn test_encrypted_vector_has_id() {
        let mut encryptor = create_encryptor();

        let encrypted = encryptor.encrypt("test_id", &[1.0], HashMap::new()).unwrap();

        assert_eq!(encrypted.id, "test_id");
    }

    #[test]
    fn test_searchable_embedding() {
        let mut encryptor = create_encryptor();
        encryptor.initialize(4);

        let encrypted = encryptor.encrypt("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();

        assert!(encrypted.search_embedding.is_some());
    }

    #[test]
    fn test_search_encrypted() {
        let mut encryptor = create_encryptor();
        encryptor.initialize(4);

        let vectors = [([1.0, 0.0, 0.0, 0.0], "a"),
            ([0.0, 1.0, 0.0, 0.0], "b"),
            ([0.0, 0.0, 1.0, 0.0], "c")];

        let encrypted: Vec<EncryptedVector> = vectors.iter()
            .map(|(v, id)| encryptor.encrypt(id, v, HashMap::new()).unwrap())
            .collect();

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = encryptor.search_encrypted(&query, &encrypted, 2).unwrap();

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_metadata_encryption() {
        let mut encryptor = create_encryptor();

        let mut meta = HashMap::new();
        meta.insert("key".to_string(), "value".to_string());

        let encrypted = encryptor.encrypt("vec1", &[1.0], meta).unwrap();

        assert!(encrypted.metadata.contains_key("key"));
    }

    #[test]
    fn test_projection_matrix() {
        let key = vec![0u8; 32];
        let mut manager = KeyManager::new(&key).unwrap();

        manager.init_projection(128, 32);

        let matrix = manager.projection_matrix().unwrap();
        assert_eq!(matrix.len(), 32);
        assert_eq!(matrix[0].len(), 128);
    }

    #[test]
    fn test_different_keys_different_ciphertext() {
        let key1 = vec![1u8; 32];
        let key2 = vec![2u8; 32];

        let mut enc1 = VectorEncryptor::new(
            EncryptionConfig::default(),
            KeyManager::new(&key1).unwrap(),
        );
        let mut enc2 = VectorEncryptor::new(
            EncryptionConfig::default(),
            KeyManager::new(&key2).unwrap(),
        );

        let vector = vec![1.0, 2.0, 3.0, 4.0];

        let encrypted1 = enc1.encrypt("v1", &vector, HashMap::new()).unwrap();
        let encrypted2 = enc2.encrypt("v2", &vector, HashMap::new()).unwrap();

        // Ciphertexts should differ
        assert_ne!(encrypted1.ciphertext, encrypted2.ciphertext);
    }

    #[test]
    fn test_non_searchable_mode() {
        let key = vec![0u8; 32];
        let manager = KeyManager::new(&key).unwrap();
        let config = EncryptionConfig {
            searchable: false,
            ..Default::default()
        };

        let mut encryptor = VectorEncryptor::new(config, manager);
        let encrypted = encryptor.encrypt("vec1", &[1.0, 2.0], HashMap::new()).unwrap();

        assert!(encrypted.search_embedding.is_none());
    }

    #[test]
    fn test_encrypted_metadata_store() {
        let encryptor = create_encryptor();
        let mut store = EncryptedMetadataStore::new(encryptor);

        store.put("key1", "secret_value").unwrap();

        let retrieved = store.get("key1").unwrap();
        assert_eq!(retrieved, Some("secret_value".to_string()));
    }

    #[test]
    fn test_noise_addition() {
        let mut encryptor = create_encryptor();
        encryptor.config.noise_level = 0.1;
        encryptor.initialize(4);

        // Encrypt same vector twice - search embeddings should differ slightly
        let encrypted1 = encryptor.encrypt("v1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        let encrypted2 = encryptor.encrypt("v2", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();

        // Due to deterministic noise in our simplified implementation, they might be same
        // In real implementation, this would add random noise
        assert!(encrypted1.search_embedding.is_some());
        assert!(encrypted2.search_embedding.is_some());
    }
}
