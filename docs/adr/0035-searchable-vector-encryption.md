# ADR-0035: Searchable Vector Encryption

## Status

Accepted

## Context

Sensitive applications require encryption of vector data:

1. **Regulatory compliance** — HIPAA, GDPR, PCI-DSS mandate encryption at rest
2. **Data sovereignty** — Vectors may encode PII (face embeddings, medical records)
3. **Multi-tenant isolation** — Tenants shouldn't access each other's vectors
4. **Breach mitigation** — Stolen encrypted data is useless without keys

Challenge: Traditional encryption prevents similarity search — encrypted vectors have random distances.

Approaches considered:

| Approach | Search on Encrypted | Security Level | Performance |
|----------|---------------------|----------------|-------------|
| No encryption | N/A | None | Baseline |
| Encrypt at rest, decrypt for search | No | High | Decrypt overhead |
| Homomorphic encryption | Yes | Very High | 1000x slower |
| Order-preserving encryption | Approximate | Medium | 2-5x slower |
| Random projection + noise | Approximate | Medium | 1.5x slower |

## Decision

Implement **searchable vector encryption** using a combination of:
1. **Standard encryption** (AES-256-GCM, ChaCha20-Poly1305) for storage
2. **Order-preserving transforms** with differential privacy noise for searchable indices

### Encryption Algorithms

```rust
pub enum EncryptionAlgorithm {
    AES256GCM,           // Standard, hardware-accelerated
    ChaCha20Poly1305,    // Fast on systems without AES-NI
    OrderPreserving,     // Enables approximate search
}

pub struct EncryptionConfig {
    pub algorithm: EncryptionAlgorithm,
    pub key_size: usize,           // 256 bits
    pub searchable: bool,          // Enable search on encrypted
    pub noise_level: f32,          // Differential privacy (0.01 default)
    pub projection_dims: usize,    // Random projection dimensions
}
```

### Key Management

```rust
pub struct KeyManager {
    master_key: EncryptionKey,
    derived_keys: HashMap<String, EncryptionKey>,
}

impl KeyManager {
    /// Derive a collection-specific key from master key
    pub fn derive_key(&self, collection: &str) -> Result<EncryptionKey> {
        let hk = Hkdf::<Sha256>::new(
            Some(collection.as_bytes()),  // Salt
            &self.master_key.bytes,
        );

        let mut derived = vec![0u8; 32];
        hk.expand(b"needle-collection-key", &mut derived)
            .map_err(|_| NeedleError::EncryptionError("Key derivation failed"))?;

        Ok(EncryptionKey::new(derived, &format!("{}:derived", collection)))
    }

    /// Rotate keys (re-encrypt with new key)
    pub fn rotate_key(&mut self, collection: &str) -> Result<RotationResult> {
        let old_key = self.derived_keys.get(collection);
        let new_key = self.derive_key(collection)?;
        // ... re-encryption logic
        Ok(RotationResult { vectors_re_encrypted: count })
    }
}
```

### Searchable Encryption

For approximate search on encrypted vectors:

```rust
pub struct SearchableEncryptor {
    config: EncryptionConfig,
    projection_matrix: Vec<Vec<f32>>,  // Random projection
    noise_generator: NoiseGenerator,
}

impl SearchableEncryptor {
    /// Create searchable index entry (lossy, privacy-preserving)
    pub fn create_search_index(&self, vector: &[f32]) -> Vec<f32> {
        // Step 1: Random projection to lower dimension
        let projected = self.random_project(vector);

        // Step 2: Add calibrated noise for differential privacy
        let noised = self.add_noise(&projected);

        // Step 3: Order-preserving transform
        self.order_preserving_encode(&noised)
    }

    /// Search returns approximate results (may have false positives)
    pub fn search_encrypted(
        &self,
        query: &[f32],
        encrypted_index: &[Vec<f32>],
        k: usize,
    ) -> Vec<(usize, f32)> {
        let query_indexed = self.create_search_index(query);

        // Distance on transformed vectors approximates original distance
        let mut distances: Vec<(usize, f32)> = encrypted_index.iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(&query_indexed, v)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances
    }
}
```

### Two-Layer Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Storage Layer                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Encrypted Vectors (AES-256-GCM)                    │    │
│  │  - Full precision vectors                            │    │
│  │  - Decrypted only for final result enrichment       │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Search Layer                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Searchable Index (Order-Preserving + Noise)        │    │
│  │  - Approximate distances preserved                   │    │
│  │  - Differential privacy prevents reconstruction     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Code References

- `src/encryption.rs:29-39` — Encryption dependency imports
- `src/encryption.rs:41-65` — EncryptionConfig structure
- `src/encryption.rs:67-76` — EncryptionAlgorithm enum
- `src/encryption.rs:78-100` — EncryptionKey with rotation tracking

## Consequences

### Benefits

1. **Compliance ready** — Meets encryption-at-rest requirements
2. **Search on encrypted** — Approximate search without decryption
3. **Privacy preserving** — Differential privacy prevents vector reconstruction
4. **Per-collection keys** — Breach of one key doesn't expose all data
5. **Key rotation** — Support for regular key rotation without downtime

### Tradeoffs

1. **Approximate results** — Searchable encryption has ~5-10% recall loss
2. **Performance overhead** — 2-5x slower than unencrypted search
3. **Key management complexity** — Must securely store and rotate keys
4. **No exact search** — Can't do exact nearest neighbor on encrypted data

### Security Properties

| Property | Guaranteed? | Notes |
|----------|-------------|-------|
| Confidentiality at rest | Yes | AES-256-GCM |
| Confidentiality in search | Partial | Approximate distances leaked |
| Forward secrecy | With rotation | Old keys can't decrypt new data |
| Key compromise scope | Per-collection | Derived keys limit blast radius |

### What This Enabled

- Healthcare applications with PHI in embeddings
- Financial services with encrypted transaction vectors
- Multi-tenant SaaS with cryptographic isolation
- Compliance certifications (SOC 2, HIPAA, PCI-DSS)

### What This Prevented

- Plaintext vectors in storage (compliance failure)
- Full corpus access from single key compromise
- Vector reconstruction from search indices

### Usage Example

```rust
// Initialize encryption
let master_key = KeyManager::generate_master_key()?;
let key_manager = KeyManager::new(master_key);

let config = EncryptionConfig {
    algorithm: EncryptionAlgorithm::AES256GCM,
    key_size: 256,
    searchable: true,
    noise_level: 0.01,  // 1% noise for differential privacy
    projection_dims: 64,
};

let encryptor = VectorEncryptor::new(config, key_manager.derive_key("documents")?);

// Encrypt vector for storage
let encrypted = encryptor.encrypt(&vector)?;

// Create searchable index entry
let search_entry = encryptor.create_search_index(&vector);

// Search (returns candidate indices)
let candidates = encryptor.search_encrypted(&query, &search_index, 100)?;

// Decrypt and re-rank final results
let results: Vec<SearchResult> = candidates.iter()
    .map(|(idx, _)| {
        let decrypted = encryptor.decrypt(&encrypted_vectors[*idx])?;
        // Re-compute exact distance on decrypted vector
        Ok(SearchResult { id: ids[*idx], distance: exact_distance(&query, &decrypted) })
    })
    .collect::<Result<_>>()?;
```
