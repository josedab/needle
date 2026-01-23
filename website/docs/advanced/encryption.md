---
sidebar_position: 3
---

# Encryption

Needle supports encryption at rest to protect your vector data. This guide covers how to enable and configure encryption.

## Overview

Needle provides two encryption modes:

| Mode | Algorithm | Key Size | Use Case |
|------|-----------|----------|----------|
| AES-256-GCM | AES-256 with Galois/Counter Mode | 256 bits | Standard encryption |
| ChaCha20-Poly1305 | ChaCha20 with Poly1305 MAC | 256 bits | Software-only (no AES-NI) |

## Enabling Encryption

### Create Encrypted Database

```rust
use needle::{Database, DatabaseConfig, EncryptionConfig};

// Generate or load encryption key
let key: [u8; 32] = generate_secure_key();

// Configure encryption
let config = DatabaseConfig::new()
    .with_encryption(EncryptionConfig::aes256(&key));

// Create database
let db = Database::create_with_config("encrypted.needle", config)?;
```

### Open Encrypted Database

```rust
let key: [u8; 32] = load_key_from_secure_storage();

let config = DatabaseConfig::new()
    .with_encryption(EncryptionConfig::aes256(&key));

let db = Database::open_with_config("encrypted.needle", config)?;
```

## Key Management

### Generating Keys

```rust
use rand::RngCore;

fn generate_secure_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut key);
    key
}
```

### Deriving Keys from Passwords

```rust
use argon2::{Argon2, password_hash::PasswordHasher};

fn derive_key(password: &str, salt: &[u8; 16]) -> [u8; 32] {
    let argon2 = Argon2::default();

    let mut key = [0u8; 32];
    argon2
        .hash_password_into(password.as_bytes(), salt, &mut key)
        .expect("key derivation failed");

    key
}
```

### Secure Key Storage

**Environment Variable:**
```rust
let key_hex = std::env::var("NEEDLE_ENCRYPTION_KEY")?;
let key = hex::decode(&key_hex)?
    .try_into()
    .map_err(|_| "Invalid key length")?;
```

**HashiCorp Vault:**
```rust
use vaultrs::client::VaultClient;

async fn get_key_from_vault() -> [u8; 32] {
    let client = VaultClient::new(
        "https://vault.example.com:8200",
        std::env::var("VAULT_TOKEN").unwrap()
    ).unwrap();

    let secret = client
        .secrets
        .kv2
        .read("secret/needle", "encryption-key")
        .await
        .unwrap();

    hex::decode(secret.data["key"].as_str().unwrap())
        .unwrap()
        .try_into()
        .unwrap()
}
```

**AWS KMS:**
```rust
use aws_sdk_kms::Client;

async fn get_key_from_kms(kms_key_id: &str) -> [u8; 32] {
    let config = aws_config::load_from_env().await;
    let client = Client::new(&config);

    // Generate data key
    let response = client
        .generate_data_key()
        .key_id(kms_key_id)
        .key_spec("AES_256")
        .send()
        .await
        .unwrap();

    response.plaintext().unwrap().as_ref().try_into().unwrap()
}
```

## Encryption Algorithms

### AES-256-GCM (Recommended)

Advanced Encryption Standard with Galois/Counter Mode. Uses hardware acceleration when available (AES-NI).

```rust
let config = DatabaseConfig::new()
    .with_encryption(EncryptionConfig::aes256(&key));
```

**Pros:**
- Hardware acceleration on modern CPUs
- Widely audited and standardized
- Authenticated encryption (integrity + confidentiality)

### ChaCha20-Poly1305

Software-efficient stream cipher with Poly1305 MAC.

```rust
let config = DatabaseConfig::new()
    .with_encryption(EncryptionConfig::chacha20(&key));
```

**Pros:**
- Fast in software (no hardware acceleration needed)
- Constant-time implementation
- Good for ARM devices without AES-NI

## What Gets Encrypted

| Component | Encrypted | Notes |
|-----------|-----------|-------|
| Vector data | Yes | All vector values |
| Metadata | Yes | All JSON metadata |
| HNSW graph | Yes | Index structure |
| Collection names | Yes | |
| File header | No | Contains format version |

## Performance Impact

Encryption adds overhead. Typical impact on 1M vectors, 384 dimensions:

| Operation | Without Encryption | With Encryption | Overhead |
|-----------|-------------------|-----------------|----------|
| Insert | 0.1 ms | 0.15 ms | +50% |
| Search (k=10) | 5 ms | 6 ms | +20% |
| Save | 2 s | 3 s | +50% |
| Load | 1 s | 1.5 s | +50% |

With AES-NI (most modern x86 CPUs), overhead is ~10-20%.

## Key Rotation

Rotate encryption keys periodically:

```rust
fn rotate_key(
    db_path: &str,
    old_key: &[u8; 32],
    new_key: &[u8; 32],
) -> needle::Result<()> {
    // Open with old key
    let old_config = DatabaseConfig::new()
        .with_encryption(EncryptionConfig::aes256(old_key));
    let db = Database::open_with_config(db_path, old_config)?;

    // Export to temporary file (unencrypted in memory)
    let temp_path = format!("{}.temp", db_path);

    // Create new database with new key
    let new_config = DatabaseConfig::new()
        .with_encryption(EncryptionConfig::aes256(new_key));
    let new_db = Database::create_with_config(&temp_path, new_config)?;

    // Copy all collections
    for name in db.list_collections()? {
        let old_coll = db.collection(&name)?;
        let info = old_coll.info()?;

        new_db.create_collection(&name, info.dimensions, info.distance)?;
        let new_coll = new_db.collection(&name)?;

        for (id, vector, metadata) in old_coll.iter()? {
            new_coll.insert(&id, &vector, metadata)?;
        }
    }

    new_db.save()?;

    // Replace old database
    std::fs::rename(&temp_path, db_path)?;

    Ok(())
}
```

## CLI Usage

```bash
# Create encrypted database
NEEDLE_ENCRYPTION_KEY=<hex_key> needle create encrypted.needle

# All operations require the key
NEEDLE_ENCRYPTION_KEY=<hex_key> needle info encrypted.needle
NEEDLE_ENCRYPTION_KEY=<hex_key> needle search encrypted.needle -c docs -q "[...]" -k 10
```

## Security Considerations

### Do's

- Use cryptographically secure random keys
- Store keys in secure key management systems
- Rotate keys periodically
- Use TLS for data in transit
- Audit access to encryption keys

### Don'ts

- Don't hardcode keys in source code
- Don't store keys alongside encrypted data
- Don't use weak key derivation functions
- Don't disable encryption in production
- Don't use the same key for multiple databases

### Threat Model

Needle's encryption protects against:
- Unauthorized access to database files
- Data theft from disk
- Backup compromise

It does **not** protect against:
- Memory inspection while database is open
- Key theft
- Side-channel attacks
- Compromised host machine

## Searchable Encryption

For applications requiring search on encrypted data without decryption, Needle supports order-preserving encryption (OPE) for metadata filtering:

```rust
let config = DatabaseConfig::new()
    .with_encryption(EncryptionConfig::aes256(&key))
    .with_searchable_encryption(true);
```

This enables:
- Range queries on encrypted numeric metadata
- Equality queries on encrypted strings
- Slight security trade-off for functionality

## Next Steps

- [Production Deployment](/docs/guides/production)
- [Sharding](/docs/advanced/sharding)
- [API Reference](/docs/api-reference)
