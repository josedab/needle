//! Encryption Example — Encrypted Vector Storage
//!
//! Demonstrates encrypting vectors using ChaCha20-Poly1305.
//!
//! Run with: cargo run --example encryption_usage --features encryption

#[cfg(feature = "encryption")]
fn main() -> needle::Result<()> {
    use needle::enterprise::encryption::{
        EncryptionConfig, KeyManager, VectorEncryptor,
    };
    use std::collections::HashMap;

    println!("=== Needle Encryption Example ===\n");

    // Create a 32-byte master key (in production, load from a secure vault)
    let master_key = b"this-is-a-32-byte-master-key!!XY";

    // Initialize the key manager with the master key
    let key_manager = KeyManager::new(master_key)?;
    println!("Key manager initialized");

    // Create an encryptor with default config
    let config = EncryptionConfig::default();
    let mut encryptor = VectorEncryptor::new(config, key_manager);
    println!("Encryptor created with {:?} algorithm\n", encryptor.config().algorithm);

    // Encrypt a vector
    let vector = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let mut metadata = HashMap::new();
    metadata.insert("title".to_string(), "Secret Document".to_string());

    let encrypted = encryptor.encrypt("doc_1", &vector, metadata)?;
    println!("Encrypted vector 'doc_1':");
    println!("  Ciphertext length: {} bytes", encrypted.ciphertext.len());
    println!("  Nonce length: {} bytes", encrypted.nonce.len());
    println!("  Key ID: {}", encrypted.key_id);

    // Decrypt the vector
    let decrypted = encryptor.decrypt(&encrypted)?;
    println!("\nDecrypted vector: {:?}", decrypted.vector);
    println!("Decrypted metadata: {:?}", decrypted.metadata);

    // Verify round-trip
    assert_eq!(vector, decrypted.vector, "Round-trip encryption failed!");
    println!("\n✓ Round-trip encryption verified successfully");

    Ok(())
}

#[cfg(not(feature = "encryption"))]
fn main() {
    eprintln!("This example requires the 'encryption' feature.");
    eprintln!("Run with: cargo run --example encryption_usage --features encryption");
}
