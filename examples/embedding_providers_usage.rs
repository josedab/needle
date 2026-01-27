//! Embedding Providers Example — OpenAI, Cohere, Ollama
//!
//! Demonstrates configuring embedding providers for automatic
//! vector generation from text. Requires API keys for cloud providers.
//!
//! Run with: cargo run --example embedding_providers_usage --features embedding-providers

#[cfg(feature = "embedding-providers")]
fn main() {
    use needle::embeddings_provider::{
        MockConfig, MockProvider, OllamaConfig, OpenAIConfig,
    };

    println!("=== Needle Embedding Providers Example ===\n");

    // --- Mock Provider (no API key needed, for testing) ---
    println!("1. Mock Provider (for testing):");
    let mock_config = MockConfig {
        dimensions: 384,
        deterministic: true,
    };
    let _mock = MockProvider::new(mock_config);
    println!("   Created mock provider (384 dimensions, deterministic)\n");

    // --- OpenAI Configuration ---
    println!("2. OpenAI Provider Configuration:");
    let openai_config = OpenAIConfig {
        api_key: "sk-your-api-key-here".to_string(),
        model: "text-embedding-3-small".to_string(),
        ..OpenAIConfig::default()
    };
    println!("   Model: {}", openai_config.model);
    println!("   Set OPENAI_API_KEY env var to use in production\n");

    // --- Ollama Configuration (local, no API key) ---
    println!("3. Ollama Provider Configuration (local):");
    let ollama_config = OllamaConfig {
        base_url: "http://localhost:11434".to_string(),
        model: "nomic-embed-text".to_string(),
    };
    println!("   Base URL: {}", ollama_config.base_url);
    println!("   Model: {}", ollama_config.model);
    println!("   No API key needed — runs locally\n");

    println!("To generate embeddings, create a provider and call embed():");
    println!("  let provider = OpenAIProvider::new(openai_config);");
    println!("  let embeddings = provider.embed(&[\"Hello world\"]).await?;");
}

#[cfg(not(feature = "embedding-providers"))]
fn main() {
    eprintln!("This example requires the 'embedding-providers' feature.");
    eprintln!("Run with: cargo run --example embedding_providers_usage --features embedding-providers");
}
