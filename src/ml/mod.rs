//! Embedding generation, models, fine-tuning, RAG.
pub mod auto_embed;
pub mod dimreduce;
pub mod embeddings_gateway;
#[cfg(feature = "embedding-providers")]
pub mod embeddings_provider;
pub mod finetuning;
#[cfg(feature = "experimental")]
pub mod inference_engine;
pub mod local_inference;
pub mod matryoshka;
pub mod model_registry;
pub mod multimodal;
pub mod rag;
