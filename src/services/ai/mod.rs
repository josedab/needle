//! Ai services.
#![allow(clippy::unwrap_used)] // tech debt: 298 unwrap() calls remaining

#[cfg(feature = "experimental")]
pub mod agentic_memory_protocol;
#[cfg(feature = "experimental")]
pub mod agentic_workflow;
#[cfg(feature = "experimental")]
pub mod graph_knowledge_service;
#[cfg(feature = "experimental")]
pub mod graph_query;
#[cfg(feature = "experimental")]
pub mod graphrag_service;
#[cfg(feature = "experimental")]
pub mod llm_cache_middleware;
#[cfg(feature = "experimental")]
pub mod llm_tools;
#[cfg(feature = "experimental")]
pub mod rag_sdk;
#[cfg(feature = "experimental")]
pub mod semantic_cache;
