//! Framework integrations: LangChain, LlamaIndex, Haystack, Semantic Kernel.

#![allow(clippy::unwrap_used)] // tech debt: per-module unwrap cleanup in progress
pub(crate) mod framework_common;
pub mod haystack;
#[cfg(feature = "integrations")]
pub mod langchain;
#[cfg(feature = "integrations")]
pub mod llamaindex;
pub mod semantic_kernel;
