//! Framework integrations: LangChain, LlamaIndex, Haystack, Semantic Kernel.

pub(crate) mod framework_common;
#[cfg(feature = "integrations")]
pub mod langchain;
#[cfg(feature = "integrations")]
pub mod llamaindex;
pub mod haystack;
pub mod semantic_kernel;
