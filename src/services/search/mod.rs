//! Search services.

#[cfg(feature = "experimental")]
pub mod adaptive_index_selector;
#[cfg(feature = "experimental")]
pub mod encrypted_search;
#[cfg(feature = "experimental")]
pub mod needleql_executor;
#[cfg(feature = "experimental")]
pub mod needleql_lsp;
#[cfg(feature = "experimental")]
pub mod nl_filter_parser;
#[cfg(feature = "experimental")]
pub mod query_cache_middleware;
#[cfg(feature = "experimental")]
pub mod query_optimizer;
#[cfg(feature = "experimental")]
pub mod query_replay;
