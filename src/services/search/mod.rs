//! Search services.

#[cfg(feature = "experimental")]
pub mod adaptive_index_selector;
#[cfg(feature = "experimental")]
pub mod encrypted_search;
/// Re-exported from [`crate::search::needleql_executor`] for backward compatibility.
#[cfg(feature = "experimental")]
pub use crate::search::needleql_executor;
/// Re-exported from [`crate::search::needleql_lsp`] for backward compatibility.
#[cfg(feature = "experimental")]
pub use crate::search::needleql_lsp;
#[cfg(feature = "experimental")]
pub mod nl_filter_parser;
#[cfg(feature = "experimental")]
pub mod query_cache_middleware;
#[cfg(feature = "experimental")]
pub mod query_optimizer;
#[cfg(feature = "experimental")]
pub mod query_replay;
