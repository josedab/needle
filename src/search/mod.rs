//! Query planning, routing, natural language filters, cross-collection search.
#[cfg(feature = "experimental")]
pub mod collaborative_search;
pub mod cross_collection;
pub mod federated;
pub mod nl_filter;
pub mod query_builder;
pub mod query_explain;
pub mod query_lang;
pub mod query_planner;
pub mod reranker;
pub mod routing;
