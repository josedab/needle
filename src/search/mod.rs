//! Query planning, routing, natural language filters, cross-collection search.
#![allow(clippy::unwrap_used)] // tech debt: 279 unwrap() calls remaining
#[cfg(feature = "experimental")]
pub mod collaborative_search;
pub mod cross_collection;
pub mod federated;
pub mod nl_filter;
/// NeedleQL query executor service for CLI, REST, and SDK consumption.
#[cfg(feature = "experimental")]
pub mod needleql_executor;
/// NeedleQL Language Server Protocol (LSP) for IDE integration.
#[cfg(feature = "experimental")]
pub mod needleql_lsp;
pub mod query_builder;
pub mod query_explain;
pub mod query_lang;
pub mod query_planner;
pub mod reranker;
pub mod cost_estimator;
pub mod routing;
pub mod collection_federation;
pub mod graphql_api;
pub mod graphrag;
pub mod pipeline;
/// Composable search pipeline DSL with built-in templates for common RAG patterns.
pub mod search_pipeline;
#[cfg(feature = "experimental")]
pub mod semantic_cache;
/// Embedded SQL analytics engine over vector metadata (COUNT, SUM, AVG, GROUP BY, HAVING).
pub mod sql_analytics;
