//! Collection services.
#![allow(clippy::unwrap_used)] // tech debt: 161 unwrap() calls remaining

#[cfg(feature = "experimental")]
pub mod collection_bundle;
#[cfg(feature = "experimental")]
pub mod collection_federation;
#[cfg(feature = "experimental")]
pub mod collection_rbac;
#[cfg(feature = "experimental")]
pub mod materialized_views;
#[cfg(feature = "experimental")]
pub mod multimodal_collection;
pub mod multimodal_service;
pub mod pitr_service;
pub mod text_collection;
#[cfg(feature = "experimental")]
pub mod typed_schema;
pub mod vector_namespace;
