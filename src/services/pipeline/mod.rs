//! Pipeline services.
#![allow(clippy::unwrap_used)] // tech debt: 192 unwrap() calls remaining

#[cfg(feature = "experimental")]
pub mod cdc_framework;
#[cfg(feature = "experimental")]
pub mod ingestion_pipeline;
pub mod ingestion_service;
#[cfg(feature = "experimental")]
pub mod pipeline_manager;
#[cfg(feature = "experimental")]
pub mod realtime_streaming;
#[cfg(feature = "experimental")]
pub mod streaming_ingest;
#[cfg(feature = "experimental")]
pub mod streaming_protocol;
#[cfg(feature = "experimental")]
pub mod vector_pipeline;
