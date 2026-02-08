//! Compute services.
#![allow(clippy::unwrap_used)] // tech debt: 116 unwrap() calls remaining

#[cfg(feature = "experimental")]
pub mod adaptive_optimizer;
#[cfg(feature = "experimental")]
pub mod adaptive_service;
#[cfg(feature = "experimental")]
pub mod gpu_kernels;
#[cfg(feature = "experimental")]
pub mod time_travel_query;
#[cfg(feature = "experimental")]
pub mod transactional_api;
#[cfg(feature = "experimental")]
pub mod vector_transactions;
