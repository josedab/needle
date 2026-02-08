//! Plugin services.
#![allow(clippy::unwrap_used)] // tech debt: 156 unwrap() calls remaining

#[cfg(feature = "experimental")]
pub mod plugin_api;
#[cfg(feature = "experimental")]
pub mod plugin_ecosystem;
#[cfg(feature = "experimental")]
pub mod plugin_runtime;
#[cfg(feature = "experimental")]
pub mod wasm_browser;
#[cfg(feature = "experimental")]
pub mod wasm_persistence;
#[cfg(feature = "experimental")]
pub mod wasm_plugin_runtime;
#[cfg(feature = "experimental")]
pub mod wasm_sdk;
