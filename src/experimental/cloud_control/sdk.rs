#![allow(clippy::unwrap_used)]

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

/// Configuration for connecting to a managed Needle Cloud instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkConfig {
    /// Endpoint URL of the Needle Cloud instance.
    pub endpoint: String,
    /// API key for authentication.
    pub api_key: String,
    /// Connection timeout.
    pub connect_timeout_ms: u64,
    /// Request timeout.
    pub request_timeout_ms: u64,
    /// Maximum number of retry attempts.
    pub max_retries: u32,
    /// Base retry delay in milliseconds (exponential backoff).
    pub retry_base_delay_ms: u64,
    /// Enable local fallback when cloud is unreachable.
    pub local_fallback: bool,
}

impl Default for SdkConfig {
    fn default() -> Self {
        Self {
            endpoint: "https://api.needle.cloud".to_string(),
            api_key: String::new(),
            connect_timeout_ms: 5_000,
            request_timeout_ms: 30_000,
            max_retries: 3,
            retry_base_delay_ms: 100,
            local_fallback: false,
        }
    }
}

/// Connection state for the SDK wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionState {
    Connected,
    Connecting,
    Disconnected,
    Fallback,
}

/// Lightweight SDK wrapper that manages connection state, retries, and
/// optional local fallback.
pub struct NeedleCloudClient {
    config: SdkConfig,
    state: RwLock<ConnectionState>,
    request_count: std::sync::atomic::AtomicU64,
    error_count: std::sync::atomic::AtomicU64,
}

impl NeedleCloudClient {
    pub fn new(config: SdkConfig) -> Self {
        let initial_state = if config.api_key.is_empty() {
            ConnectionState::Disconnected
        } else {
            ConnectionState::Connected
        };

        Self {
            config,
            state: RwLock::new(initial_state),
            request_count: std::sync::atomic::AtomicU64::new(0),
            error_count: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Current connection state.
    pub fn state(&self) -> ConnectionState {
        *self.state.read()
    }

    /// Check if the client is ready to make requests.
    pub fn is_ready(&self) -> bool {
        matches!(
            *self.state.read(),
            ConnectionState::Connected | ConnectionState::Fallback
        )
    }

    /// Simulate a connection attempt (for testing and SDK bootstrapping).
    pub fn connect(&self) -> Result<()> {
        if self.config.api_key.is_empty() {
            return Err(NeedleError::Unauthorized("No API key configured".into()));
        }

        *self.state.write() = ConnectionState::Connected;
        Ok(())
    }

    /// Record a successful request.
    pub fn record_request(&self) {
        self.request_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record a failed request. Transitions to fallback if too many errors.
    pub fn record_error(&self) {
        let errors = self
            .error_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        let requests = self
            .request_count
            .load(std::sync::atomic::Ordering::Relaxed);

        // If error rate exceeds 50% over recent requests, switch to fallback.
        if requests > 10 && errors as f64 / requests as f64 > 0.5 && self.config.local_fallback {
            *self.state.write() = ConnectionState::Fallback;
        }
    }

    /// Compute the retry delay for a given attempt using exponential backoff.
    pub fn retry_delay_ms(&self, attempt: u32) -> u64 {
        self.config.retry_base_delay_ms * 2u64.pow(attempt.min(10))
    }

    /// Get connection statistics.
    pub fn stats(&self) -> SdkStats {
        SdkStats {
            state: *self.state.read(),
            total_requests: self
                .request_count
                .load(std::sync::atomic::Ordering::Relaxed),
            total_errors: self.error_count.load(std::sync::atomic::Ordering::Relaxed),
            endpoint: self.config.endpoint.clone(),
        }
    }
}

/// Statistics from the SDK connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SdkStats {
    pub state: ConnectionState,
    pub total_requests: u64,
    pub total_errors: u64,
    pub endpoint: String,
}
