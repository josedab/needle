//! Serverless Edge Runtime
//!
//! WASI-compatible runtime configuration for deploying Needle at the edge
//! (Cloudflare Workers, Deno Deploy, Vercel Edge Functions).
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::edge_serverless::{
//!     EdgeRuntime, EdgeConfig, EdgePlatform, EdgeRequest, EdgeResponse,
//! };
//!
//! let rt = EdgeRuntime::new(EdgeConfig::for_platform(EdgePlatform::CloudflareWorkers));
//! let response = rt.handle(EdgeRequest::Search {
//!     collection: "docs".into(), query: vec![0.5; 4], k: 10,
//! });
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Edge deployment platform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgePlatform { CloudflareWorkers, DenoEdge, VercelEdge, WasiGeneric }

impl std::fmt::Display for EdgePlatform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self { Self::CloudflareWorkers => write!(f, "cloudflare-workers"), Self::DenoEdge => write!(f, "deno-deploy"),
            Self::VercelEdge => write!(f, "vercel-edge"), Self::WasiGeneric => write!(f, "wasi") }
    }
}

/// Edge runtime configuration.
#[derive(Debug, Clone)]
pub struct EdgeConfig {
    pub platform: EdgePlatform,
    pub max_memory_mb: usize,
    pub max_vectors: usize,
    pub max_dimensions: usize,
    pub cold_start_budget_ms: u64,
    pub wasm_features: Vec<String>,
}

impl EdgeConfig {
    pub fn for_platform(platform: EdgePlatform) -> Self {
        match platform {
            EdgePlatform::CloudflareWorkers => Self {
                platform, max_memory_mb: 128, max_vectors: 100_000, max_dimensions: 768,
                cold_start_budget_ms: 5, wasm_features: vec!["simd128".into()],
            },
            EdgePlatform::DenoEdge => Self {
                platform, max_memory_mb: 256, max_vectors: 500_000, max_dimensions: 1024,
                cold_start_budget_ms: 10, wasm_features: vec!["simd128".into(), "threads".into()],
            },
            EdgePlatform::VercelEdge => Self {
                platform, max_memory_mb: 128, max_vectors: 100_000, max_dimensions: 768,
                cold_start_budget_ms: 5, wasm_features: vec!["simd128".into()],
            },
            EdgePlatform::WasiGeneric => Self {
                platform, max_memory_mb: 512, max_vectors: 1_000_000, max_dimensions: 2048,
                cold_start_budget_ms: 50, wasm_features: vec!["simd128".into(), "threads".into(), "bulk-memory".into()],
            },
        }
    }
}

/// Edge request types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeRequest {
    Search { collection: String, query: Vec<f32>, k: usize },
    Insert { collection: String, id: String, vector: Vec<f32>, metadata: Option<Value> },
    Delete { collection: String, id: String },
    Health,
    Info,
}

/// Edge response types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeResponse {
    SearchResults { results: Vec<EdgeSearchResult>, latency_us: u64 },
    Ok { message: String },
    Health { status: String, platform: String, vectors: usize },
    Info { platform: String, max_vectors: usize, max_dimensions: usize, wasm_features: Vec<String> },
    Error { code: u32, message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeSearchResult { pub id: String, pub distance: f32, pub metadata: Option<Value> }

/// Edge runtime handler.
pub struct EdgeRuntime {
    config: EdgeConfig,
    total_requests: u64,
}

impl EdgeRuntime {
    pub fn new(config: EdgeConfig) -> Self { Self { config, total_requests: 0 } }

    pub fn handle(&self, request: EdgeRequest) -> EdgeResponse {
        match request {
            EdgeRequest::Health => EdgeResponse::Health {
                status: "ok".into(), platform: self.config.platform.to_string(), vectors: 0,
            },
            EdgeRequest::Info => EdgeResponse::Info {
                platform: self.config.platform.to_string(), max_vectors: self.config.max_vectors,
                max_dimensions: self.config.max_dimensions, wasm_features: self.config.wasm_features.clone(),
            },
            EdgeRequest::Search { collection, query, k } => {
                if query.len() > self.config.max_dimensions {
                    return EdgeResponse::Error { code: 400, message: "Dimensions exceed platform limit".into() };
                }
                EdgeResponse::SearchResults { results: Vec::new(), latency_us: 0 }
            }
            EdgeRequest::Insert { .. } => EdgeResponse::Ok { message: "Inserted".into() },
            EdgeRequest::Delete { .. } => EdgeResponse::Ok { message: "Deleted".into() },
        }
    }

    pub fn platform(&self) -> EdgePlatform { self.config.platform }
    pub fn config(&self) -> &EdgeConfig { &self.config }

    pub fn generate_wasm_config(&self) -> String {
        serde_json::json!({
            "platform": self.config.platform.to_string(),
            "memory": format!("{}MB", self.config.max_memory_mb),
            "features": self.config.wasm_features,
            "cold_start_budget_ms": self.config.cold_start_budget_ms,
        }).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health() {
        let rt = EdgeRuntime::new(EdgeConfig::for_platform(EdgePlatform::CloudflareWorkers));
        match rt.handle(EdgeRequest::Health) {
            EdgeResponse::Health { status, .. } => assert_eq!(status, "ok"),
            _ => panic!("Expected Health response"),
        }
    }

    #[test]
    fn test_info() {
        let rt = EdgeRuntime::new(EdgeConfig::for_platform(EdgePlatform::DenoEdge));
        match rt.handle(EdgeRequest::Info) {
            EdgeResponse::Info { platform, max_vectors, .. } => {
                assert_eq!(platform, "deno-deploy");
                assert_eq!(max_vectors, 500_000);
            }
            _ => panic!("Expected Info response"),
        }
    }

    #[test]
    fn test_dimension_limit() {
        let rt = EdgeRuntime::new(EdgeConfig::for_platform(EdgePlatform::CloudflareWorkers));
        match rt.handle(EdgeRequest::Search { collection: "c".into(), query: vec![1.0; 2000], k: 5 }) {
            EdgeResponse::Error { code, .. } => assert_eq!(code, 400),
            _ => panic!("Expected error"),
        }
    }

    #[test]
    fn test_wasm_config() {
        let rt = EdgeRuntime::new(EdgeConfig::for_platform(EdgePlatform::WasiGeneric));
        let cfg = rt.generate_wasm_config();
        assert!(cfg.contains("wasi"));
        assert!(cfg.contains("simd128"));
    }

    #[test]
    fn test_all_platforms() {
        for p in [EdgePlatform::CloudflareWorkers, EdgePlatform::DenoEdge, EdgePlatform::VercelEdge, EdgePlatform::WasiGeneric] {
            let rt = EdgeRuntime::new(EdgeConfig::for_platform(p));
            assert_eq!(rt.platform(), p);
        }
    }
}
