//! Observability: telemetry, drift detection, anomaly detection, profiling.

#![allow(missing_docs)]
#![allow(clippy::unwrap_used)] // tech debt: 126 unwrap() calls remaining
pub mod anomaly;
pub mod audit;
pub mod dashboard;
pub mod drift;
pub mod lineage;
pub mod observability;
pub mod otel_service;
pub mod profiler;
pub mod telemetry;
