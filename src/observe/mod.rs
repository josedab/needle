//! Observability: telemetry, drift detection, anomaly detection, profiling.

#![allow(clippy::unwrap_used)] // tech debt: per-module unwrap cleanup in progress
pub mod anomaly;
pub mod audit;
pub mod dashboard;
pub mod drift;
pub mod lineage;
pub mod observability;
pub mod otel_service;
pub mod profiler;
pub mod telemetry;
