//! Observability services.
#![allow(clippy::unwrap_used)] // tech debt: 12 unwrap() calls remaining

#[cfg(feature = "experimental")]
pub mod ann_benchmark;
#[cfg(feature = "experimental")]
pub mod benchmark_runner;
#[cfg(feature = "experimental")]
pub mod benchmark_suite;
#[cfg(feature = "experimental")]
pub mod drift_monitor;
#[cfg(feature = "experimental")]
pub mod evidence_collector;
#[cfg(feature = "experimental")]
pub mod triage_report;
#[cfg(feature = "experimental")]
pub mod vector_lineage;
#[cfg(feature = "experimental")]
pub mod visual_explorer;
