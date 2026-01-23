//! Declarative Pipeline Manager
//!
//! Manages the lifecycle of declarative vector pipelines: create from YAML/JSON
//! config, start/stop/restart, monitor health, and report statistics.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::pipeline_manager::{
//!     PipelineManager, PipelineDef, SourceDef, StageDef,
//!     PipelineStatus, PipelineHealth,
//! };
//!
//! let mut mgr = PipelineManager::new(16); // max 16 pipelines
//!
//! let def = PipelineDef::new("my-etl")
//!     .source(SourceDef::Webhook { path: "/ingest".into() })
//!     .stage(StageDef::Chunk { size: 512, overlap: 64 })
//!     .stage(StageDef::Embed { model: "mock-384".into() })
//!     .stage(StageDef::Index { collection: "docs".into() });
//!
//! let id = mgr.create(def).unwrap();
//! mgr.start(&id).unwrap();
//! assert_eq!(mgr.status(&id).unwrap(), PipelineStatus::Running);
//! mgr.stop(&id).unwrap();
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Pipeline Definition ─────────────────────────────────────────────────────

/// Source connector definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceDef {
    /// HTTP webhook endpoint.
    Webhook { path: String },
    /// File system watcher.
    FileWatch { directory: String, pattern: String },
    /// Kafka consumer.
    Kafka { brokers: Vec<String>, topic: String, group: String },
    /// PostgreSQL CDC.
    PostgresCdc { connection: String, slot: String },
    /// Manual push (no auto-source).
    Manual,
}

/// Processing stage definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StageDef {
    /// Text chunking.
    Chunk { size: usize, overlap: usize },
    /// Embedding generation.
    Embed { model: String },
    /// Index into collection.
    Index { collection: String },
    /// Metadata enrichment.
    Enrich { fields: HashMap<String, String> },
    /// Filter by field value.
    Filter { field: String, value: String },
    /// Deduplication.
    Dedup,
}

/// A complete pipeline definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineDef {
    pub name: String,
    pub source: SourceDef,
    pub stages: Vec<StageDef>,
    pub description: Option<String>,
}

impl PipelineDef {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            source: SourceDef::Manual,
            stages: Vec::new(),
            description: None,
        }
    }

    #[must_use]
    pub fn source(mut self, source: SourceDef) -> Self {
        self.source = source;
        self
    }

    #[must_use]
    pub fn stage(mut self, stage: StageDef) -> Self {
        self.stages.push(stage);
        self
    }

    #[must_use]
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Parse a pipeline definition from a YAML-like string.
    /// Simplified parser that handles key: value format.
    pub fn from_yaml(yaml: &str) -> Result<Self> {
        let mut name = String::new();
        let mut stages = Vec::new();
        let mut source = SourceDef::Manual;

        for line in yaml.lines() {
            let line = line.trim();
            if line.starts_with("name:") {
                name = line.trim_start_matches("name:").trim().to_string();
            } else if line.starts_with("source:") {
                let src = line.trim_start_matches("source:").trim();
                source = match src {
                    "webhook" => SourceDef::Webhook { path: "/ingest".into() },
                    "manual" => SourceDef::Manual,
                    _ => SourceDef::Manual,
                };
            } else if line.starts_with("- chunk:") {
                let size: usize = line
                    .trim_start_matches("- chunk:")
                    .trim()
                    .parse()
                    .unwrap_or(512);
                stages.push(StageDef::Chunk { size, overlap: size / 8 });
            } else if line.starts_with("- embed:") {
                let model = line.trim_start_matches("- embed:").trim().to_string();
                stages.push(StageDef::Embed { model });
            } else if line.starts_with("- index:") {
                let coll = line.trim_start_matches("- index:").trim().to_string();
                stages.push(StageDef::Index { collection: coll });
            } else if line.starts_with("- dedup") {
                stages.push(StageDef::Dedup);
            }
        }

        if name.is_empty() {
            return Err(NeedleError::InvalidArgument("Pipeline name required".into()));
        }

        Ok(Self { name, source, stages, description: None })
    }
}

// ── Pipeline Status ─────────────────────────────────────────────────────────

/// Pipeline lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStatus {
    Created,
    Running,
    Stopped,
    Failed,
}

/// Runtime state of a managed pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineState {
    pub id: String,
    pub definition: PipelineDef,
    pub status: PipelineStatus,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub stopped_at: Option<u64>,
    pub records_processed: u64,
    pub records_failed: u64,
    pub error_message: Option<String>,
}

/// Pipeline health summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineHealth {
    pub total_pipelines: usize,
    pub running: usize,
    pub stopped: usize,
    pub failed: usize,
}

// ── Pipeline Manager ────────────────────────────────────────────────────────

/// Manages pipeline lifecycles.
pub struct PipelineManager {
    pipelines: HashMap<String, PipelineState>,
    max_pipelines: usize,
    next_id: u64,
}

impl PipelineManager {
    pub fn new(max_pipelines: usize) -> Self {
        Self {
            pipelines: HashMap::new(),
            max_pipelines,
            next_id: 0,
        }
    }

    /// Create a pipeline from a definition.
    pub fn create(&mut self, def: PipelineDef) -> Result<String> {
        if self.pipelines.len() >= self.max_pipelines {
            return Err(NeedleError::CapacityExceeded(format!(
                "Max pipelines ({}) reached", self.max_pipelines
            )));
        }

        let id = format!("pipe_{}", self.next_id);
        self.next_id += 1;

        let state = PipelineState {
            id: id.clone(),
            definition: def,
            status: PipelineStatus::Created,
            created_at: now_secs(),
            started_at: None,
            stopped_at: None,
            records_processed: 0,
            records_failed: 0,
            error_message: None,
        };

        self.pipelines.insert(id.clone(), state);
        Ok(id)
    }

    /// Start a pipeline.
    pub fn start(&mut self, id: &str) -> Result<()> {
        let state = self.pipelines.get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Pipeline '{id}'")))?;
        state.status = PipelineStatus::Running;
        state.started_at = Some(now_secs());
        state.stopped_at = None;
        Ok(())
    }

    /// Stop a pipeline.
    pub fn stop(&mut self, id: &str) -> Result<()> {
        let state = self.pipelines.get_mut(id)
            .ok_or_else(|| NeedleError::NotFound(format!("Pipeline '{id}'")))?;
        state.status = PipelineStatus::Stopped;
        state.stopped_at = Some(now_secs());
        Ok(())
    }

    /// Get pipeline status.
    pub fn status(&self, id: &str) -> Result<PipelineStatus> {
        self.pipelines.get(id)
            .map(|s| s.status)
            .ok_or_else(|| NeedleError::NotFound(format!("Pipeline '{id}'")))
    }

    /// Get pipeline details.
    pub fn get(&self, id: &str) -> Option<&PipelineState> {
        self.pipelines.get(id)
    }

    /// List all pipelines.
    pub fn list(&self) -> Vec<&PipelineState> {
        self.pipelines.values().collect()
    }

    /// Remove a pipeline (must be stopped first).
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        if let Some(state) = self.pipelines.get(id) {
            if state.status == PipelineStatus::Running {
                return Err(NeedleError::InvalidArgument("Stop pipeline before removing".into()));
            }
        }
        Ok(self.pipelines.remove(id).is_some())
    }

    /// Record that records were processed by a pipeline.
    pub fn record_progress(&mut self, id: &str, processed: u64, failed: u64) {
        if let Some(state) = self.pipelines.get_mut(id) {
            state.records_processed += processed;
            state.records_failed += failed;
        }
    }

    /// Get overall health summary.
    pub fn health(&self) -> PipelineHealth {
        let mut h = PipelineHealth { total_pipelines: self.pipelines.len(), running: 0, stopped: 0, failed: 0 };
        for s in self.pipelines.values() {
            match s.status {
                PipelineStatus::Running => h.running += 1,
                PipelineStatus::Stopped | PipelineStatus::Created => h.stopped += 1,
                PipelineStatus::Failed => h.failed += 1,
            }
        }
        h
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_start() {
        let mut mgr = PipelineManager::new(10);
        let def = PipelineDef::new("test")
            .source(SourceDef::Manual)
            .stage(StageDef::Chunk { size: 512, overlap: 64 })
            .stage(StageDef::Index { collection: "docs".into() });

        let id = mgr.create(def).unwrap();
        assert_eq!(mgr.status(&id).unwrap(), PipelineStatus::Created);

        mgr.start(&id).unwrap();
        assert_eq!(mgr.status(&id).unwrap(), PipelineStatus::Running);
    }

    #[test]
    fn test_stop_and_remove() {
        let mut mgr = PipelineManager::new(10);
        let id = mgr.create(PipelineDef::new("test")).unwrap();
        mgr.start(&id).unwrap();

        // Can't remove while running
        assert!(mgr.remove(&id).is_err());

        mgr.stop(&id).unwrap();
        assert!(mgr.remove(&id).unwrap());
        assert!(mgr.get(&id).is_none());
    }

    #[test]
    fn test_capacity_limit() {
        let mut mgr = PipelineManager::new(2);
        mgr.create(PipelineDef::new("a")).unwrap();
        mgr.create(PipelineDef::new("b")).unwrap();
        assert!(mgr.create(PipelineDef::new("c")).is_err());
    }

    #[test]
    fn test_health() {
        let mut mgr = PipelineManager::new(10);
        let id1 = mgr.create(PipelineDef::new("a")).unwrap();
        let _id2 = mgr.create(PipelineDef::new("b")).unwrap();
        mgr.start(&id1).unwrap();

        let h = mgr.health();
        assert_eq!(h.total_pipelines, 2);
        assert_eq!(h.running, 1);
        assert_eq!(h.stopped, 1);
    }

    #[test]
    fn test_record_progress() {
        let mut mgr = PipelineManager::new(10);
        let id = mgr.create(PipelineDef::new("test")).unwrap();
        mgr.start(&id).unwrap();

        mgr.record_progress(&id, 100, 5);
        let state = mgr.get(&id).unwrap();
        assert_eq!(state.records_processed, 100);
        assert_eq!(state.records_failed, 5);
    }

    #[test]
    fn test_from_yaml() {
        let yaml = "name: my-pipeline\nsource: webhook\n- chunk: 256\n- embed: mock-384\n- index: docs\n- dedup";
        let def = PipelineDef::from_yaml(yaml).unwrap();
        assert_eq!(def.name, "my-pipeline");
        assert_eq!(def.stages.len(), 4);
        assert!(matches!(def.source, SourceDef::Webhook { .. }));
    }

    #[test]
    fn test_yaml_missing_name() {
        let yaml = "source: webhook\n- chunk: 512";
        assert!(PipelineDef::from_yaml(yaml).is_err());
    }

    #[test]
    fn test_list_pipelines() {
        let mut mgr = PipelineManager::new(10);
        mgr.create(PipelineDef::new("a")).unwrap();
        mgr.create(PipelineDef::new("b")).unwrap();
        assert_eq!(mgr.list().len(), 2);
    }
}
