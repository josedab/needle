#![allow(clippy::unwrap_used)]
//! Agentic Workflow Engine
//!
//! Multi-agent shared memory, workflow primitives, chain-of-thought memory,
//! tool-use tracking, and automatic context window management.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::agentic_workflow::{
//!     WorkflowEngine, WorkflowConfig, AgentId, WorkflowStep,
//! };
//!
//! let mut engine = WorkflowEngine::new(WorkflowConfig::default());
//!
//! // Register agents
//! let agent_a = engine.register_agent("researcher", 4096);
//! let agent_b = engine.register_agent("writer", 4096);
//!
//! // Write to shared memory
//! engine.write_shared_memory(&agent_a, "findings", "Vector DBs are fast").unwrap();
//!
//! // Read from shared memory (cross-agent)
//! let val = engine.read_shared_memory(&agent_b, "findings").unwrap();
//! assert_eq!(val, "Vector DBs are fast");
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Agent Types ─────────────────────────────────────────────────────────────

/// Agent identifier.
pub type AgentId = String;

/// An agent in the workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    /// Unique agent ID.
    pub id: AgentId,
    /// Agent role/name.
    pub role: String,
    /// Maximum context window size (tokens).
    pub context_window: usize,
    /// Current context usage (tokens).
    pub context_used: usize,
    /// Agent status.
    pub status: AgentStatus,
    /// Registered at timestamp.
    pub registered_at: u64,
    /// Tool calls made by this agent.
    pub tool_calls: Vec<ToolCall>,
    /// Chain-of-thought entries.
    pub thought_chain: Vec<ThoughtEntry>,
}

/// Agent status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    Idle,
    Working,
    WaitingForInput,
    Completed,
    Failed,
}

/// A tool call made by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    /// Tool name.
    pub tool: String,
    /// Input parameters.
    pub input: serde_json::Value,
    /// Output result.
    pub output: Option<serde_json::Value>,
    /// Call timestamp.
    pub timestamp: u64,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Whether the call succeeded.
    pub success: bool,
}

/// A chain-of-thought entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThoughtEntry {
    /// Step number.
    pub step: usize,
    /// The thought/reasoning.
    pub thought: String,
    /// Action taken.
    pub action: Option<String>,
    /// Observation from action.
    pub observation: Option<String>,
    /// Timestamp.
    pub timestamp: u64,
}

// ── Shared Memory ───────────────────────────────────────────────────────────

/// An entry in shared memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedMemoryEntry {
    /// Memory key.
    pub key: String,
    /// Memory value.
    pub value: String,
    /// Who wrote it.
    pub written_by: AgentId,
    /// When it was written.
    pub written_at: u64,
    /// Access count.
    pub access_count: u64,
    /// Priority (higher = more important).
    pub priority: u32,
}

// ── Workflow Types ───────────────────────────────────────────────────────────

/// A workflow definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    /// Workflow ID.
    pub id: String,
    /// Workflow name.
    pub name: String,
    /// Steps in the workflow.
    pub steps: Vec<WorkflowStep>,
    /// Current step index.
    pub current_step: usize,
    /// Workflow status.
    pub status: WorkflowStatus,
    /// Created at.
    pub created_at: u64,
    /// Completed at.
    pub completed_at: Option<u64>,
}

/// A step in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowStep {
    /// Step name.
    pub name: String,
    /// Agent assigned to this step.
    pub agent_id: AgentId,
    /// Step type.
    pub step_type: StepType,
    /// Step status.
    pub status: StepStatus,
    /// Step output.
    pub output: Option<String>,
    /// Dependencies (step indices that must complete first).
    pub depends_on: Vec<usize>,
}

/// Workflow step type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepType {
    /// Agent executes a task.
    Execute { prompt: String },
    /// Agent searches vector memory.
    Search { query: String, k: usize },
    /// Agent writes to shared memory.
    Store { key: String, value: String },
    /// Conditional branch.
    Condition { condition: String },
    /// Wait for external input.
    WaitForInput,
    /// Fan-out to multiple agents.
    FanOut { agent_ids: Vec<AgentId> },
}

/// Step status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Skipped,
}

/// Workflow status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorkflowStatus {
    Created,
    Running,
    Paused,
    Completed,
    Failed,
}

// ── Observability & Replay ──────────────────────────────────────────────────

/// A recorded event for replay.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowEvent {
    /// Event type.
    pub event_type: EventType,
    /// Related agent.
    pub agent_id: Option<AgentId>,
    /// Related workflow.
    pub workflow_id: Option<String>,
    /// Event data.
    pub data: serde_json::Value,
    /// Timestamp.
    pub timestamp: u64,
}

/// Event type for the event log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    AgentRegistered,
    AgentStatusChange,
    ToolCallStart,
    ToolCallEnd,
    ThoughtAdded,
    MemoryWrite,
    MemoryRead,
    WorkflowCreated,
    StepStarted,
    StepCompleted,
    StepFailed,
    WorkflowCompleted,
    ContextEviction,
}

// ── Engine Config ───────────────────────────────────────────────────────────

/// Workflow engine configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowConfig {
    /// Maximum number of agents.
    pub max_agents: usize,
    /// Maximum shared memory entries.
    pub max_shared_memory: usize,
    /// Maximum event log size.
    pub max_events: usize,
    /// Default context window size.
    pub default_context_window: usize,
    /// Context eviction threshold (percentage of window used).
    pub context_eviction_threshold: f64,
    /// Maximum workflows.
    pub max_workflows: usize,
}

impl Default for WorkflowConfig {
    fn default() -> Self {
        Self {
            max_agents: 64,
            max_shared_memory: 10_000,
            max_events: 100_000,
            default_context_window: 8192,
            context_eviction_threshold: 0.9,
            max_workflows: 100,
        }
    }
}

/// Engine statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineStats {
    pub total_tool_calls: u64,
    pub total_thoughts: u64,
    pub total_memory_writes: u64,
    pub total_memory_reads: u64,
    pub total_context_evictions: u64,
    pub active_agents: usize,
    pub active_workflows: usize,
}

// ── Workflow Engine ─────────────────────────────────────────────────────────

/// The agentic workflow engine.
pub struct WorkflowEngine {
    config: WorkflowConfig,
    agents: HashMap<AgentId, Agent>,
    shared_memory: HashMap<String, SharedMemoryEntry>,
    workflows: HashMap<String, Workflow>,
    events: Vec<WorkflowEvent>,
    stats: EngineStats,
    next_agent_id: u64,
    next_workflow_id: u64,
}

impl WorkflowEngine {
    /// Create a new workflow engine.
    pub fn new(config: WorkflowConfig) -> Self {
        Self {
            config,
            agents: HashMap::new(),
            shared_memory: HashMap::new(),
            workflows: HashMap::new(),
            events: Vec::new(),
            stats: EngineStats::default(),
            next_agent_id: 1,
            next_workflow_id: 1,
        }
    }

    // ── Agent Management ────────────────────────────────────────────────

    /// Register a new agent.
    pub fn register_agent(&mut self, role: &str, context_window: usize) -> AgentId {
        let id = format!("agent-{:04}", self.next_agent_id);
        self.next_agent_id += 1;
        let ctx = if context_window == 0 {
            self.config.default_context_window
        } else {
            context_window
        };
        let agent = Agent {
            id: id.clone(),
            role: role.to_string(),
            context_window: ctx,
            context_used: 0,
            status: AgentStatus::Idle,
            registered_at: now_secs(),
            tool_calls: Vec::new(),
            thought_chain: Vec::new(),
        };
        self.agents.insert(id.clone(), agent);
        self.stats.active_agents += 1;
        self.record_event(EventType::AgentRegistered, Some(&id), None, serde_json::json!({ "role": role }));
        id
    }

    /// Get agent info.
    pub fn agent(&self, agent_id: &str) -> Option<&Agent> {
        self.agents.get(agent_id)
    }

    /// Update agent status.
    pub fn set_agent_status(&mut self, agent_id: &str, status: AgentStatus) -> Result<()> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Agent '{agent_id}'"))
        })?;
        agent.status = status;
        self.record_event(
            EventType::AgentStatusChange,
            Some(agent_id),
            None,
            serde_json::json!({ "status": format!("{:?}", status) }),
        );
        Ok(())
    }

    /// Record a tool call for an agent.
    pub fn record_tool_call(
        &mut self,
        agent_id: &str,
        tool: &str,
        input: serde_json::Value,
        output: Option<serde_json::Value>,
        duration_ms: u64,
        success: bool,
    ) -> Result<()> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Agent '{agent_id}'"))
        })?;
        let call = ToolCall {
            tool: tool.to_string(),
            input: input.clone(),
            output,
            timestamp: now_secs(),
            duration_ms,
            success,
        };
        agent.tool_calls.push(call);
        agent.context_used += 100; // estimate token usage
        self.stats.total_tool_calls += 1;

        self.record_event(
            if success { EventType::ToolCallEnd } else { EventType::ToolCallEnd },
            Some(agent_id),
            None,
            serde_json::json!({ "tool": tool, "success": success, "duration_ms": duration_ms }),
        );

        // Check context window management
        self.check_context_eviction(agent_id);
        Ok(())
    }

    /// Add a chain-of-thought entry.
    pub fn add_thought(
        &mut self,
        agent_id: &str,
        thought: &str,
        action: Option<String>,
        observation: Option<String>,
    ) -> Result<()> {
        let agent = self.agents.get_mut(agent_id).ok_or_else(|| {
            NeedleError::NotFound(format!("Agent '{agent_id}'"))
        })?;
        let step = agent.thought_chain.len() + 1;
        agent.thought_chain.push(ThoughtEntry {
            step,
            thought: thought.to_string(),
            action,
            observation,
            timestamp: now_secs(),
        });
        agent.context_used += thought.len() / 4; // rough token estimate
        self.stats.total_thoughts += 1;

        self.record_event(
            EventType::ThoughtAdded,
            Some(agent_id),
            None,
            serde_json::json!({ "step": step }),
        );
        self.check_context_eviction(agent_id);
        Ok(())
    }

    // ── Shared Memory ───────────────────────────────────────────────────

    /// Write to shared memory.
    pub fn write_shared_memory(
        &mut self,
        agent_id: &str,
        key: &str,
        value: &str,
    ) -> Result<()> {
        if !self.agents.contains_key(agent_id) {
            return Err(NeedleError::NotFound(format!("Agent '{agent_id}'")));
        }
        if self.shared_memory.len() >= self.config.max_shared_memory
            && !self.shared_memory.contains_key(key)
        {
            // Evict lowest priority entry
            if let Some(lowest) = self
                .shared_memory
                .iter()
                .min_by_key(|(_, e)| e.priority)
                .map(|(k, _)| k.clone())
            {
                self.shared_memory.remove(&lowest);
            }
        }
        self.shared_memory.insert(
            key.to_string(),
            SharedMemoryEntry {
                key: key.to_string(),
                value: value.to_string(),
                written_by: agent_id.to_string(),
                written_at: now_secs(),
                access_count: 0,
                priority: 1,
            },
        );
        self.stats.total_memory_writes += 1;
        self.record_event(
            EventType::MemoryWrite,
            Some(agent_id),
            None,
            serde_json::json!({ "key": key }),
        );
        Ok(())
    }

    /// Read from shared memory.
    pub fn read_shared_memory(&mut self, agent_id: &str, key: &str) -> Result<String> {
        if !self.agents.contains_key(agent_id) {
            return Err(NeedleError::NotFound(format!("Agent '{agent_id}'")));
        }
        let entry = self.shared_memory.get_mut(key).ok_or_else(|| {
            NeedleError::NotFound(format!("Memory key '{key}'"))
        })?;
        entry.access_count += 1;
        let value = entry.value.clone();
        self.stats.total_memory_reads += 1;
        self.record_event(
            EventType::MemoryRead,
            Some(agent_id),
            None,
            serde_json::json!({ "key": key }),
        );
        Ok(value)
    }

    /// List all shared memory keys.
    pub fn list_shared_memory(&self) -> Vec<&SharedMemoryEntry> {
        self.shared_memory.values().collect()
    }

    /// Set priority for a shared memory entry.
    pub fn set_memory_priority(&mut self, key: &str, priority: u32) -> Result<()> {
        let entry = self.shared_memory.get_mut(key).ok_or_else(|| {
            NeedleError::NotFound(format!("Memory key '{key}'"))
        })?;
        entry.priority = priority;
        Ok(())
    }

    // ── Workflow Management ─────────────────────────────────────────────

    /// Create a new workflow.
    pub fn create_workflow(&mut self, name: &str, steps: Vec<WorkflowStep>) -> Result<String> {
        if self.workflows.len() >= self.config.max_workflows {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum workflows ({}) reached",
                self.config.max_workflows
            )));
        }
        let id = format!("wf-{:04}", self.next_workflow_id);
        self.next_workflow_id += 1;
        let workflow = Workflow {
            id: id.clone(),
            name: name.to_string(),
            steps,
            current_step: 0,
            status: WorkflowStatus::Created,
            created_at: now_secs(),
            completed_at: None,
        };
        self.workflows.insert(id.clone(), workflow);
        self.stats.active_workflows += 1;
        self.record_event(
            EventType::WorkflowCreated,
            None,
            Some(&id),
            serde_json::json!({ "name": name }),
        );
        Ok(id)
    }

    /// Advance a workflow to the next step.
    pub fn advance_workflow(&mut self, workflow_id: &str) -> Result<Option<&WorkflowStep>> {
        // Validate workflow state first
        {
            let wf = self.workflows.get(workflow_id).ok_or_else(|| {
                NeedleError::NotFound(format!("Workflow '{workflow_id}'"))
            })?;
            if wf.status == WorkflowStatus::Completed || wf.status == WorkflowStatus::Failed {
                return Err(NeedleError::InvalidOperation(format!(
                    "Workflow '{workflow_id}' is already {:?}",
                    wf.status
                )));
            }
        }

        // Mutate the workflow
        let (completed_step, is_finished, next_step) = {
            let wf = self.workflows.get_mut(workflow_id).ok_or_else(|| {
                NeedleError::InvalidOperation(format!("Workflow '{workflow_id}' not found"))
            })?;
            wf.status = WorkflowStatus::Running;

            let completed_step = if wf.current_step < wf.steps.len() {
                wf.steps[wf.current_step].status = StepStatus::Completed;
                let step = wf.current_step;
                wf.current_step += 1;
                Some(step)
            } else {
                None
            };

            if wf.current_step >= wf.steps.len() {
                wf.status = WorkflowStatus::Completed;
                wf.completed_at = Some(now_secs());
                (completed_step, true, 0)
            } else {
                let next = wf.current_step;
                wf.steps[next].status = StepStatus::Running;
                (completed_step, false, next)
            }
        };

        // Record events after releasing the mutable borrow
        if let Some(step) = completed_step {
            self.record_event(
                EventType::StepCompleted,
                None,
                Some(workflow_id),
                serde_json::json!({ "step": step }),
            );
        }

        if is_finished {
            self.stats.active_workflows = self.stats.active_workflows.saturating_sub(1);
            self.record_event(
                EventType::WorkflowCompleted,
                None,
                Some(workflow_id),
                serde_json::json!({}),
            );
            return Ok(None);
        }

        self.record_event(
            EventType::StepStarted,
            None,
            Some(workflow_id),
            serde_json::json!({ "step": next_step }),
        );

        let wf = self.workflows.get(workflow_id).ok_or_else(|| {
            NeedleError::InvalidOperation(format!("Workflow '{workflow_id}' not found"))
        })?;
        Ok(Some(&wf.steps[wf.current_step]))
    }

    /// Get workflow info.
    pub fn workflow(&self, workflow_id: &str) -> Option<&Workflow> {
        self.workflows.get(workflow_id)
    }

    /// List all workflows.
    pub fn list_workflows(&self) -> Vec<&Workflow> {
        self.workflows.values().collect()
    }

    // ── Context Window Management ───────────────────────────────────────

    fn check_context_eviction(&mut self, agent_id: &str) {
        let agent = match self.agents.get_mut(agent_id) {
            Some(a) => a,
            None => return,
        };
        let threshold =
            (agent.context_window as f64 * self.config.context_eviction_threshold) as usize;
        if agent.context_used > threshold {
            // Evict oldest thoughts to free up context
            let to_remove = agent.thought_chain.len() / 3;
            if to_remove > 0 {
                agent.thought_chain.drain(..to_remove);
                agent.context_used = agent.context_used / 2;
                self.stats.total_context_evictions += 1;
            }
        }
    }

    // ── Observability ───────────────────────────────────────────────────

    fn record_event(
        &mut self,
        event_type: EventType,
        agent_id: Option<&str>,
        workflow_id: Option<&str>,
        data: serde_json::Value,
    ) {
        if self.events.len() >= self.config.max_events {
            self.events.remove(0);
        }
        self.events.push(WorkflowEvent {
            event_type,
            agent_id: agent_id.map(String::from),
            workflow_id: workflow_id.map(String::from),
            data,
            timestamp: now_secs(),
        });
    }

    /// Get the event log.
    pub fn events(&self) -> &[WorkflowEvent] {
        &self.events
    }

    /// Get events for a specific agent.
    pub fn agent_events(&self, agent_id: &str) -> Vec<&WorkflowEvent> {
        self.events
            .iter()
            .filter(|e| e.agent_id.as_deref() == Some(agent_id))
            .collect()
    }

    /// Get events for a specific workflow.
    pub fn workflow_events(&self, workflow_id: &str) -> Vec<&WorkflowEvent> {
        self.events
            .iter()
            .filter(|e| e.workflow_id.as_deref() == Some(workflow_id))
            .collect()
    }

    /// Get engine stats.
    pub fn stats(&self) -> &EngineStats {
        &self.stats
    }

    /// Get config.
    pub fn config(&self) -> &WorkflowConfig {
        &self.config
    }

    /// Count active agents.
    pub fn agent_count(&self) -> usize {
        self.agents.len()
    }
}

impl Default for WorkflowEngine {
    fn default() -> Self {
        Self::new(WorkflowConfig::default())
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> WorkflowEngine {
        WorkflowEngine::new(WorkflowConfig::default())
    }

    #[test]
    fn test_register_agent() {
        let mut engine = make_engine();
        let id = engine.register_agent("researcher", 4096);
        assert!(id.starts_with("agent-"));
        assert_eq!(engine.agent(&id).unwrap().role, "researcher");
    }

    #[test]
    fn test_agent_status() {
        let mut engine = make_engine();
        let id = engine.register_agent("test", 4096);
        engine.set_agent_status(&id, AgentStatus::Working).unwrap();
        assert_eq!(engine.agent(&id).unwrap().status, AgentStatus::Working);
    }

    #[test]
    fn test_tool_call() {
        let mut engine = make_engine();
        let id = engine.register_agent("test", 4096);
        engine
            .record_tool_call(
                &id,
                "search",
                serde_json::json!({"query": "test"}),
                Some(serde_json::json!({"results": []})),
                50,
                true,
            )
            .unwrap();
        assert_eq!(engine.agent(&id).unwrap().tool_calls.len(), 1);
        assert_eq!(engine.stats().total_tool_calls, 1);
    }

    #[test]
    fn test_thought_chain() {
        let mut engine = make_engine();
        let id = engine.register_agent("test", 4096);
        engine
            .add_thought(&id, "I need to search for vectors", Some("search".into()), None)
            .unwrap();
        engine
            .add_thought(&id, "Results look good", None, Some("Found 5 results".into()))
            .unwrap();
        assert_eq!(engine.agent(&id).unwrap().thought_chain.len(), 2);
    }

    #[test]
    fn test_shared_memory() {
        let mut engine = make_engine();
        let a = engine.register_agent("a", 4096);
        let b = engine.register_agent("b", 4096);

        engine
            .write_shared_memory(&a, "key1", "value1")
            .unwrap();
        let val = engine.read_shared_memory(&b, "key1").unwrap();
        assert_eq!(val, "value1");
    }

    #[test]
    fn test_shared_memory_not_found() {
        let mut engine = make_engine();
        let a = engine.register_agent("a", 4096);
        assert!(engine.read_shared_memory(&a, "nonexistent").is_err());
    }

    #[test]
    fn test_memory_priority() {
        let mut engine = make_engine();
        let a = engine.register_agent("a", 4096);
        engine.write_shared_memory(&a, "key1", "val").unwrap();
        engine.set_memory_priority("key1", 10).unwrap();
        let entries = engine.list_shared_memory();
        assert_eq!(entries[0].priority, 10);
    }

    #[test]
    fn test_workflow_creation() {
        let mut engine = make_engine();
        let agent_id = engine.register_agent("worker", 4096);
        let steps = vec![
            WorkflowStep {
                name: "Step 1".into(),
                agent_id: agent_id.clone(),
                step_type: StepType::Execute {
                    prompt: "Do something".into(),
                },
                status: StepStatus::Pending,
                output: None,
                depends_on: vec![],
            },
            WorkflowStep {
                name: "Step 2".into(),
                agent_id,
                step_type: StepType::Store {
                    key: "result".into(),
                    value: "done".into(),
                },
                status: StepStatus::Pending,
                output: None,
                depends_on: vec![0],
            },
        ];
        let wf_id = engine.create_workflow("test-flow", steps).unwrap();
        assert!(engine.workflow(&wf_id).is_some());
    }

    #[test]
    fn test_workflow_advance() {
        let mut engine = make_engine();
        let agent_id = engine.register_agent("worker", 4096);
        let steps = vec![
            WorkflowStep {
                name: "Step 1".into(),
                agent_id: agent_id.clone(),
                step_type: StepType::Execute {
                    prompt: "Do something".into(),
                },
                status: StepStatus::Pending,
                output: None,
                depends_on: vec![],
            },
            WorkflowStep {
                name: "Step 2".into(),
                agent_id,
                step_type: StepType::Execute {
                    prompt: "Do more".into(),
                },
                status: StepStatus::Pending,
                output: None,
                depends_on: vec![],
            },
        ];
        let wf_id = engine.create_workflow("test-flow", steps).unwrap();

        // Advance to step 1 (completes step 0 which is conceptually "start")
        let step = engine.advance_workflow(&wf_id).unwrap();
        assert!(step.is_some());
        assert_eq!(step.unwrap().name, "Step 2");

        // Advance past last step -> completed
        let step = engine.advance_workflow(&wf_id).unwrap();
        assert!(step.is_none());
        assert_eq!(
            engine.workflow(&wf_id).unwrap().status,
            WorkflowStatus::Completed
        );
    }

    #[test]
    fn test_event_log() {
        let mut engine = make_engine();
        let id = engine.register_agent("test", 4096);
        engine.write_shared_memory(&id, "k", "v").unwrap();

        let events = engine.events();
        assert!(events.len() >= 2); // register + write
        let agent_events = engine.agent_events(&id);
        assert!(!agent_events.is_empty());
    }

    #[test]
    fn test_context_eviction() {
        let mut engine = WorkflowEngine::new(WorkflowConfig {
            context_eviction_threshold: 0.5,
            ..Default::default()
        });
        let id = engine.register_agent("test", 200);
        // Add many thoughts to trigger eviction
        for i in 0..50 {
            let _ = engine.add_thought(
                &id,
                &format!("Thought number {i} with some substantial text content to increase usage"),
                None,
                None,
            );
        }
        // Some thoughts should have been evicted
        assert!(engine.agent(&id).unwrap().thought_chain.len() < 50);
    }

    #[test]
    fn test_invalid_agent_error() {
        let mut engine = make_engine();
        assert!(engine.set_agent_status("fake-agent", AgentStatus::Working).is_err());
        assert!(engine.write_shared_memory("fake-agent", "k", "v").is_err());
    }

    #[test]
    fn test_workflow_events() {
        let mut engine = make_engine();
        let agent_id = engine.register_agent("worker", 4096);
        let steps = vec![WorkflowStep {
            name: "Step 1".into(),
            agent_id,
            step_type: StepType::Execute { prompt: "test".into() },
            status: StepStatus::Pending,
            output: None,
            depends_on: vec![],
        }];
        let wf_id = engine.create_workflow("test", steps).unwrap();
        engine.advance_workflow(&wf_id).unwrap();
        let wf_events = engine.workflow_events(&wf_id);
        assert!(!wf_events.is_empty());
    }

    #[test]
    fn test_list_workflows() {
        let mut engine = make_engine();
        let agent_id = engine.register_agent("worker", 4096);
        let steps = vec![WorkflowStep {
            name: "S".into(),
            agent_id,
            step_type: StepType::Execute { prompt: "x".into() },
            status: StepStatus::Pending,
            output: None,
            depends_on: vec![],
        }];
        engine.create_workflow("wf1", steps.clone()).unwrap();
        engine.create_workflow("wf2", steps).unwrap();
        assert_eq!(engine.list_workflows().len(), 2);
    }
}
