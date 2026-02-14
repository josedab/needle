//! ⚠️ **Experimental**: This module is under active development. APIs may change without notice.
//!
//! Vector Migration and Rebalancing Protocol.
//!
//! This module provides mechanisms for migrating vectors between shards
//! during cluster rebalancing operations.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Rebalance Coordinator                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
//! │  │ Migration    │  │ Transfer     │  │ Progress             │  │
//! │  │ Planner      │  │ Executor     │  │ Tracker              │  │
//! │  └──────────────┘  └──────────────┘  └──────────────────────┘  │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Source Shard ────[vectors]────► Target Shard                   │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use needle::rebalance::{RebalanceCoordinator, RebalanceConfig};
//!
//! let config = RebalanceConfig::default();
//! let coordinator = RebalanceCoordinator::new(config);
//!
//! // Plan migration for adding a new shard
//! let plan = coordinator.plan_add_shard(new_shard_id)?;
//!
//! // Execute migration
//! coordinator.execute(plan).await?;
//! ```

use crate::error::{NeedleError, Result};
use crate::shard::{ShardId, ShardManager};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Configuration for rebalancing operations.
#[derive(Debug, Clone)]
pub struct RebalanceConfig {
    /// Maximum concurrent migrations.
    pub max_concurrent_migrations: usize,
    /// Batch size for vector transfers.
    pub batch_size: usize,
    /// Timeout for individual transfers.
    pub transfer_timeout: Duration,
    /// Retry count for failed transfers.
    pub retry_count: usize,
    /// Delay between retries.
    pub retry_delay: Duration,
    /// Whether to verify transfers.
    pub verify_transfers: bool,
    /// Throttle rate (vectors per second, 0 = unlimited).
    pub throttle_rate: u64,
    /// Checkpoint interval for progress.
    pub checkpoint_interval: Duration,
}

impl Default for RebalanceConfig {
    fn default() -> Self {
        Self {
            max_concurrent_migrations: 4,
            batch_size: 1000,
            transfer_timeout: Duration::from_secs(60),
            retry_count: 3,
            retry_delay: Duration::from_secs(1),
            verify_transfers: true,
            throttle_rate: 0,
            checkpoint_interval: Duration::from_secs(10),
        }
    }
}

impl RebalanceConfig {
    /// Create a new rebalance configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum concurrent migrations.
    pub fn max_concurrent_migrations(mut self, max: usize) -> Self {
        self.max_concurrent_migrations = max;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set transfer timeout.
    pub fn transfer_timeout(mut self, timeout: Duration) -> Self {
        self.transfer_timeout = timeout;
        self
    }

    /// Set throttle rate.
    pub fn throttle_rate(mut self, rate: u64) -> Self {
        self.throttle_rate = rate;
        self
    }

    /// Enable or disable transfer verification.
    pub fn verify_transfers(mut self, verify: bool) -> Self {
        self.verify_transfers = verify;
        self
    }
}

/// A single migration task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationTask {
    /// Unique task ID.
    pub id: String,
    /// Source shard.
    pub source_shard: ShardId,
    /// Target shard.
    pub target_shard: ShardId,
    /// Collection name.
    pub collection: String,
    /// Vector IDs to migrate.
    pub vector_ids: Vec<String>,
    /// Current state.
    pub state: MigrationState,
    /// Number of vectors migrated.
    pub migrated_count: usize,
    /// Number of vectors failed.
    pub failed_count: usize,
    /// Created timestamp.
    pub created_at: u64,
    /// Last updated timestamp.
    pub updated_at: u64,
    /// Error message if failed.
    pub error: Option<String>,
}

impl MigrationTask {
    /// Create a new migration task.
    pub fn new(
        source_shard: ShardId,
        target_shard: ShardId,
        collection: String,
        vector_ids: Vec<String>,
    ) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            id: format!("migration_{}_{}_{}", source_shard.0, target_shard.0, now),
            source_shard,
            target_shard,
            collection,
            vector_ids,
            state: MigrationState::Pending,
            migrated_count: 0,
            failed_count: 0,
            created_at: now,
            updated_at: now,
            error: None,
        }
    }

    /// Get the total number of vectors to migrate.
    pub fn total(&self) -> usize {
        self.vector_ids.len()
    }

    /// Get the progress percentage.
    pub fn progress(&self) -> f64 {
        if self.vector_ids.is_empty() {
            return 100.0;
        }
        (self.migrated_count as f64 / self.vector_ids.len() as f64) * 100.0
    }

    /// Check if the migration is complete.
    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            MigrationState::Completed | MigrationState::Failed
        )
    }
}

/// State of a migration task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationState {
    /// Pending execution.
    Pending,
    /// Currently running.
    Running,
    /// Paused.
    Paused,
    /// Completed successfully.
    Completed,
    /// Failed.
    Failed,
    /// Cancelled.
    Cancelled,
}

/// A rebalance plan consisting of multiple migrations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RebalancePlan {
    /// Unique plan ID.
    pub id: String,
    /// Plan description.
    pub description: String,
    /// Migration tasks.
    pub tasks: Vec<MigrationTask>,
    /// Plan state.
    pub state: PlanState,
    /// Total vectors to migrate.
    pub total_vectors: usize,
    /// Vectors migrated so far.
    pub migrated_vectors: usize,
    /// Created timestamp.
    pub created_at: u64,
    /// Estimated completion time (optional).
    pub estimated_completion: Option<u64>,
}

impl RebalancePlan {
    /// Create a new rebalance plan.
    pub fn new(description: String, tasks: Vec<MigrationTask>) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let total_vectors: usize = tasks.iter().map(|t| t.vector_ids.len()).sum();

        Self {
            id: format!("plan_{}", now),
            description,
            tasks,
            state: PlanState::Created,
            total_vectors,
            migrated_vectors: 0,
            created_at: now,
            estimated_completion: None,
        }
    }

    /// Get overall progress percentage.
    pub fn progress(&self) -> f64 {
        if self.total_vectors == 0 {
            return 100.0;
        }
        (self.migrated_vectors as f64 / self.total_vectors as f64) * 100.0
    }

    /// Get the number of completed tasks.
    pub fn completed_tasks(&self) -> usize {
        self.tasks.iter().filter(|t| t.is_complete()).count()
    }

    /// Check if all tasks are complete.
    pub fn is_complete(&self) -> bool {
        self.tasks.iter().all(|t| t.is_complete())
    }
}

/// State of a rebalance plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlanState {
    /// Plan created but not started.
    Created,
    /// Plan is executing.
    Executing,
    /// Plan is paused.
    Paused,
    /// Plan completed successfully.
    Completed,
    /// Plan failed.
    Failed,
    /// Plan was cancelled.
    Cancelled,
}

/// A vector being transferred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorTransfer {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// Metadata.
    pub metadata: Option<Value>,
}

/// Transfer batch for efficient migration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferBatch {
    /// Batch ID.
    pub batch_id: u64,
    /// Source shard.
    pub source_shard: ShardId,
    /// Target shard.
    pub target_shard: ShardId,
    /// Collection name.
    pub collection: String,
    /// Vectors in this batch.
    pub vectors: Vec<VectorTransfer>,
    /// Checksum for verification.
    pub checksum: u64,
}

impl TransferBatch {
    /// Create a new transfer batch.
    pub fn new(
        batch_id: u64,
        source_shard: ShardId,
        target_shard: ShardId,
        collection: String,
        vectors: Vec<VectorTransfer>,
    ) -> Self {
        let checksum = Self::compute_checksum(&vectors);
        Self {
            batch_id,
            source_shard,
            target_shard,
            collection,
            vectors,
            checksum,
        }
    }

    fn compute_checksum(vectors: &[VectorTransfer]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        vectors.len().hash(&mut hasher);
        for v in vectors {
            v.id.hash(&mut hasher);
            for &f in &v.vector {
                f.to_bits().hash(&mut hasher);
            }
        }
        hasher.finish()
    }

    /// Verify the batch checksum.
    pub fn verify(&self) -> bool {
        Self::compute_checksum(&self.vectors) == self.checksum
    }
}

/// Statistics for rebalancing operations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RebalanceStats {
    /// Total migrations started.
    pub migrations_started: u64,
    /// Total migrations completed.
    pub migrations_completed: u64,
    /// Total migrations failed.
    pub migrations_failed: u64,
    /// Total vectors migrated.
    pub vectors_migrated: u64,
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Average migration time (ms).
    pub avg_migration_time_ms: f64,
    /// Current active migrations.
    pub active_migrations: usize,
    /// Retries performed.
    pub retries: u64,
}

/// Progress checkpoint for resumable migrations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationCheckpoint {
    /// Plan ID.
    pub plan_id: String,
    /// Task ID.
    pub task_id: String,
    /// Last migrated vector ID.
    pub last_vector_id: String,
    /// Migrated count.
    pub migrated_count: usize,
    /// Checkpoint timestamp.
    pub timestamp: u64,
}

/// Trait for migration data source.
pub trait MigrationSource: Send + Sync {
    /// Get vectors by IDs from a shard.
    fn get_vectors(
        &self,
        shard: ShardId,
        collection: &str,
        ids: &[String],
    ) -> Result<Vec<VectorTransfer>>;

    /// Delete vectors from a shard (after successful migration).
    fn delete_vectors(&self, shard: ShardId, collection: &str, ids: &[String]) -> Result<()>;

    /// Get all vector IDs in a shard for a collection.
    fn list_vectors(&self, shard: ShardId, collection: &str) -> Result<Vec<String>>;
}

/// Trait for migration data target.
pub trait MigrationTarget: Send + Sync {
    /// Insert vectors into a shard.
    fn insert_vectors(
        &self,
        shard: ShardId,
        collection: &str,
        vectors: &[VectorTransfer],
    ) -> Result<()>;

    /// Verify vectors exist in target shard.
    fn verify_vectors(&self, shard: ShardId, collection: &str, ids: &[String]) -> Result<bool>;
}

/// Rebalance coordinator for managing migrations.
pub struct RebalanceCoordinator<S, T>
where
    S: MigrationSource,
    T: MigrationTarget,
{
    /// Configuration.
    config: RebalanceConfig,
    /// Migration source.
    source: Arc<S>,
    /// Migration target.
    target: Arc<T>,
    /// Active plans.
    plans: RwLock<HashMap<String, RebalancePlan>>,
    /// Checkpoints for resumption.
    checkpoints: RwLock<HashMap<String, MigrationCheckpoint>>,
    /// Statistics.
    stats: RwLock<RebalanceStats>,
    /// Cancellation flag.
    cancelled: AtomicBool,
    /// Active migration count.
    active_count: AtomicU64,
}

impl<S, T> RebalanceCoordinator<S, T>
where
    S: MigrationSource,
    T: MigrationTarget,
{
    /// Create a new rebalance coordinator.
    pub fn new(config: RebalanceConfig, source: Arc<S>, target: Arc<T>) -> Self {
        Self {
            config,
            source,
            target,
            plans: RwLock::new(HashMap::new()),
            checkpoints: RwLock::new(HashMap::new()),
            stats: RwLock::new(RebalanceStats::default()),
            cancelled: AtomicBool::new(false),
            active_count: AtomicU64::new(0),
        }
    }

    /// Plan migrations for adding a new shard.
    pub fn plan_add_shard(
        &self,
        new_shard: ShardId,
        existing_shards: &[ShardId],
        shard_manager: &ShardManager,
        collection: &str,
    ) -> Result<RebalancePlan> {
        let mut tasks = Vec::new();

        // Calculate which vectors should move to the new shard
        for &source_shard in existing_shards {
            let vector_ids = self.source.list_vectors(source_shard, collection)?;

            // Find vectors that should now belong to the new shard
            let vectors_to_move: Vec<String> = vector_ids
                .into_iter()
                .filter(|id| shard_manager.route_id(id) == new_shard)
                .collect();

            if !vectors_to_move.is_empty() {
                tasks.push(MigrationTask::new(
                    source_shard,
                    new_shard,
                    collection.to_string(),
                    vectors_to_move,
                ));
            }
        }

        let description = format!(
            "Add shard {} - migrating vectors from {} existing shards",
            new_shard.0,
            existing_shards.len()
        );

        Ok(RebalancePlan::new(description, tasks))
    }

    /// Plan migrations for removing a shard.
    pub fn plan_remove_shard(
        &self,
        shard_to_remove: ShardId,
        remaining_shards: &[ShardId],
        shard_manager: &ShardManager,
        collection: &str,
    ) -> Result<RebalancePlan> {
        let vector_ids = self.source.list_vectors(shard_to_remove, collection)?;

        // Group vectors by their new target shard
        let mut vectors_by_target: HashMap<ShardId, Vec<String>> = HashMap::new();

        for id in vector_ids {
            let target = shard_manager.route_id(&id);
            if target != shard_to_remove && remaining_shards.contains(&target) {
                vectors_by_target.entry(target).or_default().push(id);
            }
        }

        let tasks: Vec<MigrationTask> = vectors_by_target
            .into_iter()
            .map(|(target, ids)| {
                MigrationTask::new(shard_to_remove, target, collection.to_string(), ids)
            })
            .collect();

        let description = format!(
            "Remove shard {} - distributing vectors to {} remaining shards",
            shard_to_remove.0,
            remaining_shards.len()
        );

        Ok(RebalancePlan::new(description, tasks))
    }

    /// Plan a full rebalance across all shards.
    pub fn plan_full_rebalance(
        &self,
        shards: &[ShardId],
        shard_manager: &ShardManager,
        collection: &str,
    ) -> Result<RebalancePlan> {
        let mut tasks = Vec::new();

        for &source_shard in shards {
            let vector_ids = self.source.list_vectors(source_shard, collection)?;

            // Group by target shard
            let mut vectors_by_target: HashMap<ShardId, Vec<String>> = HashMap::new();

            for id in vector_ids {
                let target = shard_manager.route_id(&id);
                if target != source_shard {
                    vectors_by_target.entry(target).or_default().push(id);
                }
            }

            for (target, ids) in vectors_by_target {
                if !ids.is_empty() {
                    tasks.push(MigrationTask::new(
                        source_shard,
                        target,
                        collection.to_string(),
                        ids,
                    ));
                }
            }
        }

        let description = format!("Full rebalance across {} shards", shards.len());

        Ok(RebalancePlan::new(description, tasks))
    }

    /// Execute a rebalance plan.
    pub fn execute(&self, mut plan: RebalancePlan) -> Result<()> {
        plan.state = PlanState::Executing;

        // Store the plan
        {
            let mut plans = self.plans.write().map_err(|_| NeedleError::LockError)?;
            plans.insert(plan.id.clone(), plan.clone());
        }

        // Execute tasks
        for task in &mut plan.tasks {
            if self.cancelled.load(Ordering::SeqCst) {
                plan.state = PlanState::Cancelled;
                break;
            }

            self.execute_task(task, &plan.id)?;
            plan.migrated_vectors += task.migrated_count;
        }

        // Update final state
        plan.state = if plan.is_complete() {
            PlanState::Completed
        } else if self.cancelled.load(Ordering::SeqCst) {
            PlanState::Cancelled
        } else {
            PlanState::Failed
        };

        // Update stored plan
        {
            let mut plans = self.plans.write().map_err(|_| NeedleError::LockError)?;
            plans.insert(plan.id.clone(), plan);
        }

        Ok(())
    }

    /// Execute a single migration task.
    fn execute_task(&self, task: &mut MigrationTask, plan_id: &str) -> Result<()> {
        task.state = MigrationState::Running;
        let start_time = Instant::now();

        self.active_count.fetch_add(1, Ordering::SeqCst);

        // Check for existing checkpoint
        let start_index = {
            let checkpoints = self
                .checkpoints
                .read()
                .map_err(|_| NeedleError::LockError)?;
            if let Some(checkpoint) = checkpoints.get(&task.id) {
                checkpoint.migrated_count
            } else {
                0
            }
        };

        let mut last_checkpoint_time = Instant::now();

        // Process vectors in batches
        for (batch_id, chunk) in task.vector_ids[start_index..]
            .chunks(self.config.batch_size)
            .enumerate()
        {
            if self.cancelled.load(Ordering::SeqCst) {
                task.state = MigrationState::Cancelled;
                break;
            }

            // Apply throttling
            if self.config.throttle_rate > 0 {
                let expected_duration =
                    Duration::from_secs_f64(chunk.len() as f64 / self.config.throttle_rate as f64);
                std::thread::sleep(expected_duration);
            }

            // Retry loop
            let mut success = false;
            for retry in 0..=self.config.retry_count {
                if retry > 0 {
                    std::thread::sleep(self.config.retry_delay);
                    let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
                    stats.retries += 1;
                }

                match self.transfer_batch(task, chunk, batch_id as u64) {
                    Ok(bytes) => {
                        task.migrated_count += chunk.len();

                        let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
                        stats.vectors_migrated += chunk.len() as u64;
                        stats.bytes_transferred += bytes;

                        success = true;
                        break;
                    }
                    Err(e) => {
                        if retry == self.config.retry_count {
                            task.failed_count += chunk.len();
                            task.error = Some(e.to_string());
                        }
                    }
                }
            }

            if !success {
                task.state = MigrationState::Failed;
                break;
            }

            // Checkpoint progress
            if last_checkpoint_time.elapsed() >= self.config.checkpoint_interval {
                self.save_checkpoint(plan_id, task)?;
                last_checkpoint_time = Instant::now();
            }
        }

        self.active_count.fetch_sub(1, Ordering::SeqCst);

        // Update task state
        if task.state != MigrationState::Failed && task.state != MigrationState::Cancelled {
            task.state = if task.migrated_count == task.vector_ids.len() {
                MigrationState::Completed
            } else {
                MigrationState::Failed
            };
        }

        // Update timestamp
        task.updated_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        // Update stats
        {
            let mut stats = self.stats.write().map_err(|_| NeedleError::LockError)?;
            match task.state {
                MigrationState::Completed => {
                    stats.migrations_completed += 1;
                    let elapsed = start_time.elapsed().as_millis() as f64;
                    stats.avg_migration_time_ms =
                        (stats.avg_migration_time_ms * stats.migrations_completed as f64 + elapsed)
                            / (stats.migrations_completed + 1) as f64;
                }
                MigrationState::Failed => {
                    stats.migrations_failed += 1;
                }
                _ => {}
            }
        }

        // Clear checkpoint on completion
        if task.state == MigrationState::Completed {
            let mut checkpoints = self
                .checkpoints
                .write()
                .map_err(|_| NeedleError::LockError)?;
            checkpoints.remove(&task.id);
        }

        Ok(())
    }

    /// Transfer a batch of vectors.
    fn transfer_batch(&self, task: &MigrationTask, ids: &[String], batch_id: u64) -> Result<u64> {
        // Get vectors from source
        let vectors = self
            .source
            .get_vectors(task.source_shard, &task.collection, ids)?;

        // Calculate bytes transferred (approximate)
        let bytes: u64 = vectors
            .iter()
            .map(|v| {
                (v.id.len()
                    + v.vector.len() * 4
                    + v.metadata
                        .as_ref()
                        .map(|m| m.to_string().len())
                        .unwrap_or(0)) as u64
            })
            .sum();

        // Create transfer batch
        let batch = TransferBatch::new(
            batch_id,
            task.source_shard,
            task.target_shard,
            task.collection.clone(),
            vectors.clone(),
        );

        // Insert into target
        self.target
            .insert_vectors(task.target_shard, &task.collection, &batch.vectors)?;

        // Verify if configured
        if self.config.verify_transfers {
            let verified = self
                .target
                .verify_vectors(task.target_shard, &task.collection, ids)?;

            if !verified {
                return Err(NeedleError::Corruption(
                    "Transfer verification failed".to_string(),
                ));
            }
        }

        // Delete from source
        self.source
            .delete_vectors(task.source_shard, &task.collection, ids)?;

        Ok(bytes)
    }

    /// Save a checkpoint for a migration task.
    fn save_checkpoint(&self, plan_id: &str, task: &MigrationTask) -> Result<()> {
        let checkpoint = MigrationCheckpoint {
            plan_id: plan_id.to_string(),
            task_id: task.id.clone(),
            last_vector_id: task
                .vector_ids
                .get(task.migrated_count.saturating_sub(1))
                .cloned()
                .unwrap_or_default(),
            migrated_count: task.migrated_count,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
        };

        let mut checkpoints = self
            .checkpoints
            .write()
            .map_err(|_| NeedleError::LockError)?;
        checkpoints.insert(task.id.clone(), checkpoint);

        Ok(())
    }

    /// Cancel ongoing migrations.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Resume from cancelled state.
    pub fn resume(&self) {
        self.cancelled.store(false, Ordering::SeqCst);
    }

    /// Get the status of a plan.
    pub fn get_plan(&self, plan_id: &str) -> Option<RebalancePlan> {
        let plans = self.plans.read().ok()?;
        plans.get(plan_id).cloned()
    }

    /// Get all active plans.
    pub fn list_plans(&self) -> Result<Vec<RebalancePlan>> {
        let plans = self.plans.read().map_err(|_| NeedleError::LockError)?;
        Ok(plans.values().cloned().collect())
    }

    /// Get statistics.
    pub fn stats(&self) -> Result<RebalanceStats> {
        let stats = self.stats.read().map_err(|_| NeedleError::LockError)?;
        let mut result = stats.clone();
        result.active_migrations = self.active_count.load(Ordering::SeqCst) as usize;
        Ok(result)
    }

    /// Pause a running plan.
    pub fn pause_plan(&self, plan_id: &str) -> Result<()> {
        let mut plans = self.plans.write().map_err(|_| NeedleError::LockError)?;
        if let Some(plan) = plans.get_mut(plan_id) {
            if plan.state == PlanState::Executing {
                plan.state = PlanState::Paused;
            }
        }
        Ok(())
    }

    /// Resume a paused plan.
    pub fn resume_plan(&self, plan_id: &str) -> Result<()> {
        let mut plans = self.plans.write().map_err(|_| NeedleError::LockError)?;
        if let Some(plan) = plans.get_mut(plan_id) {
            if plan.state == PlanState::Paused {
                plan.state = PlanState::Executing;
            }
        }
        Ok(())
    }

    /// Get checkpoint for a task.
    pub fn get_checkpoint(&self, task_id: &str) -> Option<MigrationCheckpoint> {
        let checkpoints = self.checkpoints.read().ok()?;
        checkpoints.get(task_id).cloned()
    }

    /// Clean up completed plans.
    pub fn cleanup_completed(&self, max_age: Duration) -> Result<usize> {
        let mut plans = self.plans.write().map_err(|_| NeedleError::LockError)?;
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let cutoff = now.saturating_sub(max_age.as_millis() as u64);

        let to_remove: Vec<String> = plans
            .iter()
            .filter(|(_, plan)| plan.is_complete() && plan.created_at < cutoff)
            .map(|(id, _)| id.clone())
            .collect();

        let count = to_remove.len();
        for id in to_remove {
            plans.remove(&id);
        }

        Ok(count)
    }
}

/// Dry-run migration for testing and validation.
pub struct DryRunMigration {
    /// Operations that would be performed.
    operations: RwLock<Vec<DryRunOperation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DryRunOperation {
    pub operation_type: String,
    pub source_shard: Option<ShardId>,
    pub target_shard: Option<ShardId>,
    pub collection: String,
    pub vector_count: usize,
}

impl Default for DryRunMigration {
    fn default() -> Self {
        Self::new()
    }
}

impl DryRunMigration {
    pub fn new() -> Self {
        Self {
            operations: RwLock::new(Vec::new()),
        }
    }

    pub fn get_operations(&self) -> Vec<DryRunOperation> {
        self.operations
            .read()
            .ok()
            .map(|o| o.clone())
            .unwrap_or_default()
    }
}

impl MigrationSource for DryRunMigration {
    fn get_vectors(
        &self,
        shard: ShardId,
        collection: &str,
        ids: &[String],
    ) -> Result<Vec<VectorTransfer>> {
        let mut ops = self
            .operations
            .write()
            .map_err(|_| NeedleError::LockError)?;
        ops.push(DryRunOperation {
            operation_type: "get_vectors".to_string(),
            source_shard: Some(shard),
            target_shard: None,
            collection: collection.to_string(),
            vector_count: ids.len(),
        });

        Ok(ids
            .iter()
            .map(|id| VectorTransfer {
                id: id.clone(),
                vector: vec![0.0; 128], // Dummy vector
                metadata: None,
            })
            .collect())
    }

    fn delete_vectors(&self, shard: ShardId, collection: &str, ids: &[String]) -> Result<()> {
        let mut ops = self
            .operations
            .write()
            .map_err(|_| NeedleError::LockError)?;
        ops.push(DryRunOperation {
            operation_type: "delete_vectors".to_string(),
            source_shard: Some(shard),
            target_shard: None,
            collection: collection.to_string(),
            vector_count: ids.len(),
        });
        Ok(())
    }

    fn list_vectors(&self, shard: ShardId, collection: &str) -> Result<Vec<String>> {
        let mut ops = self
            .operations
            .write()
            .map_err(|_| NeedleError::LockError)?;
        ops.push(DryRunOperation {
            operation_type: "list_vectors".to_string(),
            source_shard: Some(shard),
            target_shard: None,
            collection: collection.to_string(),
            vector_count: 0,
        });
        Ok(Vec::new())
    }
}

impl MigrationTarget for DryRunMigration {
    fn insert_vectors(
        &self,
        shard: ShardId,
        collection: &str,
        vectors: &[VectorTransfer],
    ) -> Result<()> {
        let mut ops = self
            .operations
            .write()
            .map_err(|_| NeedleError::LockError)?;
        ops.push(DryRunOperation {
            operation_type: "insert_vectors".to_string(),
            source_shard: None,
            target_shard: Some(shard),
            collection: collection.to_string(),
            vector_count: vectors.len(),
        });
        Ok(())
    }

    fn verify_vectors(&self, shard: ShardId, collection: &str, ids: &[String]) -> Result<bool> {
        let mut ops = self
            .operations
            .write()
            .map_err(|_| NeedleError::LockError)?;
        ops.push(DryRunOperation {
            operation_type: "verify_vectors".to_string(),
            source_shard: None,
            target_shard: Some(shard),
            collection: collection.to_string(),
            vector_count: ids.len(),
        });
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_migration_task_creation() {
        let task = MigrationTask::new(
            ShardId(1),
            ShardId(2),
            "test_collection".to_string(),
            vec!["vec1".to_string(), "vec2".to_string()],
        );

        assert_eq!(task.source_shard, ShardId(1));
        assert_eq!(task.target_shard, ShardId(2));
        assert_eq!(task.total(), 2);
        assert_eq!(task.progress(), 0.0);
        assert!(!task.is_complete());
    }

    #[test]
    fn test_migration_task_progress() {
        let mut task = MigrationTask::new(
            ShardId(1),
            ShardId(2),
            "test".to_string(),
            vec![
                "v1".to_string(),
                "v2".to_string(),
                "v3".to_string(),
                "v4".to_string(),
            ],
        );

        task.migrated_count = 2;
        assert_eq!(task.progress(), 50.0);

        task.migrated_count = 4;
        assert_eq!(task.progress(), 100.0);
    }

    #[test]
    fn test_rebalance_plan_creation() {
        let tasks = vec![
            MigrationTask::new(
                ShardId(1),
                ShardId(2),
                "test".to_string(),
                vec!["v1".to_string(), "v2".to_string()],
            ),
            MigrationTask::new(
                ShardId(1),
                ShardId(3),
                "test".to_string(),
                vec!["v3".to_string()],
            ),
        ];

        let plan = RebalancePlan::new("Test plan".to_string(), tasks);

        assert_eq!(plan.total_vectors, 3);
        assert_eq!(plan.progress(), 0.0);
        assert!(!plan.is_complete());
    }

    #[test]
    fn test_transfer_batch_verification() {
        let vectors = vec![
            VectorTransfer {
                id: "v1".to_string(),
                vector: vec![0.1, 0.2, 0.3],
                metadata: None,
            },
            VectorTransfer {
                id: "v2".to_string(),
                vector: vec![0.4, 0.5, 0.6],
                metadata: Some(serde_json::json!({"key": "value"})),
            },
        ];

        let batch = TransferBatch::new(1, ShardId(1), ShardId(2), "test".to_string(), vectors);

        assert!(batch.verify());
    }

    #[test]
    fn test_rebalance_config_builder() {
        let config = RebalanceConfig::new()
            .max_concurrent_migrations(8)
            .batch_size(500)
            .throttle_rate(10000)
            .verify_transfers(false);

        assert_eq!(config.max_concurrent_migrations, 8);
        assert_eq!(config.batch_size, 500);
        assert_eq!(config.throttle_rate, 10000);
        assert!(!config.verify_transfers);
    }

    #[test]
    fn test_dry_run_migration() {
        let dry_run = DryRunMigration::new();

        // Simulate operations
        dry_run.list_vectors(ShardId(1), "test").unwrap();
        dry_run
            .get_vectors(ShardId(1), "test", &["v1".to_string()])
            .unwrap();
        dry_run
            .insert_vectors(
                ShardId(2),
                "test",
                &[VectorTransfer {
                    id: "v1".to_string(),
                    vector: vec![0.1],
                    metadata: None,
                }],
            )
            .unwrap();

        let ops = dry_run.get_operations();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].operation_type, "list_vectors");
        assert_eq!(ops[1].operation_type, "get_vectors");
        assert_eq!(ops[2].operation_type, "insert_vectors");
    }

    #[test]
    fn test_migration_states() {
        assert_eq!(MigrationState::Pending, MigrationState::Pending);
        assert_ne!(MigrationState::Running, MigrationState::Completed);
    }

    #[test]
    fn test_plan_states() {
        assert_eq!(PlanState::Created, PlanState::Created);
        assert_ne!(PlanState::Executing, PlanState::Completed);
    }
}
