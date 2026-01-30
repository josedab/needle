//! Vector Diff & Version Control Service
//!
//! Git-like versioning for vector collections with commit, branch, diff, merge,
//! and rollback operations. Leverages MVCC for point-in-time queries.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::version_control::{
//!     VersionControlService, VcsConfig, VectorChange,
//! };
//!
//! let mut vcs = VersionControlService::new(VcsConfig::default());
//!
//! // Stage changes and commit
//! vcs.stage_insert("doc-1", vec![0.1, 0.2], None);
//! let hash = vcs.commit("Initial vectors", "user@example.com").unwrap();
//!
//! // Create a branch and make changes
//! vcs.create_branch("feature", &hash).unwrap();
//! vcs.checkout("feature").unwrap();
//! vcs.stage_insert("doc-2", vec![0.3, 0.4], None);
//! vcs.commit("Add doc-2", "user@example.com").unwrap();
//!
//! // Diff branches
//! let diff = vcs.diff("main", "feature").unwrap();
//! assert_eq!(diff.inserts, 1);
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::error::{NeedleError, Result};

// ── Commit & Log ────────────────────────────────────────────────────────────

/// A commit hash (SHA-like hex string).
pub type CommitHash = String;

/// A commit in the version history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    /// Unique commit hash.
    pub hash: CommitHash,
    /// Parent commit hash (None for root).
    pub parent: Option<CommitHash>,
    /// Commit message.
    pub message: String,
    /// Author.
    pub author: String,
    /// Timestamp (unix seconds).
    pub timestamp: u64,
    /// Changes in this commit.
    pub changes: Vec<VectorChange>,
    /// Snapshot of vector state at this commit.
    pub snapshot: HashMap<String, VectorSnapshot>,
}

/// A change to a vector in a commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorChange {
    Insert {
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
    Update {
        id: String,
        old_vector: Vec<f32>,
        new_vector: Vec<f32>,
        old_metadata: Option<serde_json::Value>,
        new_metadata: Option<serde_json::Value>,
    },
    Delete {
        id: String,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    },
}

/// Snapshot of a vector at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorSnapshot {
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

// ── Branch ──────────────────────────────────────────────────────────────────

/// A branch pointing to a commit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub name: String,
    pub head: CommitHash,
    pub created_at: u64,
}

// ── Diff ────────────────────────────────────────────────────────────────────

/// Diff between two commits/branches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    /// Base reference (branch/commit).
    pub base: String,
    /// Target reference.
    pub target: String,
    /// Number of inserted vectors.
    pub inserts: usize,
    /// Number of updated vectors.
    pub updates: usize,
    /// Number of deleted vectors.
    pub deletes: usize,
    /// Detailed changes.
    pub changes: Vec<DiffEntry>,
}

/// A single entry in a diff.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffEntry {
    pub vector_id: String,
    pub change_type: DiffChangeType,
    /// Cosine similarity between old and new vectors (for updates).
    pub similarity: Option<f32>,
}

/// Type of diff change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffChangeType {
    Added,
    Modified,
    Deleted,
}

// ── Merge ───────────────────────────────────────────────────────────────────

/// Merge result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// Resulting commit hash.
    pub commit_hash: CommitHash,
    /// Number of vectors merged.
    pub merged_count: usize,
    /// Conflicts encountered.
    pub conflicts: Vec<MergeConflict>,
    /// Whether the merge was clean (no conflicts).
    pub clean: bool,
}

/// A merge conflict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    pub vector_id: String,
    pub base_vector: Option<Vec<f32>>,
    pub ours_vector: Option<Vec<f32>>,
    pub theirs_vector: Option<Vec<f32>>,
    pub resolution: MergeResolution,
}

/// How a merge conflict is resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeResolution {
    /// Keep our version.
    Ours,
    /// Keep their version.
    Theirs,
    /// Average the vectors.
    Average,
    /// Keep both (suffix IDs).
    Both,
}

// ── Version Control Config ──────────────────────────────────────────────────

/// VCS configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VcsConfig {
    /// Maximum commits to retain.
    pub max_commits: usize,
    /// Maximum branches.
    pub max_branches: usize,
    /// Default merge resolution strategy.
    pub merge_strategy: MergeResolution,
    /// Enable automatic snapshots.
    pub auto_snapshot: bool,
}

impl Default for VcsConfig {
    fn default() -> Self {
        Self {
            max_commits: 10_000,
            max_branches: 100,
            merge_strategy: MergeResolution::Theirs,
            auto_snapshot: true,
        }
    }
}

// ── VCS Service ─────────────────────────────────────────────────────────────

/// Version control service for vector collections.
pub struct VersionControlService {
    config: VcsConfig,
    commits: HashMap<CommitHash, Commit>,
    branches: HashMap<String, Branch>,
    current_branch: String,
    staging: Vec<VectorChange>,
    commit_order: Vec<CommitHash>,
    next_hash_idx: u64,
}

impl VersionControlService {
    /// Create a new version control service.
    pub fn new(config: VcsConfig) -> Self {
        let root_hash = "0000000000".to_string();
        let mut commits = HashMap::new();
        commits.insert(
            root_hash.clone(),
            Commit {
                hash: root_hash.clone(),
                parent: None,
                message: "Initial commit".into(),
                author: "system".into(),
                timestamp: now_secs(),
                changes: vec![],
                snapshot: HashMap::new(),
            },
        );
        let mut branches = HashMap::new();
        branches.insert(
            "main".into(),
            Branch {
                name: "main".into(),
                head: root_hash.clone(),
                created_at: now_secs(),
            },
        );
        Self {
            config,
            commits,
            branches,
            current_branch: "main".into(),
            staging: Vec::new(),
            commit_order: vec![root_hash],
            next_hash_idx: 1,
        }
    }

    /// Stage a vector insert.
    pub fn stage_insert(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) {
        self.staging.push(VectorChange::Insert {
            id: id.into(),
            vector,
            metadata,
        });
    }

    /// Stage a vector update.
    pub fn stage_update(
        &mut self,
        id: impl Into<String>,
        old_vector: Vec<f32>,
        new_vector: Vec<f32>,
        old_metadata: Option<serde_json::Value>,
        new_metadata: Option<serde_json::Value>,
    ) {
        self.staging.push(VectorChange::Update {
            id: id.into(),
            old_vector,
            new_vector,
            old_metadata,
            new_metadata,
        });
    }

    /// Stage a vector delete.
    pub fn stage_delete(
        &mut self,
        id: impl Into<String>,
        vector: Vec<f32>,
        metadata: Option<serde_json::Value>,
    ) {
        self.staging.push(VectorChange::Delete {
            id: id.into(),
            vector,
            metadata,
        });
    }

    /// Get the number of staged changes.
    pub fn staged_count(&self) -> usize {
        self.staging.len()
    }

    /// Commit staged changes.
    pub fn commit(&mut self, message: &str, author: &str) -> Result<CommitHash> {
        if self.staging.is_empty() {
            return Err(NeedleError::InvalidOperation(
                "Nothing staged to commit".into(),
            ));
        }
        if self.commits.len() >= self.config.max_commits {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum commits ({}) reached",
                self.config.max_commits
            )));
        }

        let branch = self
            .branches
            .get(&self.current_branch)
            .ok_or_else(|| NeedleError::NotFound(format!("Branch '{}'", self.current_branch)))?;
        let parent_hash = branch.head.clone();
        let parent_snapshot = self
            .commits
            .get(&parent_hash)
            .map(|c| c.snapshot.clone())
            .unwrap_or_default();

        // Build new snapshot by applying changes
        let mut snapshot = parent_snapshot;
        for change in &self.staging {
            match change {
                VectorChange::Insert {
                    id,
                    vector,
                    metadata,
                } => {
                    snapshot.insert(
                        id.clone(),
                        VectorSnapshot {
                            vector: vector.clone(),
                            metadata: metadata.clone(),
                        },
                    );
                }
                VectorChange::Update {
                    id,
                    new_vector,
                    new_metadata,
                    ..
                } => {
                    snapshot.insert(
                        id.clone(),
                        VectorSnapshot {
                            vector: new_vector.clone(),
                            metadata: new_metadata.clone(),
                        },
                    );
                }
                VectorChange::Delete { id, .. } => {
                    snapshot.remove(id);
                }
            }
        }

        let hash = self.generate_hash();
        let commit = Commit {
            hash: hash.clone(),
            parent: Some(parent_hash),
            message: message.to_string(),
            author: author.to_string(),
            timestamp: now_secs(),
            changes: std::mem::take(&mut self.staging),
            snapshot,
        };

        self.commits.insert(hash.clone(), commit);
        self.commit_order.push(hash.clone());

        // Update branch head
        if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = hash.clone();
        }

        Ok(hash)
    }

    /// Create a new branch at the given commit.
    pub fn create_branch(&mut self, name: &str, at_commit: &CommitHash) -> Result<()> {
        if self.branches.len() >= self.config.max_branches {
            return Err(NeedleError::CapacityExceeded(format!(
                "Maximum branches ({}) reached",
                self.config.max_branches
            )));
        }
        if self.branches.contains_key(name) {
            return Err(NeedleError::Conflict(format!("Branch '{name}' already exists")));
        }
        if !self.commits.contains_key(at_commit) {
            return Err(NeedleError::NotFound(format!("Commit '{at_commit}'")));
        }
        self.branches.insert(
            name.to_string(),
            Branch {
                name: name.to_string(),
                head: at_commit.clone(),
                created_at: now_secs(),
            },
        );
        Ok(())
    }

    /// Switch to a branch.
    pub fn checkout(&mut self, branch_name: &str) -> Result<()> {
        if !self.branches.contains_key(branch_name) {
            return Err(NeedleError::NotFound(format!("Branch '{branch_name}'")));
        }
        if !self.staging.is_empty() {
            return Err(NeedleError::InvalidOperation(
                "Cannot checkout with uncommitted changes".into(),
            ));
        }
        self.current_branch = branch_name.to_string();
        Ok(())
    }

    /// Delete a branch.
    pub fn delete_branch(&mut self, name: &str) -> Result<()> {
        if name == "main" {
            return Err(NeedleError::InvalidOperation(
                "Cannot delete 'main' branch".into(),
            ));
        }
        if name == self.current_branch {
            return Err(NeedleError::InvalidOperation(
                "Cannot delete current branch".into(),
            ));
        }
        self.branches
            .remove(name)
            .ok_or_else(|| NeedleError::NotFound(format!("Branch '{name}'")))?;
        Ok(())
    }

    /// List all branches.
    pub fn list_branches(&self) -> Vec<&Branch> {
        self.branches.values().collect()
    }

    /// Get current branch name.
    pub fn current_branch(&self) -> &str {
        &self.current_branch
    }

    /// Get a commit by hash.
    pub fn get_commit(&self, hash: &CommitHash) -> Option<&Commit> {
        self.commits.get(hash)
    }

    /// Get the commit log for the current branch.
    pub fn log(&self, limit: usize) -> Vec<&Commit> {
        let mut result = Vec::new();
        let branch = match self.branches.get(&self.current_branch) {
            Some(b) => b,
            None => return result,
        };
        let mut current = Some(branch.head.clone());
        while let Some(hash) = current {
            if result.len() >= limit {
                break;
            }
            if let Some(commit) = self.commits.get(&hash) {
                result.push(commit);
                current = commit.parent.clone();
            } else {
                break;
            }
        }
        result
    }

    /// Diff between two branches or commits.
    pub fn diff(&self, base: &str, target: &str) -> Result<DiffResult> {
        let base_snapshot = self.resolve_snapshot(base)?;
        let target_snapshot = self.resolve_snapshot(target)?;

        let mut changes = Vec::new();
        let mut inserts = 0;
        let mut updates = 0;
        let mut deletes = 0;

        // Find inserts and updates in target
        for (id, target_vec) in &target_snapshot {
            if let Some(base_vec) = base_snapshot.get(id) {
                if base_vec.vector != target_vec.vector {
                    let sim = cosine_similarity(&base_vec.vector, &target_vec.vector);
                    changes.push(DiffEntry {
                        vector_id: id.clone(),
                        change_type: DiffChangeType::Modified,
                        similarity: Some(sim),
                    });
                    updates += 1;
                }
            } else {
                changes.push(DiffEntry {
                    vector_id: id.clone(),
                    change_type: DiffChangeType::Added,
                    similarity: None,
                });
                inserts += 1;
            }
        }

        // Find deletes
        for id in base_snapshot.keys() {
            if !target_snapshot.contains_key(id) {
                changes.push(DiffEntry {
                    vector_id: id.clone(),
                    change_type: DiffChangeType::Deleted,
                    similarity: None,
                });
                deletes += 1;
            }
        }

        Ok(DiffResult {
            base: base.into(),
            target: target.into(),
            inserts,
            updates,
            deletes,
            changes,
        })
    }

    /// Merge a source branch into the current branch.
    pub fn merge(&mut self, source_branch: &str, author: &str) -> Result<MergeResult> {
        if source_branch == self.current_branch {
            return Err(NeedleError::InvalidOperation(
                "Cannot merge a branch into itself".into(),
            ));
        }
        let source_head = self
            .branches
            .get(source_branch)
            .ok_or_else(|| NeedleError::NotFound(format!("Branch '{source_branch}'")))?
            .head
            .clone();
        let target_head = self
            .branches
            .get(&self.current_branch)
            .ok_or_else(|| NeedleError::NotFound(format!("Branch '{}'", self.current_branch)))?
            .head
            .clone();

        let source_snapshot = self
            .commits
            .get(&source_head)
            .map(|c| c.snapshot.clone())
            .unwrap_or_default();
        let target_snapshot = self
            .commits
            .get(&target_head)
            .map(|c| c.snapshot.clone())
            .unwrap_or_default();

        let mut merged_snapshot = target_snapshot.clone();
        let mut conflicts = Vec::new();
        let mut merged_count = 0;

        for (id, source_vec) in &source_snapshot {
            if let Some(target_vec) = target_snapshot.get(id) {
                if source_vec.vector != target_vec.vector {
                    // Conflict: both branches modified the same vector
                    let resolution = self.config.merge_strategy;
                    match resolution {
                        MergeResolution::Ours => {} // keep target
                        MergeResolution::Theirs => {
                            merged_snapshot.insert(id.clone(), source_vec.clone());
                        }
                        MergeResolution::Average => {
                            let avg: Vec<f32> = source_vec
                                .vector
                                .iter()
                                .zip(target_vec.vector.iter())
                                .map(|(a, b)| (a + b) / 2.0)
                                .collect();
                            merged_snapshot.insert(
                                id.clone(),
                                VectorSnapshot {
                                    vector: avg,
                                    metadata: source_vec.metadata.clone(),
                                },
                            );
                        }
                        MergeResolution::Both => {
                            let dup_id = format!("{id}_merged");
                            merged_snapshot.insert(dup_id, source_vec.clone());
                        }
                    }
                    conflicts.push(MergeConflict {
                        vector_id: id.clone(),
                        base_vector: None,
                        ours_vector: Some(target_vec.vector.clone()),
                        theirs_vector: Some(source_vec.vector.clone()),
                        resolution,
                    });
                }
            } else {
                // New vector from source
                merged_snapshot.insert(id.clone(), source_vec.clone());
                merged_count += 1;
            }
        }

        // Create merge commit
        let hash = self.generate_hash();
        let commit = Commit {
            hash: hash.clone(),
            parent: Some(target_head),
            message: format!("Merge branch '{source_branch}' into {}", self.current_branch),
            author: author.to_string(),
            timestamp: now_secs(),
            changes: vec![],
            snapshot: merged_snapshot,
        };
        self.commits.insert(hash.clone(), commit);
        self.commit_order.push(hash.clone());

        if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = hash.clone();
        }

        let clean = conflicts.is_empty();
        Ok(MergeResult {
            commit_hash: hash,
            merged_count,
            conflicts,
            clean,
        })
    }

    /// Rollback to a specific commit.
    pub fn rollback(&mut self, commit_hash: &CommitHash, author: &str) -> Result<CommitHash> {
        let target = self
            .commits
            .get(commit_hash)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{commit_hash}'")))?;
        let snapshot = target.snapshot.clone();

        // Create a rollback commit
        let hash = self.generate_hash();
        let branch_head = self
            .branches
            .get(&self.current_branch)
            .map(|b| b.head.clone())
            .unwrap_or_default();
        let commit = Commit {
            hash: hash.clone(),
            parent: Some(branch_head),
            message: format!("Rollback to {commit_hash}"),
            author: author.to_string(),
            timestamp: now_secs(),
            changes: vec![],
            snapshot,
        };
        self.commits.insert(hash.clone(), commit);
        self.commit_order.push(hash.clone());

        if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = hash.clone();
        }

        Ok(hash)
    }

    /// Get the snapshot at the head of the current branch.
    pub fn head_snapshot(&self) -> HashMap<String, VectorSnapshot> {
        self.branches
            .get(&self.current_branch)
            .and_then(|b| self.commits.get(&b.head))
            .map(|c| c.snapshot.clone())
            .unwrap_or_default()
    }

    /// Get total commit count.
    pub fn commit_count(&self) -> usize {
        self.commits.len()
    }

    /// Get total branch count.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Get config.
    pub fn config(&self) -> &VcsConfig {
        &self.config
    }

    fn resolve_snapshot(&self, ref_name: &str) -> Result<HashMap<String, VectorSnapshot>> {
        // Try as branch first, then as commit hash
        if let Some(branch) = self.branches.get(ref_name) {
            return self
                .commits
                .get(&branch.head)
                .map(|c| c.snapshot.clone())
                .ok_or_else(|| NeedleError::NotFound(format!("Commit for branch '{ref_name}'")));
        }
        self.commits
            .get(ref_name)
            .map(|c| c.snapshot.clone())
            .ok_or_else(|| NeedleError::NotFound(format!("Ref '{ref_name}'")))
    }

    fn generate_hash(&mut self) -> CommitHash {
        let hash = format!("{:010x}", self.next_hash_idx);
        self.next_hash_idx += 1;
        hash
    }
}

impl Default for VersionControlService {
    fn default() -> Self {
        Self::new(VcsConfig::default())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
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

    fn make_vcs() -> VersionControlService {
        VersionControlService::new(VcsConfig::default())
    }

    #[test]
    fn test_commit() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1, 0.2], None);
        let hash = vcs.commit("Add doc-1", "test@test.com").unwrap();
        assert!(!hash.is_empty());
        assert_eq!(vcs.commit_count(), 2); // root + this
    }

    #[test]
    fn test_empty_commit_error() {
        let mut vcs = make_vcs();
        assert!(vcs.commit("empty", "test").is_err());
    }

    #[test]
    fn test_staging() {
        let mut vcs = make_vcs();
        assert_eq!(vcs.staged_count(), 0);
        vcs.stage_insert("doc-1", vec![0.1], None);
        assert_eq!(vcs.staged_count(), 1);
    }

    #[test]
    fn test_branch_create_checkout() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let hash = vcs.commit("first", "test").unwrap();

        vcs.create_branch("feature", &hash).unwrap();
        vcs.checkout("feature").unwrap();
        assert_eq!(vcs.current_branch(), "feature");
    }

    #[test]
    fn test_branch_delete() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let hash = vcs.commit("first", "test").unwrap();
        vcs.create_branch("feature", &hash).unwrap();
        vcs.delete_branch("feature").unwrap();
        assert_eq!(vcs.branch_count(), 1);
    }

    #[test]
    fn test_cannot_delete_main() {
        let mut vcs = make_vcs();
        assert!(vcs.delete_branch("main").is_err());
    }

    #[test]
    fn test_log() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        vcs.commit("first", "test").unwrap();
        vcs.stage_insert("doc-2", vec![0.2], None);
        vcs.commit("second", "test").unwrap();

        let log = vcs.log(10);
        assert_eq!(log.len(), 3); // root + 2
        assert_eq!(log[0].message, "second");
    }

    #[test]
    fn test_diff() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1, 0.2], None);
        let h1 = vcs.commit("first", "test").unwrap();

        vcs.create_branch("feature", &h1).unwrap();
        vcs.checkout("feature").unwrap();
        vcs.stage_insert("doc-2", vec![0.3, 0.4], None);
        vcs.commit("add doc-2", "test").unwrap();

        let diff = vcs.diff("main", "feature").unwrap();
        assert_eq!(diff.inserts, 1);
        assert_eq!(diff.deletes, 0);
    }

    #[test]
    fn test_diff_with_update() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1, 0.2], None);
        let h1 = vcs.commit("first", "test").unwrap();

        vcs.create_branch("feature", &h1).unwrap();
        vcs.checkout("feature").unwrap();
        vcs.stage_update("doc-1", vec![0.1, 0.2], vec![0.5, 0.6], None, None);
        vcs.commit("update doc-1", "test").unwrap();

        let diff = vcs.diff("main", "feature").unwrap();
        assert_eq!(diff.updates, 1);
        assert!(diff.changes[0].similarity.is_some());
    }

    #[test]
    fn test_merge_clean() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let h1 = vcs.commit("first", "test").unwrap();

        vcs.create_branch("feature", &h1).unwrap();
        vcs.checkout("feature").unwrap();
        vcs.stage_insert("doc-2", vec![0.2], None);
        vcs.commit("add doc-2 on feature", "test").unwrap();

        vcs.checkout("main").unwrap();
        let result = vcs.merge("feature", "test").unwrap();
        assert!(result.clean);
        assert_eq!(result.merged_count, 1);

        let snapshot = vcs.head_snapshot();
        assert!(snapshot.contains_key("doc-1"));
        assert!(snapshot.contains_key("doc-2"));
    }

    #[test]
    fn test_merge_conflict() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let h1 = vcs.commit("first", "test").unwrap();

        // Branch A: modify doc-1
        vcs.create_branch("feature", &h1).unwrap();
        vcs.checkout("feature").unwrap();
        vcs.stage_update("doc-1", vec![0.1], vec![0.5], None, None);
        vcs.commit("update on feature", "test").unwrap();

        // Main: modify doc-1 differently
        vcs.checkout("main").unwrap();
        vcs.stage_update("doc-1", vec![0.1], vec![0.9], None, None);
        vcs.commit("update on main", "test").unwrap();

        let result = vcs.merge("feature", "test").unwrap();
        assert!(!result.clean);
        assert_eq!(result.conflicts.len(), 1);
    }

    #[test]
    fn test_rollback() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let h1 = vcs.commit("first", "test").unwrap();
        vcs.stage_insert("doc-2", vec![0.2], None);
        vcs.commit("second", "test").unwrap();

        vcs.rollback(&h1, "test").unwrap();
        let snapshot = vcs.head_snapshot();
        assert!(snapshot.contains_key("doc-1"));
        assert!(!snapshot.contains_key("doc-2"));
    }

    #[test]
    fn test_merge_self_error() {
        let mut vcs = make_vcs();
        assert!(vcs.merge("main", "test").is_err());
    }

    #[test]
    fn test_checkout_with_staged_error() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let h = vcs.commit("first", "test").unwrap();
        vcs.create_branch("feature", &h).unwrap();
        vcs.stage_insert("doc-2", vec![0.2], None);
        assert!(vcs.checkout("feature").is_err());
    }

    #[test]
    fn test_duplicate_branch_error() {
        let mut vcs = make_vcs();
        vcs.stage_insert("doc-1", vec![0.1], None);
        let h = vcs.commit("first", "test").unwrap();
        vcs.create_branch("feature", &h).unwrap();
        assert!(vcs.create_branch("feature", &h).is_err());
    }

    #[test]
    fn test_cosine_similarity() {
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 0.001);

        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 0.001);
    }
}
