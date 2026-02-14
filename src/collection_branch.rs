//! Collection Branching & Merge
//!
//! Git-like branching for vector collections using copy-on-write semantics.
//! Enables experimentation, A/B testing, and staging workflows.
//!
//! # Architecture
//!
//! ```text
//! main ────────────────────────────────►
//!          ╲
//!           ╲── experiment/v2 ──────────►
//!                      ╲
//!                       ╲── ab-test ────►
//! ```
//!
//! Each branch maintains a set of changes (inserts, updates, deletes) over a
//! parent branch. Reads check the branch's local changes first, then fall back
//! to the parent chain. Merges detect conflicts when the same vector ID was
//! modified in both branches.
//!
//! # Example
//!
//! ```rust
//! use needle::collection_branch::*;
//!
//! let mut tree = BranchTree::new();
//!
//! // Insert on main
//! tree.insert("main", "doc1", vec![1.0, 2.0]).unwrap();
//!
//! // Create a branch
//! tree.create_branch("experiment", "main").unwrap();
//!
//! // Modify on the branch without affecting main
//! tree.insert("experiment", "doc1", vec![3.0, 4.0]).unwrap();
//! tree.insert("experiment", "doc2", vec![5.0, 6.0]).unwrap();
//!
//! // Main still has original
//! assert_eq!(tree.get("main", "doc1").unwrap(), &[1.0, 2.0]);
//! // Branch has override
//! assert_eq!(tree.get("experiment", "doc1").unwrap(), &[3.0, 4.0]);
//!
//! // Merge back
//! let result = tree.merge("experiment", "main", MergeStrategy::SourceWins).unwrap();
//! assert_eq!(result.merged, 2);
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// Branch Metadata
// ============================================================================

/// Metadata for a branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchInfo {
    /// Branch name.
    pub name: String,
    /// Parent branch name (None for root/main).
    pub parent: Option<String>,
    /// Creation timestamp.
    pub created_at: u64,
    /// Number of local changes.
    pub change_count: usize,
    /// Whether this branch is read-only (frozen).
    pub frozen: bool,
}

/// A change to a vector in a branch.
#[derive(Debug, Clone)]
enum VectorChange {
    /// Vector was inserted or updated.
    Upsert(Vec<f32>),
    /// Vector was deleted.
    Delete,
}

/// A branch's local changes (copy-on-write layer).
struct BranchLayer {
    info: BranchInfo,
    changes: HashMap<String, VectorChange>,
}

// ============================================================================
// Merge Types
// ============================================================================

/// Strategy for resolving merge conflicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Source branch wins on conflict.
    SourceWins,
    /// Target branch wins on conflict.
    TargetWins,
    /// Skip conflicting vectors (don't merge them).
    Skip,
}

/// A conflict detected during merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeConflict {
    /// Vector ID.
    pub vector_id: String,
    /// Description of the conflict.
    pub description: String,
}

/// Result of a merge operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MergeResult {
    /// Number of vectors merged (applied to target).
    pub merged: usize,
    /// Number of conflicts detected.
    pub conflicts: usize,
    /// Number of conflicts skipped.
    pub skipped: usize,
    /// Detailed conflict list.
    pub conflict_details: Vec<MergeConflict>,
}

// ============================================================================
// Diff Types
// ============================================================================

/// Difference between two branches for a single vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DiffEntry {
    /// Vector exists only in the source branch.
    Added { id: String },
    /// Vector was deleted in the source branch.
    Deleted { id: String },
    /// Vector was modified in the source branch.
    Modified { id: String },
}

// ============================================================================
// Branch Tree
// ============================================================================

/// Manages a tree of branches with copy-on-write vector storage.
pub struct BranchTree {
    branches: HashMap<String, BranchLayer>,
}

impl BranchTree {
    /// Create a new branch tree with a default "main" branch.
    pub fn new() -> Self {
        let now = Self::now();
        let mut branches = HashMap::new();
        branches.insert(
            "main".to_string(),
            BranchLayer {
                info: BranchInfo {
                    name: "main".to_string(),
                    parent: None,
                    created_at: now,
                    change_count: 0,
                    frozen: false,
                },
                changes: HashMap::new(),
            },
        );
        Self { branches }
    }

    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Create a new branch from a parent branch.
    pub fn create_branch(&mut self, name: &str, parent: &str) -> Result<()> {
        if self.branches.contains_key(name) {
            return Err(NeedleError::InvalidInput(format!(
                "Branch '{}' already exists",
                name
            )));
        }
        if !self.branches.contains_key(parent) {
            return Err(NeedleError::InvalidInput(format!(
                "Parent branch '{}' not found",
                parent
            )));
        }
        self.branches.insert(
            name.to_string(),
            BranchLayer {
                info: BranchInfo {
                    name: name.to_string(),
                    parent: Some(parent.to_string()),
                    created_at: Self::now(),
                    change_count: 0,
                    frozen: false,
                },
                changes: HashMap::new(),
            },
        );
        Ok(())
    }

    /// Delete a branch (cannot delete "main").
    pub fn delete_branch(&mut self, name: &str) -> Result<()> {
        if name == "main" {
            return Err(NeedleError::InvalidInput(
                "Cannot delete the main branch".to_string(),
            ));
        }
        // Check no other branches depend on this one
        let has_children = self
            .branches
            .values()
            .any(|b| b.info.parent.as_deref() == Some(name));
        if has_children {
            return Err(NeedleError::InvalidInput(format!(
                "Branch '{}' has child branches; delete them first",
                name
            )));
        }
        self.branches.remove(name).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Branch '{}' not found", name))
        })?;
        Ok(())
    }

    /// List all branches.
    pub fn list_branches(&self) -> Vec<BranchInfo> {
        self.branches.values().map(|b| b.info.clone()).collect()
    }

    /// Insert or update a vector on a branch.
    pub fn insert(&mut self, branch: &str, id: &str, vector: Vec<f32>) -> Result<()> {
        let layer = self.branches.get_mut(branch).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Branch '{}' not found", branch))
        })?;
        if layer.info.frozen {
            return Err(NeedleError::InvalidInput(format!(
                "Branch '{}' is frozen",
                branch
            )));
        }
        layer
            .changes
            .insert(id.to_string(), VectorChange::Upsert(vector));
        layer.info.change_count = layer.changes.len();
        Ok(())
    }

    /// Delete a vector on a branch.
    pub fn delete(&mut self, branch: &str, id: &str) -> Result<()> {
        let layer = self.branches.get_mut(branch).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Branch '{}' not found", branch))
        })?;
        if layer.info.frozen {
            return Err(NeedleError::InvalidInput(format!(
                "Branch '{}' is frozen",
                branch
            )));
        }
        layer
            .changes
            .insert(id.to_string(), VectorChange::Delete);
        layer.info.change_count = layer.changes.len();
        Ok(())
    }

    /// Get a vector from a branch (walks parent chain).
    pub fn get(&self, branch: &str, id: &str) -> Result<&[f32]> {
        let mut current = branch;
        loop {
            let layer = self.branches.get(current).ok_or_else(|| {
                NeedleError::InvalidInput(format!("Branch '{}' not found", current))
            })?;
            match layer.changes.get(id) {
                Some(VectorChange::Upsert(vec)) => return Ok(vec),
                Some(VectorChange::Delete) => {
                    return Err(NeedleError::NotFound(format!(
                        "Vector '{}' deleted in branch '{}'",
                        id, current
                    )));
                }
                None => {
                    // Walk to parent
                    match &layer.info.parent {
                        Some(parent) => current = parent,
                        None => {
                            return Err(NeedleError::NotFound(format!(
                                "Vector '{}' not found in branch chain",
                                id
                            )));
                        }
                    }
                }
            }
        }
    }

    /// Check if a vector exists on a branch (walks parent chain).
    pub fn contains(&self, branch: &str, id: &str) -> bool {
        self.get(branch, id).is_ok()
    }

    /// List all vector IDs visible on a branch (walks parent chain).
    pub fn list_ids(&self, branch: &str) -> Result<Vec<String>> {
        let mut visible: HashSet<String> = HashSet::new();
        let mut deleted: HashSet<String> = HashSet::new();
        let mut current = branch;
        loop {
            let layer = self.branches.get(current).ok_or_else(|| {
                NeedleError::InvalidInput(format!("Branch '{}' not found", current))
            })?;
            for (id, change) in &layer.changes {
                if deleted.contains(id) || visible.contains(id) {
                    continue;
                }
                match change {
                    VectorChange::Upsert(_) => {
                        visible.insert(id.clone());
                    }
                    VectorChange::Delete => {
                        deleted.insert(id.clone());
                    }
                }
            }
            match &layer.info.parent {
                Some(parent) => current = parent,
                None => break,
            }
        }
        let mut ids: Vec<String> = visible.into_iter().collect();
        ids.sort();
        Ok(ids)
    }

    /// Compute diff: changes in source relative to target.
    pub fn diff(&self, source: &str, target: &str) -> Result<Vec<DiffEntry>> {
        let source_layer = self.branches.get(source).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Branch '{}' not found", source))
        })?;
        let mut diffs = Vec::new();

        for (id, change) in &source_layer.changes {
            let exists_in_target = self.contains(target, id);
            match change {
                VectorChange::Upsert(_) => {
                    if exists_in_target {
                        diffs.push(DiffEntry::Modified { id: id.clone() });
                    } else {
                        diffs.push(DiffEntry::Added { id: id.clone() });
                    }
                }
                VectorChange::Delete => {
                    if exists_in_target {
                        diffs.push(DiffEntry::Deleted { id: id.clone() });
                    }
                }
            }
        }

        diffs.sort_by(|a, b| {
            let id_a = match a {
                DiffEntry::Added { id } | DiffEntry::Deleted { id } | DiffEntry::Modified { id } => id,
            };
            let id_b = match b {
                DiffEntry::Added { id } | DiffEntry::Deleted { id } | DiffEntry::Modified { id } => id,
            };
            id_a.cmp(id_b)
        });
        Ok(diffs)
    }

    /// Merge source branch into target branch.
    pub fn merge(
        &mut self,
        source: &str,
        target: &str,
        strategy: MergeStrategy,
    ) -> Result<MergeResult> {
        if source == target {
            return Err(NeedleError::InvalidInput(
                "Cannot merge a branch into itself".to_string(),
            ));
        }
        if !self.branches.contains_key(source) {
            return Err(NeedleError::InvalidInput(format!(
                "Source branch '{}' not found",
                source
            )));
        }
        if !self.branches.contains_key(target) {
            return Err(NeedleError::InvalidInput(format!(
                "Target branch '{}' not found",
                target
            )));
        }

        // Collect changes to merge
        let source_changes: Vec<(String, VectorChange)> = self
            .branches
            .get(source)
            .map(|l| {
                l.changes
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect()
            })
            .unwrap_or_default();

        // Detect conflicts: IDs that have local changes in both branches
        let target_changed_ids: HashSet<String> = self
            .branches
            .get(target)
            .map(|l| l.changes.keys().cloned().collect())
            .unwrap_or_default();

        let mut result = MergeResult {
            merged: 0,
            conflicts: 0,
            skipped: 0,
            conflict_details: Vec::new(),
        };

        let mut to_apply: Vec<(String, VectorChange)> = Vec::new();

        for (id, change) in &source_changes {
            if target_changed_ids.contains(id) {
                // Conflict
                result.conflicts += 1;
                result.conflict_details.push(MergeConflict {
                    vector_id: id.clone(),
                    description: "Modified in both branches".to_string(),
                });
                match strategy {
                    MergeStrategy::SourceWins => {
                        to_apply.push((id.clone(), change.clone()));
                    }
                    MergeStrategy::TargetWins => {
                        result.skipped += 1;
                    }
                    MergeStrategy::Skip => {
                        result.skipped += 1;
                    }
                }
            } else {
                to_apply.push((id.clone(), change.clone()));
            }
        }

        // Apply
        if let Some(target_layer) = self.branches.get_mut(target) {
            for (id, change) in to_apply {
                target_layer.changes.insert(id, change);
                result.merged += 1;
            }
            target_layer.info.change_count = target_layer.changes.len();
        }

        Ok(result)
    }

    /// Freeze a branch (make read-only).
    pub fn freeze(&mut self, branch: &str) -> Result<()> {
        let layer = self.branches.get_mut(branch).ok_or_else(|| {
            NeedleError::InvalidInput(format!("Branch '{}' not found", branch))
        })?;
        layer.info.frozen = true;
        Ok(())
    }

    /// Get branch info.
    pub fn branch_info(&self, name: &str) -> Option<&BranchInfo> {
        self.branches.get(name).map(|l| &l.info)
    }
}

impl Default for BranchTree {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_list_branches() {
        let mut tree = BranchTree::new();
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main' — 'main' should exist by default");
        tree.create_branch("feature", "dev")
            .expect("failed to create 'feature' branch from 'dev' — 'dev' was just created");

        let branches = tree.list_branches();
        assert_eq!(branches.len(), 3);
    }

    #[test]
    fn test_duplicate_branch() {
        let mut tree = BranchTree::new();
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        assert!(tree.create_branch("dev", "main").is_err());
    }

    #[test]
    fn test_insert_and_get_with_cow() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0, 2.0])
            .expect("failed to insert 'd1' into 'main' branch");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");

        // Dev inherits from main
        assert_eq!(
            tree.get("dev", "d1")
                .expect("failed to get 'd1' from 'dev' — should inherit from 'main'"),
            &[1.0, 2.0]
        );

        // Override on dev
        tree.insert("dev", "d1", vec![3.0, 4.0])
            .expect("failed to insert override for 'd1' on 'dev' branch");
        assert_eq!(
            tree.get("dev", "d1")
                .expect("failed to get 'd1' from 'dev' after override"),
            &[3.0, 4.0]
        );

        // Main unchanged
        assert_eq!(
            tree.get("main", "d1")
                .expect("failed to get 'd1' from 'main' — should be unchanged after dev override"),
            &[1.0, 2.0]
        );
    }

    #[test]
    fn test_delete_on_branch() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0])
            .expect("failed to insert 'd1' into 'main'");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.delete("dev", "d1")
            .expect("failed to delete 'd1' from 'dev' branch");

        // Deleted on dev
        assert!(tree.get("dev", "d1").is_err());
        // Still on main
        assert!(tree.get("main", "d1").is_ok());
    }

    #[test]
    fn test_list_ids() {
        let mut tree = BranchTree::new();
        tree.insert("main", "a", vec![1.0])
            .expect("failed to insert 'a' into 'main'");
        tree.insert("main", "b", vec![2.0])
            .expect("failed to insert 'b' into 'main'");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.insert("dev", "c", vec![3.0])
            .expect("failed to insert 'c' into 'dev'");
        tree.delete("dev", "a")
            .expect("failed to delete 'a' from 'dev'");

        let ids = tree.list_ids("dev")
            .expect("failed to list IDs for 'dev' branch");
        assert_eq!(ids, vec!["b", "c"]);

        let main_ids = tree.list_ids("main")
            .expect("failed to list IDs for 'main' branch");
        assert_eq!(main_ids, vec!["a", "b"]);
    }

    #[test]
    fn test_diff() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0])
            .expect("failed to insert 'd1' into 'main'");
        tree.insert("main", "d2", vec![2.0])
            .expect("failed to insert 'd2' into 'main'");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.insert("dev", "d1", vec![1.1])
            .expect("failed to modify 'd1' on 'dev'");
        tree.insert("dev", "d3", vec![3.0])
            .expect("failed to insert new 'd3' on 'dev'");
        tree.delete("dev", "d2")
            .expect("failed to delete 'd2' from 'dev'");

        let diffs = tree.diff("dev", "main")
            .expect("failed to diff 'dev' against 'main'");
        assert_eq!(diffs.len(), 3);
    }

    #[test]
    fn test_merge_no_conflicts() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0])
            .expect("failed to insert 'd1' into 'main'");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.insert("dev", "d2", vec![2.0])
            .expect("failed to insert 'd2' into 'dev'");

        let result = tree
            .merge("dev", "main", MergeStrategy::SourceWins)
            .expect("failed to merge 'dev' into 'main' with SourceWins strategy");
        assert_eq!(result.merged, 1);
        assert_eq!(result.conflicts, 0);
        assert!(tree.contains("main", "d2"));
    }

    #[test]
    fn test_merge_with_conflicts() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0])
            .expect("failed to insert 'd1' into 'main'");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");

        // Both branches modify d1
        tree.insert("main", "d1", vec![1.1])
            .expect("failed to modify 'd1' on 'main' — simulating concurrent edit");
        tree.insert("dev", "d1", vec![1.2])
            .expect("failed to modify 'd1' on 'dev' — simulating concurrent edit");
        tree.insert("dev", "d2", vec![2.0])
            .expect("failed to insert new 'd2' on 'dev'");

        // Source wins: dev's d1 overwrites main's
        let result = tree
            .merge("dev", "main", MergeStrategy::SourceWins)
            .expect("failed to merge 'dev' into 'main' with SourceWins — conflict on 'd1' expected");
        assert_eq!(result.conflicts, 1);
        assert_eq!(result.merged, 2); // d1 (conflict resolved) + d2
        assert_eq!(
            tree.get("main", "d1")
                .expect("failed to get 'd1' from 'main' after SourceWins merge"),
            &[1.2]
        );
    }

    #[test]
    fn test_merge_target_wins() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0])
            .expect("failed to insert 'd1' into 'main'");
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.insert("main", "d1", vec![1.1])
            .expect("failed to modify 'd1' on 'main'");
        tree.insert("dev", "d1", vec![1.2])
            .expect("failed to modify 'd1' on 'dev'");

        let result = tree
            .merge("dev", "main", MergeStrategy::TargetWins)
            .expect("failed to merge 'dev' into 'main' with TargetWins strategy");
        assert_eq!(result.conflicts, 1);
        assert_eq!(result.skipped, 1);
        // Main keeps its version
        assert_eq!(
            tree.get("main", "d1")
                .expect("failed to get 'd1' from 'main' after TargetWins merge"),
            &[1.1]
        );
    }

    #[test]
    fn test_frozen_branch() {
        let mut tree = BranchTree::new();
        tree.insert("main", "d1", vec![1.0])
            .expect("failed to insert 'd1' into 'main'");
        tree.create_branch("stable", "main")
            .expect("failed to create 'stable' branch from 'main'");
        tree.freeze("stable")
            .expect("failed to freeze 'stable' branch");

        assert!(tree.insert("stable", "d2", vec![2.0]).is_err());
        assert!(tree.delete("stable", "d1").is_err());
    }

    #[test]
    fn test_delete_branch() {
        let mut tree = BranchTree::new();
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.delete_branch("dev")
            .expect("failed to delete 'dev' branch — it has no children and is not 'main'");
        assert!(tree.branch_info("dev").is_none());

        // Cannot delete main
        assert!(tree.delete_branch("main").is_err());
    }

    #[test]
    fn test_cannot_delete_branch_with_children() {
        let mut tree = BranchTree::new();
        tree.create_branch("dev", "main")
            .expect("failed to create 'dev' branch from 'main'");
        tree.create_branch("feature", "dev")
            .expect("failed to create 'feature' branch from 'dev'");
        assert!(tree.delete_branch("dev").is_err());
    }
}
