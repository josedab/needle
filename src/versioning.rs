//! Vector Versioning - Git-like version control for vector embeddings.
//!
//! Provides complete version history for vectors with branching, commits,
//! and the ability to query at any point in time.
//!
//! # Features
//!
//! - **Commit history**: Track all changes to vectors with messages
//! - **Branching**: Create branches for experimentation
//! - **Point-in-time queries**: Search vectors as they existed at any commit
//! - **Diff and merge**: Compare and merge vector changes
//! - **Rollback**: Revert to any previous state
//!
//! # Example
//!
//! ```ignore
//! use needle::versioning::{VectorRepo, Commit};
//!
//! let mut repo = VectorRepo::new("my_vectors", 128);
//!
//! // Add vectors and commit
//! repo.add("vec1", &embedding1, metadata)?;
//! repo.add("vec2", &embedding2, metadata)?;
//! let commit1 = repo.commit("Initial embeddings")?;
//!
//! // Update and commit again
//! repo.update("vec1", &new_embedding)?;
//! let commit2 = repo.commit("Updated vec1 embedding")?;
//!
//! // Query at previous commit
//! let old_vec = repo.get_at("vec1", &commit1)?;
//! ```

use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

/// A commit hash (simplified to u64 for this implementation).
pub type CommitHash = String;

/// Generate a commit hash.
fn generate_hash() -> CommitHash {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();

    let mut hasher = DefaultHasher::new();
    now.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// A versioned vector entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Vector ID.
    pub id: String,
    /// Vector data.
    pub vector: Vec<f32>,
    /// User metadata.
    pub metadata: HashMap<String, String>,
    /// Creation timestamp.
    pub created_at: u64,
    /// Last modification timestamp.
    pub modified_at: u64,
}

/// A commit in the version history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    /// Commit hash.
    pub hash: CommitHash,
    /// Parent commit hash (None for initial commit).
    pub parent: Option<CommitHash>,
    /// Commit message.
    pub message: String,
    /// Author (optional).
    pub author: Option<String>,
    /// Timestamp.
    pub timestamp: u64,
    /// Vector IDs added in this commit.
    pub added: Vec<String>,
    /// Vector IDs modified in this commit.
    pub modified: Vec<String>,
    /// Vector IDs deleted in this commit.
    pub deleted: Vec<String>,
    /// Snapshot of all vectors at this commit.
    snapshot: HashMap<String, VectorEntry>,
}

impl Commit {
    /// Get the number of changes in this commit.
    pub fn change_count(&self) -> usize {
        self.added.len() + self.modified.len() + self.deleted.len()
    }

    /// Check if commit is empty.
    pub fn is_empty(&self) -> bool {
        self.change_count() == 0
    }
}

/// A branch in the repository.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Branch name.
    pub name: String,
    /// Current commit hash.
    pub head: CommitHash,
    /// Creation timestamp.
    pub created_at: u64,
    /// Is this the default branch.
    pub is_default: bool,
}

/// Change type for diff.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    /// Vector was added.
    Added,
    /// Vector was modified.
    Modified,
    /// Vector was deleted.
    Deleted,
    /// Vector is unchanged.
    Unchanged,
}

/// Difference between two vectors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorDiff {
    /// Vector ID.
    pub id: String,
    /// Type of change.
    pub change_type: ChangeType,
    /// Old vector (if modified or deleted).
    pub old_vector: Option<Vec<f32>>,
    /// New vector (if added or modified).
    pub new_vector: Option<Vec<f32>>,
    /// Cosine similarity between versions (for modifications).
    pub similarity: Option<f32>,
}

/// Version control repository for vectors.
#[allow(dead_code)]
pub struct VectorRepo {
    /// Repository name (for display/identification).
    name: String,
    /// Vector dimensions.
    dimensions: usize,
    /// All commits by hash.
    commits: HashMap<CommitHash, Commit>,
    /// All branches.
    branches: HashMap<String, Branch>,
    /// Current branch name.
    current_branch: String,
    /// Working directory (uncommitted changes).
    working: HashMap<String, VectorEntry>,
    /// Staged changes.
    staged: StagedChanges,
    /// Deleted IDs in working directory.
    working_deleted: HashSet<String>,
}

/// Staged changes waiting to be committed.
#[derive(Debug, Clone, Default)]
struct StagedChanges {
    added: HashSet<String>,
    modified: HashSet<String>,
    deleted: HashSet<String>,
}

impl VectorRepo {
    /// Create a new vector repository.
    pub fn new(name: &str, dimensions: usize) -> Self {
        let main_branch = Branch {
            name: "main".to_string(),
            head: String::new(),
            created_at: Self::now(),
            is_default: true,
        };

        let mut branches = HashMap::new();
        branches.insert("main".to_string(), main_branch);

        Self {
            name: name.to_string(),
            dimensions,
            commits: HashMap::new(),
            branches,
            current_branch: "main".to_string(),
            working: HashMap::new(),
            staged: StagedChanges::default(),
            working_deleted: HashSet::new(),
        }
    }

    /// Get current timestamp.
    fn now() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Add a vector to the working directory.
    pub fn add(
        &mut self,
        id: &str,
        vector: &[f32],
        metadata: HashMap<String, String>,
    ) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        let now = Self::now();
        let entry = VectorEntry {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata,
            created_at: now,
            modified_at: now,
        };

        let is_new = self.get_latest(id).is_err();
        self.working.insert(id.to_string(), entry);
        self.working_deleted.remove(id);

        if is_new {
            self.staged.added.insert(id.to_string());
            self.staged.modified.remove(id);
        } else {
            self.staged.modified.insert(id.to_string());
            self.staged.added.remove(id);
        }
        self.staged.deleted.remove(id);

        Ok(())
    }

    /// Update an existing vector.
    pub fn update(&mut self, id: &str, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Dimension mismatch: expected {}, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        let existing = self.get_latest(id)?;
        let now = Self::now();

        let entry = VectorEntry {
            id: id.to_string(),
            vector: vector.to_vec(),
            metadata: existing.metadata.clone(),
            created_at: existing.created_at,
            modified_at: now,
        };

        self.working.insert(id.to_string(), entry);
        self.staged.modified.insert(id.to_string());
        self.staged.deleted.remove(id);

        Ok(())
    }

    /// Delete a vector.
    pub fn delete(&mut self, id: &str) -> Result<()> {
        // Check if exists in latest
        if !self.exists_at_head(id) && !self.working.contains_key(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        self.working.remove(id);
        self.working_deleted.insert(id.to_string());

        self.staged.deleted.insert(id.to_string());
        self.staged.added.remove(id);
        self.staged.modified.remove(id);

        Ok(())
    }

    /// Check if vector exists at HEAD.
    fn exists_at_head(&self, id: &str) -> bool {
        let head = &self.branches[&self.current_branch].head;
        if head.is_empty() {
            return false;
        }
        if let Some(commit) = self.commits.get(head) {
            return commit.snapshot.contains_key(id);
        }
        false
    }

    /// Get the latest version of a vector.
    pub fn get_latest(&self, id: &str) -> Result<VectorEntry> {
        // Check working directory first
        if let Some(entry) = self.working.get(id) {
            return Ok(entry.clone());
        }

        // Check if deleted
        if self.working_deleted.contains(id) {
            return Err(NeedleError::NotFound(format!("Vector '{}' was deleted", id)));
        }

        // Get from HEAD commit
        let head = &self.branches[&self.current_branch].head;
        if head.is_empty() {
            return Err(NeedleError::NotFound(format!("Vector '{}' not found", id)));
        }

        if let Some(commit) = self.commits.get(head) {
            if let Some(entry) = commit.snapshot.get(id) {
                return Ok(entry.clone());
            }
        }

        Err(NeedleError::NotFound(format!("Vector '{}' not found", id)))
    }

    /// Get a vector at a specific commit.
    pub fn get_at(&self, id: &str, commit_hash: &str) -> Result<VectorEntry> {
        let commit = self.commits.get(commit_hash)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", commit_hash)))?;

        commit.snapshot.get(id)
            .cloned()
            .ok_or_else(|| NeedleError::NotFound(format!(
                "Vector '{}' not found at commit '{}'",
                id, commit_hash
            )))
    }

    /// Commit staged changes.
    pub fn commit(&mut self, message: &str) -> Result<CommitHash> {
        self.commit_with_author(message, None)
    }

    /// Commit with author.
    pub fn commit_with_author(&mut self, message: &str, author: Option<&str>) -> Result<CommitHash> {
        if self.staged.added.is_empty()
            && self.staged.modified.is_empty()
            && self.staged.deleted.is_empty()
        {
            return Err(NeedleError::InvalidInput("Nothing to commit".to_string()));
        }

        let hash = generate_hash();
        let head = &self.branches[&self.current_branch].head;
        let parent = if head.is_empty() { None } else { Some(head.clone()) };

        // Build snapshot
        let mut snapshot = if let Some(ref p) = parent {
            self.commits[p].snapshot.clone()
        } else {
            HashMap::new()
        };

        // Apply changes
        for id in &self.staged.added {
            if let Some(entry) = self.working.get(id) {
                snapshot.insert(id.clone(), entry.clone());
            }
        }
        for id in &self.staged.modified {
            if let Some(entry) = self.working.get(id) {
                snapshot.insert(id.clone(), entry.clone());
            }
        }
        for id in &self.staged.deleted {
            snapshot.remove(id);
        }

        let commit = Commit {
            hash: hash.clone(),
            parent,
            message: message.to_string(),
            author: author.map(|s| s.to_string()),
            timestamp: Self::now(),
            added: self.staged.added.iter().cloned().collect(),
            modified: self.staged.modified.iter().cloned().collect(),
            deleted: self.staged.deleted.iter().cloned().collect(),
            snapshot,
        };

        self.commits.insert(hash.clone(), commit);

        // Update branch head
        if let Some(branch) = self.branches.get_mut(&self.current_branch) {
            branch.head = hash.clone();
        }

        // Clear staged changes
        self.staged = StagedChanges::default();
        self.working.clear();
        self.working_deleted.clear();

        Ok(hash)
    }

    /// Create a new branch.
    pub fn create_branch(&mut self, name: &str) -> Result<()> {
        if self.branches.contains_key(name) {
            return Err(NeedleError::InvalidInput(format!(
                "Branch '{}' already exists",
                name
            )));
        }

        let current_head = self.branches[&self.current_branch].head.clone();

        let branch = Branch {
            name: name.to_string(),
            head: current_head,
            created_at: Self::now(),
            is_default: false,
        };

        self.branches.insert(name.to_string(), branch);
        Ok(())
    }

    /// Switch to a branch.
    pub fn checkout(&mut self, branch_name: &str) -> Result<()> {
        if !self.branches.contains_key(branch_name) {
            return Err(NeedleError::NotFound(format!(
                "Branch '{}' not found",
                branch_name
            )));
        }

        // Check for uncommitted changes
        if !self.staged.added.is_empty()
            || !self.staged.modified.is_empty()
            || !self.staged.deleted.is_empty()
        {
            return Err(NeedleError::InvalidInput(
                "Cannot checkout with uncommitted changes".to_string(),
            ));
        }

        self.current_branch = branch_name.to_string();
        self.working.clear();
        self.working_deleted.clear();

        Ok(())
    }

    /// Checkout a specific commit (detached HEAD).
    pub fn checkout_commit(&mut self, commit_hash: &str) -> Result<()> {
        if !self.commits.contains_key(commit_hash) {
            return Err(NeedleError::NotFound(format!(
                "Commit '{}' not found",
                commit_hash
            )));
        }

        // Check for uncommitted changes
        if !self.staged.added.is_empty()
            || !self.staged.modified.is_empty()
            || !self.staged.deleted.is_empty()
        {
            return Err(NeedleError::InvalidInput(
                "Cannot checkout with uncommitted changes".to_string(),
            ));
        }

        // Create a temporary detached branch
        let detached_name = format!("detached-{}", &commit_hash[..8]);
        let branch = Branch {
            name: detached_name.clone(),
            head: commit_hash.to_string(),
            created_at: Self::now(),
            is_default: false,
        };

        self.branches.insert(detached_name.clone(), branch);
        self.current_branch = detached_name;
        self.working.clear();
        self.working_deleted.clear();

        Ok(())
    }

    /// Get commit history.
    pub fn log(&self, limit: Option<usize>) -> Vec<&Commit> {
        let mut history = Vec::new();
        let mut current = self.branches[&self.current_branch].head.clone();

        while !current.is_empty() {
            if let Some(commit) = self.commits.get(&current) {
                history.push(commit);
                if let Some(limit) = limit {
                    if history.len() >= limit {
                        break;
                    }
                }
                current = commit.parent.clone().unwrap_or_default();
            } else {
                break;
            }
        }

        history
    }

    /// Diff between two commits.
    pub fn diff(&self, from: &str, to: &str) -> Result<Vec<VectorDiff>> {
        let from_commit = self.commits.get(from)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", from)))?;
        let to_commit = self.commits.get(to)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", to)))?;

        let mut diffs = Vec::new();
        let mut all_ids: HashSet<&String> = from_commit.snapshot.keys().collect();
        all_ids.extend(to_commit.snapshot.keys());

        for id in all_ids {
            let old = from_commit.snapshot.get(id);
            let new = to_commit.snapshot.get(id);

            let diff = match (old, new) {
                (None, Some(n)) => VectorDiff {
                    id: id.clone(),
                    change_type: ChangeType::Added,
                    old_vector: None,
                    new_vector: Some(n.vector.clone()),
                    similarity: None,
                },
                (Some(o), None) => VectorDiff {
                    id: id.clone(),
                    change_type: ChangeType::Deleted,
                    old_vector: Some(o.vector.clone()),
                    new_vector: None,
                    similarity: None,
                },
                (Some(o), Some(n)) => {
                    if o.vector == n.vector {
                        VectorDiff {
                            id: id.clone(),
                            change_type: ChangeType::Unchanged,
                            old_vector: None,
                            new_vector: None,
                            similarity: Some(1.0),
                        }
                    } else {
                        let sim = self.cosine_similarity(&o.vector, &n.vector);
                        VectorDiff {
                            id: id.clone(),
                            change_type: ChangeType::Modified,
                            old_vector: Some(o.vector.clone()),
                            new_vector: Some(n.vector.clone()),
                            similarity: Some(sim),
                        }
                    }
                }
                (None, None) => continue,
            };

            diffs.push(diff);
        }

        Ok(diffs)
    }

    /// Compute cosine similarity.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if mag_a == 0.0 || mag_b == 0.0 {
            return 0.0;
        }

        dot / (mag_a * mag_b)
    }

    /// Rollback to a previous commit.
    pub fn rollback(&mut self, commit_hash: &str) -> Result<CommitHash> {
        let target = self.commits.get(commit_hash)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", commit_hash)))?
            .clone();

        // Stage all vectors from target commit
        for (id, entry) in &target.snapshot {
            self.working.insert(id.clone(), entry.clone());
            self.staged.added.insert(id.clone());
        }

        // Mark current vectors not in target as deleted
        let current_head = &self.branches[&self.current_branch].head;
        if let Some(current) = self.commits.get(current_head) {
            for id in current.snapshot.keys() {
                if !target.snapshot.contains_key(id) {
                    self.staged.deleted.insert(id.clone());
                }
            }
        }

        // Commit the rollback
        let message = format!("Rollback to {}", &commit_hash[..8]);
        self.commit(&message)
    }

    /// Merge a branch into current branch.
    pub fn merge(&mut self, branch_name: &str) -> Result<MergeResult> {
        let source_branch = self.branches.get(branch_name)
            .ok_or_else(|| NeedleError::NotFound(format!("Branch '{}' not found", branch_name)))?
            .clone();

        let current_head = &self.branches[&self.current_branch].head;
        if current_head.is_empty() {
            return Err(NeedleError::InvalidInput("Cannot merge into empty branch".to_string()));
        }

        let source_commit = self.commits.get(&source_branch.head)
            .ok_or_else(|| NeedleError::NotFound("Source commit not found".to_string()))?
            .clone();
        let current_commit = self.commits.get(current_head)
            .ok_or_else(|| NeedleError::NotFound("Current commit not found".to_string()))?
            .clone();

        // Simple merge: apply all vectors from source that aren't in current
        let mut added = 0;
        let mut modified = 0;
        let mut conflicts = Vec::new();

        for (id, entry) in &source_commit.snapshot {
            if let Some(current_entry) = current_commit.snapshot.get(id) {
                if current_entry.vector != entry.vector {
                    // Conflict - for now, prefer source (theirs)
                    conflicts.push(id.clone());
                    self.working.insert(id.clone(), entry.clone());
                    self.staged.modified.insert(id.clone());
                    modified += 1;
                }
            } else {
                self.working.insert(id.clone(), entry.clone());
                self.staged.added.insert(id.clone());
                added += 1;
            }
        }

        let has_conflicts = !conflicts.is_empty();

        if !self.staged.added.is_empty() || !self.staged.modified.is_empty() {
            let message = format!("Merge branch '{}' into {}", branch_name, self.current_branch);
            let commit_hash = self.commit(&message)?;

            Ok(MergeResult {
                success: true,
                commit: Some(commit_hash),
                added,
                modified,
                conflicts,
                has_conflicts,
            })
        } else {
            Ok(MergeResult {
                success: true,
                commit: None,
                added: 0,
                modified: 0,
                conflicts: Vec::new(),
                has_conflicts: false,
            })
        }
    }

    /// List all vectors at current HEAD.
    pub fn list_vectors(&self) -> Vec<String> {
        let head = &self.branches[&self.current_branch].head;
        if head.is_empty() {
            return self.working.keys().cloned().collect();
        }

        if let Some(commit) = self.commits.get(head) {
            let mut ids: Vec<String> = commit.snapshot.keys().cloned().collect();
            // Add working directory additions
            for id in self.working.keys() {
                if !ids.contains(id) {
                    ids.push(id.clone());
                }
            }
            // Remove working directory deletions
            ids.retain(|id| !self.working_deleted.contains(id));
            ids
        } else {
            self.working.keys().cloned().collect()
        }
    }

    /// List all branches.
    pub fn list_branches(&self) -> Vec<&Branch> {
        self.branches.values().collect()
    }

    /// Get current branch name.
    pub fn current_branch(&self) -> &str {
        &self.current_branch
    }

    /// Get repository status.
    pub fn status(&self) -> RepoStatus {
        RepoStatus {
            branch: self.current_branch.clone(),
            staged_added: self.staged.added.len(),
            staged_modified: self.staged.modified.len(),
            staged_deleted: self.staged.deleted.len(),
            total_vectors: self.list_vectors().len(),
            total_commits: self.commits.len(),
        }
    }

    /// Search vectors at HEAD.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }

        let vectors = self.list_vectors();
        let mut results: Vec<SearchResult> = vectors
            .iter()
            .filter_map(|id| {
                self.get_latest(id).ok().map(|entry| {
                    let similarity = self.cosine_similarity(query, &entry.vector);
                    SearchResult {
                        id: id.clone(),
                        similarity,
                        entry,
                    }
                })
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);

        Ok(results)
    }

    /// Search vectors at a specific commit.
    pub fn search_at(&self, query: &[f32], commit_hash: &str, k: usize) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }

        let commit = self.commits.get(commit_hash)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", commit_hash)))?;

        let mut results: Vec<SearchResult> = commit.snapshot
            .iter()
            .map(|(id, entry)| {
                let similarity = self.cosine_similarity(query, &entry.vector);
                SearchResult {
                    id: id.clone(),
                    similarity,
                    entry: entry.clone(),
                }
            })
            .collect();

        results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());
        results.truncate(k);

        Ok(results)
    }
}

/// Result of a merge operation.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// Whether merge succeeded.
    pub success: bool,
    /// Merge commit hash.
    pub commit: Option<CommitHash>,
    /// Number of vectors added.
    pub added: usize,
    /// Number of vectors modified.
    pub modified: usize,
    /// Conflicting vector IDs.
    pub conflicts: Vec<String>,
    /// Whether there were conflicts.
    pub has_conflicts: bool,
}

/// Repository status.
#[derive(Debug, Clone)]
pub struct RepoStatus {
    /// Current branch.
    pub branch: String,
    /// Staged additions.
    pub staged_added: usize,
    /// Staged modifications.
    pub staged_modified: usize,
    /// Staged deletions.
    pub staged_deleted: usize,
    /// Total vectors at HEAD.
    pub total_vectors: usize,
    /// Total commits.
    pub total_commits: usize,
}

/// Search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Vector ID.
    pub id: String,
    /// Cosine similarity.
    pub similarity: f32,
    /// Vector entry.
    pub entry: VectorEntry,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_repo() {
        let repo = VectorRepo::new("test", 128);
        assert_eq!(repo.current_branch(), "main");
        assert_eq!(repo.list_vectors().len(), 0);
    }

    #[test]
    fn test_add_vector() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();

        let status = repo.status();
        assert_eq!(status.staged_added, 1);
    }

    #[test]
    fn test_commit() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();

        let hash = repo.commit("Initial commit").unwrap();

        assert!(!hash.is_empty());
        assert_eq!(repo.list_vectors().len(), 2);

        let status = repo.status();
        assert_eq!(status.staged_added, 0);
        assert_eq!(status.total_commits, 1);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut repo = VectorRepo::new("test", 4);

        let result = repo.add("vec1", &[1.0, 2.0, 3.0], HashMap::new());
        assert!(result.is_err());
    }

    #[test]
    fn test_get_vector() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Add vec1").unwrap();

        let entry = repo.get_latest("vec1").unwrap();
        assert_eq!(entry.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_update_vector() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Add vec1").unwrap();

        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        repo.commit("Update vec1").unwrap();

        let entry = repo.get_latest("vec1").unwrap();
        assert_eq!(entry.vector, vec![9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn test_delete_vector() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Add vec1").unwrap();

        repo.delete("vec1").unwrap();
        repo.commit("Delete vec1").unwrap();

        let result = repo.get_latest("vec1");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_at_commit() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        let _commit2 = repo.commit("V2").unwrap();

        // Get at first commit
        let old = repo.get_at("vec1", &commit1).unwrap();
        assert_eq!(old.vector, vec![1.0, 2.0, 3.0, 4.0]);

        // Get latest
        let new = repo.get_latest("vec1").unwrap();
        assert_eq!(new.vector, vec![9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn test_log() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Commit 1").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();
        repo.commit("Commit 2").unwrap();

        let history = repo.log(None);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].message, "Commit 2");
        assert_eq!(history[1].message, "Commit 1");
    }

    #[test]
    fn test_branching() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Main commit").unwrap();

        repo.create_branch("feature").unwrap();
        repo.checkout("feature").unwrap();

        assert_eq!(repo.current_branch(), "feature");

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();
        repo.commit("Feature commit").unwrap();

        // Feature branch has vec2
        assert_eq!(repo.list_vectors().len(), 2);

        // Switch back to main
        repo.checkout("main").unwrap();
        assert_eq!(repo.list_vectors().len(), 1);
    }

    #[test]
    fn test_diff() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        let commit1 = repo.commit("Add vec1").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();
        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        let commit2 = repo.commit("Changes").unwrap();

        let diffs = repo.diff(&commit1, &commit2).unwrap();

        let added: Vec<_> = diffs.iter().filter(|d| d.change_type == ChangeType::Added).collect();
        let modified: Vec<_> = diffs.iter().filter(|d| d.change_type == ChangeType::Modified).collect();

        assert_eq!(added.len(), 1);
        assert_eq!(modified.len(), 1);
    }

    #[test]
    fn test_rollback() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        repo.commit("V2").unwrap();

        // Rollback to commit1
        repo.rollback(&commit1).unwrap();

        let entry = repo.get_latest("vec1").unwrap();
        assert_eq!(entry.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_merge() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Main").unwrap();

        repo.create_branch("feature").unwrap();
        repo.checkout("feature").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();
        repo.commit("Feature").unwrap();

        repo.checkout("main").unwrap();
        let result = repo.merge("feature").unwrap();

        assert!(result.success);
        assert_eq!(result.added, 1);
        assert_eq!(repo.list_vectors().len(), 2);
    }

    #[test]
    fn test_search() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("a", &[1.0, 0.0, 0.0, 0.0], HashMap::new()).unwrap();
        repo.add("b", &[0.0, 1.0, 0.0, 0.0], HashMap::new()).unwrap();
        repo.add("c", &[1.0, 1.0, 0.0, 0.0], HashMap::new()).unwrap();
        repo.commit("Vectors").unwrap();

        let results = repo.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a"); // Exact match
    }

    #[test]
    fn test_search_at_commit() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("a", &[1.0, 0.0, 0.0, 0.0], HashMap::new()).unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.update("a", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        repo.commit("V2").unwrap();

        // Search at old commit
        let results = repo.search_at(&[1.0, 0.0, 0.0, 0.0], &commit1, 1).unwrap();
        assert!(results[0].similarity > 0.99);

        // Search at current (different vector)
        let results = repo.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert!(results[0].similarity < 0.1);
    }

    #[test]
    fn test_cannot_commit_empty() {
        let mut repo = VectorRepo::new("test", 4);

        let result = repo.commit("Empty");
        assert!(result.is_err());
    }

    #[test]
    fn test_cannot_checkout_with_changes() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Main").unwrap();

        repo.create_branch("feature").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new()).unwrap();
        // Don't commit

        let result = repo.checkout("feature");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_branches() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new()).unwrap();
        repo.commit("Init").unwrap();

        repo.create_branch("feature1").unwrap();
        repo.create_branch("feature2").unwrap();

        let branches = repo.list_branches();
        assert_eq!(branches.len(), 3);
    }

    #[test]
    fn test_metadata() {
        let mut repo = VectorRepo::new("test", 4);

        let mut meta = HashMap::new();
        meta.insert("category".to_string(), "test".to_string());
        meta.insert("source".to_string(), "unit_test".to_string());

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], meta).unwrap();
        repo.commit("With metadata").unwrap();

        let entry = repo.get_latest("vec1").unwrap();
        assert_eq!(entry.metadata.get("category"), Some(&"test".to_string()));
    }
}
