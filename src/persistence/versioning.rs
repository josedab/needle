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
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
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
            .unwrap_or_default()
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
            return Err(NeedleError::NotFound(format!(
                "Vector '{}' was deleted",
                id
            )));
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
        let commit = self
            .commits
            .get(commit_hash)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", commit_hash)))?;

        commit.snapshot.get(id).cloned().ok_or_else(|| {
            NeedleError::NotFound(format!(
                "Vector '{}' not found at commit '{}'",
                id, commit_hash
            ))
        })
    }

    /// Commit staged changes.
    pub fn commit(&mut self, message: &str) -> Result<CommitHash> {
        self.commit_with_author(message, None)
    }

    /// Commit with author.
    pub fn commit_with_author(
        &mut self,
        message: &str,
        author: Option<&str>,
    ) -> Result<CommitHash> {
        if self.staged.added.is_empty()
            && self.staged.modified.is_empty()
            && self.staged.deleted.is_empty()
        {
            return Err(NeedleError::InvalidInput("Nothing to commit".to_string()));
        }

        let hash = generate_hash();
        let head = &self.branches[&self.current_branch].head;
        let parent = if head.is_empty() {
            None
        } else {
            Some(head.clone())
        };

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
        let from_commit = self
            .commits
            .get(from)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", from)))?;
        let to_commit = self
            .commits
            .get(to)
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
        let target = self
            .commits
            .get(commit_hash)
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
        let source_branch = self
            .branches
            .get(branch_name)
            .ok_or_else(|| NeedleError::NotFound(format!("Branch '{}' not found", branch_name)))?
            .clone();

        let current_head = &self.branches[&self.current_branch].head;
        if current_head.is_empty() {
            return Err(NeedleError::InvalidInput(
                "Cannot merge into empty branch".to_string(),
            ));
        }

        let source_commit = self
            .commits
            .get(&source_branch.head)
            .ok_or_else(|| NeedleError::NotFound("Source commit not found".to_string()))?
            .clone();
        let current_commit = self
            .commits
            .get(current_head)
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
            let message = format!(
                "Merge branch '{}' into {}",
                branch_name, self.current_branch
            );
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

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Search vectors at a specific commit.
    pub fn search_at(
        &self,
        query: &[f32],
        commit_hash: &str,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }

        let commit = self
            .commits
            .get(commit_hash)
            .ok_or_else(|| NeedleError::NotFound(format!("Commit '{}' not found", commit_hash)))?;

        let mut results: Vec<SearchResult> = commit
            .snapshot
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

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
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

// ============================================================================
// Semantic Time-Travel Extensions
// ============================================================================

/// Time specification for time-travel queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeSpec {
    /// Unix timestamp (seconds since epoch)
    Timestamp(u64),
    /// ISO 8601 datetime string
    DateTime(String),
    /// Relative time (e.g., "1 hour ago", "last tuesday")
    Relative(String),
    /// Specific commit hash
    Commit(CommitHash),
}

impl TimeSpec {
    /// Convert to Unix timestamp
    pub fn to_timestamp(&self) -> Result<u64> {
        match self {
            TimeSpec::Timestamp(ts) => Ok(*ts),
            TimeSpec::Commit(_) => Err(NeedleError::InvalidInput(
                "Commit hash cannot be converted to timestamp".to_string(),
            )),
            TimeSpec::DateTime(s) => {
                // Parse ISO 8601 format: "2024-01-15T10:30:00Z"
                parse_datetime(s)
            }
            TimeSpec::Relative(s) => parse_relative_time(s),
        }
    }
}

/// Parse ISO 8601 datetime to Unix timestamp
fn parse_datetime(s: &str) -> Result<u64> {
    // Simple parser for common formats
    // Format: "2024-01-15T10:30:00Z" or "2024-01-15 10:30:00"
    let s = s.trim().replace('T', " ").replace('Z', "");
    let parts: Vec<&str> = s.split(' ').collect();

    if parts.is_empty() {
        return Err(NeedleError::InvalidInput(format!(
            "Invalid datetime: {}",
            s
        )));
    }

    let date_parts: Vec<&str> = parts[0].split('-').collect();
    if date_parts.len() != 3 {
        return Err(NeedleError::InvalidInput(format!(
            "Invalid date format: {}",
            parts[0]
        )));
    }

    let year: i32 = date_parts[0]
        .parse()
        .map_err(|_| NeedleError::InvalidInput("Invalid year".to_string()))?;
    let month: u32 = date_parts[1]
        .parse()
        .map_err(|_| NeedleError::InvalidInput("Invalid month".to_string()))?;
    let day: u32 = date_parts[2]
        .parse()
        .map_err(|_| NeedleError::InvalidInput("Invalid day".to_string()))?;

    let (hour, minute, second) = if parts.len() > 1 {
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        (
            time_parts.first().and_then(|s| s.parse().ok()).unwrap_or(0),
            time_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0),
            time_parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0),
        )
    } else {
        (0, 0, 0)
    };

    // Simple timestamp calculation using chrono-style calculation
    // Days from year 1 to the given date, then subtract days to Unix epoch
    let days = days_from_year_1(year, month, day);
    let unix_epoch_days = days_from_year_1(1970, 1, 1);
    let days_since_epoch = days - unix_epoch_days;

    if days_since_epoch < 0 {
        return Err(NeedleError::InvalidInput(
            "Date is before Unix epoch".to_string(),
        ));
    }

    let seconds =
        days_since_epoch as u64 * 86400 + hour as u64 * 3600 + minute as u64 * 60 + second as u64;

    Ok(seconds)
}

/// Calculate days from year 1 to a given date (simplified)
fn days_from_year_1(year: i32, month: u32, day: u32) -> i64 {
    let y = year as i64;
    let m = month as i64;
    let d = day as i64;

    // Simplified calculation
    let mut days = (y - 1) * 365 + (y - 1) / 4 - (y - 1) / 100 + (y - 1) / 400;

    // Add days for months
    let month_days = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if (1..=12).contains(&m) {
        days += month_days[(m - 1) as usize] as i64;
    }

    // Add leap day if applicable
    let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
    if is_leap && m > 2 {
        days += 1;
    }

    days + d
}

/// Calculate days since Unix epoch (1970-01-01) - kept for compatibility
#[allow(dead_code)]
fn days_since_unix_epoch(year: i32, month: u32, day: u32) -> i64 {
    let mut y = year as i64;
    let m = month as i64;
    let d = day as i64;

    // Adjust for months
    let a = (14 - m) / 12;
    y -= a;
    let m_adj = m + 12 * a - 3;

    // Julian day number
    let jdn = d + (153 * m_adj + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045;

    // Unix epoch is Julian day 2440588
    jdn - 2440588
}

/// Parse relative time string to Unix timestamp
fn parse_relative_time(s: &str) -> Result<u64> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let s = s.to_lowercase();

    // Handle common patterns
    if s == "now" {
        return Ok(now);
    }

    // "X units ago" pattern
    if s.ends_with(" ago") {
        let parts: Vec<&str> = s.trim_end_matches(" ago").split_whitespace().collect();
        if parts.len() >= 2 {
            let amount: u64 = parts[0].parse().map_err(|_| {
                NeedleError::InvalidInput(format!("Invalid amount in relative time: {}", s))
            })?;
            let unit = parts[1];

            let seconds = match unit {
                "second" | "seconds" | "sec" | "secs" => amount,
                "minute" | "minutes" | "min" | "mins" => amount * 60,
                "hour" | "hours" | "hr" | "hrs" => amount * 3600,
                "day" | "days" => amount * 86400,
                "week" | "weeks" => amount * 86400 * 7,
                "month" | "months" => amount * 86400 * 30,
                "year" | "years" => amount * 86400 * 365,
                _ => {
                    return Err(NeedleError::InvalidInput(format!(
                        "Unknown time unit: {}",
                        unit
                    )))
                }
            };

            return Ok(now.saturating_sub(seconds));
        }
    }

    // Handle day names (last monday, last tuesday, etc.)
    let weekdays = [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ];
    for (i, day_name) in weekdays.iter().enumerate() {
        if s.contains(day_name) {
            // Calculate days back to that weekday
            let current_day = (now / 86400 + 4) % 7; // Days since epoch mod 7, adjusted for Thursday epoch
            let target_day = i as u64;
            let days_back = if current_day >= target_day {
                current_day - target_day
            } else {
                7 - (target_day - current_day)
            };
            // If "last", go back one more week if we're on that day
            let days_back = if s.contains("last") && days_back == 0 {
                7
            } else {
                days_back
            };
            return Ok(now - days_back * 86400);
        }
    }

    // Handle "yesterday"
    if s == "yesterday" {
        return Ok(now - 86400);
    }

    Err(NeedleError::InvalidInput(format!(
        "Cannot parse relative time: {}",
        s
    )))
}

/// History entry for a vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorHistoryEntry {
    /// Commit hash where this version exists
    pub commit_hash: CommitHash,
    /// Commit message
    pub commit_message: String,
    /// Timestamp of the commit
    pub timestamp: u64,
    /// Change type at this commit
    pub change_type: ChangeType,
    /// The vector data at this version (None if deleted)
    pub vector: Option<Vec<f32>>,
    /// Similarity to previous version (for modifications)
    pub similarity_to_previous: Option<f32>,
}

/// Time range for querying changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start time (inclusive)
    pub start: TimeSpec,
    /// End time (inclusive)
    pub end: TimeSpec,
}

impl VectorRepo {
    /// Find the commit that was active at a specific time
    pub fn find_commit_at_time(&self, time: &TimeSpec) -> Result<Option<&Commit>> {
        let target_ts =
            match time {
                TimeSpec::Commit(hash) => {
                    return self.commits.get(hash).map(Some).ok_or_else(|| {
                        NeedleError::NotFound(format!("Commit '{}' not found", hash))
                    });
                }
                _ => time.to_timestamp()?,
            };

        // Find the most recent commit at or before the target timestamp
        let mut best_commit: Option<&Commit> = None;

        for commit in self.commits.values() {
            if commit.timestamp <= target_ts {
                match best_commit {
                    None => best_commit = Some(commit),
                    Some(current_best) if commit.timestamp > current_best.timestamp => {
                        best_commit = Some(commit);
                    }
                    _ => {}
                }
            }
        }

        Ok(best_commit)
    }

    /// Search vectors as they existed at a specific time
    ///
    /// This is the core "semantic time-travel" functionality - query your vectors
    /// as they existed at any point in time.
    ///
    /// # Example
    /// ```ignore
    /// // Search as of last Tuesday
    /// let results = repo.search_at_time(
    ///     &query_vec,
    ///     &TimeSpec::Relative("last tuesday".to_string()),
    ///     10
    /// )?;
    /// ```
    pub fn search_at_time(
        &self,
        query: &[f32],
        time: &TimeSpec,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dimensions {
            return Err(NeedleError::InvalidInput(format!(
                "Query dimension mismatch: expected {}, got {}",
                self.dimensions,
                query.len()
            )));
        }

        // Find the commit at the specified time
        let commit = self.find_commit_at_time(time)?.ok_or_else(|| {
            NeedleError::NotFound("No commit found at the specified time".to_string())
        })?;

        // Search within that commit's snapshot
        let mut results: Vec<SearchResult> = commit
            .snapshot
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

        results.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);

        Ok(results)
    }

    /// Get the complete history of a vector across all commits
    pub fn vector_history(&self, vector_id: &str) -> Vec<VectorHistoryEntry> {
        let mut history = Vec::new();

        // Build ordered commit list by following parent chain from HEAD
        let mut ordered_commits = Vec::new();
        let head = &self.branches[&self.current_branch].head;

        if !head.is_empty() {
            let mut current = head.clone();
            while !current.is_empty() {
                if let Some(commit) = self.commits.get(&current) {
                    ordered_commits.push(commit);
                    current = commit.parent.clone().unwrap_or_default();
                } else {
                    break;
                }
            }
        }

        // Reverse to get chronological order (oldest first)
        ordered_commits.reverse();

        let mut previous_vector: Option<Vec<f32>> = None;

        for commit in ordered_commits {
            let current_vector = commit.snapshot.get(vector_id);

            let change_type = if commit.added.contains(&vector_id.to_string()) {
                ChangeType::Added
            } else if commit.modified.contains(&vector_id.to_string()) {
                ChangeType::Modified
            } else if commit.deleted.contains(&vector_id.to_string()) {
                ChangeType::Deleted
            } else if current_vector.is_some() {
                // Vector exists but wasn't changed in this commit
                continue;
            } else {
                continue;
            };

            let similarity = match (&previous_vector, &current_vector) {
                (Some(prev), Some(curr)) => Some(self.cosine_similarity(prev, &curr.vector)),
                _ => None,
            };

            history.push(VectorHistoryEntry {
                commit_hash: commit.hash.clone(),
                commit_message: commit.message.clone(),
                timestamp: commit.timestamp,
                change_type,
                vector: current_vector.map(|e| e.vector.clone()),
                similarity_to_previous: similarity,
            });

            previous_vector = current_vector.map(|e| e.vector.clone());
        }

        history
    }

    /// Get all changes between two points in time
    pub fn changes_between(&self, range: &TimeRange) -> Result<Vec<VectorDiff>> {
        let start_commit = self.find_commit_at_time(&range.start)?;
        let end_commit = self.find_commit_at_time(&range.end)?;

        match (start_commit, end_commit) {
            (Some(start), Some(end)) => self.diff(&start.hash, &end.hash),
            (None, Some(end)) => {
                // From beginning of time to end
                let mut diffs = Vec::new();

                for (id, entry) in &end.snapshot {
                    diffs.push(VectorDiff {
                        id: id.clone(),
                        change_type: ChangeType::Added,
                        old_vector: None,
                        new_vector: Some(entry.vector.clone()),
                        similarity: None,
                    });
                }

                Ok(diffs)
            }
            (Some(start), None) => {
                // All vectors at start were "deleted" by end
                let mut diffs = Vec::new();

                for (id, entry) in &start.snapshot {
                    diffs.push(VectorDiff {
                        id: id.clone(),
                        change_type: ChangeType::Deleted,
                        old_vector: Some(entry.vector.clone()),
                        new_vector: None,
                        similarity: None,
                    });
                }

                Ok(diffs)
            }
            (None, None) => Ok(Vec::new()),
        }
    }

    /// Get a vector at a specific time
    pub fn get_at_time(&self, vector_id: &str, time: &TimeSpec) -> Result<VectorEntry> {
        let commit = self.find_commit_at_time(time)?.ok_or_else(|| {
            NeedleError::NotFound("No commit found at the specified time".to_string())
        })?;

        commit.snapshot.get(vector_id).cloned().ok_or_else(|| {
            NeedleError::NotFound(format!(
                "Vector '{}' not found at specified time",
                vector_id
            ))
        })
    }

    /// Compare a vector between two points in time
    pub fn compare_at_times(
        &self,
        vector_id: &str,
        time1: &TimeSpec,
        time2: &TimeSpec,
    ) -> Result<VectorDiff> {
        let entry1 = self.get_at_time(vector_id, time1).ok();
        let entry2 = self.get_at_time(vector_id, time2).ok();

        match (entry1, entry2) {
            (None, None) => Err(NeedleError::NotFound(format!(
                "Vector '{}' not found at either time point",
                vector_id
            ))),
            (None, Some(e2)) => Ok(VectorDiff {
                id: vector_id.to_string(),
                change_type: ChangeType::Added,
                old_vector: None,
                new_vector: Some(e2.vector),
                similarity: None,
            }),
            (Some(e1), None) => Ok(VectorDiff {
                id: vector_id.to_string(),
                change_type: ChangeType::Deleted,
                old_vector: Some(e1.vector),
                new_vector: None,
                similarity: None,
            }),
            (Some(e1), Some(e2)) => {
                if e1.vector == e2.vector {
                    Ok(VectorDiff {
                        id: vector_id.to_string(),
                        change_type: ChangeType::Unchanged,
                        old_vector: None,
                        new_vector: None,
                        similarity: Some(1.0),
                    })
                } else {
                    let sim = self.cosine_similarity(&e1.vector, &e2.vector);
                    Ok(VectorDiff {
                        id: vector_id.to_string(),
                        change_type: ChangeType::Modified,
                        old_vector: Some(e1.vector),
                        new_vector: Some(e2.vector),
                        similarity: Some(sim),
                    })
                }
            }
        }
    }

    /// List all commits within a time range
    pub fn commits_in_range(&self, range: &TimeRange) -> Result<Vec<&Commit>> {
        let start_ts = range.start.to_timestamp()?;
        let end_ts = range.end.to_timestamp()?;

        let mut commits: Vec<&Commit> = self
            .commits
            .values()
            .filter(|c| c.timestamp >= start_ts && c.timestamp <= end_ts)
            .collect();

        commits.sort_by_key(|c| c.timestamp);
        Ok(commits)
    }
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

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();

        let status = repo.status();
        assert_eq!(status.staged_added, 1);
    }

    #[test]
    fn test_commit() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new())
            .unwrap();

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

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Add vec1").unwrap();

        let entry = repo.get_latest("vec1").unwrap();
        assert_eq!(entry.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_update_vector() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Add vec1").unwrap();

        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        repo.commit("Update vec1").unwrap();

        let entry = repo.get_latest("vec1").unwrap();
        assert_eq!(entry.vector, vec![9.0, 8.0, 7.0, 6.0]);
    }

    #[test]
    fn test_delete_vector() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Add vec1").unwrap();

        repo.delete("vec1").unwrap();
        repo.commit("Delete vec1").unwrap();

        let result = repo.get_latest("vec1");
        assert!(result.is_err());
    }

    #[test]
    fn test_get_at_commit() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
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

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Commit 1").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new())
            .unwrap();
        repo.commit("Commit 2").unwrap();

        let history = repo.log(None);
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].message, "Commit 2");
        assert_eq!(history[1].message, "Commit 1");
    }

    #[test]
    fn test_branching() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Main commit").unwrap();

        repo.create_branch("feature").unwrap();
        repo.checkout("feature").unwrap();

        assert_eq!(repo.current_branch(), "feature");

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new())
            .unwrap();
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

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("Add vec1").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new())
            .unwrap();
        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        let commit2 = repo.commit("Changes").unwrap();

        let diffs = repo.diff(&commit1, &commit2).unwrap();

        let added: Vec<_> = diffs
            .iter()
            .filter(|d| d.change_type == ChangeType::Added)
            .collect();
        let modified: Vec<_> = diffs
            .iter()
            .filter(|d| d.change_type == ChangeType::Modified)
            .collect();

        assert_eq!(added.len(), 1);
        assert_eq!(modified.len(), 1);
    }

    #[test]
    fn test_rollback() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
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

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Main").unwrap();

        repo.create_branch("feature").unwrap();
        repo.checkout("feature").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new())
            .unwrap();
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

        repo.add("a", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.add("b", &[0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.add("c", &[1.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.commit("Vectors").unwrap();

        let results = repo.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, "a"); // Exact match
    }

    #[test]
    fn test_search_at_commit() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("a", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
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

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        repo.commit("Main").unwrap();

        repo.create_branch("feature").unwrap();

        repo.add("vec2", &[5.0, 6.0, 7.0, 8.0], HashMap::new())
            .unwrap();
        // Don't commit

        let result = repo.checkout("feature");
        assert!(result.is_err());
    }

    #[test]
    fn test_list_branches() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
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

    // ========================================================================
    // Semantic Time-Travel Tests
    // ========================================================================

    #[test]
    fn test_parse_relative_time_ago() {
        // Test "X units ago" parsing
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let result = parse_relative_time("1 hour ago").unwrap();
        assert!(result > now - 3700 && result < now - 3500);

        let result = parse_relative_time("2 days ago").unwrap();
        assert!(result > now - 2 * 86400 - 100 && result < now - 2 * 86400 + 100);
    }

    #[test]
    fn test_parse_relative_time_now() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let result = parse_relative_time("now").unwrap();
        assert!(result >= now - 1 && result <= now + 1);
    }

    #[test]
    fn test_parse_relative_time_yesterday() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let result = parse_relative_time("yesterday").unwrap();
        assert!(result > now - 86500 && result < now - 86300);
    }

    #[test]
    fn test_parse_datetime() {
        // Test ISO 8601 format
        let result = parse_datetime("2024-01-15T10:30:00Z").unwrap();
        assert!(result > 0); // Should parse successfully

        let result = parse_datetime("2024-01-15 10:30:00").unwrap();
        assert!(result > 0);
    }

    #[test]
    fn test_timespec_to_timestamp() {
        let ts = TimeSpec::Timestamp(1705312200);
        assert_eq!(ts.to_timestamp().unwrap(), 1705312200);

        let dt = TimeSpec::DateTime("2024-01-15T10:30:00Z".to_string());
        assert!(dt.to_timestamp().is_ok());

        let rel = TimeSpec::Relative("1 hour ago".to_string());
        assert!(rel.to_timestamp().is_ok());
    }

    #[test]
    fn test_find_commit_at_time() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("V1").unwrap();
        let ts1 = repo.commits.get(&commit1).unwrap().timestamp;

        // Use TimeSpec::Commit for exact commit lookup
        let found = repo
            .find_commit_at_time(&TimeSpec::Commit(commit1.clone()))
            .unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().hash, commit1);

        // Ensure we can find a commit at a given timestamp
        let found = repo.find_commit_at_time(&TimeSpec::Timestamp(ts1)).unwrap();
        assert!(found.is_some());

        // Far past should return None
        let found = repo.find_commit_at_time(&TimeSpec::Timestamp(0)).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_search_at_time() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("a", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.update("a", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        repo.commit("V2").unwrap();

        // Search at first commit (using commit-based lookup)
        let results = repo
            .search_at_time(&[1.0, 0.0, 0.0, 0.0], &TimeSpec::Commit(commit1), 1)
            .unwrap();

        // Should find vector with high similarity (it was [1,0,0,0] at that time)
        assert!(results[0].similarity > 0.99);
    }

    #[test]
    fn test_vector_history() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.commit("Added").unwrap();

        repo.update("vec1", &[0.5, 0.5, 0.0, 0.0]).unwrap();
        repo.commit("Modified").unwrap();

        repo.update("vec1", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        repo.commit("Modified again").unwrap();

        let history = repo.vector_history("vec1");

        assert_eq!(history.len(), 3);
        assert_eq!(history[0].change_type, ChangeType::Added);
        assert_eq!(history[1].change_type, ChangeType::Modified);
        assert_eq!(history[2].change_type, ChangeType::Modified);

        // Check similarity tracking
        assert!(history[1].similarity_to_previous.is_some());
        assert!(history[2].similarity_to_previous.is_some());
    }

    #[test]
    fn test_get_at_time() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.update("vec1", &[9.0, 8.0, 7.0, 6.0]).unwrap();
        repo.commit("V2").unwrap();

        // Get at commit1 (using commit-based lookup)
        let entry = repo
            .get_at_time("vec1", &TimeSpec::Commit(commit1))
            .unwrap();
        assert_eq!(entry.vector, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compare_at_times() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.update("vec1", &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let commit2 = repo.commit("V2").unwrap();

        // Use commit-based time specs for precise comparison
        let diff = repo
            .compare_at_times(
                "vec1",
                &TimeSpec::Commit(commit1),
                &TimeSpec::Commit(commit2),
            )
            .unwrap();

        assert_eq!(diff.change_type, ChangeType::Modified);
        assert!(diff.similarity.unwrap() < 0.1); // Orthogonal vectors
    }

    #[test]
    fn test_changes_between_times() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("V1").unwrap();

        repo.add("vec2", &[0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.update("vec1", &[0.5, 0.5, 0.0, 0.0]).unwrap();
        let commit2 = repo.commit("V2").unwrap();

        // Use commit-based time specs
        let range = TimeRange {
            start: TimeSpec::Commit(commit1),
            end: TimeSpec::Commit(commit2),
        };

        let changes = repo.changes_between(&range).unwrap();

        let added: Vec<_> = changes
            .iter()
            .filter(|d| d.change_type == ChangeType::Added)
            .collect();
        let modified: Vec<_> = changes
            .iter()
            .filter(|d| d.change_type == ChangeType::Modified)
            .collect();

        assert_eq!(added.len(), 1);
        assert_eq!(modified.len(), 1);
    }

    #[test]
    fn test_commits_in_range() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 0.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.commit("V1").unwrap();

        repo.add("vec2", &[0.0, 1.0, 0.0, 0.0], HashMap::new())
            .unwrap();
        repo.commit("V2").unwrap();

        repo.add("vec3", &[0.0, 0.0, 1.0, 0.0], HashMap::new())
            .unwrap();
        repo.commit("V3").unwrap();

        // Get all commits (use wide timestamp range)
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let range = TimeRange {
            start: TimeSpec::Timestamp(now - 10),
            end: TimeSpec::Timestamp(now + 10),
        };
        let commits = repo.commits_in_range(&range).unwrap();
        assert_eq!(commits.len(), 3);
    }

    #[test]
    fn test_time_travel_with_deletion() {
        let mut repo = VectorRepo::new("test", 4);

        repo.add("vec1", &[1.0, 2.0, 3.0, 4.0], HashMap::new())
            .unwrap();
        let commit1 = repo.commit("Added").unwrap();

        repo.delete("vec1").unwrap();
        let commit2 = repo.commit("Deleted").unwrap();

        // Can still retrieve at commit1 using commit-based lookup
        let entry = repo
            .get_at_time("vec1", &TimeSpec::Commit(commit1))
            .unwrap();
        assert_eq!(entry.vector, vec![1.0, 2.0, 3.0, 4.0]);

        // Cannot retrieve at commit2 (deleted)
        let result = repo.get_at_time("vec1", &TimeSpec::Commit(commit2));
        assert!(result.is_err());

        // History shows deletion
        let history = repo.vector_history("vec1");
        assert_eq!(history.len(), 2);
        assert_eq!(history[1].change_type, ChangeType::Deleted);
    }
}
