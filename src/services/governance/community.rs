//! Community Bootstrap Infrastructure
//!
//! Contributor tracking, project showcase registry, and good-first-issue
//! generator for building an open-source community around Needle.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::community::{
//!     CommunityManager, Contributor, ShowcaseEntry, IssueTemplate,
//!     IssueDifficulty,
//! };
//!
//! let mut mgr = CommunityManager::new();
//!
//! // Track contributors
//! mgr.add_contributor(Contributor::new("alice", "Alice Smith"));
//!
//! // Generate good-first-issues
//! let issues = mgr.generate_starter_issues();
//! assert!(!issues.is_empty());
//! ```

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

// ── Contributor ──────────────────────────────────────────────────────────────

/// A project contributor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contributor {
    /// GitHub username.
    pub username: String,
    /// Display name.
    pub name: String,
    /// Number of contributions.
    pub contributions: u32,
    /// First contribution date.
    pub first_contribution: u64,
    /// Areas of contribution.
    pub areas: Vec<String>,
    /// Whether they're a maintainer.
    pub maintainer: bool,
}

impl Contributor {
    /// Create a new contributor.
    pub fn new(username: &str, name: &str) -> Self {
        Self {
            username: username.into(),
            name: name.into(),
            contributions: 0,
            first_contribution: now_secs(),
            areas: Vec::new(),
            maintainer: false,
        }
    }

    /// Record a contribution.
    #[must_use]
    pub fn with_contribution(mut self, area: &str) -> Self {
        self.contributions += 1;
        if !self.areas.contains(&area.to_string()) {
            self.areas.push(area.into());
        }
        self
    }
}

// ── Showcase Entry ───────────────────────────────────────────────────────────

/// A "built with Needle" showcase entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShowcaseEntry {
    /// Project name.
    pub name: String,
    /// Description.
    pub description: String,
    /// URL.
    pub url: String,
    /// Author.
    pub author: String,
    /// Tags.
    pub tags: Vec<String>,
    /// Submission date.
    pub submitted_at: u64,
    /// Whether approved.
    pub approved: bool,
}

// ── Issue Template ───────────────────────────────────────────────────────────

/// A generated issue template.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueTemplate {
    /// Issue title.
    pub title: String,
    /// Issue body.
    pub body: String,
    /// Labels.
    pub labels: Vec<String>,
    /// Difficulty.
    pub difficulty: IssueDifficulty,
    /// Estimated time.
    pub estimated_hours: f32,
    /// Area of codebase.
    pub area: String,
}

/// Issue difficulty level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueDifficulty {
    GoodFirstIssue,
    Beginner,
    Intermediate,
    Advanced,
}

impl std::fmt::Display for IssueDifficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GoodFirstIssue => write!(f, "good first issue"),
            Self::Beginner => write!(f, "beginner"),
            Self::Intermediate => write!(f, "intermediate"),
            Self::Advanced => write!(f, "advanced"),
        }
    }
}

// ── Community Stats ──────────────────────────────────────────────────────────

/// Community health statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommunityStats {
    /// Total contributors.
    pub total_contributors: usize,
    /// Active contributors (contributed in last 90 days).
    pub active_contributors: usize,
    /// Total showcase entries.
    pub showcase_entries: usize,
    /// Open starter issues.
    pub open_starter_issues: usize,
    /// Total contributions.
    pub total_contributions: u32,
}

// ── Community Manager ────────────────────────────────────────────────────────

/// Manages community infrastructure.
pub struct CommunityManager {
    contributors: HashMap<String, Contributor>,
    showcase: Vec<ShowcaseEntry>,
    issues: Vec<IssueTemplate>,
}

impl CommunityManager {
    /// Create a new community manager.
    pub fn new() -> Self {
        Self {
            contributors: HashMap::new(),
            showcase: Vec::new(),
            issues: Vec::new(),
        }
    }

    /// Add a contributor.
    pub fn add_contributor(&mut self, contributor: Contributor) {
        self.contributors.insert(contributor.username.clone(), contributor);
    }

    /// Record a contribution for an existing contributor.
    pub fn record_contribution(&mut self, username: &str, area: &str) {
        if let Some(c) = self.contributors.get_mut(username) {
            c.contributions += 1;
            if !c.areas.contains(&area.to_string()) {
                c.areas.push(area.into());
            }
        }
    }

    /// Submit a showcase entry.
    pub fn submit_showcase(&mut self, entry: ShowcaseEntry) {
        self.showcase.push(entry);
    }

    /// Approve a showcase entry.
    pub fn approve_showcase(&mut self, index: usize) -> bool {
        if let Some(entry) = self.showcase.get_mut(index) {
            entry.approved = true;
            true
        } else {
            false
        }
    }

    /// Generate starter issues for new contributors.
    pub fn generate_starter_issues(&self) -> Vec<IssueTemplate> {
        vec![
            IssueTemplate {
                title: "Add doc comments to src/services/vector_pipeline.rs".into(),
                body: "Add `///` documentation comments to all public types and methods in the vector pipeline module. Follow existing patterns in src/collection/mod.rs.".into(),
                labels: vec!["good first issue".into(), "documentation".into()],
                difficulty: IssueDifficulty::GoodFirstIssue,
                estimated_hours: 2.0,
                area: "documentation".into(),
            },
            IssueTemplate {
                title: "Add unit tests for src/services/drift_monitor.rs edge cases".into(),
                body: "Add tests for: empty baseline, single-dimension vectors, NaN handling, very large vectors. Use existing test patterns.".into(),
                labels: vec!["good first issue".into(), "testing".into()],
                difficulty: IssueDifficulty::GoodFirstIssue,
                estimated_hours: 3.0,
                area: "testing".into(),
            },
            IssueTemplate {
                title: "Replace unwrap() with ? in src/services/ modules".into(),
                body: "Find and replace unwrap() calls in production code paths with proper ? error propagation. Tests can keep unwrap().".into(),
                labels: vec!["good first issue".into(), "code quality".into()],
                difficulty: IssueDifficulty::Beginner,
                estimated_hours: 4.0,
                area: "error-handling".into(),
            },
            IssueTemplate {
                title: "Add example: RAG chatbot with semantic cache".into(),
                body: "Create examples/rag_with_cache.rs showing how to combine TextVectorCollection with SemanticCache for a cost-efficient RAG pipeline.".into(),
                labels: vec!["good first issue".into(), "examples".into()],
                difficulty: IssueDifficulty::Beginner,
                estimated_hours: 4.0,
                area: "examples".into(),
            },
            IssueTemplate {
                title: "Add CLI command: needle benchmark".into(),
                body: "Add a `benchmark` subcommand to the CLI that runs the ann_benchmark module against a user-provided dataset and prints recall/QPS.".into(),
                labels: vec!["enhancement".into(), "cli".into()],
                difficulty: IssueDifficulty::Intermediate,
                estimated_hours: 6.0,
                area: "cli".into(),
            },
        ]
    }

    /// Get community statistics.
    pub fn stats(&self) -> CommunityStats {
        let total_contributions: u32 = self.contributors.values().map(|c| c.contributions).sum();
        CommunityStats {
            total_contributors: self.contributors.len(),
            active_contributors: self.contributors.len(), // simplified
            showcase_entries: self.showcase.iter().filter(|e| e.approved).count(),
            open_starter_issues: self.generate_starter_issues().len(),
            total_contributions,
        }
    }

    /// Get all contributors sorted by contributions.
    pub fn leaderboard(&self) -> Vec<&Contributor> {
        let mut sorted: Vec<_> = self.contributors.values().collect();
        sorted.sort_by(|a, b| b.contributions.cmp(&a.contributions));
        sorted
    }

    /// Get approved showcase entries.
    pub fn approved_showcase(&self) -> Vec<&ShowcaseEntry> {
        self.showcase.iter().filter(|e| e.approved).collect()
    }

    /// Contributor count.
    pub fn contributor_count(&self) -> usize {
        self.contributors.len()
    }
}

impl Default for CommunityManager {
    fn default() -> Self { Self::new() }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contributor_tracking() {
        let mut mgr = CommunityManager::new();
        mgr.add_contributor(Contributor::new("alice", "Alice"));
        mgr.record_contribution("alice", "docs");
        mgr.record_contribution("alice", "tests");

        let leaderboard = mgr.leaderboard();
        assert_eq!(leaderboard[0].contributions, 2);
        assert_eq!(leaderboard[0].areas.len(), 2);
    }

    #[test]
    fn test_showcase() {
        let mut mgr = CommunityManager::new();
        mgr.submit_showcase(ShowcaseEntry {
            name: "MyApp".into(), description: "Cool app".into(),
            url: "https://example.com".into(), author: "bob".into(),
            tags: vec!["rag".into()], submitted_at: now_secs(), approved: false,
        });
        assert!(mgr.approved_showcase().is_empty());
        mgr.approve_showcase(0);
        assert_eq!(mgr.approved_showcase().len(), 1);
    }

    #[test]
    fn test_starter_issues() {
        let mgr = CommunityManager::new();
        let issues = mgr.generate_starter_issues();
        assert!(issues.len() >= 5);
        assert!(issues.iter().any(|i| i.labels.contains(&"good first issue".to_string())));
    }

    #[test]
    fn test_stats() {
        let mut mgr = CommunityManager::new();
        mgr.add_contributor(Contributor::new("a", "A").with_contribution("core"));
        mgr.add_contributor(Contributor::new("b", "B"));
        let stats = mgr.stats();
        assert_eq!(stats.total_contributors, 2);
        assert_eq!(stats.total_contributions, 1);
    }

    #[test]
    fn test_difficulty_display() {
        assert_eq!(format!("{}", IssueDifficulty::GoodFirstIssue), "good first issue");
    }
}
