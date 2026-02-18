//! Experimental Module Audit System
//!
//! Classifies experimental modules by maturity level and generates
//! promotion/archive/delete recommendations based on code metrics.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::module_audit::{
//!     ModuleAuditor, ModuleInfo, MaturityLevel, AuditReport,
//! };
//!
//! let mut auditor = ModuleAuditor::new();
//! auditor.register_module(ModuleInfo::new("gpu", 3653, 41, MaturityLevel::Scaffolding));
//! auditor.register_module(ModuleInfo::new("clustering", 1200, 20, MaturityLevel::Functional));
//!
//! let report = auditor.audit();
//! for rec in &report.recommendations {
//!     println!("{}: {} → {}", rec.module, rec.current_level, rec.recommended_action);
//! }
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ── Maturity Level ───────────────────────────────────────────────────────────

/// Maturity classification for experimental modules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MaturityLevel {
    /// Just type stubs and placeholder implementations.
    Scaffolding,
    /// Basic functionality works but not production-ready.
    Prototype,
    /// Core functionality complete with tests.
    Functional,
    /// Ready for promotion to beta/stable.
    Mature,
}

impl std::fmt::Display for MaturityLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scaffolding => write!(f, "scaffolding"),
            Self::Prototype => write!(f, "prototype"),
            Self::Functional => write!(f, "functional"),
            Self::Mature => write!(f, "mature"),
        }
    }
}

// ── Recommended Action ───────────────────────────────────────────────────────

/// Recommended action for a module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecommendedAction {
    /// Promote to stable API.
    Promote,
    /// Keep in experimental with improvements.
    Improve,
    /// Archive as reference implementation.
    Archive,
    /// Delete (pure scaffolding with no value).
    Delete,
}

impl std::fmt::Display for RecommendedAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Promote => write!(f, "PROMOTE"),
            Self::Improve => write!(f, "IMPROVE"),
            Self::Archive => write!(f, "ARCHIVE"),
            Self::Delete => write!(f, "DELETE"),
        }
    }
}

// ── Module Info ──────────────────────────────────────────────────────────────

/// Information about an experimental module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleInfo {
    /// Module name.
    pub name: String,
    /// Lines of code.
    pub loc: usize,
    /// Number of tests.
    pub test_count: usize,
    /// Current maturity level.
    pub maturity: MaturityLevel,
    /// Whether it has external dependencies.
    pub has_external_deps: bool,
    /// Whether it duplicates existing functionality.
    pub is_duplicate: bool,
    /// Usage count (references from other modules).
    pub usage_count: usize,
    /// Last modified timestamp.
    pub last_modified: Option<u64>,
}

impl ModuleInfo {
    /// Create a new module info.
    pub fn new(name: &str, loc: usize, test_count: usize, maturity: MaturityLevel) -> Self {
        Self {
            name: name.into(),
            loc,
            test_count,
            maturity,
            has_external_deps: false,
            is_duplicate: false,
            usage_count: 0,
            last_modified: None,
        }
    }

    /// Set external dependency flag.
    #[must_use]
    pub fn with_external_deps(mut self, has: bool) -> Self {
        self.has_external_deps = has;
        self
    }

    /// Set usage count.
    #[must_use]
    pub fn with_usage(mut self, count: usize) -> Self {
        self.usage_count = count;
        self
    }

    /// Test density (tests per 100 LOC).
    pub fn test_density(&self) -> f32 {
        if self.loc == 0 { return 0.0; }
        (self.test_count as f32 / self.loc as f32) * 100.0
    }
}

// ── Audit Recommendation ─────────────────────────────────────────────────────

/// A specific recommendation for a module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditRecommendation {
    /// Module name.
    pub module: String,
    /// Current maturity level.
    pub current_level: MaturityLevel,
    /// Recommended action.
    pub recommended_action: RecommendedAction,
    /// Rationale.
    pub rationale: String,
    /// Priority (1 = highest).
    pub priority: u32,
}

// ── Audit Report ─────────────────────────────────────────────────────────────

/// Complete audit report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Total modules audited.
    pub total_modules: usize,
    /// Total LOC in experimental.
    pub total_loc: usize,
    /// Recommendations.
    pub recommendations: Vec<AuditRecommendation>,
    /// LOC to be removed if recommendations followed.
    pub loc_reduction: usize,
    /// Summary by action.
    pub summary: HashMap<String, usize>,
}

// ── Module Auditor ───────────────────────────────────────────────────────────

/// Audits experimental modules for maturity and recommends actions.
pub struct ModuleAuditor {
    modules: Vec<ModuleInfo>,
}

impl ModuleAuditor {
    /// Create a new auditor.
    pub fn new() -> Self {
        Self { modules: Vec::new() }
    }

    /// Create with the default Needle experimental modules pre-loaded.
    pub fn with_defaults() -> Self {
        let mut auditor = Self::new();
        for (name, loc, tests, maturity) in DEFAULT_MODULES {
            auditor.register_module(ModuleInfo::new(name, *loc, *tests, *maturity));
        }
        auditor
    }

    /// Register a module for audit.
    pub fn register_module(&mut self, info: ModuleInfo) {
        self.modules.push(info);
    }

    /// Run the audit and generate recommendations.
    pub fn audit(&self) -> AuditReport {
        let mut recommendations = Vec::new();
        let mut loc_reduction = 0;
        let mut summary: HashMap<String, usize> = HashMap::new();

        for module in &self.modules {
            let action = Self::recommend(module);
            let rationale = Self::rationale(module, action);
            let priority = match action {
                RecommendedAction::Delete => 1,
                RecommendedAction::Archive => 2,
                RecommendedAction::Improve => 3,
                RecommendedAction::Promote => 4,
            };

            if matches!(action, RecommendedAction::Delete | RecommendedAction::Archive) {
                loc_reduction += module.loc;
            }

            *summary.entry(action.to_string()).or_default() += 1;

            recommendations.push(AuditRecommendation {
                module: module.name.clone(),
                current_level: module.maturity,
                recommended_action: action,
                rationale,
                priority,
            });
        }

        recommendations.sort_by_key(|r| r.priority);

        AuditReport {
            total_modules: self.modules.len(),
            total_loc: self.modules.iter().map(|m| m.loc).sum(),
            recommendations,
            loc_reduction,
            summary,
        }
    }

    /// Module count.
    pub fn module_count(&self) -> usize {
        self.modules.len()
    }

    fn recommend(module: &ModuleInfo) -> RecommendedAction {
        match module.maturity {
            MaturityLevel::Scaffolding => {
                if module.test_count == 0 && module.usage_count == 0 {
                    RecommendedAction::Delete
                } else {
                    RecommendedAction::Archive
                }
            }
            MaturityLevel::Prototype => {
                if module.test_density() < 0.5 {
                    RecommendedAction::Archive
                } else {
                    RecommendedAction::Improve
                }
            }
            MaturityLevel::Functional => {
                if module.test_density() >= 1.0 && !module.is_duplicate {
                    RecommendedAction::Promote
                } else {
                    RecommendedAction::Improve
                }
            }
            MaturityLevel::Mature => RecommendedAction::Promote,
        }
    }

    fn rationale(module: &ModuleInfo, action: RecommendedAction) -> String {
        match action {
            RecommendedAction::Delete => format!(
                "{} is scaffolding ({} LOC, {} tests, {} usage). No value to retain.",
                module.name, module.loc, module.test_count, module.usage_count
            ),
            RecommendedAction::Archive => format!(
                "{} is {}, low test density ({:.1}%). Archive as reference.",
                module.name, module.maturity, module.test_density()
            ),
            RecommendedAction::Improve => format!(
                "{} is {} with {:.1}% test density. Needs more tests before promotion.",
                module.name, module.maturity, module.test_density()
            ),
            RecommendedAction::Promote => format!(
                "{} is {} with good test coverage ({:.1}%). Ready for stable API.",
                module.name, module.maturity, module.test_density()
            ),
        }
    }
}

impl Default for ModuleAuditor {
    fn default() -> Self { Self::new() }
}

const DEFAULT_MODULES: &[(&str, usize, usize, MaturityLevel)] = &[
    ("gpu", 3653, 41, MaturityLevel::Scaffolding),
    ("cloud_control", 2934, 41, MaturityLevel::Scaffolding),
    ("vector_streaming", 2930, 17, MaturityLevel::Prototype),
    ("edge_runtime", 2204, 27, MaturityLevel::Functional),
    ("agentic_memory", 1801, 25, MaturityLevel::Functional),
    ("knowledge_graph", 1721, 22, MaturityLevel::Functional),
    ("clustering", 1200, 20, MaturityLevel::Functional),
    ("temporal", 1100, 18, MaturityLevel::Functional),
    ("crdt", 1050, 15, MaturityLevel::Functional),
    ("adaptive_index", 980, 12, MaturityLevel::Prototype),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_with_defaults() {
        let auditor = ModuleAuditor::with_defaults();
        let report = auditor.audit();
        assert!(report.total_modules >= 10);
        assert!(report.total_loc > 10000);
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_scaffolding_gets_deleted() {
        let mut auditor = ModuleAuditor::new();
        auditor.register_module(ModuleInfo::new("stub", 100, 0, MaturityLevel::Scaffolding));
        let report = auditor.audit();
        assert_eq!(report.recommendations[0].recommended_action, RecommendedAction::Delete);
    }

    #[test]
    fn test_mature_gets_promoted() {
        let mut auditor = ModuleAuditor::new();
        auditor.register_module(ModuleInfo::new("ready", 500, 20, MaturityLevel::Mature));
        let report = auditor.audit();
        assert_eq!(report.recommendations[0].recommended_action, RecommendedAction::Promote);
    }

    #[test]
    fn test_loc_reduction() {
        let mut auditor = ModuleAuditor::new();
        auditor.register_module(ModuleInfo::new("a", 1000, 0, MaturityLevel::Scaffolding));
        auditor.register_module(ModuleInfo::new("b", 500, 20, MaturityLevel::Mature));
        let report = auditor.audit();
        assert_eq!(report.loc_reduction, 1000);
    }

    #[test]
    fn test_test_density() {
        let m = ModuleInfo::new("test", 1000, 10, MaturityLevel::Functional);
        assert!((m.test_density() - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_summary() {
        let auditor = ModuleAuditor::with_defaults();
        let report = auditor.audit();
        assert!(report.summary.values().sum::<usize>() == report.total_modules);
    }
}
