//! Experimental Triage Report
//!
//! Generates actionable triage recommendations from module_audit results,
//! producing a formatted report with promotion/archive/delete decisions.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::triage_report::{TriageReporter, TriageConfig};
//!
//! let reporter = TriageReporter::new(TriageConfig::default());
//! let report = reporter.generate();
//! println!("{}", report.markdown);
//! ```

use serde::{Deserialize, Serialize};
use crate::services::module_audit::{ModuleAuditor, AuditReport, RecommendedAction};

/// Triage configuration.
#[derive(Debug, Clone)]
pub struct TriageConfig {
    /// Minimum test density for promotion (tests per 100 LOC).
    pub min_promote_density: f32,
    /// Maximum LOC target for experimental/.
    pub target_loc: usize,
}

impl Default for TriageConfig {
    fn default() -> Self { Self { min_promote_density: 1.0, target_loc: 15_000 } }
}

/// Formatted triage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageResult {
    pub markdown: String,
    pub promote: Vec<String>,
    pub archive: Vec<String>,
    pub delete: Vec<String>,
    pub improve: Vec<String>,
    pub current_loc: usize,
    pub projected_loc: usize,
    pub loc_reduction_pct: f32,
}

/// Triage report generator.
pub struct TriageReporter { config: TriageConfig }

impl TriageReporter {
    pub fn new(config: TriageConfig) -> Self { Self { config } }

    /// Generate a triage report from the default module auditor.
    pub fn generate(&self) -> TriageResult {
        let auditor = ModuleAuditor::with_defaults();
        let audit = auditor.audit();
        self.format_report(&audit)
    }

    /// Format an existing audit report.
    pub fn format_report(&self, audit: &AuditReport) -> TriageResult {
        let mut promote = Vec::new();
        let mut archive = Vec::new();
        let mut delete = Vec::new();
        let mut improve = Vec::new();
        let mut reduction = 0usize;

        for rec in &audit.recommendations {
            match rec.recommended_action {
                RecommendedAction::Promote => promote.push(rec.module.clone()),
                RecommendedAction::Archive => { archive.push(rec.module.clone()); reduction += audit.total_loc / audit.total_modules.max(1); }
                RecommendedAction::Delete => { delete.push(rec.module.clone()); reduction += audit.total_loc / audit.total_modules.max(1); }
                RecommendedAction::Improve => improve.push(rec.module.clone()),
            }
        }

        let projected = audit.total_loc.saturating_sub(reduction);
        let pct = if audit.total_loc > 0 { reduction as f32 / audit.total_loc as f32 * 100.0 } else { 0.0 };

        let mut md = String::new();
        md.push_str("# Experimental Module Triage Report\n\n");
        md.push_str(&format!("**Current**: {} modules, {} LOC\n", audit.total_modules, audit.total_loc));
        md.push_str(&format!("**Target**: <{} LOC\n", self.config.target_loc));
        md.push_str(&format!("**Projected**: {} LOC ({:.0}% reduction)\n\n", projected, pct));

        if !promote.is_empty() {
            md.push_str("## ✅ Promote to Stable\n");
            for m in &promote { md.push_str(&format!("- `{m}`\n")); }
            md.push('\n');
        }
        if !improve.is_empty() {
            md.push_str("## 🔨 Improve (Needs Work)\n");
            for m in &improve { md.push_str(&format!("- `{m}`\n")); }
            md.push('\n');
        }
        if !archive.is_empty() {
            md.push_str("## 📦 Archive as Reference\n");
            for m in &archive { md.push_str(&format!("- `{m}`\n")); }
            md.push('\n');
        }
        if !delete.is_empty() {
            md.push_str("## 🗑️ Delete (Scaffolding)\n");
            for m in &delete { md.push_str(&format!("- `{m}`\n")); }
            md.push('\n');
        }

        TriageResult { markdown: md, promote, archive, delete, improve, current_loc: audit.total_loc, projected_loc: projected, loc_reduction_pct: pct }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate() {
        let reporter = TriageReporter::new(TriageConfig::default());
        let result = reporter.generate();
        assert!(result.markdown.contains("Triage Report"));
        assert!(!result.promote.is_empty() || !result.delete.is_empty());
        assert!(result.current_loc > 0);
    }

    #[test]
    fn test_reduction() {
        let reporter = TriageReporter::new(TriageConfig::default());
        let result = reporter.generate();
        assert!(result.projected_loc <= result.current_loc);
    }

    #[test]
    fn test_categories_present() {
        let reporter = TriageReporter::new(TriageConfig::default());
        let result = reporter.generate();
        let total = result.promote.len() + result.archive.len() + result.delete.len() + result.improve.len();
        assert!(total >= 5);
    }
}
