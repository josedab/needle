#![allow(clippy::unwrap_used)]
//! Unwrap Audit Tool
//!
//! Programmatic scanner that inventories `unwrap()` calls, classifies them by
//! risk level, and generates prioritized fix recommendations.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::unwrap_audit::{UnwrapAuditor, AuditConfig, RiskLevel};
//!
//! let auditor = UnwrapAuditor::new(AuditConfig::default());
//! let report = auditor.scan_module("collection/mod.rs", SAMPLE_CODE);
//! println!("Found {} unwraps ({} critical)", report.total, report.critical);
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Risk level for an unwrap call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskLevel {
    /// In production hot path — must fix.
    Critical,
    /// In error handling or setup — should fix.
    High,
    /// In rarely-reached code — can fix.
    Medium,
    /// In test/example code — acceptable.
    Low,
}

/// A single unwrap occurrence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnwrapOccurrence {
    pub file: String,
    pub line: usize,
    pub context: String,
    pub risk: RiskLevel,
    pub suggestion: String,
}

/// Audit configuration.
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Files considered critical (highest risk).
    pub critical_files: Vec<String>,
    /// Patterns that indicate test code.
    pub test_patterns: Vec<String>,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            critical_files: vec![
                "collection/mod.rs".into(), "database/mod.rs".into(),
                "database/collection_ref.rs".into(), "server.rs".into(),
                "storage.rs".into(), "error.rs".into(),
            ],
            test_patterns: vec!["#[test]".into(), "#[cfg(test)]".into(), "mod tests".into()],
        }
    }
}

/// Module scan report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleReport {
    pub file: String,
    pub total: usize,
    pub critical: usize,
    pub high: usize,
    pub medium: usize,
    pub low: usize,
    pub occurrences: Vec<UnwrapOccurrence>,
}

/// Full codebase audit report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub total_unwraps: usize,
    pub total_files: usize,
    pub by_risk: HashMap<String, usize>,
    pub top_files: Vec<(String, usize)>,
    pub fix_priority: Vec<String>,
    pub estimated_effort_hours: f32,
}

/// Unwrap auditor.
pub struct UnwrapAuditor {
    config: AuditConfig,
}

impl UnwrapAuditor {
    pub fn new(config: AuditConfig) -> Self { Self { config } }

    /// Scan a single module's source code.
    pub fn scan_module(&self, file: &str, source: &str) -> ModuleReport {
        let mut occurrences = Vec::new();
        let mut in_test = false;

        for (line_num, line) in source.lines().enumerate() {
            if self.config.test_patterns.iter().any(|p| line.contains(p)) { in_test = true; }
            if line.contains("unwrap()") {
                let risk = if in_test {
                    RiskLevel::Low
                } else if self.config.critical_files.iter().any(|f| file.contains(f)) {
                    RiskLevel::Critical
                } else if line.contains("lock()") || line.contains("read()") || line.contains("write()") {
                    RiskLevel::High
                } else {
                    RiskLevel::Medium
                };

                let suggestion = if line.contains(".get(") {
                    "Use .ok_or_else(|| NeedleError::NotFound(...))? instead".into()
                } else if line.contains(".lock(") {
                    "Use .map_err(|_| NeedleError::LockError)? instead".into()
                } else if line.contains(".parse(") {
                    "Use .map_err(|e| NeedleError::InvalidInput(e.to_string()))? instead".into()
                } else {
                    "Replace .unwrap() with ? and appropriate error conversion".into()
                };

                occurrences.push(UnwrapOccurrence {
                    file: file.into(), line: line_num + 1,
                    context: line.trim().to_string(), risk, suggestion,
                });
            }
        }

        let critical = occurrences.iter().filter(|o| o.risk == RiskLevel::Critical).count();
        let high = occurrences.iter().filter(|o| o.risk == RiskLevel::High).count();
        let medium = occurrences.iter().filter(|o| o.risk == RiskLevel::Medium).count();
        let low = occurrences.iter().filter(|o| o.risk == RiskLevel::Low).count();

        ModuleReport { file: file.into(), total: occurrences.len(), critical, high, medium, low, occurrences }
    }

    /// Generate a full audit report from multiple module reports.
    pub fn aggregate(&self, reports: &[ModuleReport]) -> AuditReport {
        let total_unwraps: usize = reports.iter().map(|r| r.total).sum();
        let mut by_risk = HashMap::new();
        by_risk.insert("critical".into(), reports.iter().map(|r| r.critical).sum());
        by_risk.insert("high".into(), reports.iter().map(|r| r.high).sum());
        by_risk.insert("medium".into(), reports.iter().map(|r| r.medium).sum());
        by_risk.insert("low".into(), reports.iter().map(|r| r.low).sum());

        let mut top_files: Vec<(String, usize)> = reports.iter()
            .filter(|r| r.total > 0)
            .map(|r| (r.file.clone(), r.total))
            .collect();
        top_files.sort_by(|a, b| b.1.cmp(&a.1));
        top_files.truncate(10);

        let fix_priority: Vec<String> = reports.iter()
            .filter(|r| r.critical > 0)
            .map(|r| format!("{}: {} critical unwraps", r.file, r.critical))
            .collect();

        // ~5 min per unwrap conversion on average
        let estimated_effort = total_unwraps as f32 * 5.0 / 60.0;

        AuditReport { total_unwraps, total_files: reports.len(), by_risk, top_files, fix_priority, estimated_effort_hours: estimated_effort }
    }
}

impl Default for UnwrapAuditor { fn default() -> Self { Self::new(AuditConfig::default()) } }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_detects_unwrap() {
        let auditor = UnwrapAuditor::default();
        let source = "fn foo() {\n    let x = some_option.unwrap();\n}\n";
        let report = auditor.scan_module("src/services/test.rs", source);
        assert_eq!(report.total, 1);
        assert_eq!(report.medium, 1);
    }

    #[test]
    fn test_critical_file() {
        let auditor = UnwrapAuditor::default();
        let source = "let v = map.get(\"k\").unwrap();\n";
        let report = auditor.scan_module("collection/mod.rs", source);
        assert_eq!(report.critical, 1);
    }

    #[test]
    fn test_test_code_is_low() {
        let auditor = UnwrapAuditor::default();
        let source = "#[cfg(test)]\nmod tests {\nfn t() { x.unwrap(); }\n}\n";
        let report = auditor.scan_module("some.rs", source);
        assert_eq!(report.low, 1);
    }

    #[test]
    fn test_suggestions() {
        let auditor = UnwrapAuditor::default();
        let source = "map.get(\"k\").unwrap()\n";
        let report = auditor.scan_module("x.rs", source);
        assert!(report.occurrences[0].suggestion.contains("ok_or_else"));
    }

    #[test]
    fn test_aggregate() {
        let auditor = UnwrapAuditor::default();
        let r1 = auditor.scan_module("collection/mod.rs", "x.unwrap();\ny.unwrap();\n");
        let r2 = auditor.scan_module("other.rs", "z.unwrap();\n");
        let report = auditor.aggregate(&[r1, r2]);
        assert_eq!(report.total_unwraps, 3);
        assert_eq!(report.total_files, 2);
        assert!(report.estimated_effort_hours > 0.0);
    }
}
