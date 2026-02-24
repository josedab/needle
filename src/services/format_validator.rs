//! File Format Specification
//!
//! Byte-level `.needle` format spec types, validator, version migration,
//! and backwards compatibility checks. Enables third-party tooling and
//! stable format guarantees for v1.0.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::format_validator::{
//!     FormatSpec, FormatVersion, ValidationResult, PageType, FormatMigrator,
//! };
//!
//! let spec = FormatSpec::v1();
//! println!("Magic: {:?}", spec.magic_bytes);
//! println!("Version: {}", spec.version);
//!
//! // Validate a file header
//! let header = vec![0x4E, 0x44, 0x4C, 0x45, 0x01, 0x00]; // "NDLE" + version 1.0
//! let result = FormatSpec::validate_header(&header);
//! assert!(result.valid);
//! ```

use serde::{Deserialize, Serialize};

/// Format version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
}

impl FormatVersion {
    pub fn new(major: u16, minor: u16) -> Self { Self { major, minor } }

    pub fn is_compatible_with(&self, other: &FormatVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

impl std::fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Page types in the .needle file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageType {
    Header,
    Index,
    Vector,
    Metadata,
    Wal,
}

impl std::fmt::Display for PageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Header => write!(f, "header"),
            Self::Index => write!(f, "index"),
            Self::Vector => write!(f, "vector"),
            Self::Metadata => write!(f, "metadata"),
            Self::Wal => write!(f, "wal"),
        }
    }
}

/// A page descriptor in the format spec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageDescriptor {
    pub page_type: PageType,
    pub offset: u64,
    pub size: u64,
    pub checksum: Option<String>,
}

/// The complete .needle format specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatSpec {
    pub magic_bytes: [u8; 4],
    pub version: FormatVersion,
    pub header_size: u64,
    pub page_alignment: u64,
    pub checksum_algorithm: String,
    pub pages: Vec<PageDescriptor>,
    pub endianness: String,
    pub max_dimensions: u32,
    pub max_vectors: u64,
    pub max_metadata_bytes: u64,
}

impl FormatSpec {
    /// Current v1 format specification.
    pub fn v1() -> Self {
        Self {
            magic_bytes: [0x4E, 0x44, 0x4C, 0x45], // "NDLE"
            version: FormatVersion::new(1, 0),
            header_size: 4096,
            page_alignment: 4096,
            checksum_algorithm: "sha256".into(),
            pages: vec![
                PageDescriptor { page_type: PageType::Header, offset: 0, size: 4096, checksum: None },
                PageDescriptor { page_type: PageType::Index, offset: 4096, size: 0, checksum: None },
                PageDescriptor { page_type: PageType::Vector, offset: 0, size: 0, checksum: None },
                PageDescriptor { page_type: PageType::Metadata, offset: 0, size: 0, checksum: None },
            ],
            endianness: "little".into(),
            max_dimensions: 65536,
            max_vectors: 1_000_000_000,
            max_metadata_bytes: 1_073_741_824, // 1GB
        }
    }

    /// Validate a file header against the spec.
    pub fn validate_header(header_bytes: &[u8]) -> ValidationResult {
        let mut issues = Vec::new();

        if header_bytes.len() < 6 {
            return ValidationResult {
                valid: false,
                issues: vec!["Header too short (need at least 6 bytes)".into()],
                version_detected: None,
            };
        }

        // Check magic bytes
        let magic = &header_bytes[0..4];
        let expected = [0x4E, 0x44, 0x4C, 0x45];
        if magic != expected {
            issues.push(format!(
                "Invalid magic bytes: got {:02X}{:02X}{:02X}{:02X}, expected 4E444C45",
                magic[0], magic[1], magic[2], magic[3]
            ));
        }

        // Check version
        let version = FormatVersion::new(
            header_bytes[4] as u16,
            header_bytes[5] as u16,
        );

        let current = FormatVersion::new(1, 0);
        if !current.is_compatible_with(&version) {
            issues.push(format!("Version {} not compatible with current {}", version, current));
        }

        ValidationResult {
            valid: issues.is_empty(),
            issues,
            version_detected: Some(version),
        }
    }

    /// Generate the format specification as a documentation string.
    pub fn to_spec_doc(&self) -> String {
        let mut doc = String::new();
        doc.push_str("# Needle File Format Specification\n\n");
        doc.push_str(&format!("Version: {}\n\n", self.version));
        doc.push_str("## Header\n\n");
        doc.push_str(&format!("- Magic bytes: `{:02X} {:02X} {:02X} {:02X}` (\"NDLE\")\n",
            self.magic_bytes[0], self.magic_bytes[1], self.magic_bytes[2], self.magic_bytes[3]));
        doc.push_str(&format!("- Header size: {} bytes\n", self.header_size));
        doc.push_str(&format!("- Page alignment: {} bytes\n", self.page_alignment));
        doc.push_str(&format!("- Checksum: {}\n", self.checksum_algorithm));
        doc.push_str(&format!("- Endianness: {}\n\n", self.endianness));
        doc.push_str("## Limits\n\n");
        doc.push_str(&format!("- Max dimensions: {}\n", self.max_dimensions));
        doc.push_str(&format!("- Max vectors: {}\n", self.max_vectors));
        doc.push_str(&format!("- Max metadata: {} bytes\n\n", self.max_metadata_bytes));
        doc.push_str("## Page Layout\n\n");
        doc.push_str("| Page Type | Offset | Size |\n");
        doc.push_str("|-----------|--------|------|\n");
        for page in &self.pages {
            doc.push_str(&format!("| {} | {} | {} |\n", page.page_type, page.offset, page.size));
        }
        doc
    }
}

/// Result of a format validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub issues: Vec<String>,
    pub version_detected: Option<FormatVersion>,
}

/// Format migration between versions.
pub struct FormatMigrator {
    migrations: Vec<(FormatVersion, FormatVersion, String)>,
}

impl FormatMigrator {
    pub fn new() -> Self {
        Self {
            migrations: vec![
                (FormatVersion::new(1, 0), FormatVersion::new(1, 1),
                 "Add metadata checksum field to header".into()),
            ],
        }
    }

    /// Check if migration is needed between two versions.
    pub fn needs_migration(&self, from: &FormatVersion, to: &FormatVersion) -> bool {
        from < to
    }

    /// Get the migration path between two versions.
    pub fn migration_path(
        &self,
        from: &FormatVersion,
        to: &FormatVersion,
    ) -> Vec<(FormatVersion, FormatVersion, String)> {
        self.migrations
            .iter()
            .filter(|(f, t, _)| f >= from && t <= to)
            .cloned()
            .collect()
    }

    /// List all available migrations.
    pub fn available_migrations(&self) -> &[(FormatVersion, FormatVersion, String)] {
        &self.migrations
    }

    /// Check backwards compatibility between two versions.
    pub fn is_backwards_compatible(from: &FormatVersion, to: &FormatVersion) -> bool {
        from.major == to.major
    }
}

impl Default for FormatMigrator {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_spec() {
        let spec = FormatSpec::v1();
        assert_eq!(spec.magic_bytes, [0x4E, 0x44, 0x4C, 0x45]);
        assert_eq!(spec.version, FormatVersion::new(1, 0));
        assert_eq!(spec.header_size, 4096);
    }

    #[test]
    fn test_validate_valid_header() {
        let header = vec![0x4E, 0x44, 0x4C, 0x45, 0x01, 0x00];
        let result = FormatSpec::validate_header(&header);
        assert!(result.valid);
        assert_eq!(result.version_detected, Some(FormatVersion::new(1, 0)));
    }

    #[test]
    fn test_validate_invalid_magic() {
        let header = vec![0xFF, 0xFF, 0xFF, 0xFF, 0x01, 0x00];
        let result = FormatSpec::validate_header(&header);
        assert!(!result.valid);
        assert!(result.issues[0].contains("Invalid magic bytes"));
    }

    #[test]
    fn test_validate_too_short() {
        let result = FormatSpec::validate_header(&[0x4E, 0x44]);
        assert!(!result.valid);
    }

    #[test]
    fn test_version_compatibility() {
        let v10 = FormatVersion::new(1, 0);
        let v11 = FormatVersion::new(1, 1);
        let v20 = FormatVersion::new(2, 0);

        assert!(v11.is_compatible_with(&v10));
        assert!(!v10.is_compatible_with(&v11));
        assert!(!v20.is_compatible_with(&v10));
    }

    #[test]
    fn test_spec_doc_generation() {
        let spec = FormatSpec::v1();
        let doc = spec.to_spec_doc();
        assert!(doc.contains("Needle File Format"));
        assert!(doc.contains("NDLE"));
        assert!(doc.contains("sha256"));
    }

    #[test]
    fn test_migration_needed() {
        let migrator = FormatMigrator::new();
        assert!(migrator.needs_migration(&FormatVersion::new(1, 0), &FormatVersion::new(1, 1)));
        assert!(!migrator.needs_migration(&FormatVersion::new(1, 1), &FormatVersion::new(1, 0)));
    }

    #[test]
    fn test_backwards_compatibility() {
        assert!(FormatMigrator::is_backwards_compatible(
            &FormatVersion::new(1, 0), &FormatVersion::new(1, 5)
        ));
        assert!(!FormatMigrator::is_backwards_compatible(
            &FormatVersion::new(1, 0), &FormatVersion::new(2, 0)
        ));
    }

    #[test]
    fn test_version_display() {
        assert_eq!(format!("{}", FormatVersion::new(1, 0)), "1.0");
        assert_eq!(format!("{}", FormatVersion::new(2, 3)), "2.3");
    }
}
