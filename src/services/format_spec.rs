//! File Format Specification Engine
//!
//! Generates byte-level documentation for the `.needle` file format,
//! manages version compatibility matrix, and provides format validation.
//!
//! # Example
//!
//! ```rust,no_run
//! use needle::services::format_spec::{
//!     FormatSpec, FormatVersion, FieldSpec, ValidationResult,
//! };
//!
//! let spec = FormatSpec::current();
//! println!("Format: {}", spec.magic);
//! println!("Version: {}", spec.version);
//!
//! for field in &spec.header_fields {
//!     println!("  {} @ offset {}: {} bytes", field.name, field.offset, field.size);
//! }
//!
//! // Validate a file header
//! let header = b"NEEDLE01";
//! let result = spec.validate_magic(header);
//! assert!(result.valid);
//! ```

use serde::{Deserialize, Serialize};

// ── Format Version ───────────────────────────────────────────────────────────

/// File format version.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FormatVersion {
    /// Major version (breaking format changes).
    pub major: u32,
    /// Minor version (backward-compatible additions).
    pub minor: u32,
}

impl FormatVersion {
    /// Create a version.
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Check backward compatibility with another version.
    pub fn is_compatible(&self, other: &FormatVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

impl std::fmt::Display for FormatVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

// ── Field Specification ──────────────────────────────────────────────────────

/// Specification of a single field in the binary format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSpec {
    /// Field name.
    pub name: String,
    /// Byte offset from section start.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Data type.
    pub data_type: FieldType,
    /// Description.
    pub description: String,
    /// Whether this field is optional.
    pub optional: bool,
}

/// Data type of a field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    /// Raw bytes (magic number, etc).
    Bytes,
    /// Unsigned 32-bit integer (little-endian).
    U32Le,
    /// Unsigned 64-bit integer (little-endian).
    U64Le,
    /// 32-bit float (little-endian).
    F32Le,
    /// Variable-length byte array with length prefix.
    VarBytes,
    /// UTF-8 string with length prefix.
    String,
    /// CRC32 checksum.
    Crc32,
}

impl std::fmt::Display for FieldType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bytes => write!(f, "bytes"),
            Self::U32Le => write!(f, "u32le"),
            Self::U64Le => write!(f, "u64le"),
            Self::F32Le => write!(f, "f32le"),
            Self::VarBytes => write!(f, "var_bytes"),
            Self::String => write!(f, "string"),
            Self::Crc32 => write!(f, "crc32"),
        }
    }
}

// ── Section Specification ────────────────────────────────────────────────────

/// A section within the file format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SectionSpec {
    /// Section name.
    pub name: String,
    /// Section description.
    pub description: String,
    /// Fields in this section.
    pub fields: Vec<FieldSpec>,
    /// Whether this section is required.
    pub required: bool,
}

// ── Compatibility Matrix ─────────────────────────────────────────────────────

/// Compatibility entry between format versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityEntry {
    /// Reader version.
    pub reader: FormatVersion,
    /// Writer version.
    pub writer: FormatVersion,
    /// Whether compatible.
    pub compatible: bool,
    /// Notes.
    pub notes: String,
}

// ── Validation Result ────────────────────────────────────────────────────────

/// Result of format validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the data is valid.
    pub valid: bool,
    /// Detected format version.
    pub version: Option<FormatVersion>,
    /// Errors found.
    pub errors: Vec<String>,
    /// Warnings.
    pub warnings: Vec<String>,
}

// ── Format Specification ─────────────────────────────────────────────────────

/// Complete file format specification.
pub struct FormatSpec {
    /// Magic number.
    pub magic: String,
    /// Current format version.
    pub version: FormatVersion,
    /// Header fields.
    pub header_fields: Vec<FieldSpec>,
    /// Sections.
    pub sections: Vec<SectionSpec>,
    /// Compatibility matrix.
    pub compatibility: Vec<CompatibilityEntry>,
}

impl FormatSpec {
    /// Get the current format specification.
    pub fn current() -> Self {
        Self {
            magic: "NEEDLE01".into(),
            version: FormatVersion::new(1, 0),
            header_fields: vec![
                FieldSpec {
                    name: "magic".into(), offset: 0, size: 8,
                    data_type: FieldType::Bytes,
                    description: "File magic number: 'NEEDLE01'".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "version".into(), offset: 8, size: 4,
                    data_type: FieldType::U32Le,
                    description: "Format version (major << 16 | minor)".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "dimensions".into(), offset: 12, size: 4,
                    data_type: FieldType::U32Le,
                    description: "Vector dimensionality".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "vector_count".into(), offset: 16, size: 8,
                    data_type: FieldType::U64Le,
                    description: "Total number of vectors".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "state_offset".into(), offset: 24, size: 8,
                    data_type: FieldType::U64Le,
                    description: "Byte offset to serialized state data".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "header_checksum".into(), offset: 32, size: 4,
                    data_type: FieldType::Crc32,
                    description: "CRC32 of header bytes 0..32".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "state_checksum".into(), offset: 36, size: 4,
                    data_type: FieldType::Crc32,
                    description: "CRC32 of state data".into(),
                    optional: false,
                },
                FieldSpec {
                    name: "reserved".into(), offset: 40, size: 4056,
                    data_type: FieldType::Bytes,
                    description: "Reserved for future use (zero-filled)".into(),
                    optional: true,
                },
            ],
            sections: vec![
                SectionSpec {
                    name: "header".into(),
                    description: "4KB fixed-size header with metadata and checksums".into(),
                    fields: Vec::new(), // fields listed above
                    required: true,
                },
                SectionSpec {
                    name: "state".into(),
                    description: "Serialized database state (collections, vectors, indices, metadata)".into(),
                    fields: vec![FieldSpec {
                        name: "state_data".into(), offset: 4096, size: 0,
                        data_type: FieldType::VarBytes,
                        description: "Bincode/JSON serialized state".into(),
                        optional: false,
                    }],
                    required: true,
                },
            ],
            compatibility: vec![
                CompatibilityEntry {
                    reader: FormatVersion::new(1, 0),
                    writer: FormatVersion::new(1, 0),
                    compatible: true,
                    notes: "Fully compatible".into(),
                },
            ],
        }
    }

    /// Validate a magic number.
    pub fn validate_magic(&self, bytes: &[u8]) -> ValidationResult {
        let mut errors = Vec::new();
        let magic_bytes = self.magic.as_bytes();

        if bytes.len() < magic_bytes.len() {
            errors.push(format!("File too short: {} bytes, need {}", bytes.len(), magic_bytes.len()));
            return ValidationResult { valid: false, version: None, errors, warnings: Vec::new() };
        }

        if &bytes[..magic_bytes.len()] != magic_bytes {
            errors.push(format!("Invalid magic: expected {:?}, got {:?}",
                magic_bytes, &bytes[..magic_bytes.len()]));
        }

        ValidationResult {
            valid: errors.is_empty(),
            version: if errors.is_empty() { Some(self.version.clone()) } else { None },
            errors,
            warnings: Vec::new(),
        }
    }

    /// Generate markdown documentation for the format.
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();
        md.push_str(&format!("# Needle File Format Specification v{}\n\n", self.version));
        md.push_str(&format!("Magic: `{}`\n\n", self.magic));

        md.push_str("## Header (4096 bytes)\n\n");
        md.push_str("| Offset | Size | Type | Name | Description |\n");
        md.push_str("|--------|------|------|------|-------------|\n");
        for f in &self.header_fields {
            md.push_str(&format!("| {} | {} | {} | {} | {} |\n",
                f.offset, f.size, f.data_type, f.name, f.description));
        }
        md.push_str("\n");

        for section in &self.sections {
            md.push_str(&format!("## {}\n\n{}\n\n", section.name, section.description));
        }

        md
    }

    /// Total header size.
    pub fn header_size(&self) -> usize {
        4096
    }

    /// Check version compatibility.
    pub fn is_version_compatible(&self, reader: &FormatVersion, writer: &FormatVersion) -> bool {
        self.compatibility.iter().any(|e|
            e.reader == *reader && e.writer == *writer && e.compatible
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_current_spec() {
        let spec = FormatSpec::current();
        assert_eq!(spec.magic, "NEEDLE01");
        assert_eq!(spec.version, FormatVersion::new(1, 0));
        assert!(!spec.header_fields.is_empty());
    }

    #[test]
    fn test_validate_magic() {
        let spec = FormatSpec::current();
        assert!(spec.validate_magic(b"NEEDLE01").valid);
        assert!(!spec.validate_magic(b"INVALID!").valid);
        assert!(!spec.validate_magic(b"SHORT").valid);
    }

    #[test]
    fn test_markdown_generation() {
        let spec = FormatSpec::current();
        let md = spec.to_markdown();
        assert!(md.contains("Needle File Format"));
        assert!(md.contains("NEEDLE01"));
        assert!(md.contains("| Offset |"));
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = FormatVersion::new(1, 0);
        assert!(v1.is_compatible(&FormatVersion::new(1, 0)));
        assert!(!v1.is_compatible(&FormatVersion::new(2, 0)));
    }

    #[test]
    fn test_header_size() {
        let spec = FormatSpec::current();
        assert_eq!(spec.header_size(), 4096);
    }

    #[test]
    fn test_compatibility_matrix() {
        let spec = FormatSpec::current();
        assert!(spec.is_version_compatible(&FormatVersion::new(1, 0), &FormatVersion::new(1, 0)));
        assert!(!spec.is_version_compatible(&FormatVersion::new(1, 0), &FormatVersion::new(2, 0)));
    }

    #[test]
    fn test_field_types() {
        assert_eq!(format!("{}", FieldType::U32Le), "u32le");
        assert_eq!(format!("{}", FieldType::Crc32), "crc32");
    }
}
