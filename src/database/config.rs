use crate::error::{NeedleError, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Database configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Path to the database file
    pub path: PathBuf,
    /// Whether to create if not exists
    pub create_if_missing: bool,
    /// Read-only mode
    pub read_only: bool,
    /// Automatically save on drop when there are dirty (unsaved) changes.
    /// Defaults to `false` for backward compatibility. I/O errors during
    /// auto-save are logged via `eprintln!` since `Drop` cannot return errors.
    #[serde(default)]
    pub auto_save: bool,
    /// Interval in seconds for automatic background flush of dirty data to disk.
    /// When set to a value > 0, a background thread will periodically call `save()`
    /// if there are unsaved changes. Set to `0` to disable.
    /// Defaults to `0` (disabled).
    #[serde(default)]
    pub auto_flush_interval_secs: u64,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            path: PathBuf::from("needle.db"),
            create_if_missing: true,
            read_only: false,
            auto_save: false,
            auto_flush_interval_secs: 0,
        }
    }
}

impl DatabaseConfig {
    /// Create a new config with the given path
    #[must_use]
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            ..Default::default()
        }
    }

    /// Enable or disable read-only mode.
    ///
    /// When enabled, [`save()`](crate::Database::save) returns an error and
    /// auto-save / auto-flush are rejected at validation time.
    #[must_use]
    pub fn with_read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Enable or disable auto-save on drop.
    ///
    /// When enabled, the database will attempt to persist unsaved changes
    /// during [`Drop`]. Defaults to `false`.
    #[must_use]
    pub fn with_auto_save(mut self, auto_save: bool) -> Self {
        self.auto_save = auto_save;
        self
    }

    /// Set the auto-flush interval in seconds.
    ///
    /// When > 0, a background thread will periodically save dirty data to disk.
    /// Set to `0` to disable.
    #[must_use]
    pub fn with_auto_flush_interval_secs(mut self, secs: u64) -> Self {
        self.auto_flush_interval_secs = secs;
        self
    }

    /// Validate the configuration for internal consistency.
    ///
    /// Returns an error if conflicting options are set, e.g. `read_only` combined
    /// with `auto_save` or a non-zero `auto_flush_interval_secs`.
    pub fn validate(&self) -> Result<()> {
        if self.read_only && self.auto_save {
            return Err(NeedleError::InvalidConfig(
                "read_only is incompatible with auto_save".to_string(),
            ));
        }
        if self.read_only && self.auto_flush_interval_secs > 0 {
            return Err(NeedleError::InvalidConfig(
                "read_only is incompatible with auto_flush_interval_secs > 0".to_string(),
            ));
        }
        if self.read_only && self.create_if_missing {
            return Err(NeedleError::InvalidConfig(
                "read_only is incompatible with create_if_missing; set create_if_missing to false for read-only databases".to_string(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DatabaseConfig::default();
        assert_eq!(config.path, PathBuf::from("needle.db"));
        assert!(config.create_if_missing);
        assert!(!config.read_only);
        assert!(!config.auto_save);
    }

    #[test]
    fn test_new_config() {
        let config = DatabaseConfig::new("/tmp/test.needle");
        assert_eq!(config.path, PathBuf::from("/tmp/test.needle"));
        assert!(config.create_if_missing);
    }

    #[test]
    fn test_with_auto_save() {
        let config = DatabaseConfig::new("test.db").with_auto_save(true);
        assert!(config.auto_save);
    }

    #[test]
    fn test_with_auto_save_false() {
        let config = DatabaseConfig::new("test.db").with_auto_save(false);
        assert!(!config.auto_save);
    }

    #[test]
    fn test_with_read_only() {
        let config = DatabaseConfig::new("test.db").with_read_only(true);
        assert!(config.read_only);
    }

    #[test]
    fn test_validate_default_ok() {
        assert!(DatabaseConfig::default().validate().is_ok());
    }

    #[test]
    fn test_validate_read_only_with_auto_save_fails() {
        let config = DatabaseConfig::new("test.db")
            .with_read_only(true)
            .with_auto_save(true);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_read_only_with_auto_flush_fails() {
        let mut config = DatabaseConfig::new("test.db");
        config.read_only = true;
        config.auto_flush_interval_secs = 30;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_read_only_alone_ok() {
        let mut config = DatabaseConfig::new("test.db").with_read_only(true);
        config.create_if_missing = false;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_auto_save_alone_ok() {
        let config = DatabaseConfig::new("test.db").with_auto_save(true);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_read_only_with_create_if_missing_fails() {
        let config = DatabaseConfig::new("test.db").with_read_only(true);
        // create_if_missing defaults to true, which conflicts with read_only
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validate_read_only_no_create_ok() {
        let mut config = DatabaseConfig::new("test.db").with_read_only(true);
        config.create_if_missing = false;
        assert!(config.validate().is_ok());
    }
}
