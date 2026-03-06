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
}
