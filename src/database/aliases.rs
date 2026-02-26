use super::Database;
use crate::error::{NeedleError, Result};
use tracing::{debug, info};

impl Database {
    /// Create an alias for a collection.
    ///
    /// Aliases provide alternative names for collections, useful for blue-green
    /// deployments where you can switch the "production" alias from one collection
    /// to another atomically.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to create
    /// * `collection` - The target collection name
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::AliasAlreadyExists`] if the alias already exists.
    /// Returns [`NeedleError::CollectionNotFound`] if the target collection doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    ///
    /// // Create alias pointing to the new version
    /// db.create_alias("docs", "docs_v2")?;
    ///
    /// // Now we can access via alias
    /// let coll = db.collection("docs")?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn create_alias(&self, alias: &str, collection: &str) -> Result<()> {
        let mut state = self.state.write();

        // Check if collection exists
        if !state.collections.contains_key(collection) {
            return Err(NeedleError::CollectionNotFound(collection.to_string()));
        }

        // Check if alias already exists (as alias or collection name)
        if state.aliases.contains_key(alias) {
            return Err(NeedleError::AliasAlreadyExists(alias.to_string()));
        }
        if state.collections.contains_key(alias) {
            return Err(NeedleError::AliasAlreadyExists(format!(
                "{} (conflicts with collection name)",
                alias
            )));
        }

        info!(alias = %alias, collection = %collection, "Creating alias");
        state
            .aliases
            .insert(alias.to_string(), collection.to_string());
        self.mark_modified();
        Ok(())
    }

    /// Delete an alias.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to delete
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the alias was deleted, `Ok(false)` if no alias
    /// with that name existed.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    ///
    /// assert!(db.delete_alias("docs")?);
    /// assert!(!db.delete_alias("nonexistent")?);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete_alias(&self, alias: &str) -> Result<bool> {
        let mut state = self.state.write();
        let removed = state.aliases.remove(alias).is_some();
        if removed {
            info!(alias = %alias, "Alias deleted");
            self.mark_modified();
        } else {
            debug!(alias = %alias, "Alias not found for delete");
        }
        Ok(removed)
    }

    /// List all aliases.
    ///
    /// Returns a list of `(alias_name, collection_name)` tuples.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v1", 128)?;
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    /// db.create_alias("old_docs", "docs_v1")?;
    ///
    /// let aliases = db.list_aliases();
    /// assert_eq!(aliases.len(), 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn list_aliases(&self) -> Vec<(String, String)> {
        self.state
            .read()
            .aliases
            .iter()
            .map(|(alias, collection)| (alias.clone(), collection.clone()))
            .collect()
    }

    /// Get the canonical collection name for an alias.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to resolve
    ///
    /// # Returns
    ///
    /// Returns `Some(collection_name)` if the alias exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    ///
    /// assert_eq!(db.get_canonical_name("docs"), Some("docs_v2".to_string()));
    /// assert_eq!(db.get_canonical_name("nonexistent"), None);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn get_canonical_name(&self, alias: &str) -> Option<String> {
        self.state.read().aliases.get(alias).cloned()
    }

    /// Get all aliases that point to a specific collection.
    ///
    /// # Arguments
    ///
    /// * `collection` - The collection name to find aliases for
    ///
    /// # Returns
    ///
    /// A vector of alias names that reference the given collection.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("docs", "docs_v2")?;
    /// db.create_alias("production", "docs_v2")?;
    ///
    /// let aliases = db.aliases_for_collection("docs_v2");
    /// assert_eq!(aliases.len(), 2);
    /// assert!(aliases.contains(&"docs".to_string()));
    /// assert!(aliases.contains(&"production".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn aliases_for_collection(&self, collection: &str) -> Vec<String> {
        self.state
            .read()
            .aliases
            .iter()
            .filter(|(_, target)| *target == collection)
            .map(|(alias, _)| alias.clone())
            .collect()
    }

    /// Update an existing alias to point to a different collection.
    ///
    /// This is useful for blue-green deployments where you want to atomically
    /// switch an alias from one collection to another.
    ///
    /// # Arguments
    ///
    /// * `alias` - The alias name to update
    /// * `collection` - The new target collection name
    ///
    /// # Errors
    ///
    /// Returns [`NeedleError::AliasNotFound`] if the alias doesn't exist.
    /// Returns [`NeedleError::CollectionNotFound`] if the target collection doesn't exist.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Database;
    ///
    /// let db = Database::in_memory();
    /// db.create_collection("docs_v1", 128)?;
    /// db.create_collection("docs_v2", 128)?;
    /// db.create_alias("production", "docs_v1")?;
    ///
    /// // Switch production to v2
    /// db.update_alias("production", "docs_v2")?;
    ///
    /// assert_eq!(db.get_canonical_name("production"), Some("docs_v2".to_string()));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn update_alias(&self, alias: &str, collection: &str) -> Result<()> {
        let mut state = self.state.write();

        // Check if collection exists
        if !state.collections.contains_key(collection) {
            return Err(NeedleError::CollectionNotFound(collection.to_string()));
        }

        // Check if alias exists
        if !state.aliases.contains_key(alias) {
            return Err(NeedleError::AliasNotFound(alias.to_string()));
        }

        info!(alias = %alias, collection = %collection, "Updating alias");
        state
            .aliases
            .insert(alias.to_string(), collection.to_string());
        self.mark_modified();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Database;
    use crate::error::NeedleError;

    fn setup_db() -> Database {
        let db = Database::in_memory();
        db.create_collection("coll_v1", 128).unwrap();
        db.create_collection("coll_v2", 128).unwrap();
        db
    }

    // ── create_alias ────────────────────────────────────────────────────

    #[test]
    fn test_create_alias() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        assert_eq!(
            db.get_canonical_name("prod"),
            Some("coll_v1".to_string())
        );
    }

    #[test]
    fn test_create_alias_duplicate() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        let result = db.create_alias("prod", "coll_v2");
        assert!(matches!(result, Err(NeedleError::AliasAlreadyExists(_))));
    }

    #[test]
    fn test_create_alias_conflicts_with_collection_name() {
        let db = setup_db();
        let result = db.create_alias("coll_v1", "coll_v2");
        assert!(matches!(result, Err(NeedleError::AliasAlreadyExists(_))));
    }

    #[test]
    fn test_create_alias_missing_collection() {
        let db = setup_db();
        let result = db.create_alias("prod", "nonexistent");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }

    // ── delete_alias ────────────────────────────────────────────────────

    #[test]
    fn test_delete_alias() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        assert!(db.delete_alias("prod").unwrap());
        assert!(db.get_canonical_name("prod").is_none());
    }

    #[test]
    fn test_delete_alias_not_found() {
        let db = setup_db();
        assert!(!db.delete_alias("nonexistent").unwrap());
    }

    // ── list_aliases ────────────────────────────────────────────────────

    #[test]
    fn test_list_aliases() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        db.create_alias("staging", "coll_v2").unwrap();
        let aliases = db.list_aliases();
        assert_eq!(aliases.len(), 2);
    }

    #[test]
    fn test_list_aliases_empty() {
        let db = setup_db();
        assert!(db.list_aliases().is_empty());
    }

    // ── get_canonical_name ──────────────────────────────────────────────

    #[test]
    fn test_get_canonical_name_not_found() {
        let db = setup_db();
        assert!(db.get_canonical_name("missing").is_none());
    }

    // ── aliases_for_collection ──────────────────────────────────────────

    #[test]
    fn test_aliases_for_collection() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        db.create_alias("latest", "coll_v1").unwrap();
        let aliases = db.aliases_for_collection("coll_v1");
        assert_eq!(aliases.len(), 2);
    }

    #[test]
    fn test_aliases_for_collection_none() {
        let db = setup_db();
        assert!(db.aliases_for_collection("coll_v1").is_empty());
    }

    // ── update_alias ────────────────────────────────────────────────────

    #[test]
    fn test_update_alias() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        db.update_alias("prod", "coll_v2").unwrap();
        assert_eq!(
            db.get_canonical_name("prod"),
            Some("coll_v2".to_string())
        );
    }

    #[test]
    fn test_update_alias_not_found() {
        let db = setup_db();
        let result = db.update_alias("nonexistent", "coll_v1");
        assert!(matches!(result, Err(NeedleError::AliasNotFound(_))));
    }

    #[test]
    fn test_update_alias_missing_collection() {
        let db = setup_db();
        db.create_alias("prod", "coll_v1").unwrap();
        let result = db.update_alias("prod", "nonexistent");
        assert!(matches!(result, Err(NeedleError::CollectionNotFound(_))));
    }
}
