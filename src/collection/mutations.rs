use super::*;

impl Collection {
    /// Delete a vector by its ID.
    ///
    /// Removes the vector from the index, but does not immediately reclaim
    /// storage space. Call [`compact()`](Self::compact) after many deletions
    /// to reclaim space.
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the vector was deleted, `Ok(false)` if no vector
    /// with that ID existed.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(collection.delete("v1")?);
    /// assert!(!collection.delete("v1")?); // Already deleted
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete(&mut self, id: &str) -> Result<bool> {
        let internal_id = match self.metadata.get_internal_id(id) {
            Some(id) => id,
            None => return Ok(false),
        };

        self.metadata.delete(internal_id);
        self.index.delete(internal_id)?;
        self.provenance_store.remove(id);

        // Record CDC event
        self.record_cdc_event(CdcEventType::Delete, id, None);

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(true)
    }

    /// Delete multiple vectors by their external IDs.
    ///
    /// Skips IDs that do not exist (no error is raised for missing IDs).
    ///
    /// # Arguments
    ///
    /// * `ids` - Slice of external vector IDs to delete
    ///
    /// # Returns
    ///
    /// The number of vectors that were actually found and deleted.
    ///
    /// # Errors
    ///
    /// Returns an error if an underlying index operation fails.
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// # collection.insert("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    /// # collection.insert("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
    /// let deleted = collection.delete_batch(&["a", "b", "nonexistent"])?;
    /// assert_eq!(deleted, 2);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn delete_batch(&mut self, ids: &[impl AsRef<str>]) -> Result<usize> {
        let mut deleted = 0;
        for id in ids {
            if self.delete(id.as_ref())? {
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    /// Update an existing vector and its metadata.
    ///
    /// Replaces both the vector data and metadata for an existing vector.
    /// The vector is re-indexed after the update.
    ///
    /// # Arguments
    ///
    /// * `id` - ID of the vector to update
    /// * `vector` - New vector data (must match collection dimensions)
    /// * `metadata` - New metadata (replaces existing metadata)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::VectorNotFound`] - No vector with the given ID exists
    /// - [`NeedleError::DimensionMismatch`] - Vector dimensions don't match
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"version": 1})))?;
    ///
    /// // Update the vector and metadata
    /// collection.update("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"version": 2})))?;
    ///
    /// let (vec, meta) = collection.get("v1").unwrap();
    /// assert_eq!(vec, &[0.0, 1.0, 0.0, 0.0]);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn update(&mut self, id: &str, vector: &[f32], metadata: Option<Value>) -> Result<()> {
        // Check dimensions
        if vector.len() != self.config.dimensions {
            return Err(NeedleError::DimensionMismatch {
                expected: self.config.dimensions,
                got: vector.len(),
            });
        }

        // Get internal ID
        let internal_id = self
            .metadata
            .get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        // Update vector in storage
        self.vectors.update(internal_id, vector.to_vec())?;

        // Update metadata
        self.metadata.update_data(internal_id, metadata)?;

        // Re-index the vector (delete and re-insert in index)
        self.index.delete(internal_id)?;
        self.index
            .insert(internal_id, vector, self.vectors.as_slice())?;

        // Record CDC event
        self.record_cdc_event(CdcEventType::Update, id, None);

        // Invalidate cache since collection changed
        self.invalidate_cache();

        Ok(())
    }

    /// Update only the metadata for an existing vector
    /// Returns error if the vector doesn't exist
    pub fn update_metadata(&mut self, id: &str, metadata: Option<Value>) -> Result<()> {
        let internal_id = self
            .metadata
            .get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        self.metadata.update_data(internal_id, metadata)?;

        // Invalidate cache since metadata is part of search results
        self.invalidate_cache();

        Ok(())
    }

    /// Insert a vector, or update it if it already exists.
    ///
    /// If a vector with the given `id` exists, its data and metadata are replaced
    /// (equivalent to calling [`update`](Self::update)). Otherwise a new vector is inserted.
    ///
    /// # Arguments
    ///
    /// * `id` - External vector ID
    /// * `vector` - Vector data (must match collection dimensions)
    /// * `metadata` - Optional JSON metadata
    ///
    /// # Returns
    ///
    /// `true` if a new vector was inserted, `false` if an existing vector was updated.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Vector dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Vector contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # use serde_json::json;
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// let is_new = collection.upsert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// assert!(is_new); // inserted
    ///
    /// let is_new = collection.upsert("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"v": 2})))?;
    /// assert!(!is_new); // updated
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn upsert(
        &mut self,
        id: impl Into<String>,
        vector: &[f32],
        metadata: Option<Value>,
    ) -> Result<bool> {
        let id = id.into();

        if self.contains(&id) {
            self.update(&id, vector, metadata)?;
            Ok(false) // Updated existing
        } else {
            self.insert(id, vector, metadata)?;
            Ok(true) // Inserted new
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::collection::Collection;
    use crate::error::NeedleError;
    use serde_json::json;

    // ── Delete ───────────────────────────────────────────────────────────

    #[test]
    fn test_delete_existing() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert!(col.delete("v1").unwrap());
        assert_eq!(col.len(), 0);
    }

    #[test]
    fn test_delete_not_found() {
        let mut col = Collection::with_dimensions("test", 4);
        assert!(!col.delete("nonexistent").unwrap());
    }

    #[test]
    fn test_delete_idempotent() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert!(col.delete("v1").unwrap());
        assert!(!col.delete("v1").unwrap());
    }

    // ── Delete batch ────────────────────────────────────────────────────

    #[test]
    fn test_delete_batch_basic() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("b", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        let deleted = col.delete_batch(&["a", "b", "nonexistent"]).unwrap();
        assert_eq!(deleted, 2);
        assert_eq!(col.len(), 0);
    }

    #[test]
    fn test_delete_batch_empty() {
        let mut col = Collection::with_dimensions("test", 4);
        let deleted = col.delete_batch(&Vec::<String>::new()).unwrap();
        assert_eq!(deleted, 0);
    }

    #[test]
    fn test_delete_batch_all_missing() {
        let mut col = Collection::with_dimensions("test", 4);
        let deleted = col.delete_batch(&["x", "y"]).unwrap();
        assert_eq!(deleted, 0);
    }

    // ── Update ──────────────────────────────────────────────────────────

    #[test]
    fn test_update_basic() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.update("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"k": "v"})))
            .unwrap();
        let (vec, meta) = col.get("v1").unwrap();
        assert_eq!(vec, &[0.0, 1.0, 0.0, 0.0]);
        assert!(meta.is_some());
    }

    #[test]
    fn test_update_not_found() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.update("missing", &[1.0, 0.0, 0.0, 0.0], None);
        assert!(matches!(result, Err(NeedleError::VectorNotFound(_))));
    }

    #[test]
    fn test_update_dimension_mismatch() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        let result = col.update("v1", &[1.0, 0.0], None);
        assert!(matches!(result, Err(NeedleError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_update_searchable_after() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("v2", &[0.0, 1.0, 0.0, 0.0], None).unwrap();
        col.update("v1", &[0.0, 0.99, 0.0, 0.0], None).unwrap();
        let results = col.search(&[0.0, 1.0, 0.0, 0.0], 1).unwrap();
        // v1 is now closer to the query than before
        assert!(!results.is_empty());
    }

    // ── Update metadata ─────────────────────────────────────────────────

    #[test]
    fn test_update_metadata_basic() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.update_metadata("v1", Some(json!({"new": true})))
            .unwrap();
        let (_, meta) = col.get("v1").unwrap();
        assert_eq!(meta.unwrap()["new"], true);
    }

    #[test]
    fn test_update_metadata_not_found() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.update_metadata("missing", Some(json!({})));
        assert!(matches!(result, Err(NeedleError::VectorNotFound(_))));
    }

    #[test]
    fn test_update_metadata_to_none() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"k": 1})))
            .unwrap();
        col.update_metadata("v1", None).unwrap();
        let (_, meta) = col.get("v1").unwrap();
        assert!(meta.is_none());
    }

    // ── Upsert ──────────────────────────────────────────────────────────

    #[test]
    fn test_upsert_insert_path() {
        let mut col = Collection::with_dimensions("test", 4);
        let is_new = col.upsert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert!(is_new);
        assert_eq!(col.len(), 1);
    }

    #[test]
    fn test_upsert_update_path() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        let is_new = col
            .upsert("v1", &[0.0, 1.0, 0.0, 0.0], Some(json!({"v": 2})))
            .unwrap();
        assert!(!is_new);
        assert_eq!(col.len(), 1);
        let (vec, _) = col.get("v1").unwrap();
        assert_eq!(vec, &[0.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn test_upsert_dimension_mismatch() {
        let mut col = Collection::with_dimensions("test", 4);
        let result = col.upsert("v1", &[1.0, 0.0], None);
        assert!(result.is_err());
    }
}
