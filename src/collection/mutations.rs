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
