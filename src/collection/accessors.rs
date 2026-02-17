use super::*;

impl Collection {
    /// Retrieve a vector and its metadata by ID.
    ///
    /// Returns a reference to the vector data and optional metadata if found.
    ///
    /// # Arguments
    ///
    /// * `id` - The external ID of the vector to retrieve
    ///
    /// # Returns
    ///
    /// Returns `Some((vector, metadata))` if the vector exists, `None` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 2.0, 3.0, 4.0], Some(json!({"name": "test"})))?;
    ///
    /// if let Some((vector, metadata)) = collection.get("v1") {
    ///     assert_eq!(vector, &[1.0, 2.0, 3.0, 4.0]);
    ///     assert!(metadata.is_some());
    /// }
    ///
    /// assert!(collection.get("nonexistent").is_none());
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn get(&self, id: &str) -> Option<(&[f32], Option<&Value>)> {
        let internal_id = self.metadata.get_internal_id(id)?;
        let vector = self.vectors.get(internal_id)?;
        let metadata = self.metadata.get(internal_id).and_then(|e| e.data.as_ref());
        Some((vector, metadata))
    }

    /// Check if a vector with the given ID exists.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    ///
    /// assert!(collection.contains("v1"));
    /// assert!(!collection.contains("v2"));
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn contains(&self, id: &str) -> bool {
        self.metadata.contains(id)
    }

    /// Get collection statistics
    pub fn stats(&self) -> CollectionStats {
        let vector_count = self.vectors.len();
        let dimensions = self.config.dimensions;

        // Estimate memory usage
        let vector_memory = vector_count * dimensions * std::mem::size_of::<f32>();
        let metadata_memory = self.metadata.estimated_memory();
        let index_memory = self.index.estimated_memory();

        CollectionStats {
            name: self.config.name.clone(),
            vector_count,
            dimensions,
            distance_function: self.config.distance,
            vector_memory_bytes: vector_memory,
            metadata_memory_bytes: metadata_memory,
            index_memory_bytes: index_memory,
            total_memory_bytes: vector_memory + metadata_memory + index_memory,
            index_stats: self.index.stats(),
        }
    }

    /// Count vectors matching an optional filter
    pub fn count(&self, filter: Option<&Filter>) -> usize {
        match filter {
            None => self.len(),
            Some(f) => self
                .metadata
                .iter()
                .filter(|(internal_id, entry)| {
                    !self.index.is_deleted(*internal_id) && f.matches(entry.data.as_ref())
                })
                .count(),
        }
    }

    /// Get the number of deleted vectors pending compaction
    pub fn deleted_count(&self) -> usize {
        self.index.deleted_count()
    }

    /// Iterate over all vectors in the collection
    /// Returns an iterator of (external_id, vector, metadata)
    pub fn iter(&self) -> impl Iterator<Item = (&str, &[f32], Option<&Value>)> {
        self.metadata
            .iter()
            .filter_map(move |(internal_id, entry)| {
                // Skip deleted vectors
                if self.index.is_deleted(internal_id) {
                    return None;
                }
                let vector = self.vectors.get(internal_id)?;
                Some((
                    entry.external_id.as_str(),
                    vector.as_slice(),
                    entry.data.as_ref(),
                ))
            })
    }

    /// Get all vector IDs in the collection
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.metadata
            .iter()
            .filter(move |(internal_id, _)| !self.index.is_deleted(*internal_id))
            .map(|(_, entry)| entry.external_id.as_str())
    }

    /// Compact the collection by removing deleted and expired vectors
    /// This rebuilds the index and reclaims storage
    /// Returns the number of vectors removed
    pub fn compact(&mut self) -> Result<usize> {
        // First, expire any TTL'd vectors
        let expired_count = self.expire_vectors()?;

        let deleted_count = self.index.deleted_count();

        if deleted_count == 0 {
            return Ok(expired_count);
        }

        // Get the ID mapping from compacting the index
        let id_map = self.index.compact(self.vectors.as_slice())?;

        // Rebuild vectors and metadata with new IDs
        let mut new_vectors = VectorStore::new(self.config.dimensions);
        let mut new_metadata = MetadataStore::new();
        let mut new_expirations = HashMap::new();
        let mut new_insertion_timestamps = HashMap::new();

        // Sort by new ID to maintain order
        let mut mappings: Vec<_> = id_map.into_iter().collect();
        mappings.sort_by_key(|(_, new_id)| *new_id);

        for (old_id, new_id) in mappings {
            if let Some(vector) = self.vectors.get(old_id) {
                let added_id = new_vectors.add(vector.clone())?;
                debug_assert_eq!(added_id, new_id);

                if let Some(entry) = self.metadata.get(old_id) {
                    new_metadata.insert(new_id, entry.external_id.clone(), entry.data.clone())?;
                }

                // Remap expiration entry if it exists
                if let Some(expiration) = self.expirations.get(&old_id) {
                    new_expirations.insert(new_id, *expiration);
                }

                // Remap insertion timestamp if it exists
                if let Some(ts) = self.insertion_timestamps.get(&old_id) {
                    new_insertion_timestamps.insert(new_id, *ts);
                }
            }
        }

        self.vectors = new_vectors;
        self.metadata = new_metadata;
        self.expirations = new_expirations;
        self.insertion_timestamps = new_insertion_timestamps;

        // Invalidate cache since internal IDs changed
        self.invalidate_cache();

        Ok(deleted_count + expired_count)
    }

    /// Check if the collection needs compaction.
    ///
    /// Returns `true` if the ratio of deleted vectors to total vectors exceeds
    /// `threshold`. Use this to decide when to call [`compact`](Self::compact)
    /// to reclaim storage space.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Ratio threshold in the range 0.0–1.0 (e.g., 0.2 = 20% deleted)
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # let config = CollectionConfig::new("test", 4);
    /// # let collection = Collection::new(config);
    /// if collection.needs_compaction(0.2) {
    ///     // More than 20% of vectors are deleted; compact to reclaim space
    /// }
    /// ```
    pub fn needs_compaction(&self, threshold: f64) -> bool {
        self.index.needs_compaction(threshold)
    }
}

/// Iterator over collection entries.
///
/// Yields `(id, vector, metadata)` tuples for each vector in the collection.
/// Deleted vectors are automatically skipped.
pub struct CollectionIter<'a> {
    collection: &'a Collection,
    ids: Vec<String>,
    index: usize,
}

impl<'a> CollectionIter<'a> {
    fn new(collection: &'a Collection) -> Self {
        let ids = collection.all_ids();
        Self {
            collection,
            ids,
            index: 0,
        }
    }
}

impl<'a> Iterator for CollectionIter<'a> {
    type Item = (String, Vec<f32>, Option<Value>);

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.ids.len() {
            let id = &self.ids[self.index];
            self.index += 1;

            if let Some((vector, metadata)) = self.collection.get(id) {
                return Some((id.clone(), vector.to_vec(), metadata.cloned()));
            }
        }
        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.ids.len().saturating_sub(self.index);
        (0, Some(remaining))
    }
}

impl<'a> IntoIterator for &'a Collection {
    type Item = (String, Vec<f32>, Option<Value>);
    type IntoIter = CollectionIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        CollectionIter::new(self)
    }
}
