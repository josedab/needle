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

    /// Get the raw vector data by external ID.
    /// Returns `None` if the vector doesn't exist.
    pub fn get_vector(&self, id: &str) -> Option<Vec<f32>> {
        let internal_id = self.metadata.get_internal_id(id)?;
        self.vectors.get(internal_id).map(|v| v.to_vec())
    }

    /// Get the insertion timestamp for a vector by external ID.
    /// Returns `None` if no timestamp is recorded.
    pub fn insertion_timestamp_by_id(&self, id: &str) -> Option<u64> {
        let internal_id = self.metadata.get_internal_id(id)?;
        self.insertion_timestamps.get(&internal_id).copied()
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

// ── CDC accessors ────────────────────────────────────────────────────────────

impl Collection {
    /// Enable CDC (change data capture) on this collection.
    pub fn enable_cdc(&mut self, max_events: usize) {
        self.cdc_log = Some(CdcLog::new(max_events));
    }

    /// Disable CDC and discard the event log.
    pub fn disable_cdc(&mut self) {
        self.cdc_log = None;
    }

    /// Check whether CDC is enabled.
    pub fn cdc_enabled(&self) -> bool {
        self.cdc_log.is_some()
    }

    /// Get CDC events after the given cursor (sequence number), up to `limit`.
    pub fn cdc_events_since(&self, after_sequence: u64, limit: usize) -> Vec<&CdcEvent> {
        self.cdc_log
            .as_ref()
            .map_or_else(Vec::new, |log| log.events_since(after_sequence, limit))
    }

    /// Get the current CDC head sequence number.
    pub fn cdc_head_sequence(&self) -> u64 {
        self.cdc_log.as_ref().map_or(0, |log| log.head_sequence())
    }

    /// Compact CDC events older than `before_sequence`.
    pub fn cdc_compact(&mut self, before_sequence: u64) {
        if let Some(log) = &mut self.cdc_log {
            log.compact(before_sequence);
        }
    }

    /// Record a CDC event (internal helper called by insert/update/delete).
    pub(crate) fn record_cdc_event(
        &mut self,
        event_type: CdcEventType,
        vector_id: &str,
        metadata: Option<serde_json::Value>,
    ) {
        if let Some(log) = &mut self.cdc_log {
            log.append(event_type, vector_id, metadata);
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::collection::Collection;
    use crate::metadata::Filter;
    use serde_json::json;

    fn populated_collection() -> Collection {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"tag": "a"})))
            .unwrap();
        col.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"tag": "b"})))
            .unwrap();
        col.insert("v3", &[0.0, 0.0, 1.0, 0.0], None).unwrap();
        col
    }

    // ── get ─────────────────────────────────────────────────────────────

    #[test]
    fn test_get_existing() {
        let col = populated_collection();
        let (vec, meta) = col.get("v1").unwrap();
        assert_eq!(vec, &[1.0, 0.0, 0.0, 0.0]);
        assert!(meta.is_some());
    }

    #[test]
    fn test_get_nonexistent() {
        let col = populated_collection();
        assert!(col.get("missing").is_none());
    }

    #[test]
    fn test_get_no_metadata() {
        let col = populated_collection();
        let (_, meta) = col.get("v3").unwrap();
        assert!(meta.is_none());
    }

    // ── contains ────────────────────────────────────────────────────────

    #[test]
    fn test_contains_true() {
        let col = populated_collection();
        assert!(col.contains("v1"));
    }

    #[test]
    fn test_contains_false() {
        let col = populated_collection();
        assert!(!col.contains("missing"));
    }

    #[test]
    fn test_contains_empty() {
        let col = Collection::with_dimensions("test", 4);
        assert!(!col.contains("anything"));
    }

    // ── stats ───────────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let col = Collection::with_dimensions("test", 4);
        let stats = col.stats();
        assert_eq!(stats.vector_count, 0);
        assert_eq!(stats.dimensions, 4);
        assert_eq!(stats.name, "test");
    }

    #[test]
    fn test_stats_populated() {
        let col = populated_collection();
        let stats = col.stats();
        assert_eq!(stats.vector_count, 3);
        assert_eq!(stats.dimensions, 4);
        assert!(stats.total_memory_bytes > 0);
    }

    // ── count ───────────────────────────────────────────────────────────

    #[test]
    fn test_count_no_filter() {
        let col = populated_collection();
        assert_eq!(col.count(None), 3);
    }

    #[test]
    fn test_count_with_filter() {
        let col = populated_collection();
        let filter = Filter::eq("tag", "a");
        assert_eq!(col.count(Some(&filter)), 1);
    }

    #[test]
    fn test_count_filter_no_match() {
        let col = populated_collection();
        let filter = Filter::eq("tag", "nonexistent");
        assert_eq!(col.count(Some(&filter)), 0);
    }

    #[test]
    fn test_count_empty() {
        let col = Collection::with_dimensions("test", 4);
        assert_eq!(col.count(None), 0);
    }

    // ── deleted_count ───────────────────────────────────────────────────

    #[test]
    fn test_deleted_count_none() {
        let col = populated_collection();
        assert_eq!(col.deleted_count(), 0);
    }

    #[test]
    fn test_deleted_count_after_delete() {
        let mut col = populated_collection();
        col.delete("v1").unwrap();
        assert_eq!(col.deleted_count(), 1);
    }

    // ── iter ────────────────────────────────────────────────────────────

    #[test]
    fn test_iter() {
        let col = populated_collection();
        let items: Vec<_> = col.iter().collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_iter_empty() {
        let col = Collection::with_dimensions("test", 4);
        let items: Vec<_> = col.iter().collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_iter_skips_deleted() {
        let mut col = populated_collection();
        col.delete("v1").unwrap();
        let items: Vec<_> = col.iter().collect();
        assert_eq!(items.len(), 2);
    }

    // ── ids ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ids() {
        let col = populated_collection();
        let ids: Vec<_> = col.ids().collect();
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_ids_skips_deleted() {
        let mut col = populated_collection();
        col.delete("v2").unwrap();
        let ids: Vec<_> = col.ids().collect();
        assert_eq!(ids.len(), 2);
        assert!(!ids.contains(&"v2"));
    }

    // ── compact ─────────────────────────────────────────────────────────

    #[test]
    fn test_compact_no_deletions() {
        let mut col = populated_collection();
        let removed = col.compact().unwrap();
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_compact_after_delete() {
        let mut col = populated_collection();
        col.delete("v1").unwrap();
        let removed = col.compact().unwrap();
        assert!(removed > 0);
        assert_eq!(col.len(), 2);
        assert!(col.get("v2").is_some());
        assert!(col.get("v3").is_some());
    }

    // ── needs_compaction ────────────────────────────────────────────────

    #[test]
    fn test_needs_compaction_false() {
        let col = populated_collection();
        assert!(!col.needs_compaction(0.5));
    }

    // ── CollectionIter (IntoIterator) ───────────────────────────────────

    #[test]
    fn test_into_iter() {
        let col = populated_collection();
        let items: Vec<_> = (&col).into_iter().collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_collection_iter_size_hint() {
        let col = populated_collection();
        let iter = (&col).into_iter();
        let (_, upper) = iter.size_hint();
        assert!(upper.is_some());
    }

    // ── CDC accessors ───────────────────────────────────────────────────

    #[test]
    fn test_cdc_disabled_by_default() {
        let col = Collection::with_dimensions("test", 4);
        assert!(!col.cdc_enabled());
    }

    #[test]
    fn test_cdc_enable_disable() {
        let mut col = Collection::with_dimensions("test", 4);
        col.enable_cdc(100);
        assert!(col.cdc_enabled());
        col.disable_cdc();
        assert!(!col.cdc_enabled());
    }

    #[test]
    fn test_cdc_events_tracked() {
        let mut col = Collection::with_dimensions("test", 4);
        col.enable_cdc(100);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        assert!(col.cdc_head_sequence() > 0);
        let events = col.cdc_events_since(0, 10);
        assert!(!events.is_empty());
    }

    #[test]
    fn test_cdc_events_since_no_cdc() {
        let col = Collection::with_dimensions("test", 4);
        let events = col.cdc_events_since(0, 10);
        assert!(events.is_empty());
    }
}
