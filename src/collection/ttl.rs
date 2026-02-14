use super::*;

impl Collection {
    // ============ TTL/Expiration Methods ============

    /// Get the current Unix timestamp in seconds
    #[inline]
    pub(super) fn now_unix() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Check if a vector has expired based on its internal ID
    #[inline]
    pub(super) fn is_expired(&self, internal_id: usize) -> bool {
        if let Some(&expiration) = self.expirations.get(&internal_id) {
            Self::now_unix() >= expiration
        } else {
            false
        }
    }

    /// Sweep and delete all expired vectors.
    ///
    /// This is the "eager" expiration strategy. Call this periodically to
    /// remove expired vectors and reclaim storage space.
    ///
    /// # Returns
    ///
    /// The number of vectors that were expired and deleted.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, CollectionConfig};
    ///
    /// let config = CollectionConfig::new("ephemeral", 4)
    ///     .with_default_ttl_seconds(1); // 1 second TTL
    /// let mut collection = Collection::new(config);
    ///
    /// collection.insert("temp", &[0.1, 0.2, 0.3, 0.4], None)?;
    ///
    /// // Wait for expiration...
    /// std::thread::sleep(std::time::Duration::from_secs(2));
    ///
    /// let expired = collection.expire_vectors()?;
    /// assert_eq!(expired, 1);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn expire_vectors(&mut self) -> Result<usize> {
        let now = Self::now_unix();
        let mut expired_ids = Vec::new();

        // Find all expired vectors
        for (&internal_id, &expiration) in &self.expirations {
            if now >= expiration {
                expired_ids.push(internal_id);
            }
        }

        // Delete each expired vector
        for internal_id in &expired_ids {
            // Mark as deleted in index
            self.index.delete(*internal_id)?;
            // Remove metadata
            self.metadata.delete(*internal_id);
            // Remove from expirations tracking
            self.expirations.remove(internal_id);
        }

        if !expired_ids.is_empty() {
            self.invalidate_cache();
        }

        Ok(expired_ids.len())
    }

    /// Check if an expiration sweep is needed based on a threshold.
    ///
    /// Returns true if the ratio of expired vectors to total vectors
    /// exceeds the given threshold (0.0-1.0).
    ///
    /// # Arguments
    ///
    /// * `threshold` - Ratio threshold (e.g., 0.1 = 10% expired)
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let collection = Collection::with_dimensions("test", 4);
    ///
    /// // Check if more than 10% of vectors are expired
    /// if collection.needs_expiration_sweep(0.1) {
    ///     // Run sweep...
    /// }
    /// ```
    pub fn needs_expiration_sweep(&self, threshold: f64) -> bool {
        if self.expirations.is_empty() {
            return false;
        }

        let now = Self::now_unix();
        let expired_count = self.expirations.values().filter(|&&exp| now >= exp).count();
        let total = self.len();

        if total == 0 {
            return expired_count > 0;
        }

        (expired_count as f64 / total as f64) > threshold
    }

    /// Get TTL statistics for the collection.
    ///
    /// Returns a tuple of (total_with_ttl, expired_count, earliest_expiration, latest_expiration).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let collection = Collection::with_dimensions("test", 4);
    /// let (total, expired, earliest, latest) = collection.ttl_stats();
    /// println!("TTL vectors: {}, expired: {}", total, expired);
    /// ```
    pub fn ttl_stats(&self) -> (usize, usize, Option<u64>, Option<u64>) {
        let now = Self::now_unix();
        let total = self.expirations.len();
        let expired = self.expirations.values().filter(|&&exp| now >= exp).count();
        let earliest = self.expirations.values().copied().min();
        let latest = self.expirations.values().copied().max();

        (total, expired, earliest, latest)
    }

    /// Get the expiration timestamp for a vector by external ID.
    ///
    /// Returns `None` if the vector doesn't exist or has no TTL set.
    pub fn get_ttl(&self, id: &str) -> Option<u64> {
        let internal_id = self.metadata.get_internal_id(id)?;
        self.expirations.get(&internal_id).copied()
    }

    /// Set or update the TTL for an existing vector.
    ///
    /// # Arguments
    ///
    /// * `id` - External vector ID
    /// * `ttl_seconds` - TTL in seconds from now, or `None` to remove TTL
    ///
    /// # Errors
    ///
    /// Returns an error if the vector doesn't exist.
    pub fn set_ttl(&mut self, id: &str, ttl_seconds: Option<u64>) -> Result<()> {
        let internal_id = self
            .metadata
            .get_internal_id(id)
            .ok_or_else(|| NeedleError::VectorNotFound(id.to_string()))?;

        match ttl_seconds {
            Some(ttl) => {
                let expiration = Self::now_unix() + ttl;
                self.expirations.insert(internal_id, expiration);
            }
            None => {
                self.expirations.remove(&internal_id);
            }
        }

        Ok(())
    }
}
