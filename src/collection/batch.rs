use super::*;

impl Collection {
    /// Batch search for multiple queries in parallel.
    ///
    /// Executes all queries concurrently using Rayon and returns results
    /// in the same order as the input queries.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query vectors (each must match collection dimensions)
    /// * `k` - Maximum number of results per query
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Any query has wrong dimensions
    /// - [`NeedleError::InvalidVector`] - Any query contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// # collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    /// let queries = vec![vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]];
    /// let all_results = collection.batch_search(&queries, 5)?;
    /// assert_eq!(all_results.len(), 2); // One result set per query
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn batch_search(&self, queries: &[Vec<f32>], k: usize) -> Result<Vec<Vec<SearchResult>>> {
        use std::time::Instant;
        let start = Instant::now();

        // Validate all queries have correct dimensions and values
        for query in queries.iter() {
            self.validate_query(query)?;
        }
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        // Perform parallel search
        let results: Vec<Result<Vec<SearchResult>>> = queries
            .par_iter()
            .map(|query| {
                let raw_results = self.index.search(query, k, self.vectors.as_slice())?;
                self.enrich_results(raw_results)
            })
            .collect();

        // Collect results, propagating any errors
        let results: Vec<Vec<SearchResult>> = results.into_iter().collect::<Result<_>>()?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    batch_size = queries.len(),
                    collection_size = self.len(),
                    query_type = "batch",
                    "slow batch query detected"
                );
            }
        }

        Ok(results)
    }

    /// Batch search with metadata filter in parallel.
    ///
    /// Like [`batch_search`](Self::batch_search), but applies a metadata filter to
    /// every query. Over-fetches candidates by `FILTER_CANDIDATE_MULTIPLIER` to
    /// compensate for filtered-out results.
    ///
    /// # Arguments
    ///
    /// * `queries` - Slice of query vectors (each must match collection dimensions)
    /// * `k` - Maximum number of results per query
    /// * `filter` - MongoDB-style metadata filter applied to all queries
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Any query has wrong dimensions
    /// - [`NeedleError::InvalidVector`] - Any query contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// # use needle::{Collection, CollectionConfig, Filter};
    /// # use serde_json::json;
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// # collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"tag": "a"}))).unwrap();
    /// let queries = vec![vec![1.0, 0.0, 0.0, 0.0]];
    /// let filter = Filter::eq("tag", "a");
    /// let results = collection.batch_search_with_filter(&queries, 5, &filter)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn batch_search_with_filter(
        &self,
        queries: &[Vec<f32>],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<Vec<SearchResult>>> {
        use std::time::Instant;
        let start = Instant::now();

        // Validate all queries have correct dimensions and values
        for query in queries.iter() {
            self.validate_query(query)?;
        }
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(vec![Vec::new(); queries.len()]);
        }

        // Perform parallel filtered search
        let results: Vec<Result<Vec<SearchResult>>> = queries
            .par_iter()
            .map(|query| {
                let candidates = self.index.search(
                    query,
                    k * FILTER_CANDIDATE_MULTIPLIER,
                    self.vectors.as_slice(),
                )?;
                let filtered: Vec<(VectorId, f32)> = candidates
                    .into_iter()
                    .filter(|(id, _)| {
                        if let Some(entry) = self.metadata.get(*id) {
                            filter.matches(entry.data.as_ref())
                        } else {
                            false
                        }
                    })
                    .take(k)
                    .collect();
                self.enrich_results(filtered)
            })
            .collect();

        let results: Vec<Vec<SearchResult>> = results.into_iter().collect::<Result<_>>()?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    batch_size = queries.len(),
                    collection_size = self.len(),
                    query_type = "batch",
                    filter_applied = true,
                    "slow batch query detected"
                );
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::collection::Collection;
    use crate::error::NeedleError;
    use crate::metadata::Filter;
    use serde_json::json;

    fn populated_collection() -> Collection {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"tag": "a"})))
            .unwrap();
        col.insert("v2", &[0.0, 1.0, 0.0, 0.0], Some(json!({"tag": "b"})))
            .unwrap();
        col.insert("v3", &[0.0, 0.0, 1.0, 0.0], Some(json!({"tag": "a"})))
            .unwrap();
        col
    }

    // ── batch_search ────────────────────────────────────────────────────

    #[test]
    fn test_batch_search_single_query() {
        let col = populated_collection();
        let results = col.batch_search(&[vec![1.0, 0.0, 0.0, 0.0]], 2).unwrap();
        assert_eq!(results.len(), 1);
        assert!(!results[0].is_empty());
    }

    #[test]
    fn test_batch_search_multiple_queries() {
        let col = populated_collection();
        let queries = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
        ];
        let results = col.batch_search(&queries, 3).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_batch_search_empty_queries() {
        let col = populated_collection();
        let results = col.batch_search(&[], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_batch_search_dimension_mismatch() {
        let col = populated_collection();
        let queries = vec![vec![1.0, 0.0]]; // wrong dims
        let result = col.batch_search(&queries, 5);
        assert!(matches!(result, Err(NeedleError::DimensionMismatch { .. })));
    }

    #[test]
    fn test_batch_search_k_zero() {
        let col = populated_collection();
        let results = col
            .batch_search(&[vec![1.0, 0.0, 0.0, 0.0]], 0)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    #[test]
    fn test_batch_search_empty_collection() {
        let col = Collection::with_dimensions("test", 4);
        let results = col
            .batch_search(&[vec![1.0, 0.0, 0.0, 0.0]], 5)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    // ── batch_search_with_filter ────────────────────────────────────────

    #[test]
    fn test_batch_search_with_filter_match() {
        let col = populated_collection();
        let filter = Filter::eq("tag", "a");
        let results = col
            .batch_search_with_filter(&[vec![1.0, 0.0, 0.0, 0.0]], 5, &filter)
            .unwrap();
        assert_eq!(results.len(), 1);
        for r in &results[0] {
            assert!(r.id == "v1" || r.id == "v3");
        }
    }

    #[test]
    fn test_batch_search_with_filter_no_match() {
        let col = populated_collection();
        let filter = Filter::eq("tag", "nonexistent");
        let results = col
            .batch_search_with_filter(&[vec![1.0, 0.0, 0.0, 0.0]], 5, &filter)
            .unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].is_empty());
    }

    #[test]
    fn test_batch_search_with_filter_dimension_mismatch() {
        let col = populated_collection();
        let filter = Filter::eq("tag", "a");
        let result = col.batch_search_with_filter(&[vec![1.0]], 5, &filter);
        assert!(matches!(result, Err(NeedleError::DimensionMismatch { .. })));
    }
}
