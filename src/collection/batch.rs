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
