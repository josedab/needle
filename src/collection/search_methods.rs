use super::*;

impl Collection {
    /// Create a search builder for fluent search configuration
    ///
    /// # Example
    /// ```ignore
    /// let results = collection
    ///     .search_builder(&query)
    ///     .k(10)
    ///     .filter(&filter)
    ///     .ef_search(100)
    ///     .execute()?;
    /// ```
    pub fn search_builder<'a>(&'a self, query: &'a [f32]) -> SearchBuilder<'a> {
        SearchBuilder::new(self, query)
    }

    /// Search for the k nearest neighbors to a query vector.
    ///
    /// Performs approximate nearest neighbor search using the HNSW index.
    /// Results are sorted by distance (ascending) and include metadata.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// Returns up to `k` [`SearchResult`]s, sorted by distance (closest first).
    /// May return fewer than `k` results if the collection has fewer vectors.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match collection
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity values
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// collection.insert("v2", &[0.0, 1.0, 0.0, 0.0], None)?;
    ///
    /// let results = collection.search(&[1.0, 0.0, 0.0, 0.0], 5)?;
    /// assert_eq!(results[0].id, "v1"); // Closest match
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Check semantic cache BEFORE exact-match LRU cache
        if let Some(ref sem_cache) = self.semantic_cache {
            let mut cache = sem_cache.lock();
            if let Some(results) = cache.lookup(query, k) {
                return Ok(results);
            }
        }

        // Use exact-match cache if enabled
        let results = self.search_with_cache(query, k, || {
            let raw_results = self.index.search(query, k, self.vectors.as_slice())?;

            // Apply lazy expiration filtering if enabled
            let filtered_results = if self.config.lazy_expiration {
                raw_results
                    .into_iter()
                    .filter(|(id, _)| !self.is_expired(*id))
                    .collect()
            } else {
                raw_results
            };

            self.enrich_results(filtered_results)
        })?;

        // Store in semantic cache
        if let Some(ref sem_cache) = self.semantic_cache {
            let mut cache = sem_cache.lock();
            cache.insert(query, k, &results);
        }

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    results_count = results.len(),
                    collection_size = self.len(),
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Search using Matryoshka-style dimension truncation for faster retrieval.
    ///
    /// Performs a two-phase search:
    /// 1. **Coarse phase**: Uses HNSW search on truncated vectors for O(log n) candidate retrieval
    /// 2. **Re-rank phase**: Re-scores candidates using full-dimension vectors and returns top `k`
    ///
    /// This works with Matryoshka-trained embeddings (e.g., OpenAI text-embedding-3, Nomic embed)
    /// where prefix truncation preserves semantic meaning.
    ///
    /// # Arguments
    /// * `query` - Full-dimension query vector
    /// * `k` - Number of results to return
    /// * `coarse_dims` - Truncated dimension count for coarse search (e.g., 256 for 1024d vectors)
    /// * `oversample` - Multiplier for candidate set size (default: 4)
    pub fn search_matryoshka(
        &self,
        query: &[f32],
        k: usize,
        coarse_dims: usize,
        oversample: usize,
    ) -> Result<Vec<SearchResult>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let dims = self.config.dimensions;
        if coarse_dims >= dims || coarse_dims == 0 {
            return self.search(query, k);
        }

        let oversample = oversample.max(2);
        let candidate_k = (k * oversample).min(self.len());

        // Build a truncated vector set for HNSW coarse search
        let all_vectors = self.vectors.as_slice();
        let truncated_vectors: Vec<Vec<f32>> = all_vectors
            .iter()
            .map(|v| v[..coarse_dims.min(v.len())].to_vec())
            .collect();

        // Phase 1: HNSW search on truncated vectors — O(log n) instead of O(n)
        let truncated_query = &query[..coarse_dims];
        let coarse_hnsw = HnswIndex::new(
            HnswConfig::default().ef_search(candidate_k.max(50)),
            self.config.distance,
        );
        // Use the primary index's HNSW graph for traversal but with truncated distance computation
        let candidates = self.index.search(truncated_query, candidate_k, &truncated_vectors)?;

        // Phase 2: Re-rank with full dimensions
        let distance_fn = self.config.distance;
        let mut reranked: Vec<(usize, f32)> = candidates
            .iter()
            .filter(|(id, _)| !self.index.is_deleted(*id) && !self.is_expired(*id))
            .filter_map(|(id, _)| {
                let full_vec = self.vectors.get(*id)?;
                let full_dist = distance_fn.compute(query, full_vec).ok()?;
                Some((*id, full_dist))
            })
            .collect();

        reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        reranked.truncate(k);

        self.enrich_results(reranked)
    }

    /// Brute-force linear search with a specified distance function.
    ///
    /// This method scans all vectors in the collection and computes distances
    /// using the specified distance function. It's used when the query's distance
    /// function differs from the index's configured function.
    ///
    /// **Warning:** This is O(n) complexity and may be slow on large collections.
    pub(super) fn brute_force_search(&self, params: &BruteForceSearchParams<'_>) -> Result<Vec<SearchResult>> {
        use std::cmp::Reverse;
        use std::collections::BinaryHeap;

        let k = self.clamp_k(params.k);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Use a max-heap (via Reverse) to track top-k by smallest distance
        let mut heap: BinaryHeap<(Reverse<OrderedFloat<f32>>, usize)> =
            BinaryHeap::with_capacity(k + 1);

        // Linear scan over all non-deleted vectors
        for (internal_id, vector) in self.vectors.as_slice().iter().enumerate() {
            // Skip deleted vectors
            if self.index.is_deleted(internal_id) {
                continue;
            }

            // Skip expired vectors if lazy expiration is enabled
            if self.config.lazy_expiration && self.is_expired(internal_id) {
                continue;
            }

            // Apply pre-filter if present
            if let Some(f) = params.filter {
                if let Some(entry) = self.metadata.get(internal_id) {
                    if !f.matches(entry.data.as_ref()) {
                        continue;
                    }
                } else {
                    continue;
                }
            }

            // Compute distance with the specified function
            let dist = params.distance_fn.compute(params.query, vector)?;

            // Add to heap
            heap.push((Reverse(OrderedFloat(dist)), internal_id));

            // Keep only top-k
            if heap.len() > k {
                heap.pop();
            }
        }

        // Extract results in order (smallest distance first)
        let mut results: Vec<_> = heap.into_sorted_vec();
        results.reverse(); // BinaryHeap::into_sorted_vec returns largest first

        // Convert to SearchResults
        let mut search_results = Vec::with_capacity(results.len());
        for (Reverse(OrderedFloat(distance)), internal_id) in results {
            if let Some(entry) = self.metadata.get(internal_id) {
                let metadata = if params.include_metadata || params.post_filter.is_some() {
                    entry.data.clone()
                } else {
                    None
                };
                search_results.push(SearchResult {
                    id: entry.external_id.clone(),
                    distance,
                    metadata,
                });
            }
        }

        // Apply post-filter if present
        if let Some(pf) = params.post_filter {
            search_results.retain(|r| pf.matches(r.metadata.as_ref()));
        }

        // Strip metadata if not requested
        if !params.include_metadata {
            for result in &mut search_results {
                result.metadata = None;
            }
        }

        Ok(search_results)
    }

    /// Search with detailed query execution profiling.
    ///
    /// Returns both the search results and a [`SearchExplain`] struct containing
    /// detailed timing and statistics for query optimization and debugging.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    ///
    /// # Returns
    ///
    /// A tuple of `(results, explain)` where:
    /// - `results` - The search results (same as [`search()`](Self::search))
    /// - `explain` - Detailed profiling information
    ///
    /// # Errors
    ///
    /// Returns an error if the query is invalid (wrong dimensions, NaN/Inf values).
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// for i in 0..100 {
    ///     collection.insert(format!("v{}", i), &[i as f32, 0.0, 0.0, 0.0], None)?;
    /// }
    ///
    /// let (results, explain) = collection.search_explain(&[50.0, 0.0, 0.0, 0.0], 10)?;
    ///
    /// println!("Search completed in {}μs", explain.total_time_us);
    /// println!("HNSW traversal: {}μs ({} nodes visited)",
    ///          explain.index_time_us, explain.hnsw_stats.visited_nodes);
    /// println!("Effective k: {} (requested: {})", explain.effective_k, explain.requested_k);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_explain(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, SearchExplain)> {
        use std::time::Instant;

        let total_start = Instant::now();
        self.validate_query(query)?;

        let effective_k = self.clamp_k(k);
        if effective_k == 0 {
            let mut explain = SearchExplain {
                dimensions: self.config.dimensions,
                collection_size: self.len(),
                requested_k: k,
                effective_k: 0,
                ef_search: self.index.config().ef_search,
                distance_function: format!("{:?}", self.config.distance),
                ..Default::default()
            };
            explain.total_time_us = total_start.elapsed().as_micros() as u64;
            return Ok((Vec::new(), explain));
        }

        // HNSW index search with stats
        let index_start = Instant::now();
        let (raw_results, hnsw_stats) =
            self.index
                .search_with_stats(query, effective_k, self.vectors.as_slice())?;
        let index_time = index_start.elapsed();

        let candidates_before_filter = raw_results.len();

        // Enrich results with metadata
        let enrich_start = Instant::now();
        let results = self.enrich_results(raw_results)?;
        let enrich_time = enrich_start.elapsed();

        let total_time = total_start.elapsed();

        let explain = SearchExplain {
            total_time_us: total_time.as_micros() as u64,
            index_time_us: index_time.as_micros() as u64,
            filter_time_us: 0,
            enrich_time_us: enrich_time.as_micros() as u64,
            candidates_before_filter,
            candidates_after_filter: results.len(),
            hnsw_stats,
            dimensions: self.config.dimensions,
            collection_size: self.len(),
            requested_k: k,
            effective_k,
            ef_search: self.index.config().ef_search,
            filter_applied: false,
            distance_function: format!("{:?}", self.config.distance),
        };

        Ok((results, explain))
    }

    /// Search with full HNSW graph traversal trace for debugging.
    ///
    /// Returns search results along with a detailed `SearchTrace` showing
    /// every hop, distance computation, and layer traversal decision.
    pub fn search_with_trace(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<(Vec<SearchResult>, crate::hnsw::SearchTrace)> {
        self.validate_query(query)?;
        let effective_k = self.clamp_k(k);
        if effective_k == 0 {
            return Ok((Vec::new(), crate::hnsw::SearchTrace::default()));
        }

        let (raw_results, trace) =
            self.index
                .search_with_trace(query, effective_k, self.vectors.as_slice())?;

        let results = self.enrich_results(raw_results)?;
        Ok((results, trace))
    }

    /// Search with metadata filter and detailed profiling.
    ///
    /// Combines filtered search with query execution profiling.
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, Filter};
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("test", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "a"})))?;
    /// collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "b"})))?;
    ///
    /// let filter = Filter::eq("type", "a");
    /// let (results, explain) = collection.search_with_filter_explain(
    ///     &[1.0, 0.0, 0.0, 0.0],
    ///     10,
    ///     &filter
    /// )?;
    ///
    /// println!("Filter reduced {} -> {} candidates",
    ///          explain.candidates_before_filter,
    ///          explain.candidates_after_filter);
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_with_filter_explain(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<(Vec<SearchResult>, SearchExplain)> {
        use std::time::Instant;

        let total_start = Instant::now();
        self.validate_query(query)?;

        let effective_k = self.clamp_k(k);
        if effective_k == 0 {
            let mut explain = SearchExplain {
                dimensions: self.config.dimensions,
                collection_size: self.len(),
                requested_k: k,
                effective_k: 0,
                ef_search: self.index.config().ef_search,
                filter_applied: true,
                distance_function: format!("{:?}", self.config.distance),
                ..Default::default()
            };
            explain.total_time_us = total_start.elapsed().as_micros() as u64;
            return Ok((Vec::new(), explain));
        }

        // HNSW index search with stats (fetch extra candidates for filtering)
        let index_start = Instant::now();
        let (candidates, hnsw_stats) = self.index.search_with_stats(
            query,
            effective_k * FILTER_CANDIDATE_MULTIPLIER,
            self.vectors.as_slice(),
        )?;
        let index_time = index_start.elapsed();

        let candidates_before_filter = candidates.len();

        // Apply filter
        let filter_start = Instant::now();
        let filtered: Vec<(crate::hnsw::VectorId, f32)> = candidates
            .into_iter()
            .filter(|(id, _)| {
                if let Some(entry) = self.metadata.get(*id) {
                    filter.matches(entry.data.as_ref())
                } else {
                    false
                }
            })
            .take(effective_k)
            .collect();
        let filter_time = filter_start.elapsed();

        let candidates_after_filter = filtered.len();

        // Enrich results with metadata
        let enrich_start = Instant::now();
        let results = self.enrich_results(filtered)?;
        let enrich_time = enrich_start.elapsed();

        let total_time = total_start.elapsed();

        let explain = SearchExplain {
            total_time_us: total_time.as_micros() as u64,
            index_time_us: index_time.as_micros() as u64,
            filter_time_us: filter_time.as_micros() as u64,
            enrich_time_us: enrich_time.as_micros() as u64,
            candidates_before_filter,
            candidates_after_filter,
            hnsw_stats,
            dimensions: self.config.dimensions,
            collection_size: self.len(),
            requested_k: k,
            effective_k,
            ef_search: self.index.config().ef_search,
            filter_applied: true,
            distance_function: format!("{:?}", self.config.distance),
        };

        Ok((results, explain))
    }

    /// Search and return only IDs (faster, no metadata lookup)
    pub fn search_ids(&self, query: &[f32], k: usize) -> Result<Vec<(String, f32)>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(query, k, self.vectors.as_slice())?;

        results
            .into_iter()
            .map(|(id, distance)| {
                let entry = self
                    .metadata
                    .get(id)
                    .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;
                Ok((entry.external_id.clone(), distance))
            })
            .collect()
    }

    /// Search and return borrowed ID references (zero-copy, fastest)
    ///
    /// This method returns references to the stored IDs rather than cloning them,
    /// making it the most efficient option when you only need IDs and distances.
    ///
    /// # Example
    /// ```
    /// # use needle::{Collection, CollectionConfig};
    /// # let config = CollectionConfig::new("test", 4);
    /// # let mut collection = Collection::new(config);
    /// # collection.insert("vec1", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
    /// let query = vec![1.0, 0.0, 0.0, 0.0];
    /// let results = collection.search_ids_ref(&query, 5).unwrap();
    /// for (id, distance) in results {
    ///     println!("{}: {}", id, distance);
    /// }
    /// ```
    pub fn search_ids_ref(&self, query: &[f32], k: usize) -> Result<Vec<(&str, f32)>> {
        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        let results = self.index.search(query, k, self.vectors.as_slice())?;

        results
            .into_iter()
            .map(|(id, distance)| {
                let entry = self
                    .metadata
                    .get(id)
                    .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;
                Ok((entry.external_id.as_str(), distance))
            })
            .collect()
    }

    /// Search for nearest neighbors with metadata filtering.
    ///
    /// Combines approximate nearest neighbor search with metadata-based
    /// pre-filtering. Only vectors matching the filter are returned.
    ///
    /// Internally, this fetches more candidates than requested (`k * FILTER_CANDIDATE_MULTIPLIER`)
    /// to compensate for filtered-out results, then filters and returns
    /// the top `k` matches.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `k` - Maximum number of results to return
    /// * `filter` - MongoDB-style filter for metadata (see [`Filter`])
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, Filter};
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("docs", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"type": "article"})))?;
    /// collection.insert("v2", &[0.9, 0.1, 0.0, 0.0], Some(json!({"type": "image"})))?;
    ///
    /// // Search only for articles
    /// let filter = Filter::eq("type", "article");
    /// let results = collection.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter)?;
    /// assert_eq!(results.len(), 1);
    /// assert_eq!(results[0].id, "v1");
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_with_filter(
        &self,
        query: &[f32],
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;
        let k = self.clamp_k(k);
        if k == 0 {
            return Ok(Vec::new());
        }

        // Bloom filter pre-check: if any equality condition references a
        // field+value that definitely doesn't exist, return early.
        let eq_conditions = filter.equality_conditions();
        for (field, value_str) in &eq_conditions {
            if !self.metadata.bloom_might_contain(field, value_str) {
                return Ok(Vec::new());
            }
        }

        // For filtered search, we need to retrieve more candidates and filter
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

        let results = self.enrich_results(filtered)?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    k = k,
                    results_count = results.len(),
                    collection_size = self.len(),
                    filter_applied = true,
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Find all vectors within a given distance from the query.
    ///
    /// Unlike top-k search which returns exactly k results regardless of distance,
    /// range queries return all vectors within `max_distance` from the query. This is
    /// useful for clustering, deduplication, or finding all semantically similar items.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return (caps output size)
    ///
    /// # Returns
    ///
    /// All vectors with `distance <= max_distance`, up to `limit` results,
    /// sorted by distance ascending.
    ///
    /// # Performance
    ///
    /// This method uses HNSW search internally with `limit` as the search bound.
    /// For very large ranges or small collections, consider using exact search instead.
    /// The method benefits from early termination when all nearby candidates exceed
    /// the distance threshold.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - [`NeedleError::DimensionMismatch`] - Query dimensions don't match
    /// - [`NeedleError::InvalidVector`] - Query contains NaN or Infinity
    ///
    /// # Example
    ///
    /// ```
    /// use needle::Collection;
    ///
    /// let mut collection = Collection::with_dimensions("embeddings", 4);
    /// collection.insert("v1", &[1.0, 0.0, 0.0, 0.0], None)?;
    /// collection.insert("v2", &[0.99, 0.1, 0.0, 0.0], None)?;  // Close to v1
    /// collection.insert("v3", &[0.0, 1.0, 0.0, 0.0], None)?;   // Far from v1
    ///
    /// // Find all vectors within distance 0.2 from the query
    /// let results = collection.search_radius(&[1.0, 0.0, 0.0, 0.0], 0.2, 100)?;
    ///
    /// // Only nearby vectors are returned
    /// for r in &results {
    ///     assert!(r.distance <= 0.2);
    /// }
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_radius(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;

        if limit == 0 || max_distance < 0.0 {
            return Ok(Vec::new());
        }

        // Search for up to limit candidates
        let candidates = self.index.search(query, limit, self.vectors.as_slice())?;

        // Filter to only include results within max_distance
        let within_range: Vec<(VectorId, f32)> = candidates
            .into_iter()
            .take_while(|(_, distance)| *distance <= max_distance)
            .collect();

        let results = self.enrich_results(within_range)?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    limit = limit,
                    max_distance = max_distance,
                    results_count = results.len(),
                    collection_size = self.len(),
                    query_type = "radius",
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Find all vectors within a given distance with metadata filtering.
    ///
    /// Combines range queries with metadata pre-filtering. Only vectors that
    /// both match the filter AND are within `max_distance` are returned.
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector (must match collection dimensions)
    /// * `max_distance` - Maximum distance threshold (inclusive)
    /// * `limit` - Maximum number of results to return
    /// * `filter` - MongoDB-style filter for metadata
    ///
    /// # Example
    ///
    /// ```
    /// use needle::{Collection, Filter};
    /// use serde_json::json;
    ///
    /// let mut collection = Collection::with_dimensions("products", 4);
    /// collection.insert("p1", &[1.0, 0.0, 0.0, 0.0], Some(json!({"category": "electronics"})))?;
    /// collection.insert("p2", &[0.99, 0.1, 0.0, 0.0], Some(json!({"category": "books"})))?;
    ///
    /// // Find electronics within distance 0.5
    /// let filter = Filter::eq("category", "electronics");
    /// let results = collection.search_radius_with_filter(&[1.0, 0.0, 0.0, 0.0], 0.5, 100, &filter)?;
    /// # Ok::<(), needle::NeedleError>(())
    /// ```
    pub fn search_radius_with_filter(
        &self,
        query: &[f32],
        max_distance: f32,
        limit: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        use std::time::Instant;
        let start = Instant::now();

        self.validate_query(query)?;

        if limit == 0 || max_distance < 0.0 {
            return Ok(Vec::new());
        }

        // Over-fetch to compensate for filtering
        let fetch_count = limit * FILTER_CANDIDATE_MULTIPLIER;
        let candidates = self
            .index
            .search(query, fetch_count, self.vectors.as_slice())?;

        // Filter by distance first (can stop early), then by metadata
        let within_range: Vec<(VectorId, f32)> = candidates
            .into_iter()
            .take_while(|(_, distance)| *distance <= max_distance)
            .filter(|(id, _)| {
                if let Some(entry) = self.metadata.get(*id) {
                    filter.matches(entry.data.as_ref())
                } else {
                    false
                }
            })
            .take(limit)
            .collect();

        let results = self.enrich_results(within_range)?;

        // Log slow queries if threshold is configured
        if let Some(threshold_us) = self.config.slow_query_threshold_us {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if elapsed_us > threshold_us {
                warn!(
                    collection = %self.config.name,
                    elapsed_us = elapsed_us,
                    threshold_us = threshold_us,
                    limit = limit,
                    max_distance = max_distance,
                    results_count = results.len(),
                    collection_size = self.len(),
                    query_type = "radius",
                    filter_applied = true,
                    "slow query detected"
                );
            }
        }

        Ok(results)
    }

    /// Convert internal results to SearchResult with metadata
    pub(super) fn enrich_results(&self, results: Vec<(VectorId, f32)>) -> Result<Vec<SearchResult>> {
        results
            .into_iter()
            .map(|(id, distance)| {
                let entry = self
                    .metadata
                    .get(id)
                    .ok_or_else(|| NeedleError::Index("Missing metadata for vector".into()))?;

                Ok(SearchResult {
                    id: entry.external_id.clone(),
                    distance,
                    metadata: entry.data.clone(),
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::random_vector;
    use serde_json::json;

    fn populated_collection(n: usize, dims: usize) -> Collection {
        let mut col = Collection::with_dimensions("test", dims);
        for i in 0..n {
            let vec = random_vector(dims);
            col.insert(format!("v{i}"), &vec, Some(json!({"idx": i})))
                .unwrap();
        }
        col
    }

    // ── Basic search ────────────────────────────────────────────────────

    #[test]
    fn test_search_returns_sorted_results() {
        let col = populated_collection(50, 16);
        let query = random_vector(16);
        let results = col.search(&query, 10).unwrap();
        assert!(results.len() <= 10);
        for w in results.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }

    #[test]
    fn test_search_k_zero_returns_empty() {
        let col = populated_collection(10, 8);
        let query = random_vector(8);
        let results = col.search(&query, 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_empty_collection() {
        let col = Collection::with_dimensions("empty", 8);
        let query = random_vector(8);
        let results = col.search(&query, 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_k_larger_than_collection() {
        let col = populated_collection(3, 8);
        let query = random_vector(8);
        let results = col.search(&query, 100).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_dimension_mismatch() {
        let col = populated_collection(5, 8);
        let query = random_vector(16);
        let result = col.search(&query, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_nan_query() {
        let col = populated_collection(5, 4);
        let query = vec![f32::NAN, 0.0, 0.0, 0.0];
        let result = col.search(&query, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_infinity_query() {
        let col = populated_collection(5, 4);
        let query = vec![f32::INFINITY, 0.0, 0.0, 0.0];
        let result = col.search(&query, 5);
        assert!(result.is_err());
    }

    // ── search_ids / search_ids_ref ─────────────────────────────────────

    #[test]
    fn test_search_ids_returns_tuples() {
        let col = populated_collection(20, 8);
        let query = random_vector(8);
        let results = col.search_ids(&query, 5).unwrap();
        assert!(results.len() <= 5);
        for (id, dist) in &results {
            assert!(!id.is_empty());
            assert!(*dist >= 0.0);
        }
    }

    #[test]
    fn test_search_ids_ref_returns_refs() {
        let col = populated_collection(20, 8);
        let query = random_vector(8);
        let results = col.search_ids_ref(&query, 5).unwrap();
        assert!(results.len() <= 5);
        for (id, dist) in &results {
            assert!(!id.is_empty());
            assert!(*dist >= 0.0);
        }
    }

    #[test]
    fn test_search_ids_k_zero() {
        let col = populated_collection(5, 4);
        let query = random_vector(4);
        assert!(col.search_ids(&query, 0).unwrap().is_empty());
        assert!(col.search_ids_ref(&query, 0).unwrap().is_empty());
    }

    // ── search_with_filter ──────────────────────────────────────────────

    #[test]
    fn test_search_with_filter_narrows_results() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], Some(json!({"cat": "x"}))).unwrap();
        col.insert("b", &[0.9, 0.1, 0.0, 0.0], Some(json!({"cat": "y"}))).unwrap();
        col.insert("c", &[0.8, 0.2, 0.0, 0.0], Some(json!({"cat": "x"}))).unwrap();

        let filter = Filter::eq("cat", "x");
        let results = col.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter).unwrap();
        assert!(results.iter().all(|r| r.id == "a" || r.id == "c"));
    }

    #[test]
    fn test_search_with_filter_no_match() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], Some(json!({"cat": "x"}))).unwrap();

        let filter = Filter::eq("cat", "nonexistent");
        let results = col.search_with_filter(&[1.0, 0.0, 0.0, 0.0], 10, &filter).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_with_filter_dimension_mismatch() {
        let col = populated_collection(5, 4);
        let filter = Filter::eq("cat", "x");
        let result = col.search_with_filter(&[1.0, 0.0], 10, &filter);
        assert!(result.is_err());
    }

    // ── search_explain ──────────────────────────────────────────────────

    #[test]
    fn test_search_explain_returns_valid_stats() {
        let col = populated_collection(50, 8);
        let query = random_vector(8);
        let (results, explain) = col.search_explain(&query, 5).unwrap();
        assert!(results.len() <= 5);
        assert_eq!(explain.dimensions, 8);
        assert_eq!(explain.collection_size, 50);
        assert_eq!(explain.requested_k, 5);
        assert!(explain.effective_k <= 50);
        assert!(!explain.filter_applied);
    }

    #[test]
    fn test_search_explain_k_zero() {
        let col = populated_collection(10, 4);
        let query = random_vector(4);
        let (results, explain) = col.search_explain(&query, 0).unwrap();
        assert!(results.is_empty());
        assert_eq!(explain.effective_k, 0);
    }

    #[test]
    fn test_search_with_filter_explain_shows_filter() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], Some(json!({"cat": "x"}))).unwrap();
        col.insert("b", &[0.0, 1.0, 0.0, 0.0], Some(json!({"cat": "y"}))).unwrap();

        let filter = Filter::eq("cat", "x");
        let (results, explain) = col.search_with_filter_explain(
            &[1.0, 0.0, 0.0, 0.0], 10, &filter
        ).unwrap();
        assert!(explain.filter_applied);
        assert!(results.len() <= 1);
    }

    // ── search_radius ───────────────────────────────────────────────────

    #[test]
    fn test_search_radius_basic() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("close", &[1.0, 0.0, 0.0, 0.0], None).unwrap();
        col.insert("far", &[0.0, 0.0, 0.0, 1.0], None).unwrap();

        let results = col.search_radius(&[1.0, 0.0, 0.0, 0.0], 0.1, 100).unwrap();
        for r in &results {
            assert!(r.distance <= 0.1);
        }
    }

    #[test]
    fn test_search_radius_negative_distance_returns_empty() {
        let col = populated_collection(10, 4);
        let query = random_vector(4);
        let results = col.search_radius(&query, -1.0, 100).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_radius_zero_limit() {
        let col = populated_collection(10, 4);
        let query = random_vector(4);
        let results = col.search_radius(&query, 1.0, 0).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_radius_with_filter() {
        let mut col = Collection::with_dimensions("test", 4);
        col.insert("a", &[1.0, 0.0, 0.0, 0.0], Some(json!({"t": "x"}))).unwrap();
        col.insert("b", &[0.99, 0.1, 0.0, 0.0], Some(json!({"t": "y"}))).unwrap();

        let filter = Filter::eq("t", "x");
        let results = col.search_radius_with_filter(
            &[1.0, 0.0, 0.0, 0.0], 0.5, 100, &filter
        ).unwrap();
        assert!(results.iter().all(|r| r.id == "a"));
    }

    // ── search_matryoshka ───────────────────────────────────────────────

    #[test]
    fn test_search_matryoshka_basic() {
        let col = populated_collection(50, 16);
        let query = random_vector(16);
        let results = col.search_matryoshka(&query, 5, 8, 4).unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_matryoshka_coarse_dims_zero_falls_back() {
        let col = populated_collection(10, 8);
        let query = random_vector(8);
        let results = col.search_matryoshka(&query, 5, 0, 4).unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_matryoshka_coarse_dims_gte_full_falls_back() {
        let col = populated_collection(10, 8);
        let query = random_vector(8);
        let results = col.search_matryoshka(&query, 5, 8, 4).unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_matryoshka_k_zero() {
        let col = populated_collection(10, 8);
        let query = random_vector(8);
        let results = col.search_matryoshka(&query, 0, 4, 4).unwrap();
        assert!(results.is_empty());
    }

    // ── search_with_trace ───────────────────────────────────────────────

    #[test]
    fn test_search_with_trace_returns_trace() {
        let col = populated_collection(30, 8);
        let query = random_vector(8);
        let (results, _trace) = col.search_with_trace(&query, 5).unwrap();
        assert!(results.len() <= 5);
    }

    #[test]
    fn test_search_with_trace_k_zero() {
        let col = populated_collection(10, 4);
        let query = random_vector(4);
        let (results, trace) = col.search_with_trace(&query, 0).unwrap();
        assert!(results.is_empty());
        assert!(trace.hops.is_empty());
    }

    // ── search_builder ──────────────────────────────────────────────────

    #[test]
    fn test_search_builder_basic() {
        let col = populated_collection(30, 8);
        let query = random_vector(8);
        let results = col.search_builder(&query).k(5).execute().unwrap();
        assert!(results.len() <= 5);
    }

    // ── enrich_results ──────────────────────────────────────────────────

    #[test]
    fn test_enrich_results_empty() {
        let col = Collection::with_dimensions("test", 4);
        let results = col.enrich_results(vec![]).unwrap();
        assert!(results.is_empty());
    }
}
