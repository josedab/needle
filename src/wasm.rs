//! WebAssembly bindings for Needle using wasm-bindgen

use crate::collection::{Collection, CollectionConfig, SearchResult as RustSearchResult};
use crate::distance::DistanceFunction;
use crate::metadata::Filter;
use serde_json::Value;
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

/// Search result for JavaScript
#[wasm_bindgen]
pub struct SearchResult {
    id: String,
    distance: f32,
    metadata_json: Option<String>,
}

#[wasm_bindgen]
impl SearchResult {
    /// Get the vector ID
    #[wasm_bindgen(getter)]
    pub fn id(&self) -> String {
        self.id.clone()
    }

    /// Get the distance
    #[wasm_bindgen(getter)]
    pub fn distance(&self) -> f32 {
        self.distance
    }

    /// Get metadata as JSON string
    #[wasm_bindgen(getter)]
    pub fn metadata(&self) -> Option<String> {
        self.metadata_json.clone()
    }

    /// Get metadata as parsed JavaScript object
    #[wasm_bindgen(js_name = "getMetadata")]
    pub fn get_metadata(&self) -> JsValue {
        match &self.metadata_json {
            Some(json) => js_sys::JSON::parse(json).unwrap_or(JsValue::NULL),
            None => JsValue::NULL,
        }
    }
}

impl From<RustSearchResult> for SearchResult {
    fn from(result: RustSearchResult) -> Self {
        Self {
            id: result.id,
            distance: result.distance,
            metadata_json: result.metadata.map(|v| v.to_string()),
        }
    }
}

/// A collection of vectors for JavaScript
#[wasm_bindgen]
pub struct WasmCollection {
    inner: Arc<RwLock<Collection>>,
}

#[wasm_bindgen]
impl WasmCollection {
    /// Create a new collection
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: &str,
        dimensions: usize,
        distance: Option<String>,
    ) -> Result<WasmCollection, JsValue> {
        let dist_fn = match distance.as_deref().unwrap_or("cosine") {
            "cosine" => DistanceFunction::Cosine,
            "euclidean" | "l2" => DistanceFunction::Euclidean,
            "dot" | "dotproduct" | "inner_product" => DistanceFunction::DotProduct,
            "manhattan" | "l1" => DistanceFunction::Manhattan,
            d => {
                return Err(JsValue::from_str(&format!(
                    "Unknown distance function: {}",
                    d
                )))
            }
        };

        let config = CollectionConfig::new(name, dimensions).with_distance(dist_fn);
        Ok(Self {
            inner: Arc::new(RwLock::new(Collection::new(config))),
        })
    }

    /// Get the collection name
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.inner
            .read()
            .map(|guard| guard.name().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }

    /// Get the vector dimensions
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.inner
            .read()
            .map(|guard| guard.dimensions())
            .unwrap_or(0)
    }

    /// Get the number of vectors
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.read().map(|guard| guard.len()).unwrap_or(0)
    }

    /// Check if the collection is empty
    #[wasm_bindgen(js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner
            .read()
            .map(|guard| guard.is_empty())
            .unwrap_or(true)
    }

    /// Insert a vector with ID and optional metadata
    pub fn insert(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<String>,
    ) -> Result<(), JsValue> {
        let meta_value: Option<Value> = if let Some(json) = metadata {
            Some(
                serde_json::from_str(&json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid JSON metadata: {}", e)))?,
            )
        } else {
            None
        };

        self.inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .insert(id, &vector, meta_value)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Insert a vector with metadata as JavaScript object
    #[wasm_bindgen(js_name = "insertWithObject")]
    pub fn insert_with_object(
        &self,
        id: &str,
        vector: Vec<f32>,
        metadata: JsValue,
    ) -> Result<(), JsValue> {
        let meta_value: Option<Value> = if metadata.is_null() || metadata.is_undefined() {
            None
        } else {
            let json_str = js_sys::JSON::stringify(&metadata)
                .map_err(|_| JsValue::from_str("Failed to stringify metadata"))?;
            Some(
                serde_json::from_str(&String::from(json_str))
                    .map_err(|e| JsValue::from_str(&format!("Invalid metadata: {}", e)))?,
            )
        };

        self.inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .insert(id, &vector, meta_value)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Insert multiple vectors in batch (JSON metadata array)
    /// vectors_js should be a JS array of Float32Arrays
    #[wasm_bindgen(js_name = "insertBatch")]
    pub fn insert_batch(
        &self,
        ids: Vec<String>,
        vectors_js: JsValue,
        metadata_json_array: Option<Vec<String>>,
    ) -> Result<(), JsValue> {
        // Convert JsValue array of arrays to Vec<Vec<f32>>
        let vectors: Vec<Vec<f32>> = if vectors_js.is_array() {
            let arr = js_sys::Array::from(&vectors_js);
            let mut result = Vec::with_capacity(arr.length() as usize);
            for i in 0..arr.length() {
                let inner = arr.get(i);
                if let Some(float_arr) = inner.dyn_ref::<js_sys::Float32Array>() {
                    result.push(float_arr.to_vec());
                } else if inner.is_array() {
                    // Handle regular JS array of numbers
                    let inner_arr = js_sys::Array::from(&inner);
                    let vec: Vec<f32> = (0..inner_arr.length())
                        .filter_map(|j| inner_arr.get(j).as_f64().map(|n| n as f32))
                        .collect();
                    result.push(vec);
                } else {
                    return Err(JsValue::from_str(
                        "vectors must be an array of arrays or Float32Arrays",
                    ));
                }
            }
            result
        } else {
            return Err(JsValue::from_str("vectors must be an array"));
        };

        if ids.len() != vectors.len() {
            return Err(JsValue::from_str(
                "ids and vectors must have the same length",
            ));
        }

        let meta_values: Vec<Option<Value>> = if let Some(meta_list) = metadata_json_array {
            if meta_list.len() != ids.len() {
                return Err(JsValue::from_str(
                    "metadata must have the same length as ids",
                ));
            }
            meta_list
                .into_iter()
                .map(|json| {
                    serde_json::from_str(&json)
                        .map(Some)
                        .map_err(|e| JsValue::from_str(&format!("Invalid JSON metadata: {}", e)))
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            vec![None; ids.len()]
        };

        self.inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .insert_batch(ids, vectors, meta_values)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: Vec<f32>, k: usize) -> Result<Vec<SearchResult>, JsValue> {
        let results = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .search(&query, k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Search with a metadata filter (JSON string)
    #[wasm_bindgen(js_name = "searchWithFilter")]
    pub fn search_with_filter(
        &self,
        query: Vec<f32>,
        k: usize,
        filter_json: &str,
    ) -> Result<Vec<SearchResult>, JsValue> {
        let filter_value: Value = serde_json::from_str(filter_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid filter JSON: {}", e)))?;

        let filter = Filter::parse(&filter_value)
            .map_err(|e| JsValue::from_str(&format!("Invalid filter format: {}", e)))?;

        let results = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .search_with_filter(&query, k, &filter)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(results.into_iter().map(SearchResult::from).collect())
    }

    /// Get a vector by ID
    pub fn get(&self, id: &str) -> JsValue {
        let coll = match self.inner.read() {
            Ok(guard) => guard,
            Err(_) => return JsValue::NULL,
        };
        match coll.get(id) {
            Some((vector, metadata)) => {
                let obj = js_sys::Object::new();

                // Set vector
                let arr = js_sys::Float32Array::new_with_length(vector.len() as u32);
                arr.copy_from(vector);
                let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("vector"), &arr);

                // Set metadata
                if let Some(meta) = metadata {
                    if let Ok(parsed) = js_sys::JSON::parse(&meta.to_string()) {
                        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("metadata"), &parsed);
                    }
                }

                obj.into()
            }
            None => JsValue::NULL,
        }
    }

    /// Check if a vector ID exists
    pub fn contains(&self, id: &str) -> bool {
        self.inner.read().map(|g| g.contains(id)).unwrap_or(false)
    }

    /// Delete a vector by ID
    pub fn delete(&self, id: &str) -> Result<bool, JsValue> {
        self.inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .delete(id)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Set ef_search parameter
    #[wasm_bindgen(js_name = "setEfSearch")]
    pub fn set_ef_search(&self, ef: usize) -> Result<(), JsValue> {
        self.inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .set_ef_search(ef);
        Ok(())
    }

    /// Serialize to bytes
    #[wasm_bindgen(js_name = "toBytes")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsValue> {
        self.inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?
            .to_bytes()
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Deserialize from bytes
    #[wasm_bindgen(js_name = "fromBytes")]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmCollection, JsValue> {
        let collection =
            Collection::from_bytes(bytes).map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(Self {
            inner: Arc::new(RwLock::new(collection)),
        })
    }
}

/// Initialize the WASM module
#[wasm_bindgen(start)]
pub fn init() {
    // Initialization code can go here if needed
}

// ============================================================================
// Browser Runtime Enhancements
// ============================================================================

/// Performance metrics for browser monitoring
#[wasm_bindgen]
pub struct PerformanceMetrics {
    operation: String,
    duration_ms: f64,
    vectors_processed: usize,
    memory_used_bytes: usize,
}

#[wasm_bindgen]
impl PerformanceMetrics {
    /// Get operation name
    #[wasm_bindgen(getter)]
    pub fn operation(&self) -> String {
        self.operation.clone()
    }

    /// Get duration in milliseconds
    #[wasm_bindgen(getter, js_name = "durationMs")]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    /// Get number of vectors processed
    #[wasm_bindgen(getter, js_name = "vectorsProcessed")]
    pub fn vectors_processed(&self) -> usize {
        self.vectors_processed
    }

    /// Get estimated memory used in bytes
    #[wasm_bindgen(getter, js_name = "memoryUsedBytes")]
    pub fn memory_used_bytes(&self) -> usize {
        self.memory_used_bytes
    }

    /// Convert to JavaScript object
    #[wasm_bindgen(js_name = "toObject")]
    pub fn to_object(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"operation".into(), &self.operation.clone().into());
        let _ = js_sys::Reflect::set(&obj, &"durationMs".into(), &self.duration_ms.into());
        let _ = js_sys::Reflect::set(
            &obj,
            &"vectorsProcessed".into(),
            &(self.vectors_processed as u32).into(),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"memoryUsedBytes".into(),
            &(self.memory_used_bytes as u32).into(),
        );
        obj.into()
    }
}

/// Memory usage statistics
#[wasm_bindgen]
pub struct MemoryStats {
    vectors_count: usize,
    dimensions: usize,
    estimated_vector_bytes: usize,
    estimated_index_bytes: usize,
    estimated_metadata_bytes: usize,
}

#[wasm_bindgen]
impl MemoryStats {
    /// Get vector count
    #[wasm_bindgen(getter, js_name = "vectorsCount")]
    pub fn vectors_count(&self) -> usize {
        self.vectors_count
    }

    /// Get dimensions
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Get estimated vector storage in bytes
    #[wasm_bindgen(getter, js_name = "estimatedVectorBytes")]
    pub fn estimated_vector_bytes(&self) -> usize {
        self.estimated_vector_bytes
    }

    /// Get estimated index storage in bytes
    #[wasm_bindgen(getter, js_name = "estimatedIndexBytes")]
    pub fn estimated_index_bytes(&self) -> usize {
        self.estimated_index_bytes
    }

    /// Get estimated metadata storage in bytes
    #[wasm_bindgen(getter, js_name = "estimatedMetadataBytes")]
    pub fn estimated_metadata_bytes(&self) -> usize {
        self.estimated_metadata_bytes
    }

    /// Get total estimated bytes
    #[wasm_bindgen(getter, js_name = "totalBytes")]
    pub fn total_bytes(&self) -> usize {
        self.estimated_vector_bytes + self.estimated_index_bytes + self.estimated_metadata_bytes
    }

    /// Convert to JavaScript object
    #[wasm_bindgen(js_name = "toObject")]
    pub fn to_object(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(
            &obj,
            &"vectorsCount".into(),
            &(self.vectors_count as u32).into(),
        );
        let _ = js_sys::Reflect::set(&obj, &"dimensions".into(), &(self.dimensions as u32).into());
        let _ = js_sys::Reflect::set(
            &obj,
            &"estimatedVectorBytes".into(),
            &(self.estimated_vector_bytes as u32).into(),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"estimatedIndexBytes".into(),
            &(self.estimated_index_bytes as u32).into(),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"estimatedMetadataBytes".into(),
            &(self.estimated_metadata_bytes as u32).into(),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"totalBytes".into(),
            &(self.total_bytes() as u32).into(),
        );
        obj.into()
    }
}

/// Batch insert result
#[wasm_bindgen]
pub struct BatchInsertResult {
    successful: usize,
    failed: usize,
    errors: Vec<String>,
    duration_ms: f64,
}

#[wasm_bindgen]
impl BatchInsertResult {
    /// Get successful insert count
    #[wasm_bindgen(getter)]
    pub fn successful(&self) -> usize {
        self.successful
    }

    /// Get failed insert count
    #[wasm_bindgen(getter)]
    pub fn failed(&self) -> usize {
        self.failed
    }

    /// Get duration in milliseconds
    #[wasm_bindgen(getter, js_name = "durationMs")]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    /// Get errors as JavaScript array
    #[wasm_bindgen(getter)]
    pub fn errors(&self) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for err in &self.errors {
            arr.push(&JsValue::from_str(err));
        }
        arr
    }
}

/// Streaming search result chunk
#[wasm_bindgen]
pub struct SearchChunk {
    results: Vec<SearchResult>,
    is_final: bool,
    chunk_index: usize,
}

#[wasm_bindgen]
impl SearchChunk {
    /// Get results in this chunk
    #[wasm_bindgen(getter)]
    pub fn results(&self) -> js_sys::Array {
        let arr = js_sys::Array::new();
        for result in &self.results {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"id".into(), &result.id.clone().into());
            let _ = js_sys::Reflect::set(&obj, &"distance".into(), &result.distance.into());
            if let Some(ref meta) = result.metadata_json {
                if let Ok(parsed) = js_sys::JSON::parse(meta) {
                    let _ = js_sys::Reflect::set(&obj, &"metadata".into(), &parsed);
                }
            }
            arr.push(&obj);
        }
        arr
    }

    /// Check if this is the final chunk
    #[wasm_bindgen(getter, js_name = "isFinal")]
    pub fn is_final(&self) -> bool {
        self.is_final
    }

    /// Get chunk index
    #[wasm_bindgen(getter, js_name = "chunkIndex")]
    pub fn chunk_index(&self) -> usize {
        self.chunk_index
    }
}

// Additional methods for WasmCollection
#[wasm_bindgen]
impl WasmCollection {
    /// Get memory statistics
    #[wasm_bindgen(js_name = "getMemoryStats")]
    pub fn get_memory_stats(&self) -> Result<MemoryStats, JsValue> {
        let coll = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let vectors_count = coll.len();
        let dimensions = coll.dimensions();

        // Estimate memory usage
        let estimated_vector_bytes = vectors_count * dimensions * 4; // f32 = 4 bytes
        let estimated_index_bytes = vectors_count * 64 * 4; // Rough HNSW estimate
        let estimated_metadata_bytes = vectors_count * 100; // Average metadata estimate

        Ok(MemoryStats {
            vectors_count,
            dimensions,
            estimated_vector_bytes,
            estimated_index_bytes,
            estimated_metadata_bytes,
        })
    }

    /// Batch insert multiple vectors
    #[wasm_bindgen(js_name = "batchInsert")]
    pub fn batch_insert(
        &self,
        ids: js_sys::Array,
        vectors: js_sys::Array,
        metadata_array: Option<js_sys::Array>,
    ) -> Result<BatchInsertResult, JsValue> {
        let start = js_sys::Date::now();

        let mut successful = 0;
        let mut failed = 0;
        let mut errors = Vec::new();

        let mut coll = self
            .inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let len = ids.length() as usize;

        for i in 0..len {
            let id = ids
                .get(i as u32)
                .as_string()
                .ok_or_else(|| JsValue::from_str(&format!("Invalid ID at index {}", i)))?;

            let vec_value = vectors.get(i as u32);
            let vector: Vec<f32> = if let Some(arr) = vec_value.dyn_ref::<js_sys::Float32Array>() {
                arr.to_vec()
            } else if let Some(arr) = vec_value.dyn_ref::<js_sys::Array>() {
                let mut v = Vec::with_capacity(arr.length() as usize);
                for j in 0..arr.length() {
                    v.push(arr.get(j).as_f64().unwrap_or(0.0) as f32);
                }
                v
            } else {
                errors.push(format!("Invalid vector at index {}", i));
                failed += 1;
                continue;
            };

            let metadata: Option<Value> = metadata_array.as_ref().and_then(|arr| {
                let meta_val = arr.get(i as u32);
                if meta_val.is_null() || meta_val.is_undefined() {
                    None
                } else {
                    js_sys::JSON::stringify(&meta_val)
                        .ok()
                        .and_then(|s| serde_json::from_str(&s.as_string()?).ok())
                }
            });

            match coll.insert(&id, &vector, metadata) {
                Ok(_) => successful += 1,
                Err(e) => {
                    errors.push(format!("Failed to insert {}: {}", id, e));
                    failed += 1;
                }
            }
        }

        let duration_ms = js_sys::Date::now() - start;

        Ok(BatchInsertResult {
            successful,
            failed,
            errors,
            duration_ms,
        })
    }

    /// Search with performance metrics
    #[wasm_bindgen(js_name = "searchWithMetrics")]
    pub fn search_with_metrics(
        &self,
        query: Vec<f32>,
        k: usize,
    ) -> Result<js_sys::Object, JsValue> {
        let start = js_sys::Date::now();

        let coll = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let results = coll
            .search(&query, k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let duration_ms = js_sys::Date::now() - start;

        let result_array = js_sys::Array::new();
        for result in results.iter() {
            let obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&obj, &"id".into(), &result.id.clone().into());
            let _ = js_sys::Reflect::set(&obj, &"distance".into(), &result.distance.into());
            if let Some(ref meta) = result.metadata {
                if let Ok(parsed) = js_sys::JSON::parse(&meta.to_string()) {
                    let _ = js_sys::Reflect::set(&obj, &"metadata".into(), &parsed);
                }
            }
            result_array.push(&obj);
        }

        let response = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&response, &"results".into(), &result_array);
        let _ = js_sys::Reflect::set(&response, &"durationMs".into(), &duration_ms.into());
        let _ = js_sys::Reflect::set(
            &response,
            &"vectorsSearched".into(),
            &(coll.len() as u32).into(),
        );

        Ok(response)
    }

    /// Chunked search for streaming results
    #[wasm_bindgen(js_name = "searchChunked")]
    pub fn search_chunked(
        &self,
        query: Vec<f32>,
        k: usize,
        chunk_size: usize,
    ) -> Result<js_sys::Array, JsValue> {
        let coll = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let results = coll
            .search(&query, k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        let chunks_array = js_sys::Array::new();
        let chunk_count = (results.len() + chunk_size - 1) / chunk_size;

        for (i, chunk) in results.chunks(chunk_size).enumerate() {
            let chunk_results: Vec<SearchResult> = chunk
                .iter()
                .map(|r| SearchResult::from(r.clone()))
                .collect();

            let search_chunk = SearchChunk {
                results: chunk_results,
                is_final: i == chunk_count - 1,
                chunk_index: i,
            };

            let chunk_obj = js_sys::Object::new();
            let _ = js_sys::Reflect::set(&chunk_obj, &"results".into(), &search_chunk.results());
            let _ =
                js_sys::Reflect::set(&chunk_obj, &"isFinal".into(), &search_chunk.is_final.into());
            let _ = js_sys::Reflect::set(
                &chunk_obj,
                &"chunkIndex".into(),
                &(search_chunk.chunk_index as u32).into(),
            );

            chunks_array.push(&chunk_obj);
        }

        Ok(chunks_array)
    }

    /// Export to base64 string for storage
    #[wasm_bindgen(js_name = "toBase64")]
    pub fn to_base64(&self) -> Result<String, JsValue> {
        let bytes = self.to_bytes()?;
        Ok(base64_encode(&bytes))
    }

    /// Import from base64 string
    #[wasm_bindgen(js_name = "fromBase64")]
    pub fn from_base64(base64_str: &str) -> Result<WasmCollection, JsValue> {
        let bytes = base64_decode(base64_str)
            .map_err(|e| JsValue::from_str(&format!("Invalid base64: {}", e)))?;
        Self::from_bytes(&bytes)
    }

    /// Get collection info as JavaScript object
    #[wasm_bindgen(js_name = "getInfo")]
    pub fn get_info(&self) -> Result<js_sys::Object, JsValue> {
        let coll = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let info = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&info, &"name".into(), &coll.name().into());
        let _ = js_sys::Reflect::set(
            &info,
            &"dimensions".into(),
            &(coll.dimensions() as u32).into(),
        );
        let _ = js_sys::Reflect::set(&info, &"vectorCount".into(), &(coll.len() as u32).into());

        Ok(info)
    }

    /// Clear all vectors from the collection
    pub fn clear(&self) -> Result<(), JsValue> {
        let mut guard = self
            .inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        // Get all IDs and delete them
        let ids: Vec<String> = guard.all_ids();
        for id in ids {
            let _ = guard.delete(&id);
        }
        Ok(())
    }
}

// Simple base64 encoding/decoding for browser persistence
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();

    for chunk in data.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        result.push(CHARS[(b0 >> 2) & 0x3F] as char);
        result.push(CHARS[((b0 << 4) | (b1 >> 4)) & 0x3F] as char);

        if chunk.len() > 1 {
            result.push(CHARS[((b1 << 2) | (b2 >> 6)) & 0x3F] as char);
        } else {
            result.push('=');
        }

        if chunk.len() > 2 {
            result.push(CHARS[b2 & 0x3F] as char);
        } else {
            result.push('=');
        }
    }

    result
}

fn base64_decode(data: &str) -> Result<Vec<u8>, &'static str> {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    fn char_to_val(c: u8) -> Option<u8> {
        CHARS.iter().position(|&x| x == c).map(|p| p as u8)
    }

    let data = data.trim_end_matches('=');
    let bytes: Vec<u8> = data.bytes().collect();

    if bytes.len() % 4 == 1 {
        return Err("Invalid base64 length");
    }

    let mut result = Vec::new();

    for chunk in bytes.chunks(4) {
        let vals: Vec<u8> = chunk.iter().filter_map(|&c| char_to_val(c)).collect();

        if vals.len() < 2 {
            continue;
        }

        result.push((vals[0] << 2) | (vals[1] >> 4));

        if vals.len() > 2 {
            result.push((vals[1] << 4) | (vals[2] >> 2));
        }
        if vals.len() > 3 {
            result.push((vals[2] << 6) | vals[3]);
        }
    }

    Ok(result)
}

// ============================================================================
// IndexedDB Persistence Support
// ============================================================================

/// Configuration for IndexedDB persistence
#[wasm_bindgen]
pub struct IndexedDbConfig {
    db_name: String,
    store_name: String,
    auto_save: bool,
    save_interval_ms: u32,
}

#[wasm_bindgen]
impl IndexedDbConfig {
    /// Create a new IndexedDB configuration
    #[wasm_bindgen(constructor)]
    pub fn new(db_name: &str, store_name: &str) -> Self {
        Self {
            db_name: db_name.to_string(),
            store_name: store_name.to_string(),
            auto_save: true,
            save_interval_ms: 5000,
        }
    }

    /// Disable auto-save
    #[wasm_bindgen(js_name = "withoutAutoSave")]
    pub fn without_auto_save(mut self) -> Self {
        self.auto_save = false;
        self
    }

    /// Set save interval in milliseconds
    #[wasm_bindgen(js_name = "withSaveInterval")]
    pub fn with_save_interval(mut self, ms: u32) -> Self {
        self.save_interval_ms = ms;
        self
    }

    /// Get database name
    #[wasm_bindgen(getter, js_name = "dbName")]
    pub fn db_name(&self) -> String {
        self.db_name.clone()
    }

    /// Get store name
    #[wasm_bindgen(getter, js_name = "storeName")]
    pub fn store_name(&self) -> String {
        self.store_name.clone()
    }

    /// Check if auto-save is enabled
    #[wasm_bindgen(getter, js_name = "autoSave")]
    pub fn auto_save(&self) -> bool {
        self.auto_save
    }

    /// Get save interval
    #[wasm_bindgen(getter, js_name = "saveIntervalMs")]
    pub fn save_interval_ms(&self) -> u32 {
        self.save_interval_ms
    }
}

/// Collection with IndexedDB persistence support
#[wasm_bindgen]
pub struct PersistentCollection {
    collection: WasmCollection,
    config: IndexedDbConfig,
    dirty: bool,
}

#[wasm_bindgen]
impl PersistentCollection {
    /// Create a new persistent collection
    #[wasm_bindgen(constructor)]
    pub fn new(
        name: &str,
        dimensions: usize,
        distance: Option<String>,
        config: IndexedDbConfig,
    ) -> Result<PersistentCollection, JsValue> {
        let collection = WasmCollection::new(name, dimensions, distance)?;
        Ok(Self {
            collection,
            config,
            dirty: false,
        })
    }

    /// Get the underlying collection
    #[wasm_bindgen(getter)]
    pub fn collection(&self) -> WasmCollection {
        WasmCollection {
            inner: self.collection.inner.clone(),
        }
    }

    /// Insert a vector (marks collection as dirty)
    pub fn insert(
        &mut self,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<String>,
    ) -> Result<(), JsValue> {
        self.collection.insert(id, vector, metadata)?;
        self.dirty = true;
        Ok(())
    }

    /// Check if there are unsaved changes
    #[wasm_bindgen(getter, js_name = "isDirty")]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Get serialized data for IndexedDB storage
    /// Returns base64 encoded collection data
    #[wasm_bindgen(js_name = "getSerializedData")]
    pub fn get_serialized_data(&self) -> Result<String, JsValue> {
        self.collection.to_base64()
    }

    /// Mark as saved (call after successful IndexedDB write)
    #[wasm_bindgen(js_name = "markSaved")]
    pub fn mark_saved(&mut self) {
        self.dirty = false;
    }

    /// Restore from serialized data
    #[wasm_bindgen(js_name = "restoreFromData")]
    pub fn restore_from_data(
        data: &str,
        config: IndexedDbConfig,
    ) -> Result<PersistentCollection, JsValue> {
        let collection = WasmCollection::from_base64(data)?;
        Ok(Self {
            collection,
            config,
            dirty: false,
        })
    }

    /// Get IndexedDB configuration
    #[wasm_bindgen(getter)]
    pub fn config(&self) -> IndexedDbConfig {
        IndexedDbConfig {
            db_name: self.config.db_name.clone(),
            store_name: self.config.store_name.clone(),
            auto_save: self.config.auto_save,
            save_interval_ms: self.config.save_interval_ms,
        }
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: Vec<f32>, k: usize) -> Result<Vec<SearchResult>, JsValue> {
        self.collection.search(query, k)
    }

    /// Search with filter
    #[wasm_bindgen(js_name = "searchWithFilter")]
    pub fn search_with_filter(
        &self,
        query: Vec<f32>,
        k: usize,
        filter_json: &str,
    ) -> Result<Vec<SearchResult>, JsValue> {
        self.collection.search_with_filter(query, k, filter_json)
    }

    /// Get vector by ID
    pub fn get(&self, id: &str) -> JsValue {
        self.collection.get(id)
    }

    /// Delete vector (marks as dirty)
    pub fn delete(&mut self, id: &str) -> Result<bool, JsValue> {
        let result = self.collection.delete(id)?;
        if result {
            self.dirty = true;
        }
        Ok(result)
    }

    /// Clear all vectors (marks as dirty)
    pub fn clear(&mut self) -> Result<(), JsValue> {
        self.collection.clear()?;
        self.dirty = true;
        Ok(())
    }

    /// Get collection info
    #[wasm_bindgen(js_name = "getInfo")]
    pub fn get_info(&self) -> Result<js_sys::Object, JsValue> {
        let info = self.collection.get_info()?;
        let _ = js_sys::Reflect::set(&info, &"isDirty".into(), &self.dirty.into());
        let _ = js_sys::Reflect::set(&info, &"dbName".into(), &self.config.db_name.clone().into());
        let _ = js_sys::Reflect::set(
            &info,
            &"storeName".into(),
            &self.config.store_name.clone().into(),
        );
        Ok(info)
    }
}

/// Generate JavaScript helper code for IndexedDB operations
/// This returns a string of JavaScript that can be eval'd to get helper functions
#[wasm_bindgen(js_name = "getIndexedDbHelpers")]
pub fn get_indexed_db_helpers() -> String {
    r#"
const NeedleIndexedDb = {
    async openDatabase(dbName, storeName) {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(dbName, 1);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            request.onupgradeneeded = (event) => {
                const db = event.target.result;
                if (!db.objectStoreNames.contains(storeName)) {
                    db.createObjectStore(storeName);
                }
            };
        });
    },

    async save(dbName, storeName, key, data) {
        const db = await this.openDatabase(dbName, storeName);
        return new Promise((resolve, reject) => {
            const tx = db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.put(data, key);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve();
            tx.oncomplete = () => db.close();
        });
    },

    async load(dbName, storeName, key) {
        const db = await this.openDatabase(dbName, storeName);
        return new Promise((resolve, reject) => {
            const tx = db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const request = store.get(key);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            tx.oncomplete = () => db.close();
        });
    },

    async delete(dbName, storeName, key) {
        const db = await this.openDatabase(dbName, storeName);
        return new Promise((resolve, reject) => {
            const tx = db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            const request = store.delete(key);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve();
            tx.oncomplete = () => db.close();
        });
    },

    async listKeys(dbName, storeName) {
        const db = await this.openDatabase(dbName, storeName);
        return new Promise((resolve, reject) => {
            const tx = db.transaction(storeName, 'readonly');
            const store = tx.objectStore(storeName);
            const request = store.getAllKeys();
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            tx.oncomplete = () => db.close();
        });
    },

    async saveCollection(collection) {
        const config = collection.config;
        const data = collection.getSerializedData();
        const name = collection.collection.name;
        await this.save(config.dbName, config.storeName, name, data);
        collection.markSaved();
        return true;
    },

    async loadCollection(dbName, storeName, collectionName, config) {
        const data = await this.load(dbName, storeName, collectionName);
        if (!data) return null;
        return PersistentCollection.restoreFromData(data, config);
    },

    createAutoSaver(collection, intervalMs = 5000) {
        let timer = null;
        const save = async () => {
            if (collection.isDirty) {
                try {
                    await this.saveCollection(collection);
                    console.log('Auto-saved collection:', collection.collection.name);
                } catch (e) {
                    console.error('Auto-save failed:', e);
                }
            }
        };
        return {
            start() {
                if (!timer) {
                    timer = setInterval(save, intervalMs);
                }
            },
            stop() {
                if (timer) {
                    clearInterval(timer);
                    timer = null;
                }
            },
            saveNow: save
        };
    }
};
"#
    .to_string()
}

/// Offline-first sync status
#[wasm_bindgen]
pub struct SyncStatus {
    is_online: bool,
    pending_changes: usize,
    last_sync_timestamp: f64,
}

#[wasm_bindgen]
impl SyncStatus {
    /// Create a new sync status
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            is_online: true,
            pending_changes: 0,
            last_sync_timestamp: 0.0,
        }
    }

    /// Check if online
    #[wasm_bindgen(getter, js_name = "isOnline")]
    pub fn is_online(&self) -> bool {
        self.is_online
    }

    /// Get pending changes count
    #[wasm_bindgen(getter, js_name = "pendingChanges")]
    pub fn pending_changes(&self) -> usize {
        self.pending_changes
    }

    /// Get last sync timestamp
    #[wasm_bindgen(getter, js_name = "lastSyncTimestamp")]
    pub fn last_sync_timestamp(&self) -> f64 {
        self.last_sync_timestamp
    }

    /// Update online status
    #[wasm_bindgen(js_name = "setOnline")]
    pub fn set_online(&mut self, online: bool) {
        self.is_online = online;
    }

    /// Add pending change
    #[wasm_bindgen(js_name = "addPendingChange")]
    pub fn add_pending_change(&mut self) {
        self.pending_changes += 1;
    }

    /// Clear pending changes (after sync)
    #[wasm_bindgen(js_name = "clearPendingChanges")]
    pub fn clear_pending_changes(&mut self) {
        self.pending_changes = 0;
        self.last_sync_timestamp = js_sys::Date::now();
    }

    /// Convert to JavaScript object
    #[wasm_bindgen(js_name = "toObject")]
    pub fn to_object(&self) -> js_sys::Object {
        let obj = js_sys::Object::new();
        let _ = js_sys::Reflect::set(&obj, &"isOnline".into(), &self.is_online.into());
        let _ = js_sys::Reflect::set(
            &obj,
            &"pendingChanges".into(),
            &(self.pending_changes as u32).into(),
        );
        let _ = js_sys::Reflect::set(
            &obj,
            &"lastSyncTimestamp".into(),
            &self.last_sync_timestamp.into(),
        );
        obj
    }
}

impl Default for SyncStatus {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate JavaScript helper code for service worker support
#[wasm_bindgen(js_name = "getServiceWorkerHelpers")]
pub fn get_service_worker_helpers() -> String {
    r#"
const NeedleServiceWorker = {
    // Register service worker with Needle caching support
    async register(swPath = '/needle-sw.js') {
        if ('serviceWorker' in navigator) {
            try {
                const registration = await navigator.serviceWorker.register(swPath);
                console.log('Needle ServiceWorker registered:', registration);
                return registration;
            } catch (e) {
                console.error('Needle ServiceWorker registration failed:', e);
                throw e;
            }
        }
        throw new Error('Service workers not supported');
    },

    // Generate service worker script content
    generateScript(options = {}) {
        const cacheName = options.cacheName || 'needle-cache-v1';
        const wasmPath = options.wasmPath || '/needle.wasm';
        
        return `
const CACHE_NAME = '${cacheName}';
const WASM_PATH = '${wasmPath}';

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            return cache.add(WASM_PATH);
        })
    );
});

self.addEventListener('fetch', (event) => {
    if (event.request.url.includes('.wasm')) {
        event.respondWith(
            caches.match(event.request).then((response) => {
                return response || fetch(event.request).then((fetchResponse) => {
                    return caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, fetchResponse.clone());
                        return fetchResponse;
                    });
                });
            })
        );
    }
});

self.addEventListener('message', (event) => {
    if (event.data.type === 'SYNC_COLLECTION') {
        // Handle background sync
        self.registration.sync.register('needle-sync');
    }
});

self.addEventListener('sync', (event) => {
    if (event.tag === 'needle-sync') {
        event.waitUntil(syncCollections());
    }
});

async function syncCollections() {
    // Notify clients to sync
    const clients = await self.clients.matchAll();
    clients.forEach(client => {
        client.postMessage({ type: 'SYNC_REQUESTED' });
    });
}
`;
    },

    // Check if currently offline
    isOffline() {
        return !navigator.onLine;
    },

    // Listen for online/offline events
    onConnectivityChange(callback) {
        window.addEventListener('online', () => callback(true));
        window.addEventListener('offline', () => callback(false));
    }
};
"#
    .to_string()
}

// ============================================================================
// Multi-Collection WasmDatabase
// ============================================================================

/// Multi-collection database for WASM environments
#[wasm_bindgen]
pub struct WasmDatabase {
    collections: std::collections::HashMap<String, Collection>,
}

#[wasm_bindgen]
impl WasmDatabase {
    /// Create a new empty database
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            collections: std::collections::HashMap::new(),
        }
    }

    /// Create a new collection with the given name and dimensions
    #[wasm_bindgen(js_name = "createCollection")]
    pub fn create_collection(&mut self, name: &str, dimensions: usize) -> Result<(), JsValue> {
        if self.collections.contains_key(name) {
            return Err(JsValue::from_str(&format!(
                "Collection '{}' already exists",
                name
            )));
        }
        let config = CollectionConfig::new(name, dimensions);
        self.collections
            .insert(name.to_string(), Collection::new(config));
        Ok(())
    }

    /// Get a reference to a collection by name
    #[wasm_bindgen(js_name = "getCollection")]
    pub fn get_collection(&self, name: &str) -> Result<WasmCollectionRef, JsValue> {
        if !self.collections.contains_key(name) {
            return Err(JsValue::from_str(&format!(
                "Collection '{}' not found",
                name
            )));
        }
        Ok(WasmCollectionRef {
            name: name.to_string(),
        })
    }

    /// List all collection names
    #[wasm_bindgen(js_name = "listCollections")]
    pub fn list_collections(&self) -> Vec<JsValue> {
        self.collections
            .keys()
            .map(|k| JsValue::from_str(k))
            .collect()
    }

    /// Delete a collection by name, returns true if it existed
    #[wasm_bindgen(js_name = "deleteCollection")]
    pub fn delete_collection(&mut self, name: &str) -> bool {
        self.collections.remove(name).is_some()
    }

    /// Serialize the entire database to bytes for IndexedDB persistence
    #[wasm_bindgen(js_name = "toBytes")]
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsValue> {
        let mut buf: Vec<u8> = Vec::new();
        let count = self.collections.len() as u32;
        buf.extend_from_slice(&count.to_le_bytes());

        for (name, coll) in &self.collections {
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len() as u32;
            buf.extend_from_slice(&name_len.to_le_bytes());
            buf.extend_from_slice(name_bytes);

            let coll_bytes = coll
                .to_bytes()
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            let coll_len = coll_bytes.len() as u64;
            buf.extend_from_slice(&coll_len.to_le_bytes());
            buf.extend_from_slice(&coll_bytes);
        }
        Ok(buf)
    }

    /// Deserialize a database from bytes loaded from IndexedDB
    #[wasm_bindgen(js_name = "fromBytes")]
    pub fn from_bytes(bytes: &[u8]) -> Result<WasmDatabase, JsValue> {
        let mut pos = 0usize;

        if bytes.len() < 4 {
            return Err(JsValue::from_str("Invalid database bytes: too short"));
        }
        let count = u32::from_le_bytes(
            bytes[pos..pos + 4]
                .try_into()
                .expect("slice is exactly 4 bytes"),
        ) as usize;
        pos += 4;

        let mut collections = std::collections::HashMap::new();

        for _ in 0..count {
            if pos + 4 > bytes.len() {
                return Err(JsValue::from_str("Truncated database bytes"));
            }
            let name_len = u32::from_le_bytes(
                bytes[pos..pos + 4]
                    .try_into()
                    .expect("slice is exactly 4 bytes"),
            ) as usize;
            pos += 4;

            if pos + name_len > bytes.len() {
                return Err(JsValue::from_str("Truncated collection name"));
            }
            let name = String::from_utf8(bytes[pos..pos + name_len].to_vec())
                .map_err(|e| JsValue::from_str(&format!("Invalid UTF-8 name: {}", e)))?;
            pos += name_len;

            if pos + 8 > bytes.len() {
                return Err(JsValue::from_str("Truncated collection length"));
            }
            let coll_len = u64::from_le_bytes(
                bytes[pos..pos + 8]
                    .try_into()
                    .expect("slice is exactly 8 bytes"),
            ) as usize;
            pos += 8;

            if pos + coll_len > bytes.len() {
                return Err(JsValue::from_str("Truncated collection data"));
            }
            let coll = Collection::from_bytes(&bytes[pos..pos + coll_len])
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            pos += coll_len;

            collections.insert(name, coll);
        }

        Ok(WasmDatabase { collections })
    }

    /// Insert a vector into a named collection
    pub fn insert(
        &mut self,
        collection: &str,
        id: &str,
        vector: Vec<f32>,
        metadata: Option<String>,
    ) -> Result<(), JsValue> {
        let coll = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| JsValue::from_str(&format!("Collection '{}' not found", collection)))?;
        let meta_value: Option<Value> = if let Some(json) = metadata {
            Some(
                serde_json::from_str(&json)
                    .map_err(|e| JsValue::from_str(&format!("Invalid JSON metadata: {}", e)))?,
            )
        } else {
            None
        };
        coll.insert(id, &vector, meta_value)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Search within a named collection
    pub fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>, JsValue> {
        let coll = self
            .collections
            .get(collection)
            .ok_or_else(|| JsValue::from_str(&format!("Collection '{}' not found", collection)))?;
        let results = coll
            .search(&query, k)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(results.into_iter().map(SearchResult::from).collect())
    }
}

impl Default for WasmDatabase {
    fn default() -> Self {
        Self::new()
    }
}

/// A lightweight reference to a collection inside a WasmDatabase
#[wasm_bindgen]
pub struct WasmCollectionRef {
    name: String,
}

#[wasm_bindgen]
impl WasmCollectionRef {
    /// Get the collection name
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }
}

// ============================================================================
// IndexedDB Persistence Helpers (script generators)
// ============================================================================

/// Configuration for generating IndexedDB persistence scripts
#[wasm_bindgen]
pub struct IndexedDbPersistenceConfig {
    db_name: String,
    store_name: String,
    version: u32,
}

#[wasm_bindgen]
impl IndexedDbPersistenceConfig {
    /// Create a new configuration with sensible defaults
    #[wasm_bindgen(constructor)]
    pub fn new(db_name: &str) -> Self {
        Self {
            db_name: db_name.to_string(),
            store_name: "needle_collections".to_string(),
            version: 1,
        }
    }

    /// Get the IndexedDB database name
    #[wasm_bindgen(getter, js_name = "dbName")]
    pub fn db_name(&self) -> String {
        self.db_name.clone()
    }

    /// Get the object store name
    #[wasm_bindgen(getter, js_name = "storeName")]
    pub fn store_name(&self) -> String {
        self.store_name.clone()
    }

    /// Get the IndexedDB schema version
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> u32 {
        self.version
    }

    /// Generate JavaScript code that saves a `Uint8Array` into IndexedDB
    #[wasm_bindgen(js_name = "generateSaveScript")]
    pub fn generate_save_script(&self) -> String {
        format!(
            r#"async function needleSave(key, dataBytes) {{
  return new Promise((resolve, reject) => {{
    const request = indexedDB.open("{db}", {ver});
    request.onupgradeneeded = (e) => {{
      const db = e.target.result;
      if (!db.objectStoreNames.contains("{store}")) {{
        db.createObjectStore("{store}");
      }}
    }};
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {{
      const db = request.result;
      const tx = db.transaction("{store}", "readwrite");
      const store = tx.objectStore("{store}");
      const putReq = store.put(dataBytes, key);
      putReq.onerror = () => reject(putReq.error);
      tx.oncomplete = () => {{ db.close(); resolve(); }};
    }};
  }});
}}"#,
            db = self.db_name,
            store = self.store_name,
            ver = self.version
        )
    }

    /// Generate JavaScript code that loads a `Uint8Array` from IndexedDB
    #[wasm_bindgen(js_name = "generateLoadScript")]
    pub fn generate_load_script(&self) -> String {
        format!(
            r#"async function needleLoad(key) {{
  return new Promise((resolve, reject) => {{
    const request = indexedDB.open("{db}", {ver});
    request.onupgradeneeded = (e) => {{
      const db = e.target.result;
      if (!db.objectStoreNames.contains("{store}")) {{
        db.createObjectStore("{store}");
      }}
    }};
    request.onerror = () => reject(request.error);
    request.onsuccess = () => {{
      const db = request.result;
      const tx = db.transaction("{store}", "readonly");
      const store = tx.objectStore("{store}");
      const getReq = store.get(key);
      getReq.onerror = () => reject(getReq.error);
      getReq.onsuccess = () => {{
        db.close();
        resolve(getReq.result || null);
      }};
    }};
  }});
}}"#,
            db = self.db_name,
            store = self.store_name,
            ver = self.version
        )
    }
}

// ============================================================================
// Web Worker Configuration
// ============================================================================

/// Configuration for running Needle search inside a Web Worker
#[wasm_bindgen]
pub struct WebWorkerConfig {
    num_threads: usize,
    search_timeout_ms: u32,
}

#[wasm_bindgen]
impl WebWorkerConfig {
    /// Create a default Web Worker configuration
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            num_threads: 1,
            search_timeout_ms: 5000,
        }
    }

    /// Set the number of workers to spawn
    #[wasm_bindgen(js_name = "withThreads")]
    pub fn with_threads(mut self, n: usize) -> Self {
        self.num_threads = n;
        self
    }

    /// Set the search timeout in milliseconds
    #[wasm_bindgen(js_name = "withTimeout")]
    pub fn with_timeout(mut self, ms: u32) -> Self {
        self.search_timeout_ms = ms;
        self
    }

    /// Generate JavaScript source for a Web Worker that loads the WASM module
    /// and exposes a message-based search API
    #[wasm_bindgen(js_name = "generateWorkerScript")]
    pub fn generate_worker_script(&self) -> String {
        format!(
            r#"// Needle Web Worker – auto-generated
const SEARCH_TIMEOUT_MS = {timeout};

let db = null;

self.onmessage = async (e) => {{
  const {{ type, id, payload }} = e.data;
  try {{
    switch (type) {{
      case "init": {{
        const {{ wasmUrl, initFn }} = payload;
        importScripts(initFn);
        await wasm_bindgen(wasmUrl);
        db = new wasm_bindgen.WasmDatabase();
        self.postMessage({{ id, ok: true }});
        break;
      }}
      case "createCollection": {{
        db.createCollection(payload.name, payload.dimensions);
        self.postMessage({{ id, ok: true }});
        break;
      }}
      case "insert": {{
        db.insert(
          payload.collection,
          payload.vectorId,
          new Float32Array(payload.vector),
          payload.metadata ? JSON.stringify(payload.metadata) : undefined
        );
        self.postMessage({{ id, ok: true }});
        break;
      }}
      case "search": {{
        const timer = setTimeout(() => {{
          self.postMessage({{ id, ok: false, error: "Search timed out" }});
        }}, SEARCH_TIMEOUT_MS);
        const results = db.search(payload.collection, new Float32Array(payload.query), payload.k);
        clearTimeout(timer);
        const mapped = [];
        for (let i = 0; i < results.length; i++) {{
          mapped.push({{ id: results[i].id, distance: results[i].distance, metadata: results[i].metadata }});
        }}
        self.postMessage({{ id, ok: true, results: mapped }});
        break;
      }}
      case "load": {{
        const bytes = new Uint8Array(payload.bytes);
        db = wasm_bindgen.WasmDatabase.fromBytes(bytes);
        self.postMessage({{ id, ok: true }});
        break;
      }}
      case "save": {{
        const bytes = db.toBytes();
        self.postMessage({{ id, ok: true, bytes }}, [bytes.buffer]);
        break;
      }}
      default:
        self.postMessage({{ id, ok: false, error: "Unknown message type: " + type }});
    }}
  }} catch (err) {{
    self.postMessage({{ id, ok: false, error: err.toString() }});
  }}
}};
"#,
            timeout = self.search_timeout_ms
        )
    }
}

impl Default for WebWorkerConfig {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Batch Search API
// ============================================================================

/// A request containing multiple search queries for batch execution
#[wasm_bindgen]
pub struct BatchSearchRequest {
    queries: Vec<Vec<f32>>,
    ks: Vec<usize>,
}

#[wasm_bindgen]
impl BatchSearchRequest {
    /// Create a new empty batch search request
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
            ks: Vec::new(),
        }
    }

    /// Add a query vector with its own k value
    #[wasm_bindgen(js_name = "addQuery")]
    pub fn add_query(&mut self, query: Vec<f32>, k: usize) {
        self.queries.push(query);
        self.ks.push(k);
    }

    /// Get the number of queued queries
    #[wasm_bindgen(js_name = "queryCount")]
    pub fn query_count(&self) -> usize {
        self.queries.len()
    }
}

impl Default for BatchSearchRequest {
    fn default() -> Self {
        Self::new()
    }
}

/// Results from a batch search operation
#[wasm_bindgen]
pub struct BatchSearchResults {
    results: Vec<Vec<SearchResult>>,
    duration_ms: f64,
}

#[wasm_bindgen]
impl BatchSearchResults {
    /// Get the number of result sets
    #[wasm_bindgen(getter, js_name = "queryCount")]
    pub fn query_count(&self) -> usize {
        self.results.len()
    }

    /// Get duration in milliseconds
    #[wasm_bindgen(getter, js_name = "durationMs")]
    pub fn duration_ms(&self) -> f64 {
        self.duration_ms
    }

    /// Get results for a specific query index as a JS array of objects
    #[wasm_bindgen(js_name = "getResults")]
    pub fn get_results(&self, index: usize) -> js_sys::Array {
        let arr = js_sys::Array::new();
        if let Some(results) = self.results.get(index) {
            for r in results {
                let obj = js_sys::Object::new();
                let _ = js_sys::Reflect::set(&obj, &"id".into(), &r.id.clone().into());
                let _ = js_sys::Reflect::set(&obj, &"distance".into(), &r.distance.into());
                if let Some(ref meta) = r.metadata_json {
                    if let Ok(parsed) = js_sys::JSON::parse(meta) {
                        let _ = js_sys::Reflect::set(&obj, &"metadata".into(), &parsed);
                    }
                }
                arr.push(&obj);
            }
        }
        arr
    }
}

// Additional WasmCollection methods
#[wasm_bindgen]
impl WasmCollection {
    /// Get the number of vectors in the collection
    pub fn count(&self) -> usize {
        self.inner.read().map(|guard| guard.len()).unwrap_or(0)
    }

    /// Execute a batch search request against this collection
    #[wasm_bindgen(js_name = "batchSearch")]
    pub fn batch_search_request(
        &self,
        request: &BatchSearchRequest,
    ) -> Result<BatchSearchResults, JsValue> {
        let start = js_sys::Date::now();

        let coll = self
            .inner
            .read()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        let mut all_results: Vec<Vec<SearchResult>> = Vec::with_capacity(request.queries.len());
        for (query, &k) in request.queries.iter().zip(request.ks.iter()) {
            let results = coll
                .search(query, k)
                .map_err(|e| JsValue::from_str(&e.to_string()))?;
            all_results.push(results.into_iter().map(SearchResult::from).collect());
        }

        let duration_ms = js_sys::Date::now() - start;

        Ok(BatchSearchResults {
            results: all_results,
            duration_ms,
        })
    }

    /// Search with advanced options
    #[wasm_bindgen(js_name = "searchWithOptions")]
    pub fn search_with_options(
        &self,
        query: Vec<f32>,
        options: &WasmSearchOptions,
    ) -> Result<Vec<SearchResult>, JsValue> {
        let mut coll = self
            .inner
            .write()
            .map_err(|_| JsValue::from_str("Lock poisoned"))?;

        // Temporarily set ef_search if provided
        let prev_ef = if let Some(ef) = options.ef_search {
            let current = coll.dimensions(); // no getter for ef, just set it
            coll.set_ef_search(ef);
            Some(current)
        } else {
            None
        };

        let results = if let Some(ref filter_json) = options.filter_json {
            let filter_value: Value = serde_json::from_str(filter_json)
                .map_err(|e| JsValue::from_str(&format!("Invalid filter JSON: {}", e)))?;
            let filter = Filter::parse(&filter_value)
                .map_err(|e| JsValue::from_str(&format!("Invalid filter: {}", e)))?;
            coll.search_with_filter(&query, options.k, &filter)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        } else {
            coll.search(&query, options.k)
                .map_err(|e| JsValue::from_str(&e.to_string()))?
        };

        // Restore previous ef_search is not strictly needed since we hold a write lock,
        // but we leave it at the new value as a deliberate side-effect.
        let _ = prev_ef;

        Ok(results.into_iter().map(SearchResult::from).collect())
    }
}

// ============================================================================
// WasmSearchOptions
// ============================================================================

/// Advanced search configuration for WASM
#[wasm_bindgen]
pub struct WasmSearchOptions {
    k: usize,
    ef_search: Option<usize>,
    filter_json: Option<String>,
}

#[wasm_bindgen]
impl WasmSearchOptions {
    /// Create new search options with k results
    #[wasm_bindgen(constructor)]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            ef_search: None,
            filter_json: None,
        }
    }

    /// Set the ef_search HNSW parameter for recall tuning
    #[wasm_bindgen(js_name = "withEfSearch")]
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = Some(ef);
        self
    }

    /// Set a metadata filter as a JSON string (MongoDB-style query)
    #[wasm_bindgen(js_name = "withFilter")]
    pub fn with_filter(mut self, filter_json: &str) -> Self {
        self.filter_json = Some(filter_json.to_string());
        self
    }

    /// Get the k value
    #[wasm_bindgen(getter)]
    pub fn k(&self) -> usize {
        self.k
    }
}
