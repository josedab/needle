//! WebAssembly bindings for Needle using wasm-bindgen

use crate::collection::{Collection, CollectionConfig, SearchResult as RustSearchResult};
use crate::distance::DistanceFunction;
use crate::metadata::parse_filter;
use serde_json::Value;
use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

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
        self.inner.read()
            .map(|guard| guard.name().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    }

    /// Get the vector dimensions
    #[wasm_bindgen(getter)]
    pub fn dimensions(&self) -> usize {
        self.inner.read()
            .map(|guard| guard.dimensions())
            .unwrap_or(0)
    }

    /// Get the number of vectors
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.read()
            .map(|guard| guard.len())
            .unwrap_or(0)
    }

    /// Check if the collection is empty
    #[wasm_bindgen(js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.inner.read()
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
    #[wasm_bindgen(js_name = "insertBatch")]
    pub fn insert_batch(
        &self,
        ids: Vec<String>,
        vectors: Vec<Vec<f32>>,
        metadata_json_array: Option<Vec<String>>,
    ) -> Result<(), JsValue> {
        if ids.len() != vectors.len() {
            return Err(JsValue::from_str("ids and vectors must have the same length"));
        }

        let meta_values: Vec<Option<Value>> = if let Some(meta_list) = metadata_json_array {
            if meta_list.len() != ids.len() {
                return Err(JsValue::from_str("metadata must have the same length as ids"));
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

        let filter = parse_filter(&filter_value)
            .ok_or_else(|| JsValue::from_str("Invalid filter format"))?;

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
        self.inner.write()
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
