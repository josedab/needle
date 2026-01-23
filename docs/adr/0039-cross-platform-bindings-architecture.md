# ADR-0039: Cross-Platform Bindings Architecture

## Status

Accepted

## Context

Needle is written in Rust, but users work in many languages:

1. **Python** — Data scientists, ML engineers, LangChain users
2. **JavaScript/TypeScript** — Web developers, Node.js backends
3. **Swift** — iOS mobile applications
4. **Kotlin** — Android mobile applications

Each platform has different FFI (Foreign Function Interface) patterns:

| Platform | FFI Approach | Key Requirements |
|----------|--------------|------------------|
| Python | PyO3 | NumPy arrays, async/await, GIL handling |
| Browser | wasm-bindgen | Small bundle, no threads, Promise-based |
| Node.js | wasm-bindgen + napi | Async workers, Buffer types |
| iOS | UniFFI | Swift types, Combine/async |
| Android | UniFFI | Kotlin types, coroutines |

## Decision

Implement **platform-specific bindings** via feature flags, using best-in-class tooling for each platform:

- **PyO3** for Python (with pythonize for serde)
- **wasm-bindgen** for Browser/Node.js
- **UniFFI** for Swift/Kotlin mobile

### Feature Flag Architecture

```toml
[features]
# Language bindings (mutually exclusive build targets)
python = ["pyo3", "pythonize", "numpy"]
wasm = ["wasm-bindgen", "js-sys", "web-sys", "getrandom/js"]
uniffi-bindings = ["uniffi"]
```

### Python Bindings (PyO3)

```rust
#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use pythonize::{depythonize, pythonize};
    use numpy::{PyArray1, PyReadonlyArray1};

    /// Python-facing Database class
    #[pyclass(name = "Database")]
    pub struct PyDatabase {
        inner: Database,
    }

    #[pymethods]
    impl PyDatabase {
        #[new]
        fn new(path: Option<&str>) -> PyResult<Self> {
            let inner = match path {
                Some(p) => Database::open(p).map_err(to_py_err)?,
                None => Database::in_memory(),
            };
            Ok(Self { inner })
        }

        /// Create a collection
        fn create_collection(
            &mut self,
            name: &str,
            dimension: usize,
            distance: Option<&str>,
        ) -> PyResult<()> {
            // ...
        }

        /// Insert vectors (accepts numpy arrays)
        fn insert(
            &mut self,
            collection: &str,
            id: &str,
            vector: PyReadonlyArray1<f32>,
            metadata: Option<&PyAny>,
        ) -> PyResult<()> {
            let vec = vector.as_slice()?.to_vec();
            let meta = metadata
                .map(|m| depythonize(m))
                .transpose()?
                .unwrap_or(Value::Null);

            self.inner.insert(collection, id, &vec, meta)
                .map_err(to_py_err)
        }

        /// Search (returns numpy array of results)
        fn search(
            &self,
            py: Python<'_>,
            collection: &str,
            query: PyReadonlyArray1<f32>,
            k: usize,
            filter: Option<&PyAny>,
        ) -> PyResult<Vec<PyObject>> {
            let query_vec = query.as_slice()?;
            let filter = filter.map(|f| Filter::parse(&depythonize(f)?)).transpose()?;

            let results = self.inner.search(collection, query_vec, k, filter.as_ref())
                .map_err(to_py_err)?;

            // Convert to Python dicts
            results.iter()
                .map(|r| pythonize(py, r).map_err(|e| e.into()))
                .collect()
        }
    }

    #[pymodule]
    fn needle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
        m.add_class::<PyDatabase>()?;
        Ok(())
    }
}
```

### WebAssembly Bindings (wasm-bindgen)

```rust
#[cfg(feature = "wasm")]
mod wasm {
    use wasm_bindgen::prelude::*;
    use js_sys::{Array, Object, Float32Array};

    #[wasm_bindgen]
    pub struct WasmDatabase {
        inner: Database,
    }

    #[wasm_bindgen]
    impl WasmDatabase {
        #[wasm_bindgen(constructor)]
        pub fn new() -> Result<WasmDatabase, JsValue> {
            // WASM: always in-memory (no file system)
            Ok(Self {
                inner: Database::in_memory(),
            })
        }

        #[wasm_bindgen(js_name = createCollection)]
        pub fn create_collection(
            &mut self,
            name: &str,
            dimension: usize,
        ) -> Result<(), JsValue> {
            self.inner.create_collection(name, dimension, Default::default())
                .map_err(to_js_err)
        }

        /// Insert vector (accepts Float32Array)
        pub fn insert(
            &mut self,
            collection: &str,
            id: &str,
            vector: Float32Array,
            metadata: JsValue,
        ) -> Result<(), JsValue> {
            let vec: Vec<f32> = vector.to_vec();
            let meta: Value = serde_wasm_bindgen::from_value(metadata)?;

            self.inner.insert(collection, id, &vec, meta)
                .map_err(to_js_err)
        }

        /// Search (returns Promise for async-friendly API)
        #[wasm_bindgen(js_name = search)]
        pub fn search(
            &self,
            collection: &str,
            query: Float32Array,
            k: usize,
        ) -> Result<Array, JsValue> {
            let query_vec: Vec<f32> = query.to_vec();

            let results = self.inner.search(collection, &query_vec, k, None)
                .map_err(to_js_err)?;

            // Convert to JS array of objects
            let array = Array::new();
            for r in results {
                let obj = Object::new();
                js_sys::Reflect::set(&obj, &"id".into(), &r.id.into())?;
                js_sys::Reflect::set(&obj, &"distance".into(), &r.distance.into())?;
                array.push(&obj);
            }
            Ok(array)
        }
    }
}
```

### UniFFI Bindings (Swift/Kotlin)

```rust
#[cfg(feature = "uniffi-bindings")]
mod uniffi_bindings {
    use uniffi;

    #[derive(uniffi::Object)]
    pub struct UniffiDatabase {
        inner: Database,
    }

    #[uniffi::export]
    impl UniffiDatabase {
        #[uniffi::constructor]
        pub fn new() -> Self {
            Self {
                inner: Database::in_memory(),
            }
        }

        #[uniffi::constructor]
        pub fn open(path: String) -> Result<Self, NeedleError> {
            Ok(Self {
                inner: Database::open(&path)?,
            })
        }

        pub fn create_collection(
            &self,
            name: String,
            dimension: u32,
        ) -> Result<(), NeedleError> {
            self.inner.create_collection(&name, dimension as usize, Default::default())
        }

        pub fn search(
            &self,
            collection: String,
            query: Vec<f32>,
            k: u32,
        ) -> Result<Vec<SearchResultUniffi>, NeedleError> {
            let results = self.inner.search(&collection, &query, k as usize, None)?;
            Ok(results.into_iter().map(SearchResultUniffi::from).collect())
        }
    }

    #[derive(uniffi::Record)]
    pub struct SearchResultUniffi {
        pub id: String,
        pub distance: f32,
        pub metadata: String,  // JSON string for cross-language simplicity
    }
}
```

### Code References

- `src/python.rs` — PyO3 Python bindings
- `src/wasm.rs` — wasm-bindgen JavaScript bindings
- `src/uniffi_bindings.rs` — UniFFI Swift/Kotlin bindings
- `Cargo.toml:172-178` — Binding feature flags

## Consequences

### Benefits

1. **Native performance** — All bindings call into same optimized Rust core
2. **Idiomatic APIs** — Each language gets natural types (numpy, Float32Array, etc.)
3. **Selective compilation** — Only compile bindings you need
4. **Single source of truth** — Core logic in Rust, bindings are thin wrappers
5. **Type safety** — Compile-time checking where possible

### Tradeoffs

1. **Maintenance overhead** — Three binding implementations to maintain
2. **API surface duplication** — Same methods defined multiple times
3. **Testing complexity** — Must test each binding separately
4. **Release coordination** — Python wheel, npm package, CocoaPod/Maven all versioned

### Platform-Specific Considerations

| Platform | Consideration | Solution |
|----------|---------------|----------|
| Python | GIL blocks parallel search | Release GIL during search |
| WASM | No threads | Main thread only, keep operations fast |
| WASM | No file system | In-memory only, IndexedDB for persistence |
| iOS | No dynamic linking | Static framework |
| Android | JNI overhead | Batch operations to reduce crossings |

### What This Enabled

- `pip install needle-db` for Python users
- `npm install needle-db` for JavaScript users
- Swift Package Manager / CocoaPods for iOS
- Maven/Gradle for Android
- Same Rust core for all platforms

### What This Prevented

- Rewriting Needle in each language
- Inconsistent behavior across platforms
- Performance loss from pure-language implementations

### Build Commands

```bash
# Python wheel (via maturin)
maturin build --release --features python

# WASM package (via wasm-pack)
wasm-pack build --release --features wasm

# UniFFI bindings generation
cargo build --release --features uniffi-bindings
uniffi-bindgen generate src/needle.udl --language swift
uniffi-bindgen generate src/needle.udl --language kotlin
```

### Usage Examples

**Python:**
```python
import needle
import numpy as np

db = needle.Database()
db.create_collection("docs", 384)

vector = np.random.randn(384).astype(np.float32)
db.insert("docs", "id1", vector, {"text": "hello"})

results = db.search("docs", vector, k=5)
```

**JavaScript (Browser):**
```javascript
import { WasmDatabase } from 'needle-db';

const db = new WasmDatabase();
db.createCollection('docs', 384);

const vector = new Float32Array(384).fill(0.1);
db.insert('docs', 'id1', vector, { text: 'hello' });

const results = db.search('docs', vector, 5);
```

**Swift:**
```swift
import Needle

let db = try UniffiDatabase.open(path: "mydb.needle")
try db.createCollection(name: "docs", dimension: 384)

let vector: [Float] = Array(repeating: 0.1, count: 384)
let results = try db.search(collection: "docs", query: vector, k: 5)
```

**Kotlin:**
```kotlin
import needle.UniffiDatabase

val db = UniffiDatabase.open("mydb.needle")
db.createCollection("docs", 384u)

val vector = FloatArray(384) { 0.1f }
val results = db.search("docs", vector.toList(), 5u)
```
