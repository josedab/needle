---
sidebar_position: 4
---

# Swift & Kotlin

Needle provides native bindings for Swift (iOS, macOS) and Kotlin (Android, JVM) through UniFFI.

## Swift (iOS/macOS)

### Installation

#### Swift Package Manager

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/anthropics/needle.git", from: "0.1.0")
]
```

#### CocoaPods

```ruby
# Podfile
pod 'NeedleDB', '~> 0.1'
```

### Quick Start

```swift
import NeedleDB

// Create database
let db = try Database.open(path: "vectors.needle")

// Create collection
try db.createCollection(
    name: "documents",
    dimensions: 384,
    distance: .cosine
)

// Get collection
let collection = try db.collection(name: "documents")

// Insert vector
let embedding: [Float] = Array(repeating: 0.1, count: 384)
try collection.insert(
    id: "doc1",
    vector: embedding,
    metadata: ["title": "Hello World"]
)

// Search
let query: [Float] = Array(repeating: 0.1, count: 384)
let results = try collection.search(vector: query, k: 10)

for result in results {
    print("ID: \(result.id), Distance: \(result.distance)")
}

// Save
try db.save()
```

### Database API

```swift
import NeedleDB

// File-based database
let db = try Database.open(path: documentsPath + "/vectors.needle")

// In-memory database
let db = try Database.inMemory()

// Collection management
try db.createCollection(name: "docs", dimensions: 384, distance: .cosine)

let config = CollectionConfig(
    dimensions: 384,
    distance: .cosine,
    hnswM: 32,
    hnswEfConstruction: 400
)
try db.createCollection(name: "high_quality", config: config)

// List collections
let names = try db.listCollections()

// Get collection
let collection = try db.collection(name: "documents")

// Delete collection
try db.deleteCollection(name: "old")
```

### Collection API

```swift
let collection = try db.collection(name: "documents")

// Insert
let embedding: [Float] = generateEmbedding(text: "Hello world")
try collection.insert(
    id: "doc1",
    vector: embedding,
    metadata: [
        "title": "Hello World",
        "category": "greeting",
        "timestamp": Date().timeIntervalSince1970
    ]
)

// Get by ID
if let entry = try collection.get(id: "doc1") {
    print("Vector: \(entry.vector)")
    print("Metadata: \(entry.metadata)")
}

// Delete
try collection.delete(id: "doc1")

// Count
let count = try collection.count()

// Clear
try collection.clear()
```

### Searching

```swift
// Basic search
let results = try collection.search(vector: queryVector, k: 10)

// With filter
let filter: [String: Any] = ["category": "programming"]
let results = try collection.search(vector: queryVector, k: 10, filter: filter)

// Complex filter
let filter: [String: Any] = [
    "$and": [
        ["category": ["$in": ["books", "articles"]]],
        ["year": ["$gte": 2020]]
    ]
]
let results = try collection.search(vector: queryVector, k: 10, filter: filter)

// With custom ef_search
let results = try collection.search(vector: queryVector, k: 10, filter: nil, efSearch: 100)
```

### Core ML Integration

```swift
import CoreML
import NeedleDB

class SemanticSearch {
    let db: Database
    let model: TextEmbedding  // Your Core ML model

    init() throws {
        db = try Database.open(path: getDocumentsPath() + "/search.needle")
        model = try TextEmbedding()

        if !db.collectionExists(name: "documents") {
            try db.createCollection(name: "documents", dimensions: 384, distance: .cosine)
        }
    }

    func index(id: String, text: String, metadata: [String: Any]) throws {
        let embedding = try model.embed(text: text)
        let collection = try db.collection(name: "documents")
        try collection.insert(id: id, vector: embedding, metadata: metadata)
    }

    func search(query: String, k: Int = 10) throws -> [SearchResult] {
        let embedding = try model.embed(text: query)
        let collection = try db.collection(name: "documents")
        return try collection.search(vector: embedding, k: k)
    }
}
```

### SwiftUI Example

```swift
import SwiftUI
import NeedleDB

struct SearchView: View {
    @StateObject private var viewModel = SearchViewModel()
    @State private var query = ""

    var body: some View {
        NavigationView {
            VStack {
                TextField("Search...", text: $query)
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .padding()
                    .onChange(of: query) { newValue in
                        viewModel.search(query: newValue)
                    }

                List(viewModel.results) { result in
                    VStack(alignment: .leading) {
                        Text(result.title)
                            .font(.headline)
                        Text("Score: \(String(format: "%.2f", 1 - result.distance))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Semantic Search")
        }
    }
}

class SearchViewModel: ObservableObject {
    @Published var results: [SearchResultItem] = []

    private let searchEngine: SemanticSearch

    init() {
        searchEngine = try! SemanticSearch()
    }

    func search(query: String) {
        guard !query.isEmpty else {
            results = []
            return
        }

        Task {
            let searchResults = try await searchEngine.search(query: query)
            await MainActor.run {
                self.results = searchResults.map { result in
                    SearchResultItem(
                        id: result.id,
                        title: result.metadata["title"] as? String ?? "",
                        distance: result.distance
                    )
                }
            }
        }
    }
}
```

## Kotlin (Android/JVM)

### Installation

#### Gradle

```kotlin
// build.gradle.kts
dependencies {
    implementation("com.anthropic:needle:0.1.0")
}
```

#### Maven

```xml
<dependency>
    <groupId>com.anthropic</groupId>
    <artifactId>needle</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Quick Start

```kotlin
import com.anthropic.needle.*

fun main() {
    // Create database
    val db = Database.open("vectors.needle")

    // Create collection
    db.createCollection(
        name = "documents",
        dimensions = 384,
        distance = DistanceFunction.COSINE
    )

    // Get collection
    val collection = db.collection("documents")

    // Insert vector
    val embedding = FloatArray(384) { 0.1f }
    collection.insert(
        id = "doc1",
        vector = embedding,
        metadata = mapOf("title" to "Hello World")
    )

    // Search
    val query = FloatArray(384) { 0.1f }
    val results = collection.search(query, k = 10)

    for (result in results) {
        println("ID: ${result.id}, Distance: ${result.distance}")
    }

    // Save
    db.save()
}
```

### Database API

```kotlin
import com.anthropic.needle.*

// File-based database
val db = Database.open("vectors.needle")

// In-memory database
val db = Database.inMemory()

// Collection management
db.createCollection("docs", dimensions = 384, distance = DistanceFunction.COSINE)

val config = CollectionConfig(
    dimensions = 384,
    distance = DistanceFunction.COSINE,
    hnswM = 32,
    hnswEfConstruction = 400
)
db.createCollection("high_quality", config)

// List collections
val names = db.listCollections()

// Get collection
val collection = db.collection("documents")

// Delete collection
db.deleteCollection("old")
```

### Collection API

```kotlin
val collection = db.collection("documents")

// Insert
val embedding = generateEmbedding("Hello world")
collection.insert(
    id = "doc1",
    vector = embedding,
    metadata = mapOf(
        "title" to "Hello World",
        "category" to "greeting",
        "timestamp" to System.currentTimeMillis()
    )
)

// Get by ID
val entry = collection.get("doc1")
entry?.let {
    println("Vector: ${it.vector.toList()}")
    println("Metadata: ${it.metadata}")
}

// Delete
collection.delete("doc1")

// Count
val count = collection.count()

// Clear
collection.clear()
```

### Searching

```kotlin
// Basic search
val results = collection.search(queryVector, k = 10)

// With filter
val filter = mapOf("category" to "programming")
val results = collection.search(queryVector, k = 10, filter = filter)

// Complex filter
val filter = mapOf(
    "\$and" to listOf(
        mapOf("category" to mapOf("\$in" to listOf("books", "articles"))),
        mapOf("year" to mapOf("\$gte" to 2020))
    )
)
val results = collection.search(queryVector, k = 10, filter = filter)

// With custom ef_search
val results = collection.search(queryVector, k = 10, filter = null, efSearch = 100)
```

### Android Integration

```kotlin
import android.app.Application
import com.anthropic.needle.*

class NeedleApplication : Application() {
    lateinit var database: Database
        private set

    override fun onCreate() {
        super.onCreate()

        // Initialize database in app's files directory
        val dbPath = filesDir.resolve("vectors.needle").absolutePath
        database = Database.open(dbPath)

        // Create collection if needed
        if (!database.collectionExists("documents")) {
            database.createCollection("documents", 384, DistanceFunction.COSINE)
        }
    }

    override fun onTerminate() {
        database.save()
        super.onTerminate()
    }
}
```

### Jetpack Compose Example

```kotlin
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.anthropic.needle.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class SearchViewModel(private val db: Database) : ViewModel() {
    private val _results = MutableStateFlow<List<SearchResult>>(emptyList())
    val results: StateFlow<List<SearchResult>> = _results

    fun search(query: String) {
        if (query.isBlank()) {
            _results.value = emptyList()
            return
        }

        viewModelScope.launch {
            val embedding = generateEmbedding(query)
            val collection = db.collection("documents")
            _results.value = collection.search(embedding, k = 10)
        }
    }
}

@Composable
fun SearchScreen(viewModel: SearchViewModel) {
    var query by remember { mutableStateOf("") }
    val results by viewModel.results.collectAsState()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        OutlinedTextField(
            value = query,
            onValueChange = {
                query = it
                viewModel.search(it)
            },
            label = { Text("Search") },
            modifier = Modifier.fillMaxWidth()
        )

        Spacer(modifier = Modifier.height(16.dp))

        LazyColumn {
            items(results) { result ->
                Card(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp)
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            text = result.metadata["title"] as? String ?: result.id,
                            style = MaterialTheme.typography.titleMedium
                        )
                        Text(
                            text = "Score: ${"%.2f".format(1 - result.distance)}",
                            style = MaterialTheme.typography.bodySmall
                        )
                    }
                }
            }
        }
    }
}
```

### Coroutines Support

```kotlin
import kotlinx.coroutines.*

class AsyncSearchRepository(private val db: Database) {
    private val dispatcher = Dispatchers.IO

    suspend fun insert(id: String, vector: FloatArray, metadata: Map<String, Any>) =
        withContext(dispatcher) {
            val collection = db.collection("documents")
            collection.insert(id, vector, metadata)
        }

    suspend fun search(query: FloatArray, k: Int): List<SearchResult> =
        withContext(dispatcher) {
            val collection = db.collection("documents")
            collection.search(query, k)
        }

    suspend fun batchInsert(items: List<Triple<String, FloatArray, Map<String, Any>>>) =
        withContext(dispatcher) {
            val collection = db.collection("documents")
            items.forEach { (id, vector, metadata) ->
                collection.insert(id, vector, metadata)
            }
            db.save()
        }
}
```

## Platform-Specific Notes

### iOS

- Database files are stored in the app's Documents directory
- Use background tasks for large indexing operations
- Consider using Core ML for on-device embeddings

### Android

- Database files are stored in the app's internal storage
- Use WorkManager for background indexing
- Consider using TensorFlow Lite for on-device embeddings

### Memory Management

```swift
// Swift: Use autoreleasepool for batch operations
autoreleasepool {
    for doc in documents {
        try collection.insert(id: doc.id, vector: doc.embedding, metadata: doc.metadata)
    }
}
```

```kotlin
// Kotlin: Save periodically during large batch operations
documents.chunked(1000).forEach { batch ->
    batch.forEach { doc ->
        collection.insert(doc.id, doc.embedding, doc.metadata)
    }
    db.save()
}
```

## Next Steps

- [API Reference](/docs/api-reference)
- [Semantic Search Guide](/docs/guides/semantic-search)
- [Production Deployment](/docs/guides/production)
