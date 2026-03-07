# Needle Java SDK

Java client for the [Needle](https://github.com/anthropics/needle) vector database REST API.

**Java 11+** — uses `java.net.http.HttpClient` with zero external dependencies.

## Installation

Add to your Maven `pom.xml`:

```xml
<dependency>
    <groupId>dev.needle</groupId>
    <artifactId>needle-sdk</artifactId>
    <version>0.1.0</version>
</dependency>
```

Or build from source:

```bash
cd sdk/java
mvn package
```

## Quick Start

```java
import dev.needle.*;

// Create client
NeedleClient client = new NeedleClient("http://localhost:8080");

// With API key authentication
NeedleClient client = new NeedleClient("http://localhost:8080", "your-api-key");

// Health check
boolean healthy = client.isHealthy();

// Create a collection
Collection col = client.createCollection("documents", 384);

// Create with specific distance function
Collection col = client.createCollection("documents", 384, "cosine");

// List collections
List<Collection> collections = client.listCollections();

// Get collection info
Collection info = client.getCollection("documents");
System.out.println("Vectors: " + info.getCount());

// Delete collection
client.deleteCollection("documents");
```

## Inserting Vectors

```java
// Simple insert
float[] embedding = new float[]{0.1f, 0.2f, 0.3f, /* ... */};
Vector vec = new Vector("doc1", embedding);
client.insert("documents", vec);

// Insert with metadata
Map<String, Object> metadata = new HashMap<>();
metadata.put("title", "Introduction to Vectors");
metadata.put("category", "tutorial");
metadata.put("page", 42);

Vector vec = new Vector("doc1", embedding, metadata);
client.insert("documents", vec);

// Insert with TTL (time-to-live in seconds)
client.insert("documents", vec, 3600); // expires in 1 hour
```

## Retrieving and Deleting Vectors

```java
// Get vector by ID
Vector vec = client.getVector("documents", "doc1");
System.out.println("ID: " + vec.getId());
System.out.println("Dimensions: " + vec.getValues().length);
System.out.println("Metadata: " + vec.getMetadata());

// Delete vector by ID
client.deleteVector("documents", "doc1");
```

## Searching

```java
// Basic search
float[] queryVec = new float[]{0.15f, 0.25f, 0.35f, /* ... */};
SearchResponse resp = client.search("documents", new SearchOptions(queryVec, 10));

for (SearchResult result : resp.getResults()) {
    System.out.printf("ID: %s, Distance: %.4f%n", result.getId(), result.getDistance());
}

// Search with metadata filter
Map<String, Object> filter = new HashMap<>();
filter.put("category", "tutorial");
SearchOptions opts = new SearchOptions(queryVec, 5).setFilter(filter);
SearchResponse resp = client.search("documents", opts);

// Search with pagination (cursor-based)
SearchOptions opts = new SearchOptions(queryVec, 10);
SearchResponse resp = client.search("documents", opts);

while (resp.isHasMore()) {
    opts.setSearchAfter(resp.getNextCursor());
    resp = client.search("documents", opts);
    // process resp.getResults()
}

// Search with all options
SearchOptions opts = new SearchOptions(queryVec)
    .setK(20)
    .setFilter(filter)
    .setIncludeVectors(true)
    .setExplain(true)
    .setDistance("euclidean");
SearchResponse resp = client.search("documents", opts);
```

## Error Handling

```java
try {
    Collection col = client.getCollection("nonexistent");
} catch (NeedleException e) {
    System.out.println("Status: " + e.getStatusCode()); // 404
    System.out.println("Code: " + e.getCode());         // COLLECTION_NOT_FOUND
    System.out.println("Message: " + e.getMessage());
    System.out.println("Help: " + e.getHelp());
}
```

## Rate Limiting

Rate limit information is extracted from response headers automatically:

```java
client.listCollections();
RateLimitInfo info = client.getLastRateLimitInfo();
if (info != null) {
    System.out.println("Limit: " + info.getLimit());
    System.out.println("Remaining: " + info.getRemaining());
    System.out.println("Retry-After: " + info.getRetryAfter());
}
```

## Thread Safety

`NeedleClient` is thread-safe and can be shared across threads. The underlying `HttpClient` handles connection pooling internally.

## API Reference

### NeedleClient

| Method | Description |
|--------|-------------|
| `createCollection(name, dimensions)` | Create collection |
| `createCollection(name, dimensions, distance)` | Create collection with distance function |
| `getCollection(name)` | Get collection info |
| `listCollections()` | List all collections |
| `deleteCollection(name)` | Delete collection |
| `insert(collection, vector)` | Insert vector |
| `insert(collection, vector, ttlSeconds)` | Insert vector with TTL |
| `getVector(collection, id)` | Get vector by ID |
| `deleteVector(collection, id)` | Delete vector by ID |
| `search(collection, options)` | Search for similar vectors |
| `isHealthy()` | Check server health |
| `getLastRateLimitInfo()` | Get rate limit info from last response |
