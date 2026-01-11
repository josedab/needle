# Migration and Upgrade Guide

This guide covers upgrading Needle versions, migrating between index types, and migrating from other vector databases.

## Version Compatibility

### Semantic Versioning

Needle follows semantic versioning (`MAJOR.MINOR.PATCH`):

- **PATCH** (0.1.x → 0.1.y): Bug fixes, no API changes, automatic file format upgrade
- **MINOR** (0.1.x → 0.2.0): New features, backward-compatible API, automatic file format upgrade
- **MAJOR** (0.x → 1.0): Breaking changes, may require migration

### File Format Versions

| Needle Version | File Format Version | Notes |
|---------------|---------------------|-------|
| 0.1.x | 1 | Initial format |
| 0.2.x | 2 | Added IVF index support |
| 0.3.x | 3 | Added encryption headers |
| 1.0.x | 4 | Optimized HNSW serialization |

### Compatibility Matrix

| From Version | To Version | Automatic Upgrade | Notes |
|--------------|------------|-------------------|-------|
| 0.1.x | 0.2.x | Yes | Seamless |
| 0.2.x | 0.3.x | Yes | Seamless |
| 0.3.x | 1.0.x | Manual | See migration guide |
| Any | Same major | Yes | Patch/minor upgrades |

---

## Upgrading Needle

### Minor/Patch Upgrades

For patch and minor version upgrades:

```bash
# Update dependency in Cargo.toml
# needle = "0.1" → needle = "0.2"

# Rebuild
cargo build --release

# No data migration needed - format upgrades automatically
```

### Major Version Upgrades

For major version upgrades (e.g., 0.x → 1.0):

1. **Backup First**

```rust
use needle::backup::{BackupManager, BackupConfig};

let manager = BackupManager::new(BackupConfig::default());
manager.create_backup(&db, BackupType::Full)?;
```

2. **Check Breaking Changes**

Review the [CHANGELOG](../CHANGELOG.md) for breaking changes:
- API changes
- Configuration changes
- File format changes
- Removed features

3. **Run Migration Tool**

```bash
# Export with old version
needle export old_db.needle --format json > backup.json

# Install new version
cargo install needle@1.0

# Import with new version
needle import new_db.needle < backup.json
```

4. **Verify Migration**

```rust
// Compare vector counts
assert_eq!(old_db.collection("docs")?.len(), new_db.collection("docs")?.len());

// Sample search comparison
let query = vec![0.1; 384];
let old_results = old_coll.search(&query, 10)?;
let new_results = new_coll.search(&query, 10)?;

// Results should be identical (or very similar for approximate search)
```

### Rolling Upgrades (Distributed)

For distributed clusters:

1. **Upgrade Followers First**
   - Stop follower
   - Upgrade binary
   - Start follower
   - Wait for sync

2. **Upgrade One at a Time**
   - Wait for replication to catch up between upgrades
   - Monitor cluster health

3. **Upgrade Leader Last**
   - Step down leader gracefully
   - Upgrade binary
   - Start node
   - Let election happen naturally

```rust
// Graceful leader step-down
if cluster.is_leader() {
    cluster.step_down().await?;
}
```

---

## Index Type Migration

### HNSW to IVF

When dataset grows beyond HNSW memory limits:

```rust
use needle::{Database, IvfIndex, IvfConfig};

// 1. Open source HNSW collection
let db = Database::open("source.needle")?;
let hnsw_coll = db.collection("documents")?;

// 2. Configure IVF
let ivf_config = IvfConfig::new(1024)  // clusters = sqrt(N)
    .with_nprobe(32)
    .with_product_quantization(8);

let mut ivf = IvfIndex::new(384, ivf_config);

// 3. Collect training samples (10% of data)
let sample_size = hnsw_coll.len() / 10;
let samples: Vec<Vec<f32>> = hnsw_coll.iter()
    .take(sample_size)
    .map(|(_, v, _)| v)
    .collect();

println!("Training IVF on {} samples...", samples.len());
ivf.train(&samples)?;

// 4. Migrate vectors
println!("Migrating {} vectors...", hnsw_coll.len());
let mut migrated = 0;

for (id, vector, metadata) in hnsw_coll.iter() {
    ivf.add_with_metadata(&id, &vector, metadata)?;
    migrated += 1;

    if migrated % 100000 == 0 {
        println!("Migrated {}/{}", migrated, hnsw_coll.len());
    }
}

// 5. Verify
println!("Verifying migration...");
let test_queries: Vec<_> = hnsw_coll.iter().take(100).map(|(_, v, _)| v).collect();

for query in &test_queries {
    let hnsw_results = hnsw_coll.search(query, 10)?;
    let ivf_results = ivf.search(query, 10)?;

    let recall = calculate_recall(&hnsw_results, &ivf_results);
    assert!(recall > 0.9, "Recall too low: {}", recall);
}

println!("Migration complete. Recall verified > 90%");
```

### IVF to DiskANN

For billion-scale datasets:

```rust
use needle::{DiskAnnIndex, DiskAnnConfig};

// 1. Configure DiskANN
let config = DiskAnnConfig::new()
    .with_max_degree(64)
    .with_build_complexity(100)
    .with_search_complexity(50)
    .with_pq_chunks(32);

// 2. Build index (requires all vectors in memory during build)
let vectors: Vec<_> = ivf.iter().collect();
let index = DiskAnnIndex::build(&vectors, &config)?;

// 3. Save to disk
index.save("vectors.diskann")?;

// 4. Load for queries
let index = DiskAnnIndex::load("vectors.diskann")?;
```

### Adding Quantization

Add quantization to existing collection:

```rust
use needle::{ScalarQuantizer, ProductQuantizer};

// Train scalar quantizer on sample
let samples: Vec<_> = collection.iter()
    .take(10000)
    .map(|(_, v, _)| v)
    .collect();

let sq = ScalarQuantizer::train(&samples);

// Create quantized collection
let new_config = CollectionConfig::new("quantized", 384)
    .with_quantization(sq);

db.create_collection_with_config(new_config)?;
let quantized_coll = db.collection("quantized")?;

// Copy with quantization
for (id, vector, metadata) in collection.iter() {
    quantized_coll.insert(&id, &vector, metadata)?;
}
```

---

## Migrating from Other Vector Databases

### From Pinecone

```python
# Export from Pinecone
import pinecone

pinecone.init(api_key="...")
index = pinecone.Index("my-index")

# Fetch all vectors (paginated)
vectors = []
for ids in paginate_ids(index):
    results = index.fetch(ids)
    vectors.extend(results["vectors"].values())

# Save to JSON
import json
with open("pinecone_export.json", "w") as f:
    for v in vectors:
        json.dump({
            "id": v["id"],
            "vector": v["values"],
            "metadata": v.get("metadata", {})
        }, f)
        f.write("\n")
```

```rust
// Import to Needle
use needle::Database;
use serde_json::Value;
use std::io::{BufRead, BufReader};
use std::fs::File;

let db = Database::open("migrated.needle")?;
db.create_collection("documents", 384)?;
let coll = db.collection("documents")?;

let file = File::open("pinecone_export.json")?;
let reader = BufReader::new(file);

for line in reader.lines() {
    let record: Value = serde_json::from_str(&line?)?;

    let id = record["id"].as_str().unwrap();
    let vector: Vec<f32> = record["vector"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let metadata = record.get("metadata").cloned();

    coll.insert(id, &vector, metadata)?;
}

db.save()?;
```

### From Milvus

```python
# Export from Milvus
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
collection = Collection("my_collection")

# Query all vectors
results = collection.query(
    expr="",
    output_fields=["id", "embedding", "metadata"]
)

# Save to JSON
import json
with open("milvus_export.json", "w") as f:
    for r in results:
        json.dump({
            "id": str(r["id"]),
            "vector": r["embedding"],
            "metadata": r.get("metadata", {})
        }, f)
        f.write("\n")
```

### From Weaviate

```python
# Export from Weaviate
import weaviate

client = weaviate.Client("http://localhost:8080")

# Get all objects with vectors
results = client.query.get(
    "Document",
    ["title", "content"]
).with_additional(["id", "vector"]).do()

# Save to JSON
import json
with open("weaviate_export.json", "w") as f:
    for obj in results["data"]["Get"]["Document"]:
        json.dump({
            "id": obj["_additional"]["id"],
            "vector": obj["_additional"]["vector"],
            "metadata": {k: v for k, v in obj.items() if k != "_additional"}
        }, f)
        f.write("\n")
```

### From Qdrant

```python
# Export from Qdrant
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# Scroll through all points
offset = None
all_points = []

while True:
    results, offset = client.scroll(
        collection_name="my_collection",
        offset=offset,
        limit=1000,
        with_vectors=True,
        with_payload=True
    )

    all_points.extend(results)

    if offset is None:
        break

# Save to JSON
import json
with open("qdrant_export.json", "w") as f:
    for point in all_points:
        json.dump({
            "id": str(point.id),
            "vector": point.vector,
            "metadata": point.payload
        }, f)
        f.write("\n")
```

### From ChromaDB

```python
# Export from ChromaDB
import chromadb

client = chromadb.Client()
collection = client.get_collection("my_collection")

# Get all embeddings
results = collection.get(
    include=["embeddings", "metadatas", "documents"]
)

# Save to JSON
import json
with open("chroma_export.json", "w") as f:
    for i, id in enumerate(results["ids"]):
        json.dump({
            "id": id,
            "vector": results["embeddings"][i],
            "metadata": results["metadatas"][i] if results["metadatas"] else {}
        }, f)
        f.write("\n")
```

---

## Embedding Model Migration

When changing embedding models:

### Same Dimensions

If new model has same dimensions, re-embed in place:

```rust
// Create new collection for re-embedded vectors
db.create_collection("documents_v2", 384)?;
let new_coll = db.collection("documents_v2")?;

// Re-embed each document
for (id, _old_vector, metadata) in old_coll.iter() {
    // Get original text from metadata or external source
    let text = get_original_text(&id);

    // Generate new embedding
    let new_vector = embedding_model.embed(&text)?;

    new_coll.insert(&id, &new_vector, metadata)?;
}

// Swap collections using alias
db.update_alias("documents", "documents_v2")?;
db.drop_collection("documents_v1")?;
```

### Different Dimensions

If dimensions change, create new collection:

```rust
// Old: 384 dimensions
// New: 1536 dimensions (e.g., text-embedding-3-large)

db.create_collection("documents_large", 1536)?;
let new_coll = db.collection("documents_large")?;

for (id, _old_vector, metadata) in old_coll.iter() {
    let text = get_original_text(&id);
    let new_vector = new_embedding_model.embed(&text)?;  // 1536 dims

    new_coll.insert(&id, &new_vector, metadata)?;
}
```

---

## Rollback Procedures

### Quick Rollback

If issues arise after upgrade:

```bash
# Stop the service
systemctl stop needle

# Restore from backup
needle restore /backups/pre-upgrade-backup.tar.gz --target data.needle

# Downgrade binary
cargo install needle@previous_version

# Start service
systemctl start needle
```

### Partial Rollback (Collections)

Roll back specific collections:

```rust
// Restore single collection from backup
let backup_db = Database::open("backup.needle")?;
let backup_coll = backup_db.collection("documents")?;

// Drop current and recreate
db.drop_collection("documents")?;
db.create_collection("documents", backup_coll.dimensions())?;
let coll = db.collection("documents")?;

// Copy from backup
for (id, vector, metadata) in backup_coll.iter() {
    coll.insert(&id, &vector, metadata)?;
}
```

---

## Migration Checklist

### Pre-Migration

- [ ] Create full backup
- [ ] Document current configuration
- [ ] Test migration in staging environment
- [ ] Prepare rollback procedure
- [ ] Schedule maintenance window
- [ ] Notify stakeholders

### During Migration

- [ ] Stop writes (if required)
- [ ] Run migration script
- [ ] Monitor progress
- [ ] Verify data integrity

### Post-Migration

- [ ] Run validation queries
- [ ] Compare result quality
- [ ] Check performance metrics
- [ ] Update documentation
- [ ] Remove old data (after verification)

### Validation Queries

```rust
// Count verification
assert_eq!(old_coll.len(), new_coll.len(), "Vector count mismatch");

// Sample search quality
let test_queries: Vec<_> = /* sample queries */;

for query in test_queries {
    let old_results = old_coll.search(&query, 100)?;
    let new_results = new_coll.search(&query, 100)?;

    let recall = calculate_recall_at_k(&old_results, &new_results, 10);
    assert!(recall >= 0.95, "Recall@10 below threshold: {}", recall);
}

// Metadata preservation
for (id, _, old_meta) in old_coll.iter().take(1000) {
    let (_, new_meta) = new_coll.get(&id)?;
    assert_eq!(old_meta, new_meta, "Metadata mismatch for {}", id);
}
```

---

## See Also

- [Index Selection Guide](index-selection-guide.md) - Choosing the right index
- [Production Checklist](production-checklist.md) - Pre-deployment verification
- [Backup and Restore](how-to-guides.md#backup-and-restore) - Backup procedures
- [CHANGELOG](../CHANGELOG.md) - Version history and breaking changes
