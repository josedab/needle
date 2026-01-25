---
sidebar_position: 2
---

# Command Line Interface

Needle includes a powerful CLI for managing databases, collections, and vectors from the command line.

## Installation

```bash
# Install from crates.io
cargo install needle

# Or build from source
git clone https://github.com/anthropics/needle
cd needle
cargo install --path .
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `create` | Create a new database |
| `create-collection` | Create a collection |
| `info` | Show database info |
| `collections` | List collections |
| `stats` | Collection statistics |
| `insert` | Insert vectors from stdin |
| `get` | Get vector by ID |
| `search` | Search similar vectors |
| `delete` | Delete vector by ID |
| `export` | Export collection to JSON |
| `import` | Import from JSON |
| `count` | Count vectors |
| `clear` | Delete all vectors |
| `compact` | Reclaim deleted space |
| `serve` | Start HTTP server |
| `tune` | Auto-tune HNSW parameters |
| `alias` | Manage collection aliases |
| `ttl` | Manage vector expiration |

## Database Operations

### Create Database

```bash
# Create new empty database
needle create mydb.needle

# Create with initial collection
needle create mydb.needle -c documents -d 384
```

### Database Info

```bash
needle info mydb.needle
```

Output:
```
Database: mydb.needle
Size: 1.2 GB
Collections: 3
Total vectors: 1,234,567
```

## Collection Operations

### Create Collection

```bash
# Basic creation
needle create-collection mydb.needle -n documents -d 384

# With distance function
needle create-collection mydb.needle -n documents -d 384 --distance cosine

# With HNSW parameters
needle create-collection mydb.needle -n documents -d 384 \
  --hnsw-m 32 \
  --hnsw-ef-construction 400

# With quantization
needle create-collection mydb.needle -n documents -d 384 \
  --quantization scalar
```

### List Collections

```bash
needle collections mydb.needle
```

Output:
```
Collections:
  - documents (384 dims, cosine, 50,000 vectors)
  - images (512 dims, euclidean, 10,000 vectors)
```

### Collection Statistics

```bash
needle stats mydb.needle -c documents
```

Output:
```
Collection: documents
Dimensions: 384
Distance: Cosine
Vector count: 50,000
HNSW M: 16
HNSW ef_construction: 200
Memory usage: 125 MB
```

### Delete Collection

```bash
needle delete-collection mydb.needle -c old_collection
```

## Vector Operations

### Insert Vectors

From JSON stdin:
```bash
# Single vector
echo '{"id":"doc1","vector":[0.1,0.2,...],"metadata":{"title":"Hello"}}' | \
  needle insert mydb.needle -c documents

# Multiple vectors (NDJSON)
cat vectors.ndjson | needle insert mydb.needle -c documents

# From file
needle insert mydb.needle -c documents < vectors.ndjson
```

JSON format:
```json
{"id": "doc1", "vector": [0.1, 0.2, 0.3], "metadata": {"title": "Hello"}}
{"id": "doc2", "vector": [0.2, 0.3, 0.4], "metadata": {"title": "World"}}
```

### Get Vector

```bash
needle get mydb.needle -c documents -i doc1
```

Output:
```json
{
  "id": "doc1",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "title": "Hello World"
  }
}
```

### Delete Vector

```bash
needle delete mydb.needle -c documents -i doc1
```

### Count Vectors

```bash
needle count mydb.needle -c documents
```

Output:
```
50000
```

### Clear Collection

```bash
# Clear all vectors (requires confirmation)
needle clear mydb.needle -c documents

# Skip confirmation
needle clear mydb.needle -c documents --yes
```

## Search

### Basic Search

```bash
# Search by vector (JSON array)
needle search mydb.needle -c documents \
  -q "[0.1, 0.2, 0.3, ...]" \
  -k 10
```

Output:
```json
[
  {"id": "doc1", "distance": 0.123, "metadata": {"title": "Hello"}},
  {"id": "doc2", "distance": 0.456, "metadata": {"title": "World"}}
]
```

### Search with Filter

```bash
needle search mydb.needle -c documents \
  -q "[0.1, 0.2, ...]" \
  -k 10 \
  -f '{"category": "programming"}'
```

### Search with Parameters

```bash
needle search mydb.needle -c documents \
  -q "[0.1, 0.2, ...]" \
  -k 10 \
  --ef-search 100
```

### Search with Distance Override

Override the distance function at query time (falls back to brute-force if different from index):

```bash
needle search mydb.needle -c documents \
  -q "[0.1, 0.2, ...]" \
  -k 10 \
  --distance euclidean
```

Available distance functions: `cosine`, `euclidean`, `dot`, `manhattan`

### Search from File

```bash
# Read query vector from file
needle search mydb.needle -c documents \
  --query-file query.json \
  -k 10
```

## Import/Export

### Export Collection

```bash
# Export to stdout (NDJSON)
needle export mydb.needle -c documents

# Export to file
needle export mydb.needle -c documents > backup.ndjson

# Export with filter
needle export mydb.needle -c documents \
  -f '{"category": "important"}'
```

### Import Collection

```bash
# Import from file
needle import mydb.needle -c documents < backup.ndjson

# Import from stdin
cat backup.ndjson | needle import mydb.needle -c documents

# Overwrite existing (default: skip duplicates)
needle import mydb.needle -c documents --overwrite < backup.ndjson
```

## Maintenance

### Compact Collection

Reclaim space from deleted vectors:

```bash
needle compact mydb.needle -c documents
```

Output:
```
Compacted collection 'documents'
Reclaimed: 50 MB
```

### Auto-Tune Parameters

```bash
needle tune mydb.needle -c documents \
  --target-vectors 1000000 \
  --profile high-recall \
  --memory-budget 4G
```

Output:
```
Recommended configuration:
  hnsw_m: 24
  hnsw_ef_construction: 300
  estimated_recall: 98.5%
  estimated_memory: 3.2 GB
```

## Alias Management

Aliases provide alternative names for collections, useful for blue-green deployments.

### Create Alias

```bash
needle alias create -d mydb.needle --alias prod --collection documents_v2
```

### List Aliases

```bash
needle alias list -d mydb.needle
```

Output:
```
Aliases:
  prod -> documents_v2
  staging -> documents_v1
```

### Resolve Alias

```bash
needle alias resolve -d mydb.needle --alias prod
```

Output:
```
prod -> documents_v2
```

### Update Alias

```bash
needle alias update -d mydb.needle --alias prod --collection documents_v3
```

### Delete Alias

```bash
needle alias delete -d mydb.needle --alias old_alias
```

## TTL / Expiration

Manage automatic expiration of vectors.

### View TTL Statistics

```bash
needle ttl stats -d mydb.needle -c documents
```

Output:
```
TTL Statistics for 'documents':
  Vectors with TTL: 10,000
  Currently expired: 234
  Nearest expiration: 2024-01-15 10:30:00 UTC
  Furthest expiration: 2024-01-22 15:45:00 UTC
```

### Sweep Expired Vectors

Remove all expired vectors from a collection:

```bash
needle ttl sweep -d mydb.needle -c documents
```

Output:
```
Swept 234 expired vectors from 'documents'
```

## HTTP Server

### Start Server

```bash
# Basic
needle serve -d mydb.needle

# Custom address
needle serve -a 0.0.0.0:8080 -d mydb.needle

# With TLS
needle serve -a 0.0.0.0:8443 -d mydb.needle \
  --tls-cert cert.pem \
  --tls-key key.pem

# With metrics
needle serve -d mydb.needle --metrics-port 9090
```

## Global Options

| Option | Description |
|--------|-------------|
| `-v, --verbose` | Enable verbose output |
| `-q, --quiet` | Suppress non-essential output |
| `--json` | Output in JSON format |
| `--help` | Show help |
| `--version` | Show version |

## Examples

### Build a Search Index

```bash
#!/bin/bash

# Create database and collection
needle create search.needle
needle create-collection search.needle -n articles -d 384 --distance cosine

# Index documents (from embeddings file)
cat embeddings.ndjson | needle insert search.needle -c articles

# Show stats
needle stats search.needle -c articles

# Search
needle search search.needle -c articles \
  -q "[$(python3 embed.py "search query")]" \
  -k 5
```

### Backup and Restore

```bash
#!/bin/bash

# Backup
needle export mydb.needle -c documents > backup_$(date +%Y%m%d).ndjson

# Restore to new database
needle create restored.needle
needle create-collection restored.needle -n documents -d 384
needle import restored.needle -c documents < backup_20240115.ndjson
```

### Batch Processing

```bash
#!/bin/bash

# Process multiple queries
while IFS= read -r query; do
  needle search mydb.needle -c documents -q "$query" -k 5 --json
done < queries.txt
```

## Next Steps

- [HTTP Server](/docs/advanced/http-server)
- [API Reference](/docs/api-reference)
- [Production Deployment](/docs/guides/production)
