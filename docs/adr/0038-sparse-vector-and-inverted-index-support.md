# ADR-0038: Sparse Vector and Inverted Index Support

## Status

Accepted

## Context

Dense vectors (BERT, OpenAI embeddings) excel at semantic similarity but have weaknesses:

1. **Lexical mismatch** — "NYC" and "New York City" may have distant embeddings
2. **Rare terms** — Uncommon words are poorly represented
3. **Exact matching** — Users often expect exact keyword matches
4. **Domain vocabulary** — Technical terms need lexical matching
5. **Interpretability** — Dense vectors are opaque; term weights are explainable

Traditional information retrieval uses **sparse vectors** (TF-IDF, BM25):
- Most dimensions are zero (sparse)
- Non-zero dimensions correspond to vocabulary terms
- Easily interpretable (term weights)
- Excellent for exact and partial term matching

Modern approaches like **SPLADE** combine the best of both:
- Learned sparse representations
- Semantic expansion (related terms get non-zero weights)
- Efficient sparse computation

## Decision

Implement **sparse vector support** with an **inverted index** for efficient sparse retrieval, complementing the dense HNSW index.

### Sparse Vector Representation

```rust
/// Sparse vector: only non-zero dimensions stored
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    /// Non-zero dimension indices (sorted)
    pub indices: Vec<u32>,

    /// Corresponding values
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(indices: Vec<u32>, values: Vec<f32>) -> Self {
        debug_assert_eq!(indices.len(), values.len());
        Self { indices, values }
    }

    /// Create from dense vector (threshold for sparsity)
    pub fn from_dense(dense: &[f32], threshold: f32) -> Self {
        let (indices, values): (Vec<u32>, Vec<f32>) = dense.iter()
            .enumerate()
            .filter(|(_, &v)| v.abs() > threshold)
            .map(|(i, &v)| (i as u32, v))
            .unzip();

        Self { indices, values }
    }

    /// Number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    /// Dot product (efficient sparse-sparse)
    pub fn dot(&self, other: &SparseVector) -> f32 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.indices.len() && j < other.indices.len() {
            match self.indices[i].cmp(&other.indices[j]) {
                std::cmp::Ordering::Equal => {
                    result += self.values[i] * other.values[j];
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }

        result
    }
}
```

### Inverted Index

```rust
/// Inverted index for sparse vector search
pub struct SparseIndex {
    /// Term → list of (doc_id, weight) postings
    inverted_lists: HashMap<u32, Vec<Posting>>,

    /// Document norms for scoring
    doc_norms: HashMap<String, f32>,

    /// Vocabulary size
    vocab_size: usize,
}

#[derive(Clone)]
pub struct Posting {
    pub doc_id: String,
    pub weight: f32,
}

impl SparseIndex {
    /// Add a sparse vector to the index
    pub fn add(&mut self, id: &str, vector: &SparseVector) {
        for (&term_idx, &weight) in vector.indices.iter().zip(&vector.values) {
            self.inverted_lists
                .entry(term_idx)
                .or_default()
                .push(Posting {
                    doc_id: id.to_string(),
                    weight,
                });
        }

        // Store document norm for scoring
        let norm = vector.values.iter().map(|v| v * v).sum::<f32>().sqrt();
        self.doc_norms.insert(id.to_string(), norm);
    }

    /// Search using WAND algorithm for efficiency
    pub fn search(
        &self,
        query: &SparseVector,
        k: usize,
    ) -> Vec<SparseSearchResult> {
        // Accumulate scores for documents
        let mut scores: HashMap<String, f32> = HashMap::new();

        for (&term_idx, &query_weight) in query.indices.iter().zip(&query.values) {
            if let Some(postings) = self.inverted_lists.get(&term_idx) {
                for posting in postings {
                    *scores.entry(posting.doc_id.clone()).or_default() +=
                        query_weight * posting.weight;
                }
            }
        }

        // Normalize and sort
        let mut results: Vec<_> = scores.into_iter()
            .map(|(id, score)| {
                let norm = self.doc_norms.get(&id).copied().unwrap_or(1.0);
                SparseSearchResult {
                    id,
                    score: score / norm,  // Cosine similarity
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        results.truncate(k);
        results
    }
}
```

### TF-IDF Builder

```rust
pub struct TfIdfVectorizer {
    vocabulary: HashMap<String, u32>,
    idf: Vec<f32>,
    doc_count: usize,
}

impl TfIdfVectorizer {
    /// Fit on corpus to learn vocabulary and IDF
    pub fn fit(&mut self, documents: &[&str]) {
        let mut term_doc_counts: HashMap<String, usize> = HashMap::new();

        for doc in documents {
            let terms: HashSet<String> = self.tokenize(doc).collect();
            for term in terms {
                *term_doc_counts.entry(term).or_default() += 1;
            }
        }

        // Build vocabulary and IDF
        for (idx, (term, doc_count)) in term_doc_counts.into_iter().enumerate() {
            self.vocabulary.insert(term, idx as u32);
            // IDF = log(N / df)
            self.idf.push((documents.len() as f32 / doc_count as f32).ln());
        }

        self.doc_count = documents.len();
    }

    /// Transform document to sparse TF-IDF vector
    pub fn transform(&self, document: &str) -> SparseVector {
        let mut term_counts: HashMap<u32, usize> = HashMap::new();

        for term in self.tokenize(document) {
            if let Some(&idx) = self.vocabulary.get(&term) {
                *term_counts.entry(idx).or_default() += 1;
            }
        }

        let total_terms = term_counts.values().sum::<usize>() as f32;

        let mut indices: Vec<u32> = term_counts.keys().copied().collect();
        indices.sort();

        let values: Vec<f32> = indices.iter()
            .map(|&idx| {
                let tf = term_counts[&idx] as f32 / total_terms;
                tf * self.idf[idx as usize]
            })
            .collect();

        SparseVector::new(indices, values)
    }
}
```

### Code References

- `src/sparse.rs` — SparseVector and SparseIndex implementation
- `src/hybrid.rs` — Integration with hybrid search (BM25 + dense)
- `src/lib.rs` — Public exports for sparse types

## Consequences

### Benefits

1. **Lexical matching** — Exact and partial term matches work reliably
2. **Interpretability** — Term weights explain why documents match
3. **Efficiency** — Inverted index is O(query terms × avg postings)
4. **Complementary to dense** — Hybrid search combines both strengths
5. **SPLADE compatibility** — Works with learned sparse models

### Tradeoffs

1. **Vocabulary dependency** — Must build vocabulary from corpus
2. **No semantic generalization** — Unlike dense, doesn't handle synonyms
3. **Storage for large vocab** — Inverted lists can be large
4. **Language-specific** — Tokenization varies by language

### What This Enabled

- Hybrid search (ADR-0012) combining BM25 and vector similarity
- SPLADE integration for learned sparse representations
- Keyword search fallback when semantic search fails
- Explainable relevance scores

### What This Prevented

- Black-box search results (users can see term matches)
- Missing exact matches for technical terms
- Over-reliance on embedding model quality

### Sparse vs. Dense Comparison

| Aspect | Sparse (TF-IDF/SPLADE) | Dense (BERT/OpenAI) |
|--------|------------------------|---------------------|
| Semantic similarity | Limited | Excellent |
| Exact matching | Excellent | Limited |
| Interpretability | High | Low |
| Storage per doc | Variable (avg ~100 terms) | Fixed (768-1536 dims) |
| Query latency | Depends on query length | Consistent |
| Domain adaptation | Needs retraining | Zero-shot capable |

### Usage Example

```rust
use needle::sparse::{SparseVector, SparseIndex, TfIdfVectorizer};

// Build TF-IDF vectorizer
let documents = vec![
    "machine learning algorithms",
    "deep learning neural networks",
    "natural language processing",
];
let mut vectorizer = TfIdfVectorizer::new();
vectorizer.fit(&documents);

// Create sparse index
let mut index = SparseIndex::new(vectorizer.vocabulary_size());

// Add documents
for (i, doc) in documents.iter().enumerate() {
    let sparse_vec = vectorizer.transform(doc);
    index.add(&format!("doc{}", i), &sparse_vec);
}

// Search
let query = "machine learning";
let query_vec = vectorizer.transform(query);
let results = index.search(&query_vec, 5);

for result in results {
    println!("ID: {}, Score: {:.4}", result.id, result.score);
}
```

### Hybrid Search Pattern

```rust
// Combine sparse (lexical) and dense (semantic) search
let sparse_results = sparse_index.search(&sparse_query, 100);
let dense_results = hnsw_index.search(&dense_query, 100);

// Reciprocal Rank Fusion (from ADR-0012)
let hybrid_results = reciprocal_rank_fusion(
    &dense_results,
    &sparse_results,
    &RrfConfig::default(),
    10,
);
```
