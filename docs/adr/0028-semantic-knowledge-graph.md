# ADR-0028: Semantic Knowledge Graph Overlay

## Status

Accepted

## Context

Vector similarity search finds semantically related content but lacks structured knowledge:

1. **No explicit relationships** — Vectors encode similarity, not "is-a" or "part-of" relations
2. **Entity ambiguity** — "Apple" the company vs "apple" the fruit have similar embeddings in some contexts
3. **Missing context** — Related entities don't appear in search results unless vectors happen to be similar
4. **No reasoning** — Can't traverse relationships like "find products by the same manufacturer"

Knowledge graphs complement vector search:
- **Structured relationships** — Explicit typed edges between entities
- **Entity disambiguation** — Canonical entity IDs resolve ambiguity
- **Graph traversal** — Multi-hop queries across relationships
- **Contextual enrichment** — Add related entities to search results

### Alternatives Considered

1. **External graph database (Neo4j)** — Operational complexity, sync challenges
2. **Metadata-only relationships** — Limited traversal, no graph queries
3. **Vector-only (encode relationships)** — Loses explicit structure

## Decision

Needle implements an optional **semantic knowledge graph overlay** that coexists with vector indices:

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Query Layer                                 │
│                                                                  │
│   Vector Search ──────────┬──────────── Graph Traversal         │
│        │                  │                    │                 │
│        ▼                  ▼                    ▼                 │
│   ┌─────────┐      ┌─────────────┐      ┌───────────┐          │
│   │  HNSW   │      │   Entity    │      │   Edge    │          │
│   │  Index  │◄────▶│   Index     │◄────▶│   Index   │          │
│   └─────────┘      └─────────────┘      └───────────┘          │
│        │                  │                    │                 │
│        └──────────────────┴────────────────────┘                │
│                           │                                      │
│                    ┌──────▼──────┐                              │
│                    │   Storage   │                              │
│                    └─────────────┘                              │
└─────────────────────────────────────────────────────────────────┘
```

### Entity Model

```rust
// src/knowledge_graph.rs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique entity identifier
    pub id: String,

    /// Entity type (e.g., "Person", "Product", "Organization")
    pub entity_type: String,

    /// Canonical name
    pub name: String,

    /// Alternative names/aliases for matching
    pub aliases: Vec<String>,

    /// Associated vector ID (if entity has a vector representation)
    pub vector_id: Option<String>,

    /// Entity properties
    pub properties: HashMap<String, PropertyValue>,

    /// External identifiers (Wikipedia, Wikidata, etc.)
    pub external_ids: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PropertyValue {
    String(String),
    Number(f64),
    Bool(bool),
    Date(String),
    List(Vec<PropertyValue>),
}
```

### Relationship Model

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relation {
    /// Unique relation identifier
    pub id: String,

    /// Source entity ID
    pub source: String,

    /// Relation type (e.g., "works_for", "manufactured_by", "located_in")
    pub relation_type: String,

    /// Target entity ID
    pub target: String,

    /// Relation properties (weight, confidence, timestamps)
    pub properties: HashMap<String, PropertyValue>,

    /// Bidirectional flag
    pub bidirectional: bool,
}

/// Predefined relation types with semantics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationType {
    // Hierarchical
    IsA,           // "dog" IsA "animal"
    PartOf,        // "wheel" PartOf "car"
    InstanceOf,    // "Fido" InstanceOf "dog"

    // Associative
    RelatedTo,     // general association
    SimilarTo,     // semantic similarity
    SynonymOf,     // same meaning
    AntonymOf,     // opposite meaning

    // Domain-specific
    WorksFor,      // employment
    LocatedIn,     // geography
    CreatedBy,     // authorship
    ManufacturedBy,// products
    OwnedBy,       // ownership

    // Temporal
    Precedes,      // time ordering
    Follows,
    During,

    // Custom
    Custom(String),
}
```

### Knowledge Graph Index

```rust
// src/knowledge_graph.rs
pub struct KnowledgeGraph {
    /// Entity storage (id -> Entity)
    entities: HashMap<String, Entity>,

    /// Relation storage (id -> Relation)
    relations: HashMap<String, Relation>,

    /// Forward edge index: source -> [(relation_type, target)]
    outgoing: HashMap<String, Vec<(String, String)>>,

    /// Reverse edge index: target -> [(relation_type, source)]
    incoming: HashMap<String, Vec<(String, String)>>,

    /// Type index: entity_type -> [entity_id]
    by_type: HashMap<String, HashSet<String>>,

    /// Alias index for entity resolution
    alias_index: HashMap<String, String>,

    /// Configuration
    config: KnowledgeGraphConfig,
}

#[derive(Debug, Clone)]
pub struct KnowledgeGraphConfig {
    /// Maximum graph traversal depth
    pub max_traversal_depth: usize,

    /// Whether to auto-link vectors to entities by ID
    pub auto_link_vectors: bool,

    /// Entity types to index
    pub indexed_types: Option<HashSet<String>>,
}
```

### Graph Operations

```rust
impl KnowledgeGraph {
    /// Add an entity to the graph
    pub fn add_entity(&mut self, entity: Entity) -> Result<()> {
        let id = entity.id.clone();

        // Index aliases for resolution
        for alias in &entity.aliases {
            self.alias_index.insert(alias.to_lowercase(), id.clone());
        }
        self.alias_index.insert(entity.name.to_lowercase(), id.clone());

        // Index by type
        self.by_type
            .entry(entity.entity_type.clone())
            .or_default()
            .insert(id.clone());

        self.entities.insert(id, entity);
        Ok(())
    }

    /// Add a relation between entities
    pub fn add_relation(&mut self, relation: Relation) -> Result<()> {
        // Validate entities exist
        if !self.entities.contains_key(&relation.source) {
            return Err(NeedleError::EntityNotFound(relation.source.clone()));
        }
        if !self.entities.contains_key(&relation.target) {
            return Err(NeedleError::EntityNotFound(relation.target.clone()));
        }

        // Index edges
        self.outgoing
            .entry(relation.source.clone())
            .or_default()
            .push((relation.relation_type.clone(), relation.target.clone()));

        self.incoming
            .entry(relation.target.clone())
            .or_default()
            .push((relation.relation_type.clone(), relation.source.clone()));

        // Handle bidirectional
        if relation.bidirectional {
            self.outgoing
                .entry(relation.target.clone())
                .or_default()
                .push((relation.relation_type.clone(), relation.source.clone()));

            self.incoming
                .entry(relation.source.clone())
                .or_default()
                .push((relation.relation_type.clone(), relation.target.clone()));
        }

        self.relations.insert(relation.id.clone(), relation);
        Ok(())
    }

    /// Resolve text to entity (entity linking)
    pub fn resolve_entity(&self, text: &str) -> Option<&Entity> {
        let normalized = text.to_lowercase();
        self.alias_index
            .get(&normalized)
            .and_then(|id| self.entities.get(id))
    }

    /// Get related entities within N hops
    pub fn traverse(
        &self,
        start: &str,
        relation_types: Option<&[RelationType]>,
        max_depth: usize,
    ) -> Vec<TraversalResult> {
        let mut visited = HashSet::new();
        let mut results = Vec::new();
        let mut queue = VecDeque::new();

        queue.push_back((start.to_string(), 0, Vec::new()));
        visited.insert(start.to_string());

        while let Some((current, depth, path)) = queue.pop_front() {
            if depth > 0 {
                results.push(TraversalResult {
                    entity_id: current.clone(),
                    depth,
                    path: path.clone(),
                });
            }

            if depth >= max_depth {
                continue;
            }

            if let Some(edges) = self.outgoing.get(&current) {
                for (rel_type, target) in edges {
                    // Filter by relation type if specified
                    if let Some(types) = relation_types {
                        if !types.iter().any(|t| t.matches(rel_type)) {
                            continue;
                        }
                    }

                    if !visited.contains(target) {
                        visited.insert(target.clone());
                        let mut new_path = path.clone();
                        new_path.push((rel_type.clone(), target.clone()));
                        queue.push_back((target.clone(), depth + 1, new_path));
                    }
                }
            }
        }

        results
    }
}

#[derive(Debug, Clone)]
pub struct TraversalResult {
    pub entity_id: String,
    pub depth: usize,
    pub path: Vec<(String, String)>, // (relation_type, entity_id) pairs
}
```

### Integration with Vector Search

```rust
impl Collection {
    /// Search with knowledge graph enrichment
    pub fn search_with_graph(
        &self,
        query: &[f32],
        k: usize,
        graph: &KnowledgeGraph,
        enrichment: GraphEnrichment,
    ) -> Result<Vec<EnrichedSearchResult>> {
        // Standard vector search
        let results = self.search(query, k)?;

        // Enrich each result with graph context
        let enriched = results
            .into_iter()
            .map(|r| {
                let mut enriched = EnrichedSearchResult {
                    id: r.id.clone(),
                    distance: r.distance,
                    metadata: r.metadata,
                    entities: Vec::new(),
                    related: Vec::new(),
                };

                // Find linked entity
                if let Some(entity) = graph.entities.values()
                    .find(|e| e.vector_id.as_ref() == Some(&r.id))
                {
                    enriched.entities.push(entity.clone());

                    // Get related entities
                    if enrichment.include_related {
                        let related = graph.traverse(
                            &entity.id,
                            enrichment.relation_types.as_deref(),
                            enrichment.max_depth,
                        );
                        enriched.related = related;
                    }
                }

                enriched
            })
            .collect();

        Ok(enriched)
    }
}

#[derive(Debug, Clone)]
pub struct GraphEnrichment {
    pub include_related: bool,
    pub relation_types: Option<Vec<RelationType>>,
    pub max_depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct EnrichedSearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: Option<Value>,
    pub entities: Vec<Entity>,
    pub related: Vec<TraversalResult>,
}
```

### Example Usage

```rust
// Build knowledge graph
let mut graph = KnowledgeGraph::new(KnowledgeGraphConfig::default());

// Add entities
graph.add_entity(Entity {
    id: "apple_inc".into(),
    entity_type: "Organization".into(),
    name: "Apple Inc.".into(),
    aliases: vec!["Apple".into(), "Apple Computer".into()],
    vector_id: Some("doc_apple_wiki".into()),
    properties: hashmap! { "founded" => PropertyValue::String("1976".into()) },
    external_ids: hashmap! { "wikidata" => "Q312".into() },
})?;

graph.add_entity(Entity {
    id: "tim_cook".into(),
    entity_type: "Person".into(),
    name: "Tim Cook".into(),
    aliases: vec!["Timothy Cook".into()],
    vector_id: Some("doc_tim_cook".into()),
    properties: HashMap::new(),
    external_ids: HashMap::new(),
})?;

// Add relationship
graph.add_relation(Relation {
    id: "rel_1".into(),
    source: "tim_cook".into(),
    relation_type: "works_for".into(),
    target: "apple_inc".into(),
    properties: hashmap! { "role" => PropertyValue::String("CEO".into()) },
    bidirectional: false,
})?;

// Search with graph enrichment
let results = collection.search_with_graph(
    &query_vector,
    10,
    &graph,
    GraphEnrichment {
        include_related: true,
        relation_types: Some(vec![RelationType::WorksFor]),
        max_depth: 2,
    },
)?;
```

## Consequences

### Benefits

1. **Structured knowledge** — Explicit relationships complement vector similarity
2. **Entity disambiguation** — Resolve "Apple" to the correct entity
3. **Contextual results** — Include related entities in search output
4. **Graph queries** — Traverse relationships independent of vector search

### Tradeoffs

1. **Manual curation** — Graph must be populated (not auto-generated)
2. **Sync complexity** — Graph and vectors must stay consistent
3. **Memory overhead** — Graph indices consume additional memory

### What This Enabled

- **Enterprise search** — Organization hierarchies, product catalogs
- **Recommendation systems** — "Users who bought X also bought Y"
- **Question answering** — Multi-hop reasoning over facts

### What This Prevented

- **Pure embedding limitations** — Not everything is captured by vectors
- **External graph dependency** — No need for separate Neo4j deployment

## References

- Knowledge graph implementation: `src/knowledge_graph.rs`
- Entity model: `src/knowledge_graph.rs:20-60`
- Graph traversal: `src/knowledge_graph.rs:200-300`
- Search integration: `src/collection.rs` (search_with_graph method)
