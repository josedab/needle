# API Stability Policy

Needle's public API is organized into three stability tiers:

## Stable

Types re-exported in the **Stable API** section of `lib.rs`. These are the core types
most users need:

- `Database`, `CollectionRef`, `Collection`, `Filter`, `SearchResult`
- `HnswConfig`, `HnswIndex`, `DistanceFunction`
- `NeedleError`, `Result`
- Quantizers, sparse vectors, multi-vectors

**Guarantee**: Breaking changes follow semver. Deprecation warnings will appear at
least one minor version before removal.

## Beta

Types re-exported in the **Beta API** section. These are feature-complete but may see
breaking changes before 1.0:

- Backup, encryption, security, Raft, sharding
- Drift detection, anomaly detection, knowledge graphs
- Query builders, rerankers, routing

**Guarantee**: Breaking changes are documented in CHANGELOG.md. We aim for stability
but reserve the right to adjust APIs based on user feedback.

## Experimental

Types re-exported in the **Experimental API** section. These are under active
development:

- Agentic memory, analytics dashboards, cloud control plane
- GPU acceleration, distributed HNSW, edge runtimes
- Plugin systems, streaming upsert, zero-copy bindings

**Guarantee**: None. APIs may change or be removed without notice. Do not depend on
these in production without pinning to an exact version.

## Promoting a Module

New modules follow this path:

1. **Experimental**: Added to `src/` with `pub mod` in the Experimental section of
   `lib.rs`. Must have unit tests.
2. **Beta**: Promoted after: API review, doc comments on all public items, integration
   tests, and an ADR documenting the design.
3. **Stable**: Promoted after: 1+ minor versions in Beta without breaking changes,
   comprehensive doc tests, and benchmarks.

## Deprecation Process

1. Add `#[deprecated(since = "X.Y.Z", note = "use `new_name` instead")]`
2. Document in CHANGELOG.md under "Deprecated"
3. Keep deprecated items for at least one minor version
4. Remove in the next major version
