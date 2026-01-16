---
slug: documentation-refresh
title: "Needle Documentation Refresh: New Guides, Architecture Deep-Dive, and More"
authors: [needle-team]
tags: [announcement, tutorial]
---

We've shipped a major documentation refresh for Needle. Whether you're deploying to production, choosing the right index, or trying to understand how HNSW works under the hood, we've got you covered.

<!-- truncate -->

## What's New

### Architecture Deep-Dive

Ever wondered how Needle stores vectors in a single file, or how the HNSW index maintains thread safety? Our new [Architecture](/docs/architecture) page walks through the entire system—from the database layer down to the storage format—with Mermaid diagrams showing data flow, module relationships, and the query pipeline.

### Production-Ready Guides

We've added guides for teams taking Needle to production:

- **[Production Checklist](/docs/guides/production-checklist)** — Capacity planning, security hardening, monitoring, and backup strategies in one place.
- **[Deployment Guide](/docs/advanced/deployment)** — Docker, Docker Compose, Kubernetes, and Helm configurations with copy-paste manifests.
- **[Operations Guide](/docs/advanced/operations)** — Prometheus metrics, Grafana dashboards, backup/restore procedures, and troubleshooting runbooks.

### Index Selection Guide

Not sure whether to use HNSW, IVF, or DiskANN? The new [Index Selection Guide](/docs/guides/index-selection) includes a decision flowchart, parameter tuning tables, and auto-tuning examples to help you pick the right index for your workload.

### Docker & HTTP Quickstart

If you prefer running Needle as an HTTP server, the [Docker Quickstart](/docs/guides/docker-quickstart) gets you from `docker pull` to your first search query in under two minutes, with the full REST API endpoint table for reference.

### Distributed Operations

For large-scale deployments, the [Distributed Operations](/docs/advanced/distributed) guide covers Raft-based replication, hash and range sharding, and cluster management patterns.

### API Stability Tiers

We've formalized our [API Stability](/docs/api-stability) policy with three tiers—Stable, Beta, and Experimental—so you know exactly what to expect from each API surface.

## Accuracy Improvements

We audited every code example in the documentation against the actual Rust API signatures. Key corrections include:

- `Database::in_memory()` returns `Database` directly (no `Result` wrapper)
- `create_collection()` takes two arguments `(name, dimensions)` with cosine distance as the default
- `search()` takes two arguments `(query, k)`; use `search_with_filter()` or the `query()` builder for filtered searches
- `insert()` metadata parameter is `Option<Value>`, wrapped with `Some(json!(...))`

All code examples across the docs are now consistent with the actual API.

## What's Next

We're continuing to improve the docs based on community feedback. If you spot an issue or want to suggest a new guide, [open an issue](https://github.com/anthropics/needle/issues) or submit a PR.

Happy searching! 🔍
