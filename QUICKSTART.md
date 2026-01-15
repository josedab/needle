# Quickstart

Pick **one** path and copy-paste the block. Each ends with visible output.

## Path 1: Rust Library

```bash
cargo add needle serde_json
```

```rust
use needle::{Database, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    let db = Database::in_memory();
    db.create_collection("docs", 3)?;
    let coll = db.collection("docs")?;

    coll.insert("a", &[1.0, 0.0, 0.0], Some(json!({"kind": "greeting"})))?;
    coll.insert("b", &[0.0, 1.0, 0.0], Some(json!({"kind": "farewell"})))?;
    coll.insert("c", &[0.0, 0.0, 1.0], Some(json!({"kind": "greeting"})))?;

    let results = coll.search(&[0.9, 0.1, 0.0], 2)?;
    for r in &results {
        println!("{} (distance: {:.4})", r.id, r.distance);
    }

    let filtered = coll.search_with_filter(&[0.9, 0.1, 0.0], 2, &Filter::eq("kind", "greeting"))?;
    println!("Filtered: {} results", filtered.len());

    Ok(())
}
```

## Path 2: HTTP Server (Docker)

```bash
git clone https://github.com/anthropics/needle.git && cd needle
docker compose --profile demo up -d --build
curl -s http://127.0.0.1:8080/collections/demo/search \
  -H "Content-Type: application/json" \
  -d '{"vector":[0.1,0.2,0.3],"k":3}' | python3 -m json.tool
```

## Path 3: HTTP Server (from source)

```bash
git clone https://github.com/anthropics/needle.git && cd needle
make demo
```

## What Next?

| Goal | Command |
|------|---------|
| Run all examples | `cargo run --example basic_usage` |
| Interactive HTTP walkthrough | `make playground` |
| Full API docs | `cargo doc --open` |
| See all CLI commands | `cargo run -- --help` |
