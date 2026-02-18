---
sidebar_position: 10
---

# MCP Integration

Needle exposes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server so AI agents can create, search, and manage vector collections directly.

## Prerequisites

```bash
# From crates.io
cargo install needle

# Or via Docker
docker pull ghcr.io/anthropics/needle:latest
```

Verify:

```bash
needle mcp --database vectors.needle
```

## Claude Desktop

Add the following to your Claude Desktop config:

| OS | Config path |
|----|-------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### Local binary

```json
{
  "mcpServers": {
    "needle": {
      "command": "needle",
      "args": ["mcp", "--database", "/path/to/vectors.needle"]
    }
  }
}
```

### Docker

```json
{
  "mcpServers": {
    "needle": {
      "command": "docker",
      "args": [
        "run", "--rm", "-i",
        "-v", "/path/to/data:/data",
        "ghcr.io/anthropics/needle:latest",
        "mcp", "--database", "/data/vectors.needle"
      ]
    }
  }
}
```

Restart Claude Desktop after saving. Needle tools appear in the tool picker (🔧).

## Transports

### stdio (default)

Run `needle mcp --database <path>` as a subprocess. The client communicates over stdin/stdout.

### HTTP

```bash
needle serve --database vectors.needle -a 127.0.0.1:8080
```

The MCP endpoint is at `POST /mcp`. Point your client at `http://127.0.0.1:8080/mcp`.

## Available Tools

| Tool | Description |
|------|-------------|
| `list_collections` | List all collections with dimensions and counts |
| `create_collection` | Create a collection with specified dimensions and distance |
| `collection_info` | Detailed statistics about a collection |
| `insert_vectors` | Insert vectors with optional JSON metadata |
| `search` | ANN search with optional metadata filters |
| `get_vector` | Retrieve a vector and metadata by ID |
| `delete_vector` | Delete a vector by ID |
| `delete_collection` | Delete an entire collection |
| `save_database` | Persist all changes to disk |
| `remember` | Store a memory with embedding for long-term recall |
| `recall` | Retrieve relevant memories by similarity |
| `forget` | Delete a specific memory by ID |
| `memory_consolidate` | Promote important memories and expire old entries |

## Resources

Collections are exposed as MCP resources:

```
needle://collections/{name}
```

Clients that support MCP resource browsing can list and inspect collections directly.

## Troubleshooting

### Tools don't appear in Claude Desktop

1. Ensure `needle` is on your `PATH` (`which needle`).
2. Check Claude Desktop logs for MCP connection errors.
3. Verify the database path exists and is writable.

### Database locked

Only one process can write to a `.needle` file at a time. Stop any running server before starting the MCP server on the same file.

---

## See Also

- [Getting Started](/docs/getting-started) — First steps with Needle
- [API Reference](/docs/api-reference) — Full REST API docs
- [CLI Reference](/docs/advanced/cli) — All CLI commands
