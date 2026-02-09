# MCP Integration Guide

Needle exposes a [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server so AI agents can create, search, and manage vector collections directly. This guide covers how to connect MCP clients to Needle.

## Prerequisites

Install Needle via one of:

```bash
# From crates.io
cargo install needle

# Or run via Docker
docker pull ghcr.io/anthropics/needle:latest
```

Verify the MCP server starts:

```bash
needle mcp --database vectors.needle
```

## Claude Desktop

Add the following to your Claude Desktop configuration file:

| OS | Config path |
|----|-------------|
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

### Using a local binary

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

### Using Docker

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

Restart Claude Desktop after saving the configuration. Needle tools will appear in the tool picker (🔧 icon).

## Other MCP Clients

Needle supports both `stdio` and `http` transports. Any MCP-compatible client can connect using the same patterns.

### stdio transport (default)

Run `needle mcp --database <path>` as a subprocess. The client communicates over stdin/stdout.

### HTTP transport

Start Needle in server mode with the MCP endpoint enabled:

```bash
needle serve --database vectors.needle -a 127.0.0.1:8080
```

The MCP HTTP endpoint is available at `POST /mcp`. Point your MCP client at `http://127.0.0.1:8080/mcp`.

## Available Tools

The MCP server exposes the following tools (defined in [`mcp-registry.json`](../mcp-registry.json)):

| Tool | Description |
|------|-------------|
| `list_collections` | List all vector collections with dimensions and counts |
| `create_collection` | Create a new collection with specified dimensions and distance function |
| `collection_info` | Get detailed statistics about a collection |
| `insert_vectors` | Insert one or more vectors with optional JSON metadata |
| `search` | ANN search with optional MongoDB-style metadata filters |
| `get_vector` | Retrieve a specific vector and its metadata by ID |
| `delete_vector` | Delete a vector from a collection by ID |
| `delete_collection` | Delete an entire collection |
| `save_database` | Persist all changes to disk |
| `remember` | Store a memory with embedding vector for long-term recall |
| `recall` | Retrieve relevant memories by vector similarity |
| `forget` | Delete a specific memory by ID |
| `memory_consolidate` | Promote important memories and expire old entries |

## Resources

The MCP server also exposes collections as MCP resources:

```
needle://collections/{name}
```

Clients that support MCP resource browsing can list and inspect collections directly.

## Troubleshooting

### Tools don't appear in Claude Desktop

1. Ensure `needle` is on your `PATH` (run `which needle` or `where needle`).
2. Check the Claude Desktop logs for MCP connection errors.
3. Verify the database path exists and is writable.

### Permission denied

If using Docker, make sure the mounted volume has the correct permissions:

```bash
chmod 777 /path/to/data
```

### Database locked

Only one process can write to a `.needle` file at a time. Stop any running Needle server or CLI before starting the MCP server on the same database file.
