# Needle Edge Deployment

Pre-built templates for deploying Needle to serverless edge platforms.

## Cloudflare Workers

```bash
cd deploy/edge/cloudflare-worker
npm install
npx wrangler dev     # local development
npx wrangler deploy  # deploy to Cloudflare
```

## Deno Deploy

```bash
cd deploy/edge/deno-deploy
deno run --allow-net --allow-read server.ts  # local
deployctl deploy --project=needle server.ts  # deploy
```

## Platform Support

| Platform | Status | Storage | Max Bundle |
|----------|--------|---------|------------|
| Cloudflare Workers | Template ready | KV | 1MB (free), 10MB (paid) |
| Deno Deploy | Template ready | Deno KV | 20MB |
| Vercel Edge | Planned | — | 4MB |

## Building the WASM Bundle

```bash
# From project root
wasm-pack build --target bundler --features wasm --out-dir deploy/edge/pkg
```
