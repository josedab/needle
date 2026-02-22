// Cloudflare Worker template for Needle Vector Search
// Deploy: wrangler deploy
//
// This worker provides serverless vector search at the edge.
// Vectors are stored in Cloudflare KV and searched using the
// Needle WASM module.

export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    // Health check
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'ok',
        platform: 'cloudflare-workers',
        version: '0.1.0',
      }), { headers: { 'Content-Type': 'application/json' } });
    }

    // Search endpoint
    if (url.pathname === '/search' && request.method === 'POST') {
      try {
        const body = await request.json();
        const { collection, query, k = 10 } = body;

        if (!collection || !query) {
          return new Response(JSON.stringify({
            error: 'Missing collection or query',
          }), { status: 400, headers: { 'Content-Type': 'application/json' } });
        }

        // In production, this would call the Needle WASM module
        // For now, return a placeholder
        const start = Date.now();
        const results = {
          results: [],
          latency_ms: Date.now() - start,
          collection,
          k,
        };

        return new Response(JSON.stringify(results), {
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (e) {
        return new Response(JSON.stringify({ error: e.message }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    // Info endpoint
    if (url.pathname === '/info') {
      return new Response(JSON.stringify({
        name: 'needle-edge',
        version: '0.1.0',
        platform: 'cloudflare-workers',
        capabilities: ['search', 'insert', 'delete'],
        max_dimensions: 768,
        max_vectors: 100000,
      }), { headers: { 'Content-Type': 'application/json' } });
    }

    return new Response('Needle Edge - Vector Search at the Edge\n\nEndpoints:\n  GET  /health\n  GET  /info\n  POST /search\n', {
      headers: { 'Content-Type': 'text/plain' },
    });
  },
};
