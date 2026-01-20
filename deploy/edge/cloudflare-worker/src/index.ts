// Needle Vector Database — Cloudflare Worker
// Deploy: npx wrangler deploy
// Dev:    npx wrangler dev

export interface Env {
  NEEDLE_KV: KVNamespace;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const headers = { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' };

    try {
      if (url.pathname === '/health') {
        return new Response(JSON.stringify({ status: 'ok', platform: 'cloudflare-workers' }), { headers });
      }

      if (url.pathname === '/insert' && request.method === 'POST') {
        const body = await request.json() as any;
        await env.NEEDLE_KV.put(`vec:${body.id}`, JSON.stringify({ vector: body.vector, metadata: body.metadata }));
        return new Response(JSON.stringify({ inserted: true, id: body.id }), { headers });
      }

      if (url.pathname === '/get' && request.method === 'GET') {
        const id = url.searchParams.get('id');
        if (!id) return new Response(JSON.stringify({ error: 'Missing id' }), { status: 400, headers });
        const data = await env.NEEDLE_KV.get(`vec:${id}`);
        if (!data) return new Response(JSON.stringify({ found: false }), { headers });
        return new Response(JSON.stringify({ found: true, ...JSON.parse(data) }), { headers });
      }

      return new Response(JSON.stringify({ error: 'Not found', endpoints: ['/health', '/insert', '/get'] }), { status: 404, headers });
    } catch (e: any) {
      return new Response(JSON.stringify({ error: e.message }), { status: 500, headers });
    }
  },
};
