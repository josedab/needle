// Needle Vector Database — Deno Deploy
// Deploy: deployctl deploy --project=needle server.ts
// Dev:    deno run --allow-net --allow-read server.ts

const kv = await Deno.openKv();

Deno.serve({ port: 8080 }, async (request: Request): Promise<Response> => {
  const url = new URL(request.url);
  const headers = { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" };

  try {
    if (url.pathname === "/health") {
      return new Response(JSON.stringify({ status: "ok", platform: "deno-deploy" }), { headers });
    }

    if (url.pathname === "/insert" && request.method === "POST") {
      const body = await request.json();
      await kv.set(["vectors", body.id], { vector: body.vector, metadata: body.metadata });
      return new Response(JSON.stringify({ inserted: true, id: body.id }), { headers });
    }

    if (url.pathname === "/get") {
      const id = url.searchParams.get("id");
      if (!id) return new Response(JSON.stringify({ error: "Missing id" }), { status: 400, headers });
      const entry = await kv.get(["vectors", id]);
      if (!entry.value) return new Response(JSON.stringify({ found: false }), { headers });
      return new Response(JSON.stringify({ found: true, ...(entry.value as object) }), { headers });
    }

    if (url.pathname === "/delete" && request.method === "DELETE") {
      const id = url.searchParams.get("id");
      if (!id) return new Response(JSON.stringify({ error: "Missing id" }), { status: 400, headers });
      await kv.delete(["vectors", id]);
      return new Response(JSON.stringify({ deleted: true }), { headers });
    }

    return new Response(JSON.stringify({ error: "Not found" }), { status: 404, headers });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500, headers });
  }
});

console.log("Needle Deno Deploy server running on :8080");
