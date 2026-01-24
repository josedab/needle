import React, { useState, useCallback } from 'react';
import Layout from '@theme/Layout';

const EXAMPLES = [
  {
    title: 'Semantic Search',
    code: `// Create a collection and search
const db = new Database();
db.createCollection("docs", 384);
db.insert("docs", "doc1", randomVector(384), { title: "Hello World" });
const results = db.search("docs", randomVector(384), 5);
console.log(results);`,
  },
  {
    title: 'Metadata Filtering',
    code: `// Filter search results by metadata
const results = db.search("docs", query, 10, {
  filter: { category: "science", year: { "$gte": 2020 } }
});`,
  },
  {
    title: 'Batch Insert',
    code: `// Insert multiple vectors at once
const vectors = Array.from({ length: 100 }, (_, i) => ({
  id: \`doc\${i}\`,
  vector: randomVector(384),
  metadata: { index: i }
}));
db.insertBatch("docs", vectors);`,
  },
];

function PlaygroundPage(): JSX.Element {
  const [code, setCode] = useState(EXAMPLES[0].code);
  const [output, setOutput] = useState('// Output will appear here');
  const [selectedExample, setSelectedExample] = useState(0);

  const handleRun = useCallback(() => {
    setOutput('// WASM runtime not loaded — this is a preview.\n// Build with: cargo build --target wasm32-unknown-unknown --features wasm');
  }, []);

  const handleShare = useCallback(() => {
    const encoded = btoa(code);
    const url = `${window.location.origin}${window.location.pathname}#snippet=${encoded}`;
    navigator.clipboard?.writeText(url);
    setOutput('// Share URL copied to clipboard!');
  }, [code]);

  const handleCopyAs = useCallback((lang: string) => {
    let generated = '';
    switch (lang) {
      case 'rust':
        generated = `use needle::Database;\n\nfn main() {\n    let db = Database::in_memory();\n    // ${code.split('\n').join('\n    // ')}\n}`;
        break;
      case 'python':
        generated = `import needle\n\ndb = needle.Database.in_memory()\n# ${code.split('\n').join('\n# ')}`;
        break;
      case 'javascript':
        generated = code;
        break;
    }
    navigator.clipboard?.writeText(generated);
    setOutput(`// ${lang.charAt(0).toUpperCase() + lang.slice(1)} code copied to clipboard!`);
  }, [code]);

  return (
    <Layout title="Playground" description="Interactive Needle vector database playground">
      <main style={{ padding: '2rem', maxWidth: '1200px', margin: '0 auto' }}>
        <h1>🎮 Needle Playground</h1>
        <p>Try Needle directly in your browser — no installation required.</p>

        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
          {EXAMPLES.map((ex, i) => (
            <button
              key={i}
              onClick={() => { setSelectedExample(i); setCode(ex.code); }}
              style={{
                padding: '0.5rem 1rem',
                border: selectedExample === i ? '2px solid var(--ifm-color-primary)' : '1px solid #ccc',
                borderRadius: '4px',
                background: selectedExample === i ? 'var(--ifm-color-primary-lightest)' : 'transparent',
                cursor: 'pointer',
              }}
            >
              {ex.title}
            </button>
          ))}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
          <div>
            <h3>Code</h3>
            <textarea
              value={code}
              onChange={(e) => setCode(e.target.value)}
              style={{
                width: '100%',
                height: '300px',
                fontFamily: 'monospace',
                fontSize: '14px',
                padding: '1rem',
                border: '1px solid #ccc',
                borderRadius: '4px',
              }}
            />
          </div>
          <div>
            <h3>Output</h3>
            <pre style={{
              width: '100%',
              height: '300px',
              padding: '1rem',
              border: '1px solid #ccc',
              borderRadius: '4px',
              overflow: 'auto',
              background: '#f6f8fa',
            }}>
              {output}
            </pre>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', flexWrap: 'wrap' }}>
          <button onClick={handleRun} style={{ padding: '0.5rem 1.5rem', background: 'var(--ifm-color-primary)', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
            ▶ Run
          </button>
          <button onClick={handleShare} style={{ padding: '0.5rem 1rem', border: '1px solid #ccc', borderRadius: '4px', cursor: 'pointer' }}>
            🔗 Share
          </button>
          <button onClick={() => handleCopyAs('rust')} style={{ padding: '0.5rem 1rem', border: '1px solid #ccc', borderRadius: '4px', cursor: 'pointer' }}>
            Copy as Rust
          </button>
          <button onClick={() => handleCopyAs('python')} style={{ padding: '0.5rem 1rem', border: '1px solid #ccc', borderRadius: '4px', cursor: 'pointer' }}>
            Copy as Python
          </button>
          <button onClick={() => handleCopyAs('javascript')} style={{ padding: '0.5rem 1rem', border: '1px solid #ccc', borderRadius: '4px', cursor: 'pointer' }}>
            Copy as JS
          </button>
        </div>
      </main>
    </Layout>
  );
}

export default PlaygroundPage;
