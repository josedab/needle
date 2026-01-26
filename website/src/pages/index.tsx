import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';
import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.badges}>
          <a href="https://github.com/anthropics/needle" target="_blank" rel="noopener noreferrer">
            <img src="https://img.shields.io/github/stars/anthropics/needle?style=social" alt="GitHub stars" />
          </a>
          <a href="https://crates.io/crates/needle" target="_blank" rel="noopener noreferrer">
            <img src="https://img.shields.io/crates/v/needle.svg" alt="Crates.io" />
          </a>
          <a href="https://github.com/anthropics/needle/actions/workflows/ci.yml" target="_blank" rel="noopener noreferrer">
            <img src="https://github.com/anthropics/needle/actions/workflows/ci.yml/badge.svg" alt="CI" />
          </a>
          <a href="https://codecov.io/gh/anthropics/needle" target="_blank" rel="noopener noreferrer">
            <img src="https://codecov.io/gh/anthropics/needle/branch/main/graph/badge.svg" alt="codecov" />
          </a>
        </div>
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.installTabs}>
          <Tabs>
            <TabItem value="rust" label="Rust" default>
              <div className={styles.installCommand}>
                <code>cargo add needle</code>
              </div>
            </TabItem>
            <TabItem value="python" label="Python">
              <div className={styles.installCommand}>
                <code>pip install needle-db</code>
              </div>
            </TabItem>
            <TabItem value="javascript" label="JavaScript">
              <div className={styles.installCommand}>
                <code>npm install @anthropic/needle</code>
              </div>
            </TabItem>
          </Tabs>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--primary button--lg"
            to="/docs/getting-started">
            Get Started
          </Link>
          <Link
            className="button button--secondary button--lg"
            to="/docs/api-reference">
            API Reference
          </Link>
          <Link
            className="button button--secondary button--lg"
            href="https://github.com/anthropics/needle">
            GitHub
          </Link>
        </div>
      </div>
    </header>
  );
}

function ArchitectureDiagram() {
  return (
    <section className={styles.architecture}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <Heading as="h2">How Needle Works</Heading>
            <p>
              Needle embeds directly into your application—no network calls, no separate processes.
              All data lives in a single <code>.needle</code> file that you can backup, copy, or distribute.
            </p>
            <ul className={styles.architectureList}>
              <li><strong>Embedded Architecture</strong> — Links directly to your app</li>
              <li><strong>HNSW Index</strong> — O(log n) approximate nearest neighbor search</li>
              <li><strong>Memory-Mapped I/O</strong> — Efficient handling of large datasets</li>
              <li><strong>Single File Storage</strong> — No directory sprawl, easy backups</li>
            </ul>
          </div>
          <div className="col col--6">
            <div className={styles.architectureDiagram}>
              <div className={styles.diagramBox}>
                <div className={styles.diagramLabel}>Your Application</div>
                <div className={styles.diagramArrow}>↓</div>
                <div className={styles.diagramNeedle}>
                  <div className={styles.diagramLabel}>Needle</div>
                  <div className={styles.diagramComponents}>
                    <span>Collections</span>
                    <span>HNSW Index</span>
                    <span>Metadata</span>
                  </div>
                </div>
                <div className={styles.diagramArrow}>↓</div>
                <div className={styles.diagramFile}>
                  <span>📄</span> vectors.needle
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function QuickExample() {
  const rustCode = `use needle::{Database, Filter};
use serde_json::json;

fn main() -> needle::Result<()> {
    // Create or open a database
    let db = Database::open("vectors.needle")?;

    // Create a collection for 384-dimensional vectors
    db.create_collection("documents", 384)?;

    // Insert vectors with metadata
    let collection = db.collection("documents")?;
    collection.insert("doc1", &embedding, Some(json!({
        "title": "Hello World",
        "category": "greeting"
    })))?;

    // Search for similar vectors
    let results = collection.search(&query_vector, 10)?;

    for result in results {
        println!("ID: {}, Distance: {:.4}", result.id, result.distance);
    }

    // Persist to disk
    db.save()?;
    Ok(())
}`;

  const pythonCode = `import needle

# Create or open a database
db = needle.Database.open("vectors.needle")

# Create a collection for 384-dimensional vectors
db.create_collection("documents", 384)

# Insert vectors with metadata
collection = db.collection("documents")
collection.insert("doc1", embedding, {
    "title": "Hello World",
    "category": "greeting"
})

# Search for similar vectors
results = collection.search(query_vector, k=10)

for result in results:
    print(f"ID: {result.id}, Distance: {result.distance:.4f}")

# Persist to disk
db.save()`;

  const jsCode = `import { Database } from '@anthropic/needle';

// Create or open a database
const db = await Database.open("vectors.needle");

// Create a collection for 384-dimensional vectors
await db.createCollection("documents", 384);

// Insert vectors with metadata
const collection = await db.collection("documents");
await collection.insert("doc1", embedding, {
    title: "Hello World",
    category: "greeting"
});

// Search for similar vectors
const results = await collection.search(queryVector, 10);

for (const result of results) {
    console.log(\`ID: \${result.id}, Distance: \${result.distance.toFixed(4)}\`);
}

// Persist to disk
await db.save();`;

  return (
    <section className={styles.quickExample}>
      <div className="container">
        <Heading as="h2" className="text--center">
          Simple, Powerful API
        </Heading>
        <p className="text--center" style={{marginBottom: '2rem', color: 'var(--ifm-color-emphasis-600)'}}>
          Get productive in minutes. The same clean API across all languages.
        </p>
        <Tabs>
          <TabItem value="rust" label="Rust" default>
            <CodeBlock language="rust" title="main.rs" showLineNumbers>
              {rustCode}
            </CodeBlock>
          </TabItem>
          <TabItem value="python" label="Python">
            <CodeBlock language="python" title="main.py" showLineNumbers>
              {pythonCode}
            </CodeBlock>
          </TabItem>
          <TabItem value="javascript" label="JavaScript">
            <CodeBlock language="javascript" title="index.js" showLineNumbers>
              {jsCode}
            </CodeBlock>
          </TabItem>
        </Tabs>
      </div>
    </section>
  );
}

function UseCases() {
  const cases = [
    {
      title: 'Semantic Search',
      description: 'Build search engines that understand meaning, not just keywords. Perfect for documentation, knowledge bases, and content discovery.',
      link: '/docs/guides/semantic-search',
    },
    {
      title: 'RAG Applications',
      description: 'Power retrieval-augmented generation for LLMs. Store and retrieve relevant context to ground AI responses in your data.',
      link: '/docs/guides/rag',
    },
    {
      title: 'Recommendation Systems',
      description: 'Create personalized recommendations using vector similarity. Match users with products, content, or other users.',
      link: '/docs/concepts/vectors',
    },
    {
      title: 'Image & Audio Search',
      description: 'Search multimedia content using embeddings from CLIP, ImageBind, or other models. Find similar images, audio, or video.',
      link: '/docs/concepts/distance-functions',
    },
  ];

  return (
    <section className={styles.useCases}>
      <div className="container">
        <Heading as="h2" className="text--center">
          Built for Modern AI Applications
        </Heading>
        <p className="text--center">
          From semantic search to RAG pipelines, Needle powers the vector search layer of AI applications.
        </p>
        <div className="row">
          {cases.map((item, idx) => (
            <div key={idx} className="col col--3">
              <div className={styles.useCaseCard}>
                <Heading as="h3">{item.title}</Heading>
                <p>{item.description}</p>
                <Link to={item.link}>Learn more →</Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function LanguageBindings() {
  return (
    <section className={styles.bindings}>
      <div className="container">
        <Heading as="h2" className="text--center">
          Use From Any Language
        </Heading>
        <p className="text--center">
          Native bindings for your favorite languages, all powered by the same high-performance Rust core.
        </p>
        <div className={styles.bindingGrid}>
          <Link to="/docs/bindings/rust" className={styles.bindingCard}>
            <span className={styles.bindingIcon}>🦀</span>
            <span>Rust</span>
          </Link>
          <Link to="/docs/bindings/python" className={styles.bindingCard}>
            <span className={styles.bindingIcon}>🐍</span>
            <span>Python</span>
          </Link>
          <Link to="/docs/bindings/javascript" className={styles.bindingCard}>
            <span className={styles.bindingIcon}>🌐</span>
            <span>JavaScript</span>
          </Link>
          <Link to="/docs/bindings/swift-kotlin" className={styles.bindingCard}>
            <span className={styles.bindingIcon}>📱</span>
            <span>Swift/Kotlin</span>
          </Link>
        </div>
      </div>
    </section>
  );
}

function Benchmarks() {
  return (
    <section className={styles.benchmarks}>
      <div className="container">
        <Heading as="h2" className="text--center" style={{color: 'white', marginBottom: '2rem'}}>
          Performance That Scales
        </Heading>
        <div className="row">
          <div className="col col--3 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>&lt;10ms</span>
              <span className={styles.statLabel}>Search Latency</span>
            </div>
          </div>
          <div className="col col--3 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>99%+</span>
              <span className={styles.statLabel}>Recall@10</span>
            </div>
          </div>
          <div className="col col--3 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>15K</span>
              <span className={styles.statLabel}>Inserts/Second</span>
            </div>
          </div>
          <div className="col col--3 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>50M+</span>
              <span className={styles.statLabel}>Vectors Supported</span>
            </div>
          </div>
        </div>
        <p className="text--center" style={{marginTop: '2rem', opacity: 0.8}}>
          Benchmarks on 1M vectors, 384 dimensions. <Link to="/docs/benchmarks" style={{color: 'white', textDecoration: 'underline'}}>View full methodology →</Link>
        </p>
      </div>
    </section>
  );
}

function WhyNeedle() {
  return (
    <section className={styles.whyNeedle}>
      <div className="container">
        <Heading as="h2" className="text--center">
          Why Choose Needle?
        </Heading>
        <p className="text--center" style={{marginBottom: '2rem', color: 'var(--ifm-color-emphasis-600)'}}>
          The right tool depends on your use case. Here's when Needle shines.
        </p>
        <div className="row">
          <div className="col col--6">
            <div className={styles.whyCard}>
              <Heading as="h3">✅ Needle is great for</Heading>
              <ul>
                <li>Embedded applications (desktop, mobile, edge)</li>
                <li>Single-node deployments up to ~50M vectors</li>
                <li>Projects that value SQLite-like simplicity</li>
                <li>RAG applications and semantic search</li>
                <li>Prototyping with easy migration to production</li>
              </ul>
            </div>
          </div>
          <div className="col col--6">
            <div className={styles.whyCard}>
              <Heading as="h3">🔄 Consider alternatives when</Heading>
              <ul>
                <li>You need billion-scale vector search</li>
                <li>You need multi-region active-active replication</li>
                <li>You want a managed cloud service with SLAs</li>
                <li>You need real-time streaming at &gt;100K vectors/sec</li>
              </ul>
              <Link to="/docs/comparison" className={styles.compareLink}>
                See detailed comparison →
              </Link>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function BuiltBy() {
  return (
    <section className={styles.builtBy}>
      <div className="container text--center">
        <p className={styles.builtByText}>
          Built with ❤️ by <a href="https://anthropic.com" target="_blank" rel="noopener noreferrer">Anthropic</a>
        </p>
        <p className={styles.builtBySubtext}>
          Open source under the MIT License
        </p>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="SQLite for Vectors"
      description="An embedded vector database written in Rust. High-performance approximate nearest neighbor search with HNSW indexing, single-file storage, and zero configuration.">
      <HomepageHeader />
      <main>
        <HomepageFeatures />
        <ArchitectureDiagram />
        <QuickExample />
        <Benchmarks />
        <WhyNeedle />
        <UseCases />
        <LanguageBindings />
        <BuiltBy />
      </main>
    </Layout>
  );
}
