import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';
import Heading from '@theme/Heading';
import CodeBlock from '@theme/CodeBlock';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.installCommand}>
          <code>cargo add needle</code>
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
        </div>
      </div>
    </header>
  );
}

function QuickExample() {
  const rustCode = `use needle::{Database, DistanceFunction};

// Create or open a database
let db = Database::open("vectors.needle")?;

// Create a collection for 384-dimensional vectors
db.create_collection("documents", 384, DistanceFunction::Cosine)?;

// Insert vectors with metadata
let collection = db.collection("documents")?;
collection.insert("doc1", &embedding, json!({"title": "Hello World"}))?;

// Search for similar vectors
let results = collection.search(&query_vector, 10, None)?;`;

  return (
    <section className={styles.quickExample}>
      <div className="container">
        <div className="row">
          <div className="col col--6">
            <Heading as="h2">Simple, Powerful API</Heading>
            <p>
              Needle provides a clean, intuitive API that lets you build vector
              search applications in minutes. No servers to manage, no complex
              configuration—just add the dependency and start building.
            </p>
            <ul className={styles.featureList}>
              <li>Single-file database—easy to backup and distribute</li>
              <li>Sub-10ms search latency with HNSW indexing</li>
              <li>MongoDB-style metadata filtering</li>
              <li>Zero configuration required</li>
            </ul>
          </div>
          <div className="col col--6">
            <CodeBlock language="rust" title="main.rs">
              {rustCode}
            </CodeBlock>
          </div>
        </div>
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
        <div className="row">
          <div className="col col--4 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>&lt;10ms</span>
              <span className={styles.statLabel}>Search Latency</span>
            </div>
          </div>
          <div className="col col--4 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>99%+</span>
              <span className={styles.statLabel}>Recall@10</span>
            </div>
          </div>
          <div className="col col--4 text--center">
            <div className={styles.stat}>
              <span className={styles.statNumber}>1M+</span>
              <span className={styles.statLabel}>Vectors/Second</span>
            </div>
          </div>
        </div>
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
        <QuickExample />
        <Benchmarks />
        <UseCases />
        <LanguageBindings />
      </main>
    </Layout>
  );
}
