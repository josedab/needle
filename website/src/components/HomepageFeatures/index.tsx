import type {ReactNode} from 'react';
import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  icon: string;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'SQLite for Vectors',
    icon: '🗄️',
    description: (
      <>
        Single-file database that's easy to backup, distribute, and embed.
        No servers, no complex setup—just add the dependency and start building.
      </>
    ),
  },
  {
    title: 'Blazing Fast Search',
    icon: '⚡',
    description: (
      <>
        HNSW indexing delivers sub-10ms approximate nearest neighbor search
        with 99%+ recall. SIMD-optimized distance functions for maximum throughput.
      </>
    ),
  },
  {
    title: 'Rich Query Capabilities',
    icon: '🔍',
    description: (
      <>
        MongoDB-style metadata filtering, hybrid search with BM25+RRF fusion,
        and multiple distance functions including Cosine, Euclidean, and Dot Product.
      </>
    ),
  },
  {
    title: 'Memory Efficient',
    icon: '💾',
    description: (
      <>
        Multiple quantization strategies (Scalar, Product, Binary) reduce memory
        usage by up to 32x while maintaining search quality.
      </>
    ),
  },
  {
    title: 'Production Ready',
    icon: '🚀',
    description: (
      <>
        Built-in HTTP server, Prometheus metrics, auto-tuning, and comprehensive
        error handling. Ready for production from day one.
      </>
    ),
  },
  {
    title: 'Written in Rust',
    icon: '🦀',
    description: (
      <>
        Memory-safe, thread-safe, and fast. Native bindings for Python,
        JavaScript/WASM, Swift, and Kotlin—all from one codebase.
      </>
    ),
  },
];

function Feature({title, icon, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className={styles.featureCard}>
        <div className={styles.featureIcon}>{icon}</div>
        <Heading as="h3" className={styles.featureTitle}>{title}</Heading>
        <p className={styles.featureDescription}>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
