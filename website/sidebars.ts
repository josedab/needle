import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [
    {
      type: 'doc',
      id: 'intro',
      label: 'Introduction',
    },
    {
      type: 'doc',
      id: 'getting-started',
      label: 'Getting Started',
    },
    {
      type: 'category',
      label: 'Core Concepts',
      collapsed: false,
      items: [
        'concepts/vectors',
        'concepts/collections',
        'concepts/hnsw-index',
        'concepts/distance-functions',
        'concepts/metadata-filtering',
        'concepts/aliasing',
        'concepts/ttl',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        'guides/semantic-search',
        'guides/rag',
        'guides/hybrid-search',
        'guides/quantization',
        'guides/production',
        'guides/index-selection',
        'guides/docker-quickstart',
        'guides/production-checklist',
      ],
    },
    {
      type: 'category',
      label: 'Language Bindings',
      collapsed: true,
      items: [
        'bindings/rust',
        'bindings/python',
        'bindings/javascript',
        'bindings/swift-kotlin',
      ],
    },
    {
      type: 'category',
      label: 'Advanced',
      collapsed: true,
      items: [
        'advanced/http-server',
        'advanced/cli',
        'advanced/encryption',
        'advanced/sharding',
        'advanced/replication',
        'advanced/operations',
        'advanced/deployment',
        'advanced/distributed',
      ],
    },
    {
      type: 'doc',
      id: 'api-reference',
      label: 'API Reference',
    },
    {
      type: 'category',
      label: 'Configuration',
      collapsed: true,
      items: [
        'configuration/hnsw-tuning',
        'configuration/feature-flags',
      ],
    },
    {
      type: 'doc',
      id: 'architecture',
      label: 'Architecture',
    },
    {
      type: 'doc',
      id: 'benchmarks',
      label: 'Performance & Benchmarks',
    },
    {
      type: 'doc',
      id: 'comparison',
      label: 'Comparison',
    },
    {
      type: 'doc',
      id: 'api-stability',
      label: 'API Stability',
    },
    {
      type: 'doc',
      id: 'migration',
      label: 'Migration Guide',
    },
    {
      type: 'doc',
      id: 'troubleshooting',
      label: 'Troubleshooting',
    },
    {
      type: 'doc',
      id: 'faq',
      label: 'FAQ',
    },
    {
      type: 'category',
      label: 'Community',
      collapsed: true,
      items: [
        'contributing',
        'changelog',
      ],
    },
  ],
};

export default sidebars;
