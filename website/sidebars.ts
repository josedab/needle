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
      id: 'comparison',
      label: 'Comparison',
    },
    {
      type: 'doc',
      id: 'faq',
      label: 'FAQ',
    },
  ],
};

export default sidebars;
