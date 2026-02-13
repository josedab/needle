import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Needle',
  tagline: 'SQLite for Vectors — An embedded vector database written in Rust',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://needle.dev',
  baseUrl: '/',

  organizationName: 'anthropics',
  projectName: 'needle',

  onBrokenLinks: 'throw',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/anthropics/needle/tree/main/website/',
          showLastUpdateTime: true,
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl: 'https://github.com/anthropics/needle/tree/main/website/',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        hashed: true,
        language: ["en"],
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
      },
    ],
  ],

  themeConfig: {
    image: 'img/needle-social-card.svg',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    announcementBar: {
      id: 'star_us',
      content: '⭐ If you like Needle, give it a star on <a target="_blank" rel="noopener noreferrer" href="https://github.com/anthropics/needle">GitHub</a>!',
      backgroundColor: '#6366f1',
      textColor: '#ffffff',
      isCloseable: true,
    },
    navbar: {
      title: 'Needle',
      logo: {
        alt: 'Needle Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Docs',
        },
        {
          to: '/docs/api-reference',
          label: 'API',
          position: 'left',
        },
        {
          to: '/blog',
          label: 'Blog',
          position: 'left',
        },
        {
          href: 'pathname:///playground',
          label: 'Playground',
          position: 'left',
        },
        {
          href: 'https://docs.rs/needle',
          label: 'Rust Docs',
          position: 'right',
        },
        {
          href: 'https://github.com/anthropics/needle',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Learn',
          items: [
            {
              label: 'Getting Started',
              to: '/docs/getting-started',
            },
            {
              label: 'Core Concepts',
              to: '/docs/concepts/vectors',
            },
            {
              label: 'API Reference',
              to: '/docs/api-reference',
            },
            {
              label: 'Architecture',
              to: '/docs/architecture',
            },
          ],
        },
        {
          title: 'Guides',
          items: [
            {
              label: 'Semantic Search',
              to: '/docs/guides/semantic-search',
            },
            {
              label: 'RAG Applications',
              to: '/docs/guides/rag',
            },
            {
              label: 'Production Deployment',
              to: '/docs/guides/production',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/anthropics/needle',
            },
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/anthropics/needle/discussions',
            },
            {
              label: 'Discord',
              href: 'https://discord.gg/anthropic',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'Rust Docs',
              href: 'https://docs.rs/needle',
            },
            {
              label: 'Crates.io',
              href: 'https://crates.io/crates/needle',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Anthropic. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml', 'bash', 'python', 'json'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
