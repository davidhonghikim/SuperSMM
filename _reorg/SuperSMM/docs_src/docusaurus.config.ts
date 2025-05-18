import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'SuperSMM Documentation',
  tagline: 'Your guide to the Super Sheet Music Master platform',
  favicon: 'img/favicon.ico', // We can update this later

  // Set the production url of your site here
  url: 'https://davidhonghikim.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/SuperSMM/',

  // GitHub pages deployment config.
  organizationName: 'davidhonghikim', // Your GitHub org/user name.
  projectName: 'SuperSMM', // Your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  markdown: {
    mermaid: true,
  },

  themes: ['@docusaurus/theme-mermaid'], // Explicitly add the theme

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
          editUrl:
            'https://github.com/davidhonghikim/SuperSMM/tree/main/docs_src/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          editUrl:
            'https://github.com/davidhonghikim/SuperSMM/tree/main/docs_src/',
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg', // Consider creating a custom social card
    navbar: {
      title: 'SuperSMM',
      logo: {
        alt: 'SuperSMM Logo',
        src: 'img/logo.svg', // We can update this later
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar', // This will be the ID for the main docs sidebar
          position: 'left',
          label: 'Documentation', // Changed from Tutorial
        },
        // {to: '/blog', label: 'Blog', position: 'left'}, // Blog removed for now
        {
          href: 'https://github.com/davidhonghikim/SuperSMM',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Introduction',
              to: '/', // Default intro page
            },
          ],
        },
        {
          title: 'Community',
          items: [
            // Add relevant community links here if any, e.g., Discord, forums
            // For now, keeping it minimal
            {
              label: 'Stack Overflow (Docusaurus)',
              href: 'https://stackoverflow.com/questions/tagged/docusaurus',
            },
            {
              label: 'Discord (Docusaurus)',
              href: 'https://discordapp.com/invite/docusaurus',
            },
          ],
        },
        {
          title: 'More',
          items: [
            // { label: 'Blog', to: '/blog' }, // Blog removed for now
            {
              label: 'GitHub',
              href: 'https://github.com/davidhonghikim/SuperSMM',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} SuperSMM. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
