import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docsSidebar: [ // Renamed from tutorialSidebar
    {
      type: 'doc',
      id: 'intro', // Links to docs/intro.md (which has slug: /)
      label: 'Overview',
    },
    {
      type: 'category',
      label: 'User Guides',
      items: [
        'user_guides/placeholder_user_guide',
      ],
    },
    {
      type: 'category',
      label: 'Development Guides',
      items: [
        'development_guides/logging_standard',
        'development_guides/ai_context_management',
      ],
    },
    {
      type: 'category',
      label: 'Technical Deep Dives & Research',
      items: [
        'technical-deep-dives-research/omr-research',
        'technical-deep-dives-research/hmm-notes',
      ],
    },
    {
      type: 'category',
      label: 'Project Management & Planning',
      items: [
        'project-management-planning/project-progress',
        'project-management-planning/reorg-optimization-plan',
        'project-management-planning/performance-requirements',
      ],
    },
    {
      type: 'category',
      label: 'Proposals & Ideas',
      items: [
        'proposals-ideas/cascade-web-interface-proposal',
      ],
    },
    {
      type: 'category',
      label: 'Protocols',
      items: [
        'protocols/agentic-automation-protocol',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/index',
      ],
    },
  ],
};

export default sidebars;

