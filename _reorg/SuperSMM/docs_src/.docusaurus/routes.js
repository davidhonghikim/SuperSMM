import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/SuperSMM/blog',
    component: ComponentCreator('/SuperSMM/blog', 'bc3'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/archive',
    component: ComponentCreator('/SuperSMM/blog/archive', '326'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/authors',
    component: ComponentCreator('/SuperSMM/blog/authors', '494'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/authors/all-sebastien-lorber-articles',
    component: ComponentCreator('/SuperSMM/blog/authors/all-sebastien-lorber-articles', 'fc8'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/authors/yangshun',
    component: ComponentCreator('/SuperSMM/blog/authors/yangshun', 'e3c'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/first-blog-post',
    component: ComponentCreator('/SuperSMM/blog/first-blog-post', '680'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/long-blog-post',
    component: ComponentCreator('/SuperSMM/blog/long-blog-post', '906'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/mdx-blog-post',
    component: ComponentCreator('/SuperSMM/blog/mdx-blog-post', '1e6'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/tags',
    component: ComponentCreator('/SuperSMM/blog/tags', 'e8e'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/tags/docusaurus',
    component: ComponentCreator('/SuperSMM/blog/tags/docusaurus', 'a1b'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/tags/facebook',
    component: ComponentCreator('/SuperSMM/blog/tags/facebook', 'ba0'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/tags/hello',
    component: ComponentCreator('/SuperSMM/blog/tags/hello', 'b65'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/tags/hola',
    component: ComponentCreator('/SuperSMM/blog/tags/hola', 'b98'),
    exact: true
  },
  {
    path: '/SuperSMM/blog/welcome',
    component: ComponentCreator('/SuperSMM/blog/welcome', '7b3'),
    exact: true
  },
  {
    path: '/SuperSMM/markdown-page',
    component: ComponentCreator('/SuperSMM/markdown-page', 'b9d'),
    exact: true
  },
  {
    path: '/SuperSMM/docs',
    component: ComponentCreator('/SuperSMM/docs', 'd26'),
    routes: [
      {
        path: '/SuperSMM/docs',
        component: ComponentCreator('/SuperSMM/docs', '627'),
        routes: [
          {
            path: '/SuperSMM/docs',
            component: ComponentCreator('/SuperSMM/docs', '491'),
            routes: [
              {
                path: '/SuperSMM/docs/',
                component: ComponentCreator('/SuperSMM/docs/', '370'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/api/core',
                component: ComponentCreator('/SuperSMM/docs/api/core', '794'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development_guides/ai_context_management',
                component: ComponentCreator('/SuperSMM/docs/development_guides/ai_context_management', '915'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/development_guides/architecture',
                component: ComponentCreator('/SuperSMM/docs/development_guides/architecture', '59c'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development_guides/debugging',
                component: ComponentCreator('/SuperSMM/docs/development_guides/debugging', 'e58'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development_guides/logging_standard',
                component: ComponentCreator('/SuperSMM/docs/development_guides/logging_standard', '5a1'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/development_guides/system_features',
                component: ComponentCreator('/SuperSMM/docs/development_guides/system_features', 'bb2'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development/coverage_improvement_plan',
                component: ComponentCreator('/SuperSMM/docs/development/coverage_improvement_plan', '3d2'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development/linting_fixes',
                component: ComponentCreator('/SuperSMM/docs/development/linting_fixes', '74b'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development/refactoring_plan',
                component: ComponentCreator('/SuperSMM/docs/development/refactoring_plan', 'd02'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development/refactoring_tasks',
                component: ComponentCreator('/SuperSMM/docs/development/refactoring_tasks', '606'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/development/scripts',
                component: ComponentCreator('/SuperSMM/docs/development/scripts', 'f9f'),
                exact: true
              },
              {
                path: '/SuperSMM/docs/project-management-planning/performance-requirements',
                component: ComponentCreator('/SuperSMM/docs/project-management-planning/performance-requirements', 'd80'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/project-management-planning/project-progress',
                component: ComponentCreator('/SuperSMM/docs/project-management-planning/project-progress', '41d'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/project-management-planning/reorg-optimization-plan',
                component: ComponentCreator('/SuperSMM/docs/project-management-planning/reorg-optimization-plan', '809'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/proposals-ideas/cascade-web-interface-proposal',
                component: ComponentCreator('/SuperSMM/docs/proposals-ideas/cascade-web-interface-proposal', '44f'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/protocols/agentic-automation-protocol',
                component: ComponentCreator('/SuperSMM/docs/protocols/agentic-automation-protocol', '121'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/reference',
                component: ComponentCreator('/SuperSMM/docs/reference', 'ab5'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/technical-deep-dives-research/hmm-notes',
                component: ComponentCreator('/SuperSMM/docs/technical-deep-dives-research/hmm-notes', '51e'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/technical-deep-dives-research/omr-research',
                component: ComponentCreator('/SuperSMM/docs/technical-deep-dives-research/omr-research', '536'),
                exact: true,
                sidebar: "docsSidebar"
              },
              {
                path: '/SuperSMM/docs/user_guides/placeholder_user_guide',
                component: ComponentCreator('/SuperSMM/docs/user_guides/placeholder_user_guide', '557'),
                exact: true,
                sidebar: "docsSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/SuperSMM/',
    component: ComponentCreator('/SuperSMM/', '9a5'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
