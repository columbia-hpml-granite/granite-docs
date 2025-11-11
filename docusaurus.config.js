const config = {
  title: "Granite Docs",
  tagline: "Columbia University High Performance Machine Learning Projects in Collaboration with IBM Research",
  favicon: "image/favicon.ico",

  url: "https://columbia-hpml-granite.github.io",
  baseUrl: "/docs/",

  organizationName: "columbia-hpml-granite",
  projectName: "granite-docs",

  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  presets: [
    [
      "classic",
      ({
        docs: {
          routeBasePath: "/", // This makes docs the root
          sidebarPath: require.resolve("./sidebars.js"),
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  // Conditionally add mermaid theme if it's installed
  themes: (() => {
    try {
      require.resolve('@docusaurus/theme-mermaid');
      return ['@docusaurus/theme-mermaid'];
    } catch {
      console.warn('Warning: @docusaurus/theme-mermaid is not installed. Mermaid diagrams will not be rendered.');
      return [];
    }
  })(),

  markdown: {
    mermaid: (() => {
      try {
        require.resolve('@docusaurus/theme-mermaid');
        return true;
      } catch {
        return false;
      }
    })(),
  },

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: "Home",
        items: [
          {
            type: "dropdown",
            label: "Weekly Updates",
            position: "left",
            items: [
              {
                label: "Week 1",
                to: "/weekly/week1",
              },
              {
                label: "Week 2",
                to: "/weekly/week2",
              },
              {
                label: "Week 3",
                to: "/weekly/week3",
              },
              {
                label: "Week 4",
                to: "/weekly/week4",
              },
              {
                label: "Week 5",
                to: "/weekly/week5",
              },
              {
                label: "Week 6",
                to: "/weekly/week6",
              },
              {
                label: "Week 7",
                to: "/weekly/week7",
              },
              {
                label: "Week 8",
                to: "/weekly/week8",
              },
            ],
          },
          {
            href: "https://github.com/your-org/granite-docs",
            label: "GitHub",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Documentation",
            items: [
              {
                label: "Home",
                to: "/",
              },
            ],
          },
          {
            title: "More",
            items: [
              {
                label: "GitHub",
                href: "https://github.com/your-org/granite-docs",
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Columbia University & IBM Research. Built with Docusaurus.`,
      },
      docs: {
        sidebar: {
          hideable: true,
          autoCollapseCategories: false,
        },
      },
      prism: (() => {
        try {
          let lightTheme, darkTheme;

          try {
            const {themes} = require('prism-react-renderer');
            lightTheme = themes.github;
            darkTheme = themes.dracula;
          } catch {
            try {
              lightTheme = require('prism-react-renderer/themes/github');
              darkTheme = require('prism-react-renderer/themes/dracula');
            } catch {
              lightTheme = undefined;
              darkTheme = undefined;
            }
          }

          return {
            theme: lightTheme,
            darkTheme: darkTheme,
            additionalLanguages: ['bash', 'python', 'javascript', 'typescript', 'json', 'yaml'],
          };
        } catch (error) {
          console.warn('Warning: Could not load prism-react-renderer. Using default code highlighting.');
          return {
            additionalLanguages: ['bash', 'python', 'javascript', 'typescript', 'json', 'yaml'],
          };
        }
      })(),
    }),
};

module.exports = config;
