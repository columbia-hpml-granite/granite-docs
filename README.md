# Granite Docs

![Deploy Status](https://github.com/your-org/granite-docs/actions/workflows/deploy.yml/badge.svg)

Technical documentation for Columbia University High Performance Machine Learning Projects in Collaboration with IBM Research.

## Project Structure

This repository uses Docusaurus to serve technical documentation with a markdown-based homepage.

```
granite-docs/
├── docs/                    # Documentation files
│   ├── intro.md            # Homepage (markdown)
│   └── weekly/             # Weekly update reports
│       ├── _template.md    # Template for weekly updates
│       ├── week1.md        # Week 1 update
│       ├── week2.md        # Week 2 update
│       └── ...             # Weeks 3-8
├── src/
│   └── css/
│       └── custom.css      # Custom styling
├── static/
│   └── img/                # Static images and assets
├── docusaurus.config.js    # Docusaurus configuration
├── sidebars.js             # Sidebar configuration
└── package.json            # Dependencies
```

## Getting Started

### Installation

```bash
npm install
```

### Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

### Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Weekly Updates

Weekly updates follow a structured template located at `docs/weekly/_template.md`. Each week's update should include:

1. **Project Overview** - Team and objective information
2. **Overall Progress Summary** - Milestones and deliverables
3. **Tasks Completed This Week** - Detailed task breakdown
4. **Plans for Next Week** - Upcoming tasks and assignments
5. **Challenges / Blockers** - Issues and proposed solutions
6. **Individual Contributions** - Team member contributions
7. **Feedback / Requests from Supervisors** - Questions and guidance needed
8. **Appendix** - Supporting materials and references

## Configuration

### Updating Project Information

Edit `docusaurus.config.js` to update:
- Site title and tagline
- GitHub organization and repository URLs
- Navigation items
- Footer content

### Updating Homepage

Edit `docs/intro.md` to update the main landing page content.

### Adding Images

Images are organized in `static/img/` directory:

- `architecture/` - Architecture diagrams
- `results/` - Performance charts and graphs
- `diagrams/` - General diagrams
- `screenshots/` - Screenshots
- `weekly/weekN/` - Weekly update specific images

**Usage in markdown:**
```markdown
![Description](/img/weekly/week2/chart-name.png)
```

See `static/img/README.md` for detailed guidelines.

## Deployment

This site automatically deploys to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

### Automatic Deployment (Recommended)

Simply push to the `main` branch:

```bash
git add .
git commit -m "Your commit message"
git push origin main
```

The site will automatically build and deploy. Check the **Actions** tab to monitor deployment progress.

### Manual Deployment

You can also trigger deployment manually:

1. Go to the **Actions** tab in your repository
2. Select "Deploy to GitHub Pages"
3. Click **Run workflow**

### Setup Requirements

Before first deployment:

1. **Enable GitHub Pages:**
   - Go to **Settings** → **Pages**
   - Set source to **GitHub Actions**

2. **Update Configuration:**
   - Edit `docusaurus.config.js`
   - Update `url`, `baseUrl`, `organizationName`, and `projectName`


## Documentation

For more information about Docusaurus, visit: https://docusaurus.io/

## License

Copyright © 2025 Columbia University & IBM Research
