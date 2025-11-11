# Deployment Guide

This document explains how to deploy the Granite Docs site to GitHub Pages.

## Automatic Deployment

The site automatically deploys to GitHub Pages when changes are pushed to the `main` branch.

### GitHub Actions Workflows

We have two workflows configured:

1. **Deploy to GitHub Pages** (`.github/workflows/deploy.yml`)
   - Triggers on push to `main` branch
   - Builds the Docusaurus site
   - Deploys to GitHub Pages
   - Can also be triggered manually from the Actions tab

2. **Test Build** (`.github/workflows/test.yml`)
   - Triggers on pull requests to `main`
   - Tests that the site builds successfully
   - Does not deploy

## Initial Setup

### 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click **Settings** → **Pages**
3. Under **Source**, select:
   - **Source:** GitHub Actions
4. Save the settings

### 2. Update Repository URLs

Edit `docusaurus.config.js` and update these values:

```javascript
url: "https://your-org.github.io",
baseUrl: "/granite-docs/",  // or "/" if using custom domain
organizationName: "your-org",
projectName: "granite-docs",
```

### 3. Push to Main Branch

```bash
git add .
git commit -m "Configure GitHub Actions deployment"
git push origin main
```

The deployment will start automatically. Check the **Actions** tab to monitor progress.

## Manual Deployment

You can also deploy manually:

### Option 1: Using GitHub Actions

1. Go to **Actions** tab in your repository
2. Select "Deploy to GitHub Pages" workflow
3. Click **Run workflow**
4. Select the `main` branch
5. Click **Run workflow**

### Option 2: Using npm script (Legacy)

```bash
npm run build
npm run deploy
```

**Note:** The npm deploy script requires additional configuration in `docusaurus.config.js` for the legacy deployment method.

## Deployment URL

After deployment, your site will be available at:
- **Default:** `https://your-org.github.io/granite-docs/`
- **Custom Domain:** Configure in GitHub Pages settings

## Troubleshooting

### Build Fails

1. Check the **Actions** tab for error logs
2. Test locally: `npm run build`
3. Ensure all dependencies are in `package.json`
4. Check for broken links or missing files

### Site Not Updating

1. Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)
2. Check the Actions tab to verify deployment succeeded
3. Wait a few minutes for GitHub Pages to propagate changes

### 404 Errors

1. Verify `baseUrl` in `docusaurus.config.js` matches your repository name
2. For organization pages: `baseUrl: "/repo-name/"`
3. For user pages: `baseUrl: "/"`

### Permission Errors

1. Go to **Settings** → **Actions** → **General**
2. Under "Workflow permissions", select:
   - ✅ Read and write permissions
   - ✅ Allow GitHub Actions to create and approve pull requests
3. Save changes

## Environment Variables

If you need environment variables for deployment:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add repository secrets
3. Reference in workflow: `${{ secrets.SECRET_NAME }}`

## Custom Domain

To use a custom domain:

1. Add `CNAME` file to `static/` directory:
   ```
   docs.yourdomain.com
   ```

2. Update DNS records:
   - Add CNAME record pointing to `your-org.github.io`

3. Update `docusaurus.config.js`:
   ```javascript
   url: "https://docs.yourdomain.com",
   baseUrl: "/",
   ```

## Monitoring Deployments

- **Actions Tab:** View deployment history and logs
- **Environments:** Check deployment status under **Settings** → **Environments**
- **Status Badge:** Add to README.md:
  ```markdown
  ![Deploy Status](https://github.com/your-org/granite-docs/actions/workflows/deploy.yml/badge.svg)
  ```

## Rollback

To rollback to a previous version:

1. Go to **Actions** tab
2. Find the successful deployment you want to restore
3. Click **Re-run jobs**

Or revert the commit:
```bash
git revert <commit-hash>
git push origin main
```

## Local Testing

Before deploying, always test locally:

```bash
npm install
npm start          # Development server
npm run build      # Production build
npm run serve      # Test production build locally
```

## Additional Resources

- [Docusaurus Deployment Docs](https://docusaurus.io/docs/deployment)
- [GitHub Pages Docs](https://docs.github.com/en/pages)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
