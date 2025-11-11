# Image Assets

This directory contains site-level image assets.

## Files

- **favicon.ico** - Site favicon (16x16, 32x32, or 48x48 pixels)
  - Place your favicon.ico file here
  - Referenced in `docusaurus.config.js` as `favicon: "image/favicon.ico"`

## Creating a Favicon

### Online Tools
- [Favicon.io](https://favicon.io/) - Generate from text, image, or emoji
- [RealFaviconGenerator](https://realfavicongenerator.net/) - Comprehensive favicon generator

### Requirements
- Format: `.ico` file
- Recommended sizes: 16x16, 32x32, 48x48 pixels
- Can include multiple sizes in one .ico file

### Quick Generation

If you have a PNG/JPG image, convert it to favicon:

```bash
# Using ImageMagick
convert logo.png -resize 32x32 favicon.ico

# Or use online converters
```

## Note

The `static/img/` directory is for documentation images (charts, diagrams, screenshots).
The `static/image/` directory is specifically for site assets like favicon.
