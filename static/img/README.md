# Images Directory

This directory contains all images used in the documentation.

## Directory Structure

```
img/
├── architecture/       # Architecture diagrams and system designs
├── results/           # Performance charts, graphs, and experiment results
├── diagrams/          # General diagrams and flowcharts
├── screenshots/       # Screenshots from experiments or tools
└── weekly/            # Weekly update images
    ├── week1/        # Week 1 specific images
    ├── week2/        # Week 2 specific images
    ├── week3/        # Week 3 specific images
    ├── week4/        # Week 4 specific images
    ├── week5/        # Week 5 specific images
    ├── week6/        # Week 6 specific images
    ├── week7/        # Week 7 specific images
    └── week8/        # Week 8 specific images
```

## How to Use Images in Markdown

### Basic Image Syntax

```markdown
![Alt text](/img/category/image-name.png)
```

### Examples

**Week 2 Results:**
```markdown
![Encoder-Decoder Breakdown](/img/weekly/week2/encoder-decoder-breakdown.png)
```

**Architecture Diagram:**
```markdown
![Granite Speech Architecture](/img/architecture/granite-speech-architecture.png)
```

**With Caption:**
```markdown
![Performance Chart](/img/results/latency-comparison.png)
*Figure 1: Latency comparison across different audio lengths*
```

### Image Best Practices

1. **File Naming:** Use lowercase with hyphens (e.g., `encoder-latency-chart.png`)
2. **Formats:** PNG for screenshots/charts, SVG for diagrams when possible
3. **Size:** Optimize images before uploading (keep under 1MB when possible)
4. **Alt Text:** Always provide descriptive alt text for accessibility

## Common Use Cases

### Weekly Reports

Place images in the corresponding week folder:
- `/img/weekly/week1/` - Week 1 images
- `/img/weekly/week2/` - Week 2 images (profiling results, charts)
- etc.

### Architecture Documentation

Place in `/img/architecture/`:
- System diagrams
- Model architecture visualizations
- Integration flowcharts

### Performance Results

Place in `/img/results/`:
- Benchmark charts
- Performance graphs
- Comparison tables (as images)

### Screenshots

Place in `/img/screenshots/`:
- Tool/UI screenshots
- Code output
- Terminal captures
