# README figures

The project README uses PNG exports for predictable rendering. Each figure also has an editable SVG source with selectable text, vector connectors and accessible title/description metadata.

| Figure | PNG export | Editable source |
| --- | --- | --- |
| System overview | [01-system-overview.png](01-system-overview.png) | [01-system-overview.svg](01-system-overview.svg) |
| EEG collaboration | [02-eeg-agent-collaboration.png](02-eeg-agent-collaboration.png) | [02-eeg-agent-collaboration.svg](02-eeg-agent-collaboration.svg) |
| fMRI screening | [03-fmri-screening-loop.png](03-fmri-screening-loop.png) | [03-fmri-screening-loop.svg](03-fmri-screening-loop.svg) |

Use Inkscape, Figma or another SVG editor to change labels and layout. SVG files are the canonical editable assets. Use DejaVu Sans, or review text widths if substituting another font.

Re-export an edited source with Inkscape, for example from the repository root:

```bash
inkscape docs/figures/02-eeg-agent-collaboration.svg \
  --export-type=png --export-width=1800 \
  --export-filename=docs/figures/02-eeg-agent-collaboration.png
```

Purple marks LLM roles, teal marks runtime/tools, and slate marks artifacts. Dashed connectors indicate conditional or memory handoffs. Match the arrows to the [architecture source map](../architecture.md) when changing orchestration.

These vector diagrams contain no generated scientific results, synthetic benchmark curves or UI screenshots.
