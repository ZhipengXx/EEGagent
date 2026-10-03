# EEGagent visual system

The README uses an editorial scientific style: warm paper, ink-blue typography, restrained copper and teal accents, thin rules and generous margins. Brain signals, cortical folds, representations and image-gallery glyphs tie the illustrations to brain decoding.

PNG illustration exports give predictable GitHub rendering. SVG sources retain selectable text, editable vectors and accessible title/description metadata. Each illustration PNG is 2160 pixels wide. The separate platform screenshots retain their original viewport sizes.

| Illustration | PNG export | Editable source |
| --- | --- | --- |
| Project identity and retrieval concept | [00-brain-decoding-hero.png](00-brain-decoding-hero.png) | [00-brain-decoding-hero.svg](00-brain-decoding-hero.svg) |
| Two research workflows | [01-system-overview.png](01-system-overview.png) | [01-system-overview.svg](01-system-overview.svg) |
| EEG role collaboration | [02-eeg-agent-collaboration.png](02-eeg-agent-collaboration.png) | [02-eeg-agent-collaboration.svg](02-eeg-agent-collaboration.svg) |
| fMRI screening loop | [03-fmri-screening-loop.png](03-fmri-screening-loop.png) | [03-fmri-screening-loop.svg](03-fmri-screening-loop.svg) |
| Planning over 2–4 next-action options | [04-planning-options.png](04-planning-options.png) | [04-planning-options.svg](04-planning-options.svg) |

## Recorded platform screenshots

[`ui-workspace/`](ui-workspace/) contains actual rendered workbench captures committed with `14ec3e6` on 2026-10-01. Their records and values are synthetic DEMO data. They show that recorded UI revision, rather than the current interface. See the [platform gallery](../platform.md) for captions, provenance and replacement guidance.

`ui-workspace/ui-workspace-1440-en.png` is an English text-localization edit of the recorded screenshot; it is not a new browser capture. The original image is retained.

## README navigation

[`navigation/`](navigation/) contains five SVG button assets in the same paper / ink / teal palette. Each README button is a separate image link, so it remains clickable on GitHub without custom CSS or JavaScript.

The illustration renderer below does not capture or refresh application screenshots.

## Palette and typography

| Token | Value | Use |
| --- | --- | --- |
| Paper | `#F7F4EE` | Canvas |
| White | `#FFFDFA` | Study panels |
| Ink | `#203744` | Headings and role names |
| Muted | `#5C6C70` | Descriptions |
| Rule | `#CCCFC7` | Dividers and outlines |
| Teal | `#35756E` | Runtime responsibilities, signal traces and feedback |
| Copper | `#AE663B` | Study numbering and representative forward paths |

DejaVu Serif sets the project wordmark and editorial titles. DejaVu Sans sets body text and role names; DejaVu Sans Mono sets study labels. These fonts are available under an open font license. Review text widths after any font substitution.

Dark bullet labels identify the eight EEG LLM roles. Runtime operations are labeled in teal. Dashed paths indicate conditional handoffs, feedback, repair or scoped-memory retrieval. All role handoffs are mediated by campaign records; the four visual stages do not impose a fixed runtime sequence.

## Re-render

With Python 3, Inkscape and the DejaVu fonts installed, run from the repository root:

```bash
python docs/figures/render_figures.py
```

[render_figures.py](render_figures.py) uses the Python standard library and Inkscape. It regenerates the four original illustration SVGs and PNG exports without a model, API, training data or additional Python packages. The SVGs can also be edited directly in Inkscape or Figma; rerunning the script replaces those manual edits.

The planning comparison figure uses the same drawing primitives and can be regenerated separately:

```bash
python docs/figures/render_planning_options.py
```

Its four actions and highlighted selection are illustrative. One planner response contains the alternatives and choice; the runtime dispatches the selected action after validation.

To export a manually edited SVG:

```bash
inkscape docs/figures/02-eeg-agent-collaboration.svg \
  --export-type=png --export-width=2160 \
  --export-filename=docs/figures/02-eeg-agent-collaboration.png
```

Brain shapes, activity colors, signal traces, representation tiles and gallery glyphs are schematic. They are not measured results, anatomical localization maps, synthetic benchmark curves or UI screenshots. Match workflow changes to the [architecture map](../architecture.md) before changing role labels or arrows.
