# Agentic workspace API

Inspection baseline: `c26f68f`. These GET endpoints are read-only projections. They do not start workers, call models, train, recompute metrics, or change budget files.

Demo data lives under `runs/eeg_research_ui_demo/` and is created only by `POST /api/agentic_demo`. Append `demo=1` to GET query strings to read that root. It is isolated from `runs/eeg_research_v18/`.

## Null policy

| Token | Meaning |
| --- | --- |
| `null` / omitted | Field was not recorded |
| `unknown` | Legacy record without a start event; do not infer that a role is still running |
| `missing` (metrics) | Epoch exists but this value was not written |
| `ok` + `0` | Zero is a real number |
| `non_finite` | NaN/Inf; do not draw an SVG point |
| `unlinked` | Two records share no identity; timestamps are not used to invent edges |

## Endpoints

- `GET /api/agentic_status` — campaign summaries only (no source, no full history)
- `GET /api/agentic_status?campaign=<id>` — one campaign detail, source preview truncated at 6000 characters
- `GET /api/agentic_timeline?campaign=<id>&cursor=0&limit=20` — `events`, `next_cursor`, `has_more`, `activity`
- `GET /api/agentic_job?campaign=<id>&job=<id>` — history from `jobs/<job_id>/history.jsonl`
- `GET /api/agentic_source?campaign=<id>&candidate=<id>` — full source; `preview=1` keeps the truncation
- `POST /api/agentic_control` — existing pause / resume / stop
- `POST /api/agentic_demo` — write the explicit demo fixture

Campaign, job, and candidate names are `Path.name` only. `..`, absolute paths, and cross-campaign jobs return `campaign_missing` / `job_missing`.

## Compatibility

`campaign_view()` still returns `decisions[-8:]` for older callers. The workbench timeline is the paginated source for more than eight events. `best_full` now ranks `0` above `-2`; `beats_control` is false when the best delta is not positive.
