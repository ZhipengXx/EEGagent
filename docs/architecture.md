# Architecture and source map

Inspection baseline: [`f698e71bdfcedefd1e44229793553c123dbef777`](https://github.com/ZhipengXx/EEGagent/tree/f698e71bdfcedefd1e44229793553c123dbef777), committed on 2026-09-30. The diagrams summarize that implementation. Revisit this map when orchestration changes.

## Reading the figures

Purple marks LLM responsibilities, teal marks runtime code or tools, and slate marks artifacts. Dashed connectors indicate conditional or memory handoffs. Boxes combining an optional role or policy with deterministic code are labeled explicitly.

Arrows represent semantic handoffs mediated by the runtime. They do not imply direct role-to-role messages, separate services, mandatory invocation order, or parallel execution. The overview groups responsibilities; the EEG figure shows a representative candidate path, and the fMRI figure shows the check-and-observe loop.

## System overview

![System overview](figures/01-system-overview.png)

| Figure element | Implementation | Boundary |
| --- | --- | --- |
| Workbench and controls | [`fmri/workbench.py`](../react-agent/src/react_agent/fmri/workbench.py) and workflow CLIs | A shared interface does not merge workflow evidence or memory |
| fMRI loop | [`fmri/graph.py`](../react-agent/src/react_agent/fmri/graph.py), [`fmri/loop.py`](../react-agent/src/react_agent/fmri/loop.py) | Numeric-consistency screening |
| Image materialization | [`fmri/pipeline_graph.py`](../react-agent/src/react_agent/fmri/pipeline_graph.py), [`fmri/generation/`](../react-agent/src/react_agent/fmri/generation/) | External TRIBE resources; optional generation before screening |
| EEG campaign | [`agentic/loop.py`](../react-agent/src/react_agent/eeg_research/agentic/loop.py), [`agentic/worker.py`](../react-agent/src/react_agent/eeg_research/agentic/worker.py) | State-driven actions and service calls |
| ReAct entry | [`graph.py`](../react-agent/src/react_agent/graph.py) | Conversation graph |
| Studio adapters | [`langgraph.json`](../react-agent/langgraph.json), [`studio_graph.py`](../react-agent/src/react_agent/eeg_research/agentic/studio_graph.py) | EEG view wraps ticks as load → step → summarize |

## EEG role collaboration

![EEG collaboration](figures/02-eeg-agent-collaboration.png)

The planner receives current observations. `available_actions()` derives executable choices from state and budget. `tick()` reconciles work and routes a selected action through the appropriate service. Job settlement and analysis can occur without an extra planner decision.

| Responsibility | Source | Output / enforcement |
| --- | --- | --- |
| Planning | [`planner.py`](../react-agent/src/react_agent/eeg_research/agentic/planner.py), [`research_plan.py`](../react-agent/src/react_agent/eeg_research/agentic/research_plan.py) | Available actions and evidence-linked plan versions |
| Local method retrieval | [`knowledge/__init__.py`](../react-agent/src/react_agent/eeg_research/agentic/knowledge/__init__.py), [`loop.py`](../react-agent/src/react_agent/eeg_research/agentic/loop.py) `_retrieve_methods` | Local cards and optional librarian interpretation; no online search |
| Experiment design | [`loop.py`](../react-agent/src/react_agent/eeg_research/agentic/loop.py) `_design_experiment`, [`experiment_designer.txt`](../react-agent/src/react_agent/eeg_research/agentic/prompts/experiment_designer.txt) | Draft with control, intervention, capabilities and evidence |
| Runtime approval | [`experiment_gate.py`](../react-agent/src/react_agent/eeg_research/agentic/experiment_gate.py), [`run_context.py`](../react-agent/src/react_agent/eeg_research/agentic/run_context.py) | Context-bound approval and frozen run specifications |
| Candidate implementation and review | [`native_patch.py`](../react-agent/src/react_agent/eeg_research/agentic/native_patch.py), [`worker.py`](../react-agent/src/react_agent/eeg_research/agentic/worker.py) | Controlled edits, checks, verdicts and bounded repairs |
| Training | [`jobs.py`](../react-agent/src/react_agent/eeg_research/agentic/jobs.py), [`train_entry.py`](../react-agent/src/react_agent/eeg_training/train_entry.py) | Training/evaluation artifacts and checkpoints |
| Evidence settlement | [`loop.py`](../react-agent/src/react_agent/eeg_research/agentic/loop.py) `_record_job`, [`comparison.py`](../react-agent/src/react_agent/eeg_research/agentic/comparison.py) | Result binding, diagnostics and matched comparison |
| Result analysis | [`worker.py`](../react-agent/src/react_agent/eeg_research/agentic/worker.py) `analyze`, [`result_analyst.txt`](../react-agent/src/react_agent/eeg_research/agentic/prompts/result_analyst.txt) | Interpretation against the bound hypothesis |
| Result audit | [`loop.py`](../react-agent/src/react_agent/eeg_research/agentic/loop.py) `_audit_result`, [`result_auditor.txt`](../react-agent/src/react_agent/eeg_research/agentic/prompts/result_auditor.txt) | Claim draft and latest result passed to the LLM; runtime state hashing tracks freshness |
| Memory and evidence level | [`memory.py`](../react-agent/src/react_agent/eeg_research/agentic/memory.py), [`promotion.py`](../react-agent/src/react_agent/eeg_research/agentic/promotion.py) | Accepted conditional lessons and runtime promotion decisions |
| Pack reuse | [`export.py`](../react-agent/src/react_agent/eeg_research/agentic/export.py) | Hashes, source availability and evaluate-only reconstruction command |

All eight role prompts live in [`agentic/prompts/`](../react-agent/src/react_agent/eeg_research/agentic/prompts/). [`llm.py`](../react-agent/src/react_agent/eeg_research/agentic/llm.py) currently uses the DeepSeek fast profile for these calls; role names do not imply distinct model backbones. [`roles.py`](../react-agent/src/react_agent/eeg_research/agentic/roles.py), [`task_ledger.py`](../react-agent/src/react_agent/eeg_research/agentic/task_ledger.py), [`identity.py`](../react-agent/src/react_agent/eeg_research/agentic/identity.py) and [`artifacts.py`](../react-agent/src/react_agent/eeg_research/agentic/artifacts.py) maintain result envelopes, task/attempt identity and provenance.

The capability manifest declares hooks, but `available` is not a verification receipt: the current manifest initially marks verification as false. The approval gate checks registered availability; it does not require a non-null verification artifact for each hook. Unknown or disabled hooks can block approval. Candidate changes must respect the frozen contract and executable hooks.

The auditor prompt asks for an exact `ReportDraft` and `DependencyManifest`. The current `_audit_result` service sends a generated claim draft, its report hash and the latest settled result, rather than separately injecting both complete structured objects. Runtime `report_dependency_hash()` and `refresh_audit_freshness()` track campaign-state evidence changes. The figure describes this narrower implemented path. An LLM audit verdict does not establish that every underlying file was independently re-read.

A pilot need not lead to a full run. Full runs need not improve performance. Confirmation and stopping depend on evidence, budget and policy. Valid negative findings can be retained and audited.

## fMRI screening collaboration

![fMRI screening](figures/03-fmri-screening-loop.png)

| Responsibility | Source | Output / enforcement |
| --- | --- | --- |
| Ingest and required checks | [`graph.py`](../react-agent/src/react_agent/fmri/graph.py), [`loop.py`](../react-agent/src/react_agent/fmri/loop.py) | Mandatory-check results before optional selection |
| Planner / replanner | [`planning/service.py`](../react-agent/src/react_agent/fmri/planning/service.py) | Structured plan proposals |
| Action validation | [`planning/validator.py`](../react-agent/src/react_agent/fmri/planning/validator.py), [`policy.py`](../react-agent/src/react_agent/fmri/policy.py), [`planning/guards.py`](../react-agent/src/react_agent/fmri/planning/guards.py) | Tool, resource, budget and policy constraints |
| Tools and evidence | [`tools/registry.py`](../react-agent/src/react_agent/fmri/tools/registry.py), [`ledger.py`](../react-agent/src/react_agent/fmri/ledger.py) | Measurements and question coverage |
| Diagnostic stopping | [`diagnostic/stop_gate.py`](../react-agent/src/react_agent/fmri/diagnostic/stop_gate.py) | Diagnostic-mode stopping within its contract |
| Final report | [`reporting.py`](../react-agent/src/react_agent/fmri/reporting.py) | Verdict, screening decision, stop reason and scope |
| Memory | [`memory/service.py`](../react-agent/src/react_agent/fmri/memory/service.py), [`memory/repository.py`](../react-agent/src/react_agent/fmri/memory/repository.py) | Deterministic episode writing; optional curator proposals |

Planner and replanner are modes of one planning responsibility. Rule screening does not require an LLM. The curator runs only when enabled and applicable.

## Evidence and historical notes

The figures describe implementation paths. They do not claim every role has been validated in a new live campaign or that performance improved. [Canonical acceptance](../react-agent/docs/revision_acceptance_canonical.md) distinguishes offline/CPU checks from live-role and GPU runs not executed in that revision.

Older documents, including `eeg_research_v1_8.md` and `implementation_inventory.md`, are revision-specific. Their role inventories and limitations may differ from the inspected source. Treat those records as historical evidence, not the latest architecture specification.
