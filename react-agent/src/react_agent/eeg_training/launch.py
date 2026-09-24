"""Prepare a retrieval campaign without importing torch."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from react_agent.eeg_research.adapters.ubp_retrieval import UbpRetrievalAdapter
from react_agent.eeg_research.executor import ExecutionRefused, ResearchBudget, run_trial
from react_agent.eeg_research.registry import Registry
from react_agent.eeg_research.report import write_campaign
from react_agent.eeg_research.schemas import ExperimentSpec, ModelRecord, ProfileRecord, TaskCard
from react_agent.eeg_research.task_contract import freeze_task
from react_agent.eeg_research.trace import (
    EVAL_SCORES,
    TrialRecord,
    append_local,
    distinct_settings,
    find_setting,
    finished_count,
    local_actions,
    needs_scoring,
    next_trial_index,
    read_trace,
    scored,
    setting_key,
    write_trial_config,
)
from react_agent.eeg_training.protocol import (
    Design,
    SplitError,
    data_root,
    describe_split,
    format_gpu_seconds,
    learning_rate,
    list_gpus,
    list_subjects,
    parse_gpu_list,
    parse_gpu_seconds,
    parse_stop,
    parse_subject_selection,
    resolve_gpu_seconds,
    options,
    probe_blockers,
    protocol_label,
    resolve_run_gpu,
    split_manifest,
    split_plan,
    STOP_LABELS,
    BASELINE_WEIGHT_DECAY,
    CHAIN_WEIGHT_DECAY,
    chains,
    train_command,
    validate_design,
)


def parse_design(form: dict[str, str]) -> Design:
    """Read one form submission."""
    lr_text = (form.get("lr") or "").strip()
    gpu_text = (form.get("gpu") or "").strip()
    return Design(
        dataset=form.get("dataset") or "eeg",
        exp_setting=form.get("exp_setting") or "intra-subject",
        subject=parse_subject_selection(form.get("subject") or ""),
        epochs=int(form.get("epochs") or "50"),
        seed=int(form.get("seed") or "0"),
        batch_size=int(form.get("batch_size") or "1024"),
        lr=float(lr_text) if lr_text else None,
        train_dir=(form.get("train_dir") or "").strip(),
        test_dir=(form.get("test_dir") or "").strip(),
        gpu=parse_gpu_list(gpu_text),
        data_root=(form.get("data_root") or "").strip(),
        gpu_seconds=parse_gpu_seconds(form.get("gpu_seconds") or ""),
        stop=parse_stop(form.get("stop") or ""),
        training_strategy=form.get("training_strategy") or "pooled_subjects",
        generalization_target=(form.get("generalization_target") or "").strip(),
        held_out_subjects=(form.get("held_out_subjects") or "").strip(),
        policy=form.get("policy") or "adaptive",
        gpu_mode=form.get("gpu_mode") or "explicit",
    )


def task_card(design: Design) -> TaskCard:
    """Record the protocol identity. Image ids are frozen inside the trainer."""
    setting = "within_subject" if design.exp_setting == "intra-subject" else "cross_subject"
    return freeze_task(
        TaskCard.model_validate(
            {
                "task_id": design.campaign_id(),
                "dataset_id": design.dataset,
                "dataset_version": design.exp_setting,
                "dataset_manifest_hash": "ubp-baseline-rn50",
                "train_ids": [f"{design.subject}:train"],
                "validation_ids": [f"{design.subject}:validation"],
                "test_ids": [f"{design.subject}:test"],
                "split_unit": "image" if design.exp_setting == "intra-subject" else "subject",
                "generalization_target": setting,
                "subject_scope": design.subject,
                "preprocessing_hash": "things-preprocessed",
                "metric_implementation_hash": "ubp_baseline_within_batch",
                "target_feature_hash": "rn50-direct",
                "candidate_bank_hash": f"{design.dataset}-{design.exp_setting}",
                "candidate_count": 200,
                "similarity": "cosine",
                "trial_aggregation": "mean",
                "query_unit": "image",
                "permitted_model_ids": ["ubp_eeg_project"],
                "permitted_profile_ids": ["profile_baseline"],
                "seed_schedule": [design.seed],
                "uses_test_for_tuning": False,
            }
        )
    )


def _registry(adapter: UbpRetrievalAdapter) -> Registry:
    model = ModelRecord(
        model_id="ubp_eeg_project",
        adapter_id="ubp_retrieval",
        supported_tasks=["eeg_image_retrieval"],
        input_contract="THINGS EEG or MEG baseline retrieval",
        code_revision="eeg_training.v1.6",
        dependency_environment="EEG_TRAIN_PYTHON",
        approved_profiles=["profile_baseline"],
        availability="unverified",
        reason=adapter.reason,
    )
    profile = ProfileRecord(profile_id="profile_baseline", allowed_changes=[])
    return Registry([model], [profile], {"ubp_retrieval": adapter})


def _budget(design: Design) -> ResearchBudget:
    cap = resolve_gpu_seconds(design.gpu_seconds)
    if design.policy == "adaptive":
        return ResearchBudget(
            max_trials=6,
            max_lm_calls=12,
            max_execution_attempts=4,
            gpu_seconds=cap,
            per_trial_timeout_s=cap,
        )
    return ResearchBudget(
        max_trials=2,
        max_lm_calls=0,
        max_execution_attempts=4,
        gpu_seconds=cap,
        per_trial_timeout_s=cap,
    )


def _base(design: Design, root: Path | None) -> Path:
    return Path(design.data_root) if design.data_root else (root if root is not None else data_root())


def _action_text(action: str) -> str:
    return {"discover": "检索", "probe": "预检", "dry_run": "试运行", "run": "开始训练"}.get(action, action)


def _protocol_text(design: Design) -> str:
    return "被试内" if design.exp_setting == "intra-subject" else "被试间"


def _subject_text(design: Design) -> str:
    if not design.subject:
        return "未选择"
    if design.subject == "all":
        return "全部"
    return design.subject


def _human(message: str) -> str:
    if message == "no_subject":
        return "还没有选择被试"
    return message


def _file_progress(plan) -> dict[str, int] | None:
    """Count unique files that exist. An empty check stays unset."""
    seen: set[Path] = set()
    paths: list[Path] = []
    for path in (*plan.train_files, *plan.val_files, *plan.forbidden_files, *plan.feature_caches):
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    if not paths:
        return None
    return {"found": sum(1 for path in paths if path.is_file()), "total": len(paths)}


def _activity_log(
    design: Design,
    action: str,
    plan,
    blockers: list[str],
    base: Path,
    out_dir: Path | None,
    *,
    campaign_written: bool,
    discovered_subjects: int | None = None,
    gpu_count: int | None = None,
) -> list[str]:
    """Lines for the workbench console. Scores are never invented here."""
    lines = [f"动作：{_action_text(action)}。协议：{protocol_label(design)}。被试：{_subject_text(design)}。"]
    if discovered_subjects is not None:
        lines.append(f"找到被试 {discovered_subjects} 个。")
    if gpu_count is not None:
        lines.append(f"读到 GPU {gpu_count} 块。")
    if plan is not None:
        lines.append(
            f"训练文件 {len(plan.train_files)} 个，验证文件 {len(plan.val_files)} 个，"
            f"不进入训练 {len(plan.forbidden_files)} 个，特征缓存 {len(plan.feature_caches)} 个。"
        )
        if out_dir is not None:
            try:
                command = train_command(design, out_dir, base)
            except SplitError:
                if not any("EEG_TRAIN_PYTHON" in item for item in blockers):
                    lines.append("未配置 EEG_TRAIN_PYTHON，无法确认 torch 解释器")
            else:
                lines.append("命令：" + " ".join(command))
    for item in blockers:
        lines.append(_human(item))
    cap = resolve_gpu_seconds(design.gpu_seconds)
    lines.append(f"GPU 秒数上限：{format_gpu_seconds(cap)}")
    lines.append(f"结束条件：{STOP_LABELS.get(design.stop, design.stop)}")
    if campaign_written:
        lines.append(f"已写入 {design.campaign_id()}")
    return lines


def discover_design(design: Design, root: Path | None = None) -> dict[str, object]:
    """Scan subject directories and GPUs. This does not write a campaign."""
    if design.dataset not in {"eeg", "meg"}:
        raise SplitError("unknown_dataset")
    base = _base(design, root)
    subjects = list_subjects(base, design.dataset)
    gpus = list_gpus()
    return {
        "ok": True,
        "started": False,
        "discovered": True,
        "subjects": subjects,
        "gpus": gpus,
        "blockers": [],
        "campaign_id": "",
        "campaign_written": False,
        "split": None,
        "options": options(),
        "test_result": None,
        "api_usd": None,
        "gpu_seconds": resolve_gpu_seconds(design.gpu_seconds),
        "data_root": str(base),
        "dataset": design.dataset,
        "exp_setting": design.exp_setting,
        "subject": design.subject,
        "batch_size": design.batch_size,
        "lr": learning_rate(design),
        "log": _activity_log(
            design,
            "discover",
            None,
            [],
            base,
            None,
            campaign_written=False,
            discovered_subjects=len(subjects),
            gpu_count=len(gpus),
        ),
        "progress": None,
    }


def _refused(design: Design, root: Path | None, message: str, action: str = "probe") -> dict[str, object]:
    base = _base(design, root)
    subjects = list_subjects(base, design.dataset) if design.dataset in {"eeg", "meg"} else []
    return {
        "ok": False,
        "started": False,
        "discovered": True,
        "blockers": [message],
        "campaign_id": "",
        "split": None,
        "subjects": subjects,
        "gpus": list_gpus(),
        "options": options(),
        "test_result": None,
        "api_usd": None,
        "gpu_seconds": resolve_gpu_seconds(design.gpu_seconds),
        "data_root": str(base),
        "dataset": design.dataset,
        "exp_setting": design.exp_setting,
        "subject": design.subject,
        "log": _activity_log(design, action, None, [message], base, None, campaign_written=False),
        "progress": None,
    }


def trial_steps(design: Design, out_dir: Path) -> list[dict[str, object]]:
    """Decide train, test-only, or skip. A finished root baseline counts as trial 1."""
    count = 2 if chains(design) else 1
    steps: list[dict[str, object]] = []
    for trial in range(1, count + 1):
        slot = out_dir / "trials" / f"t{trial}"
        target = slot
        if trial == 1 and not (slot / "metrics.json").is_file() and (out_dir / "metrics.json").is_file():
            target = out_dir
        metrics_path = target / "metrics.json"
        weight = BASELINE_WEIGHT_DECAY if trial == 1 else CHAIN_WEIGHT_DECAY
        profile = "profile_baseline" if trial == 1 else "profile_weight_decay"
        primary = None
        if metrics_path.is_file():
            stored = json.loads(metrics_path.read_text(encoding="utf-8"))
            if isinstance(stored, dict):
                primary = stored.get("primary_metric")
            action = "skip"
        else:
            action = "train"
            target = slot
        steps.append(
            {
                "trial": trial,
                "action": action,
                "directory": str(target),
                "weight_decay": weight,
                "profile": profile,
                "primary_metric": primary,
            }
        )
    return steps


def _write_chain(out_dir: Path, design: Design, steps: list[dict[str, object]], current: int | None) -> None:
    """Record which trial is running. This file is not a score."""
    payload = {
        "stop": design.stop,
        "label": STOP_LABELS[design.stop],
        "current": current,
        "trials": [
            {
                "trial": step["trial"],
                "profile": step["profile"],
                "weight_decay": step["weight_decay"],
                "action": "running" if step["trial"] == current else step["action"],
                "primary_metric": step["primary_metric"],
            }
            for step in steps
        ],
    }
    (out_dir / "chain.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _apply_changes(design: Design, changes: dict[str, object], *, seed: int) -> Design:
    """Apply the planner's allowed knobs. Other fields stay on the frozen design."""
    weight = design.weight_decay
    learning = design.lr
    if "weight_decay" in changes:
        weight = float(changes["weight_decay"])
    if "learning_rate" in changes:
        learning = float(changes["learning_rate"])
    return replace(design, weight_decay=weight, lr=learning, seed=seed, test_only=False)


def _start_one_trial(
    design: Design,
    out_dir: Path,
    trial_index: int,
    budget: ResearchBudget,
    card: TaskCard,
    base: Path,
) -> object:
    """Start one existing retrieval trial. The caller already chose this action."""
    trial_out = out_dir / "trials" / f"t{trial_index}"
    adapter = UbpRetrievalAdapter(design, base, resolve_gpu_seconds(design.gpu_seconds))
    spec = ExperimentSpec(
        id=f"{design.campaign_id()}-t{trial_index}",
        campaign_id=design.campaign_id(),
        hypothesis_id=f"h-adaptive-{trial_index}",
        model_id="ubp_eeg_project",
        profile_id="profile_baseline",
        frozen_task_hash=card.frozen_task_hash,
        seed=design.seed,
        resolved_config_hash=design.campaign_id(),
        output_dir=str(trial_out),
    )
    return run_trial(adapter, spec, str(trial_out), budget, dry_run=False)


def _base_config(design: Design) -> dict[str, object]:
    """The setting a trial runs with. Legacy trials without a config count as the frozen design."""
    return {
        "seed": design.seed,
        "learning_rate": learning_rate(design),
        "weight_decay": design.weight_decay,
        "epochs": design.epochs,
    }


def _design_for(design: Design, config: dict[str, object]) -> Design:
    return replace(
        design,
        seed=int(config["seed"]),  # type: ignore[arg-type]
        lr=float(config["learning_rate"]),  # type: ignore[arg-type]
        weight_decay=float(config["weight_decay"]),  # type: ignore[arg-type]
        test_only=False,
    )


def _next_seed(trials: list[TrialRecord], design: Design, base: dict[str, object]) -> int:
    """The first seed with this learning rate and weight decay that has no trial yet."""
    knobs = setting_key(_base_config(design))[1:]
    used = {row.key(base)[0] for row in trials if row.status in {"finished", "running"} and row.key(base)[1:] == knobs}
    seed = design.seed
    while seed in used:
        seed += 1
    return seed


def _score_checkpoint(design: Design, trial_dir: Path, base: Path) -> None:
    """Score an existing checkpoint in the child process. Training does not restart."""
    adapter = UbpRetrievalAdapter(design, base, resolve_gpu_seconds(design.gpu_seconds))
    adapter.evaluate(trial_dir)


def _write_score_error(trial_dir: Path, detail: str) -> None:
    (trial_dir / EVAL_SCORES).write_text(json.dumps({"error": detail}, ensure_ascii=False), encoding="utf-8")


def _curve_summary(row: TrialRecord) -> dict[str, object]:
    """Best epoch from the stored curve. Fixed-bank rows win over within-batch rows."""
    rows = []
    path = row.path / "history.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except ValueError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    if not rows:
        return {"trial": row.name, "epochs_recorded": 0}
    if not all("fixed_bank_top1" in item for item in rows):
        return {"trial": row.name, "epochs_recorded": len(rows), "curve_metric": "fixed_bank_top1", "best_value": None}
    best = max(rows, key=lambda item: float(item.get("fixed_bank_top1") or 0.0))
    return {
        "trial": row.name,
        "epochs_recorded": len(rows),
        "curve_metric": "fixed_bank_top1",
        "best_epoch": best.get("epoch"),
        "best_value": best.get("fixed_bank_top1"),
        "last_train_loss": rows[-1].get("train_loss"),
    }


def _split_counts(design: Design, base: Path) -> dict[str, object]:
    """Count frozen files. A missing split stays uncounted."""
    try:
        plan = split_plan(base, design)
    except (SplitError, OSError, ValueError):
        return {"split": "划分尚未确认"}
    return {
        "train_files": len(plan.train_files),
        "validation_files": len(plan.val_files),
        "held_out_files": len(plan.forbidden_files),
    }


def _local_findings(action: str, design: Design, trials: list[TrialRecord], base: Path) -> dict[str, object]:
    """Record one non-training action from the trace. Test scores are not read."""
    unique = [row for row in trials if row.duplicate_of is None]
    if action == "inspect_task_data":
        found: dict[str, object] = {"protocol": protocol_label(design), "trials": len(unique)}
        found.update(_split_counts(design, base))
        return found
    if action == "analyze_learning_curve":
        return {"curves": [_curve_summary(row) for row in unique]}
    if action == "evaluate_validation":
        return {
            "validation": [
                {"trial": row.name, "fixed_bank_top1": row.fixed_bank_top1, "fixed_bank_top5": row.fixed_bank_top5}
                for row in unique
            ]
        }
    if action == "retrieve_research_memory":
        return {"memory": [row.name for row in scored(trials)]}
    if action == "analyze_retrieval_errors":
        return {"retrieval_errors": "没有逐条 query 的错误记录"}
    if action == "propose_experiment":
        return {"proposal": "实验提议已记下"}
    return {"note": action}


def _resolved_setting(row: TrialRecord, base: dict[str, object]) -> dict[str, object]:
    """Numeric knobs. A trial without its own config uses the protocol default."""
    source = row.config or base
    return {
        "seed": source.get("seed"),
        "learning_rate": source.get("learning_rate"),
        "weight_decay": source.get("weight_decay"),
    }


def _tried_row(row: TrialRecord, base: dict[str, object]) -> dict[str, object]:
    """One row of the search table. Batch-local scores are not included."""
    setting = _resolved_setting(row, base)
    return {
        "trial": row.name,
        "learning_rate": setting["learning_rate"],
        "weight_decay": setting["weight_decay"],
        "status": row.status,
        "fixed_bank_top1": row.fixed_bank_top1,
        "validation_candidate_count": row.validation_candidate_count,
    }


def _planner_trial(row: TrialRecord, base: dict[str, object]) -> dict[str, object]:
    """One trial for the planner. The setting is always numeric."""
    return {
        "trial": row.name,
        "status": row.status,
        "setting": _resolved_setting(row, base),
        "fixed_bank_top1": row.fixed_bank_top1,
        "fixed_bank_top5": row.fixed_bank_top5,
        "validation_candidate_count": row.validation_candidate_count,
        "score_error": row.score_error,
    }


def _observation(
    design: Design,
    trials: list[TrialRecord],
    base_config: dict[str, object],
    seen: list[dict[str, object]],
) -> dict[str, object]:
    """What the planner sees. Everything comes from the trace on disk."""
    unique = [row for row in trials if row.duplicate_of is None]
    best = max(scored(trials), key=lambda row: row.fixed_bank_top1 or 0.0, default=None)
    checked: dict[str, object] = {}
    for row in seen:
        checked[str(row.get("action"))] = row.get("findings")
    observation: dict[str, object] = {
        "protocol": protocol_label(design),
        "base_setting": base_config,
        "trials": [_planner_trial(row, base_config) for row in unique],
        "tried_settings": [_tried_row(row, base_config) for row in unique],
        "duplicate_trials": [row.name for row in trials if row.duplicate_of],
        "distinct_settings": distinct_settings(trials, base_config),
        "selection_metric": "fixed_bank_top1",
        "validation_metric": None if best is None else best.fixed_bank_top1,
        "best_trial": None if best is None else best.name,
    }
    if checked:
        observation["checked"] = checked
    return observation


def _memory_rows(design: Design, trials: list[TrialRecord]) -> list[dict[str, object]]:
    return [
        {
            "memory_id": row.name,
            "task_hash": design.campaign_id(),
            "metric": "fixed_bank_top1",
            "primary_metric": row.fixed_bank_top1,
        }
        for row in scored(trials)
    ]


def run_adaptive_loop(
    design: Design,
    out_dir: Path,
    budget: ResearchBudget,
    card: TaskCard,
    base: Path,
    *,
    backend: Any = None,
    scorer: Any = None,
) -> dict[str, object]:
    """Read the trace, score what is missing, ask for one action, then read the trace again."""
    from react_agent.eeg_research.controller import TRAIN_ACTIONS, ResearchController

    controller = ResearchController(out_dir)
    score = scorer or _score_checkpoint
    base_config = _base_config(design)
    notes: list[str] = []
    blockers: list[str] = []
    started = False
    record: dict[str, object] = {}
    attempted: set[str] = set()
    repeated: set[object] = set()
    while True:
        trials = read_trace(out_dir)
        pending = needs_scoring(trials)
        if pending:
            for row in pending:
                if row.name in attempted:
                    _write_score_error(row.path, "补算没有写出分数")
                    notes.append(f"{row.name} 补算没有写出分数。")
                    continue
                attempted.add(row.name)
                notes.append(f"{row.name} 还没有验证固定候选集分数，用已有 checkpoint 补算。")
                try:
                    score(_design_for(design, row.config or base_config), row.path, base)
                except (RuntimeError, SplitError, ValueError, OSError, ExecutionRefused) as exc:
                    _write_score_error(row.path, str(exc))
                    notes.append(f"{row.name} 补算失败：{exc}")
            continue
        budget.trials = distinct_settings(trials, base_config)
        if budget.lm_calls >= budget.max_lm_calls:
            blockers = ["无法继续规划：语言模型次数已用完，不会退回固定两试。"]
            notes.extend(blockers)
            break
        done_now = finished_count(trials)
        seen = local_actions(out_dir)
        completed = sorted({str(row.get("action")) for row in seen if row.get("trial_count") == done_now})
        record = controller.decide(
            campaign_id=design.campaign_id(),
            observation=_observation(design, trials, base_config, seen),
            trials_used=budget.trials,
            max_trials=budget.max_trials,
            memory_rows=_memory_rows(design, trials),
            backend=backend,
            budget=budget,
            completed_actions=completed,
        )
        if not record.get("action_label"):
            notes.append("正在请求规划")
            notes.append(str(record.get("detail") or ""))
        if record.get("status") != "planned":
            blockers = [str(record.get("detail") or "planning_blocked")]
            break
        action = str(record.get("action") or "")
        if action == "stop_research":
            break
        if action not in TRAIN_ACTIONS:
            if action in completed:
                if action in repeated:
                    note = f"{record.get('action_label') or action}这一步已经做过。"
                    notes.append(note)
                    record = controller.note_repeat(record, note)
                    break
                repeated.add(action)
                continue
            append_local(out_dir, action, _local_findings(action, design, trials, base), done_now)
            continue
        if any(row.status == "running" for row in trials):
            note = "已有试验在训练，这一轮不再启动新训练。"
            notes.append(note)
            paused = dict(record)
            paused["action"] = ""
            paused["action_label"] = ""
            record = controller.note_repeat(paused, note)
            break
        changes = record.get("changes") if isinstance(record.get("changes"), dict) else {}
        try:
            trial_design = _apply_changes(design, changes, seed=design.seed)
            if action == "replicate_candidate":
                trial_design = replace(trial_design, seed=_next_seed(trials, trial_design, base_config))
            validate_design(trial_design)
        except (SplitError, ValueError) as exc:
            blockers = [str(exc)]
            notes.append(str(exc))
            break
        config = _base_config(trial_design)
        key = setting_key(config)
        existing = find_setting(trials, key, base_config)
        if existing is not None:
            if key in repeated:
                note = f"这个设置已有结果（{existing.name}），仍被要求再训练一次。"
                notes.append(note)
                record = controller.note_repeat(record, note)
                break
            repeated.add(key)
            notes.append(f"这个设置已有结果（{existing.name}），直接复用，不再训练。")
            continue
        if budget.trials >= budget.max_trials:
            notes.append("试验次数已到上限。")
            break
        index = next_trial_index(trials)
        write_trial_config(out_dir / "trials" / f"t{index}", config)
        try:
            _start_one_trial(trial_design, out_dir, index, budget, card, base)
        except (RuntimeError, SplitError, ValueError, ExecutionRefused) as exc:
            blockers = [str(exc)]
            notes.append(str(exc))
            break
        started = True
        notes.append(f"t{index} 已写入。")
    budget.trials = distinct_settings(read_trace(out_dir), base_config)
    return {
        "started": started,
        "blockers": blockers,
        "notes": notes,
        "research": record,
        "stop_reason": record.get("status"),
        "trials_used": budget.trials,
    }


def launch_design(design: Design, out_dir: Path, action: str, root: Path | None = None) -> dict[str, object]:
    """Probe, write a dry-run campaign, or start one trial when the cap is known."""
    try:
        validate_design(design)
    except (SplitError, ValueError) as exc:
        return _refused(design, root, str(exc), action)
    base = _base(design, root)
    try:
        blockers = probe_blockers(design, base)
        plan = split_plan(base, design)
    except SplitError as exc:
        return _refused(design, root, str(exc), action)
    adapter = UbpRetrievalAdapter(design, base, resolve_gpu_seconds(design.gpu_seconds))
    registry = _registry(adapter)
    registry.probe_all()
    budget = _budget(design)
    card = task_card(design)
    state = {
        "campaign_id": design.campaign_id(),
        "hypotheses": [],
        "experiments": [],
        "evidence": [],
        "completed_ids": [],
        "decision": {},
        "stop_reason": None,
    }
    if action == "run" and budget.gpu_seconds is None and "没有 GPU 秒数上限，只允许试运行" not in blockers:
        blockers.append("没有 GPU 秒数上限，只允许试运行")
    if action == "probe":
        return _payload(design, plan, blockers, base, out_dir, action, started=False, campaign_written=False)
    if design.policy == "agentic":
        note = "代码级研究在 runs/eeg_research_v18 下单独运行，这次点击不写旧 campaign。"
        return _payload(
            design, plan, blockers, base, out_dir, action,
            started=False, campaign_written=False, notes=[note],
        )
    state["stop_reason"] = "; ".join(blockers) if blockers else "dry_run"
    notes: list[str] = []
    started = False
    if action == "run" and not blockers:
        try:
            design = resolve_run_gpu(design)
        except SplitError as exc:
            blockers = [str(exc)]
    research: dict[str, object] | None = None
    if action == "run" and not blockers and design.policy == "adaptive":
        outcome = run_adaptive_loop(design, out_dir, budget, card, base)
        notes.extend(outcome["notes"])  # type: ignore[arg-type]
        research = outcome["research"] if isinstance(outcome["research"], dict) else None
        state["stop_reason"] = outcome["stop_reason"]
        state["decision"] = research or {}
        blockers = list(outcome["blockers"])  # type: ignore[arg-type]
        started = bool(outcome["started"])
    elif action == "run" and not blockers:
        steps = trial_steps(design, out_dir)
        for step in steps:
            label = f"第 {step['trial']} 次 {step['profile']}，weight decay {step['weight_decay']}"
            if step["action"] == "skip":
                notes.append(f"{label} 已有训练记录，跳过。")
                continue
            trial_design = replace(
                design,
                weight_decay=float(step["weight_decay"]),
                test_only=step["action"] == "test_only",
            )
            trial_out = Path(str(step["directory"]))
            _write_chain(out_dir, design, steps, int(step["trial"]))
            notes.append(f"{label} {'补测试' if trial_design.test_only else '开始'}。")
            adapter = UbpRetrievalAdapter(trial_design, base, resolve_gpu_seconds(design.gpu_seconds))
            spec = ExperimentSpec(
                id=f"{design.campaign_id()}-t{step['trial']}",
                campaign_id=design.campaign_id(),
                hypothesis_id="h-baseline" if step["trial"] == 1 else "h-weight-decay",
                model_id="ubp_eeg_project",
                profile_id=str(step["profile"]),
                frozen_task_hash=card.frozen_task_hash,
                seed=design.seed,
                resolved_config_hash=design.campaign_id(),
                output_dir=str(trial_out),
            )
            try:
                bundle = run_trial(adapter, spec, str(trial_out), budget, dry_run=False)
            except (RuntimeError, SplitError) as exc:
                state["stop_reason"] = str(exc)
                notes.append(str(exc))
                write_campaign(out_dir, card, registry, state, budget)
                return _payload(
                    design, plan, [str(exc)], base, out_dir, action,
                    started=False, campaign_written=True, notes=notes,
                )
            started = True
            step["action"] = "done"
            step["primary_metric"] = bundle.primary_metric
            state["experiments"].append(bundle.model_dump())
        state["stop_reason"] = "trial_budget_exhausted" if chains(design) else "trial_recorded"
        budget.trials = len([step for step in steps if step["action"] == "done"])
        _write_chain(out_dir, design, steps, None)
    elif action != "probe":
        started = False
    write_campaign(out_dir, card, registry, state, budget)
    (out_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest(plan, design), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "split_plan.json").write_text(
        json.dumps(
            {
                "val_mode": plan.val_mode,
                "train_files": [str(path) for path in plan.train_files],
                "val_files": [str(path) for path in plan.val_files],
                "forbidden_files": [str(path) for path in plan.forbidden_files],
                "test_result": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return _payload(
        design,
        plan,
        blockers,
        base,
        out_dir,
        action,
        started=started,
        campaign_written=True,
        notes=notes,
        research=_research_view(research),
    )


def _payload(
    design: Design,
    plan,
    blockers: list[str],
    base: Path,
    out_dir: Path,
    action: str,
    *,
    started: bool,
    campaign_written: bool,
    notes: list[str] | None = None,
    research: dict[str, object] | None = None,
) -> dict[str, object]:
    log = _activity_log(design, action, plan, blockers, base, out_dir, campaign_written=campaign_written)
    log.extend(notes or [])
    return {
        "ok": not blockers,
        "started": started,
        "blockers": blockers,
        "campaign_id": design.campaign_id(),
        "campaign_written": campaign_written,
        "dataset": design.dataset,
        "exp_setting": design.exp_setting,
        "subject": design.subject,
        "val_mode": plan.val_mode,
        "forbidden": [str(path) for path in plan.forbidden_files],
        "test_result": None,
        "api_usd": None,
        "gpu_seconds": resolve_gpu_seconds(design.gpu_seconds),
        "batch_size": design.batch_size,
        "lr": learning_rate(design),
        "gpu": design.gpu,
        "data_root": str(base),
        "split": describe_split(plan),
        "protocol_label": protocol_label(design),
        "subjects": list_subjects(base, design.dataset),
        "discovered": True,
        "gpus": list_gpus(),
        "options": options(),
        "log": log,
        "progress": _file_progress(plan),
        "research": research,
    }


def _research_view(record: dict[str, object] | None) -> dict[str, object] | None:
    """Fields the console can show. The raw object stays out of the summary line."""
    if not record:
        return None
    keys = ("phase", "status", "action", "action_label", "reason", "detail", "raw", "legacy_fixed")
    return {key: record.get(key) for key in keys}
