"""Typed configuration for the fMRI check loop."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "fmri_check.v1.1"


class NearConstantTol(BaseModel):
    """Absolute and relative tolerances for near-constant series."""

    model_config = ConfigDict(extra="forbid")

    abs_tol: float = 1e-8
    rel_tol: float = 1e-6


class BasicStatsParams(BaseModel):
    """Parameters for basic_statistics."""

    model_config = ConfigDict(extra="forbid")

    near_constant: NearConstantTol = Field(default_factory=NearConstantTol)
    quantiles: list[float] = Field(default_factory=lambda: [0.05, 0.5, 0.95])
    demo_temporal_change_low: float | None = 0.05
    demo_temporal_spike_ratio: float | None = 8.0
    enable_demo_temporal_heuristics: bool = True


class TemporalParams(BaseModel):
    """Parameters for temporal_diagnostics."""

    model_config = ConfigDict(extra="forbid")

    min_length: int = 4
    spike_ratio_heuristic: float = 8.0
    threshold_source: Literal["heuristic", "reference_calibrated"] = "heuristic"


class ReferenceParams(BaseModel):
    """Parameters for reference_distribution."""

    model_config = ConfigDict(extra="forbid")

    stats_path: str | None = None
    min_samples: int = 3
    mad_eps: float = 1e-12
    robust_z_flag: float = 6.0
    quality_verdict_enabled: bool = False
    quality_min_n: int = 30
    required_compat_fields: list[str] = Field(
        default_factory=lambda: [
            "generation_profile_id",
            "spatial_representation",
            "normalization",
        ]
    )


class ResourceParams(BaseModel):
    """Explicit resource bindings. IDs only; no free paths from the LM."""

    model_config = ConfigDict(extra="forbid")

    gray_control_id: str | None = None
    gray_control_path: str | None = None
    atlas_dir: str | None = None
    atlas_id: str = "destrieux_fsaverage5_lh_rh"
    cohort_index_path: str | None = None
    cohort_manifest_id: str | None = None
    asset_root: str | None = "../assets"
    cortexmae_model_id: str = "cortex_mae_parcel"
    schaefer_atlas_dir: str | None = None
    image_encoder_id: str = "clip"
    image_embedding_dir: str | None = None
    samples_manifest_path: str | None = None


class CheckProfileParams(BaseModel):
    """Which questions a profile must answer."""

    model_config = ConfigDict(extra="forbid")

    name: Literal[
        "numeric_minimal",
        "tribe_diagnostic",
        "image16_numeric_diagnostic",
    ] = "numeric_minimal"
    required_questions: list[str] = Field(default_factory=list)


class ScoreWeights(BaseModel):
    """Heuristic priority weights. Not a calibrated risk model."""

    model_config = ConfigDict(extra="forbid")

    w_a: float = 0.6
    w_m: float = 0.4
    w_d: float = 0.0


class LmProfile(BaseModel):
    """One DeepSeek call profile."""

    model_config = ConfigDict(extra="forbid")

    model: str = "deepseek-chat"
    thinking: bool = False
    reasoning_effort: str | None = None
    timeout_s: float = 60.0
    max_tokens: int = 800
    temperature: float | None = None
    supports_temperature: bool = False
    supports_thinking: bool = True
    supports_json_object: bool = True


class GenerationParams(BaseModel):
    """TRIBE image→fMRI generation. Independent of the decision LM backend."""

    model_config = ConfigDict(extra="forbid")

    backend: Literal["worker", "mock"] = "worker"
    repo_path: str = "../../tribev2"
    python_executable: str = "${TRIBE_PYTHON}"
    checkpoint: str = "facebook/tribev2"
    cache_folder: str = "../assets/generation/hf_cache"
    artifact_root: str = "../assets/generation"
    device: str = "cuda"
    cuda_visible_devices: str | None = None
    profile_id: str = "static_gray4_image1_gray11_tribev2_v1"
    fps: int = 10
    pre_gray_s: float = 4.0
    image_s: float = 1.0
    post_gray_s: float = 11.0
    gray_rgb: tuple[int, int, int] = (128, 128, 128)
    codec: str = "libx264"
    missing_audio: str = "zero_fill_aggregate_features"
    missing_text: str = "zero_fill_aggregate_features"
    export_rule_id: str = "segment_time_window_0_16_exclusive_v1"


class Budgets(BaseModel):
    """Per-sample and batch budgets."""

    model_config = ConfigDict(extra="forbid")

    max_rounds: int = 6
    max_tool_calls: int = 6
    max_lm_calls: int = 6
    no_progress_limit: int = 2
    max_retries: int = 1
    walltime_s: float | None = 120.0
    max_batch_lm_calls: int | None = None
    max_batch_usd: float | None = None
    usd_input_per_million: float | None = None
    usd_output_per_million: float | None = None
    price_source: str | None = None
    price_date: str | None = None
    currency: str = "USD"
    max_generation_requests: int | None = None
    max_new_tribe_predictions: int | None = None
    generation_walltime_s: float | None = None
    max_replans: int = 2
    max_json_repairs: int = 1
    max_optional_tool_executions_per_sample: int | None = None


class HybridPolicyCfg(BaseModel):
    """Hybrid routing knobs."""

    model_config = ConfigDict(extra="forbid")

    upgrade_once: bool = True
    candidate_count_for_reasoning: int = 2


class PlanningCfg(BaseModel):
    """Persistent-plan policy. Disabled by default so old YAML stays rule/hybrid."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    prompt_version: str = "planner_v1"
    max_replans: int = 2
    require_initial_plan: bool = True
    require_gray_question_for_non_control: bool = True
    require_temporal_description_for_non_control: bool = True
    api_failure_mode: Literal["fail", "degrade_to_rule"] = "fail"
    run_goal: Literal["quality_screening", "feature_extraction"] = "quality_screening"


class MemoryCfg(BaseModel):
    """Long-term episode store. Separate from LangGraph checkpoints."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["off", "read_write", "read_only"] = "off"
    repository: Literal["sqlite"] = "sqlite"
    path: str = "../assets/memory/fmri_memory.sqlite3"
    namespace: str = "default"
    max_retrieved_items: int = 6
    curator_enabled: bool = False
    promote_candidates_automatically: bool = False
    prompt_version: str = "memory_curator_v1"


class RoutingRuleCfg(BaseModel):
    """When to open a follow-up ticket. Heuristic rules cannot flag quality."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    threshold: float = 5.0
    basis: Literal["heuristic", "calibrated", "exact_contract"] = "heuristic"
    decision_effect: Literal["none"] = "none"
    question: str = "contrast_transition_localization"


class RoutingCfg(BaseModel):
    """exploratory_v1 routing. Thresholds are inspection hints, not physiology."""

    model_config = ConfigDict(extra="forbid")

    profile: str = "exploratory_v1"
    contrast_rms_step_ratio: RoutingRuleCfg = Field(
        default_factory=lambda: RoutingRuleCfg(
            enabled=False,
            question="contrast_transition_localization",
        )
    )
    spatial_concentration: RoutingRuleCfg = Field(
        default_factory=lambda: RoutingRuleCfg(enabled=False, question="spatial_concentration")
    )


class DiagnosticCfg(BaseModel):
    """V1.4 numeric diagnostic profile. Off by default so V1.3 YAML is unchanged."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    analysis_goal: str = "image16_numeric_diagnostic"
    coverage_profile: str = "image16_numeric_diagnostic_v1"
    coverage_profile_version: str = "v1.4"
    metric_definition_version: str = "ras_v1"
    require_contrast_for_temporal: bool = True
    routing: RoutingCfg = Field(default_factory=RoutingCfg)
    screen_depth: Literal["standard", "deep"] = "standard"


class ReportingCfg(BaseModel):
    """Which extra artifacts a planned run writes."""

    model_config = ConfigDict(extra="forbid")

    emit_plan: bool = True
    emit_memory_retrieval: bool = True
    emit_llm_usage: bool = True


class FmriCheckConfig(BaseModel):
    """Resolved fMRI check configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    required_checks: list[str] = Field(
        default_factory=lambda: ["validate_input", "basic_statistics"]
    )
    enabled_tools: list[str] = Field(
        default_factory=lambda: [
            "validate_input",
            "basic_statistics",
            "temporal_diagnostics",
            "roi_summary",
            "reference_distribution",
        ]
    )
    basic_statistics: BasicStatsParams = Field(default_factory=BasicStatsParams)
    temporal_diagnostics: TemporalParams = Field(default_factory=TemporalParams)
    reference: ReferenceParams = Field(default_factory=ReferenceParams)
    resources: ResourceParams = Field(default_factory=ResourceParams)
    check_profile: CheckProfileParams = Field(default_factory=CheckProfileParams)
    score: ScoreWeights = Field(default_factory=ScoreWeights)
    budgets: Budgets = Field(default_factory=Budgets)
    fast: LmProfile = Field(default_factory=LmProfile)
    reasoning: LmProfile = Field(
        default_factory=lambda: LmProfile(thinking=True, max_tokens=1200)
    )
    hybrid: HybridPolicyCfg = Field(default_factory=HybridPolicyCfg)
    planning: PlanningCfg = Field(default_factory=PlanningCfg)
    diagnostic: DiagnosticCfg = Field(default_factory=DiagnosticCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    reporting: ReportingCfg = Field(default_factory=ReportingCfg)
    cache_enabled: bool = True
    generation: GenerationParams = Field(default_factory=GenerationParams)
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_api_key: str | None = None
    backend_default: Literal["mock", "deepseek", "none"] = "mock"
    policy_default: Literal["rule", "hybrid", "planned"] = "rule"
    prompt_version: str = "fmri_decision.v1"

    @model_validator(mode="after")
    def _deep_screen_questions(self) -> "FmriCheckConfig":
        """Deep screening makes the L2/L3 questions required. Standard runs are unchanged."""
        if self.diagnostic.enabled and self.diagnostic.screen_depth == "deep":
            required = self.check_profile.required_questions
            for question_id, tool in DEEP_SCREEN_QUESTIONS:
                if question_id not in required:
                    required.append(question_id)
                if tool not in self.enabled_tools:
                    self.enabled_tools.append(tool)
        return self


DEEP_SCREEN_QUESTIONS: list[tuple[str, str]] = [
    ("roi", "surface_roi_profile"),
    ("specificity", "cross_image_specificity"),
    ("calibrated_reference", "reference_distribution"),
    ("image_fmri_rsa", "semantic_consistency"),
]


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge nested dictionaries without mutating inputs."""
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _ensure_tribe_python() -> None:
    """Read only TRIBE_PYTHON from .env. Do not load API keys into the process."""
    if os.environ.get("TRIBE_PYTHON"):
        return
    env_path = _REPO_ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "TRIBE_PYTHON":
            os.environ["TRIBE_PYTHON"] = value.strip().strip('"').strip("'")
            return


def _resolve_configured_path(raw: str) -> str:
    """Expand ``$VAR`` / ``${VAR}``, then resolve relative paths from react-agent.

    Unset variables stay literal so a missing interpreter is not joined onto
    the repository path.
    """
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if "$" in expanded:
        return expanded
    path = Path(expanded)
    if not path.is_absolute():
        path = (_REPO_ROOT / path).resolve()
    return str(path)


def load_config(
    path: str | Path | None = None,
    *,
    cli_overrides: dict[str, Any] | None = None,
) -> FmriCheckConfig:
    """Load YAML defaults, then credentials from env, then CLI overrides.

    Priority: CLI explicit > credential env vars > YAML > coded defaults.
    Environment variables never override scientific thresholds.
    """
    _ensure_tribe_python()
    data: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("config YAML must be a mapping")
        data = loaded
    cfg = FmriCheckConfig.model_validate(data)
    if cfg.reference.stats_path:
        cfg.reference.stats_path = _resolve_configured_path(cfg.reference.stats_path)
    if cfg.resources.atlas_dir:
        cfg.resources.atlas_dir = _resolve_configured_path(cfg.resources.atlas_dir)
    if cfg.resources.cohort_index_path:
        cfg.resources.cohort_index_path = _resolve_configured_path(cfg.resources.cohort_index_path)
    if cfg.resources.gray_control_path:
        cfg.resources.gray_control_path = _resolve_configured_path(cfg.resources.gray_control_path)
    for field in (
        "asset_root",
        "schaefer_atlas_dir",
        "image_embedding_dir",
        "samples_manifest_path",
    ):
        raw = getattr(cfg.resources, field)
        if raw:
            setattr(cfg.resources, field, _resolve_configured_path(raw))
    for field in ("repo_path", "python_executable", "cache_folder", "artifact_root"):
        raw = getattr(cfg.generation, field)
        if raw:
            setattr(cfg.generation, field, _resolve_configured_path(raw))
    if cfg.memory.path:
        cfg.memory.path = _resolve_configured_path(cfg.memory.path)

    key = os.environ.get("DEEPSEEK_API_KEY")
    if key and key not in {"", "...", "...."}:
        cfg.deepseek_api_key = key
    base = os.environ.get("DEEPSEEK_BASE_URL")
    if base:
        cfg.deepseek_base_url = base
    fast_model = os.environ.get("DEEPSEEK_FAST_MODEL")
    if fast_model:
        cfg.fast.model = fast_model
    reason_model = os.environ.get("DEEPSEEK_REASONING_MODEL")
    if reason_model:
        cfg.reasoning.model = reason_model

    if cli_overrides:
        merged = _deep_merge(cfg.model_dump(), cli_overrides)
        cfg = FmriCheckConfig.model_validate(merged)
    if cfg.budgets.max_batch_usd is not None and cfg.budgets.usd_input_per_million is None:
        raise ValueError("strict USD cap requires usd_input_per_million price fields")
    return cfg


def redacted_config(cfg: FmriCheckConfig) -> dict[str, Any]:
    """Dump config with secrets removed."""
    payload = cfg.model_dump()
    if payload.get("deepseek_api_key"):
        payload["deepseek_api_key"] = "***"
    return payload


def resolve_policy_backend(
    cfg: FmriCheckConfig,
    *,
    policy: str | None = None,
    backend: str | None = None,
) -> tuple[str, str]:
    """Resolve CLI/Studio policy+backend. Only rule forces backend=none."""
    resolved_policy = policy or cfg.policy_default
    if resolved_policy == "rule":
        return "rule", "none"
    resolved_backend = backend or cfg.backend_default
    if resolved_policy in {"hybrid", "planned"} and resolved_backend == "none":
        raise ValueError(f"{resolved_policy} policy requires --backend mock or deepseek")
    if resolved_backend == "deepseek" and not cfg.deepseek_api_key:
        raise ValueError("backend=deepseek requires DEEPSEEK_API_KEY")
    return resolved_policy, resolved_backend


def planned_preflight(cfg: FmriCheckConfig, *, backend: str, policy: str) -> dict[str, Any]:
    """Local checks before expensive generation. Does not prove the key works."""
    if policy == "planned" and backend == "deepseek" and not cfg.deepseek_api_key:
        raise ValueError("planned+deepseek requires DEEPSEEK_API_KEY; refusing silent rule fallback")
    if policy == "planned" and backend not in {"mock", "deepseek"}:
        raise ValueError("planned policy requires backend mock or deepseek")
    return {
        "policy": policy,
        "backend": backend,
        "model": cfg.fast.model,
        "run_mode": "planned" if policy == "planned" else policy,
        "memory_mode": cfg.memory.mode if cfg.memory.enabled else "off",
        "memory_namespace": cfg.memory.namespace,
        "planning_enabled": cfg.planning.enabled or policy == "planned",
        "max_lm_calls": cfg.budgets.max_lm_calls,
        "max_replans": cfg.planning.max_replans,
        "max_batch_lm_calls": cfg.budgets.max_batch_lm_calls,
        "api_failure_mode": cfg.planning.api_failure_mode,
        "key_present": bool(cfg.deepseek_api_key),
        "key_validated": False,
    }
