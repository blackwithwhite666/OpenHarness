"""Eval spec + preset resolution for ``ohmo evals run``.

A *spec* (``evals/spec.json`` in an eval bundle) freezes **how** a bundle is run,
so a replay can't silently drift from the recording. It declares named *presets*
— e.g. ``inner`` (offline, cache-replayed) and ``faithful`` (live) — and each
preset pins the runner, scorer, model, fixture-match, the frozen-input files
(system prompt, baked histories, completion cache) and, optionally, the gate
thresholds. ``ohmo evals run --preset inner`` resolves every option from the
spec; an explicit CLI flag still overrides its preset value for ad-hoc runs.

Why this exists: the inner-loop bundle only replays byte-for-byte when the record
environment equals the replay environment (pinned prompt + baked history + mock
skill + fixed cache). Spelling those couplings out as ~10 CLI flags on every run
(record *and* replay, in CI *and* the README) is how they drift. A spec is the
single source of truth for both lanes — record reads the same preset it replays.

Paths inside a preset are resolved **relative to the spec file's directory**, so
a bundle stays portable: copy ``evals/`` anywhere and ``--preset inner`` still
finds ``system_prompt.txt`` / ``histories.json`` / ``completions.json`` next to
``spec.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_SPEC_FILENAME = "spec.json"
SUPPORTED_SPEC_VERSIONS = (1,)
CACHE_MODES = ("off", "record", "strict")

# ResolvedRun fields a CLI flag may override (see ``resolve_run(overrides=...)``).
OVERRIDABLE_FIELDS = (
    "pack_filename",
    "agent_runner_name",
    "scorer",
    "model",
    "provider_profile",
    "fixture_match",
    "samples",
    "judge_grounding",
    "judge_votes",
    "system_prompt_file",
    "histories_file",
    "rubrics_file",
    "grounding_mode",
    "live_skill",
    "cache_completions",
    "cache_mode",
    "cache_prune_to",
)


class SpecError(ValueError):
    """A spec file is missing, malformed, or names an unknown preset.

    Subclasses ``ValueError`` so ``ohmo evals run``'s existing error handling
    prints it to stderr and exits 1.
    """


@dataclass(frozen=True)
class GateThresholds:
    """Pass/fail bar for a preset's gate."""

    hit_floor: float
    passed_baseline: int


@dataclass(frozen=True)
class GateResult:
    """Outcome of applying a preset's gate to an eval report."""

    ok: bool
    hit_rate: float
    hits: int
    misses: int
    passed: int
    total: int
    thresholds: GateThresholds
    reasons: tuple[str, ...]

    def describe(self, label: str) -> str:
        """Render a human, CI-log-friendly multi-line verdict."""
        head = (
            f"{label} gate: cases={self.total} passed={self.passed} "
            f"cache_hit_rate={self.hit_rate:.4f} "
            f"(hits={self.hits} misses={self.misses})"
        )
        body = "\n".join(f"  FAIL: {reason}" for reason in self.reasons)
        tail = f"{label} gate: {'OK' if self.ok else 'FAILED'}"
        return "\n".join(part for part in (head, body, tail) if part)


@dataclass(frozen=True)
class ResolvedRun:
    """A fully-resolved run configuration derived from a spec preset.

    File paths are absolute. ``cache_mode`` is one of :data:`CACHE_MODES`; the
    caller translates it to the ``run_ohmo_eval_report`` cache kwargs.
    """

    spec_path: Path
    preset: str
    pack_filename: str
    agent_runner_name: str
    scorer: str | None
    model: str | None
    provider_profile: str | None
    fixture_match: str
    samples: int
    judge_grounding: bool
    judge_votes: int
    system_prompt_file: str | None
    histories_file: str | None
    rubrics_file: str | None
    grounding_mode: str
    live_skill: bool
    cache_completions: str | None
    cache_mode: str
    cache_prune_to: str | None
    gate: GateThresholds | None


def default_spec_path(workspace: Path | None) -> Path | None:
    """The conventional spec location for a workspace: ``<ws>/evals/spec.json``.

    Returns the path only if it exists, else ``None`` (so callers can fall back
    to plain flag-driven runs).
    """
    if workspace is None:
        return None
    candidate = Path(workspace) / "evals" / DEFAULT_SPEC_FILENAME
    return candidate if candidate.is_file() else None


def load_spec(spec_path: str | Path) -> dict[str, Any]:
    """Read + validate a spec file into a plain dict."""
    path = Path(spec_path).expanduser()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SpecError(f"eval spec not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read eval spec {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SpecError(f"eval spec {path} must be a JSON object")
    version = raw.get("version")
    if version not in SUPPORTED_SPEC_VERSIONS:
        supported = ", ".join(str(v) for v in SUPPORTED_SPEC_VERSIONS)
        raise SpecError(
            f"eval spec {path} has unsupported version {version!r}; "
            f"supported: {supported}"
        )
    presets = raw.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise SpecError(f"eval spec {path} must define a non-empty 'presets' object")
    return raw


def resolve_run(
    spec_path: str | Path,
    preset_name: str,
    *,
    overrides: dict[str, Any] | None = None,
) -> ResolvedRun:
    """Resolve ``preset_name`` in ``spec_path`` into a :class:`ResolvedRun`.

    Merge order (last wins): spec ``defaults`` -> the named preset -> ``overrides``
    (values a CLI flag was explicitly given). Relative file paths in the spec are
    resolved against the spec file's directory; ``overrides`` paths are taken
    as-is (the caller expands them).
    """
    path = Path(spec_path).expanduser().resolve()
    spec = load_spec(path)
    presets = spec["presets"]
    if preset_name not in presets:
        available = ", ".join(sorted(presets)) or "(none)"
        raise SpecError(
            f"unknown preset {preset_name!r} in {path}. Available: {available}"
        )
    preset = presets[preset_name]
    if not isinstance(preset, dict):
        raise SpecError(f"preset {preset_name!r} in {path} must be an object")

    defaults = spec.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SpecError(f"'defaults' in {path} must be an object")
    merged: dict[str, Any] = {**defaults, **preset}

    spec_dir = path.parent

    def _abs(value: Any) -> str | None:
        if not value:
            return None
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = spec_dir / candidate
        return str(candidate)

    cache_cfg = merged.get("cache") or {}
    if not isinstance(cache_cfg, dict):
        raise SpecError(
            f"preset {preset_name!r} 'cache' in {path} must be an object"
        )
    cache_mode = str(cache_cfg.get("mode", "off")).strip().lower()
    if cache_mode not in CACHE_MODES:
        supported = ", ".join(CACHE_MODES)
        raise SpecError(
            f"preset {preset_name!r} cache.mode {cache_mode!r} in {path} "
            f"is invalid; supported: {supported}"
        )

    gate_cfg = merged.get("gate")
    gate: GateThresholds | None = None
    if gate_cfg is not None:
        if not isinstance(gate_cfg, dict):
            raise SpecError(
                f"preset {preset_name!r} 'gate' in {path} must be an object"
            )
        try:
            gate = GateThresholds(
                hit_floor=float(gate_cfg["hit_floor"]),
                passed_baseline=int(gate_cfg["passed_baseline"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SpecError(
                f"preset {preset_name!r} 'gate' in {path} needs numeric "
                f"'hit_floor' and 'passed_baseline': {exc}"
            ) from exc

    resolved = ResolvedRun(
        spec_path=path,
        preset=preset_name,
        pack_filename=str(merged.get("pack", "eval_pack.json")),
        agent_runner_name=str(merged.get("agent_runner", "scripted")),
        scorer=(str(merged["scorer"]) if merged.get("scorer") else None),
        model=(str(merged["model"]) if merged.get("model") else None),
        provider_profile=(
            str(merged["provider_profile"]) if merged.get("provider_profile") else None
        ),
        fixture_match=str(merged.get("fixture_match", "args_then_order")),
        samples=int(merged.get("samples", 1)),
        judge_grounding=bool(merged.get("judge_grounding", False)),
        judge_votes=int(merged.get("judge_votes", 3)),
        system_prompt_file=_abs(merged.get("system_prompt_file")),
        histories_file=_abs(merged.get("histories_file")),
        rubrics_file=_abs(merged.get("rubrics_file")),
        grounding_mode=str(merged.get("grounding_mode", "process")),
        live_skill=bool(merged.get("live_skill", True)),
        cache_completions=_abs(cache_cfg.get("completions")),
        cache_mode=cache_mode,
        cache_prune_to=_abs(cache_cfg.get("prune_to")),
        gate=gate,
    )

    if overrides:
        unknown = set(overrides) - set(OVERRIDABLE_FIELDS)
        if unknown:
            raise SpecError(
                f"cannot override non-preset fields: {', '.join(sorted(unknown))}"
            )
        resolved = replace(resolved, **overrides)

    _validate_cache(resolved)
    return resolved


def _validate_cache(run: ResolvedRun) -> None:
    if run.cache_mode in ("record", "strict") and not run.cache_completions:
        raise SpecError(
            f"preset {run.preset!r} cache.mode={run.cache_mode} needs a cache file "
            "(preset cache.completions or --cache-completions)"
        )


def evaluate_gate(
    thresholds: GateThresholds,
    *,
    hit_rate: float,
    hits: int,
    misses: int,
    passed: int,
    total: int,
) -> GateResult:
    """Apply a gate's thresholds to observed report metrics.

    Two independent bars (mirrors the legacy ``check_inner_eval.py``):

    * ``hit_floor`` — the completion cache must still replay. A drop means the
      agent system prompt, a tool schema, or the model changed, so the recording
      is stale (re-record + re-commit ``completions.json`` / ``system_prompt.txt``).
    * ``passed_baseline`` — no scored regression on the cases that reproduce.
    """
    reasons: list[str] = []
    if hit_rate < thresholds.hit_floor:
        reasons.append(
            f"cache hit-rate {hit_rate:.4f} < floor {thresholds.hit_floor:.4f} — "
            "prompt/tool-schema/model changed; re-record the completion cache "
            "(see the bundle README)."
        )
    if passed < thresholds.passed_baseline:
        reasons.append(
            f"passed {passed} < baseline {thresholds.passed_baseline} — "
            "scored regression."
        )
    return GateResult(
        ok=not reasons,
        hit_rate=hit_rate,
        hits=hits,
        misses=misses,
        passed=passed,
        total=total,
        thresholds=thresholds,
        reasons=tuple(reasons),
    )
