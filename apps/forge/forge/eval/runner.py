"""Gold-set eval runner.

Reads ``apps/forge/forge/eval/golds/forge_tasks.jsonl``, runs each task
through a fresh ``ForgeEngine`` (per-task ``FORGE_REPO`` if a fixture is
declared), scores the answer via the inlined task-success rubric, prints
pass/fail + an overall success rate. CSV side-effect at
``<repo>/.forge/eval_results/<timestamp>.csv``.

Single topology only. Earlier versions of Forge had a ``solo`` /
``supervisor`` split and this runner used to sweep across both; we
collapsed everything into one main agent + delegation tools, so each
task is now scored once. The ``topology`` field on ``EvalRow`` and
``TaskResult`` is kept ("main") for CSV/back-compat but is no longer
printed in the CLI output.

End-of-turn reflection is disabled in eval — the per-task workspace is
a temp dir that gets rmtree'd at the end of the task, so reflection
has nowhere durable to write, and a background reflection task can
race with that cleanup.

Public surface:

    run_eval_cli(paths, cfg, *, limit=None) -> int   # used by `forge eval`
    run_eval(paths, cfg, *, limit=None) -> list[EvalRow]
"""
from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared import judge_with_rubric

from ..config import ForgeConfig, ensure_config
from ..paths import ForgePaths


GOLDS_PATH = Path(__file__).resolve().parent / "golds" / "forge_tasks.jsonl"

# Task-success rubric — uses ``shared.judge_with_rubric`` (the W1 judge).
# ``success_criteria`` is treated as the reference answer; the judge scores
# the agent's final message 0-5; ``score >= 4`` is the cut-off for "pass"
# (the same cut-off used in the bake-off notebook). We do NOT optimize the
# judge for Forge's output style — same teaching trade-off as the W1 sweep.
_PASS_THRESHOLD = 4


@dataclass
class TaskScore:
    score: float
    passed: bool
    rationale: str = ""


def score_task(
    *, task: str, success_criteria: str, answer: str,
    evidence: list[str] | None = None,
    judge_model: str | None = None,
) -> TaskScore:
    """Run the stock rubric judge over a Forge task.

    ``judge_model`` defaults to the judge in the W1 module (Claude Opus 4.7);
    passing a slug here swaps it.

    ``evidence`` — optional list of "what actually happened" snippets
    (post-task file contents, observed tool calls). When supplied, the
    W1 rubric judge grounds its faithfulness score in this evidence
    instead of taking the agent's answer at face value. Necessary for
    edit/tool-call tasks where the agent could say "Done!" without
    actually doing the work, or do the work without describing it.
    """
    judge_llm = None
    if judge_model:
        from shared import get_llm
        judge_llm = get_llm(judge_model)
    rv = judge_with_rubric(
        question=task,
        reference=success_criteria,
        answer=answer,
        evidence=evidence,
        judge_llm=judge_llm,
    )
    overall = float(getattr(rv, "overall", 0) or 0)
    rationale = str(getattr(rv, "notes", "") or getattr(rv, "rationale", "") or "")
    return TaskScore(
        score=overall,
        passed=overall >= _PASS_THRESHOLD,
        rationale=rationale,
    )


@dataclass
class GoldTask:
    id: str
    task: str
    success_criteria: str
    bucket: str = "misc"
    difficulty: int = 1
    blast_radius: str = "low"   # low | med | high
    expected_plan_required: bool = False
    fixture: str | None = None   # repo-relative path to a fixture dir
    files: dict[str, str] | None = None  # inline fixture: {path: content}


@dataclass
class EvalRow:
    task_id: str
    bucket: str
    topology: str
    planned: bool
    answer: str
    score: float
    passed: bool
    rationale: str
    elapsed_s: float
    error: str = ""


def _load_golds() -> list[GoldTask]:
    if not GOLDS_PATH.is_file():
        return []
    out: list[GoldTask] = []
    with GOLDS_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            out.append(GoldTask(**data))
    return out


def _materialise_fixture(task: GoldTask, base_paths: ForgePaths) -> Path:
    """Create a temp dir holding the fixture (file copy or inline writes) and
    return that path. Caller is responsible for cleanup."""
    if task.fixture:
        src = (Path(__file__).resolve().parent / "golds" / task.fixture).resolve()
        if not src.is_dir():
            raise FileNotFoundError(f"fixture missing: {src}")
        dst = Path(tempfile.mkdtemp(prefix=f"forge-eval-{task.id}-"))
        for child in src.iterdir():
            target = dst / child.name
            if child.is_dir():
                shutil.copytree(child, target)
            else:
                shutil.copy2(child, target)
        return dst
    if task.files:
        dst = Path(tempfile.mkdtemp(prefix=f"forge-eval-{task.id}-"))
        for rel, content in task.files.items():
            p = dst / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return dst
    # No fixture — run against the *current* repo. We still create an
    # isolated .forge dir under a temp prefix so the eval doesn't pollute
    # the user's checkpoints / audit log.
    dst = Path(tempfile.mkdtemp(prefix=f"forge-eval-{task.id}-"))
    return dst


async def _auto_approver(*, tool_name, args, agent_name, reason) -> bool:
    """Eval is headless — auto-approve every gate so tasks can execute. The
    audit log still captures the decision + the "human approved" source so
    we can review after the fact."""
    return True


def _collect_evidence(
    *, task: GoldTask, work_dir: Path, trace_path: Path,
    max_file_bytes: int = 4096, max_tool_calls: int = 40,
) -> list[str]:
    """Snapshot what the agent actually *did* so the judge can verify
    success_criteria against ground truth instead of the answer string.

    Returns one entry per fixture file (post-task contents) plus a
    single "TOOL CALLS" summary. Truncates aggressively — the judge
    just needs enough signal to grade, not a full audit log.
    """
    out: list[str] = []
    # Post-task contents of every file we know about (initial fixture).
    # Walk the work_dir for these explicit paths only — we deliberately
    # skip new files outside the fixture to keep evidence focused and
    # avoid noise from .forge/, .git/, RAG index, etc.
    fixture_paths: list[str] = list(task.files.keys()) if task.files else []
    for rel in fixture_paths:
        p = work_dir / rel
        if not p.exists():
            out.append(f"FILE {rel} (after task): <deleted>")
            continue
        try:
            data = p.read_bytes()[: max_file_bytes + 1]
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            out.append(f"FILE {rel} (after task): <read error: {exc!r}>")
            continue
        truncated = " [...truncated]" if len(data) > max_file_bytes else ""
        out.append(f"FILE {rel} (after task):\n{text}{truncated}")

    # Tool-call sequence from trace.jsonl. We pull main-agent tool calls
    # so the judge can verify e.g. "agent called semantic_write" without
    # having to ask the agent to recite its trajectory.
    if trace_path.is_file():
        try:
            calls: list[str] = []
            with trace_path.open(encoding="utf-8") as fh:
                for line in fh:
                    try:
                        ev = json.loads(line)
                    except Exception:  # noqa: BLE001
                        continue
                    if ev.get("type") != "tool_call":
                        continue
                    # Skip reflector — it's disabled in eval anyway,
                    # but belt-and-suspenders for old traces.
                    if ev.get("agent_name") == "reflector":
                        continue
                    tool = ev.get("tool") or "<?>"
                    args = ev.get("args") or {}
                    args_preview = json.dumps(args, default=str, ensure_ascii=False)
                    if len(args_preview) > 200:
                        args_preview = args_preview[:200] + "..."
                    calls.append(f"  - {tool}({args_preview})")
                    if len(calls) >= max_tool_calls:
                        calls.append(f"  - ... (truncated at {max_tool_calls})")
                        break
            if calls:
                out.append("TOOL CALLS (in order):\n" + "\n".join(calls))
            else:
                out.append("TOOL CALLS: <none>")
        except Exception as exc:  # noqa: BLE001
            out.append(f"TOOL CALLS: <trace read error: {exc!r}>")

    return out


async def _run_one(
    task: GoldTask, base_paths: ForgePaths, base_cfg: ForgeConfig,
) -> EvalRow:
    from ..agent.engine import ForgeEngine  # late import: keeps `forge eval` cheap if there's a deep error
    from ..repo_rag import build_index

    work_dir = _materialise_fixture(task, base_paths)
    # Pin the engine to the fixture dir via FORGE_REPO; reuse the user's
    # config for model choices so eval reflects their settings.
    saved = os.environ.get("FORGE_REPO")
    os.environ["FORGE_REPO"] = str(work_dir)
    paths = ForgePaths.for_repo(work_dir)
    paths.ensure()
    cfg, _ = ensure_config(paths)
    # Copy model + compaction + permissions + memory settings from the
    # user's main config so a single ``forge eval`` reflects their config
    # choices.
    cfg.models = base_cfg.models
    cfg.compaction = base_cfg.compaction
    cfg.permissions = base_cfg.permissions
    # Per-task workspaces are temp dirs that get rmtree'd at the end of
    # _run_one — reflection has nowhere durable to write and the
    # background task can race with the cleanup (FileNotFoundError on
    # trace.jsonl). Disable it for eval; we're scoring the main agent's
    # answer, not its memory write-back.
    cfg.memory = base_cfg.memory.model_copy(update={"reflect_on_thread_end": False})

    # Build the RAG index over the fixture so repo_rag_hybrid_retrieve works.
    # Cheap on a small fixture; ignored on first-call errors.
    try:
        await asyncio.to_thread(build_index, paths=paths, cfg=cfg.repo_rag, force=True)
    except Exception:  # noqa: BLE001
        pass

    started = datetime.now(timezone.utc)
    error = ""
    answer = ""
    topology = "main"
    planned = False
    engine = None
    evidence: list[str] = []
    try:
        engine = await ForgeEngine.start(
            paths=paths, cfg=cfg, approver=_auto_approver,
        )
        result = await engine.run_task(task.task)
        answer = result.answer
        topology = result.topology
        planned = result.planned
        # Snapshot post-task fixture state + tool-call trace BEFORE
        # the finally block rmtree's the work_dir. The judge uses
        # this as ground-truth evidence for grading.
        try:
            evidence = _collect_evidence(
                task=task, work_dir=work_dir, trace_path=paths.trace_jsonl,
            )
        except Exception:  # noqa: BLE001
            evidence = []
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if engine is not None:
            try:
                await engine.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if saved is None:
            os.environ.pop("FORGE_REPO", None)
        else:
            os.environ["FORGE_REPO"] = saved
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    if error:
        return EvalRow(
            task_id=task.id, bucket=task.bucket, topology=topology,
            planned=planned, answer="", score=0.0, passed=False,
            rationale="", elapsed_s=elapsed, error=error,
        )
    try:
        scored: TaskScore = await asyncio.to_thread(
            score_task,
            task=task.task, success_criteria=task.success_criteria, answer=answer,
            evidence=evidence or None,
            judge_model=base_cfg.models.judge,
        )
    except Exception as exc:  # noqa: BLE001
        return EvalRow(
            task_id=task.id, bucket=task.bucket, topology=topology,
            planned=planned, answer=answer, score=0.0, passed=False,
            rationale="", elapsed_s=elapsed, error=f"judge failed: {exc!r}",
        )
    return EvalRow(
        task_id=task.id, bucket=task.bucket, topology=topology,
        planned=planned, answer=answer, score=scored.score,
        passed=scored.passed, rationale=scored.rationale,
        elapsed_s=elapsed,
    )


async def run_eval(
    paths: ForgePaths, cfg: ForgeConfig, *, limit: int | None = None,
) -> list[EvalRow]:
    """Run the bundled gold set. Returns one ``EvalRow`` per task."""
    golds = _load_golds()
    if limit is not None:
        golds = golds[:limit]
    rows: list[EvalRow] = []
    for task in golds:
        row = await _run_one(task, paths, cfg)
        rows.append(row)
    return rows


def _write_csv(paths: ForgePaths, rows: list[EvalRow]) -> Path:
    paths.eval_results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = paths.eval_results_dir / f"eval-{stamp}.csv"
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0]).keys()) if rows else
                                ["task_id", "bucket", "topology", "planned", "answer",
                                 "score", "passed", "rationale", "elapsed_s", "error"])
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return out


def run_eval_cli(
    paths: ForgePaths, cfg: ForgeConfig, *, limit: int | None = None,
) -> int:
    """CLI entry. Returns 0 on >= 80% pass rate, 1 otherwise (for CI)."""
    try:
        rows = asyncio.run(run_eval(paths, cfg, limit=limit))
    except KeyboardInterrupt:
        return 130
    if not rows:
        print("forge eval: no tasks in gold set", file=sys.stderr)
        return 1
    n_pass = sum(1 for r in rows if r.passed)
    n = len(rows)
    out_path = _write_csv(paths, rows)
    print()
    for r in rows:
        mark = "PASS" if r.passed else ("ERR " if r.error else "FAIL")
        line = f"  [{mark}] {r.task_id:<28} score={r.score:.2f} t={r.elapsed_s:.1f}s"
        if r.error:
            line += f" err={r.error[:80]}"
        print(line)
    print()
    print(f"forge eval: {n_pass}/{n} passed ({100*n_pass/n:.1f}%). csv -> {out_path}")
    return 0 if n_pass >= 0.8 * n else 1


__all__ = ["GoldTask", "EvalRow", "run_eval", "run_eval_cli"]
