from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunDir:
    root: Path
    config_snapshot: Path
    meta: Path
    metrics: Path
    artifacts: Path


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.STDOUT,
        )
        return out.decode().strip()
    except Exception:
        return "nogit"


def make_run_dir(tag: str, root: str | None = None) -> RunDir:
    """
    Create a run directory.

    - If `root` is provided (e.g., "outputs/phase2"), runs are created under that directory.
    - Otherwise, runs are created under "outputs/runs" (Phase-1 default behavior).

    The run directory name keeps the original "{timestamp}_{tag}_{git|nogit}" pattern.
    """
    base = Path(root) if root else (Path("outputs") / "runs")
    base.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = base / f"{stamp}_{tag}_{_git_commit()}"
    run_root.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_root / "config.yaml"
    meta = run_root / "meta.json"
    metrics = run_root / "metrics.json"
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    return RunDir(
        root=run_root,
        config_snapshot=config_snapshot,
        meta=meta,
        metrics=metrics,
        artifacts=artifacts,
    )


def write_meta(run: RunDir, extra: dict | None = None) -> None:
    import sys

    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    if extra:
        meta.update(extra)
    run.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
