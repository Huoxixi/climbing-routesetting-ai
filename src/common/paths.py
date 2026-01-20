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
        out = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], stderr=subprocess.STDOUT)
        return out.decode().strip()
    except Exception:
        return 'nogit'

def make_run_dir(tag: str) -> RunDir:
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    commit = _git_commit()
    root = Path('outputs') / 'runs' / f'{ts}_{tag}_{commit}'
    root.mkdir(parents=True, exist_ok=True)

    artifacts = root / 'artifacts'
    artifacts.mkdir(parents=True, exist_ok=True)

    return RunDir(
        root=root,
        config_snapshot=root / 'config.yaml',
        meta=root / 'meta.json',
        metrics=root / 'metrics.json',
        artifacts=artifacts,
    )

def write_meta(run: RunDir, extra: dict | None = None) -> None:
    import sys
    meta = {
        'python': sys.version,
        'platform': platform.platform(),
        'git_commit': _git_commit(),
    }
    if extra:
        meta.update(extra)
    run.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
