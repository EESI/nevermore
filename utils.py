from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def ensure_repo_root_on_path(repo_root: Path) -> None:
    """Add repo_root to sys.path once so local Firm-DTI modules resolve."""
    repo_root = Path(repo_root).resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def dump_json(data: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
