from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from .utils import dump_json


def hash_file(path: Path, chunk_size: int = 1 << 20) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return f"{path.stat().st_size}:{h.hexdigest()}"


def normalize_outputs(outputs: Dict[str, Path]) -> Dict[str, Path]:
    return {k: Path(v).resolve() for k, v in outputs.items()}


@dataclass
class StepResult:
    name: str
    signature: str
    outputs: Dict[str, Path]
    details: Dict[str, Any]
    metadata_path: Path
    cached: bool = False


class CacheManager:
    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _signature_payload(self, name: str, payload: Dict[str, Any], files: Dict[str, Path]) -> str:
        file_hashes = {k: hash_file(v) for k, v in sorted(files.items())}
        serializable = {"step": name, "config": payload, "files": file_hashes}
        blob = json.dumps(serializable, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:12]

    def _step_dir(self, name: str, signature: str) -> Path:
        return self.output_root / name / signature

    def load_cached(self, name: str, signature: str) -> Optional[StepResult]:
        meta_path = self._step_dir(name, signature) / "metadata.json"
        if not meta_path.exists():
            return None
        data = json.loads(meta_path.read_text())
        outputs = normalize_outputs(data.get("outputs", {}))
        details = data.get("details", {})
        missing = [p for p in outputs.values() if not p.exists()]
        if missing:
            return None
        return StepResult(
            name=name, signature=signature, outputs=outputs, details=details, metadata_path=meta_path, cached=True
        )

    def store(
        self,
        name: str,
        signature: str,
        payload: Dict[str, Any],
        files: Dict[str, Path],
        outputs: Dict[str, Path],
        details: Dict[str, Any],
    ) -> StepResult:
        step_dir = self._step_dir(name, signature)
        step_dir.mkdir(parents=True, exist_ok=True)
        outputs = normalize_outputs(outputs)
        metadata = {
            "step": name,
            "signature": signature,
            "config": payload,
            "files": {k: str(Path(v)) for k, v in files.items()},
            "file_hashes": {k: hash_file(v) for k, v in files.items()},
            "outputs": {k: str(v) for k, v in outputs.items()},
            "details": details,
            "created": datetime.utcnow().isoformat() + "Z",
        }
        meta_path = step_dir / "metadata.json"
        dump_json(metadata, meta_path)
        return StepResult(
            name=name, signature=signature, outputs=outputs, details=details, metadata_path=meta_path, cached=False
        )

    def run_step(
        self,
        name: str,
        payload: Dict[str, Any],
        files: Dict[str, Path],
        runner: Callable[[Path], Tuple[Dict[str, Path], Dict[str, Any]]],
    ) -> StepResult:
        sig = self._signature_payload(name, payload, files)
        cached = self.load_cached(name, sig)
        if cached:
            return cached

        step_dir = self._step_dir(name, sig)
        outputs, details = runner(step_dir)
        return self.store(name, sig, payload, files, outputs, details)
