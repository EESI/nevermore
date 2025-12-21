from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class PathsConfig:
    repo_root: Optional[Path] = None
    output_root: Optional[Path] = None

    def finalize(self) -> "PathsConfig":
        repo = self.repo_root or _default_repo_root()
        out_root = self.output_root or (repo / "nevermore" / "outputs")
        return replace(self, repo_root=Path(repo).resolve(), output_root=Path(out_root).resolve())


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    optimization_db: str = "optimization.csv"
    retrieval_db: str = "retrieval.csv"

    def optimization_path(self, repo_root: Path) -> Path:
        base = Path(self.data_dir)
        if not base.is_absolute():
            base = Path(repo_root) / base
        return base / self.optimization_db

    def retrieval_path(self, repo_root: Path) -> Path:
        base = Path(self.data_dir)
        if not base.is_absolute():
            base = Path(repo_root) / base
        return base / self.retrieval_db


@dataclass
class FeatureConfig:
    protein_encoder: str = "esm"  # "handcrafted" | "esm"
    esm_model: str = "facebook/esm2_t12_35M_UR50D"
    max_token_length: int = 1500
    morgan_bits: int = 1024
    morgan_radius: int = 2
    skip_protein_features: bool = False
    checkpoint: Path = Path("./output/model_2/trainer2/Firm-D4-prj2/checkpoint.pt")
    batch_size: int = 64
    device: Optional[str] = None  # "cuda", "cpu", or None for auto
    admet_in_features: bool = False
    admet_keys: List[str] = field(
        default_factory=lambda: ["molecular_weight", "logP", "HIA_Hou", "hERG", "QED", "Lipinski"]
    )


@dataclass
class OptimizationConfig:
    target_affinity: float = 12.0
    sample_index: int = 10
    protein_features: int = 0
    ligand_features: int = 15
    allow_protein_adjustments: bool = False
    allow_ligand_adjustments: bool = True
    frozen_protein_features: List[int] = field(default_factory=list)
    frozen_ligand_features: List[int] = field(default_factory=list)
    budget: int = 300
    regularization: float = 0.001
    beta: float = 0.0  # optional extra weight (e.g., ADMET penalty)
    manifold_weight: float = 0.0  # weight for dataset-manifold proximity penalty
    target_sequence: Optional[str] = None
    baseline_smiles: Optional[str] = None
    # Optional ADMET penalties applied during optimization. Each entry: {"key": str, "min": float?, "max": float?, "weight": float}
    admet_constraints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RetrievalConfig:
    max_rel_smiles_len_diff: float = 0.30
    max_rel_mw_diff: float = 0.30
    max_l1_distance: Optional[float] = None
    top_candidates: int = 20


@dataclass
class VisualizationConfig:
    enabled: bool = True
    max_mols: Optional[int] = 40
    mols_per_row: int = 6
    panel_size: List[int] = field(default_factory=lambda: [300, 240])
    id_col: str = "dataset_index"
    out_svg: str = "candidate_ligands_grid.svg"
    out_png: str = "candidate_ligands_grid.png"


@dataclass
class DockingConfig:
    enabled: bool = False
    target_key: str = "3U84"
    receptor_pdbqt: Path = Path("../3U84.pdbqt")
    out_root: Optional[Path] = None
    exhaustiveness: int = 32
    num_modes: int = 20
    seed: int = 42
    limit: Optional[int] = None
    center: Optional[List[float]] = None
    size: List[float] = field(default_factory=lambda: [30.0, 30.0, 30.0])


@dataclass
class AdmetConfig:
    enabled: bool = False
    keys: List[str] = field(
        default_factory=lambda: ["molecular_weight", "logP", "HIA_Hou", "hERG", "QED", "Lipinski"]
    )
    batch_size: int = 256


@dataclass
class ReportConfig:
    enabled: bool = True


@dataclass
class PipelineConfig:
    run_name: str = "nevermore"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    admet: AdmetConfig = field(default_factory=AdmetConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def finalize(self) -> "PipelineConfig":
        finalized_paths = self.paths.finalize()
        resolved_checkpoint = (Path(finalized_paths.repo_root) / self.features.checkpoint).resolve()
        features = replace(self.features, checkpoint=resolved_checkpoint)
        # Expand relative data_dir against repo root
        data_dir = Path(self.data.data_dir)
        if not data_dir.is_absolute():
            data_dir = (finalized_paths.repo_root / data_dir).resolve()
        data = replace(self.data, data_dir=data_dir)
        # Docking output defaults to within output_root/ docking
        docking_out = self.docking.out_root
        if docking_out is None:
            docking_out = finalized_paths.output_root / "docking"
        docking_out = Path(docking_out)
        docking_receptor = Path(self.docking.receptor_pdbqt)
        if not docking_receptor.is_absolute():
            docking_receptor = (finalized_paths.repo_root / docking_receptor).resolve()
        docking = replace(self.docking, out_root=Path(docking_out).resolve(), receptor_pdbqt=docking_receptor)
        return replace(
            self,
            paths=finalized_paths,
            features=features,
            data=data,
            docking=docking,
        )

    def to_dict(self) -> Dict[str, Any]:
        def _convert(val):
            if isinstance(val, Path):
                return str(val)
            if isinstance(val, list):
                return [_convert(v) for v in val]
            if isinstance(val, dict):
                return {k: _convert(v) for k, v in val.items()}
            return val

        return {k: _convert(v) for k, v in asdict(self).items()}


def _update_dataclass(obj, updates: Dict[str, Any]):
    if updates is None:
        return obj
    kwargs = {}
    for f in fields(obj):
        if f.name not in updates:
            kwargs[f.name] = getattr(obj, f.name)
            continue
        val = updates[f.name]
        current = getattr(obj, f.name)
        if dataclass_isinstance(current):
            kwargs[f.name] = _update_dataclass(current, val or {})
        else:
            kwargs[f.name] = val
    return obj.__class__(**kwargs)


def dataclass_isinstance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def load_config(path: Optional[Path] = None) -> PipelineConfig:
    base = PipelineConfig()
    if path is None:
        return base.finalize()
    path = Path(path)
    user_cfg = yaml.safe_load(path.read_text()) if path.exists() else {}
    merged = _update_dataclass(base, user_cfg or {})
    return merged.finalize()
