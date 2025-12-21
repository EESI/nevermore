from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .cache import CacheManager, StepResult
from .config import PipelineConfig, load_config
from . import steps
from .utils import ensure_repo_root_on_path


STAGE_ORDER = [
    "ingest",
    "features",
    "optimization",
    "retrieval",
    "visualization",
    "docking",
    "report",
]


class NevermorePipeline:
    def __init__(self, config_path: Optional[Path] = None, config: Optional[PipelineConfig] = None):
        self.config = config or load_config(config_path)
        ensure_repo_root_on_path(self.config.paths.repo_root)
        self.cache = CacheManager(self.config.paths.output_root)
        self.results: Dict[str, StepResult] = {}
        self._verbose: bool = True

    def run(self, up_to: str = "report", verbose: bool = True) -> Dict[str, StepResult]:
        if up_to not in STAGE_ORDER:
            raise ValueError(f"Unknown stage '{up_to}'. Valid stages: {STAGE_ORDER}")
        self._verbose = verbose
        for stage in STAGE_ORDER:
            if verbose:
                print(f"[{stage}] starting...")
            runner = getattr(self, f"_run_{stage}")
            self.results[stage] = runner()
            if verbose:
                res = self.results[stage]
                state = "cached" if getattr(res, "cached", False) else "done"
                print(f"[{stage}] {state} (sig={res.signature})")
            if stage == up_to:
                break
        return self.results

    # ------------------------- stage runners ------------------------- #
    def _run_ingest(self) -> StepResult:
        cfg = self.config.data
        repo_root = self.config.paths.repo_root
        payload = {
            "data_dir": str(cfg.data_dir),
            "optimization_db": cfg.optimization_db,
            "retrieval_db": cfg.retrieval_db,
        }
        files = {"optimization": cfg.optimization_path(repo_root), "retrieval": cfg.retrieval_path(repo_root)}
        return self.cache.run_step(
            "ingest",
            payload,
            files,
            lambda step_dir: steps.ingest_data(cfg, repo_root, step_dir, verbose=self._verbose),
        )

    def _run_features(self) -> StepResult:
        ingest_res = self.results.get("ingest")
        if ingest_res is None:
            raise RuntimeError("Run ingest before features")
        cfg = self.config.features
        payload = {
            "protein_encoder": cfg.protein_encoder,
            "esm_model": cfg.esm_model,
            "max_token_length": cfg.max_token_length,
            "skip_protein_features": getattr(cfg, "skip_protein_features", False),
            "morgan_bits": cfg.morgan_bits,
            "morgan_radius": cfg.morgan_radius,
            "admet_in_features": getattr(cfg, "admet_in_features", False),
            "admet_keys": getattr(cfg, "admet_keys", []),
        }
        files = {
            "raw_optimization": ingest_res.outputs["raw_optimization"],
            "raw_retrieval": ingest_res.outputs["raw_retrieval"],
        }

        def _run_features_dual(step_dir: Path):
            # Optimization features (with optional ADMET)
            opt_dir = step_dir / "optimization"
            opt_outputs, opt_details = steps.build_features(
                cfg, opt_dir, ingest_res.outputs["raw_optimization"], verbose=self._verbose
            )
            # Retrieval features (no ADMET to avoid extra work unless enabled)
            ret_cfg = cfg
            ret_dir = step_dir / "retrieval"
            ret_outputs, ret_details = steps.build_features(
                ret_cfg, ret_dir, ingest_res.outputs["raw_retrieval"], verbose=self._verbose
            )
            # Prefix keys
            outputs = {f"opt_{k}": v for k, v in opt_outputs.items()} | {f"ret_{k}": v for k, v in ret_outputs.items()}
            details = {"optimization": opt_details, "retrieval": ret_details}
            return outputs, details

        return self.cache.run_step("features", payload, files, _run_features_dual)

    def _run_optimization(self) -> StepResult:
        feat_res = self.results.get("features")
        if feat_res is None:
            raise RuntimeError("Run features before optimization")
        opt_cfg = self.config.optimization
        feat_cfg = self.config.features
        payload = {
            "target_affinity": opt_cfg.target_affinity,
            "sample_index": opt_cfg.sample_index,
            "protein_features": opt_cfg.protein_features,
            "ligand_features": opt_cfg.ligand_features,
            "allow_protein_adjustments": opt_cfg.allow_protein_adjustments,
            "allow_ligand_adjustments": opt_cfg.allow_ligand_adjustments,
            "frozen_protein_features": opt_cfg.frozen_protein_features,
            "frozen_ligand_features": opt_cfg.frozen_ligand_features,
            "budget": opt_cfg.budget,
            "regularization": opt_cfg.regularization,
            "beta": getattr(opt_cfg, "beta", 0.0),
            "manifold_weight": opt_cfg.manifold_weight,
            "target_sequence": opt_cfg.target_sequence,
            "baseline_smiles": opt_cfg.baseline_smiles,
            "admet_constraints": opt_cfg.admet_constraints,
            "skip_protein_features": getattr(feat_cfg, "skip_protein_features", False),
        }
        files = {
            "processed": feat_res.outputs["opt_processed"],
            "protein": feat_res.outputs["opt_protein"],
            "ligand": feat_res.outputs["opt_ligand"],
            "checkpoint": feat_cfg.checkpoint,
        }
        if "opt_admet" in feat_res.outputs:
            files["admet"] = feat_res.outputs["opt_admet"]
        return self.cache.run_step(
            "optimization",
            payload,
            files,
            lambda step_dir: steps.run_optimization(
                opt_cfg,
                feat_cfg,
                step_dir,
                feat_res.outputs["opt_processed"],
                feat_res.outputs["opt_protein"],
                feat_res.outputs["opt_ligand"],
                feat_res.outputs.get("opt_admet"),
            ),
        )

    def _run_retrieval(self) -> StepResult:
        feat_res = self.results.get("features")
        opt_res = self.results.get("optimization")
        if feat_res is None or opt_res is None:
            raise RuntimeError("Run features + optimization before retrieval")
        cfg = self.config.retrieval
        payload = {
            "max_rel_smiles_len_diff": cfg.max_rel_smiles_len_diff,
            "max_rel_mw_diff": cfg.max_rel_mw_diff,
            "max_l1_distance": cfg.max_l1_distance,
            "top_candidates": cfg.top_candidates,
        } | {
            "ligand_indices": opt_res.details.get("ligand_indices"),
            "target_counts": opt_res.details.get("target_counts"),
            "baseline_smiles": opt_res.details.get("baseline_smiles"),
        }
        files = {"processed": feat_res.outputs["ret_processed"], "ligand": feat_res.outputs["ret_ligand"]}
        return self.cache.run_step(
            "retrieval",
            payload,
            files,
            lambda step_dir: steps.retrieve_candidates(
                cfg, step_dir, feat_res.outputs["ret_processed"], feat_res.outputs["ret_ligand"], opt_res.details
            ),
        )

    def _run_visualization(self) -> StepResult:
        retrieval_res = self.results.get("retrieval")
        if retrieval_res is None:
            raise RuntimeError("Run retrieval before visualization")
        cfg = self.config.visualization
        candidates = retrieval_res.outputs.get("candidates")
        payload = {"enabled": cfg.enabled, "max_mols": cfg.max_mols, "mols_per_row": cfg.mols_per_row}
        files = {"candidates": candidates} if candidates else {}
        return self.cache.run_step(
            "visualization",
            payload,
            files,
            lambda step_dir: steps.visualize_candidates(cfg, step_dir, candidates) if candidates else ({}, {}),
        )

    def _run_docking(self) -> StepResult:
        retrieval_res = self.results.get("retrieval")
        if retrieval_res is None:
            raise RuntimeError("Run retrieval before docking")
        cfg = self.config.docking
        candidates = retrieval_res.outputs.get("candidates")
        payload = {
            "enabled": cfg.enabled,
            "target_key": cfg.target_key,
            "receptor_pdbqt": str(cfg.receptor_pdbqt),
            "center": cfg.center,
            "size": cfg.size,
            "limit": cfg.limit,
            "baseline_index": retrieval_res.details.get("baseline_index") if retrieval_res else None,
            "baseline_smiles": retrieval_res.details.get("baseline_smiles") if retrieval_res else None,
        }
        files = {"candidates": candidates} if candidates else {}
        return self.cache.run_step(
            "docking",
            payload,
            files,
            lambda step_dir: steps.dock_candidates(
                cfg,
                step_dir,
                candidates,
                retrieval_res.details.get("baseline_smiles") if retrieval_res else None,
                retrieval_res.details.get("baseline_index") if retrieval_res else None,
            )
            if candidates
            else ({}, {}),
        )

    def _run_admet(self) -> StepResult:
        retrieval_res = self.results.get("retrieval")
        if retrieval_res is None:
            raise RuntimeError("Run retrieval before admet")
        cfg = self.config.admet
        candidates = retrieval_res.outputs.get("candidates")
        payload = {"enabled": cfg.enabled, "keys": cfg.keys, "batch_size": getattr(cfg, "batch_size", 256)}
        files = {"candidates": candidates} if candidates else {}
        return self.cache.run_step(
            "admet",
            payload,
            files,
            lambda step_dir: steps.admet_predictions(cfg, step_dir, candidates) if candidates else ({}, {}),
        )

    def _run_report(self) -> StepResult:
        retrieval_res = self.results.get("retrieval")
        if retrieval_res is None:
            raise RuntimeError("Run retrieval before report")
        candidates = retrieval_res.outputs.get("candidates")
        opt_res = self.results.get("optimization")
        feat_res = self.results.get("features")
        admet_csv = feat_res.outputs.get("ret_admet") if feat_res else None
        docking_res = self.results.get("docking")
        docking_csv = docking_res.outputs.get("docking") if docking_res else None
        payload = {
            "enabled": self.config.report.enabled,
            "baseline_index": retrieval_res.details.get("baseline_index") if retrieval_res else None,
            "baseline_smiles": retrieval_res.details.get("baseline_smiles") if retrieval_res else None,
            "optimization_signature": opt_res.signature if opt_res else None,
        }
        files = {
            k: v
            for k, v in {
                "candidates": candidates,
                "docking": docking_csv,
                "admet": admet_csv,
                "processed": feat_res.outputs.get("ret_processed") if feat_res else None,
                "protein": feat_res.outputs.get("ret_protein") if feat_res else None,
                "ligand": feat_res.outputs.get("ret_ligand") if feat_res else None,
                "checkpoint": self.config.features.checkpoint,
            }.items()
            if v
        }
        return self.cache.run_step(
            "report",
            payload,
            files,
            lambda step_dir: steps.build_report(
                step_dir,
                candidates,
                docking_csv,
                admet_csv,
                retrieval_res.details.get("baseline_index") if retrieval_res else None,
                retrieval_res.details.get("baseline_smiles") if retrieval_res else None,
                feat_res.outputs.get("ret_processed") if feat_res else None,
                feat_res.outputs.get("ret_protein") if feat_res else None,
                feat_res.outputs.get("ret_ligand") if feat_res else None,
                self.config.features,
                opt_res.details if opt_res else None,
            )
            if candidates
            else ({}, {}),
        )
