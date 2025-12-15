from __future__ import annotations

import io
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from contextlib import redirect_stdout, redirect_stderr
from torch.cuda.amp import autocast
from tqdm import tqdm

from .config import (
    AdmetConfig,
    DataConfig,
    DockingConfig,
    FeatureConfig,
    OptimizationConfig,
    RetrievalConfig,
    VisualizationConfig,
)
from .Firm_dti import ligand_feature_vector, protein_feature_vector, is_kekulizable


StepOutputs = Tuple[Dict[str, Path], Dict[str, Any]]


# Suppress noisy libraries that print to stdout/stderr even in non-verbose mode.
def _silent_run(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


# Torch checkpoints may reference original module names ("model", "model2").
# Install aliases that point to the bundled Firm-DTI model so torch.load can resolve them.
def _alias_firm_dti_modules() -> None:
    try:
        import importlib

        from . import Firm_dti

        sys.modules.setdefault("model", importlib.import_module(".Firm_dti.model", __package__))
        sys.modules.setdefault("model2", importlib.import_module(".Firm_dti.model", __package__))
    except Exception:
        # If aliasing fails, fall through and let torch.load raise a clear error.
        pass


# ------------------------------ ingest ------------------------------ #
def ingest_data(cfg: DataConfig, repo_root: Path, step_dir: Path, verbose: bool = False) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    train_path = cfg.train_path(repo_root)
    test_path = cfg.test_path(repo_root)
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    test_df["split"] = "test"
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined_path = step_dir / "raw_combined.csv"
    combined.to_csv(combined_path, index=False)
    details = {
        "rows": len(combined),
        "columns": combined.columns.tolist(),
        "split_counts": {"train": len(train_df), "test": len(test_df)},
    }
    if verbose:
        print(f"[ingest] combined {len(combined)} rows (train={len(train_df)}, test={len(test_df)}) -> {combined_path}")
    return {"raw_combined": combined_path}, details


# ------------------------------ features ------------------------------ #
def build_features(cfg: FeatureConfig, step_dir: Path, raw_csv: Path, verbose: bool = False) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    try:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
    except Exception:
        pass
    df = pd.read_csv(raw_csv)
    filtered = df[df["Drug"].apply(is_kekulizable)].reset_index(drop=True).copy()
    filtered["dataset_index"] = np.arange(len(filtered))
    filtered["smiles"] = filtered["Drug"]

    protein_feat_list = []
    ligand_feat_list = []
    iterable = filtered.itertuples(index=False)
    if verbose:
        iterable = tqdm(iterable, total=len(filtered), desc="[features] featurizing")

    use_esm = cfg.protein_encoder.lower() == "esm"
    tokenizer = None
    max_len = None
    if use_esm:
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise RuntimeError("Transformers is required for ESM tokenization") from e
        tokenizer = AutoTokenizer.from_pretrained(cfg.esm_model)
        max_len = min(cfg.max_token_length, tokenizer.model_max_length)

    for row in iterable:
        if use_esm:
            tokens = tokenizer(
                row.Target,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )["input_ids"].squeeze(0).numpy()
            protein_feat_list.append(tokens.astype(np.int64, copy=False))
        else:
            protein_feat_list.append(protein_feature_vector(row.Target))
        ligand_feat_list.append(ligand_feature_vector(row.Drug, cfg.morgan_bits, cfg.morgan_radius))

    protein_mat = np.stack(protein_feat_list)
    ligand_mat = np.stack(ligand_feat_list)

    admet_path: Optional[Path] = None
    admet_rows = 0
    if getattr(cfg, "admet_in_features", False) and len(filtered):
        try:
            from admet_ai import ADMETModel
        except ImportError as e:
            raise RuntimeError("admet-ai is required for ADMET feature computation") from e

        model = _silent_run(ADMETModel)
        smiles_list = filtered["smiles"].astype(str).tolist()
        preds = _silent_run(model.predict, smiles_list)
        if isinstance(preds, pd.DataFrame):
            pred_df = preds.reset_index(drop=True)
        elif isinstance(preds, (list, tuple)):
            pred_df = pd.DataFrame(preds)
        else:
            raise TypeError(f"Unexpected ADMET predictions type: {type(preds)}")
        if len(pred_df) != len(filtered):
            raise RuntimeError(f"ADMET prediction count mismatch: got {len(pred_df)} vs {len(filtered)} rows")

        admet_cols = list(getattr(cfg, "admet_keys", []) or pred_df.columns.tolist())
        for col in admet_cols:
            filtered[col] = pred_df.get(col)
        admet_export = pd.concat([filtered[["dataset_index", "smiles"]], pred_df[admet_cols]], axis=1)
        admet_path = step_dir / "admet_features.csv"
        admet_export.to_csv(admet_path, index=False)
        admet_rows = len(admet_export)

    processed_path = step_dir / "processed.csv"
    protein_path = step_dir / "protein.npy"
    ligand_path = step_dir / "ligand.npy"
    filtered.to_csv(processed_path, index=False)
    np.save(protein_path, protein_mat)
    np.save(ligand_path, ligand_mat)

    details = {
        "kept_rows": len(filtered),
        "dropped_rows": len(df) - len(filtered),
        "protein_dim": int(protein_mat.shape[1]),
        "ligand_dim": int(ligand_mat.shape[1]),
        "protein_encoder": cfg.protein_encoder,
    }
    if admet_path:
        details["admet_rows"] = admet_rows
        details["admet_keys"] = admet_cols
    if verbose:
        print(f"[features] wrote {processed_path}, protein.npy, ligand.npy (kept {len(filtered)} rows)")
        if admet_path:
            print(f"[features] ADMET predictions -> {admet_path}")
    outputs = {"processed": processed_path, "protein": protein_path, "ligand": ligand_path}
    if admet_path:
        outputs["admet"] = admet_path
    return outputs, details


# ------------------------------ optimization ------------------------------ #
def _load_model(checkpoint: Path, protein_dim: int, ligand_dim: int, device: torch.device, feat_cfg: FeatureConfig):
    from .Firm_dti import FeatureTuner, EsmMorganTuner

    _alias_firm_dti_modules()
    # Explicitly allow loading pickled modules (old checkpoints) by opting out of weights_only=True default.
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    use_esm = feat_cfg.protein_encoder.lower() == "esm"

    if isinstance(state, dict) and "Tuner" in state:
        model = state["Tuner"]
    else:
        state_dict = state.get("model_state_dict", state)
        if use_esm:
            model = EsmMorganTuner(ligand_dim=ligand_dim, pretrained_model=feat_cfg.esm_model)
        else:
            hidden_dim = state_dict["protein_proj.weight"].shape[0]
            rbf_k = state_dict["aff_head.out.weight"].shape[1]
            model = FeatureTuner(protein_dim=protein_dim, ligand_dim=ligand_dim, hidden_dim=hidden_dim, rbf_k=rbf_k)
        model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def _select_indices(std_array: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=int)
    return np.argsort(std_array)[-count:]




def run_optimization(
    opt_cfg: "OptimizationConfig",
    feat_cfg: "FeatureConfig",
    step_dir: Path,
    processed_csv: Path,
    protein_path: Path,
    ligand_path: Path,
    admet_path: Optional[Path] = None,
) -> "StepOutputs":
    """
    Projection-in-loop optimization:
      Nevergrad edits a "bucket" fingerprint -> we project it to top-K REAL fingerprints
      -> score model on those K, then return an objective to Nevergrad.

    This version:
      - evaluates ALL top-K neighbors each step
      - logs best/avg/softmin over top-K to a separate CSV
      - optimizes a smoother objective (softmin over top-K) to avoid "constant/stuck" behavior
      - optional novelty penalty to avoid repeating the same projected_idx forever
      - optional disable rounding during optimization (round only at the end)

    Requires external helpers used in your original code:
      - ligand_feature_vector(smiles, bits, radius)
      - _load_model(checkpoint, protein_dim, ligand_dim, device, feat_cfg)
      - _select_indices(std_vec, count)
    """

    step_dir.mkdir(parents=True, exist_ok=True)

    processed = pd.read_csv(processed_csv)
    protein_mat = np.load(protein_path)
    ligand_mat = np.load(ligand_path)

    # if opt_cfg.sample_index >= len(processed):
    #     raise IndexError(
    #         f"sample_index {opt_cfg.sample_index} out of range for processed dataset of size {len(processed)}"
    #     )

    target_sequence = opt_cfg.target_sequence # or processed.loc[opt_cfg.sample_index, "Target"]
    baseline_smiles = opt_cfg.baseline_smiles # or processed.loc[opt_cfg.sample_index, "Drug"]

    use_esm = feat_cfg.protein_encoder.lower() == "esm"
    if opt_cfg.target_sequence:
        if use_esm:
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise RuntimeError("Transformers is required for ESM tokenization") from e
            tokenizer = AutoTokenizer.from_pretrained(feat_cfg.esm_model)
            max_len = min(feat_cfg.max_token_length, tokenizer.model_max_length)
            tokens = tokenizer(
                opt_cfg.target_sequence,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )["input_ids"].squeeze(0).numpy()
            base_protein = tokens.astype(np.int64, copy=False)
        else:
            base_protein = protein_feature_vector(opt_cfg.target_sequence)   
    base_ligand = ligand_feature_vector(baseline_smiles, feat_cfg.morgan_bits, feat_cfg.morgan_radius)

    device = (
        torch.device(feat_cfg.device)
        if feat_cfg.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = _load_model(feat_cfg.checkpoint, protein_mat.shape[1], ligand_mat.shape[1], device, feat_cfg)
    use_amp = device.type == "cuda" and not use_esm

    protein_count = 0 if use_esm else (opt_cfg.protein_features if opt_cfg.allow_protein_adjustments else 0)
    ligand_count = opt_cfg.ligand_features if opt_cfg.allow_ligand_adjustments else 0

    protein_std = protein_mat.std(axis=0) if protein_count > 0 else np.zeros(0, dtype=np.float32)
    ligand_std = ligand_mat.std(axis=0)

    protein_indices = _select_indices(protein_std, protein_count)
    ligand_indices = _select_indices(ligand_std, ligand_count)

    ligand_weights = np.ones(ligand_mat.shape[1], dtype=np.float32)
    if ligand_count > 0:
        ligand_weights.fill(1)               # downweight untouched buckets
        ligand_weights[ligand_indices] = 1.0   # emphasize editable buckets

    # ---------------- ADMET constraints ----------------
    admet_array = None
    admet_penalties: List[Dict[str, Any]] = []
    admet_keys: List[str] = []
    if admet_path and opt_cfg.admet_constraints:
        admet_df = pd.read_csv(admet_path)
        if "dataset_index" in admet_df.columns:
            admet_df = admet_df.set_index("dataset_index")
        # align to processed order
        admet_df = admet_df.reindex(processed["dataset_index"])

        constraint_keys: List[str] = []
        for c in opt_cfg.admet_constraints:
            key = c.get("key")
            if key in admet_df.columns:
                constraint_keys.append(key)
                admet_penalties.append(
                    {
                        "key": key,
                        "min": c.get("min"),
                        "max": c.get("max"),
                        "weight": float(c.get("weight", 1.0)),
                    }
                )

        if constraint_keys:
            admet_array = admet_df[constraint_keys].to_numpy()
            admet_keys = constraint_keys

    admet_key2idx = {k: i for i, k in enumerate(admet_keys)} if admet_keys else {}

    # ---------------- bounds ----------------
    protein_bounds = np.zeros(protein_count, dtype=np.float32)
    if protein_count > 0:
        protein_bounds = np.where(
            protein_std[protein_indices] > 0, 2 * protein_std[protein_indices], 1.0
        ).astype(np.float32)
        if opt_cfg.frozen_protein_features:
            frozen_mask = np.isin(protein_indices, np.array(opt_cfg.frozen_protein_features, dtype=int))
            protein_bounds[frozen_mask] = 0.0

    ligand_bounds = np.zeros(ligand_count, dtype=np.float32)
    if ligand_count > 0:
        ligand_bounds = np.where(
            ligand_std[ligand_indices] > 0, 2 * ligand_std[ligand_indices], 1.0
        ).astype(np.float32)
        if opt_cfg.frozen_ligand_features:
            frozen_mask = np.isin(ligand_indices, np.array(opt_cfg.frozen_ligand_features, dtype=int))
            ligand_bounds[frozen_mask] = 0.0

    init = np.zeros(protein_count + ligand_count, dtype=float)
    lower = np.concatenate((-protein_bounds, -ligand_bounds))
    upper = np.concatenate((protein_bounds, ligand_bounds))

    try:
        import nevergrad as ng
    except ImportError as e:
        raise RuntimeError("nevergrad is required for optimization") from e

    parametrization = ng.p.Array(init=init).set_bounds(lower=lower, upper=upper)

    # ---------------- model scoring ----------------
    def score_pair(protein_vec: np.ndarray, ligand_vec: np.ndarray) -> float:
        ligand_tensor = torch.tensor(ligand_vec, dtype=torch.float32, device=device)
        with torch.no_grad():
            if use_esm:
                protein_tensor = torch.tensor(protein_vec, dtype=torch.long, device=device)
                if protein_tensor.dim() == 1:
                    protein_tensor = protein_tensor.unsqueeze(0)
                if ligand_tensor.dim() == 1:
                    ligand_tensor = ligand_tensor.unsqueeze(0)
                _, _, pred = model.inference(ligand_tensor, protein_tensor)
            else:
                protein_tensor = torch.tensor(protein_vec, dtype=torch.float32, device=device)
                with autocast(enabled=use_amp):
                    _, _, pred = model.inference(ligand_tensor.unsqueeze(0), protein_tensor.unsqueeze(0))
        return float(pred.item())

    # ---- knobs to reduce "stuck constant" behavior ----
    use_rounding = bool(getattr(opt_cfg, "use_rounding", True))
    soft_T = float(getattr(opt_cfg, "softmin_T", 0.75) or 0.75)          # smaller -> closer to min, larger -> closer to mean
    novelty_weight = float(getattr(opt_cfg, "novelty_weight", 0) or 0)  # >0 penalizes repeated projected_idx_best

    def discretize_ligand(vec: np.ndarray) -> np.ndarray:
        if not use_rounding:
            out = vec.astype(np.float32, copy=True)
            np.clip(out, 0.0, None, out=out)
            return out
        rounded = np.rint(vec).astype(np.float32, copy=False)
        np.clip(rounded, 0.0, None, out=rounded)
        return rounded

    baseline_affinity = score_pair(base_protein, base_ligand)

    manifold_weight = float(getattr(opt_cfg, "manifold_weight", 0.0) or 0.0)

    # --------------- Projection-in-loop: fast NN on edited indices ---------------
    S = ligand_indices.astype(int)
    nn_k = int(getattr(opt_cfg, "nn_k", 30) or 30)  # top-K projection size

    lig_mat_S = ligand_mat[:, S].astype(np.float32, copy=False) if len(S) else None
    wS = ligand_weights[S].astype(np.float32, copy=False) if len(S) else None

    def project_neighbors(lig_bucket: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          idxs: (K,) indices of nearest real ligands
          dist: (N,) weighted L1 distances used (for manifold penalty/logging)
        """
        if ligand_count == 0:
            dist = np.sum(np.abs(ligand_mat - lig_bucket[None, :]) * ligand_weights[None, :], axis=1).astype(np.float32)
            idx = int(np.argmin(dist))
            return np.array([idx], dtype=int), dist

        # distance only on editable dims (fast)
        var = np.sum(np.abs(lig_mat_S - lig_bucket[S][None, :]) * wS[None, :], axis=1).astype(np.float32)
        dist = var

        K = max(1, min(nn_k, dist.shape[0]))
        if K == 1:
            return np.array([int(np.argmin(dist))], dtype=int), dist

        if K == dist.shape[0]:
            idxs = np.argsort(dist)
            return idxs.astype(int), dist

        idxs = np.argpartition(dist, K)[:K]
        idxs = idxs[np.argsort(dist[idxs])]
        return idxs.astype(int), dist
    # ---------------------------------------------------------------------------

    # track repeats for novelty penalty
    seen_counts: Dict[int, int] = {}

    def evaluate(delta: np.ndarray, include_rep_penalty: bool = True) -> Dict[str, Any]:
        """
        Evaluate one Nevergrad candidate.
        Computes:
          - best-of-K neighbor (for reporting + manifest)
          - avg-of-K loss (for analysis)
          - softmin-of-K loss (for smoother optimization)
        """
        delta = np.asarray(delta, dtype=np.float32)

        protein_delta = delta[:protein_count]
        ligand_delta = delta[protein_count:]

        protein_adjusted = base_protein.copy()
        ligand_bucket = base_ligand.copy()

        if protein_count > 0:
            protein_adjusted[protein_indices] += protein_delta

        if ligand_count > 0:
            ligand_bucket[ligand_indices] += ligand_delta
            ligand_bucket = discretize_ligand(ligand_bucket)

        idxs, dist = project_neighbors(ligand_bucket)

        # evaluate ALL K neighbors
        losses: List[float] = []
        preds: List[float] = []
        admet_penalties_k: List[float] = []
        manifold_distances_k: List[float] = []

        best = {
            "loss": float("inf"),
            "pred": None,
            "admet_penalty": 0.0,
            "manifold_distance": None,
            "projected_idx": None,
            "admet_vals": None,
        }

        for idx in idxs:
            ligand_proj = ligand_mat[idx]
            pred = score_pair(protein_adjusted, ligand_proj)
            deviation = pred - opt_cfg.target_affinity

            admet_penalty = 0.0
            admet_vals = None
            if admet_array is not None:
                admet_vec = admet_array[idx]
                admet_vals = {
                    k: (float(admet_vec[i]) if not np.isnan(admet_vec[i]) else np.nan)
                    for i, k in enumerate(admet_keys)
                }
                for pen in admet_penalties:
                    j = admet_key2idx[pen["key"]]
                    val = admet_vec[j]
                    if np.isnan(val):
                        continue
                    if pen["min"] is not None and val < pen["min"]:
                        admet_penalty += pen["weight"] * (pen["min"] - val) ** 2
                    if pen["max"] is not None and val > pen["max"]:
                        admet_penalty += pen["weight"] * (val - pen["max"]) ** 2

            manifold_distance = float(dist[idx])
            manifold_penalty = manifold_weight * manifold_distance if manifold_weight > 0.0 else 0.0

            loss = float(deviation**2 + admet_penalty + manifold_penalty)

            losses.append(loss)
            preds.append(float(pred))
            admet_penalties_k.append(float(admet_penalty))
            manifold_distances_k.append(float(manifold_distance))

            if loss < best["loss"]:
                best.update(
                    {
                        "loss": float(loss),
                        "pred": float(pred),
                        "admet_penalty": float(admet_penalty),
                        "manifold_distance": manifold_distance if manifold_weight > 0.0 else None,
                        "projected_idx": int(idx),
                        "admet_vals": admet_vals,
                    }
                )

        losses_np = np.asarray(losses, dtype=np.float32)

        # avg-of-K
        loss_avg = float(np.mean(losses_np))

        # softmin-of-K: w_i ∝ exp(-loss_i / T)
        if len(losses_np) > 1:
            T = max(soft_T, 1e-8)
            x = -losses_np / T
            x = x - np.max(x)  # stabilize
            w = np.exp(x)
            w = w / (np.sum(w) + 1e-12)
            loss_soft = float(np.sum(w * losses_np))
            pred_avg = float(np.mean(np.asarray(preds, dtype=np.float32)))
        else:
            loss_soft = float(losses_np[0])
            pred_avg = float(preds[0])

        # novelty/repeat penalty (optional)
        rep_penalty = 0.0
        if include_rep_penalty and novelty_weight > 0.0 and best["projected_idx"] is not None:
            c = seen_counts.get(best["projected_idx"], 0)
            rep_penalty = novelty_weight * float(c)

        ng_objective = float(loss_soft + rep_penalty)

        return {
            # best-of-K (use this for final selection/manifest)
            **best,

            # top-K stats
            "loss_best": float(best["loss"]),
            "pred_best": float(best["pred"]) if best["pred"] is not None else np.nan,
            "loss_avg_topk": loss_avg,
            "loss_soft_topk": loss_soft,
            "pred_avg_topk": pred_avg,
            "rep_penalty": float(rep_penalty),
            "ng_objective": ng_objective,

            # raw top-K lists (JSON-logged)
            "topk_idxs": [int(i) for i in idxs.tolist()],
            "topk_preds": [float(p) for p in preds],
            "topk_losses": [float(l) for l in losses],
            "topk_admet_penalties": [float(a) for a in admet_penalties_k],
            "topk_manifold_distances": [float(m) for m in manifold_distances_k],

            # payload
            "protein_adjusted": protein_adjusted,
            "ligand_bucket": ligand_bucket,
            "protein_delta": protein_delta,
            "ligand_delta": ligand_delta,
        }

    # ---------------- Nevergrad loop ----------------
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=opt_cfg.budget)

    trace_main: List[Dict[str, Any]] = []
    trace_topk: List[Dict[str, Any]] = []

    def logging_objective(delta):
        out = evaluate(delta, include_rep_penalty=True)

        # update repeat counts AFTER evaluation
        if out["projected_idx"] is not None:
            pi = int(out["projected_idx"])
            seen_counts[pi] = seen_counts.get(pi, 0) + 1

        it = len(trace_main)

        # main trace (compact)
        trace_main.append(
            {
                "iter": it,
                "projected_idx_best": int(out["projected_idx"]) if out["projected_idx"] is not None else None,
                "predicted_affinity_best": float(out["pred_best"]),
                "loss_best": float(out["loss_best"]),
                "ng_objective": float(out["ng_objective"]),
                "admet_penalty_best": float(out["admet_penalty"]),
                "manifold_distance_best": out["manifold_distance"],
                "protein_delta_L1": float(np.sum(np.abs(out["protein_delta"]))) if protein_count > 0 else 0.0,
                "ligand_delta_L1": float(np.sum(np.abs(out["ligand_delta"]))) if ligand_count > 0 else 0.0,
                "ligand_counts": json.dumps(out["ligand_bucket"][ligand_indices].astype(int).tolist())
                if ligand_count > 0 and use_rounding
                else None,
            }
        )

        # top-K stats trace (for plotting/diagnostics)
        trace_topk.append(
            {
                "iter": it,
                "K": int(len(out["topk_idxs"])),
                "nn_k": int(nn_k),
                "loss_best": float(out["loss_best"]),
                "loss_avg_topk": float(out["loss_avg_topk"]),
                "loss_soft_topk": float(out["loss_soft_topk"]),
                "pred_best": float(out["pred_best"]),
                "pred_avg_topk": float(out["pred_avg_topk"]),
                "rep_penalty": float(out["rep_penalty"]),
                "ng_objective": float(out["ng_objective"]),
                "projected_idx_best": int(out["projected_idx"]) if out["projected_idx"] is not None else None,
                "topk_projected_idxs": json.dumps(out["topk_idxs"]),
                "topk_preds": json.dumps(out["topk_preds"]),
                "topk_losses": json.dumps(out["topk_losses"]),
                "topk_admet_penalties": json.dumps(out["topk_admet_penalties"]),
                "topk_manifold_distances": json.dumps(out["topk_manifold_distances"]),
            }
        )

        return out["ng_objective"]

    recommendation = optimizer.minimize(logging_objective)
    best_delta = np.asarray(recommendation.value, dtype=np.float32)

    # Final evaluation (no novelty penalty)
    final = evaluate(best_delta, include_rep_penalty=False)

    optimized_protein = final["protein_adjusted"]
    optimized_bucket = final["ligand_bucket"]
    projected_idx = final["projected_idx"]
    optimized_affinity = float(final["pred_best"])

    # ---------------- summaries ----------------
    protein_summary = pd.DataFrame(
        {
            "feature_type": ["protein"] * len(protein_indices),
            "feature_index": protein_indices.astype(int),
            "baseline": base_protein[protein_indices] if len(protein_indices) else [],
            "new_value": optimized_protein[protein_indices] if len(protein_indices) else [],
        }
    )
    if len(protein_indices):
        protein_summary["adjustment"] = protein_summary["new_value"] - protein_summary["baseline"]

    ligand_baseline = discretize_ligand(base_ligand)
    ligand_summary = pd.DataFrame(
        {
            "feature_type": ["ligand"] * len(ligand_indices),
            "feature_index": ligand_indices.astype(int),
            "baseline": ligand_baseline[ligand_indices].astype(int) if use_rounding else ligand_baseline[ligand_indices],
            "new_value": optimized_bucket[ligand_indices].astype(int) if use_rounding else optimized_bucket[ligand_indices],
        }
    )
    ligand_summary["adjustment"] = ligand_summary["new_value"] - ligand_summary["baseline"]
    ligand_adjustments = ligand_summary.sort_values("feature_index")["adjustment"].tolist()

    summary_df = pd.concat([protein_summary, ligand_summary], ignore_index=True)
    summary_df["abs_adjustment"] = summary_df["adjustment"].abs()

    summary_path = step_dir / "optimization_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Save traces
    trace_path = step_dir / "optimization_trace.csv"
    topk_stats_path = step_dir / "optimization_trace_topk_stats.csv"

    try:
        pd.DataFrame(trace_main).to_csv(trace_path, index=False)
    except Exception:
        trace_path = None

    try:
        pd.DataFrame(trace_topk).to_csv(topk_stats_path, index=False)
    except Exception:
        topk_stats_path = None

    # Manifest: includes both the BUCKET target and the PROJECTED real ligand index
    manifest = {
        "baseline_affinity": float(baseline_affinity),
        "optimized_affinity": float(optimized_affinity),
        "sample_index": int(opt_cfg.sample_index),
        "target_sequence": target_sequence,
        "baseline_smiles": baseline_smiles,
        "ligand_indices": ligand_indices.astype(int).tolist(),
        "target_counts": ligand_summary.sort_values("feature_index")["new_value"].tolist(),
        "ligand_adjustments": ligand_adjustments,
        "manifold_weight": float(manifold_weight),
        "nn_k": int(nn_k),
        "softmin_T": float(soft_T),
        "use_rounding": bool(use_rounding),
        "novelty_weight": float(novelty_weight),
        "projected_neighbor_index": int(projected_idx) if projected_idx is not None else None,
    }
    (step_dir / "target_manifest.json").write_text(json.dumps(manifest, indent=2))

    details = manifest #| {"protein_indices": protein_indices.astype(int).tolist()}
    outputs = {"summary": summary_path, "manifest": step_dir / "target_manifest.json"}
    if trace_path:
        outputs["trace"] = trace_path
    if topk_stats_path:
        outputs["topk_stats"] = topk_stats_path

    return outputs, details




# ------------------------------ retrieval ------------------------------ #
def _compute_mol_weight(smiles: str) -> Optional[float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
    except ImportError:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return float(Descriptors.MolWt(mol))


def _canonical_smiles(smiles: str) -> str:
    """Return canonical SMILES if RDKit is available, else the original string."""
    try:
        from rdkit import Chem
    except ImportError:
        return smiles
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return smiles
    return Chem.MolToSmiles(m, canonical=True)


def retrieve_candidates(
    cfg: RetrievalConfig,
    step_dir: Path,
    processed_csv: Path,
    ligand_path: Path,
    optimization_details: Dict[str, Any],
) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    ligand_indices = optimization_details.get("ligand_indices", [])
    target_counts = optimization_details.get("target_counts", [])
    ligand_adjustments = optimization_details.get("ligand_adjustments", [])
    if len(ligand_indices) == 0 or len(target_counts) == 0:
        return {}, {"skipped": True, "reason": "No ligand feature adjustments to anchor retrieval."}

    processed = pd.read_csv(processed_csv)
    ligand_mat = np.load(ligand_path)
    bucket_array = np.array(ligand_indices, dtype=int)
    target_counts_arr = np.array(target_counts, dtype=np.float32)

    fingerprint_subset = ligand_mat[:, bucket_array].astype(np.float32, copy=False)
    diff_matrix = np.abs(fingerprint_subset - target_counts_arr)
    # Heavily weight edited buckets; lightly weight unchanged ones.
    weights = np.ones_like(target_counts_arr) #* 0.2
    # if len(ligand_adjustments) == len(target_counts_arr):
    #     adj = np.asarray(ligand_adjustments, dtype=np.float32)
    #     weights = np.where(np.abs(adj) > 0, 5.0 * (1.0 + np.abs(adj)), 0.2)
    # else:
    #     weights = np.ones_like(target_counts_arr)
    weighted_diff = diff_matrix * weights
    l1_distance = weighted_diff.sum(axis=1)
    max_bucket_diff = diff_matrix.max(axis=1)

    baseline_smiles = optimization_details.get("baseline_smiles")
    if not baseline_smiles:
        raise KeyError("baseline_smiles is missing in optimization_details")
    
    baseline_smiles_len = len(baseline_smiles)
    baseline_mw = _compute_mol_weight(baseline_smiles)

    candidate_records = []
    seen_smiles = set()
    seen_canonical = set()
    for idx, (distance, row) in enumerate(zip(l1_distance, processed.itertuples(index=False))):
        smi = getattr(row, "smiles", getattr(row, "Drug", None))
        if not smi or smi in seen_smiles:
            continue
        can_smi = _canonical_smiles(smi)
        if can_smi in seen_canonical:
            continue
        smiles_len = len(smi)
        if baseline_smiles_len > 0:
            rel_len_diff = abs(smiles_len - baseline_smiles_len) / baseline_smiles_len
            if rel_len_diff > cfg.max_rel_smiles_len_diff:
                continue
        cand_mw = getattr(row, "MolWt", None)
        if pd.notna(cand_mw):
            cand_mw = float(cand_mw)
        else:
            cand_mw = _compute_mol_weight(smi)
        if baseline_mw and cand_mw:
            rel_mw_diff = abs(cand_mw - baseline_mw) / baseline_mw
            if rel_mw_diff > cfg.max_rel_mw_diff:
                continue
        if cfg.max_l1_distance is not None and distance > cfg.max_l1_distance:
            continue

        counts = fingerprint_subset[idx].astype(int).tolist()
        candidate_records.append(
            {
                "dataset_index": int(getattr(row, "dataset_index", idx)),
                "smiles": smi,
                "smiles_length": smiles_len,
                "mol_weight": cand_mw,
                "distance_L1": float(distance),
                "max_bucket_difference": float(max_bucket_diff[idx]),
                "bucket_counts": json.dumps(counts),
            }
        )
        seen_smiles.add(smi)
        seen_canonical.add(can_smi)
        if cfg.top_candidates and len(candidate_records) >= cfg.top_candidates:
            break

    if not candidate_records:
        return {}, {"skipped": True, "reason": "No candidates matched retrieval thresholds."}

    candidate_df = pd.DataFrame(candidate_records)
    candidate_path = step_dir / "candidate_ligands.csv"
    candidate_df.to_csv(candidate_path, index=False)
    details = {
        "baseline_smiles": baseline_smiles,
        "candidates": len(candidate_df),
        "bucket_array": bucket_array.tolist(),
        "target_counts": target_counts,
    }
    return {"candidates": candidate_path}, details


# ------------------------------ visualization ------------------------------ #
def visualize_candidates(cfg: VisualizationConfig, step_dir: Path, candidates_csv: Path) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.enabled:
        return {}, {"skipped": True, "reason": "Visualization disabled"}
    df = pd.read_csv(candidates_csv)
    if df.empty:
        return {}, {"skipped": True, "reason": "No candidates to render"}

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Draw, rdDepictor
        from rdkit.Chem.Draw import rdMolDraw2D
        from IPython.display import SVG  # noqa: F401
    except ImportError as e:
        raise RuntimeError("RDKit is required for visualization") from e

    id_col = cfg.id_col if cfg.id_col in df.columns else None
    ids = df[id_col].astype(str).tolist() if id_col else df.index.astype(str).tolist()
    smiles_series = df["smiles"].astype(str)

    rdDepictor.SetPreferCoordGen(True)
    mols, legends = [], []
    for cid, smi in zip(ids, smiles_series):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        AllChem.Compute2DCoords(m)
        mols.append(m)
        legends.append(f"ID={cid}")

    if cfg.max_mols is not None:
        mols = mols[: cfg.max_mols]
        legends = legends[: cfg.max_mols]

    if not mols:
        return {}, {"skipped": True, "reason": "No valid molecules parsed"}

    W, H = cfg.panel_size
    mols_per_row = max(1, cfg.mols_per_row)
    rows = math.ceil(len(mols) / mols_per_row)
    total_W, total_H = W * mols_per_row, H * rows

    drawer_svg = rdMolDraw2D.MolDraw2DSVG(total_W, total_H, W, H)
    drawer_svg.DrawMolecules(mols, legends=legends)
    drawer_svg.FinishDrawing()
    out_svg_path = step_dir / cfg.out_svg
    out_svg_path.write_text(drawer_svg.GetDrawingText())

    drawer_png = rdMolDraw2D.MolDraw2DCairo(total_W, total_H, W, H)
    drawer_png.DrawMolecules(mols, legends=legends)
    drawer_png.FinishDrawing()
    out_png_path = step_dir / cfg.out_png
    out_png_path.write_bytes(drawer_png.GetDrawingText())

    details = {"rendered": len(mols), "id_col": id_col or "index"}
    return {"grid_svg": out_svg_path, "grid_png": out_png_path}, details


# ------------------------------ docking ------------------------------ #
def _resolve_meeko() -> str:
    cand = shutil.which("mk_prepare_ligand.py")
    if cand:
        return cand
    alt = Path(sys.prefix) / "bin" / "mk_prepare_ligand.py"
    if alt.is_file():
        return str(alt)
    raise FileNotFoundError("mk_prepare_ligand.py not found on PATH")


def _smiles_to_files(smi: str, out_dir: Path, base: str = "ligand") -> Tuple[Path, Path]:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    out_dir.mkdir(parents=True, exist_ok=True)
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise ValueError(f"Bad SMILES: {smi}")
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=True)
    m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(m)
    sdf_path = out_dir / f"{base}.sdf"
    with Chem.SDWriter(str(sdf_path)) as w:
        w.write(m)
    mk_cli = _resolve_meeko()
    pdbqt_path = out_dir / f"{base}.pdbqt"
    r = subprocess.run([mk_cli, "-i", str(sdf_path), "-o", str(pdbqt_path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Meeko failed for SMILES: {smi}\n{r.stderr[:400]}")
    return sdf_path, pdbqt_path


def _parse_vina_score(stdout_text: str, log_path: Path) -> Optional[float]:
    lines = stdout_text.splitlines()
    if log_path.exists():
        try:
            lines += log_path.read_text().splitlines()
        except Exception:
            pass
    for L in lines:
        if "VINA RESULT" in L:
            try:
                return float(L.split()[3])
            except Exception:
                pass
    start = False
    for L in lines:
        if L.strip().startswith("mode") and "affinity" in L:
            start = True
            continue
        if start and L.strip() and L.strip()[0].isdigit():
            try:
                return float(L.split()[1])
            except Exception:
                break
    return None


def dock_candidates(
    cfg: DockingConfig,
    step_dir: Path,
    candidates_csv: Path,
    baseline_smiles: Optional[str] = None,
    baseline_index: Optional[int] = None,
) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.enabled:
        return {}, {"skipped": True, "reason": "Docking disabled"}
    df = pd.read_csv(candidates_csv)
    if df.empty:
        return {}, {"skipped": True, "reason": "No candidates to dock"}

    if baseline_smiles:
        already_has_baseline = False
        if "smiles" in df.columns:
            already_has_baseline = df["smiles"].astype(str).eq(str(baseline_smiles)).any()
        if not already_has_baseline:
            extra = {
                "dataset_index": -1,
                "smiles": baseline_smiles,
            }
            df = pd.concat([pd.DataFrame([extra]), df], ignore_index=True)

    center = cfg.center
    size = cfg.size
    if center is None:
        try:
            from dock_with_tdc_box import get_tdc_box_for_target

            _, center, size = get_tdc_box_for_target(cfg.target_key)
        except Exception as e:
            raise RuntimeError("Docking box center/size not provided and could not be inferred") from e

    cx, cy, cz = center
    sx, sy, sz = size
    out_root = step_dir
    out_root.mkdir(parents=True, exist_ok=True)

    results = []
    limit = cfg.limit or len(df)
    iterator = enumerate(df.itertuples(index=False))
    try:
        iterator = tqdm(iterator, total=len(df), desc="[docking] vina")  # type: ignore[arg-type]
    except Exception:
        pass
    for idx, row in iterator:
        cand_id = getattr(row, "dataset_index", None) or getattr(row, "candidate_id", None) or idx
        smi = getattr(row, "smiles")
        cand_dir = out_root / f"cand_{cand_id}"
        try:
            sdf_path, lig_pdbqt = _smiles_to_files(smi, cand_dir, base=str(cand_id))
            docked_pdbqt = cand_dir / f"docked_{cand_id}.pdbqt"
            log_path = cand_dir / f"vina_{cand_id}.log"
            cmd = [
                "vina",
                "--receptor",
                str(cfg.receptor_pdbqt),
                "--ligand",
                str(lig_pdbqt),
                "--center_x",
                str(cx),
                "--center_y",
                str(cy),
                "--center_z",
                str(cz),
                "--size_x",
                str(sx),
                "--size_y",
                str(sy),
                "--size_z",
                str(sz),
                "--exhaustiveness",
                str(cfg.exhaustiveness),
                "--num_modes",
                str(cfg.num_modes),
                "--seed",
                str(cfg.seed),
                "--out",
                str(docked_pdbqt),
                "--log",
                str(log_path),
            ]
            r = subprocess.run(cmd, capture_output=True, text=True)
            score = _parse_vina_score(r.stdout, log_path)
            results.append(
                {
                    "candidate_id": cand_id,
                    "smiles": smi,
                    "vina_score": score,
                    "ligand_sdf": str(sdf_path),
                    "ligand_pdbqt": str(lig_pdbqt),
                    "docked_pdbqt": str(docked_pdbqt),
                    "vina_log": str(log_path),
                    "returncode": r.returncode,
                    "stderr": r.stderr[:300],
                }
            )
        except Exception as e:
            results.append({"candidate_id": cand_id, "smiles": smi, "vina_score": None, "error": str(e)})
        if len(results) >= limit:
            break

    summary_df = pd.DataFrame(results)
    summary_path = step_dir / "docking_scores.csv"
    summary_df.to_csv(summary_path, index=False)
    return {"docking": summary_path}, {"docked": int(summary_df["vina_score"].notna().sum())}


# ------------------------------ ADMET ------------------------------ #
def admet_predictions(cfg: AdmetConfig, step_dir: Path, candidates_csv: Path) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.enabled:
        return {}, {"skipped": True, "reason": "ADMET disabled"}
    df = pd.read_csv(candidates_csv)
    if "smiles" not in df.columns or df.empty:
        return {}, {"skipped": True, "reason": "No SMILES available for ADMET"}
    try:
        from admet_ai import ADMETModel
    except ImportError as e:
        raise RuntimeError("admet-ai is required for ADMET predictions") from e

    model = _silent_run(ADMETModel)
    preds = _silent_run(model.predict, df["smiles"].astype(str).tolist())
    if isinstance(preds, pd.DataFrame):
        pred_rows = preds.to_dict(orient="records")
    elif isinstance(preds, (list, tuple)):
        pred_rows = list(preds)
    else:
        raise TypeError(f"Unexpected preds type: {type(preds)}")
    if len(pred_rows) != len(df):
        raise RuntimeError(f"Prediction count mismatch: got {len(pred_rows)} vs {len(df)} rows")

    out_df = df.copy()
    for col in cfg.keys:
        out_df[col] = pd.NA
    for idx, pred in enumerate(pred_rows):
        for key in cfg.keys:
            out_df.at[idx, key] = pred.get(key)
    out_path = step_dir / "admet_results.csv"
    out_df.to_csv(out_path, index=False)
    return {"admet": out_path}, {"rows": len(out_df)}


# ------------------------------ report ------------------------------ #
def build_report(
    step_dir: Path,
    candidates_csv: Path,
    docking_csv: Optional[Path],
    admet_csv: Optional[Path],
    baseline_index: Optional[int] = None,
    baseline_smiles: Optional[str] = None,
    processed_csv: Optional[Path] = None,
    protein_path: Optional[Path] = None,
    ligand_path: Optional[Path] = None,
    feat_cfg: Optional[FeatureConfig] = None,
    optimization_details: Optional[Dict[str, Any]] = None,
) -> StepOutputs:
    step_dir.mkdir(parents=True, exist_ok=True)
    base_df = pd.read_csv(candidates_csv) if candidates_csv and Path(candidates_csv).exists() else pd.DataFrame()
    if base_df.empty:
        return {}, {"skipped": True, "reason": "No candidates available for report"}

    def _build_override_protein() -> Optional[np.ndarray]:
        if optimization_details is None or feat_cfg is None:
            return None
        seq = optimization_details.get("target_sequence")
        use_esm_local = feat_cfg.protein_encoder.lower() == "esm"
        if use_esm_local:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                return None
            tokenizer = AutoTokenizer.from_pretrained(feat_cfg.esm_model)
            max_len = min(feat_cfg.max_token_length, tokenizer.model_max_length)
            tokens = tokenizer(
                seq,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=max_len,
            )["input_ids"].squeeze(0).numpy()
            return tokens.astype(np.int64, copy=False)
        return protein_feature_vector(seq)

    def _attach_affinity(df: pd.DataFrame) -> pd.DataFrame:
        if processed_csv is None or protein_path is None or ligand_path is None or feat_cfg is None:
            return df
        if not (Path(processed_csv).exists() and Path(protein_path).exists() and Path(ligand_path).exists()):
            return df
        try:
            protein_mat = np.load(protein_path)
            ligand_mat = np.load(ligand_path)
            device = torch.device(feat_cfg.device) if feat_cfg.device else torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            model = _load_model(feat_cfg.checkpoint, protein_mat.shape[1], ligand_mat.shape[1], device, feat_cfg)
            use_esm = feat_cfg.protein_encoder.lower() == "esm"
            use_amp = device.type == "cuda" and not use_esm
            override_protein_vec = _build_override_protein()
            override_ligand_smiles = (
                (optimization_details or {}).get("baseline_smiles") if optimization_details else None
            ) or baseline_smiles
            override_ligand_vec = (
                ligand_feature_vector(override_ligand_smiles, feat_cfg.morgan_bits, feat_cfg.morgan_radius)
                if override_ligand_smiles
                else None
            )
            affinities: List[float] = []
            with torch.no_grad():
                for idx in df["dataset_index"].astype(int):

                    protein_vec = override_protein_vec
                    if idx != -1 :
                        ligand_vec = ligand_mat[idx]
                    else :
                        # baseline 
                        ligand_vec = override_ligand_vec

                    ligand_tensor = torch.tensor(ligand_vec, dtype=torch.float32, device=device)
                    if use_esm:
                        protein_tensor = torch.tensor(protein_vec, dtype=torch.long, device=device)
                        if protein_tensor.dim() == 1:
                            protein_tensor = protein_tensor.unsqueeze(0)
                        if ligand_tensor.dim() == 1:
                            ligand_tensor = ligand_tensor.unsqueeze(0)
                        _, _, pred = model.inference(ligand_tensor, protein_tensor)
                    else:
                        protein_tensor = torch.tensor(protein_vec, dtype=torch.float32, device=device)
                        with autocast(enabled=use_amp):
                            _, _, pred = model.inference(ligand_tensor.unsqueeze(0), protein_tensor.unsqueeze(0))
                    affinities.append(float(pred.item()))
            df = df.copy()
            df["predicted_affinity"] = affinities
            return df
        except Exception:
            return df

    merged = base_df.copy()

    # Prepend the baseline molecule for easy reference.
    if baseline_smiles is not None or baseline_index is not None:
        baseline_row = {
            "dataset_index": -1,
            "smiles": baseline_smiles,
        }
        merged = pd.concat([pd.DataFrame([baseline_row]), merged], ignore_index=True)

    merged.insert(0, "start_mol", np.arange(len(merged), dtype=int))
    merged = _attach_affinity(merged)
    if docking_csv and Path(docking_csv).exists():
        dock_df = pd.read_csv(docking_csv)
        merged = merged.merge(dock_df[["candidate_id", "vina_score"]], left_on="dataset_index", right_on="candidate_id", how="left")
        merged = merged.drop(columns=["candidate_id"])
    if admet_csv and Path(admet_csv).exists():
        admet_df = pd.read_csv(admet_csv)
        merged = merged.merge(admet_df.drop(columns=["distance_L1", "max_bucket_difference"], errors="ignore"), on="smiles", how="left")

    def _attach_baseline_admet(df: pd.DataFrame) -> pd.DataFrame:
        if baseline_smiles is None:
            return df
        baseline_mask = df["smiles"].astype(str) == str(baseline_smiles)
        if not baseline_mask.any():
            return df
        # If we already have ADMET values for baseline, keep them.
        admet_cols = list(getattr(feat_cfg, "admet_keys", []) or [])
        if admet_cols and df.loc[baseline_mask, admet_cols].notna().any(axis=None):
            return df
        try:
            from admet_ai import ADMETModel
        except Exception:
            return df
        model = _silent_run(ADMETModel)
        preds = _silent_run(model.predict, [baseline_smiles])
        if isinstance(preds, pd.DataFrame):
            pred_row = preds.iloc[0].to_dict()
        elif isinstance(preds, (list, tuple)) and preds:
            pred_row = dict(preds[0])
        elif isinstance(preds, dict):
            pred_row = preds
        else:
            return df
        if not admet_cols:
            admet_cols = list(pred_row.keys())
        df = df.copy()
        for col in admet_cols:
            df.loc[baseline_mask, col] = pred_row.get(col)
        return df

    merged = _attach_baseline_admet(merged)

    report_path = step_dir / "nevermore_report.csv"
    merged.to_csv(report_path, index=False)
    return {"report": report_path}, {"rows": len(merged)}
