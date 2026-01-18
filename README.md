# Nevermore Pipeline

Nevermore is a **target-conditioned, closed-loop framework for ligand optimization** that combines (1) a protein–ligand **binding-affinity oracle**, (2) **derivative-free multi-objective optimization** in a learned representation/descriptor space, and (3) **database-grounded retrieval** to keep proposed molecules anchored to valid chemical matter. This repository implements the end-to-end experimental workflow described in the JCIM manuscript: **data ingest → feature building → optimization → retrieval → visualization → (optional) docking → (optional) ADMET → reporting**.

The codebase is **modular and cached** for reproducibility and efficient iteration. Each stage writes its artifacts to  
`nevermore/outputs/<step>/<signature>/` and is automatically **skipped** when inputs and configuration are unchanged. Signatures are derived deterministically from the stage configuration and upstream artifacts; changing any config value or upstream file produces a new signature and a new output directory.

## Layout
- `nevermore/configs/default.yaml` — editable defaults for every stage (targets, checkpoints, retrieval settings, etc.).
- `nevermore/` — pipeline implementation and stage modules.
- `nevermore/notebooks/run_nevermore.ipynb` — notebook entrypoint (optional; everything is runnable via CLI).


## Quick start
```bash

python -m nevermore.cli --config nevermore/configs/default.yaml --up-to retrieval
```
Change `--up-to` to run deeper (visualization, docking, admet, report). Outputs are printed with their cache signature.

Everything is scriptable; no notebook required. From the repo root:
```bash
# ingest → features → optimization → retrieval
python -m nevermore.cli --config nevermore/configs/default.yaml --up-to retrieval

# full stack including docking + ADMET (enable them in the config first)
python -m nevermore.cli --config nevermore/configs/default.yaml --up-to report
```
You can edit `nevermore/configs/default.yaml` directly to change the checkpoint, ESM model, or enable docking/ADMET. Cached outputs go to `nevermore/outputs/<step>/<signature>/`.

## Notes
- Docking and ADMET are disabled by default; flip `enabled: true` in the config to run them.
- Retrieval reuses previous steps when inputs match. Change any config value or upstream file to force a new signature/output set.
