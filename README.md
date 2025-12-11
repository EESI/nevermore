# Nevermore Pipeline

Modular, cached version of the Nevermore notebook that chains data ingest, feature building, optimization, retrieval, visualization, docking, ADMET, and reporting. Each step writes to `nevermore/outputs/<step>/<signature>/` and is skipped automatically if inputs/configs are unchanged.

## Layout
- `nevermore/configs/default.yaml` — editable defaults for every stage.
- `nevermore/` — pipeline + step implementations.
- `nevermore/notebooks/run_nevermore.ipynb` — quick notebook entrypoint.

## Quick start
```bash
# from Firm-DTI/Firm-DTI2
python -m nevermore.cli --config nevermore/configs/default.yaml --up-to retrieval
```
Change `--up-to` to run deeper (visualization, docking, admet, report). Outputs are printed with their cache signature.

### Run without notebooks / VS Code
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
- The pipeline assumes the Firm-DTI2 repo root contains `data/train.csv` and the checkpoint from the original notebook. Adjust paths in the config if yours differ.
