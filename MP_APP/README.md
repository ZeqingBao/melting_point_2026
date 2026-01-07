# Melting-Point Predictor (placeholder)

Simple Flask web app that takes a SMILES string, shows the structure and basic descriptors from RDKit, and returns a placeholder predicted melting point.

Quick start (recommended: conda)

1. Create environment and install RDKit (recommended):

```bash
conda env create -f environment.yml
conda activate mp_app
# Melting-Point Predictor

Minimal Flask app that computes RDKit descriptors and a placeholder melting-point prediction for SMILES.

Quick start

1. Create environment (RDKit via conda-forge recommended):

```bash
conda env create -f environment.yml
conda activate mp_app
```

2. Run the app (default port 5005):

```bash
python app.py
```

3. Open http://127.0.0.1:5001

Notes

- UI modes: `Single` (one SMILES) and `Batch` (CSV upload).
- Batch preview shows up to 50 uploaded rows and 50 predicted rows; full CSV output is saved to `tmp/<uuid>_out.csv` and downloadable.
- RDKit is easiest to install via conda: `conda install -c conda-forge rdkit`.
- If the port is in use, change `PORT` in `app.py` or stop the process gracefully.

Key files

- `app.py` — Flask app and endpoints (`/`, `/batch`, `/batch_ajax`, `/download/<filename>`)
- `templates/index.html` — UI
- `tmp/` — temporary CSV outputs
