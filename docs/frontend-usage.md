# VOID Production Bay Usage

## Prerequisites

- Python environment with backend dependencies installed:
  - `pip install -r backend/requirements.txt`
- Frontend dependencies:
  - `cd frontend && npm install`

## Start Services

From repo root, terminal 1:

```bash
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

From repo root, terminal 2:

```bash
cd frontend
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Main Flows

## Flow A: Pass 1 Inference

1. Open `Data & Inputs` and set sequence root (for example `./sample`), then click `Load Sequences`.
2. Toggle sequences to include in run and edit each prompt `bg` text if needed.
3. Open `Launch Workflow`, choose `Pass 1 Inference`.
4. Configure model and output paths, then submit.
5. Track logs/status in `Runs` and inspect output files in `Artifacts`.

## Flow B: Mask Pipeline

1. In `Launch Workflow`, choose `Mask Pipeline`.
2. Set `config_points_json`.
3. Choose execution mode:
   - `Full pipeline`
   - `Stage 1/2/3/4` only
4. Use GUI helper buttons from `Run Details` as needed:
   - `Run Point Selector GUI`
   - `Run Quadmask Editor GUI`
5. Launch run and monitor logs.

## Flow C: Pass 2 Refinement

1. Choose `Pass 2 Refinement`.
2. Enter `video names`, `data root`, `pass1 output dir`, output path, model/checkpoint paths.
3. Set dimensions/guidance/steps and cache options.
4. Launch run, then review generated artifacts.

## Presets and Reproducibility

- Save current form state with `Save Current As Preset`.
- Load prior presets from the dropdown.
- Export a run config JSON with `Export Current Config`.
- Import workflow config JSON with `Import Config File`.
- Starter importable configs are provided in `docs/presets/`.

## Testing

Backend tests:

```bash
python -m unittest discover -s backend/tests -p "test_*.py" -v
```

Frontend tests:

```bash
cd frontend
npm test
```
