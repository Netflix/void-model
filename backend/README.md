# VOID FastAPI Backend

## Run

From repository root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

## What it does

- Launches allowlisted VOID workflows:
- `pass1_inference`
- `pass2_refine`
- `mask_pipeline` (stage1-4 wrapper)
- `mask_stage1_sam2`
- `mask_stage2_vlm`
- `mask_stage3_grey`
- `mask_stage4_combine`
- `point_selector_gui`
- `edit_quadmask_gui`
- Tracks run status
- Stores run logs in `backend/state/logs/`
- Exposes run and environment APIs for the frontend

## Endpoints

- `GET /health`
- `GET /env/check`
- `GET /runs`
- `POST /runs`
- `GET /runs/{id}`
- `POST /runs/{id}/cancel`
- `GET /runs/{id}/logs`
- `GET /runs/{id}/logs/stream`
- `GET /artifacts?runId=...`
- `GET /presets`
- `POST /presets`
- `POST /validate/config`
