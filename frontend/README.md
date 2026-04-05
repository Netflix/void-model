## VOID Frontend (Next.js)

This app is the UI layer for running VOID workflows through the FastAPI backend.

### 1) Start backend

From repo root:

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

### 2) Start frontend

```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```

Open `http://localhost:3000`.

### 3) Run tests

```bash
cd frontend
npm test
```

Watch mode:

```bash
npm run test:watch
```

E2E smoke scaffolding (Playwright):

```bash
cd frontend
npx playwright install
npm run test:e2e
```

### Implemented in this phase

- Environment checks (`python`, `ffmpeg`, `CUDA`, `GEMINI_API_KEY`)
- Run launcher forms for:
  - Pass 1 inference
  - Pass 2 refinement
  - Mask pipeline (full or stage-by-stage)
- Quick launch buttons for:
  - Point selector GUI
  - Quadmask editor GUI
- Full Pass 1 config surface (basic + advanced) mapped to runtime `--config.*` overrides
- Preset save/load (workflow + params)
- Run cloning from run history (clone to form / clone and launch)
- Config export/import (JSON)
- Run list + status polling
- Run cancellation
- Log viewer
- Artifact browser (per selected run)
- Preflight validation before run launch (`/validate/config`)
- Pass 2 cache management UX (inspect/clear cache dir)

### Backend API expected

- `GET /env/check`
- `GET /runs`
- `POST /runs`
- `POST /runs/:id/cancel`
- `GET /runs/:id/logs`
- `GET /presets`
- `POST /presets`
- `POST /validate/config`
