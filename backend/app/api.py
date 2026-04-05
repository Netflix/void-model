from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import LOG_DIR, PROJECT_ROOT, RUNS_FILE, PRESETS_FILE, ensure_state_dirs, now_iso
from .jobs import run_worker
from .models import PresetRecord, RunRecord
from .schemas import (
    CachePathRequest,
    PresetCreateRequest,
    PromptUpdateRequest,
    RunCreateRequest,
    ValidateConfigRequest,
)
from .stores import PresetStore, RunStore
from .validators import (
    directory_stats,
    load_prompt_bg,
    path_under_project,
    safe_path,
    validate_paths,
)
from .workflows import build_command


def create_app() -> FastAPI:
    ensure_state_dirs()

    run_store = RunStore(RUNS_FILE, now_iso)
    preset_store = PresetStore(PRESETS_FILE)

    app = FastAPI(title="VOID Frontend Backend", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "time": now_iso()}

    @app.get("/env/check")
    def env_check() -> dict[str, Any]:
        python_path = shutil.which("python")
        ffmpeg_path = shutil.which("ffmpeg")
        nvidia_smi = shutil.which("nvidia-smi")

        cuda = {"available": None, "error": None}
        try:
            probe = subprocess_run_torch_cuda()
            if probe["ok"]:
                cuda["available"] = probe["available"]
            else:
                cuda["error"] = probe["error"]
        except Exception as exc:
            cuda["error"] = str(exc)

        return {
            "project_root": str(PROJECT_ROOT),
            "python": python_path,
            "ffmpeg": ffmpeg_path,
            "nvidia_smi": nvidia_smi,
            "cuda": cuda,
            "gemini_api_key_set": bool(os.environ.get("GEMINI_API_KEY")),
        }

    @app.get("/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [asdict(r) for r in run_store.list()]

    @app.get("/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = run_store.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        return asdict(run)

    @app.post("/runs")
    def create_run(req: RunCreateRequest) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:12]
        command, output_dir = build_command(req.workflow, req.params)
        errors, warnings = validate_paths(req.workflow, req.params, command)
        if errors:
            raise HTTPException(
                status_code=400,
                detail={"errors": errors, "warnings": warnings, "command": command},
            )

        log_path = LOG_DIR / f"{run_id}.log"
        log_path.touch()

        record = RunRecord(
            id=run_id,
            workflow=req.workflow,
            status="queued",
            created_at=now_iso(),
            started_at=None,
            ended_at=None,
            exit_code=None,
            command=command,
            name=req.name,
            params=req.params,
            log_path=str(log_path),
            output_dir=output_dir,
            error=None,
        )
        run_store.create(record)

        thread = threading.Thread(target=run_worker, args=(run_id, command, log_path, run_store), daemon=True)
        thread.start()

        return asdict(record)

    @app.post("/validate/config")
    def validate_config(req: ValidateConfigRequest) -> dict[str, Any]:
        command, _ = build_command(req.workflow, req.params)
        errors, warnings = validate_paths(req.workflow, req.params, command)
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "command_preview": command,
        }

    @app.get("/presets")
    def list_presets() -> list[dict[str, Any]]:
        return [asdict(p) for p in preset_store.list()]

    @app.post("/presets")
    def create_preset(req: PresetCreateRequest) -> dict[str, Any]:
        preset_id = uuid.uuid4().hex[:12]
        preset = PresetRecord(
            id=preset_id,
            name=req.name.strip(),
            workflow=req.workflow,
            params=req.params,
            created_at=now_iso(),
        )
        preset_store.create(preset)
        return asdict(preset)

    @app.post("/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        run = run_store.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        proc = run_store.get_proc(run_id)
        if not proc:
            return asdict(run)

        try:
            if os.name != "nt":
                os.kill(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass

        run_store.patch(run_id, status="cancelled", ended_at=now_iso(), exit_code=-15)
        current = run_store.get(run_id)
        return asdict(current) if current else asdict(run)

    @app.get("/runs/{run_id}/logs")
    def read_logs(run_id: str, tail: int = 400) -> dict[str, Any]:
        run = run_store.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        log_path = Path(run.log_path)
        if not log_path.exists():
            return {"lines": [], "text": ""}

        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        selected = lines[-tail:]
        return {"lines": selected, "text": "\\n".join(selected)}

    @app.get("/runs/{run_id}/logs/stream")
    async def stream_logs(run_id: str):
        run = run_store.get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        log_path = Path(run.log_path)

        async def event_source():
            last_size = 0
            while True:
                if log_path.exists():
                    current = log_path.read_text(encoding="utf-8", errors="replace")
                    chunk = current[last_size:]
                    if chunk:
                        for line in chunk.splitlines():
                            yield f"data: {line}\\n\\n"
                        last_size = len(current)

                status = run_store.get(run_id).status
                if status in {"completed", "failed", "cancelled"}:
                    yield f"event: end\\ndata: {status}\\n\\n"
                    break
                await asyncio.sleep(1)

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @app.get("/artifacts")
    def artifacts(runId: str) -> dict[str, Any]:
        run = run_store.get(runId)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")

        if not run.output_dir:
            return {"files": []}

        output_path = Path(safe_path(run.output_dir))
        if not output_path.exists():
            return {"files": []}

        files = []
        for path in output_path.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mp4", ".json", ".gif", ".png", ".jpg", ".jpeg"}:
                files.append(
                    {
                        "path": str(path),
                        "relative": str(path.relative_to(output_path)),
                        "size_bytes": path.stat().st_size,
                    }
                )

        return {"files": sorted(files, key=lambda x: x["relative"])}

    @app.get("/data/sequences")
    def list_sequences(root: str) -> dict[str, Any]:
        root_path = path_under_project(root)
        if not root_path.exists() or not root_path.is_dir():
            raise HTTPException(status_code=400, detail=f"Invalid data root directory: {root}")

        sequences: list[dict[str, Any]] = []
        for child in sorted(root_path.iterdir()):
            if not child.is_dir():
                continue

            input_video = child / "input_video.mp4"
            quadmask = child / "quadmask_0.mp4"
            prompt_json = child / "prompt.json"
            first_frame = child / "first_frame.jpg"

            sequences.append(
                {
                    "name": child.name,
                    "path": str(child),
                    "has_input_video": input_video.exists(),
                    "has_quadmask": quadmask.exists(),
                    "has_prompt": prompt_json.exists(),
                    "has_first_frame": first_frame.exists(),
                    "prompt_bg": load_prompt_bg(prompt_json),
                }
            )

        return {"root": str(root_path), "sequences": sequences}

    @app.post("/data/prompt")
    def update_prompt(req: PromptUpdateRequest) -> dict[str, Any]:
        sequence_path = path_under_project(req.sequence_path)
        if not sequence_path.exists() or not sequence_path.is_dir():
            raise HTTPException(status_code=400, detail="Sequence folder not found")

        prompt_path = sequence_path / "prompt.json"
        payload = {"bg": req.bg.strip()}
        prompt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"ok": True, "path": str(prompt_path), "prompt": payload}

    @app.get("/cache/info")
    def cache_info(path: str) -> dict[str, Any]:
        cache_path = path_under_project(path)
        stats = directory_stats(cache_path)
        return {"path": str(cache_path), **stats}

    @app.post("/cache/clear")
    def clear_cache(req: CachePathRequest) -> dict[str, Any]:
        cache_path = path_under_project(req.path)
        if cache_path.exists() and cache_path.is_dir():
            for child in cache_path.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
        else:
            cache_path.mkdir(parents=True, exist_ok=True)

        stats = directory_stats(cache_path)
        return {"ok": True, "path": str(cache_path), **stats}

    return app


def subprocess_run_torch_cuda() -> dict[str, Any]:
    import subprocess

    probe = subprocess.run(
        ["python", "-c", "import torch; print('true' if torch.cuda.is_available() else 'false')"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return {"ok": True, "available": probe.stdout.strip() == "true"}
    return {"ok": False, "error": probe.stderr.strip() or probe.stdout.strip()}
