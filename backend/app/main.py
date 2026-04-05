from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_ROOT.parent
STATE_DIR = BACKEND_ROOT / "state"
LOG_DIR = STATE_DIR / "logs"
RUNS_FILE = STATE_DIR / "runs.json"

LOG_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Pass1Params(BaseModel):
    config_path: str = "config/quadmask_cogvideox.py"
    config_overrides: dict[str, Any] = Field(default_factory=dict)


class Pass2Params(BaseModel):
    video_name: Optional[str] = None
    video_names: list[str] = Field(default_factory=list)
    data_rootdir: str = "./data"
    pass1_dir: str = "./pass1_outputs"
    output_dir: str = "./inference_with_warped_noise"
    model_name: str = "./CogVideoX-Fun-V1.5-5b-InP"
    model_checkpoint: str
    max_video_length: int = 197
    temporal_window_size: int = 85
    height: int = 384
    width: int = 672
    seed: int = 42
    guidance_scale: float = 6.0
    num_inference_steps: int = 50
    warped_noise_cache_dir: str = "./pass1_warped_noise_cache"
    skip_noise_generation: bool = False
    use_quadmask: bool = True


class MaskPipelineParams(BaseModel):
    config_points_json: str
    sam2_checkpoint: str = "../sam2_hiera_large.pt"
    device: Literal["cuda", "cpu"] = "cuda"


class RunCreateRequest(BaseModel):
    workflow: Literal["pass1_inference", "pass2_refine", "mask_pipeline"]
    params: dict[str, Any]
    name: Optional[str] = None


@dataclass
class RunRecord:
    id: str
    workflow: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    created_at: str
    started_at: Optional[str]
    ended_at: Optional[str]
    exit_code: Optional[int]
    command: list[str]
    name: Optional[str]
    params: dict[str, Any]
    log_path: str
    output_dir: Optional[str]
    error: Optional[str] = None


class RunStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}
        self._procs: dict[str, subprocess.Popen[str]] = {}
        self._load()

    def _load(self) -> None:
        if not RUNS_FILE.exists():
            return
        try:
            payload = json.loads(RUNS_FILE.read_text())
            for item in payload:
                record = RunRecord(**item)
                # Old process state cannot be resumed safely after restart.
                if record.status == "running":
                    record.status = "failed"
                    record.error = "Backend restarted while run was active"
                    record.ended_at = now_iso()
                self._runs[record.id] = record
        except Exception:
            self._runs = {}

    def _save(self) -> None:
        RUNS_FILE.write_text(
            json.dumps([asdict(v) for v in self._runs.values()], indent=2)
        )

    def list(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, run_id: str) -> Optional[RunRecord]:
        with self._lock:
            return self._runs.get(run_id)

    def create(self, record: RunRecord) -> None:
        with self._lock:
            self._runs[record.id] = record
            self._save()

    def set_proc(self, run_id: str, proc: subprocess.Popen[str]) -> None:
        with self._lock:
            self._procs[run_id] = proc

    def get_proc(self, run_id: str) -> Optional[subprocess.Popen[str]]:
        with self._lock:
            return self._procs.get(run_id)

    def clear_proc(self, run_id: str) -> None:
        with self._lock:
            self._procs.pop(run_id, None)

    def patch(self, run_id: str, **updates: Any) -> None:
        with self._lock:
            run = self._runs[run_id]
            for k, v in updates.items():
                setattr(run, k, v)
            self._save()


store = RunStore()


app = FastAPI(title="VOID Frontend Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def safe_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def override_args(config_overrides: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in config_overrides.items():
        if not key.startswith("config."):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid override key '{key}'. Keys must start with 'config.'",
            )
        if isinstance(value, bool):
            normalized = "true" if value else "false"
        else:
            normalized = str(value)
        args.append(f"--{key}={normalized}")
    return args


def build_command(workflow: str, params: dict[str, Any]) -> tuple[list[str], Optional[str]]:
    if workflow == "pass1_inference":
        p = Pass1Params(**params)
        cmd = [
            "python",
            "inference/cogvideox_fun/predict_v2v.py",
            "--config",
            p.config_path,
            *override_args(p.config_overrides),
        ]
        output_dir = p.config_overrides.get("config.experiment.save_path")
        return cmd, str(output_dir) if output_dir else None

    if workflow == "pass2_refine":
        p = Pass2Params(**params)
        if not p.video_name and not p.video_names:
            raise HTTPException(
                status_code=400,
                detail="Provide either video_name or video_names for pass2_refine",
            )
        cmd = [
            "python",
            "inference/cogvideox_fun/inference_with_pass1_warped_noise.py",
            "--data_rootdir",
            p.data_rootdir,
            "--pass1_dir",
            p.pass1_dir,
            "--output_dir",
            p.output_dir,
            "--model_name",
            p.model_name,
            "--model_checkpoint",
            p.model_checkpoint,
            "--max_video_length",
            str(p.max_video_length),
            "--temporal_window_size",
            str(p.temporal_window_size),
            "--height",
            str(p.height),
            "--width",
            str(p.width),
            "--seed",
            str(p.seed),
            "--guidance_scale",
            str(p.guidance_scale),
            "--num_inference_steps",
            str(p.num_inference_steps),
            "--warped_noise_cache_dir",
            p.warped_noise_cache_dir,
        ]
        if p.video_name:
            cmd.extend(["--video_name", p.video_name])
        if p.video_names:
            cmd.extend(["--video_names", *p.video_names])
        if p.skip_noise_generation:
            cmd.append("--skip_noise_generation")
        if p.use_quadmask:
            cmd.append("--use_quadmask")
        return cmd, p.output_dir

    if workflow == "mask_pipeline":
        p = MaskPipelineParams(**params)
        cmd = [
            "bash",
            "VLM-MASK-REASONER/run_pipeline.sh",
            p.config_points_json,
            "--sam2-checkpoint",
            p.sam2_checkpoint,
            "--device",
            p.device,
        ]
        return cmd, None

    raise HTTPException(status_code=400, detail=f"Unsupported workflow: {workflow}")


def run_worker(run_id: str, cmd: list[str], log_path: Path) -> None:
    store.patch(run_id, status="running", started_at=now_iso())
    env = os.environ.copy()

    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(cmd)}\n\n")
        log_file.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        store.set_proc(run_id, proc)

        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
            exit_code = proc.wait()
        finally:
            store.clear_proc(run_id)

    if store.get(run_id) and store.get(run_id).status == "cancelled":
        return

    if exit_code == 0:
        store.patch(run_id, status="completed", ended_at=now_iso(), exit_code=0)
    else:
        store.patch(
            run_id,
            status="failed",
            ended_at=now_iso(),
            exit_code=exit_code,
            error=f"Process exited with code {exit_code}",
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
        probe = subprocess.run(
            [
                "python",
                "-c",
                "import torch; print('true' if torch.cuda.is_available() else 'false')",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if probe.returncode == 0:
            cuda["available"] = probe.stdout.strip() == "true"
        else:
            cuda["error"] = probe.stderr.strip() or probe.stdout.strip()
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
    return [asdict(r) for r in store.list()]


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return asdict(run)


@app.post("/runs")
def create_run(req: RunCreateRequest) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    command, output_dir = build_command(req.workflow, req.params)
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
    store.create(record)

    thread = threading.Thread(target=run_worker, args=(run_id, command, log_path), daemon=True)
    thread.start()

    return asdict(record)


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    proc = store.get_proc(run_id)
    if not proc:
        return asdict(run)

    try:
        if os.name != "nt":
            os.kill(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        pass

    store.patch(run_id, status="cancelled", ended_at=now_iso(), exit_code=-15)
    return asdict(store.get(run_id))


@app.get("/runs/{run_id}/logs")
def read_logs(run_id: str, tail: int = 400) -> dict[str, Any]:
    run = store.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    log_path = Path(run.log_path)
    if not log_path.exists():
        return {"lines": [], "text": ""}

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    selected = lines[-tail:]
    return {"lines": selected, "text": "\n".join(selected)}


@app.get("/runs/{run_id}/logs/stream")
async def stream_logs(run_id: str):
    run = store.get(run_id)
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
                        yield f"data: {line}\n\n"
                    last_size = len(current)

            status = store.get(run_id).status
            if status in {"completed", "failed", "cancelled"}:
                yield f"event: end\ndata: {status}\n\n"
                break
            await asyncio.sleep(1)

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.get("/artifacts")
def artifacts(runId: str) -> dict[str, Any]:
    run = store.get(runId)
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
