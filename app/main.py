"""
main.py
-------
FastAPI server exposing video upload, job status, and result retrieval.

Endpoints:
  POST /analyze          — upload a football play video, get a job_id back
  GET  /result/{job_id}  — poll for completion; returns JSON report + video URL
  GET  /video/{job_id}   — stream the annotated video
  GET  /health           — liveness check
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.analyzer import analyze_video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Simple in-memory job store  {job_id: {"status": ..., "result": ...}}
_jobs: dict[str, dict] = {}

app = FastAPI(
    title="Football AI — Defensive Effort Analyzer",
    description="Upload a football play clip and get per-player effort scores for blue defensive players.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend
app.mount("/static", StaticFiles(directory="frontend"), name="static")

_executor = ProcessPoolExecutor(max_workers=2)


# ── helpers ────────────────────────────────────────────────────────────

def _run_analysis(input_path: str, job_id: str) -> dict:
    """Runs in a separate process (CPU-bound)."""
    return analyze_video(
        input_path=input_path,
        output_dir=str(OUTPUT_DIR),
        job_id=job_id,
    )


async def _background_analyze(job_id: str, video_path: Path) -> None:
    _jobs[job_id]["status"] = "processing"
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            _executor, _run_analysis, str(video_path), job_id
        )
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
    except Exception as exc:
        logger.exception("Analysis failed for job %s", job_id)
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["error"] = str(exc)


# ── routes ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze", status_code=202)
async def analyze(file: UploadFile = File(...)):
    """
    Upload a football play video (.mp4 / .mov / .avi).
    Returns a job_id you can poll with GET /result/{job_id}.
    """
    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Use .mp4, .mov, .avi, or .mkv.",
        )

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}{ext}"

    # Save uploaded file
    async with aiofiles.open(video_path, "wb") as out:
        content = await file.read()
        await out.write(content)

    logger.info("Received upload: %s → %s (job=%s)", file.filename, video_path, job_id)
    _jobs[job_id] = {"status": "queued", "filename": file.filename}

    asyncio.create_task(_background_analyze(job_id, video_path))

    return JSONResponse(
        status_code=202,
        content={"job_id": job_id, "status": "queued"},
    )


@app.get("/result/{job_id}")
async def result(job_id: str):
    """
    Poll job status. Possible statuses: queued | processing | done | error.
    When done, includes the full effort report and a link to the annotated video.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found.")

    job = _jobs[job_id]
    if job["status"] != "done":
        return {"job_id": job_id, "status": job["status"]}

    r = job["result"]
    return {
        "job_id": job_id,
        "status": "done",
        "processing_time_seconds": r["processing_time_seconds"],
        "player_reports": r["player_reports"],
        "video_url": f"/video/{job_id}",
    }


@app.get("/video/{job_id}")
async def get_video(job_id: str):
    """Stream the annotated output video."""
    if job_id not in _jobs or _jobs[job_id]["status"] != "done":
        raise HTTPException(status_code=404, detail="Video not ready.")

    video_path = Path(_jobs[job_id]["result"]["output_video"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file missing.")

    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=video_path.name,
    )


@app.get("/")
async def root():
    """Redirect to the frontend."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
