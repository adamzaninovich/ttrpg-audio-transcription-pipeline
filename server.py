#!/usr/bin/env python3
"""FastAPI HTTP service for audio transcription.

Accepts multipart uploads of audio files, runs transcription in a background
worker thread (one job at a time), and streams progress via SSE.
"""

import asyncio
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from transcribe import (
    Config,
    FileResult,
    load_vocab,
    process_file,
    write_merged_output,
)

app = FastAPI(title="Transcription Service")


@app.get("/version")
async def version():
    return {"version": "2"}


# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    status: str  # "queued" | "running" | "done" | "failed"
    result: dict | None = None
    error: str | None = None
    workdir: str | None = None
    events: list[dict] = field(default_factory=list)
    _event: threading.Event = field(default_factory=threading.Event)


jobs: dict[str, Job] = {}
job_queue: queue.Queue[str] = queue.Queue()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _build_config(workdir: Path, vocab_prompt: str | None) -> Config:
    model_cache = os.environ.get("MODEL_CACHE_DIR", "/models")
    return Config(
        model_name="large-v3",
        device="cuda",
        compute_type="float16",
        sample_rate=16000,
        max_gap=1.5,
        snap_distance=0.5,
        seg_batch_size=32,
        emb_batch_size=32,
        min_speakers=1,
        max_speakers=1,
        use_diarization=False,
        hf_token="",
        vocab_prompt=vocab_prompt,
        input_dir=workdir,
        output_dir=workdir,
        model_cache=model_cache,
    )


def _run_job(job: Job) -> None:
    workdir = Path(job.workdir)

    def emit(event: dict) -> None:
        job.events.append(event)
        job._event.set()
        job._event.clear()

    try:
        job.status = "running"

        audio_files = sorted(
            p for p in workdir.iterdir()
            if p.is_file() and p.suffix.lower()
            in {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}
        )
        if not audio_files:
            raise ValueError("No audio files found in upload")

        # Load vocab from workdir if present
        vocab_prompt = load_vocab(workdir)
        config = _build_config(workdir, vocab_prompt)

        results: list[FileResult] = []
        for i, audio_path in enumerate(audio_files):
            result = process_file(
                audio_path, config,
                on_event=emit,
                file_index=i,
                file_total=len(audio_files),
            )
            results.append(result)

        if len(results) > 1:
            final_result = write_merged_output(results, config)
            srt_path = workdir / "merged.srt"
        else:
            r = results[0]
            final_result = {
                "speaker": r.speaker_label,
                "audio_file": r.audio_file,
                "duration": r.duration,
                "diarization": False,
                "segments": r.segments,
            }
            srt_path = workdir / f"{r.speaker_label}.srt"

        print(f"SRT path: {srt_path}, exists: {srt_path.exists()}")
        srt_text = srt_path.read_text(encoding="utf-8")
        print(f"SRT length: {len(srt_text)}")
        final_result["srt"] = srt_text
        print(f"Result keys after SRT: {list(final_result.keys())}")
        job.result = final_result
        job.status = "done"
        emit({"type": "done", "result": final_result})

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        emit({"type": "error", "error": str(exc)})

    finally:
        try:
            shutil.rmtree(job.workdir, ignore_errors=True)
        except Exception:
            pass


def _worker() -> None:
    while True:
        job_id = job_queue.get()
        job = jobs.get(job_id)
        if job is None:
            continue
        try:
            _run_job(job)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.events.append({"type": "error", "error": str(exc)})
            job._event.set()


_worker_thread = threading.Thread(target=_worker, daemon=True)
_worker_thread.start()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/jobs")
async def create_job(
    files: list[UploadFile] = File(..., alias="files[]"),
) -> JSONResponse:
    """Accept audio files + optional vocab.txt, enqueue a transcription job."""
    job_id = uuid.uuid4().hex[:8]
    workdir = tempfile.mkdtemp(prefix=f"trans_{job_id}_")

    # Save uploaded files (audio + optional vocab.txt)
    for upload in files:
        dest = Path(workdir) / upload.filename
        with open(dest, "wb") as f:
            content = await upload.read()
            f.write(content)

    job = Job(id=job_id, status="queued", workdir=workdir)
    job.events.append({"type": "queued"})
    jobs[job_id] = job
    job_queue.put(job_id)

    return JSONResponse({"job_id": job_id})


@app.get("/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    """SSE stream of progress events for a job."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def stream() -> AsyncGenerator[str, None]:
        cursor = 0
        while True:
            new = job.events[cursor:]
            for ev in new:
                yield f"data: {json.dumps(ev)}\n\n"
                cursor += 1
                if ev.get("type") in ("done", "error"):
                    return
            if job.status in ("done", "failed") and cursor >= len(job.events):
                break
            await asyncio.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/jobs/{job_id}")
async def get_job(job_id: str) -> JSONResponse:
    """Polling fallback — returns current job status and result."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return JSONResponse({
        "status": job.status,
        "result": job.result,
        "error": job.error,
    })
