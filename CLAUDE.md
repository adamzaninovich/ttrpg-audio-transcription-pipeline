# CLAUDE.md

GPU-accelerated audio transcription using faster-whisper (large-v3). Primary use case is TTRPG session recordings — multitrack audio (one file per speaker from Craig bot). No diarization.

Two interfaces: **CLI** (`transcribe.py` via `docker compose run --rm transcribe` from the source dir) and **HTTP service** (`server.py` FastAPI app, deployed via `scripts/install`).

## Architecture

- `transcribe.py` — entire transcription pipeline, single file. Also imported by server.py.
- `server.py` — FastAPI HTTP service. Single background worker thread (one job at a time — GPU is the bottleneck). Temp dirs cleaned up after each job.
- `scripts/install <dest>` — builds the image and writes a deploy-ready `compose.yaml` (serve only, no `build:` context) to the destination. This repo is the build dir; the install dir is the run dir.
- Source `compose.yaml` services all use `profiles: [manual]` — bare `docker compose build` is a no-op; always specify the service name. The installed compose has no profiles.

## Pitfalls

- **`.dockerignore` invalidates all BuildKit cache layers** when changed — the context fingerprint changes even if the Python files didn't. Be deliberate about edits to it.
- **`nvidia-smi` won't work on the NixOS host** — run it inside the container.
- **VRAM cleanup** after unloading a model requires `del model` + `torch.cuda.empty_cache()` + `gc.collect()` — missing any of these causes VRAM leaks between files.
- **`transcribe.py` still contains diarization code** (`Config.use_diarization`, pyannote imports, etc.) — the HTTP service hardcodes `use_diarization=False` and the CLI reads it from the `SPEAKERS` env var. Don't remove that code without checking both callers.

## Environment

- RTX 4090 on a Linux machine running NixOS in WSL2, Docker Desktop
- Model cache (`models/`) is bind-mounted and persists across runs (~3GB for whisper large-v3)
