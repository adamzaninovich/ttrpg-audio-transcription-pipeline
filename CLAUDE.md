# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

GPU-accelerated audio transcription pipeline using faster-whisper (large-v3) with optional speaker diarization via pyannote.audio. Runs in Docker with NVIDIA GPU passthrough. Primary use case is TTRPG session recordings.

Two modes: **multitrack** (one file per speaker, e.g. Craig bot) and **single file** (mixed audio with diarization via `SPEAKERS` env var).

## Build and Run

```bash
# Build the Docker image
docker compose build transcribe

# Multitrack mode (no diarization)
INPUT_DIR=/path/to/audio OUTPUT_DIR=/path/to/output docker compose run --rm transcribe

# Single file with diarization
INPUT_DIR=/path/to/audio OUTPUT_DIR=/path/to/output SPEAKERS=5 docker compose run --rm transcribe

# Verify GPU/library setup
docker compose run --rm check
```

All services use `profiles: [manual]` — you must specify the service name explicitly (bare `docker compose build` is a no-op).

### CLI wrappers (`scripts/`)

`scripts/transcribe-audio` — end-to-end wrapper that validates args, sources `.env`, builds Docker image, and runs transcription. Output goes to `<input-dir>/transcripts/`.

```bash
scripts/transcribe-audio ~/sessions/craig-session/          # multitrack
scripts/transcribe-audio --speakers 5 ~/sessions/mixed/     # diarization
scripts/transcribe-audio -p ~/sessions/craig-session/       # transcribe + process
```

`scripts/process-transcript` — sends transcript JSON to an n8n webhook for LLM processing into structured notes. Handles interactive speaker name mapping for diarized transcripts (saves to `speaker_mapping.json` for reuse). Requires auth token at `/run/secrets/n8n-webhook-token` (sops-nix managed).

```bash
scripts/process-transcript ~/sessions/session-1/            # interactive speaker mapping
scripts/process-transcript -t ~/sessions/session-1/         # use test webhook
```

## Architecture

The entire pipeline lives in `transcribe.py` (single file, no modules). It runs inside a Docker container built from `Dockerfile` (CUDA 12.8.1 + Ubuntu 24.04).

```
scripts/transcribe-audio (bash wrapper)
  │
  ├── validates args, finds audio files, sources .env for HF_TOKEN
  ├── docker compose build [--quiet] transcribe
  └── docker compose run --rm transcribe
        │
        └── transcribe.py (inside container)
              │
              ├── ffmpeg: convert input → 16kHz mono WAV (temp file)
              ├── faster-whisper: WAV → word-level transcript
              ├── pyannote: WAV → in-memory waveform → speaker diarization
              ├── merge: overlap-based word→speaker assignment
              └── output: per-speaker JSON + SRT, merged JSON + SRT (multitrack)
```

**Pipeline per audio file:**
1. **WAV conversion**: `ffmpeg -i input -ar 16000 -ac 1 -y tmp.wav` — done once, used by both whisper and diarization
2. **Whisper transcription**: Loads large-v3 on GPU (float16), transcribes with word-level timestamps, VAD filter, optional `initial_prompt` from vocab.txt. Progress bar via `rich` tracks `segment.end / duration`
3. **Model swap**: Whisper unloaded, VRAM freed (`del` + `torch.cuda.empty_cache()` + `gc.collect()`), then pyannote loaded
4. **Audio into memory**: WAV loaded via `torchaudio.load()` → waveform tensor, passed to pyannote as `{"waveform": tensor, "sample_rate": 16000}`. Temp WAV deleted. Eliminates pyannote's crop() I/O bottleneck
5. **Diarization**: pyannote/speaker-diarization-3.1 with batch sizes 32 for segmentation and embedding. ProgressHook shows per-stage progress bars
6. **pyannote 4.x handling**: Pipeline returns `DiarizeOutput` dataclass, not bare `Annotation`. Extract via `result.speaker_diarization` (checked with `hasattr` for 3.x compat)
7. **Speaker-word merge**: Overlap-based — for each word's `[start, end]`, compute overlap with each speaker's turns, assign speaker with most overlap. Fallback: snap to nearest turn within 0.5s. Words >0.5s from any turn → UNKNOWN
8. **Re-segmentation**: Consecutive words from same speaker grouped into segments. Correctly splits whisper segments that span speaker turns
9. **Output**: Per-speaker `{name}.json` and `{name}.srt`. Multitrack also writes `merged.json` and `merged.srt` (all segments sorted by timestamp)

**Key design decisions:**
- **In-memory waveform** passed to pyannote instead of file path (diarization went from 5+ hours → ~1.5 min for 3-hour file)
- **Overlap-based speaker assignment** (not midpoint) handles crosstalk correctly — speaker whose turn covers more of the word wins. Also reduces UNKNOWNs since words only need *any* overlap, not midpoint-in-turn
- **WAV conversion upfront**: Both whisper and pyannote use the same WAV. Avoids double-decoding compressed audio
- **One model at a time**: Whisper (~3GB VRAM) and pyannote (~2GB VRAM) don't fit simultaneously with comfortable headroom

**Vocabulary hints:** Place `vocab.txt` in the input directory (one term per line, `#` comments). Terms are joined with commas and passed to Whisper's `initial_prompt`. Keep to ~20-50 terms (shares Whisper's 448-token context window).

## Environment

- **GPU**: RTX 4090 (24GB VRAM). Whisper ~3GB + pyannote ~2GB run sequentially.
- **System**: 32GB RAM, WSL2 configured for 24GB (`~/.wslconfig`). Docker Desktop on NixOS WSL distro.
- `nvidia-smi` must be run inside the container (NixOS can't run it on host)
- Container paths: `/input` (ro), `/output`, `/models` (whisper cache), `/hf-cache` (pyannote cache)
- Host bind mounts configured in `compose.yaml`
- `HF_TOKEN` in `.env` required only for diarization (`SPEAKERS` > 1). Must also accept model licenses for `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and `pyannote/speaker-diarization-community-1`
- `.dockerignore` allows only `transcribe.py` into the build context. Changes to `.dockerignore` invalidate BuildKit cache for all layers (context fingerprint changes)
- Model caches: `models/` (~3GB whisper large-v3), `hf-cache/` (~600MB pyannote) — bind-mounted volumes, persist across runs

## Nix Wrapper

A `transcribe-audio` CLI command is also provided by a Nix wrapper at `~/.config/snowfall/packages/transcribe-audio/default.nix` (`writeShellScriptBin`). Added to `home.packages` via `homes/x86_64-linux/adam@wsl/default.nix` as `bravo.transcribe-audio`.

```bash
# Rebuild Nix wrapper after editing it
sudo nixos-rebuild switch --flake ~/.config/snowfall#wsl
```

## Supported Formats

flac, wav, mp3, ogg, m4a, opus, webm

## Test Sessions

- `~/sessions/test-session/` — ~3 hours, 5 speakers, single mixed m4a file
- `~/sessions/craig-test/` — multitrack (one flac per speaker from Craig bot)

## Potential Improvements

- **LLM post-processing**: Pass transcript through LLM to correct speaker assignments using semantic context (conversational flow, TTRPG patterns). Optional `--llm-correct` pass via local ollama or API.
- **Speaker name mapping**: `--speakers-file speakers.json` for mapping `SPEAKER_00` → actual names.
- **Pin pyannote version**: Currently unpinned in Dockerfile, so pip pulls latest. Pinning prevents surprise API changes.

## Key References

- [pyannote #1403](https://github.com/pyannote/pyannote-audio/issues/1403) — crop() I/O bottleneck (why we use in-memory waveforms)
- [pyannote #1452](https://github.com/pyannote/pyannote-audio/issues/1452) — diarization extremely slow (loading whisper first helps)
- [pyannote #1486](https://github.com/pyannote/pyannote-audio/issues/1486) — increasing embedding batch size
- [pyannote #1963](https://github.com/pyannote/pyannote-audio/issues/1963) — 6x VRAM regression in 4.0.x
