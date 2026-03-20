# transcribe

GPU-accelerated audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3). Runs in Docker with NVIDIA GPU passthrough. Primary use case is TTRPG session recordings from [Craig bot](https://craig.chat/) — one audio file per speaker, producing per-speaker transcripts plus a merged timeline.

![Screen Recording 2026-02-22 005431_50x](https://github.com/user-attachments/assets/77a4f308-b863-46d1-be04-67ed05666199)

## Usage modes

**CLI** — batch transcription, writes JSON + SRT to disk:

```bash
docker compose run --rm transcribe
```

**HTTP service** — accepts uploads, streams progress via SSE, returns transcript JSON. Intended for programmatic use (e.g., a Phoenix/Elixir app on the same LAN):

```bash
docker compose up serve
```

## Requirements

- Docker with NVIDIA GPU runtime (Docker Desktop or nvidia-container-toolkit)
- NVIDIA GPU with ~3GB VRAM

## CLI usage

```bash
# Build the image
docker compose build transcribe

# Transcribe all audio files in a directory
INPUT_DIR=/path/to/audio OUTPUT_DIR=/path/to/output \
  docker compose run --rm transcribe
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `./input` | Directory containing audio files |
| `OUTPUT_DIR` | `./output` | Directory for transcript output |

### Supported formats

flac, wav, mp3, ogg, m4a, opus, webm

## HTTP service

The `serve` Docker service exposes a REST API on port 8000. Jobs are processed one at a time (GPU bottleneck); progress streams via SSE.

### Start the server

```bash
docker compose build transcribe   # build image (shared with CLI)
docker compose up serve           # starts uvicorn on :8000
```

### Submit a job

```bash
curl -X POST http://localhost:8000/jobs \
  -F "files[]=@GM.flac" \
  -F "files[]=@Kai.flac" \
  -F "files[]=@Milo.flac"
# → {"job_id": "a1b2c3d4"}
```

Include an optional vocab hint file:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "files[]=@GM.flac" \
  -F "vocab=Xanathar, Phandalin, Eldritch Blast"
```

### Stream progress (SSE)

```bash
curl -N http://localhost:8000/jobs/a1b2c3d4/events
```

Event stream:

```
data: {"type": "queued"}
data: {"type": "file_start", "file": "GM.flac", "index": 0, "total": 3}
data: {"type": "progress", "pct": 12}
data: {"type": "progress", "pct": 34}
...
data: {"type": "file_done", "file": "GM.flac", "duration": 3600.0, "elapsed": 480.1}
data: {"type": "file_start", "file": "Kai.flac", "index": 1, "total": 3}
...
data: {"type": "done", "result": { ... merged transcript JSON ... }}
```

On failure: `data: {"type": "error", "error": "message"}`

### Poll for status

```bash
curl http://localhost:8000/jobs/a1b2c3d4
# → {"status": "running", "result": null, "error": null}
# → {"status": "done",    "result": { ... }, "error": null}
# → {"status": "failed",  "result": null, "error": "message"}
```

### Result format

For multiple files, `result` is the merged transcript:

```json
{
  "speakers": ["GM", "Kai", "Milo"],
  "duration": 10842.5,
  "segments": [ ... ]
}
```

For a single file, `result` is the per-speaker shape (same as the CLI JSON output).

## Output

Each audio file produces a JSON and SRT file named after the input file stem:

```
output/
  speaker1.json     # word-level timestamps, speaker labels
  speaker1.srt      # subtitle file
  speaker2.json
  speaker2.srt
  merged.json       # all speakers sorted by timestamp
  merged.srt
```

### JSON format

```json
{
  "speaker": "speaker1",
  "audio_file": "speaker1.flac",
  "language": "en",
  "language_probability": 0.9987,
  "duration": 10842.528,
  "segments": [
    {
      "speaker": "speaker1",
      "start": 0.0,
      "end": 4.52,
      "text": "The quick brown fox",
      "words": [
        { "start": 0.0, "end": 0.38, "word": " The", "probability": 0.99 },
        { "start": 0.38, "end": 0.72, "word": " quick", "probability": 0.97 }
      ]
    }
  ]
}
```

## Vocabulary hints

Place a `vocab.txt` file in the input directory to improve recognition of proper nouns, domain-specific terms, or commonly mis-transcribed words. The file is automatically detected.

Format: one term per line, `#` for comments, blank lines ignored.

```
# Characters
Xanathar
Phandalin

# Spells
Eldritch Blast
```

Terms are joined and passed to Whisper's `initial_prompt`, which conditions the model to prefer these spellings when it hears something phonetically similar. Keep to roughly 20-50 terms (the prompt shares Whisper's 448-token context window).

## Verify setup

```bash
docker compose run --rm check
```

Prints versions of faster-whisper, torch, and CUDA availability.
