# Transcription Service API

Base URL: `http://<host>:8000`

## Endpoints

### POST /jobs

Submit audio files for transcription. Jobs are processed sequentially (one at a time). Upload one file per speaker — the filename stem becomes the speaker label (e.g. `Adam.flac` → speaker "Adam").

**Request:** `multipart/form-data`

| Field      | Type   | Required | Description |
|------------|--------|----------|-------------|
| `files[]`  | file   | yes      | Audio files (.flac, .wav, .mp3, .ogg, .m4a, .opus, .webm). One per speaker. |
| `files[]`  | file   | no       | Optional `vocab.txt` — one word/phrase per line. Primes the model to recognize these terms. |

**Response:** `200 OK`
```json
{"job_id": "e84569ef"}
```

**Example:**
```bash
curl -X POST http://host:8000/jobs \
  -F "files[]=@Adam.flac" \
  -F "files[]=@Nate.flac" \
  -F "files[]=@vocab.txt"
```

---

### GET /jobs/{job_id}/events

SSE stream of real-time progress events. Connect after submitting a job. The stream closes after a `done` or `error` event.

**Response:** `text/event-stream`

Each event is a JSON object on a `data:` line. Event types in order:

#### `queued`
Job is waiting for the worker.
```json
{"type": "queued"}
```

#### `file_start`
Transcription of a file has begun.
```json
{"type": "file_start", "file": "Adam.flac", "index": 0, "total": 2}
```
- `file` — filename being transcribed
- `index` — 0-based file index
- `total` — total number of audio files

#### `progress`
Transcription progress for the current file. Emitted after each segment whisper produces — frequency depends on speech density (long silences cause jumps).
```json
{"type": "progress", "pct": 37}
```
- `pct` — 0–100, percentage of the current file transcribed

#### `file_done`
One file finished.
```json
{"type": "file_done", "file": "Adam.flac", "duration": 900.032, "elapsed": 9.8}
```
- `duration` — audio length in seconds
- `elapsed` — wall-clock processing time in seconds

#### `done`
Job complete. Contains the full result (same as polling endpoint).
```json
{"type": "done", "result": { ... }}
```

#### `error`
Job failed.
```json
{"type": "error", "error": "No audio files found in upload"}
```

**Overall progress formula:**
```
overall_pct = (completed_files + current_pct / 100) / total_files * 100
```

---

### GET /jobs/{job_id}

Polling fallback. Returns current job state.

**Response:** `200 OK`
```json
{
  "status": "done",
  "result": { ... },
  "error": null
}
```
- `status` — `"queued"` | `"running"` | `"done"` | `"failed"`
- `result` — null until done, then the result object (see below)
- `error` — null unless failed

---

## Result Object

Returned in the `done` SSE event and the polling endpoint.

```json
{
  "speakers": ["Adam", "Nate"],
  "duration": 900.032,
  "segments": [ ... ],
  "srt": "1\n00:00:02,640 --> 00:00:03,899\n[Adam] I love it so much,\n\n..."
}
```

### Fields
- `speakers` — speaker labels, derived from filename stems
- `duration` — audio duration in seconds
- `srt` — full SRT subtitle text with `[Speaker]` prefixes
- `segments` — array of transcript segments:

```json
{
  "start": 2.64,
  "end": 3.9,
  "text": "I love it so much,",
  "speaker": "Adam",
  "words": [
    {"start": 2.64, "end": 2.98, "word": " I", "probability": 0.9888},
    {"start": 2.98, "end": 3.24, "word": " love", "probability": 0.9995}
  ]
}
```
- `start` / `end` — seconds
- `text` — segment text
- `speaker` — speaker label
- `words` — word-level timestamps with confidence scores (0–1)

## Notes

- **One job at a time.** The GPU is the bottleneck. Additional jobs queue and process in order.
- **Supported formats:** .flac, .wav, .mp3, .ogg, .m4a, .opus, .webm
- **Speaker labels** come from filenames — designed for multitrack recordings (e.g. Craig bot) where each file is one speaker.
- **vocab.txt** helps with proper nouns, game terms, etc. One term per line.
