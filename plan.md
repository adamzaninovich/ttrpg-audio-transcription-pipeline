# Plan: Normalize result shape and remove diarization field

## Goal
The transcription service should always return the same result shape regardless of how many files are uploaded, and should not include the `diarization` field (which is always `false` in the HTTP service).

## Current behavior

### Multi-file result (from `write_merged_output` in transcribe.py:538)
```json
{
  "speakers": ["Adam", "Nate"],
  "duration": 900.032,
  "diarization": false,
  "segments": [ ... ],
  "srt": "..."
}
```

### Single-file result (built inline in server.py:119-125)
```json
{
  "speaker": "Adam",
  "audio_file": "Adam.flac",
  "duration": 900.032,
  "diarization": false,
  "segments": [ ... ],
  "srt": "..."
}
```

These differ in: `speakers` (array) vs `speaker` (string), presence of `audio_file`, and `diarization` is in both.

## Target result shape
```json
{
  "speakers": ["Adam", "Nate"],
  "duration": 900.032,
  "segments": [ ... ],
  "srt": "..."
}
```
Single-file would return `"speakers": ["Adam"]` (array with one element). No `audio_file`, no `diarization`.

## Changes needed

### 1. server.py — single-file branch (lines 117-126)
The `else` branch currently builds its own dict. Change it to match the multi-file shape:
```python
else:
    r = results[0]
    final_result = {
        "speakers": [r.speaker_label],
        "duration": r.duration,
        "segments": r.segments,
    }
    srt_path = workdir / f"{r.speaker_label}.srt"
```

### 2. transcribe.py — `write_merged_output` (line 547-552)
Remove `"diarization"` from the `merged_result` dict:
```python
merged_result = {
    "speakers": [r.speaker_label for r in results],
    "duration": max_duration,
    "segments": merged_segments,
}
```

### 3. API.md — already correct
The docs already show the target shape. No changes needed after steps 1-2.

## Files to modify
- `server.py` — single-file result branch (~line 119)
- `transcribe.py` — `write_merged_output` (~line 547)

## What NOT to change
- The CLI caller (`transcribe.py` main) also uses `write_merged_output` — removing `diarization` from that dict is fine since the CLI writes it to a JSON file and nothing downstream depends on that field.
- `write_outputs` (per-speaker output in the CLI) still includes `diarization` in its dict — this function isn't used by the HTTP service, so leave it alone unless we want to clean it up too.
- The `diarization` code paths in `transcribe.py` (Config.use_diarization, pyannote imports, etc.) should NOT be removed — the CLI still supports them via the SPEAKERS env var.

## Testing
After changes, submit a job with 1 file and with 2+ files. Both should return the same shape with `speakers` as an array and no `diarization` field. Verify the SRT is still present in both cases.
