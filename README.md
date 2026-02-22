# transcribe

GPU-accelerated audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (large-v3) with optional speaker diarization via [pyannote.audio](https://github.com/pyannote/pyannote-audio). Runs in Docker with NVIDIA GPU passthrough.

![Screen Recording 2026-02-22 005431_50x](https://github.com/user-attachments/assets/77a4f308-b863-46d1-be04-67ed05666199)

## Modes

**Multitrack** (e.g., Craig bot recordings): One audio file per speaker. No diarization needed - speaker identity comes from the filename. Produces per-speaker transcripts plus a merged timeline.

**Single file** (e.g., phone/mic recording): One mixed audio file with multiple speakers. Uses pyannote diarization to separate speakers.

## Requirements

- Docker with NVIDIA GPU runtime (Docker Desktop or nvidia-container-toolkit)
- NVIDIA GPU with enough VRAM for the models (~5GB, see [Memory](#memory))
- [HuggingFace token](https://huggingface.co/settings/tokens) (only for diarization)

### HuggingFace setup (diarization only)

Create a `.env` file:

```
HF_TOKEN=hf_your_token_here
```

Accept the model licenses:
- [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

## Usage

```bash
# Build the image
docker compose build transcribe

# Multitrack - one file per speaker, no diarization
INPUT_DIR=/path/to/audio OUTPUT_DIR=/path/to/output \
  docker compose run --rm transcribe

# Single file - diarize into 5 speakers
INPUT_DIR=/path/to/audio OUTPUT_DIR=/path/to/output SPEAKERS=5 \
  docker compose run --rm transcribe

# Speaker range - let pyannote decide within bounds
INPUT_DIR=/path/to/audio OUTPUT_DIR=/path/to/output SPEAKERS=3-7 \
  docker compose run --rm transcribe
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `./input` | Directory containing audio files |
| `OUTPUT_DIR` | `./output` | Directory for transcript output |
| `SPEAKERS` | `1` | `1` = no diarization, `N` = exact count, `MIN-MAX` = range |
| `HF_TOKEN` | | HuggingFace token (required when `SPEAKERS` > 1) |

### Supported formats

flac, wav, mp3, ogg, m4a, opus, webm

## Output

Each audio file produces a JSON and SRT file named after the input file stem:

```
output/
  speaker1.json     # word-level timestamps, speaker labels
  speaker1.srt      # subtitle file
  speaker2.json
  speaker2.srt
  merged.json       # all speakers sorted by timestamp (multitrack only)
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
  "diarization": false,
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

## Pipeline

For each audio file:

1. **WAV conversion**: ffmpeg converts input to 16kHz mono WAV (used by both whisper and diarization)
2. **Transcription**: Whisper large-v3 on GPU (float16) with word-level timestamps and VAD filter
3. **Model swap**: Whisper unloaded from VRAM, then pyannote loaded (only one model at a time)
4. **Diarization** (if enabled): Audio loaded into memory as a waveform tensor, then pyannote runs speaker diarization with batch sizes of 32
5. **Speaker assignment**: Each word assigned to a speaker using overlap duration with diarization turns. Fallback: nearest turn within 0.5s
6. **Re-segmentation**: Consecutive words from the same speaker grouped into segments, correctly splitting whisper segments that span speaker turns
7. **Output**: Per-speaker JSON + SRT. Multitrack mode also writes merged JSON + SRT sorted by timestamp

### Why in-memory waveforms?

Pyannote's default behavior calls `torchaudio.load()` with `crop()` for every tiny audio slice during diarization - thousands of disk I/O calls. Loading the full WAV into memory first and passing it as `{"waveform": tensor, "sample_rate": 16000}` eliminates this bottleneck entirely. Diarization of a 3-hour file went from 5+ hours to ~1.5 minutes.

## Memory

Whisper large-v3 uses ~3GB VRAM. Pyannote uses ~2GB. They run sequentially (not simultaneously) to fit on GPUs with limited VRAM. System RAM usage peaks during the in-memory waveform load (~1GB for a 3-hour 16kHz mono WAV).

## Verify setup

```bash
docker compose run --rm check
```

Prints versions of faster-whisper, pyannote, torch, and CUDA availability.
