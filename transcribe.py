#!/usr/bin/env python3
"""Batch transcription of multi-track audio files using faster-whisper.

Discovers audio files in /input, transcribes each with the large-v3 model
on GPU, and writes per-speaker JSON + SRT to /output.

Supports optional speaker diarization via pyannote.audio when SPEAKERS > 1.
"""

import gc
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch

from faster_whisper import WhisperModel

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}
VOCAB_FILENAME = "vocab.txt"
INPUT_DIR = Path("/input")
OUTPUT_DIR = Path("/output")
MODEL_CACHE = os.environ.get("MODEL_CACHE_DIR", "/models")


def load_vocab(input_dir: Path) -> str | None:
    """Load vocabulary hints from vocab.txt in the input directory.

    Returns a comma-separated string of terms for Whisper's initial_prompt,
    or None if no vocab file is found.
    """
    vocab_path = input_dir / VOCAB_FILENAME
    if not vocab_path.is_file():
        return None

    terms = []
    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        terms.append(line)

    if not terms:
        return None

    prompt = ", ".join(terms)
    print(f"Vocabulary: {len(terms)} terms from {VOCAB_FILENAME}")
    return prompt


def parse_speakers(speakers_str: str) -> tuple[int, int]:
    """Parse SPEAKERS env var: '1', '3', or '2-6' → (min, max)."""
    if "-" in speakers_str:
        parts = speakers_str.split("-", 1)
        return int(parts[0]), int(parts[1])
    n = int(speakers_str)
    return n, n


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def transcribe_file(model: WhisperModel, audio_path: Path, initial_prompt: str | None = None) -> dict:
    """Transcribe a single audio file and return structured result."""
    from rich.progress import Progress, BarColumn, TimeRemainingColumn, TextColumn

    transcribe_kwargs = dict(
        language="en",
        word_timestamps=True,
        vad_filter=True,
    )
    if initial_prompt:
        transcribe_kwargs["initial_prompt"] = initial_prompt

    segments_iter, info = model.transcribe(str(audio_path), **transcribe_kwargs)

    total_seconds = int(info.duration)
    segments = []

    with Progress(
        TextColumn("transcription"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeRemainingColumn(elapsed_when_finished=True),
    ) as progress:
        task = progress.add_task("transcription", total=total_seconds)

        for segment in segments_iter:
            words = []
            if segment.words:
                words = [
                    {
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "word": w.word,
                        "probability": round(w.probability, 4),
                    }
                    for w in segment.words
                ]

            segments.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                    "words": words,
                }
            )

            progress.update(task, completed=min(int(segment.end), total_seconds))

        progress.update(task, completed=total_seconds)

    return {
        "audio_file": audio_path.name,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration, 3),
        "segments": segments,
    }


def run_diarization(pipeline, audio_input, min_speakers: int, max_speakers: int, hook=None):
    """Run speaker diarization and return the annotation object.

    audio_input can be a file path string or a dict with 'waveform' and 'sample_rate'.
    """
    kwargs = {}
    if min_speakers > 0:
        kwargs["min_speakers"] = min_speakers
    if max_speakers > 0:
        kwargs["max_speakers"] = max_speakers
    if hook is not None:
        kwargs["hook"] = hook
    return pipeline(audio_input, **kwargs)


def get_speaker_for_word(word_start: float, word_end: float, speaker_timeline: list,
                         max_gap: float = 0.5) -> str:
    """Find the best speaker for a word using overlap duration.

    Computes how much each speaker's turns overlap with the word's time range.
    The speaker with the most overlap wins. If no speaker overlaps at all,
    snaps to the nearest turn boundary within max_gap seconds.
    """
    # Compute overlap duration per speaker
    overlaps = {}
    for turn_start, turn_end, speaker in speaker_timeline:
        if turn_end <= word_start or turn_start >= word_end:
            continue
        overlap = min(turn_end, word_end) - max(turn_start, word_start)
        overlaps[speaker] = overlaps.get(speaker, 0.0) + overlap

    if overlaps:
        return max(overlaps, key=overlaps.get)

    # No overlap — snap to nearest turn boundary
    best_speaker = "UNKNOWN"
    best_dist = max_gap
    for turn_start, turn_end, speaker in speaker_timeline:
        dist = min(abs(turn_start - word_end), abs(turn_end - word_start))
        if dist < best_dist:
            best_dist = dist
            best_speaker = speaker

    return best_speaker


def merge_with_diarization(whisper_result: dict, diarization) -> list:
    """Merge whisper segments with diarization speaker labels.

    Assigns a speaker to each word using overlap duration with diarization
    turns, then re-segments: consecutive words from the same speaker
    are grouped together. This correctly splits whisper segments
    that span speaker turns.
    """
    # Build speaker timeline from diarization
    speaker_timeline = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    if not speaker_timeline:
        return whisper_result["segments"]

    result_segments = []

    for seg in whisper_result["segments"]:
        if not seg.get("words"):
            # No word-level timestamps — assign speaker by segment overlap
            seg["speaker"] = get_speaker_for_word(
                seg["start"], seg["end"], speaker_timeline,
            )
            result_segments.append(seg)
            continue

        # Assign speaker to each word
        for word in seg["words"]:
            word["speaker"] = get_speaker_for_word(
                word["start"], word["end"], speaker_timeline,
            )

        # Re-segment by speaker turns within this whisper segment
        current_words = [seg["words"][0]]
        current_speaker = seg["words"][0]["speaker"]

        for word in seg["words"][1:]:
            if word["speaker"] == current_speaker:
                current_words.append(word)
            else:
                # Speaker changed — emit segment
                result_segments.append({
                    "speaker": current_speaker,
                    "start": current_words[0]["start"],
                    "end": current_words[-1]["end"],
                    "text": "".join(w["word"] for w in current_words).strip(),
                    "words": current_words,
                })
                current_speaker = word["speaker"]
                current_words = [word]

        # Emit final group
        if current_words:
            result_segments.append({
                "speaker": current_speaker,
                "start": current_words[0]["start"],
                "end": current_words[-1]["end"],
                "text": "".join(w["word"] for w in current_words).strip(),
                "words": current_words,
            })

    return result_segments


def write_json(result: dict, output_path: Path) -> None:
    """Write transcription result as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def write_srt(segments: list, output_path: Path, include_speaker: bool = False) -> None:
    """Write segments as an SRT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(
                f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n"
            )
            if include_speaker and "speaker" in seg:
                f.write(f"[{seg['speaker']}] {seg['text']}\n\n")
            else:
                f.write(f"{seg['text']}\n\n")


def main():
    # Parse configuration
    speakers_str = os.environ.get("SPEAKERS", "1")
    try:
        min_speakers, max_speakers = parse_speakers(speakers_str)
    except ValueError:
        print(f"Error: Invalid SPEAKERS value: {speakers_str!r}")
        print("Expected: N (e.g., 3) or MIN-MAX (e.g., 2-6)")
        sys.exit(1)

    use_diarization = max_speakers > 1
    hf_token = os.environ.get("HF_TOKEN", "")

    if use_diarization and not hf_token:
        print("Error: HF_TOKEN is required for speaker diarization (SPEAKERS > 1)")
        print("Get a token at https://huggingface.co/settings/tokens")
        print("You must also accept the model licenses at:")
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  https://huggingface.co/pyannote/segmentation-3.0")
        print("  https://huggingface.co/pyannote/speaker-diarization-community-1")
        sys.exit(1)

    # Discover audio files
    audio_files = sorted(
        p
        for p in INPUT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"No audio files found in {INPUT_DIR}")
        print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    # Load vocabulary hints
    vocab_prompt = load_vocab(INPUT_DIR)

    print(f"Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"  - {f.name}")
    if use_diarization:
        print(f"Diarization: enabled (speakers: {speakers_str})")
    else:
        print("Diarization: disabled (1 speaker per file)")
    print()

    # Verify models are downloaded (first run may download ~3GB + ~600MB)
    print("Checking models...")
    t0 = time.time()
    whisper_model = WhisperModel(
        "large-v3",
        device="cuda",
        compute_type="float16",
        download_root=MODEL_CACHE,
    )
    if use_diarization:
        from pyannote.audio import Pipeline

        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
        # Unload both — they'll be loaded one at a time per file to save memory
        del diarization_pipeline
    del whisper_model
    torch.cuda.empty_cache()
    gc.collect()
    print(f"Models ready in {time.time() - t0:.1f}s")

    print()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transcribe each file
    results = []
    all_segments = []
    failures = []

    for audio_path in audio_files:
        speaker_label = audio_path.stem
        print(f"Transcribing: {audio_path.name} (label: {speaker_label})")
        t0 = time.time()

        try:
            # Convert to 16kHz mono WAV upfront — used by both whisper and
            # diarization. Avoids redundant decoding of compressed formats,
            # and eliminates pyannote's crop() I/O bottleneck.
            print("  Converting audio to WAV...")
            wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_path = Path(wav_tmp.name)
            wav_tmp.close()
            subprocess.run(
                ["ffmpeg", "-i", str(audio_path),
                 "-ar", "16000", "-ac", "1", "-y", str(wav_path)],
                check=True, capture_output=True,
            )

            # Step 1: Whisper transcription
            print("  Loading whisper model...")
            whisper_model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float16",
                download_root=MODEL_CACHE,
            )
            print("  Transcribing speech...")
            whisper_result = transcribe_file(whisper_model, wav_path, initial_prompt=vocab_prompt)
            # Preserve original filename in output
            whisper_result["audio_file"] = audio_path.name

            # Free whisper before diarization
            if use_diarization:
                del whisper_model
                torch.cuda.empty_cache()
                gc.collect()

            # Step 2: Diarization (if enabled)
            if use_diarization:
                print("  Loading diarization model...")

                # Warm up CUDA context
                torch.zeros(1).cuda()

                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    token=hf_token,
                )
                diarization_pipeline.to(torch.device("cuda"))

                # Increase batch sizes (defaults are 1, starving the GPU)
                diarization_pipeline.segmentation_batch_size = 32
                diarization_pipeline.embedding_batch_size = 32

                # Load WAV into memory so pyannote's crop() never hits disk
                print("  Loading audio into memory...")
                import torchaudio
                waveform, sample_rate = torchaudio.load(str(wav_path))
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}

                # WAV file no longer needed
                wav_path.unlink(missing_ok=True)

                print("  Running diarization...")
                from pyannote.audio.pipelines.utils.hook import ProgressHook
                with ProgressHook() as hook:
                    diarization = run_diarization(
                        diarization_pipeline, audio_input, min_speakers, max_speakers,
                        hook=hook,
                    )
                # pyannote 4.x returns DiarizeOutput; extract the Annotation
                if hasattr(diarization, "speaker_diarization"):
                    diarization = diarization.speaker_diarization
                # Free diarization pipeline and waveform
                del diarization_pipeline, waveform, audio_input
                torch.cuda.empty_cache()
                gc.collect()

                segments = merge_with_diarization(whisper_result, diarization)
                unique_speakers = {s.get("speaker", "UNKNOWN") for s in segments}
                print(f"  Found {len(unique_speakers)} speaker(s): {', '.join(sorted(unique_speakers))}")
            else:
                wav_path.unlink(missing_ok=True)
                # No diarization — tag all segments with filename as speaker
                segments = whisper_result["segments"]
                for seg in segments:
                    seg["speaker"] = speaker_label

            elapsed = time.time() - t0
            ratio = whisper_result["duration"] / elapsed if elapsed > 0 else 0

            # Build final result
            result = {
                "speaker": speaker_label,
                "audio_file": whisper_result["audio_file"],
                "language": whisper_result["language"],
                "language_probability": whisper_result["language_probability"],
                "duration": whisper_result["duration"],
                "diarization": use_diarization,
                "segments": segments,
            }

            # Write JSON
            json_path = OUTPUT_DIR / f"{speaker_label}.json"
            write_json(result, json_path)

            # Write SRT (include speaker labels when diarization is active)
            srt_path = OUTPUT_DIR / f"{speaker_label}.srt"
            write_srt(segments, srt_path, include_speaker=use_diarization)

            seg_count = len(segments)
            print(
                f"  Done in {elapsed:.1f}s ({ratio:.1f}x realtime) "
                f"— {seg_count} segments, {whisper_result['duration']:.0f}s audio"
            )
            results.append((speaker_label, whisper_result["duration"], elapsed))
            all_segments.extend(segments)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}", file=sys.stderr)
            wav_path.unlink(missing_ok=True)
            failures.append((speaker_label, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        total_audio = sum(d for _, d, _ in results)
        total_time = sum(t for _, _, t in results)
        print(f"Succeeded: {len(results)}/{len(audio_files)}")
        print(f"Total audio: {total_audio / 60:.1f} minutes")
        print(f"Total time:  {total_time / 60:.1f} minutes")
        if total_time > 0:
            print(f"Overall speed: {total_audio / total_time:.1f}x realtime")

    # Write merged transcript when multiple files were processed
    if len(results) > 1 and all_segments:
        merged_segments = sorted(all_segments, key=lambda s: s["start"])
        max_duration = max(d for _, d, _ in results)

        merged_result = {
            "speakers": [r[0] for r in results],
            "duration": max_duration,
            "diarization": use_diarization,
            "segments": merged_segments,
        }

        write_json(merged_result, OUTPUT_DIR / "merged.json")
        write_srt(merged_segments, OUTPUT_DIR / "merged.srt", include_speaker=True)
        print(f"Merged transcript: {len(merged_segments)} segments → merged.json, merged.srt")

    if failures:
        print(f"\nFailed: {len(failures)}/{len(audio_files)}")
        for speaker, error in failures:
            print(f"  - {speaker}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
