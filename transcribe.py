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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypedDict

import torch
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}
VOCAB_FILENAME = "vocab.txt"

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """All tunable parameters for a transcription run."""

    model_name: str
    device: str
    compute_type: str
    sample_rate: int
    max_gap: float
    snap_distance: float
    seg_batch_size: int
    emb_batch_size: int
    min_speakers: int
    max_speakers: int
    use_diarization: bool
    hf_token: str
    vocab_prompt: str | None
    input_dir: Path
    output_dir: Path
    model_cache: str

    @classmethod
    def from_env(cls) -> "Config":
        """Parse all configuration from environment variables.

        Exits with a diagnostic message on invalid input.
        """
        input_dir = Path(os.environ.get("INPUT_DIR", "/input"))
        output_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
        model_cache = os.environ.get("MODEL_CACHE_DIR", "/models")

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

        vocab_prompt = load_vocab(input_dir)

        return cls(
            model_name="large-v3",
            device="cuda",
            compute_type="float16",
            sample_rate=16000,
            max_gap=1.5,
            snap_distance=0.5,
            seg_batch_size=32,
            emb_batch_size=32,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            use_diarization=use_diarization,
            hf_token=hf_token,
            vocab_prompt=vocab_prompt,
            input_dir=input_dir,
            output_dir=output_dir,
            model_cache=model_cache,
        )


class SpeakerTurn(NamedTuple):
    start: float
    end: float
    speaker: str


class Word(TypedDict):
    start: float
    end: float
    word: str
    probability: float


class Segment(TypedDict, total=False):
    start: float
    end: float
    text: str
    words: list[Word]
    speaker: str


@dataclass
class TranscriptionResult:
    """Raw output from whisper transcription."""

    audio_file: str
    language: str
    language_probability: float
    duration: float
    segments: list[Segment]


@dataclass
class FileResult:
    """What process_file returns for one audio file."""

    speaker_label: str
    audio_file: str
    duration: float
    elapsed: float
    segments: list[Segment]
    transcription: TranscriptionResult


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextmanager
def whisper_model(config: Config):
    """Load a WhisperModel, yield it, then free VRAM."""
    model = WhisperModel(
        config.model_name,
        device=config.device,
        compute_type=config.compute_type,
        download_root=config.model_cache,
    )
    try:
        yield model
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def diarization_pipeline(config: Config):
    """Load pyannote diarization pipeline on CUDA, yield it, then free VRAM."""
    from pyannote.audio import Pipeline

    torch.zeros(1).cuda()

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=config.hf_token,
    )
    pipeline.to(torch.device("cuda"))
    pipeline.segmentation_batch_size = config.seg_batch_size
    pipeline.embedding_batch_size = config.emb_batch_size
    try:
        yield pipeline
    finally:
        del pipeline
        torch.cuda.empty_cache()
        gc.collect()


@contextmanager
def temp_wav(audio_path: Path, sample_rate: int = 16000):
    """Convert audio to 16kHz mono WAV in a temp file, yield path, clean up."""
    wav_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav_path = Path(wav_tmp.name)
    wav_tmp.close()
    try:
        subprocess.run(
            ["ffmpeg", "-i", str(audio_path),
             "-ar", str(sample_rate), "-ac", "1", "-y", str(wav_path)],
            check=True, capture_output=True,
        )
        yield wav_path
    finally:
        wav_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_speakers(speakers_str: str) -> tuple[int, int]:
    """Parse SPEAKERS env var: '1', '3', or '2-6' -> (min, max)."""
    if "-" in speakers_str:
        parts = speakers_str.split("-", 1)
        return int(parts[0]), int(parts[1])
    n = int(speakers_str)
    return n, n


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


def get_speaker_for_word(word_start: float, word_end: float,
                         speaker_timeline: list[SpeakerTurn],
                         max_gap: float = 0.5) -> str:
    """Find the best speaker for a word using overlap duration.

    Computes how much each speaker's turns overlap with the word's time range.
    The speaker with the most overlap wins. If no speaker overlaps at all,
    snaps to the nearest turn boundary within max_gap seconds.
    """
    overlaps: dict[str, float] = {}
    for turn in speaker_timeline:
        if turn.end <= word_start or turn.start >= word_end:
            continue
        overlap = min(turn.end, word_end) - max(turn.start, word_start)
        overlaps[turn.speaker] = overlaps.get(turn.speaker, 0.0) + overlap

    if overlaps:
        return max(overlaps, key=overlaps.get)

    best_speaker = "UNKNOWN"
    best_dist = max_gap
    for turn in speaker_timeline:
        dist = min(abs(turn.start - word_end), abs(turn.end - word_start))
        if dist < best_dist:
            best_dist = dist
            best_speaker = turn.speaker

    return best_speaker


def group_words_into_segments(words: list[Word],
                              should_split: callable) -> list[Segment]:
    """Accumulate words into segments, splitting when should_split returns True.

    should_split(prev_word, curr_word) -> bool determines where to cut.
    """
    if not words:
        return []

    groups: list[list[Word]] = []
    current: list[Word] = [words[0]]

    for word in words[1:]:
        if should_split(current[-1], word):
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    groups.append(current)

    return [
        {
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "text": "".join(w["word"] for w in group).strip(),
            "words": group,
        }
        for group in groups
    ]


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def transcribe_file(model: WhisperModel, audio_path: Path,
                    initial_prompt: str | None = None) -> TranscriptionResult:
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
    segments: list[Segment] = []

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

    return TranscriptionResult(
        audio_file=audio_path.name,
        language=info.language,
        language_probability=round(info.language_probability, 4),
        duration=round(info.duration, 3),
        segments=segments,
    )


def run_diarization(pipeline, audio_input, min_speakers: int,
                    max_speakers: int, hook=None):
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

    result = pipeline(audio_input, **kwargs)
    # pyannote 4.x returns DiarizeOutput; extract the Annotation
    if hasattr(result, "speaker_diarization"):
        result = result.speaker_diarization
    return result


def merge_with_diarization(transcription: TranscriptionResult,
                           diarization) -> list[Segment]:
    """Merge whisper segments with diarization speaker labels.

    Assigns a speaker to each word using overlap duration with diarization
    turns, then re-segments: consecutive words from the same speaker
    are grouped together. This correctly splits whisper segments
    that span speaker turns.
    """
    speaker_timeline = [
        SpeakerTurn(turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

    if not speaker_timeline:
        return transcription.segments

    result_segments: list[Segment] = []

    for seg in transcription.segments:
        if not seg.get("words"):
            seg["speaker"] = get_speaker_for_word(
                seg["start"], seg["end"], speaker_timeline,
            )
            result_segments.append(seg)
            continue

        for word in seg["words"]:
            word["speaker"] = get_speaker_for_word(
                word["start"], word["end"], speaker_timeline,
            )

        def split_on_speaker(prev, curr):
            return curr["speaker"] != prev["speaker"]

        groups = group_words_into_segments(seg["words"], split_on_speaker)
        for group in groups:
            group["speaker"] = group["words"][0]["speaker"]
        result_segments.extend(groups)

    return result_segments


def resegment_by_time_gap(segments: list[Segment],
                          max_gap: float = 1.5) -> list[Segment]:
    """Split segments when word-level timestamps show large gaps.

    In multitrack recordings each speaker's file spans the full session, with
    silence wherever they aren't speaking. Whisper's VAD skips the silence but
    its segmenter groups the surviving word islands into massive segments that
    can span minutes. This function walks each segment's word list and cuts a
    new segment whenever consecutive words are separated by more than max_gap
    seconds.

    Segments without word-level timestamps pass through unchanged.
    """
    result: list[Segment] = []
    for seg in segments:
        words = seg.get("words")
        if not words:
            result.append(seg)
            continue

        def split_on_gap(prev, curr):
            return curr["start"] - prev["end"] > max_gap

        result.extend(group_words_into_segments(words, split_on_gap))

    return result


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_json(data: dict, output_path: Path) -> None:
    """Write data as JSON."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_srt(segments: list[Segment], output_path: Path,
              include_speaker: bool = False) -> None:
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


def write_outputs(speaker_label: str, segments: list[Segment],
                  transcription: TranscriptionResult,
                  use_diarization: bool, config: Config) -> None:
    """Write JSON + SRT for one speaker/file."""
    result = {
        "speaker": speaker_label,
        "audio_file": transcription.audio_file,
        "language": transcription.language,
        "language_probability": transcription.language_probability,
        "duration": transcription.duration,
        "diarization": use_diarization,
        "segments": segments,
    }

    write_json(result, config.output_dir / f"{speaker_label}.json")
    write_srt(segments, config.output_dir / f"{speaker_label}.srt",
              include_speaker=use_diarization)


def write_merged_output(results: list[FileResult], config: Config) -> None:
    """Write merged JSON + SRT across all processed files."""
    all_segments = []
    for r in results:
        all_segments.extend(r.segments)

    merged_segments = sorted(all_segments, key=lambda s: s["start"])
    max_duration = max(r.duration for r in results)

    merged_result = {
        "speakers": [r.speaker_label for r in results],
        "duration": max_duration,
        "diarization": config.use_diarization,
        "segments": merged_segments,
    }

    write_json(merged_result, config.output_dir / "merged.json")
    write_srt(merged_segments, config.output_dir / "merged.srt", include_speaker=True)
    print(f"Merged transcript: {len(merged_segments)} segments -> merged.json, merged.srt")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def discover_audio_files(config: Config) -> list[Path]:
    """Find and report audio files in the input directory."""
    audio_files = sorted(
        p
        for p in config.input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        print(f"No audio files found in {config.input_dir}")
        print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"  - {f.name}")
    if config.use_diarization:
        speaker_range = (f"{config.min_speakers}-{config.max_speakers}"
                         if config.min_speakers != config.max_speakers
                         else str(config.max_speakers))
        print(f"Diarization: enabled (speakers: {speaker_range})")
    else:
        print("Diarization: disabled (1 speaker per file)")
    print()

    return audio_files


def process_file(audio_path: Path, config: Config) -> FileResult:
    """Run the full transcription pipeline for one audio file."""
    speaker_label = audio_path.stem
    print(f"Transcribing: {audio_path.name} (label: {speaker_label})")
    t0 = time.time()

    print("  Converting audio to WAV...")
    with temp_wav(audio_path, config.sample_rate) as wav_path:
        # Step 1: Whisper transcription
        print("  Loading whisper model...")
        with whisper_model(config) as model:
            print("  Transcribing speech...")
            transcription = transcribe_file(model, wav_path,
                                            initial_prompt=config.vocab_prompt)
        # Preserve original filename in output
        transcription.audio_file = audio_path.name

        # Step 2: Diarization or time-gap resegmentation
        if config.use_diarization:
            import torchaudio
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            print("  Loading diarization model...")
            with diarization_pipeline(config) as pipeline:
                print("  Loading audio into memory...")
                waveform, sample_rate = torchaudio.load(str(wav_path))
                audio_input = {"waveform": waveform, "sample_rate": sample_rate}

                print("  Running diarization...")
                with ProgressHook() as hook:
                    diarization = run_diarization(
                        pipeline, audio_input,
                        config.min_speakers, config.max_speakers,
                        hook=hook,
                    )
                del waveform, audio_input

            segments = merge_with_diarization(transcription, diarization)
            unique_speakers = {s.get("speaker", "UNKNOWN") for s in segments}
            print(f"  Found {len(unique_speakers)} speaker(s): "
                  f"{', '.join(sorted(unique_speakers))}")
        else:
            segments = resegment_by_time_gap(transcription.segments,
                                             max_gap=config.max_gap)
            for seg in segments:
                seg["speaker"] = speaker_label

    elapsed = time.time() - t0
    ratio = transcription.duration / elapsed if elapsed > 0 else 0

    write_outputs(speaker_label, segments, transcription,
                  config.use_diarization, config)

    print(
        f"  Done in {elapsed:.1f}s ({ratio:.1f}x realtime) "
        f"-- {len(segments)} segments, {transcription.duration:.0f}s audio"
    )

    return FileResult(
        speaker_label=speaker_label,
        audio_file=transcription.audio_file,
        duration=transcription.duration,
        elapsed=elapsed,
        segments=segments,
        transcription=transcription,
    )


def print_summary(results: list[FileResult], failures: list[tuple[str, str]],
                  total_files: int) -> None:
    """Print end-of-run summary."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        total_audio = sum(r.duration for r in results)
        total_time = sum(r.elapsed for r in results)
        print(f"Succeeded: {len(results)}/{total_files}")
        print(f"Total audio: {total_audio / 60:.1f} minutes")
        print(f"Total time:  {total_time / 60:.1f} minutes")
        if total_time > 0:
            print(f"Overall speed: {total_audio / total_time:.1f}x realtime")

    if failures:
        print(f"\nFailed: {len(failures)}/{total_files}")
        for speaker, error in failures:
            print(f"  - {speaker}: {error}")


def main():
    config = Config.from_env()
    audio_files = discover_audio_files(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[FileResult] = []
    failures: list[tuple[str, str]] = []

    for audio_path in audio_files:
        try:
            result = process_file(audio_path, config)
            results.append(result)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failures.append((audio_path.stem, str(e)))

    if len(results) > 1:
        write_merged_output(results, config)

    print_summary(results, failures, len(audio_files))
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
