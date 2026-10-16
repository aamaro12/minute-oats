"""Speech to text with Parakeet (MLX), plus merging the mic and call tracks into one transcript."""

import json
import os
import re
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

ASR_MODEL = os.environ.get("MINUTES_ASR_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
SAMPLE_RATE = 16_000


@dataclass
class Segment:
    start: float
    end: float
    speaker: str
    text: str


def read_wav(path: Path) -> np.ndarray:
    """Our 16 kHz mono 16-bit WAVs as float32. Reads to the end of the file instead of trusting
    the header, which is only finalised when recording stops cleanly."""
    raw = Path(path).read_bytes()
    start = raw.index(b"data") + 8
    return np.frombuffer(raw, np.int16, count=(len(raw) - start) // 2, offset=start).astype(np.float32) / 32768.0


def write_wav(path: Path, audio: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def cancel_echo(mic: np.ndarray, call: np.ndarray) -> np.ndarray:
    """Remove the call audio that the mic picked up from the speakers, using WebRTC's echo
    canceller with the call track as its reference. The mic is delayed by 50 ms first so the
    echo always comes after its reference, even if the two tracks are slightly out of line."""
    from livekit import rtc

    apm = rtc.AudioProcessingModule(echo_cancellation=True, high_pass_filter=True)
    frame, pad = SAMPLE_RATE // 100, SAMPLE_RATE // 20
    to_int16 = lambda a: (np.clip(a, -1, 1) * 32767).astype(np.int16)
    m = to_int16(np.pad(mic, (pad, 0)))
    r = to_int16(np.pad(call[: len(m)], (0, max(0, len(m) - len(call)))))
    out = np.zeros_like(m)
    for i in range(0, len(m) - frame + 1, frame):
        apm.process_reverse_stream(rtc.AudioFrame(r[i:i + frame].tobytes(), SAMPLE_RATE, 1, frame))
        f = rtc.AudioFrame(m[i:i + frame].tobytes(), SAMPLE_RATE, 1, frame)
        apm.set_stream_delay_ms(0)
        apm.process_stream(f)
        out[i:i + frame] = np.frombuffer(f.data, np.int16)
    return out[pad:].astype(np.float32) / 32768.0


def has_sound(audio: np.ndarray) -> bool:
    return audio.size > SAMPLE_RATE / 2 and float(np.abs(audio).max()) > 0.003


def load_model(name: str = ASR_MODEL):
    import mlx.core as mx
    import parakeet_mlx.parakeet as pk
    from parakeet_mlx import from_pretrained

    # parakeet-mlx decodes files with ffmpeg; our tracks are already 16 kHz WAV, so read them directly.
    pk.load_audio = lambda path, sr, dtype=mx.bfloat16: mx.array(read_wav(path))
    return from_pretrained(name)


def transcribe_track(model, audio: np.ndarray, speaker: str, label: str) -> list[Segment]:
    from parakeet_mlx.alignment import SentenceConfig
    from parakeet_mlx.parakeet import DecodingConfig

    if not has_sound(audio):
        print(f"  {label}: silent, skipped", file=sys.stderr)
        return []
    minutes = audio.size / SAMPLE_RATE / 60

    def progress(done, total):
        print(f"\r  {label}: {minutes:.0f} min of audio, {100 * done / total:.0f}%", end="", file=sys.stderr)

    config = DecodingConfig(sentence=SentenceConfig(silence_gap=1.5, max_duration=30))
    # Short windows: over long ones Parakeet can skip speech when voices overlap or languages mix.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "track.wav"
        write_wav(path, audio)
        result = model.transcribe(path, decoding_config=config, chunk_duration=15, overlap_duration=3,
                                  chunk_callback=progress)
    print(f"\r\033[2K  {label}: {minutes:.0f} min of audio, done", file=sys.stderr)
    segments: list[Segment] = []
    for s in result.sentences:
        text = s.text.strip()
        if not re.search(r"\w", text):
            # A lone punctuation mark split off by a pause belongs to the previous sentence.
            if segments and text:
                segments[-1].text += text
            continue
        segments.append(Segment(round(s.start, 2), round(s.end, 2), speaker, text))
    return segments


def to_turns(segments: list[Segment], gap: float = 2.0) -> list[Segment]:
    """Join consecutive segments from the same speaker into one turn."""
    turns: list[Segment] = []
    for seg in sorted(segments, key=lambda s: s.start):
        last = turns[-1] if turns else None
        if last and last.speaker == seg.speaker and seg.start - last.end < gap:
            last.end = seg.end
            last.text = f"{last.text} {seg.text}"
        else:
            turns.append(Segment(seg.start, seg.end, seg.speaker, seg.text))
    return turns


def load_corrections(path: Path) -> list[tuple[re.Pattern, str]]:
    """Lines like `Acme Core => AcmeCorp` in vocabulary.txt fix names the speech model gets wrong.
    Lines starting with # are comments."""
    rules = []
    if path.exists():
        for line in path.read_text().splitlines():
            if "=>" in line and not line.lstrip().startswith("#"):
                wrong, right = (x.strip() for x in line.split("=>", 1))
                if wrong:
                    rules.append((re.compile(rf"\b{re.escape(wrong)}\b", re.I), right))
    return rules


def apply_corrections(segments: list[Segment], rules) -> None:
    for seg in segments:
        for pattern, right in rules:
            seg.text = pattern.sub(lambda _, r=right: r, seg.text)


def clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def transcript_text(turns: list[Segment]) -> str:
    """Compact form used as input to the notes model."""
    return "\n".join(f"[{clock(t.start)}] {t.speaker + ': ' if t.speaker else ''}{t.text}" for t in turns)


def transcribe_session(session: Path) -> list[Segment]:
    meta = json.loads((session / "recording.json").read_text())
    tracks = meta["tracks"]
    both = "mic" in tracks and "system" in tracks
    model = load_model()
    print("Transcribing", file=sys.stderr)
    audio = {name: read_wav(session / file) for name, file in tracks.items()}
    if both:
        print("  removing call echo from your mic", file=sys.stderr)
        audio["mic"] = cancel_echo(audio["mic"], audio["system"])
    mine = transcribe_track(model, audio["mic"], "Me" if both else "", "you") if "mic" in audio else []
    theirs = transcribe_track(model, audio["system"], "Them", "call") if "system" in audio else []
    imported = transcribe_track(model, audio["audio"], "", "audio") if "audio" in audio else []
    turns = to_turns(mine + theirs + imported)
    apply_corrections(turns, load_corrections(session.parent / "vocabulary.txt"))
    (session / "transcript.json").write_text(json.dumps([asdict(t) for t in turns], indent=1, ensure_ascii=False))
    write_transcript_md(session, meta, turns)
    return turns


def write_transcript_md(session: Path, meta: dict, turns: list[Segment]) -> None:
    started = datetime.fromisoformat(meta["started_at"]).astimezone()
    seconds = meta.get("duration_seconds") or (turns[-1].end if turns else 0)  # 0 if the recorder crashed
    length = f"{round(seconds / 60)} min" if seconds >= 60 else f"{seconds:.0f} s"
    lines = ["# Transcript", "", f"{started:%-d %b %Y, %H:%M}, {length}", ""]
    for t in turns:
        who = f"**{t.speaker}** " if t.speaker else ""
        lines += [f"{who}`{clock(t.start)}`  ", t.text, ""]
    if not turns:
        lines.append("_No speech detected._")
    (session / "transcript.md").write_text("\n".join(lines))
