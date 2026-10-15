"""minutes: record a meeting and transcribe it, all on this Mac.

  minutes                         start recording (mic + call audio); Ctrl+C to stop, then the transcript
  minutes "title" [options]       same with a title; options: --summary --edit --no-system --no-mic
  minutes import <audio file>     transcribe an existing recording (m4a, mp3, wav, aiff)
  minutes transcribe [session]    redo the transcript of a session (--summary to add a summary)
  minutes summary [session]       write or redo the summary of a session
  minutes list                    list sessions
  minutes open [session]          open a session folder in Finder
  minutes doctor                  check permissions, models and paths
  minutes setup                   build the recorder and download the models (one time)

[session] is a folder path or part of its name; the latest session is used when omitted.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RECORDER = ROOT / "recorder/.build/release/minutes-rec"
HOME = Path(os.environ.get("MINUTES_DIR", Path.home() / "Documents/Minutes"))
COMMANDS = {"record", "import", "transcribe", "summary", "list", "open", "doctor", "setup"}


def hf_cached(repo: str) -> bool:
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    hub = Path(os.environ.get("HF_HUB_CACHE", hf_home / "hub"))
    return any(hub.glob(f"models--{repo.replace('/', '--')}/snapshots/*/config.json"))


def hf_models() -> list[str]:
    """Hugging Face models in use: speech, plus the summary model when it runs in-process (MLX)."""
    from importlib.util import find_spec

    from .notes import LLM_MODEL, LLM_URL
    from .transcribe import ASR_MODEL

    return [ASR_MODEL] + ([LLM_MODEL] if not LLM_URL and find_spec("mlx_lm") else [])


def configure_env(allow_download: bool = False) -> None:
    """Keep model libraries quiet and offline. Downloads only happen in `setup`, or the first
    time a model is missing."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not allow_download and all(hf_cached(m) for m in hf_models()):
        os.environ["HF_HUB_OFFLINE"] = "1"
    import warnings

    warnings.filterwarnings("ignore")


def safe_name(text: str) -> str:
    text = re.sub(r"[/:\\\n\r\t]+", " ", text)
    text = re.sub(r"[^\w\s,.&()'+-]", "", text)
    return re.sub(r"\s+", " ", text).strip()[:80].strip() or "Meeting"


def ensure_recorder(force: bool = False) -> None:
    """Build the Swift recorder if it is missing or older than its source."""
    source = ROOT / "recorder/Sources/minutes-rec/main.swift"
    if not source.exists():
        sys.exit("The recorder source is missing: run minutes from a clone of the repo (see the README).")
    if force or not RECORDER.exists() or RECORDER.stat().st_mtime < source.stat().st_mtime:
        if not shutil.which("swift"):
            sys.exit("Building the recorder needs Swift: install the Xcode Command Line Tools (xcode-select --install).")
        print("Building the recorder", file=sys.stderr)
        if subprocess.run(["swift", "build", "-c", "release"], cwd=ROOT / "recorder").returncode:
            sys.exit("The recorder did not build; see the Swift errors above.")


def sessions() -> list[Path]:
    if not HOME.exists():
        return []
    return sorted(d for d in HOME.iterdir() if d.is_dir() and (d / "recording.json").exists())


def find_session(query: str | None) -> Path:
    if query and (Path(query).expanduser() / "recording.json").exists():
        return Path(query).expanduser().resolve()
    found = [s for s in sessions() if not query or query.lower() in s.name.lower()]
    exact = [s for s in found if query and s.name[16:].lower() == query.lower()]
    found = exact or found
    if not found:
        sys.exit(f"No session matching {query!r} in {HOME}" if query else f"No sessions yet in {HOME}")
    return found[-1]


def read_meta(session: Path) -> dict:
    return json.loads((session / "recording.json").read_text())


def update_meta(session: Path, **changes) -> None:
    meta = read_meta(session) | changes
    (session / "recording.json").write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False))


def new_session(title: str) -> Path:
    base = HOME / f"{datetime.now():%Y-%m-%d %H%M} {safe_name(title)}"
    session, n = base, 1
    while session.exists():
        n += 1
        session = base.with_name(f"{base.name} {n}")
    session.mkdir(parents=True)
    (session / "my-notes.md").write_text(f"# {title or 'Meeting'}\n\n")
    return session


def my_notes(session: Path) -> str:
    f = session / "my-notes.md"
    if not f.exists():
        return ""
    lines = f.read_text().splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    return "\n".join(lines).strip()


def free_memory() -> None:
    import gc

    import mlx.core as mx

    gc.collect()
    mx.clear_cache()


def process(session: Path, summary: bool) -> Path:
    from .transcribe import transcribe_session

    turns = transcribe_session(session)
    free_memory()
    print(f"Transcript: {session / 'transcript.md'}")
    if summary:
        session = write_summary(session, turns)
    return session


def write_summary(session: Path, turns=None) -> Path:
    from .notes import Notes
    from .transcribe import Segment, transcript_text

    if turns is None:
        turns = [Segment(**t) for t in json.loads((session / "transcript.json").read_text())]
    if not turns:
        print("No speech in the transcript, so no summary.")
        return session
    title = read_meta(session).get("title") or None
    text = Notes().write(transcript_text(turns), my_notes(session), title)
    (session / "summary.md").write_text(text + "\n")
    if not title:
        heading = next((l[2:].strip() for l in text.splitlines() if l.startswith("# ")), "")
        renamed = session.with_name(f"{session.name[:15]} {safe_name(heading)}")
        if heading and not renamed.exists():
            session = session.rename(renamed)
            update_meta(session, title=heading)
    print(f"Summary:    {session / 'summary.md'}")
    return session


# ---- commands --------------------------------------------------------------


def cmd_record(a) -> None:
    if a.no_mic and a.no_system:
        sys.exit("--no-mic and --no-system together would record nothing.")
    ensure_recorder()
    title = " ".join(a.title).strip()
    session = new_session(title)
    print(f"Jot your own notes in: {session / 'my-notes.md'}")
    if a.edit:
        subprocess.run(["open", "-t", str(session / "my-notes.md")])
    cmd = [str(RECORDER), "record", str(session)]
    if a.no_mic:
        cmd.append("--no-mic")
    if a.no_system:
        cmd.append("--no-system")
    proc = subprocess.Popen(cmd)
    try:
        code = proc.wait()
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the recorder's save short
        proc.send_signal(signal.SIGINT)
        code = proc.wait()
        signal.signal(signal.SIGINT, signal.default_int_handler)
    if not (session / "recording.json").exists():  # the recorder never started, so there is no audio
        shutil.rmtree(session, ignore_errors=True)
        sys.exit(code or 1)
    update_meta(session, title=title)
    try:
        session = process(session, summary=a.summary)
    except KeyboardInterrupt:
        sys.exit(f"\nStopped before the transcript was done. The audio is saved; finish with:\n"
                 f"  minutes transcribe {session.name[16:]!r}")
    print(f"Folder:     {session}")


def cmd_import(a) -> None:
    ensure_recorder()
    src = Path(a.file).expanduser()
    if not src.is_file():
        sys.exit(f"No such file: {src}")
    session = new_session(a.title or src.stem)
    out = subprocess.run([str(RECORDER), "convert", str(src), str(session / "audio.wav")],
                         capture_output=True, text=True)
    if out.returncode:
        shutil.rmtree(session, ignore_errors=True)
        sys.exit(out.stderr.strip())
    started = datetime.fromtimestamp(src.stat().st_mtime).astimezone().isoformat()
    meta = {"title": a.title or src.stem, "started_at": started, "duration_seconds": float(out.stdout or 0),
            "tracks": {"audio": "audio.wav"}, "source": str(src)}
    (session / "recording.json").write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False))
    session = process(session, summary=a.summary)
    print(f"Folder:     {session}")


def cmd_transcribe(a) -> None:
    session = process(find_session(a.session), summary=a.summary)
    print(f"Folder:     {session}")


def cmd_summary(a) -> None:
    session = find_session(a.session)
    if not (session / "transcript.json").exists():
        sys.exit(f"No transcript yet. Run: minutes transcribe {session.name!r}")
    write_summary(session)


def cmd_list(a) -> None:
    rows = sessions()
    if not rows:
        print(f"No sessions yet in {HOME}")
    for s in rows:
        secs = read_meta(s).get("duration_seconds", 0)
        length = f"{secs / 60:.0f} min" if secs >= 60 else f"{secs:.0f} s"
        state = "summary" if (s / "summary.md").exists() else "transcript" if (s / "transcript.md").exists() else "audio only"
        print(f"{s.name:<60} {length:>7}  {state}")


def cmd_open(a) -> None:
    subprocess.run(["open", str(find_session(a.session))])


def cmd_doctor(a) -> None:
    from importlib.util import find_spec

    from .notes import LLM_MODEL, LLM_URL

    ensure_recorder()
    subprocess.run([str(RECORDER), "check"])
    for m in hf_models():
        print(f"model {m}: {'downloaded' if hf_cached(m) else 'missing (run: minutes setup)'}")
    if LLM_URL:
        print(f"summaries: {LLM_MODEL or '(set MINUTES_LLM_MODEL)'} at {LLM_URL}")
    else:
        print(f"summaries: {LLM_MODEL} with MLX" if find_spec("mlx_lm") else "summaries: not set up (see README)")
    print(f"sessions folder: {HOME}")


def cmd_setup(a) -> None:
    ensure_recorder(force=True)
    from huggingface_hub import snapshot_download

    for m in hf_models():
        print(f"Downloading {m} (one time)")
        snapshot_download(m)
    HOME.mkdir(parents=True, exist_ok=True)
    print("Ready. Run `minutes doctor` to check permissions.")


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    argv = sys.argv[1:]
    if not argv or (argv[0] not in COMMANDS and argv[0] not in ("-h", "--help")):
        argv = ["record", *argv]
    p = argparse.ArgumentParser(prog="minutes", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record a meeting")
    r.add_argument("title", nargs="*")
    r.add_argument("--no-mic", action="store_true", help="record call audio only")
    r.add_argument("--no-system", action="store_true", help="record the microphone only (in-person meeting)")
    r.add_argument("--edit", action="store_true", help="open my-notes.md in a text editor while recording")
    r.add_argument("--summary", action="store_true", help="also write a summary after the transcript")
    i = sub.add_parser("import", help="transcribe an existing audio file")
    i.add_argument("file")
    i.add_argument("--title")
    i.add_argument("--summary", action="store_true", help="also write a summary")
    t = sub.add_parser("transcribe", help="redo a session's transcript")
    t.add_argument("session", nargs="?")
    t.add_argument("--summary", action="store_true", help="also write a summary")
    n = sub.add_parser("summary", help="write or redo a session's summary")
    n.add_argument("session", nargs="?")
    sub.add_parser("list", help="list sessions")
    o = sub.add_parser("open", help="open a session folder")
    o.add_argument("session", nargs="?")
    sub.add_parser("doctor", help="check permissions and models")
    sub.add_parser("setup", help="build recorder and download models")
    a = p.parse_args(argv)
    configure_env(allow_download=a.cmd == "setup")
    {"record": cmd_record, "import": cmd_import, "transcribe": cmd_transcribe, "summary": cmd_summary,
     "list": cmd_list, "open": cmd_open, "doctor": cmd_doctor, "setup": cmd_setup}[a.cmd](a)


if __name__ == "__main__":
    main()
