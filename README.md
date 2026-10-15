# Minute Oats

Minute Oats records a meeting on your Mac and turns it into a transcript, without anything leaving the Mac. The command is `minutes`. It captures your microphone and the call audio from any app (Zoom, Meet, Teams, a browser), so no bot joins the call. Your lines are labelled **Me** and everyone else's **Them**. If you want a summary, it uses a language model you choose, run on your own machine.

```
$ minutes "Weekly sync"
Jot your own notes in: /Users/you/Documents/Minutes/2026-10-15 1400 Weekly sync/my-notes.md
Recording to /Users/you/Documents/Minutes/2026-10-15 1400 Weekly sync
  you:  MacBook Air Microphone
  call: system audio
Press Ctrl+C to stop.
● REC 00:12:41   you ███████···   call █████·····
Stopped. 761 s recorded.
Transcribing
  removing call echo from your mic
  you: 13 min of audio, done
  call: 13 min of audio, done
Transcript: /Users/you/Documents/Minutes/2026-10-15 1400 Weekly sync/transcript.md
Folder:     /Users/you/Documents/Minutes/2026-10-15 1400 Weekly sync
```

`transcript.md` then reads:

```
**Them** `00:00:00`
Thanks for joining. Let's review the quarterly numbers first. Revenue grew 12%, but churn in small business went up.

**Me** `00:00:09`
Sure, I think the churn is mostly the pricing change in July. I'll send the cohort analysis tomorrow.
```

## How it works

| Step | What happens |
|---|---|
| Record | A small Swift tool (`recorder/`) uses ScreenCaptureKit to write two time-aligned 16 kHz tracks: `mic.wav` (you) and `system.wav` (the call). |
| Clean | On laptop speakers your mic also hears the call. WebRTC's echo canceller, from the `livekit` package, removes it from the mic track, using the call track as its reference. If most of a mic line still repeats what the call said at that moment, the repeated words are cut. `mic.wav` itself is left untouched. |
| Transcribe | [Parakeet TDT 0.6B v3](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v3) on MLX, in 15-second windows. It handles English, Italian and 23 other European languages, and works out which one is spoken. Each track is transcribed on its own, which is how lines get their Me and Them labels. |
| Summarise | Only when you ask. The transcript, plus anything you typed in `my-notes.md` during the call, becomes a short summary with key points, decisions, action items and open questions. |

Models are downloaded once from Hugging Face. After that the tool runs offline.

## Install

You need an Apple Silicon Mac with macOS 15 or later, the Xcode Command Line Tools 16 or later (`xcode-select --install`, which brings Swift 6) and [uv](https://docs.astral.sh/uv/).

```
git clone https://github.com/aamaro12/minute-oats
cd minute-oats
uv sync --extra mlx        # leave out --extra mlx if you will use your own model server for summaries
mkdir -p ~/.local/bin && ln -s "$PWD/.venv/bin/minutes" ~/.local/bin/minutes
minutes setup              # builds the recorder and downloads the models (2.3 GB, plus 2.9 GB with --extra mlx)
minutes doctor
```

`~/.local/bin` needs to be on your `PATH`. Keep the clone where it is: `minutes` builds and runs the recorder from it, and after a `git pull` it rebuilds the recorder by itself.

macOS gives the permissions to the terminal app you run `minutes` from, and `minutes doctor` asks for them:

- **Microphone**: allow when prompted.
- **Screen & System Audio Recording**: needed for every recording, because ScreenCaptureKit records the microphone as well as the call. Switch your terminal on in System Settings > Privacy & Security, then quit and reopen the terminal if `doctor` still says "not granted".

## Try it

Importing a file needs no permissions. Make a short clip with the Mac's own voice and transcribe it:

```
say -o hello.aiff "Morning everyone. The launch moves to Thursday, and I will send the test results by five."
minutes import hello.aiff
minutes open
```

## Use

```
minutes                             # start recording; Ctrl+C to stop; the transcript follows
minutes "Call with Sam"             # same, with a title
minutes "Call with Sam" --summary   # also write a summary
minutes --edit                      # open my-notes.md to jot notes during the call
minutes --no-system                 # in-person meeting, microphone only
minutes record "Summary of Q3"      # a title that starts with a command word needs `record`
minutes import recording.m4a        # transcribe a file (m4a, mp3, wav, aiff)
minutes list                        # all sessions
minutes open                        # latest session in Finder
minutes summary                     # add or redo the summary of the latest session
minutes transcribe "Sam"            # redo the transcript of a session matching "Sam"
```

Each session is a folder in `~/Documents/Minutes`:

| File | What |
|---|---|
| `mic.wav`, `system.wav` | the audio, about 115 MB per track per hour (`audio.wav` for an imported file) |
| `recording.json` | title, start time, length, tracks |
| `transcript.md`, `transcript.json` | the timestamped transcript |
| `my-notes.md` | your own notes |
| `summary.md` | the summary, if you asked for one |

If the recorder is killed or crashes, the audio up to that point is kept and `minutes transcribe` still works on it.

## Choose the summary model

Summaries are optional, and the model is your choice.

**Built in, with MLX.** Install with `--extra mlx`. The default is Qwen3.5 4B (2.9 GB). Any MLX chat model on Hugging Face works:

```
export MINUTES_LLM_MODEL=mlx-community/Qwen3.5-9B-4bit
```

**Any OpenAI-compatible server**, such as Ollama, LM Studio, llama.cpp or `mlx_lm.server`. Set the URL and the model name. For example:

```
# Ollama
ollama pull gemma3:12b
export MINUTES_LLM_URL=http://localhost:11434/v1
export MINUTES_LLM_MODEL=gemma3:12b

# LM Studio
export MINUTES_LLM_URL=http://localhost:1234/v1
export MINUTES_LLM_MODEL=<the model you loaded>
```

A transcript longer than `MINUTES_LLM_CONTEXT` tokens (default 24,000, about an hour and a half of talk) is summarised in parts and then combined, so the model's context window must be larger than that. Ollama's default window is smaller: start it with `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`, or lower `MINUTES_LLM_CONTEXT`. Models that think before answering work too; their thinking is left out of the summary. For a server that needs a key, set `MINUTES_LLM_API_KEY`. A hosted API would send your transcript off the Mac.

## Names it mishears

Add corrections to `vocabulary.txt` in the sessions folder (`~/Documents/Minutes`), one per line, then run `minutes transcribe` again. Lines starting with `#` are comments.

```
Acme Core => AcmeCorp
```

## Settings

| Variable | Default |
|---|---|
| `MINUTES_DIR` | `~/Documents/Minutes` |
| `MINUTES_ASR_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` |
| `MINUTES_LLM_URL` | not set: summaries run in-process with MLX |
| `MINUTES_LLM_MODEL` | `mlx-community/Qwen3.5-4B-4bit` with MLX |
| `MINUTES_LLM_CONTEXT` | `24000` |
| `MINUTES_LLM_API_KEY` | not set |

## Speed

Measured on a MacBook Air M2 with 16 GB, with nothing else using the GPU:

| Step | Time |
|---|---|
| Transcribe a 10-minute recording, start to finish | 24 s |
| Echo cancelling, per 10 minutes | 4.5 s |
| Summary of an hour-long transcript (14,500 tokens) with Qwen3.5 4B | 105 s, 6.1 GB peak memory |

Transcription runs on the GPU, so it slows down a lot while another app is using it.

## Limits

- Everyone on the far side of the call is "Them": it does not tell remote speakers apart.
- The transcript comes after you stop, not during the call.
- A microphone-only recording of a room is one unlabelled stream.
- Anything else your mic hears in the room, such as a phone or a TV, is labelled Me.
- On speakers, words you say while the call is loud are kept but can be misheard.
- macOS and Apple Silicon only, and so far tested on one Mac.

## Licence

MIT
