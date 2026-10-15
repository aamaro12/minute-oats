"""Meeting notes from a transcript plus your own notes, with a language model you choose.

- MINUTES_LLM_URL set: any OpenAI-compatible server (Ollama, LM Studio, llama.cpp, mlx_lm.server),
  with MINUTES_LLM_MODEL naming the model on that server.
- Otherwise: MLX in this process (install the `mlx` extra), with MINUTES_LLM_MODEL a Hugging Face
  MLX model. Default: Qwen3.5 4B.
"""

import json
import os
import re
import sys
import urllib.error
import urllib.request

LLM_URL = os.environ.get("MINUTES_LLM_URL", "").rstrip("/")
LLM_MODEL = os.environ.get("MINUTES_LLM_MODEL", "" if LLM_URL else "mlx-community/Qwen3.5-4B-4bit")
# Transcripts longer than this many tokens are summarised in parts first, then combined.
# Keep it below the model's context window. 24,000 fits about an hour and a half of talk.
CONTEXT = int(os.environ.get("MINUTES_LLM_CONTEXT", 24_000))

SYSTEM = """You write meeting notes from a transcript of a call recorded on the user's Mac.
"Me" is the user; "Them" is everyone else on the call (the transcript cannot tell them apart, so use names only when someone is named).

Write markdown in exactly this shape:
# <short title, 3 to 8 words>
<one or two sentence summary>

## Key points
- ...

## Decisions
- ...

## Action items
- [ ] <owner>: <task> (<deadline if one was stated>)

## Open questions
- ...

Rules:
- Use only what is said in the transcript. Never invent numbers, names, owners or deadlines.
- An action item needs someone to say they will do something. Do not turn a mention of something into a task.
- If the user wrote their own notes, keep every point they wrote and expand it with detail from the transcript.
- Leave out a section entirely if there is nothing for it; never write "None".
- Be concise and specific. Write in the main language of the meeting, with British spelling for English."""

PART_SYSTEM = """You summarise one part of a long meeting transcript. List the key points, decisions,
action items (with owner and deadline when stated) and open questions in this part as short bullets.
Use only what is said. "Me" is the user; "Them" is everyone else."""


def tokens(text: str) -> int:
    return len(text) // 3  # errs high, which is the safe side for deciding when to split


class MLXModel:
    def __init__(self, name: str):
        try:
            from mlx_lm import load
        except ImportError:
            sys.exit("Summaries need a model: install the MLX extra (uv sync --extra mlx), or set MINUTES_LLM_URL "
                     "and MINUTES_LLM_MODEL to use a local server such as Ollama. See the README.")
        print(f"Loading {name}", file=sys.stderr)
        self.model, self.tok = load(name)

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                              enable_thinking=False)
        return generate(self.model, self.tok, prompt, max_tokens=max_tokens, sampler=make_sampler(temp=0.3))


class ServerModel:
    def __init__(self, url: str, name: str):
        if not name:
            sys.exit("Set MINUTES_LLM_MODEL to the model name on your server, e.g. gemma3:12b for Ollama.")
        self.url, self.name = url, name
        self.key = os.environ.get("MINUTES_LLM_API_KEY", "")

    def chat(self, messages: list[dict], max_tokens: int) -> str:
        body = {"model": self.name, "messages": messages, "max_tokens": max_tokens, "temperature": 0.3}
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        req = urllib.request.Request(f"{self.url}/chat/completions", json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                return json.load(resp)["choices"][0]["message"]["content"] or ""
        except urllib.error.URLError as e:
            sys.exit(f"Could not get a summary from {self.url}: {getattr(e, 'reason', e)}")


class Notes:
    def __init__(self):
        self.llm = ServerModel(LLM_URL, LLM_MODEL) if LLM_URL else MLXModel(LLM_MODEL)

    def _ask(self, system: str, user: str, max_tokens: int = 4000) -> str:
        # The limit leaves room for models that think before answering; the thinking is dropped.
        out = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens)
        return re.sub(r"<think>.*?(</think>|$)", "", out, flags=re.S).strip()

    def write(self, transcript: str, my_notes: str, title: str | None) -> str:
        if tokens(transcript) > CONTEXT:
            parts = split_lines(transcript, CONTEXT * 3 // 8)
            summaries = []
            for i, part in enumerate(parts, 1):
                print(f"  summarising part {i} of {len(parts)}", file=sys.stderr)
                summaries.append(f"### Part {i}\n" + self._ask(PART_SYSTEM, part))
            transcript = "Summaries of each part of the meeting, in order:\n\n" + "\n\n".join(summaries)
        print("Writing summary", file=sys.stderr)
        user = ""
        if title:
            user += f"Meeting title: {title}\n\n"
        if my_notes.strip():
            user += f"My notes from the meeting:\n{my_notes.strip()}\n\n"
        user += f"Transcript:\n{transcript}"
        return drop_empty_sections(self._ask(SYSTEM, user))


# "- None", "- [ ] N/A", "- Nothing stated." but not "- None of the vendors was chosen".
EMPTY_BULLET = re.compile(r"^\s*[-*]\s*(\[ \]\s*)?(none|n/?a|nothing)( \w+)?\W*$", re.I)


def drop_empty_sections(md: str) -> str:
    """Remove `## ` sections whose only content is a 'None' bullet; small models write them anyway."""
    out, section = [], []

    def flush():
        body = [l for l in section[1:] if l.strip()]
        if section and (not section[0].startswith("## ") or not all(EMPTY_BULLET.match(l) for l in body)):
            out.extend(section)

    for line in md.splitlines():
        if line.startswith("## "):
            flush()
            section = [line]
        else:
            section.append(line)
    flush()
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def split_lines(text: str, limit: int) -> list[str]:
    parts, current, size = [], [], 0
    for line in text.splitlines():
        n = tokens(line) + 1
        if current and size + n > limit:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        parts.append("\n".join(current))
    return parts
