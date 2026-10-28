"""Fade on a watermark: keep the conversation append-only, and only when it nears the context limit shrink stale
file reads, in one batch, down to a low-water mark.

Why batch: a server with a prefix cache (llama-server, vLLM, hosted APIs) reuses the computed prefix of the history.
Rewriting an old message invalidates everything after it, so fading one read per turn (the SWE A/B's fade_age arm)
pays a re-prefill on every fade. One batch above the high mark, down to the low mark, pays it once and leaves room
for many more append-only turns.

What shrinks, oldest first, never the reads of files touched in the last `keep` tool calls:
  1. superseded reads: a read_file of a path that is read_file'd again later (the later copy carries it) -> a one-line
     pointer. With outline-first reads this is most of the waste in a long session (the same file read per question).
  2. stale reads: read_file / read_lines of a file not touched recently -> the file's outline (if shorter).

Messages are OpenAI Chat style: assistant messages carry tool_calls (id, function.name, function.arguments JSON),
tool messages carry tool_call_id and content.
"""
import json

from .outline import tokens

FADE_TOOLS = ("read_file", "read_lines")
MARK = "[faded to save context: "


def tool_results(msgs):
    """[(tool message, name, args)] in order. Each result is paired with the assistant message just before it (by id
    within that message, else by position), because ids are not reliably unique across a conversation (some servers
    and small models reuse them)."""
    out, calls, k = [], [], 0
    for m in msgs:
        if m.get("role") == "assistant":
            calls, k = [], 0
            for c in m.get("tool_calls") or []:
                f = c.get("function") or {}
                try:
                    args = json.loads(f.get("arguments") or "{}")
                except ValueError:
                    args = {}
                calls.append((c.get("id"), f.get("name"), args if isinstance(args, dict) else {}))
        elif m.get("role") == "tool":
            hit = next(((n, a) for i, n, a in calls if i is not None and i == m.get("tool_call_id")), None)
            if hit is None:
                hit = calls[k][1:] if k < len(calls) else (m.get("name"), {})
            out.append((m, *hit))
            k += 1
    return out


def watermark_fade(msgs, est_tokens, limit, outline_of, mark=0.85, low=0.5, keep=4, keep_frac=0.25, scale=1.0):
    """Shrink in place if est_tokens > mark * limit, oldest first, until the estimate is below low * limit.
    outline_of(path) -> outline text, or None to leave that read; scale: server tokens per tokens() unit, so the
    running estimate stays in the server's units. Returns the number of messages changed.
    Protected: the newest result always, and the newest `keep` results while their total stays within
    keep_frac * limit (a count alone let one 3.4k-token outline block every batch in the 8k laptop session)."""
    if est_tokens <= mark * limit:
        return 0
    res = tool_results(msgs)
    last_k, used = set(), 0
    for j in range(len(res) - 1, max(-1, len(res) - 1 - keep), -1):
        used += scale * tokens(res[j][0].get("content") or "")
        if last_k and used > keep_frac * limit:
            break
        last_k.add(j)                                          # the newest results are never touched
    recent = {res[j][2].get("path") for j in last_k} - {None}
    later_read = {}
    for j, (_, name, args) in enumerate(res):
        if name == "read_file" and args.get("path"):
            later_read[args["path"]] = j                       # index of the last read_file of each path
    # (pass, index): 0 superseded copies, 1 stale reads (to the outline when it at least halves them, else a pointer),
    # 2 stale reads already faded to an outline (to a pointer), each pass oldest first
    cands = []
    for j, (m, name, args) in enumerate(res):
        path, content = args.get("path"), m.get("content") or ""
        if name not in FADE_TOOLS or not path or (content.startswith(MARK) and "outline is below" not in content):
            continue
        if name == "read_file" and later_read.get(path, j) > j:
            cands.append((0, j))                               # a later copy exists: eligible even if recent
        elif j not in last_k and path not in recent:
            cands.append((2 if content.startswith(MARK) else 1, j))
    est, n = est_tokens, 0
    for kind, j in sorted(cands):
        if est <= low * limit:
            break
        m, name, args = res[j]
        path, content = args["path"], m.get("content") or ""
        new = f"{MARK}{path}; read it again if you need it]"
        if kind == 0:
            new = f"{MARK}{path}; superseded by a later read of the same file]"
        elif kind == 1:
            head = outline_of(path)
            if head is not None and 2 * len(head) < len(content):
                new = f"{MARK}{path}; its outline is below, use read_lines for any body]\n{head}"
        if len(new) < len(content):
            est -= scale * (tokens(content) - tokens(new))
            m["content"] = new
            n += 1
    return n
