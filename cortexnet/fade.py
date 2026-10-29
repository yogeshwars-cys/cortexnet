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

Order inside each pass: oldest first (order="age", the default), or order="charge": every result keeps a leaky
relevance charge, +1 for each later tool call that touches its file (by path or file name in the call's arguments),
decaying by `leak` per later call; the least-charged result shrinks first, ties oldest first. A file the agent keeps
coming back to stays; one it touched once long ago goes, even if it was read more recently than the first.

Messages are OpenAI Chat style: assistant messages carry tool_calls (id, function.name, function.arguments JSON),
tool messages carry tool_call_id and content.
"""
import json
import os

from .outline import tokens

FADE_TOOLS = ("read_file", "read_lines")
PIN_TOOLS = ("note",)          # the task card (cortexnet/card.py): the newest copy is never faded
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


def charges(res, leak=0.8):
    """Leaky relevance charge per tool result: sum over later calls touching its file of leak ** (calls since)."""
    n = len(res)
    keys = [json.dumps(a, sort_keys=True) for _, _, a in res]
    out = []
    for j, (_, _, args) in enumerate(res):
        path = args.get("path")
        if not path:
            out.append(0.0)
            continue
        base = os.path.basename(path)
        q = 0.0
        for i in range(j + 1, n):
            if res[i][2].get("path") == path or (base and base in keys[i]):
                q += leak ** (n - 1 - i)
        out.append(q)
    return out


def _plan(msgs, est_tokens, limit, outline_of, low, keep, keep_frac, scale, pinned, order="age", leak=0.8):
    """The batch a fade from est_tokens down to low * limit would apply, in order:
    [(kind, tool message, name, path, new content, tokens saved)]. Changes nothing."""
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
    last_pin = max((j for j, (_, name, _) in enumerate(res) if name in PIN_TOOLS), default=-1)
    for j, (m, name, args) in enumerate(res):
        path, content = args.get("path"), m.get("content") or ""
        if name in PIN_TOOLS:                                  # working memory: only older copies may go
            if j < last_pin and not content.startswith(MARK):
                cands.append((0, j))
            continue
        if name not in FADE_TOOLS or not path or (content.startswith(MARK) and "outline is below" not in content):
            continue
        if name == "read_file" and later_read.get(path, j) > j:
            cands.append((0, j))                               # a later copy exists: eligible even if recent or pinned
        elif j not in last_k and path not in recent and path not in pinned:
            cands.append((2 if content.startswith(MARK) else 1, j))
    est, plan = est_tokens, []
    if order == "charge":
        q = charges(res, leak)
        cands = sorted(cands, key=lambda c: (c[0], q[c[1]], c[1]))
    for kind, j in sorted(cands) if order != "charge" else cands:
        if est <= low * limit:
            break
        m, name, args = res[j]
        path, content = args.get("path"), m.get("content") or ""
        new = f"{MARK}{path}; read it again if you need it]"
        if name in PIN_TOOLS:
            new = f"{MARK}an older task card; the newest card is below]"
        elif kind == 0:
            new = f"{MARK}{path}; superseded by a later read of the same file]"
        elif kind == 1:
            head = outline_of(path)
            if head is not None and 2 * len(head) < len(content):
                new = f"{MARK}{path}; its outline is below, use read_lines for any body]\n{head}"
        if len(new) < len(content):
            saved = scale * (tokens(content) - tokens(new))
            est -= saved
            plan.append((kind, m, name, path, new, saved))
    return plan


def watermark_fade(msgs, est_tokens, limit, outline_of, mark=0.85, low=0.5, keep=4, keep_frac=0.25, scale=1.0,
                   pinned=(), order="age", leak=0.8, on_fade=None):
    """Shrink in place if est_tokens > mark * limit, oldest first, until the estimate is below low * limit.
    outline_of(path) -> outline text, or None to leave that read; scale: server tokens per tokens() unit, so the
    running estimate stays in the server's units. Returns the number of messages changed.
    Protected: the newest result always, and the newest `keep` results while their total stays within
    keep_frac * limit (a count alone let one 3.4k-token outline block every batch in the 8k laptop session).
    pinned: paths whose reads stay as they are (only a copy superseded by a later read of the same path may go).
    on_fade(path): called for every read shrunk (e.g. Reader.forget, so its bodies may be sent again)."""
    if est_tokens <= mark * limit:
        return 0
    plan = _plan(msgs, est_tokens, limit, outline_of, low, keep, keep_frac, scale, set(pinned), order, leak)
    for _, m, name, path, new, _ in plan:
        m["content"] = new
        if on_fade is not None and path and name in FADE_TOOLS:
            on_fade(path)
    return len(plan)


def preview(msgs, est_tokens, limit, outline_of, mark=0.85, low=0.5, keep=4, keep_frac=0.25, scale=1.0, pinned=(),
            order="age", leak=0.8):
    """What the next fade would do, without changing anything. A fade starts at mark * limit, so this plans from
    there (or from est_tokens if already above it). Returns [dict(path, what, saved)] in the order it would apply."""
    plan = _plan(msgs, max(est_tokens, mark * limit), limit, outline_of, low, keep, keep_frac, scale, set(pinned),
                 order, leak)
    out = []
    for kind, _, name, path, new, saved in plan:
        what = ("older task card" if name in PIN_TOOLS else "superseded" if kind == 0
                else "to outline" if "outline is below" in new else "to pointer")
        out.append(dict(path=path or "(card)", tool=name, what=what, saved=int(saved)))
    return out
