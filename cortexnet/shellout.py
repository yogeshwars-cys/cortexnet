"""Outline a long shell output (test runs, builds, logs, greps) instead of handing all of it to the agent.

Kept, in full and in order: the first `head` and last `tail` lines (the command's start, summary and exit status), every
line that signals a failure (Traceback, Error, FAIL, panic, ...) with `ctx` lines around it, and every line holding one of
the agent's rare intent words with 2 lines around it. Each omitted run becomes one line saying how many lines were left
out and how to see them. Returns None when the output is short or the outline would not save enough.
"""
import re

from .outline import tokens

SIGNAL = re.compile(r"Traceback|Exception\b|\bError\b|error[:\[]|\berror\b|ERROR|\bFAIL|FAILED|failed|fatal|panic|"
                    r"Assertion|assert |Warning|warning:|WARN|exit (code|status)|returned non-zero|Killed|"
                    r"No such file|not found|denied|^E\s{2,}", re.I)
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{2,}")
# Commands whose output IS the content the agent asked for (every line may be the answer): never outlined. The rest
# (test runners, builds, installs, linters, training and server logs) are mostly repeated noise around a few signals.
CONTENT = re.compile(r"(^|[|;&]\s*|\bxargs\s+)(grep|rg|ag|ack|findstr|select-string|sls|git\s+(diff|show|log|grep|blame|"
                     r"status)|diff|cat|type|head|tail|less|more|get-content|gc|ls|dir|tree|find|get-childitem|gci|"
                     r"echo|printf|write-output|jq|yq|sed|awk|wc|sort|uniq|cut|curl|wget|env|printenv|which|where|"
                     r"python\d?(\.exe)?\s+-c|node\s+-e|sqlite3|psql)\b", re.I)


def is_content_command(cmd):
    """True when the command's output is the content itself (search hits, diffs, listings, printed values)."""
    return bool(CONTENT.search(cmd or ""))


def outline_output(text, words=(), head=10, tail=10, ctx=8, min_chars=1000, keep=0.6):
    """text -> outlined text or None. words: the agent's intent (its command, task, reasoning); the ones that occur on
    at most 5 lines of the output count as rare and their lines are kept."""
    if not text or len(text) < min_chars:
        return None
    lines = text.split("\n")
    n = len(lines)
    if n <= head + tail + 2:
        return None
    show = set(range(min(head, n))) | set(range(max(0, n - tail), n))
    for i, l in enumerate(lines):
        if SIGNAL.search(l):
            show.update(range(max(0, i - ctx), min(n, i + ctx + 1)))
    cand = {w for w in words if len(w) >= 3}
    if cand:
        where = {}
        for i, l in enumerate(lines):
            for w in cand & set(WORD.findall(l)):
                where.setdefault(w, []).append(i)
        for w, idx in where.items():
            if len(idx) <= 5:
                for i in idx:
                    show.update(range(max(0, i - 2), min(n, i + 3)))
    out, i = [], 0
    while i < n:
        if i in show:
            out.append(lines[i])
            i += 1
            continue
        j = i
        while j < n and j not in show:
            j += 1
        out.append(f"[... {j - i} lines omitted (lines {i + 1}-{j}); kept: start, end, errors, lines matching your task;"
                   f" rerun with grep/head/tail or a narrower command to see them ...]")
        i = j
    new = "\n".join(out)
    if tokens(new) > keep * tokens(text):
        return None
    return f"[cortexnet] long output: {n} lines, ~{tokens(text)} tokens, outlined to ~{tokens(new)}.\n" + new
