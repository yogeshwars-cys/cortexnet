"""Shell-output outlining on real agent logs: tokens saved, and whether what the agent used next stays visible.

Reads the local Claude Code logs (private: only aggregates are printed and written). For every Bash / PowerShell tool
result of at least 1,000 characters, the output is outlined as the proxy would (cortexnet.shellout.outline_output,
intent = the command + the agent's words in that turn). Reference recall: the words the agent used in its NEXT turn
(its text and the inputs of its next tool calls) that came from this output (present in it, not in the command) -
the share of them still visible in the outline. 1.0 means nothing the agent went on to use was hidden.
Usage: python -I bench/shell_eval.py [logs dir] [--out results/shell_eval.md]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cortexnet.outline import tokens  # noqa: E402
from cortexnet.shellout import is_content_command, outline_output  # noqa: E402

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_./-]{3,}|\d{3,}")
SHELL = {"Bash", "PowerShell"}


def text_of(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
    return ""


def episodes(src):
    """Yields (command, said, output, next_turn_text) per shell result, in log order."""
    files = glob.glob(os.path.join(src, "*", "*.jsonl")) + glob.glob(os.path.join(src, "*", "*", "subagents", "*.jsonl"))
    for f in files:
        pend, waiting = {}, []                          # call id -> (command, said); results awaiting a next turn
        for line in open(f, encoding="utf8", errors="replace"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            c = (r.get("message") or {}).get("content")
            if r.get("type") == "user" and isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result" and b.get("tool_use_id") in pend:
                        cmd, said = pend.pop(b["tool_use_id"])
                        body = b["content"] if isinstance(b.get("content"), str) else text_of(b.get("content"))
                        if len(body) >= 1000:
                            waiting.append((cmd, said, body))
                continue
            if r.get("type") != "assistant" or not isinstance(c, list):
                continue
            said = " ".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
            nxt = said + " " + " ".join(json.dumps(b.get("input") or {}) for b in c
                                       if isinstance(b, dict) and b.get("type") == "tool_use")
            for cmd, s, body in waiting:                  # this assistant message is the next turn after them
                yield cmd, s, body, nxt
            waiting = []
            for b in c:
                if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("name") in SHELL:
                    pend[b.get("id")] = (str((b.get("input") or {}).get("command") or ""), said)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", nargs="?", default=os.path.join(os.path.expanduser("~"), ".claude", "projects"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "shell_eval.md"))
    a = ap.parse_args()
    rules = {"every command": lambda cmd: True, "log-like commands only (the proxy's rule)":
             lambda cmd: not is_content_command(cmd)}
    R = {k: dict(before=[], after=[], rec=[], outlined=0) for k in rules}
    for cmd, said, body, nxt in episodes(a.src):
        words = WORD.findall(cmd + " " + said)
        refs = (set(WORD.findall(nxt)) & set(WORD.findall(body))) - set(WORD.findall(cmd))
        for k, ok in rules.items():
            new = outline_output(body, words) if ok(cmd) else None
            out = new if new is not None else body
            r = R[k]
            r["outlined"] += new is not None
            r["before"].append(tokens(body))
            r["after"].append(tokens(out))
            if refs and new is not None:
                r["rec"].append(len(refs & set(WORD.findall(out))) / len(refs))
    n = len(R["every command"]["before"])
    L = ["# Shell-output outlining on logged agent sessions", "",
         f"Local Claude Code logs, {n:,} Bash/PowerShell results of >= 1,000 characters (private data: aggregates "
         f"only), outlined as cortexnet.shellout makes them. Reference recall: of the words the agent used in its next "
         f"turn that came from the output (not from the command), the share still visible after outlining.", "",
         "| rule | outlined | tokens before -> after | kept per outlined output (median) | outlined outputs the "
         "next turn drew on | reference recall (mean) | with every used word visible |", "|---|---|---|---|---|---|---|"]
    for k, r in R.items():
        b, f = sum(r["before"]), sum(r["after"])
        kept = [x / y for x, y in zip(r["after"], r["before"]) if x < y] or [1]
        L.append(f"| {k} | {r['outlined']:,} ({r['outlined'] / max(n, 1):.0%}) | {b:,} -> {f:,} ({f / max(b, 1):.0%}) | "
                 f"{np.median(kept):.0%} | {len(r['rec']):,} | "
                 f"{np.mean(r['rec']) if r['rec'] else float('nan'):.3f} | "
                 f"{np.mean([x >= 1 - 1e-9 for x in r['rec']]) if r['rec'] else 0:.0%} |")
    L += ["", "Reference recall is a proxy for harm: a word the agent used next that the outline hid would have cost a "
          "rerun or a wrong step. The log-like rule was fixed from what the commands do, before this measurement."]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
