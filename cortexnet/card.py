"""The task card: the agent's working memory, kept outside the raw history.

The agent calls `note` to record what it did, what it found (with file:line), its plan, and what it wants next. Each
call returns the whole current card as a tool result, so the card is appended to the conversation (cache-friendly,
the system prompt never changes) and fade treats older copies as superseded and never touches the newest. With the
findings on the card, fade can drop the raw reads they came from.

The `want` field is also a latent request to CortexNET: the prefetcher reads it to warm the next files, and reads use
it as part of their query (cortexnet/prefetch.py).
"""

PHASES = ("explore", "plan", "edit", "verify")


class TaskCard:
    def __init__(self, goal=""):
        self.goal, self.phase, self.version = goal, "explore", 0
        self.plan = []          # [[text, done]]
        self.findings, self.done = [], []
        self.want = ""

    def apply(self, done="", finding="", want="", plan=None, check=None, phase=""):
        """Update from a `note` call; returns the rendered card."""
        if plan:
            items = [plan] if isinstance(plan, str) else list(plan)
            have = {p[0] for p in self.plan}
            self.plan += [[str(t), False] for t in items if str(t).strip() and str(t) not in have]
        for c in ([check] if isinstance(check, (int, str)) else list(check or [])):
            try:
                self.plan[int(c) - 1][1] = True
            except (ValueError, IndexError):
                pass
        if done and done not in self.done:
            self.done.append(str(done))
        if finding and finding not in self.findings:
            self.findings.append(str(finding))
        if want:
            self.want = str(want)
        elif self.plan and all(d for _, d in self.plan):
            self.want = ""                                 # finished: no stale latent request for the prefetcher
        if phase in PHASES:
            self.phase = phase
        self.version += 1
        return self.render()

    def render(self):
        lines = [f"[task card v{self.version}] goal: {self.goal}", f"phase: {self.phase}"]
        if self.plan:
            lines.append("plan:")
            lines += [f"  {i}. [{'x' if d else ' '}] {t}" for i, (t, d) in enumerate(self.plan, 1)]
        if self.findings:
            lines.append("findings:")
            lines += [f"  - {f}" for f in self.findings[-30:]]
        if self.done:
            lines.append("done:")
            lines += [f"  - {d}" for d in self.done[-15:]]
        if self.want:
            lines.append(f"next: {self.want}")
        return "\n".join(lines)

    def empty(self):
        return not (self.plan or self.findings or self.done or self.want)
