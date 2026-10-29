"""Terminal UI for Cortex CLI. RichUI (Rich + prompt_toolkit, when installed) or PlainUI (plain text, for pipes, CI
and machines without them). The agent loop only calls: note, thought, step, thinking, and Workspace calls confirm."""
import contextlib
import os
import re

from .outline import tokens

# The logo's letter colors, left to right (sampled from docs/cortex_logo.png)
STOPS = [(0x2e, 0x93, 0x45), (0x3a, 0x9e, 0x4b), (0x72, 0xe0, 0x2f), (0xa3, 0xf3, 0x19), (0xcd, 0xf4, 0x1d),
         (0xec, 0xed, 0x62)]
WORDMARK = [  # "CORTEX", figlet ANSI Shadow style
    " ██████╗ ██████╗ ██████╗ ████████╗███████╗██╗  ██╗",
    "██╔════╝██╔═══██╗██╔══██╗╚══██╔══╝██╔════╝╚██╗██╔╝",
    "██║     ██║   ██║██████╔╝   ██║   █████╗   ╚███╔╝ ",
    "██║     ██║   ██║██╔══██╗   ██║   ██╔══╝   ██╔██╗ ",
    "╚██████╗╚██████╔╝██║  ██║   ██║   ███████╗██╔╝ ██╗",
    " ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝",
]


def gradient(t):
    """A hex color at position t (0..1) along the logo's gradient."""
    t = min(max(t, 0.0), 1.0) * (len(STOPS) - 1)
    i = min(int(t), len(STOPS) - 2)
    f = t - i
    a, b = STOPS[i], STOPS[i + 1]
    return "#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * f)) for k in range(3))


def BRAIN_ART():
    """The brain as half-block ANSI art (built by scripts/make_banner.py), or None if the asset is missing."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "brain.ans")
    try:
        with open(p, encoding="utf8") as f:
            return f.read().rstrip("\n")
    except OSError:
        return None


def summarize(name, args, out):
    """A one-line description of a tool step's result."""
    first = out.split("\n", 1)[0]
    if out.startswith("error:") or "declined" in first:
        return first[:120]
    if name == "read_file":
        m = re.search(r": (\d+) lines, ~([\d,]+) tokens\. Outline", first)
        if m:
            bodies = len(re.findall(r"^--- (?:lines|x-ray)", out, re.M))
            menu = " + menu" if "[connected]" in out else ""
            return f"outline + {bodies} bod{'y' if bodies == 1 else 'ies'}{menu}, {tokens(out):,} of {m.group(2)} tok"
        return f"whole file, {tokens(out):,} tok"
    if name == "run":
        return first + (", outlined" if "[cortexnet] long output" in out else "") + f", {tokens(out):,} tok"
    if name == "edit_file" and out.startswith("edited"):
        add = len(re.findall(r"^\+(?!\+\+)", out, re.M))
        rem = len(re.findall(r"^-(?!--)", out, re.M))
        return f"+{add} -{rem}"
    if name == "grep":
        return "no matches" if out == "no matches" else f"{len(out.splitlines())} matches"
    if name == "note":
        return first.split("]", 1)[0].lstrip("[")
    if name == "scout":
        return f"answered, {tokens(out):,} tok"
    return f"{tokens(out):,} tok"


def usage(s):
    """Session totals: tokens sent to the model (and the share its cache served), output, calls answered from context."""
    sent, cached = s.get("sent", 0), s.get("cached", 0)
    cache = f" ({cached / sent:.0%} cached)" if sent and cached else ""
    reuse = f" | {s['reused']} repeat reads answered from context" if s.get("reused") else ""
    return f"sent {sent:,}{cache}, out {s.get('out', 0):,}{reuse}"


def brief(name, args):
    if name in ("read_file", "read_lines", "list_dir", "edit_file", "write_file") and "path" in args:
        extra = f" {args['start']}-{args['end']}" if name == "read_lines" else ""
        q = f"  \"{args['query'][:50]}\"" if args.get("query") else ""
        return f"{args['path']}{extra}{q}"
    if name == "run":
        return str(args.get("command", ""))[:100]
    if name == "grep":
        return f"/{args.get('pattern', '')}/ {args.get('path', '')}".strip()
    if name == "scout":
        return str(args.get("question", ""))[:100]
    if name == "note":
        parts = [k for k in ("plan", "check", "finding", "done", "want", "phase") if args.get(k)]
        return ", ".join(parts) or "read"
    s = " ".join(f"{k}={v!r}" for k, v in args.items())
    return s if len(s) < 100 else s[:97] + "..."


def edit_text(text, suffix=".txt"):
    """Open `text` in $VISUAL / $EDITOR (notepad on Windows, vi elsewhere) and return what was saved, or None when
    nothing changed or no editor could run."""
    import shlex
    import subprocess
    import tempfile
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or ("notepad" if os.name == "nt" else "vi")
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="cortex-edit-")
    try:
        with os.fdopen(fd, "w", encoding="utf8", newline="") as f:
            f.write(text)
        subprocess.call([t.strip('"') for t in shlex.split(editor, posix=os.name != "nt")] + [path])
        with open(path, encoding="utf8", newline="") as f:
            new = f.read()
    except (OSError, ValueError):
        return None
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return None if new == text else new


def table_lines(header, rows):
    """Aligned plain-text columns."""
    rows = [tuple(str(c) for c in r) for r in rows]
    w = [max(len(str(h)), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = lambda r: "  " + "  ".join(c.ljust(w[i]) for i, c in enumerate(r)).rstrip()  # noqa: E731
    return [fmt(header)] + [fmt(r) for r in rows]


class PlainUI:
    editor = staticmethod(edit_text)          # `e` at an edit prompt: revise the change in this editor

    def __init__(self, out=print, inp=input):
        self.out, self.inp = out, inp

    def table(self, title, header, rows):
        self.out(title)
        self.out("\n".join(table_lines(header, rows)))

    def header(self, info):
        self.out(f"Cortex CLI | {info['repo']} | {info['backend']} | context {info['ctx']:,}"
                 f"{' | ' + info['mode'] if info.get('mode') else ''}")

    def note(self, text):
        self.out(f"  [{text}]")

    def say(self, text, style=""):
        self.out(text)

    @contextlib.contextmanager
    def nested(self, label=""):
        self.out(f"  > scout {label[:90]}")
        yield

    def thought(self, text):
        self.out(text)

    def step(self, name, args, out):
        self.out(f"  > {name} {brief(name, args)}  ({summarize(name, args, out)})")
        if name == "note" and not out.startswith("error"):
            self.out("\n".join("    " + line for line in out.split("\n")[1:]))

    @contextlib.contextmanager
    def thinking(self):
        yield

    def confirm(self, question, detail, editable=None):
        """True / False, or the revised text when `editable` is given and the user answers e."""
        if detail:
            self.out(detail if len(detail) < 4000 else detail[:4000] + "\n[...]")
        while True:
            try:
                ans = self.inp(f"  {question} [{'y/N/e' if editable is not None else 'y/N'}] ").strip().lower()
            except EOFError:
                return False
            if ans != "e" or editable is None:
                return ans in ("y", "yes")
            new = self.editor(editable)
            if new is not None:
                return new
            self.out("  [nothing changed in the editor]")

    def answer(self, text):
        self.out("\n" + text)

    def error(self, text):
        self.out(f"error: {text}")

    def status(self, s):
        f = s.get("files")
        snn = f" | spiking file manager: hit@3 {f['hit@3']:.0%}" if f else ""
        self.out(f"[context {s['context']:,} / {s['ctx']:,} tokens | faded {s['faded']} | CortexNET reads saved "
                 f"{s['saved']:,} tokens{snn} | {usage(s)}]")

    def read(self):
        try:
            return self.inp("\n> ").strip()
        except EOFError:
            return None


class RichUI(PlainUI):
    """Claude Code style: a dot per action (`● Read(path)`), its result on an elbow line under it, a timed spinner,
    the answer as Markdown after a dot, line-numbered diffs before an edit, and a prompt with slash-command completion
    and a bottom toolbar (context gauge, cache share). Falls back to ASCII glyphs where UTF-8 cannot be encoded."""
    VERB = {"read_file": "Read", "read_lines": "Read", "cortex_approve": "Approve", "grep": "Search", "list_dir": "List",
            "edit_file": "Update", "write_file": "Write", "run": "Bash", "note": "Card", "scout": "Scout"}

    def __init__(self, console=None, history=None, inp=None, commands=()):
        from rich.console import Console
        self.console = console or Console(highlight=False)
        self.stats = dict(context=0, ctx=1, faded=0, saved=0)
        self.inp, self.session, self.depth = inp, None, 0
        if inp is None:
            try:
                from prompt_toolkit import PromptSession
                from prompt_toolkit.history import FileHistory
                os.makedirs(os.path.dirname(history), exist_ok=True) if history else None
                self.session = PromptSession(history=FileHistory(history) if history else None,
                                             completer=slash_completer(commands), complete_while_typing=True)
            except (ImportError, OSError):
                self.session = None
                self.inp = input

    def utf(self):
        return (self.console.encoding or "").lower().replace("-", "").startswith("utf")

    def g(self, name):
        """A glyph, or its ASCII stand-in on a legacy code page."""
        return (GLYPHS if self.utf() else ASCII)[name]

    def pad(self):
        return "    " * self.depth

    def wordmark(self):
        """CORTEX in block letters, colored left to right along the logo's gradient."""
        from rich.text import Text
        width = max(len(line) for line in WORDMARK)
        out = Text()
        for i, line in enumerate(WORDMARK):
            for x, ch in enumerate(line):
                out.append(ch, style=f"bold {gradient(x / max(width - 1, 1))}" if ch != " " else None)
            if i < len(WORDMARK) - 1:
                out.append("\n")
        return out

    def banner(self, info):
        """The logo header, or None where it cannot be drawn well (legacy code page, no color, too narrow)."""
        from rich.table import Table
        from rich.text import Text
        if not self.utf() or self.console.color_system is None or self.console.width < 56:
            return None
        right = Table.grid()
        right.add_column()
        right.add_row(self.wordmark())
        right.add_row(Text("outline-first context for coding agents", style=f"italic {gradient(0.55)}"))
        right.add_row("")
        meta = Table.grid(padding=(0, 2))
        meta.add_column(style=f"bold {gradient(0.15)}")
        meta.add_column(style="dim")
        meta.add_row("model", info["backend"])
        meta.add_row("repo", info["repo"])
        meta.add_row("context", f"{info['ctx']:,} tokens, fades near {int(0.85 * info['ctx']):,}")
        if info.get("mode"):
            meta.add_row("mode", info["mode"])
        right.add_row(meta)
        brain = BRAIN_ART()
        need = (max(len(re.sub(r"\x1b\[[0-9;]*m", "", line)) for line in brain.split("\n")) + 3
                + max(len(line) for line in WORDMARK)) if brain else 0
        if brain is None or self.console.width < need:
            return right                                      # narrow: the wordmark and the session details
        grid = Table.grid(padding=(0, 3))
        grid.add_column(vertical="middle")
        grid.add_column(vertical="middle")
        grid.add_row(Text.from_ansi(brain), right)
        return grid

    def header(self, info):
        from rich.panel import Panel
        from rich.table import Table
        art = self.banner(info)
        sep = self.g("sep")
        tips = f"[dim]  /help for commands {sep} ctrl-c stops a task {sep} ctrl-d quits[/]"
        if art is not None:
            self.console.print()
            self.console.print(art)
            self.console.print("\n" + tips)
            return
        t = Table.grid(padding=(0, 2))
        t.add_column(style="bold")
        t.add_column()
        t.add_row("repo", info["repo"])
        t.add_row("model", info["backend"])
        t.add_row("context", f"{info['ctx']:,} tokens, fades near {int(0.85 * info['ctx']):,}")
        if info.get("mode"):
            t.add_row("mode", info["mode"])
        self.console.print(Panel(t, title="[bold]Cortex CLI[/]", subtitle="outline-first reads, fade on a watermark",
                                 border_style=gradient(0.3), expand=False))
        self.console.print(tips)

    def say(self, text, style=""):
        """Plain output for slash commands (no markup interpretation)."""
        from rich.text import Text
        self.console.print(Text(text, style=style))

    def table(self, title, header, rows):
        from rich import box
        from rich.table import Table
        t = Table(title=title, title_justify="left", title_style="bold", box=box.SIMPLE_HEAD, pad_edge=False,
                  header_style="dim")
        for h in header:
            t.add_column(h, overflow="fold")
        for r in rows:
            t.add_row(*(str(c) for c in r))
        self.console.print(t)

    def note(self, text):
        self.result(text, style="dim italic")

    def _dot(self, color, body):
        from rich.table import Table
        t = Table.grid(padding=(0, 1, 0, 0))
        t.add_column(width=len(self.pad()) + 1, no_wrap=True)
        t.add_column()
        t.add_row(f"{self.pad()}[{color}]{self.g('dot')}[/]", body)
        self.console.print(t)

    def thought(self, text):
        from rich.markdown import Markdown
        self._dot("white", Markdown(text))

    def result(self, text, style="dim"):
        """Elbow line(s) under an action: the first with the elbow, the rest aligned under it."""
        from rich.text import Text
        for k, line in enumerate(text.split("\n")):
            lead = f"{self.pad()}  {self.g('elbow')}  " if k == 0 else f"{self.pad()}     "
            self.console.print(Text(lead + line, style=style))

    @contextlib.contextmanager
    def nested(self, label=""):
        """A scout: its header now, its steps indented under it."""
        from rich.text import Text
        head = Text(self.pad())
        head.append(self.g("dot"), style=gradient(0.6))
        head.append(" Scout", style="bold")
        head.append(f"({label[:90]})")
        self.console.print(head)
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def step(self, name, args, out):
        from rich.text import Text
        res = summarize(name, args, out)
        bad = res.startswith(("error", "the user declined")) or bool(re.match(r"exit code (?!0\b)", res))
        if name == "scout":                                   # its header was printed when it started
            body = out.split("\n", 1)[1] if out.startswith("scout answer:") else out
            self.result("\n".join(body.strip().split("\n")[:6]), style="red" if bad else "dim")
            return
        line = Text(self.pad())
        line.append(self.g("dot"), style="red" if bad else gradient(0.45))
        line.append(" " + self.VERB.get(name, name), style="bold")
        phase = re.search(r"^phase: (\w+)", out, re.M) if name == "note" else None
        line.append(f"({brief(name, args)}{' ' + self.g('sep') + ' ' + phase.group(1) if phase else ''})")
        self.console.print(line)
        if name == "note" and not bad:
            self.card(out)
        elif name in ("edit_file", "write_file") and not bad:
            add = len(re.findall(r"^\+(?!\+\+)", out, re.M))
            rem = len(re.findall(r"^-(?!--)", out, re.M))
            what = f"with {add} addition{'s' * (add != 1)} and {rem} removal{'s' * (rem != 1)}" if add or rem else res
            self.result(f"{'Updated' if name == 'edit_file' else 'Wrote'} {args.get('path', '')} {what}")
        else:
            self.result(res, style="red" if bad else "dim")

    def card(self, text):
        """The task card as a todo list, with its findings and the next want."""
        from rich.text import Text
        rows, section = [], ""
        for line in text.split("\n")[1:]:
            m = re.match(r"\s+\d+\. \[(x| )\] (.*)", line)
            if line.endswith(":") and not line.startswith(" "):
                section = line[:-1]
            elif m:
                done = m.group(1) == "x"
                rows.append((self.g("done" if done else "todo") + " " + m.group(2), "dim strike" if done else ""))
            elif section == "findings" and line.startswith("  - "):
                rows.append((self.g("sep") + " " + line[4:], "dim"))
            elif line.startswith("next:"):
                rows.append((line, "dim italic"))
        if not rows:
            rows = [(text.split("\n")[0], "dim")]
        for k, (body, style) in enumerate(rows):
            lead = f"{self.pad()}  {self.g('elbow')}  " if k == 0 else f"{self.pad()}     "
            self.console.print(Text(lead) + Text(body, style=style))

    @contextlib.contextmanager
    def thinking(self):
        """`✻ Thinking… (4s · ctrl-c to interrupt)`: the star cycling, the seconds counting, gone when done."""
        import threading
        import time
        from rich.live import Live
        from rich.text import Text
        t0, stop, frames = time.time(), threading.Event(), self.g("spin")

        def label(k):
            t = Text(self.pad())
            t.append(frames[k % len(frames)], style=gradient(0.6))
            t.append(" Thinking" + self.g("ellipsis"), style=f"bold {gradient(0.35)}")
            t.append(f" ({int(time.time() - t0)}s {self.g('sep')} ctrl-c to interrupt)", style="dim")
            return t

        with Live(label(0), console=self.console, transient=True, refresh_per_second=8) as live:
            def tick():
                k = 0
                while not stop.wait(0.15):
                    k += 1
                    live.update(label(k))
            th = threading.Thread(target=tick, daemon=True)
            th.start()
            try:
                yield
            finally:
                stop.set()
                th.join(1)

    def confirm(self, question, detail, editable=None):
        from rich.panel import Panel
        if detail:
            target = re.search(r"^\+\+\+ (?:b/)?(\S+)", detail, re.M)
            body = numbered_diff(detail, self.utf()) if target else _plain(detail[:6000])
            title = f"[bold]Update[/]({_esc(target.group(1))})" if target else "[bold]Bash[/]"
            self.console.print(Panel(body, border_style=gradient(0.3), title=title, title_align="left"))
        while True:
            try:
                ans = self._input(f"  {question} ({'y/n/e = edit' if editable is not None else 'y/n'}) ")
            except (EOFError, KeyboardInterrupt):
                return False
            ans = ans.strip().lower()
            if ans != "e" or editable is None:
                return ans in ("y", "yes")
            new = self.editor(editable)
            if new is not None:
                return new
            self.note("nothing changed in the editor")

    def answer(self, text):
        from rich.markdown import Markdown
        self.console.print()
        self._dot("white", Markdown(text or "(no answer)"))

    def error(self, text):
        self.console.print(f"{self.pad()}[bold red]{self.g('dot')} error:[/] {_esc(text)}")

    def gauge(self, s, width=20):
        frac = min(1.0, s["context"] / max(s["ctx"], 1))
        full = int(round(frac * width))
        return f"{self.g('bar_on') * full}{self.g('bar_off') * (width - full)} {frac:.0%}"

    def status(self, s):
        self.stats = dict(s)
        frac = s["context"] / max(s["ctx"], 1)
        color = "green" if frac < 0.6 else "yellow" if frac < 0.85 else "red"
        bits = [f"[{color}]context {s['context']:,}/{s['ctx']:,} ({frac:.0%})[/]"]
        if s.get("sent"):
            bits.append(f"cache {s.get('cached', 0) / s['sent']:.0%}")
        if s.get("saved"):
            bits.append(f"reads saved {s['saved']:,} tok")
        if s.get("faded"):
            bits.append(f"faded {s['faded']}")
        hits, miss = s.get("prefetch_hits", 0), s.get("prefetch_misses", 0)
        if hits + miss:
            bits.append(f"prefetch {hits}/{hits + miss}")
        if s.get("files"):
            bits.append(f"snn hit@3 {s['files']['hit@3']:.0%}")
        self.console.print(f"\n  [dim]{(' ' + self.g('sep') + ' ').join(bits)}[/]")

    def _input(self, prompt):
        if self.session is not None:
            return self.session.prompt(prompt)
        return self.inp(prompt)

    def toolbar(self):
        s, sep = self.stats, f" {self.g('sep')} "
        cache = f"{sep}cache {s.get('cached', 0) / s['sent']:.0%}" if s.get("sent") else ""
        return f" context {self.gauge(s)}{cache}{sep}/ for commands{sep}ctrl-d quits "

    def read(self):
        from rich.rule import Rule
        try:
            self.console.print(Rule(style="dim"))
            if self.session is not None:
                return self.session.prompt("> ", bottom_toolbar=self.toolbar).strip()
            return self.inp("> ").strip()
        except EOFError:
            return None
        except KeyboardInterrupt:
            return ""


GLYPHS = dict(dot="●", elbow="⎿", sep="·", ellipsis="…", todo="☐", done="☒", bar_on="█", bar_off="░",
              spin="✻✽✶✳✢·✢✳✶✽")
ASCII = dict(dot="*", elbow="L", sep="|", ellipsis="...", todo="[ ]", done="[x]", bar_on="#", bar_off="-",
             spin="-\\|/")


def _esc(text):
    from rich.markup import escape
    return escape(str(text))


def _plain(text):
    from rich.text import Text
    return Text(text)


def numbered_diff(diff, utf=True):
    """A unified diff with line numbers, removed lines on red and added ones on green (like Claude Code's edit view)."""
    from rich.text import Text
    out, old, new = Text(), 0, 0
    for line in diff[:12000].split("\n"):
        if line.startswith(("---", "+++")):
            continue
        m = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
        if m:
            if old or new:
                out.append("      ...\n", style="dim")
            old, new = int(m.group(1)), int(m.group(2))
        elif line.startswith("-"):
            out.append(f"{old:>5} ", style="dim")
            out.append(f"- {line[1:]}\n", style="on #3c1414" if utf else "red")
            old += 1
        elif line.startswith("+"):
            out.append(f"{new:>5} ", style="dim")
            out.append(f"+ {line[1:]}\n", style="on #143c1e" if utf else "green")
            new += 1
        elif line.startswith(" "):
            out.append(f"{new:>5}   {line[1:]}\n", style="dim")
            old, new = old + 1, new + 1
    out.rstrip()
    return out


def slash_completer(commands):
    """prompt_toolkit completion for /commands, only while the line is a lone /word."""
    from prompt_toolkit.completion import Completer, Completion

    class Slash(Completer):
        def get_completions(self, document, event):
            t = document.text_before_cursor
            if not t.startswith("/") or " " in t:
                return
            for name, desc in commands:
                if name.startswith(t):
                    yield Completion(name, start_position=-len(t), display_meta=desc)
    return Slash()


def make_ui(plain=False, history=None, commands=()):
    """RichUI on a terminal with Rich installed, otherwise PlainUI."""
    if not plain:
        try:
            import rich  # noqa: F401
            import sys
            if sys.stdout.isatty():
                return RichUI(history=history, commands=commands)
        except ImportError:
            pass
    return PlainUI()
