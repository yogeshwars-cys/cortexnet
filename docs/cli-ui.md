# Cortex CLI — UI Features Implementation Plan

Grounding: current CLI is `cortexnet/cli.py` + `cortexnet/cli_ui.py`.
`Agent` (`cli.py:319-449`) owns `msgs`, `recs`, `faded/reused/usage`, calls
`watermark_fade()` each step in `turn()` (`cli.py:385-430`) and reports
`status()` (`cli.py:432-434`). `Workspace` (`cli.py:84-205`) owns
`call()/t_read_file()/outline_of()`. UI is `PlainUI` / `RichUI(PlainUI)`
(`cli_ui.py:85-293`) made by `make_ui()` (`cli_ui.py:296-306`); the REPL loop
is `main()` (`cli.py:469-526`), input via `ui.read()`, output via
`ui.thought/step/answer/note/status/error/confirm`, context gauge via
`RichUI.gauge()` (`cli_ui.py:264-269`) + `stats` dict updated in `status()`.
History file is already `.cortexnet/cli_history` (`cli.py:499`).

Goal: keep every feature small, offline-first, and consistent with
outline-first + fade-on-watermark. No new server dependency.

## 0. Prereq: command plumbing (do first, ~1 day)

All slash features below hang off this.

- In `main()` loop (`cli.py:506-516`), before `agent.turn(user)`, intercept:
  `if user.startswith("/") or user in ("#…", "read …"): handle_local(user); continue`.
- Add `Agent.local(cmd) -> str | None` returning text to show via
  `ui.answer()`, or `None` to fall through to the model. Keep model path
  untouched.
- Add `PlainUI` no-op defaults + `RichUI` rich versions for each new screen
  so `--plain` never breaks: `show_reads/show_why/show_context/show_map`.
- Rule: slash commands never call the backend, never append to `msgs`
  (except `/resume` which restores `msgs`). They may read `agent.recs/msgs`,
  `ws.reader`, `ws.saved`.

Tests: extend `tests/test_cli.py` with a fake backend; assert slash input
produces no `chat()` call.

## 1. See what CortexNET is doing

### 1.1 `/reads` — last reads table
- Source: `Agent.recs` (each has `name/args/sections/key`) + `Agent._live/_valid`
  (`cli.py:337-345`) + `ws.saved`, `summarize()` (`cli_ui.py:41-61`).
- `PlainUI.show_reads(rows)`: one line each —
  `path | outline + N bodies [+ menu] | whole | faded/stale/live | tok`.
- `RichUI`: `rich.table.Table` with columns File / Shown / State / Saved.
- State per rec: `faded` if `not _live`, `stale` if `not _valid`, else `live`.
- Tokens: reuse `tokens()` from `outline` (already used in `summarize`).

### 1.2 `/why <file>` — why these bodies
- Source: `ws.reader` last decision. Needs a small addition: `Reader` stores
  `self.last = dict(path, P, Pc, gate, included)` in `read()` (currently
  returns only text; add attribute, no signature change).
- Display: per included symbol `P_include`, `P_coproc` (or `—`), gate weights
  `q/h/trail` from `Coprocessor.gate_weights()`, and policy cause
  (`rule_ab bypass` vs `budget top-k` vs `coproc_top force`).
- If no coproc, show v1 features that mattered (`query_overlap/trail/centrality`)
  from `learn.py:FEATURES`. Keep to top-3 to avoid noise.
- `RichUI`: bar per symbol (`█` reuse `gauge` style); `PlainUI`: numbers.

### 1.3 `/context` — what fades next
- Source: `fade.tool_results(msgs)` order + `watermark_fade(..., mark, low)`
  params from `Agent.ctx/mark/low`. Dry-run: copy the candidate ordering
  logic (`fade.py:49-97` passes 0/1/2) without mutating; list oldest
  `read_file` results with est. tokens and pass (superseded → outline →
  pointer). Reuse `msg_tokens()` (`cli.py:314-316`).
- Show: `est context / ctx`, watermark line (`0.85*ctx`), `low` line, per-item
  `pass + tok + path`. This demystifies the `faded N old reads` note
  (`cli.py:394`).

## 2. Use the outline better

### 2.1 `read N[,M]` — pick by number
- Source: last `read_file` outline for a path. Map `N` → symbol via the same
  `Symbol.label()` order shown in `outline()` (`outline.py:218-222`).
- Implement in `Workspace`: `number_map: dict[path, list[(start,end)]]` set in
  `t_read_file`; `t_read_num(path, nums)` calls existing `t_read_lines` ranges.
  No `Reader` change; misses still logged via `Reader.miss()` path because
  `read_lines` goes through the same tool.
- Accept `read 3,5 path/to/file.py` and bare `read 3` when only one file was
  outlined last. Parse in `main()` before slash dispatch.

### 2.2 `#id` — approve from keyboard
- Source: `[connected]` menu ids already emitted by `Reader.read()` when
  `menu=True`. `Workspace` has `t_cortex_approve` (`cli.py:125-126`).
- In REPL: input matching `^#[\w-]+( [\w-]+)*$` (e.g. `> #3 #7`) → call
  `ws.call("cortex_approve", {"ids": [...]})`, show via `ui.step`, append
  result as a local tool message pair without a model round-trip (same shape
  as `turn()` lines 415-427 but with `role: user = "#3 …"` prefix). Simplest
  correct version: treat as one `agent.turn("#… — approve …")` so history stays
  consistent; optimize to direct append later.

### 2.3 `/map [dir]` — repo symbol map
- Source: `Workspace.files()` + `outline_of()` (`cli.py:128-205`) + optional
  `CoNeedGraph.neighbours()` if `--coproc` graph present.
- Output: per file `path — N symbols: a(), b(), …` truncated to 40 files like
  the system prompt listing (`cli.py:323`). `RichUI`: tree; `PlainUI`: indent.
- Cache per `epoch`/`mtime` (reuse `Agent._mtime`); invalidate on edit like
  grep/list invalidation (`_valid`, `cli.py:342-345`).

## 3. Session survival

### 3.1 `/compact` — fade now
- Call `watermark_fade(msgs, context(), ctx, outline_of, mark=0, low=low)` —
  i.e. force down to `low` without waiting for `0.85*limit`. Reuse the
  existing `turn()` fade block; surface count via `ui.note` (same string).
- Add `Agent.compact(frac=low) -> n`.

### 3.2 `/pin <file>` + `/unpin` — exempt from fade
- Add `Agent.pinned: set[path]`. In `turn()`, wrap `outline_of` passed to
  `watermark_fade` so pinned paths return full text (skip outline/pointer
  shrink for those `tool_results` entries). Simplest: filter `msgs` indices
  whose tool args path is pinned out of fade candidates — needs a 5-line hook
  in `fade.watermark_fade` (`keep` predicate param, default keeps behavior).
- `/pins` lists; pins persist in session file (below).

### 3.3 Resume / fork — `cortex --resume [--fork]`
- Persist: `msgs + recs(meta only) + epoch + pinned + usage + ws.saved` to
  `.cortexnet/session.jsonl` after each turn (append snapshot; cap 5MB, rotate).
  `rec.msg` object refs can't serialize — store `key/name/args` and re-link by
  matching `tool_call_id` on load; `_live` degrades to content check (accept).
- `--resume`: load file if present, rebuild `Agent` state, continue REPL.
  `--fork` (with resume): copy session to `.cortexnet/session-fork-<ts>.jsonl`
  first. Document that faded bodies stay faded (pointers, not refetch).

### 3.4 `/undo` — revert last edit
- `Workspace.t_edit_file/t_write_file` already take exact `old/new`
  (`cli.py:158-181`). Add `ws.undo_stack: list[(path, old_content)]` — read file
  bytes before each mutation (cap 20 entries, 1MB each).
- `/undo` restores bytes, bumps `epoch` (so grep/list invalidate), notes via
  `ui.note`. `/diff` shows last undo entry as diff (reuse `difflib`, already
  imported in `cli.py`).

## 4. Safer + faster loop

### 4.1 `--plan` mode
- `argparse --plan` (`cli.py:469-489`): `confirm = decline-all` for
  `MUTATING = edit/write/run` (`cli.py:73`); pass `mode="plan"` into
  `ui.header` (already renders `info['mode']`). System prompt: append one line
  `You are in plan mode: read-only.` to `SYSTEM`.
- One-line change, high value for hosted APIs where turns cost.

### 4.2 `//test`, `//lint` presets + outlined runs
- `Workspace` keeps `last_cmds: list`; `//test` re-runs last `run` starting
  with `pytest|go test|cargo test|npm test`, else prompts. Long-output
  outlining already exists via `shellout.py` (`summarize` detects
  `[cortexnet] long output`); wire `t_run` through it by default with
  `/full` to bypass.

### 4.3 Confirm with diff + `y/n/e`
- `ui.confirm(question, detail)` already shows diff panel in `RichUI`
  (`cli_ui.py:242-254`). Extend return to `y/n/e`: on `e`, open `$EDITOR` on a
  temp file with `new` text, re-read, retry `edit_file` once. `PlainUI` keeps
  `y/N` (return False on `e` with a note).

## Status (2026-10-28)

Built, with tests in `tests/test_cli_commands.py` (12) and `tests/test_workflow.py`; full suite 99 passed.

- [x] §0 plumbing: `slash()` and `typed()` run before `agent.turn()`; neither calls the model.
- [x] 1.1 `/reads`: table from `Agent.read_rows()`: shown, live / stale / faded, tokens, pinned marker.
- [x] 1.2 `/why [file]`: `Reader.last[path]` (mode, policy, per-leaf P and coprocessor P, cause: `rule_ab`,
  `budget`, `coproc_top`); shows what was included and the 3 closest left out.
- [x] 1.3 `/context`: adds `fade.preview()` (the same plan `watermark_fade` applies, no mutation) as a table.
- [x] 2.1 `read N[,M][-K] [file]` with `/outline file` numbering (`Workspace.symbol_list`).
- [x] 2.2 `#id ...`: `Agent.inject("cortex_approve")`. Injected calls are a user line, an assistant tool call and
  its result, so roles alternate; `AnthropicMessages.convert` also merges same-role neighbours.
- [x] 2.3 `/map [dir]`: top-level symbols per source file, cached by mtime.
- [x] 3.1 `/compact` (already there; now honours pins).
- [x] 3.2 `/pin`, `/unpin`, `/pin` (list): `pinned=` in `watermark_fade` / `preview`; superseded copies still go.
- [x] 3.3 `--resume [file]`, `--fork`: `Sessions` (atomic save after every turn and state-changing command; a fresh
  run archives the last session to `.cortexnet/sessions/`, keeps 10).
- [x] 3.4 `/undo`, `/diff`: `Workspace.undo_stack` (20 entries); the model is told with the next user message.
- [x] 4.1 `--plan`: mutating tools removed from the tool list, `PLAN_LINE` added to the system prompt.
- [x] 4.2 `/test [cmd]`, `/lint [cmd]` (single slash; `//` collided with slash parsing): given command, else the last
  matching `run`, else a default from pyproject / package.json / go.mod / Cargo.toml.
- [x] 4.3 `y/n/e`: `e` opens `$VISUAL` / `$EDITOR` (notepad / vi fallback) on the proposed `new` or `content`, then
  shows the diff again; the result tells the model the user revised it.
- Fixed on the way: `edit_file` turned CRLF files into LF; it now keeps the file's line ending.
- Not done: `/full` (bypass the outlining of long `run` output).

## Build order (suggested PRs)

1. PR1: §0 plumbing + `/reads` + `/context` (read-only, no state changes).
2. PR2: `read N` + `#id` (input parsing + Workspace map).
3. PR3: `/why` (Reader.last) + `/map`.
4. PR4: `/compact` + `/pin` (fade hook) + `--plan`.
5. PR5: resume/fork + `/undo` (persistence, needs care).
6. PR6: presets + confirm-e (polish).

Each PR: update this doc's checkbox, add tests in `tests/test_cli.py`
(fake backend + `PlainUI(out=list.append)`), keep `RichUI` import-guarded
(`make_ui` already falls back when not a tty).
