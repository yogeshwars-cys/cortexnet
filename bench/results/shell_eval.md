# Shell-output outlining on logged agent sessions

Local Claude Code logs, 3,203 Bash/PowerShell results of >= 1,000 characters (private data: aggregates only), outlined as cortexnet.shellout makes them. Reference recall: of the words the agent used in its next turn that came from the output (not from the command), the share still visible after outlining.

| rule | outlined | tokens before -> after | kept per outlined output (median) | outlined outputs the next turn drew on | reference recall (mean) | with every used word visible |
|---|---|---|---|---|---|---|
| every command | 1,047 (33%) | 3,043,067 -> 1,953,740 (64%) | 40% | 86 | 0.662 | 42% |
| log-like commands only (the proxy's rule) | 23 (1%) | 3,043,067 -> 3,020,994 (99%) | 38% | 0 | nan | 0% |

Reference recall is a proxy for harm: a word the agent used next that the outline hid would have cost a rerun or a wrong step. The log-like rule was fixed from what the commands do, before this measurement.
