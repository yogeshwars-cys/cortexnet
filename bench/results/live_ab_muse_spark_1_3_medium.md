# Live A/B: muse-spark-1.3 (Meta Model API, reasoning effort medium), 14 of 20 code questions

Run stopped by hand after task 14 (enough for a first read); numbers from the per-task log.

| arm | accuracy | input tokens (sum over turns) | tool calls | cost |
|---|---|---|---|---|
| baseline (whole-file reads) | 14/14 | 119,568 | 16 | $0.153 |
| cortex (outline-first) | 14/14 | 62,751 | 25 | $0.065 |

Cortex input tokens = 52% of baseline, cost = 43%. Same accuracy.

Read: the gain is all on big files (argparse, 2,690 lines: 32% and 18% of baseline input). On small files the
outline costs an extra round trip and can be worse (tasks 4, 6, 7, 8). One run, easy lookup questions, no repeats.

Per task (baseline / cortex): input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  6,390/2,415, 1/1
- 2. What value does the DECAY constant in reader.py have?  6,389/2,430, 1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  3,420/2,291, 1/1
- 4. What is the default learning rate (lr) of IncludeModel?  1,722/3,440, 1/2
- 5. What is the default section size used by sections() in outline.py?  4,775/2,305, 1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  3,176/4,792, 1/2
- 7. Which Reader method records whether a request was a miss?  6,385/11,467, 1/4
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  3,416/3,777, 1/2
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  1,665/1,929, 1/1
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  33,746/10,751, 2/2
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  33,571/6,161, 2/3
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  7,715/4,473, 1/2
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  5,487/4,544, 1/2
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  1,711/1,976, 1/1
