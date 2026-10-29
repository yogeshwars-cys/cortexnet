# Live A/B: meta-models/muse-spark-open-1026, 20 code questions (349s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 20/20 | 2,727,167 | 2,568,576 | 5,508 (3,302) | 143,599 | 22 | $0.000 |
| cortex | 20/20 | 447,347 | 428,928 | 5,349 (2,030) | 8,158 | 35 | $0.000 |
| full | 20/20 | 423,288 | 387,712 | 5,072 (1,934) | 7,330 | 33 | $0.000 |
| full+fade | 20/20 | 608,294 | 571,264 | 5,757 (2,375) | 11,753 | 38 | $0.000 |

cortex input tokens = 16% of baseline.
full input tokens = 16% of baseline.
full+fade input tokens = 22% of baseline.

Cached and reasoning columns recomputed from the trace's usage (the harness first read only the Responses API's field names; fixed). Baseten's prefix cache served 94% of the baseline's input; uncached input: baseline 158,591, cortex 18,419 (12%), full 35,576 (22%), full+fade 37,030 (23%). Cost shows $0 because Baseten's hackathon credit is not priced in the harness.

Session mode: one conversation per arm, 20 questions in a row, server context 32,768; +fade arms shrink stale reads in one batch above 85%, down to 50% (newest 4 results kept, at most 25% of the context).

- baseline: reads faded 0, context at the end 155,657, first context overflow at question none
- cortex: reads faded 0, context at the end 14,326, first context overflow at question none
- full: reads faded 0, context at the end 13,453, first context overflow at question none
- full+fade: reads faded 0, context at the end 18,712, first context overflow at question none

Per task (baseline / cortex / full / full+fade): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  True/True/True/True, 10,124/5,626/5,875/5,829, 1/2/2/2
- 2. What value does the DECAY constant in reader.py have?  True/True/True/True, 18,742/5,567/5,714/5,673, 1/1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  True/True/True/True, 20,914/6,371/6,560/6,539, 1/1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  True/True/True/True, 22,997/7,229/7,479/7,459, 1/1/1/1
- 5. What is the default section size used by sections() in outline.py?  True/True/True/True, 26,942/8,312/8,640/8,603, 1/1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  True/True/True/True, 32,875/14,710/10,299/10,229, 1/2/1/1
- 7. Which Reader method records whether a request was a miss?  True/True/True/True, 35,295/17,087/17,636/33,641, 1/2/2/5
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  True/True/True/True, 35,714/12,473/12,782/20,411, 1/1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  True/True/True/True, 36,038/19,790/20,168/31,697, 1/2/2/2
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  True/True/True/True, 85,774/21,624/22,002/33,890, 2/2/2/3
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  True/True/True/True, 179,358/23,929/24,256/36,955, 2/2/2/2
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  True/True/True/True, 167,264/26,161/26,386/39,187, 1/2/2/2
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  True/True/True/True, 177,492/27,738/27,963/40,829, 1/2/2/2
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  True/True/True/True, 197,866/28,975/29,218/42,245, 1/2/2/2
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  True/True/True/True, 214,425/41,817/30,617/44,233, 1/3/2/2
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  True/True/True/True, 242,842/35,211/32,580/47,686, 1/2/2/3
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  True/True/True/True, 290,573/36,709/34,087/49,939, 1/2/2/2
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  True/True/True/True, 310,241/38,459/35,837/51,689, 1/2/2/2
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  True/True/True/True, 310,614/41,110/38,488/54,319, 1/2/2/2
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  True/True/True/True, 311,077/28,449/26,701/37,241, 1/1/1/1
