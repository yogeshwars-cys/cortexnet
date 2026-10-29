# Live A/B: muse-spark-1.3 (reasoning effort medium), 20 code questions (1921s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 19/20 | 3,255,752 | 1,983,159 | 5,634 (3,143) | 130,134 | 25 | $1.912 |
| cortex | 20/20 | 380,533 | 288,063 | 5,490 (2,560) | 7,990 | 29 | $0.182 |
| full | 20/20 | 449,825 | 333,789 | 5,463 (2,345) | 9,149 | 30 | $0.218 |

cortex input tokens = 12% of baseline.
full input tokens = 14% of baseline.

Session mode: one conversation per arm, 20 questions in a row, server context 8,192; +fade arms shrink stale reads in one batch above 85%, down to 50% (newest 4 results kept, at most 25% of the context).

- baseline: reads faded 0, context at the end 138,964, first context overflow at question none
- cortex: reads faded 0, context at the end 13,305, first context overflow at question none
- full: reads faded 0, context at the end 15,234, first context overflow at question none

3 turns resent the full history because the chained response was gone.

Per task (baseline / cortex / full): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  True/True/True, 9,952/5,400/5,654, 1/2/2
- 2. What value does the DECAY constant in reader.py have?  True/True/True, 18,599/5,456/5,616, 1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  True/True/True, 20,810/6,282/6,519, 1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  True/True/True, 23,005/7,335/7,646, 1/1/1
- 5. What is the default section size used by sections() in outline.py?  True/True/True, 27,051/8,606/8,974, 1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  True/True/True, 33,013/15,248/10,610, 1/2/1
- 7. Which Reader method records whether a request was a miss?  True/True/True, 35,464/17,719/26,045, 1/2/3
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  True/True/True, 35,997/13,096/16,325, 1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  True/True/True, 85,752/21,070/25,809, 2/2/2
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  True/True/True, 179,559/31,376/28,160, 2/3/3
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  True/True/True, 273,279/25,647/31,426, 2/2/2
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  True/True/True, 224,031/18,036/22,266, 1/1/1
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  True/True/True, 224,570/18,555/34,659, 1/1/2
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  True/True/True, 225,022/18,987/23,983, 1/1/1
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  True/True/True, 354,762/30,593/50,666, 2/2/3
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  True/True/True, 258,869/34,933/28,010, 1/2/1
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  False/True/True, 395,356/24,747/28,527, 2/1/1
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  True/True/True, 275,859/25,257/29,063, 1/1/1
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  True/True/True, 277,133/25,840/29,666, 1/1/1
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  True/True/True, 277,669/26,350/30,201, 1/1/1
