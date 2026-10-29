# Live A/B: llama-3.1-8b, 20 code questions (597s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| full+fade | 14/20 | 188,697 | 0 | 836 (0) | 32,002 | 20 | $0.000 |
| neuro+fade | 12/20 | 185,139 | 0 | 826 (0) | 32,002 | 20 | $0.000 |
| neurodyn+fade | 13/20 | 188,619 | 0 | 831 (0) | 32,004 | 20 | $0.000 |

neuro+fade input tokens = 98% of full+fade.
neurodyn+fade input tokens = 100% of full+fade.

Local run: prefill tokens are what the server actually computed (after its prefix cache); peak context is the largest single request (prompt + completion).

| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | wall s | peak context (max / mean) |
|---|---|---|---|---|---|---|---|---|
| full+fade | 14/20 | 0 | 65,095 | 595 | 109.5 | 44.0 | 183 | 6,781 / 5,190 |
| neuro+fade | 12/20 | 1 | 85,035 | 604 | 140.7 | 44.4 | 220 | 6,781 / 5,184 |
| neurodyn+fade | 13/20 | 0 | 86,816 | 596 | 145.6 | 45.0 | 195 | 6,776 / 5,188 |

Session mode: one conversation per arm, 20 questions in a row, server context 8,192; +fade arms shrink stale reads in one batch above 85%, down to 50% (newest 4 results kept, at most 25% of the context).

- full+fade: reads faded 20, context at the end 5,294, first context overflow at question none
- neuro+fade: reads faded 20, context at the end 5,280, first context overflow at question none
- neurodyn+fade: reads faded 20, context at the end 5,289, first context overflow at question none

Server errors counted as failures:
- task 13 neuro+fade: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~2,843 tokens)

Per task (full+fade / neuro+fade / neurodyn+fade): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  True/True/True, 2,597/2,597/2,597, 1/1/1
- 2. What value does the DECAY constant in reader.py have?  True/True/True, 4,573/4,573/4,573, 1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  True/True/True, 6,096/6,096/6,096, 1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  True/True/True, 7,051/7,051/7,051, 1/1/1
- 5. What is the default section size used by sections() in outline.py?  True/True/True, 8,334/8,334/8,334, 1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  True/True/True, 10,014/10,014/10,014, 1/1/1
- 7. Which Reader method records whether a request was a miss?  False/False/False, 12,073/12,073/12,073, 1/1/1
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  True/True/True, 10,620/10,620/10,620, 1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  True/True/True, 9,987/9,987/9,987, 1/1/1
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  False/False/False, 11,712/11,712/11,712, 1/1/1
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  False/False/False, 11,639/11,639/11,634, 1/1/1
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  False/False/False, 12,672/12,672/12,662, 1/1/1
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  True/False/True, 10,084/6,749/10,079, 1/1/1
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  True/True/True, 8,374/8,342/8,366, 1/1/1
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  True/True/True, 10,593/10,561/10,585, 1/1/1
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  False/False/False, 10,890/10,858/10,882, 1/1/1
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  True/True/True, 10,286/10,254/10,278, 1/1/1
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  False/False/False, 10,042/10,010/10,034, 1/1/1
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  True/False/False, 10,408/10,376/10,400, 1/1/1
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  True/True/True, 10,652/10,621/10,642, 1/1/1
