# Live A/B: llama-3.1-8b, 20 code questions (824s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 8,592 | 0 | 44 (0) | 11,606 | 2 | $0.000 |
| cortex | 6/20 | 58,308 | 0 | 304 (0) | 6,236 | 9 | $0.000 |
| cortex+fade | 8/20 | 124,103 | 0 | 470 (0) | 16,350 | 14 | $0.000 |
| coproc+fade | 8/20 | 119,339 | 0 | 468 (0) | 15,942 | 14 | $0.000 |

cortex input tokens = 679% of baseline.
cortex+fade input tokens = 1444% of baseline.
coproc+fade input tokens = 1389% of baseline.

Local run: prefill tokens are what the server actually computed (after its prefix cache); peak context is the largest single request (prompt + completion).

| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | wall s | peak context (max / mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 20 | 8,051 | 186 | 43.2 | 2.0 | 46 | 8,094 / 405 |
| cortex | 6/20 | 12 | 15,869 | 197 | 80.4 | 13.0 | 95 | 5,521 / 1,713 |
| cortex+fade | 8/20 | 7 | 53,247 | 193 | 275.3 | 20.7 | 299 | 7,025 / 3,460 |
| coproc+fade | 8/20 | 7 | 62,940 | 196 | 321.7 | 20.6 | 383 | 6,816 / 3,341 |

Session mode: one conversation per arm, 20 questions in a row, server context 8,192; +fade arms shrink stale reads in one batch above 85%, down to 50% (last 4 calls kept).

- baseline: reads faded 0, context at the end 16,327, first context overflow at question 1
- cortex: reads faded 0, context at the end 11,109, first context overflow at question 9
- cortex+fade: reads faded 10, context at the end 10,429, first context overflow at question 14
- coproc+fade: reads faded 10, context at the end 10,196, first context overflow at question 14

Server errors counted as failures:
- task 1 baseline: "request (15624 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15624,"n_ctx":8192}} (rejected request ~15,860 tokens)
- task 2 baseline: "request (15641 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15641,"n_ctx":8192}} (rejected request ~15,877 tokens)
- task 3 baseline: "request (15661 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15661,"n_ctx":8192}} (rejected request ~15,897 tokens)
- task 4 baseline: "request (15679 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15679,"n_ctx":8192}} (rejected request ~15,915 tokens)
- task 5 baseline: "request (15698 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15698,"n_ctx":8192}} (rejected request ~15,936 tokens)
- task 6 baseline: "request (15719 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15719,"n_ctx":8192}} (rejected request ~15,964 tokens)
- task 7 baseline: "request (15735 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15735,"n_ctx":8192}} (rejected request ~15,983 tokens)
- task 8 baseline: "request (15756 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15756,"n_ctx":8192}} (rejected request ~16,011 tokens)
- task 9 baseline: "request (15780 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15780,"n_ctx":8192}} (rejected request ~16,036 tokens)
- task 10 baseline: "request (15807 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15807,"n_ctx":8192}} (rejected request ~16,071 tokens)
- task 11 baseline: "request (15832 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15832,"n_ctx":8192}} (rejected request ~16,099 tokens)
- task 12 baseline: "request (15853 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15853,"n_ctx":8192}} (rejected request ~16,122 tokens)
- task 13 baseline: "request (15878 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15878,"n_ctx":8192}} (rejected request ~16,148 tokens)
- task 14 baseline: "request (15898 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15898,"n_ctx":8192}} (rejected request ~16,166 tokens)
- task 15 baseline: "request (15922 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15922,"n_ctx":8192}} (rejected request ~16,197 tokens)
- task 16 baseline: "request (15950 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15950,"n_ctx":8192}} (rejected request ~16,229 tokens)
- task 17 baseline: "request (15973 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15973,"n_ctx":8192}} (rejected request ~16,251 tokens)
- task 18 baseline: "request (16002 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16002,"n_ctx":8192}} (rejected request ~16,279 tokens)
- task 19 baseline: "request (16026 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16026,"n_ctx":8192}} (rejected request ~16,304 tokens)
- task 20 baseline: "request (16051 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16051,"n_ctx":8192}} (rejected request ~16,327 tokens)
- task 9 cortex: ":"request (9377 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9377,"n_ctx":8192}} (rejected request ~10,764 tokens)
- task 10 cortex: ":"request (9404 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9404,"n_ctx":8192}} (rejected request ~10,805 tokens)
- task 11 cortex: ":"request (9429 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9429,"n_ctx":8192}} (rejected request ~10,839 tokens)
- task 12 cortex: ":"request (9450 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9450,"n_ctx":8192}} (rejected request ~10,866 tokens)
- task 13 cortex: ":"request (9475 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9475,"n_ctx":8192}} (rejected request ~10,896 tokens)
- task 14 cortex: ":"request (9495 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9495,"n_ctx":8192}} (rejected request ~10,918 tokens)
- task 15 cortex: ":"request (9519 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9519,"n_ctx":8192}} (rejected request ~10,955 tokens)
- task 16 cortex: ":"request (9547 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9547,"n_ctx":8192}} (rejected request ~10,993 tokens)
- task 17 cortex: ":"request (9570 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9570,"n_ctx":8192}} (rejected request ~11,018 tokens)
- task 18 cortex: ":"request (9599 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9599,"n_ctx":8192}} (rejected request ~11,051 tokens)
- task 19 cortex: ":"request (9623 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9623,"n_ctx":8192}} (rejected request ~11,082 tokens)
- task 20 cortex: ":"request (9648 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9648,"n_ctx":8192}} (rejected request ~11,109 tokens)
- task 14 cortex+fade: ":"request (9876 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9876,"n_ctx":8192}} (rejected request ~10,271 tokens)
- task 15 cortex+fade: ":"request (9900 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9900,"n_ctx":8192}} (rejected request ~10,301 tokens)
- task 16 cortex+fade: ":"request (9928 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9928,"n_ctx":8192}} (rejected request ~10,333 tokens)
- task 17 cortex+fade: ":"request (9951 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9951,"n_ctx":8192}} (rejected request ~10,354 tokens)
- task 18 cortex+fade: ":"request (9980 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9980,"n_ctx":8192}} (rejected request ~10,382 tokens)
- task 19 cortex+fade: "request (10004 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":10004,"n_ctx":8192}} (rejected request ~10,407 tokens)
- task 20 cortex+fade: "request (10029 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":10029,"n_ctx":8192}} (rejected request ~10,429 tokens)
- task 14 coproc+fade: ":"request (9645 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9645,"n_ctx":8192}} (rejected request ~10,037 tokens)
- task 15 coproc+fade: ":"request (9669 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9669,"n_ctx":8192}} (rejected request ~10,068 tokens)
- task 16 coproc+fade: ":"request (9697 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9697,"n_ctx":8192}} (rejected request ~10,100 tokens)
- task 17 coproc+fade: ":"request (9720 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9720,"n_ctx":8192}} (rejected request ~10,121 tokens)
- task 18 coproc+fade: ":"request (9749 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9749,"n_ctx":8192}} (rejected request ~10,148 tokens)
- task 19 coproc+fade: ":"request (9773 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9773,"n_ctx":8192}} (rejected request ~10,173 tokens)
- task 20 coproc+fade: ":"request (9798 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9798,"n_ctx":8192}} (rejected request ~10,196 tokens)

Per task (baseline / cortex / cortex+fade / coproc+fade): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  False/True/True/True, 8,592/2,197/2,197/2,197, 2/1/1/1
- 2. What value does the DECAY constant in reader.py have?  False/True/True/True, 0/3,880/3,880/3,859, 0/1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  False/True/True/True, 0/5,252/5,252/5,126, 0/1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  False/True/True/True, 0/6,194/6,194/5,984, 0/1/1/1
- 5. What is the default section size used by sections() in outline.py?  False/True/True/True, 0/7,301/7,301/6,820, 0/1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  False/False/False/False, 0/8,206/8,206/7,454, 0/1/1/1
- 7. Which Reader method records whether a request was a miss?  False/False/False/False, 0/9,259/9,259/8,507, 0/1/1/1
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  False/True/True/True, 0/10,524/10,524/9,772, 0/1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  False/False/True/True, 0/5,495/12,022/11,646, 0/1/1/1
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  False/False/False/False, 0/0/13,075/13,055, 0/0/1/1
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  False/False/False/False, 0/0/12,601/12,372, 0/0/1/1
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  False/False/False/False, 0/0/12,837/12,419, 0/0/1/1
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  False/False/True/True, 0/0/13,755/13,337, 0/0/1/1
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  False/False/False/False, 0/0/7,000/6,791, 0/0/1/1
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  False/False/False/False, 0/0/0/0, 0/0/0/0
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  False/False/False/False, 0/0/0/0, 0/0/0/0
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  False/False/False/False, 0/0/0/0, 0/0/0/0
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  False/False/False/False, 0/0/0/0, 0/0/0/0
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  False/False/False/False, 0/0/0/0, 0/0/0/0
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  False/False/False/False, 0/0/0/0, 0/0/0/0
