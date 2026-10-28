# Live A/B: llama-3.1-8b, 20 code questions (700s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 520 | 0 | 22 (0) | 6,144 | 1 | $0.000 |
| cortex | 6/20 | 59,634 | 0 | 304 (0) | 6,317 | 9 | $0.000 |
| cortex+fade | 13/20 | 229,472 | 0 | 954 (0) | 62,629 | 30 | $0.000 |
| coproc+fade | 13/20 | 230,158 | 0 | 842 (0) | 48,924 | 26 | $0.000 |
| full+fade | 13/20 | 192,173 | 0 | 840 (0) | 32,159 | 20 | $0.000 |

cortex input tokens = 11468% of baseline.
cortex+fade input tokens = 44129% of baseline.
coproc+fade input tokens = 44261% of baseline.
full+fade input tokens = 36956% of baseline.

Local run: prefill tokens are what the server actually computed (after its prefix cache); peak context is the largest single request (prompt + completion).

| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | wall s | peak context (max / mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 20 | 520 | 753 | 0.7 | 0.9 | 2 | 542 / 27 |
| cortex | 6/20 | 12 | 28,991 | 870 | 33.3 | 12.6 | 47 | 5,638 / 1,752 |
| cortex+fade | 13/20 | 1 | 121,405 | 798 | 152.2 | 41.3 | 204 | 6,323 / 4,760 |
| coproc+fade | 13/20 | 0 | 142,026 | 810 | 175.4 | 36.6 | 247 | 6,873 / 5,452 |
| full+fade | 13/20 | 0 | 124,416 | 808 | 154.1 | 36.7 | 199 | 6,833 / 5,289 |

Session mode: one conversation per arm, 20 questions in a row, server context 8,192; +fade arms shrink stale reads in one batch above 85%, down to 50% (newest 4 results kept, at most 25% of the context).

- baseline: reads faded 0, context at the end 16,485, first context overflow at question 1
- cortex: reads faded 0, context at the end 11,214, first context overflow at question 9
- cortex+fade: reads faded 29, context at the end 5,632, first context overflow at question none
- coproc+fade: reads faded 26, context at the end 5,382, first context overflow at question none
- full+fade: reads faded 20, context at the end 5,289, first context overflow at question none

Server errors counted as failures:
- task 1 baseline: ":"request (8488 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8488,"n_ctx":8192}} (rejected request ~15,629 tokens)
- task 2 baseline: ":"request (8505 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8505,"n_ctx":8192}} (rejected request ~15,661 tokens)
- task 3 baseline: ":"request (8525 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8525,"n_ctx":8192}} (rejected request ~15,698 tokens)
- task 4 baseline: ":"request (8543 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8543,"n_ctx":8192}} (rejected request ~15,730 tokens)
- task 5 baseline: ":"request (8562 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8562,"n_ctx":8192}} (rejected request ~15,769 tokens)
- task 6 baseline: ":"request (8583 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8583,"n_ctx":8192}} (rejected request ~15,820 tokens)
- task 7 baseline: ":"request (8599 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8599,"n_ctx":8192}} (rejected request ~15,855 tokens)
- task 8 baseline: ":"request (8620 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8620,"n_ctx":8192}} (rejected request ~15,906 tokens)
- task 9 baseline: ":"request (8644 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8644,"n_ctx":8192}} (rejected request ~15,953 tokens)
- task 10 baseline: ":"request (8671 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8671,"n_ctx":8192}} (rejected request ~16,016 tokens)
- task 11 baseline: ":"request (8696 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8696,"n_ctx":8192}} (rejected request ~16,068 tokens)
- task 12 baseline: ":"request (8717 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8717,"n_ctx":8192}} (rejected request ~16,110 tokens)
- task 13 baseline: ":"request (8742 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8742,"n_ctx":8192}} (rejected request ~16,156 tokens)
- task 14 baseline: ":"request (8762 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8762,"n_ctx":8192}} (rejected request ~16,191 tokens)
- task 15 baseline: ":"request (8786 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8786,"n_ctx":8192}} (rejected request ~16,247 tokens)
- task 16 baseline: ":"request (8814 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8814,"n_ctx":8192}} (rejected request ~16,306 tokens)
- task 17 baseline: ":"request (8837 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8837,"n_ctx":8192}} (rejected request ~16,345 tokens)
- task 18 baseline: ":"request (8866 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8866,"n_ctx":8192}} (rejected request ~16,397 tokens)
- task 19 baseline: ":"request (8890 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8890,"n_ctx":8192}} (rejected request ~16,443 tokens)
- task 20 baseline: ":"request (8915 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8915,"n_ctx":8192}} (rejected request ~16,485 tokens)
- task 9 cortex: ":"request (9494 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9494,"n_ctx":8192}} (rejected request ~10,870 tokens)
- task 10 cortex: ":"request (9521 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9521,"n_ctx":8192}} (rejected request ~10,911 tokens)
- task 11 cortex: ":"request (9546 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9546,"n_ctx":8192}} (rejected request ~10,945 tokens)
- task 12 cortex: ":"request (9567 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9567,"n_ctx":8192}} (rejected request ~10,972 tokens)
- task 13 cortex: ":"request (9592 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9592,"n_ctx":8192}} (rejected request ~11,002 tokens)
- task 14 cortex: ":"request (9612 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9612,"n_ctx":8192}} (rejected request ~11,024 tokens)
- task 15 cortex: ":"request (9636 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9636,"n_ctx":8192}} (rejected request ~11,060 tokens)
- task 16 cortex: ":"request (9664 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9664,"n_ctx":8192}} (rejected request ~11,098 tokens)
- task 17 cortex: ":"request (9687 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9687,"n_ctx":8192}} (rejected request ~11,124 tokens)
- task 18 cortex: ":"request (9716 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9716,"n_ctx":8192}} (rejected request ~11,157 tokens)
- task 19 cortex: ":"request (9740 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9740,"n_ctx":8192}} (rejected request ~11,187 tokens)
- task 20 cortex: ":"request (9765 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9765,"n_ctx":8192}} (rejected request ~11,214 tokens)
- task 11 cortex+fade: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~6,270 tokens)

Per task (baseline / cortex / cortex+fade / coproc+fade / full+fade): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  False/True/True/True/True, 520/2,236/2,236/5,727/2,979, 1/1/1/2/1
- 2. What value does the DECAY constant in reader.py have?  False/True/True/True/True, 0/3,997/3,997/7,161/5,251, 0/1/1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  False/True/True/True/True, 0/5,408/5,408/8,572/6,718, 0/1/1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  False/True/True/True/True, 0/6,350/6,350/9,514/7,718, 0/1/1/1/1
- 5. What is the default section size used by sections() in outline.py?  False/True/True/True/True, 0/7,457/7,457/10,621/8,903, 0/1/1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  False/False/False/True/True, 0/8,362/8,362/12,074/10,485, 0/1/1/1/1
- 7. Which Reader method records whether a request was a miss?  False/False/False/False/False, 0/9,454/9,454/10,436/12,439, 0/1/1/1/1
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  False/True/True/True/True, 0/10,758/10,758/8,569/10,639, 0/1/1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  False/False/True/True/True, 0/5,612/10,918/10,111/9,867, 0/1/1/1/1
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  False/False/False/False/False, 0/0/33,079/33,638/11,697, 0/0/5/5/1
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  False/False/False/False/False, 0/0/36,194/17,867/11,625, 0/0/6/2/1
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  False/False/False/False/False, 0/0/9,228/12,923/12,657, 0/0/1/1/1
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  False/False/True/True/True, 0/0/6,455/10,190/10,068, 0/0/1/1/1
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  False/False/True/True/True, 0/0/8,844/8,508/8,351, 0/0/1/1/1
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  False/False/True/True/True, 0/0/10,751/10,844/10,571, 0/0/1/1/1
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  False/False/False/False/False, 0/0/10,617/11,073/10,872, 0/0/1/1/1
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  False/False/True/True/True, 0/0/10,468/10,455/10,272, 0/0/1/1/1
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  False/False/False/False/False, 0/0/10,621/10,245/10,028, 0/0/1/1/1
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  False/False/True/False/False, 0/0/16,842/10,697/10,394, 0/0/2/1/1
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  False/False/True/True/True, 0/0/11,433/10,933/10,639, 0/0/1/1/1
