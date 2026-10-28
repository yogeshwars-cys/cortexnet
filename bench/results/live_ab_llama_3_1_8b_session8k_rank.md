# Live A/B: llama-3.1-8b, 20 code questions (1177s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 520 | 0 | 22 (0) | 6,144 | 1 | $0.000 |
| cortex | 6/20 | 59,562 | 0 | 304 (0) | 6,317 | 9 | $0.000 |
| cortex+fade | 12/20 | 208,461 | 0 | 811 (0) | 46,447 | 25 | $0.000 |
| coproc+fade | 16/20 | 194,053 | 0 | 909 (0) | 32,231 | 21 | $0.000 |

cortex input tokens = 11454% of baseline.
cortex+fade input tokens = 40089% of baseline.
coproc+fade input tokens = 37318% of baseline.

Local run: prefill tokens are what the server actually computed (after its prefix cache); peak context is the largest single request (prompt + completion).

| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | wall s | peak context (max / mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 20 | 520 | 297 | 1.7 | 0.9 | 4 | 542 / 27 |
| cortex | 6/20 | 12 | 23,735 | 378 | 62.7 | 14.8 | 79 | 5,630 / 1,750 |
| cortex+fade | 12/20 | 0 | 107,701 | 240 | 449.1 | 43.4 | 497 | 6,766 / 4,833 |
| coproc+fade | 16/20 | 1 | 118,077 | 241 | 489.0 | 50.4 | 597 | 6,710 / 5,376 |

Session mode: one conversation per arm, 20 questions in a row, server context 8,192; +fade arms shrink stale reads in one batch above 85%, down to 50% (newest 4 results kept, at most 25% of the context).

- baseline: reads faded 0, context at the end 16,485, first context overflow at question 1
- cortex: reads faded 0, context at the end 11,198, first context overflow at question 9
- cortex+fade: reads faded 24, context at the end 5,316, first context overflow at question none
- coproc+fade: reads faded 21, context at the end 5,270, first context overflow at question none

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
- task 9 cortex: ":"request (9485 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9485,"n_ctx":8192}} (rejected request ~10,855 tokens)
- task 10 cortex: ":"request (9512 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9512,"n_ctx":8192}} (rejected request ~10,896 tokens)
- task 11 cortex: ":"request (9537 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9537,"n_ctx":8192}} (rejected request ~10,929 tokens)
- task 12 cortex: ":"request (9558 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9558,"n_ctx":8192}} (rejected request ~10,956 tokens)
- task 13 cortex: ":"request (9583 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9583,"n_ctx":8192}} (rejected request ~10,986 tokens)
- task 14 cortex: ":"request (9603 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9603,"n_ctx":8192}} (rejected request ~11,008 tokens)
- task 15 cortex: ":"request (9627 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9627,"n_ctx":8192}} (rejected request ~11,045 tokens)
- task 16 cortex: ":"request (9655 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9655,"n_ctx":8192}} (rejected request ~11,083 tokens)
- task 17 cortex: ":"request (9678 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9678,"n_ctx":8192}} (rejected request ~11,108 tokens)
- task 18 cortex: ":"request (9707 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9707,"n_ctx":8192}} (rejected request ~11,141 tokens)
- task 19 cortex: ":"request (9731 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9731,"n_ctx":8192}} (rejected request ~11,171 tokens)
- task 20 cortex: ":"request (9756 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9756,"n_ctx":8192}} (rejected request ~11,198 tokens)
- task 15 coproc+fade: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~5,611 tokens)

Per task (baseline / cortex / cortex+fade / coproc+fade): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  False/True/True/True, 520/2,235/2,235/5,724, 1/1/1/2
- 2. What value does the DECAY constant in reader.py have?  False/True/True/True, 0/3,994/3,994/7,178, 0/1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  False/True/True/True, 0/5,403/5,403/8,609, 0/1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  False/True/True/True, 0/6,343/6,343/9,569, 0/1/1/1
- 5. What is the default section size used by sections() in outline.py?  False/True/True/True, 0/7,448/7,448/10,696, 0/1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  False/False/False/True, 0/8,351/8,351/12,171, 0/1/1/1
- 7. Which Reader method records whether a request was a miss?  False/False/False/False, 0/9,441/9,441/10,563, 0/1/1/1
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  False/True/True/True, 0/10,743/10,743/8,722, 0/1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  False/False/True/True, 0/5,604/10,909/10,292, 0/1/1/1
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  False/False/False/True, 0/0/33,073/11,267, 0/0/5/1
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  False/False/False/True, 0/0/17,669/11,542, 0/0/2/1
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  False/False/False/False, 0/0/12,750/12,584, 0/0/1/1
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  False/False/True/True, 0/0/9,976/9,864, 0/0/1/1
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  False/False/True/True, 0/0/8,335/8,209, 0/0/1/1
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  False/False/True/False, 0/0/10,243/5,026, 0/0/1/1
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  False/False/False/False, 0/0/10,109/10,763, 0/0/1/1
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  False/False/True/True, 0/0/9,960/10,167, 0/0/1/1
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  False/False/False/True, 0/0/10,113/9,971, 0/0/1/1
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  False/False/False/True, 0/0/10,565/10,443, 0/0/1/1
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  False/False/True/True, 0/0/10,801/10,693, 0/0/1/1
