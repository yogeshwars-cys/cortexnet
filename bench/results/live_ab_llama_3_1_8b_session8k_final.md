# Live A/B: llama-3.1-8b, 20 code questions (621s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 8,669 | 0 | 44 (0) | 11,722 | 2 | $0.000 |
| cortex | 6/20 | 58,308 | 0 | 304 (0) | 6,236 | 9 | $0.000 |
| cortex+fade | 13/20 | 207,047 | 0 | 808 (0) | 46,228 | 25 | $0.000 |
| coproc+fade | 13/20 | 207,047 | 0 | 808 (0) | 46,228 | 25 | $0.000 |

cortex input tokens = 673% of baseline.
cortex+fade input tokens = 2388% of baseline.
coproc+fade input tokens = 2388% of baseline.

Local run: prefill tokens are what the server actually computed (after its prefix cache); peak context is the largest single request (prompt + completion).

| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | wall s | peak context (max / mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 0/20 | 20 | 8,128 | 374 | 21.7 | 1.9 | 25 | 8,171 / 409 |
| cortex | 6/20 | 12 | 4,909 | 415 | 11.8 | 12.7 | 26 | 5,521 / 1,713 |
| cortex+fade | 13/20 | 0 | 58,099 | 363 | 159.9 | 35.6 | 201 | 6,768 / 4,799 |
| coproc+fade | 13/20 | 0 | 92,445 | 362 | 255.2 | 35.8 | 368 | 6,768 / 4,799 |

Session mode: one conversation per arm, 20 questions in a row, server context 8,192; +fade arms shrink stale reads in one batch above 85%, down to 50% (newest 4 results kept, at most 25% of the context).

- baseline: reads faded 0, context at the end 16,481, first context overflow at question 1
- cortex: reads faded 0, context at the end 11,109, first context overflow at question 9
- cortex+fade: reads faded 24, context at the end 5,314, first context overflow at question none
- coproc+fade: reads faded 24, context at the end 5,314, first context overflow at question none

Server errors counted as failures:
- task 1 baseline: "request (15778 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15778,"n_ctx":8192}} (rejected request ~16,014 tokens)
- task 2 baseline: "request (15795 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15795,"n_ctx":8192}} (rejected request ~16,031 tokens)
- task 3 baseline: "request (15815 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15815,"n_ctx":8192}} (rejected request ~16,051 tokens)
- task 4 baseline: "request (15833 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15833,"n_ctx":8192}} (rejected request ~16,069 tokens)
- task 5 baseline: "request (15852 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15852,"n_ctx":8192}} (rejected request ~16,090 tokens)
- task 6 baseline: "request (15873 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15873,"n_ctx":8192}} (rejected request ~16,118 tokens)
- task 7 baseline: "request (15889 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15889,"n_ctx":8192}} (rejected request ~16,137 tokens)
- task 8 baseline: "request (15910 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15910,"n_ctx":8192}} (rejected request ~16,165 tokens)
- task 9 baseline: "request (15934 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15934,"n_ctx":8192}} (rejected request ~16,190 tokens)
- task 10 baseline: "request (15961 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15961,"n_ctx":8192}} (rejected request ~16,225 tokens)
- task 11 baseline: "request (15986 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15986,"n_ctx":8192}} (rejected request ~16,253 tokens)
- task 12 baseline: "request (16007 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16007,"n_ctx":8192}} (rejected request ~16,276 tokens)
- task 13 baseline: "request (16032 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16032,"n_ctx":8192}} (rejected request ~16,302 tokens)
- task 14 baseline: "request (16052 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16052,"n_ctx":8192}} (rejected request ~16,320 tokens)
- task 15 baseline: "request (16076 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16076,"n_ctx":8192}} (rejected request ~16,351 tokens)
- task 16 baseline: "request (16104 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16104,"n_ctx":8192}} (rejected request ~16,383 tokens)
- task 17 baseline: "request (16127 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16127,"n_ctx":8192}} (rejected request ~16,405 tokens)
- task 18 baseline: "request (16156 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16156,"n_ctx":8192}} (rejected request ~16,433 tokens)
- task 19 baseline: "request (16180 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16180,"n_ctx":8192}} (rejected request ~16,458 tokens)
- task 20 baseline: "request (16205 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":16205,"n_ctx":8192}} (rejected request ~16,481 tokens)
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

Per task (baseline / cortex / cortex+fade / coproc+fade): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  False/True/True/True, 8,669/2,197/2,197/2,197, 2/1/1/1
- 2. What value does the DECAY constant in reader.py have?  False/True/True/True, 0/3,880/3,880/3,880, 0/1/1/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  False/True/True/True, 0/5,252/5,252/5,252, 0/1/1/1
- 4. What is the default learning rate (lr) of IncludeModel?  False/True/True/True, 0/6,194/6,194/6,194, 0/1/1/1
- 5. What is the default section size used by sections() in outline.py?  False/True/True/True, 0/7,301/7,301/7,301, 0/1/1/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  False/False/False/False, 0/8,206/8,206/8,206, 0/1/1/1
- 7. Which Reader method records whether a request was a miss?  False/False/False/False, 0/9,259/9,259/9,259, 0/1/1/1
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  False/True/True/True, 0/10,524/10,524/10,524, 0/1/1/1
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  False/False/True/True, 0/5,495/10,801/10,801, 0/1/1/1
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  False/False/False/False, 0/0/38,809/38,809, 0/0/6/6
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  False/False/False/False, 0/0/11,780/11,780, 0/0/1/1
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  False/False/False/False, 0/0/12,753/12,753, 0/0/1/1
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  False/False/True/True, 0/0/9,980/9,980, 0/0/1/1
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  False/False/True/True, 0/0/8,334/8,334, 0/0/1/1
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  False/False/True/True, 0/0/10,241/10,241, 0/0/1/1
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  False/False/False/False, 0/0/10,107/10,107, 0/0/1/1
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  False/False/True/True, 0/0/9,958/9,958, 0/0/1/1
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  False/False/False/False, 0/0/10,111/10,111, 0/0/1/1
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  False/False/True/True, 0/0/10,563/10,563, 0/0/1/1
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  False/False/True/True, 0/0/10,797/10,797, 0/0/1/1
