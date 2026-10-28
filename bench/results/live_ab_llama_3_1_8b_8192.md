# Live A/B: llama-3.1-8b, 20 code questions (1177s)

| arm | accuracy | input tokens (sum over turns) | cached | output (reasoning) | tool-output tokens | tool calls | cost |
|---|---|---|---|---|---|---|---|
| baseline | 3/20 | 77,668 | 0 | 538 (0) | 115,149 | 22 | $0.000 |
| cortex | 7/20 | 23,695 | 0 | 374 (0) | 7,419 | 10 | $0.000 |

Cortex input tokens = 31% of baseline.

Local run: prefill tokens are what the server actually computed (after its prefix cache); peak context is the largest single request (prompt + completion).

| arm | answered | server errors | prefill tokens | prefill tok/s | prefill s | generation s | wall s | peak context (max / mean) |
|---|---|---|---|---|---|---|---|---|
| baseline | 3/20 | 17 | 44,119 | 200 | 220.7 | 26.8 | 630 | 8,087 / 2,390 |
| cortex | 7/20 | 11 | 12,955 | 218 | 59.4 | 17.8 | 547 | 3,238 / 802 |

Server errors counted as failures:
- task 1 baseline: "request (15617 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15617,"n_ctx":8192}} (rejected request ~11,816 tokens)
- task 2 baseline: "request (15616 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":15616,"n_ctx":8192}} (rejected request ~11,814 tokens)
- task 5 baseline: :"request (11629 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":11629,"n_ctx":8192}} (rejected request ~7,961 tokens)
- task 6 baseline: e":"request (9672 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":9672,"n_ctx":8192}} (rejected request ~6,938 tokens)
- task 8 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~214 tokens)
- task 9 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~212 tokens)
- task 10 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~219 tokens)
- task 11 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~214 tokens)
- task 12 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~210 tokens)
- task 13 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~212 tokens)
- task 14 baseline: "request (19332 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":19332,"n_ctx":8192}} (rejected request ~16,094 tokens)
- task 15 baseline: "request (19336 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":19336,"n_ctx":8192}} (rejected request ~16,103 tokens)
- task 16 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~217 tokens)
- task 17 baseline: "request (22829 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":22829,"n_ctx":8192}} (rejected request ~18,693 tokens)
- task 18 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~214 tokens)
- task 19 baseline: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~212 tokens)
- task 20 baseline: "request (22831 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":22831,"n_ctx":8192}} (rejected request ~18,694 tokens)
- task 8 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~214 tokens)
- task 9 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~212 tokens)
- task 10 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~219 tokens)
- task 11 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~214 tokens)
- task 12 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~210 tokens)
- task 13 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~212 tokens)
- task 14 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~207 tokens)
- task 15 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~216 tokens)
- task 16 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~217 tokens)
- task 18 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~214 tokens)
- task 19 cortex: HTTP 500: {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}} (rejected request ~212 tokens)

Per task (baseline / cortex): ok, input tokens, tool calls

- 1. What is the default value of full_below in the Reader class?  False/True, 8,578/2,182, 2/1
- 2. What value does the DECAY constant in reader.py have?  False/True, 8,576/2,203, 2/1
- 3. In learn.py, what prior weight does the 'changed' feature get?  True/True, 14,390/1,872, 3/1
- 4. What is the default learning rate (lr) of IncludeModel?  True/True, 14,382/1,780, 3/1
- 5. What is the default section size used by sections() in outline.py?  False/True, 12,657/1,758, 3/1
- 6. Which protocolVersion does the MCP server return by default when the client sends none?  False/False, 15,798/1,583, 4/1
- 7. Which Reader method records whether a request was a miss?  True/False, 1,215/4,473, 1/2
- 8. In learn.py, which function computes the symbol-graph edge weight between two symbols?  False/False, 0/0, 0/0
- 9. In stdlib/argparse.py, what is the default max_help_position of HelpFormatter?  False/False, 0/0, 0/0
- 10. In stdlib/argparse.py, which ArgumentParser method expands arguments that start with fromfile_prefix_chars?  False/False, 0/0, 0/0
- 11. In stdlib/argparse.py, which ArgumentParser method raises the 'invalid choice' error?  False/False, 0/0, 0/0
- 12. In stdlib/textwrap.py, what is the default placeholder of TextWrapper?  False/False, 0/0, 0/0
- 13. In stdlib/shlex.py, what does the shlex class set as its default commenters?  False/False, 0/0, 0/0
- 14. In stdlib/configparser.py, what is MAX_INTERPOLATION_DEPTH?  False/False, 515/0, 1/0
- 15. In stdlib/configparser.py, which interpolation class implements the ${section:option} syntax?  False/False, 519/0, 1/0
- 16. In stdlib/difflib.py, above what sequence length does SequenceMatcher's autojunk heuristic apply?  False/False, 0/0, 0/0
- 17. In stdlib/logging/handlers.py, what is DEFAULT_SOAP_LOGGING_PORT?  False/True, 518/3,916, 1/1
- 18. In stdlib/logging/handlers.py, what is the default `when` of TimedRotatingFileHandler?  False/False, 0/0, 0/0
- 19. In stdlib/logging/handlers.py, what is the default flushLevel of MemoryHandler?  False/False, 0/0, 0/0
- 20. In stdlib/logging/handlers.py, how many seconds does _MIDNIGHT hold?  False/True, 520/3,928, 1/1
