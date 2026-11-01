# Held-out session stress test

`bench/heldout.py`: 364 machine-written questions over 8 package subtrees of SWE-bench Lite repositories (none used to build or tune CortexNET), one conversation per subtree and arm (up to 40 questions in a row).

## llama-1b, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 25/364 (7%) | 0 | 8/8 | never | 2,107 | 0.0 | 2,118 | 1 |
| cortex+fade | 22/364 (6%) | 0 | 8/8 | never | 2,162 | 0.0 | 2,172 | 1 |
| cortexf+fade | 33/364 (9%) | 0 | 8/8 | never | 2,130 | 0.0 | 2,139 | 1 |

- cortex+fade minus paged+fade: -0.8% correct (95% CI -2.2% to +0.3%, paired over 364 questions)
- cortexf+fade minus cortex+fade: +3.0% correct (95% CI +1.1% to +4.9%, paired over 364 questions)
- cortexf+fade minus paged+fade: +2.2% correct (95% CI +0.0% to +4.4%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 1/60 | 0/60 | 3/60 |
| autodoc | 0/60 | 0/60 | 0/60 |
| extensions | 0/10 | 0/10 | 0/10 |
| featext | 3/36 | 3/36 | 3/36 |
| modeling | 10/60 | 10/60 | 13/60 |
| mpl | 11/60 | 8/60 | 8/60 |
| plot | 0/34 | 0/34 | 0/34 |
| printing | 0/44 | 1/44 | 6/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 7/59 | 7/59 | 7/59 |
| const | 8/73 | 9/73 | 13/73 |
| raises | 0/81 | 0/81 | 0/81 |
| base | 7/131 | 3/131 | 11/131 |
| hop_call | 0/6 | 0/6 | 0/6 |
| hop_default | 1/1 | 1/1 | 1/1 |
| hop_const | 2/13 | 2/13 | 1/13 |

## llama-1b-full, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 23/364 (6%) | 0 | 8/8 | never | 22,695 | 3.8 | 5,759 | 411 |
| cortex+fade | 83/364 (23%) | 0 | 8/8 | never | 8,691 | 0.6 | 5,761 | 32 |
| cortexf+fade | 96/364 (26%) | 0 | 8/8 | never | 9,248 | 0.8 | 5,789 | 29 |

- cortex+fade minus paged+fade: +16.5% correct (95% CI +11.8% to +21.4%, paired over 364 questions)
- cortexf+fade minus cortex+fade: +3.6% correct (95% CI -1.9% to +9.1%, paired over 364 questions)
- cortexf+fade minus paged+fade: +20.1% correct (95% CI +14.8% to +25.0%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 8/60 | 16/60 | 17/60 |
| autodoc | 3/60 | 10/60 | 8/60 |
| extensions | 1/10 | 0/10 | 6/10 |
| featext | 0/36 | 17/36 | 9/36 |
| modeling | 4/60 | 28/60 | 30/60 |
| mpl | 3/60 | 4/60 | 10/60 |
| plot | 2/34 | 5/34 | 4/34 |
| printing | 2/44 | 3/44 | 12/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 4/59 | 22/59 | 23/59 |
| const | 11/73 | 23/73 | 29/73 |
| raises | 1/81 | 11/81 | 9/81 |
| base | 6/131 | 27/131 | 34/131 |
| hop_call | 0/6 | 0/6 | 0/6 |
| hop_default | 0/1 | 0/1 | 0/1 |
| hop_const | 1/13 | 0/13 | 1/13 |

## llama-1b-pre, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 9/364 (2%) | 190 | 1/8 | 16 | 11,776 | 1.6 | 3,272 | 95 |
| cortex+fade | 77/364 (21%) | 43 | 3/8 | 53 | 8,225 | 0.5 | 5,539 | 19 |
| cortexf+fade | 68/364 (19%) | 94 | 1/8 | 32 | 9,342 | 0.8 | 4,765 | 25 |

- cortex+fade minus paged+fade: +18.7% correct (95% CI +14.0% to +23.4%, paired over 364 questions)
- cortexf+fade minus cortex+fade: -2.5% correct (95% CI -7.1% to +2.2%, paired over 364 questions)
- cortexf+fade minus paged+fade: +16.2% correct (95% CI +12.1% to +20.6%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 2/60 | 15/60 | 18/60 |
| autodoc | 4/60 | 3/60 | 8/60 |
| extensions | 0/10 | 0/10 | 0/10 |
| featext | 1/36 | 3/36 | 1/36 |
| modeling | 0/60 | 34/60 | 23/60 |
| mpl | 2/60 | 9/60 | 6/60 |
| plot | 0/34 | 6/34 | 2/34 |
| printing | 0/44 | 7/44 | 10/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 1/59 | 15/59 | 8/59 |
| const | 1/73 | 16/73 | 16/73 |
| raises | 1/81 | 12/81 | 8/81 |
| base | 5/131 | 32/131 | 35/131 |
| hop_call | 0/6 | 0/6 | 0/6 |
| hop_default | 0/1 | 0/1 | 0/1 |
| hop_const | 1/13 | 2/13 | 1/13 |

## llama-1b-prerec, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 7/364 (2%) | 202 | 2/8 | 13 | 10,214 | 1.4 | 2,988 | 99 |
| cortex+fade | 61/364 (17%) | 31 | 4/8 | 49 | 8,102 | 0.4 | 5,790 | 19 |
| cortexf+fade | 75/364 (21%) | 85 | 3/8 | 36 | 7,589 | 0.5 | 4,846 | 25 |

- cortex+fade minus paged+fade: +14.8% correct (95% CI +11.0% to +19.0%, paired over 364 questions)
- cortexf+fade minus cortex+fade: +3.8% correct (95% CI -1.1% to +9.1%, paired over 364 questions)
- cortexf+fade minus paged+fade: +18.7% correct (95% CI +14.3% to +23.1%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 3/60 | 10/60 | 8/60 |
| autodoc | 3/60 | 2/60 | 12/60 |
| extensions | 0/10 | 10/10 | 7/10 |
| featext | 0/36 | 4/36 | 5/36 |
| modeling | 0/60 | 10/60 | 30/60 |
| mpl | 1/60 | 0/60 | 3/60 |
| plot | 0/34 | 13/34 | 2/34 |
| printing | 0/44 | 12/44 | 8/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 0/59 | 5/59 | 16/59 |
| const | 3/73 | 18/73 | 16/73 |
| raises | 0/81 | 13/81 | 8/81 |
| base | 2/131 | 25/131 | 34/131 |
| hop_call | 0/6 | 0/6 | 0/6 |
| hop_default | 0/1 | 0/1 | 0/1 |
| hop_const | 2/13 | 0/13 | 1/13 |

## llama-3b, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 29/364 (8%) | 138 | 2/8 | 26 | 12,670 | 1.6 | 4,288 | 2 |
| cortex+fade | 144/364 (40%) | 57 | 2/8 | 45 | 13,127 | 1.4 | 5,605 | 12 |
| cortexf+fade | 147/364 (40%) | 57 | 3/8 | 45 | 13,324 | 1.4 | 5,694 | 9 |

- cortex+fade minus paged+fade: +31.6% correct (95% CI +26.1% to +37.1%, paired over 364 questions)
- cortexf+fade minus cortex+fade: +0.8% correct (95% CI -4.7% to +6.6%, paired over 364 questions)
- cortexf+fade minus paged+fade: +32.4% correct (95% CI +26.9% to +38.2%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 7/60 | 33/60 | 23/60 |
| autodoc | 3/60 | 19/60 | 20/60 |
| extensions | 1/10 | 9/10 | 9/10 |
| featext | 0/36 | 18/36 | 19/36 |
| modeling | 6/60 | 40/60 | 15/60 |
| mpl | 0/60 | 7/60 | 35/60 |
| plot | 8/34 | 0/34 | 0/34 |
| printing | 4/44 | 18/44 | 26/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 5/59 | 32/59 | 31/59 |
| const | 8/73 | 40/73 | 41/73 |
| raises | 3/81 | 10/81 | 12/81 |
| base | 12/131 | 62/131 | 57/131 |
| hop_call | 0/6 | 0/6 | 1/6 |
| hop_default | 0/1 | 0/1 | 1/1 |
| hop_const | 1/13 | 0/13 | 4/13 |

## llama-3b, context 16,384 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 51/364 (14%) | 22 | 5/8 | 49 | 31,807 | 2.0 | 11,452 | 21 |
| cortex+fade | 195/364 (54%) | 0 | 8/8 | never | 22,277 | 1.2 | 11,398 | 17 |
| cortexf+fade | 237/364 (65%) | 0 | 8/8 | never | 20,629 | 1.1 | 11,177 | 15 |

- cortex+fade minus paged+fade: +39.6% correct (95% CI +33.5% to +45.3%, paired over 364 questions)
- cortexf+fade minus cortex+fade: +11.5% correct (95% CI +6.6% to +16.2%, paired over 364 questions)
- cortexf+fade minus paged+fade: +51.1% correct (95% CI +45.9% to +56.3%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 28/60 | 48/60 | 50/60 |
| autodoc | 0/60 | 19/60 | 39/60 |
| extensions | 1/10 | 10/10 | 10/10 |
| featext | 3/36 | 19/36 | 20/36 |
| modeling | 10/60 | 40/60 | 31/60 |
| mpl | 3/60 | 20/60 | 39/60 |
| plot | 5/34 | 24/34 | 26/34 |
| printing | 1/44 | 15/44 | 22/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 11/59 | 45/59 | 51/59 |
| const | 17/73 | 53/73 | 59/73 |
| raises | 1/81 | 26/81 | 29/81 |
| base | 20/131 | 68/131 | 94/131 |
| hop_call | 0/6 | 2/6 | 2/6 |
| hop_default | 0/1 | 1/1 | 1/1 |
| hop_const | 2/13 | 0/13 | 1/13 |

## llama-3b-full, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged+fade | 66/364 (18%) | 4 | 8/8 | 39 | 16,068 | 2.3 | 5,685 | 20 |
| cortex+fade | 214/364 (59%) | 0 | 8/8 | never | 6,502 | 0.2 | 5,573 | 10 |
| cortexf+fade | 185/364 (51%) | 0 | 8/8 | never | 8,266 | 0.5 | 5,691 | 22 |

- cortex+fade minus paged+fade: +40.7% correct (95% CI +35.2% to +46.2%, paired over 364 questions)
- cortexf+fade minus cortex+fade: -8.0% correct (95% CI -13.5% to -2.5%, paired over 364 questions)
- cortexf+fade minus paged+fade: +32.7% correct (95% CI +27.2% to +37.9%, paired over 364 questions)

| subtree | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| admin | 11/60 | 57/60 | 24/60 |
| autodoc | 9/60 | 21/60 | 29/60 |
| extensions | 0/10 | 9/10 | 8/10 |
| featext | 5/36 | 30/36 | 12/36 |
| modeling | 24/60 | 39/60 | 41/60 |
| mpl | 5/60 | 21/60 | 26/60 |
| plot | 2/34 | 16/34 | 25/34 |
| printing | 10/44 | 21/44 | 20/44 |

| question kind | paged+fade | cortex+fade | cortexf+fade |
|---|---|---|---|
| default | 11/59 | 50/59 | 35/59 |
| const | 22/73 | 67/73 | 48/73 |
| raises | 2/81 | 21/81 | 29/81 |
| base | 30/131 | 72/131 | 70/131 |
| hop_call | 0/6 | 1/6 | 1/6 |
| hop_default | 1/1 | 1/1 | 0/1 |
| hop_const | 0/13 | 2/13 | 2/13 |

