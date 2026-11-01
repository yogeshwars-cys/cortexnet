# Held-out session stress test

`bench/heldout.py`: 320 machine-written questions over 8 package subtrees of SWE-bench Lite repositories (none used to build or tune CortexNET), one conversation per subtree and arm (up to 40 questions in a row).

## llama-3.1-8b, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| baseline | 1/240 (0%) | 237 | 0/6 | 1 | 229 | 0.1 | 135 | 4 |
| paged | 1/280 (0%) | 237 | 1/7 | 2 | 610 | 0.1 | 498 | 7 |
| paged+fade | 23/40 (57%) | 0 | 1/1 | never | 15,356 | 1.6 | 6,784 | 0 |
| cortex | 16/200 (8%) | 176 | 0/5 | 7 | 1,113 | 0.1 | 705 | 3 |
| cortex+fade | 78/120 (65%) | 2 | 3/3 | 41 | 10,688 | 1.0 | 5,880 | 5 |

- cortex minus paged: +7.5% correct (95% CI +4.0% to +11.0%, paired over 200 questions)
- cortex+fade minus baseline: +66.2% correct (95% CI +55.0% to +76.2%, paired over 80 questions)
- cortex+fade minus cortex: +61.3% correct (95% CI +51.2% to +71.3%, paired over 80 questions)
- paged+fade minus paged: +57.5% correct (95% CI +42.5% to +72.5%, paired over 40 questions)

| subtree | baseline | paged | paged+fade | cortex | cortex+fade |
|---|---|---|---|---|---|
| django | 0/40 | 0/40 | - | 4/40 | - |
| pylint | 0/40 | 0/40 | 23/40 | 6/40 | - |
| pytest | - | - | - | - | 24/40 |
| seaborn | - | 0/40 | - | - | - |
| sklearn | 0/40 | 1/40 | - | 1/40 | - |
| sphinx | 0/40 | 0/40 | - | - | - |
| sympy | 1/40 | 0/40 | - | 3/40 | 28/40 |
| xarray | 0/40 | 0/40 | - | 2/40 | 26/40 |

| question kind | baseline | paged | paged+fade | cortex | cortex+fade |
|---|---|---|---|---|---|
| default | 0/50 | 1/64 | 0/2 | 2/50 | 33/40 |
| const | 0/48 | 0/56 | 10/15 | 6/38 | 10/16 |
| raises | 1/46 | 0/56 | 0/3 | 2/46 | 9/33 |
| base | 0/96 | 0/104 | 13/20 | 6/66 | 26/31 |

## llama-3.1-8b, context 16,384 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| baseline | 2/320 (1%) | 315 | 0/8 | 1 | 432 | 0.1 | 252 | 8 |
| paged | 4/320 (1%) | 273 | 1/8 | 3 | 1,222 | 0.1 | 861 | 9 |
| paged+fade | 138/320 (43%) | 0 | 8/8 | never | 20,326 | 1.1 | 10,406 | 16 |
| cortex | 49/320 (15%) | 255 | 0/8 | 9 | 3,800 | 0.3 | 2,200 | 3 |
| cortex+fade | 191/280 (68%) | 0 | 7/7 | never | 19,084 | 1.0 | 10,639 | 2 |

- cortex minus paged: +14.1% correct (95% CI +10.3% to +18.1%, paired over 320 questions)
- cortex+fade minus paged+fade: +26.4% correct (95% CI +18.9% to +33.6%, paired over 280 questions)
- cortex+fade minus baseline: +67.5% correct (95% CI +61.4% to +72.9%, paired over 280 questions)
- cortex+fade minus cortex: +52.5% correct (95% CI +46.4% to +58.6%, paired over 280 questions)
- paged+fade minus paged: +41.9% correct (95% CI +36.2% to +47.2%, paired over 320 questions)

| subtree | baseline | paged | paged+fade | cortex | cortex+fade |
|---|---|---|---|---|---|
| django | 0/40 | 0/40 | 15/40 | 4/40 | 19/40 |
| pylint | 1/40 | 2/40 | 27/40 | 12/40 | 32/40 |
| pytest | 0/40 | 0/40 | 21/40 | 5/40 | - |
| seaborn | 0/40 | 0/40 | 0/40 | 10/40 | 26/40 |
| sklearn | 0/40 | 1/40 | 20/40 | 1/40 | 25/40 |
| sphinx | 0/40 | 0/40 | 12/40 | 6/40 | 34/40 |
| sympy | 1/40 | 1/40 | 18/40 | 5/40 | 26/40 |
| xarray | 0/40 | 0/40 | 25/40 | 6/40 | 29/40 |

| question kind | baseline | paged | paged+fade | cortex | cortex+fade |
|---|---|---|---|---|---|
| default | 0/78 | 1/78 | 26/78 | 13/78 | 58/64 |
| const | 1/64 | 2/64 | 39/64 | 8/64 | 36/56 |
| raises | 1/66 | 1/66 | 19/66 | 6/66 | 11/56 |
| base | 0/112 | 0/112 | 54/112 | 22/112 | 86/104 |

## llama-3.1-8b, context 32,768 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| baseline | 17/320 (5%) | 275 | 1/8 | 1 | 11,950 | 0.6 | 2,747 | 8 |
| paged | 3/320 (1%) | 269 | 1/8 | 3 | 2,461 | 0.1 | 1,495 | 7 |
| paged+fade | 159/320 (50%) | 1 | 8/8 | 41 | 44,114 | 1.3 | 23,101 | 15 |
| cortex | 71/320 (22%) | 224 | 0/8 | 16 | 10,376 | 0.3 | 5,923 | 4 |
| cortex+fade | 203/320 (63%) | 0 | 8/8 | never | 33,455 | 0.9 | 18,274 | 5 |

- cortex minus paged: +21.2% correct (95% CI +16.9% to +25.9%, paired over 320 questions)
- cortex+fade minus paged+fade: +13.8% correct (95% CI +6.9% to +20.3%, paired over 320 questions)
- cortex+fade minus baseline: +58.1% correct (95% CI +52.2% to +63.7%, paired over 320 questions)
- cortex+fade minus cortex: +41.2% correct (95% CI +35.0% to +47.5%, paired over 320 questions)
- paged+fade minus paged: +48.8% correct (95% CI +43.4% to +54.4%, paired over 320 questions)

| subtree | baseline | paged | paged+fade | cortex | cortex+fade |
|---|---|---|---|---|---|
| django | 0/40 | 0/40 | 17/40 | 6/40 | 20/40 |
| pylint | 15/40 | 2/40 | 24/40 | 13/40 | 31/40 |
| pytest | 0/40 | 0/40 | 32/40 | 10/40 | 30/40 |
| seaborn | 1/40 | 0/40 | 19/40 | 11/40 | 1/40 |
| sklearn | 0/40 | 1/40 | 25/40 | 3/40 | 26/40 |
| sphinx | 0/40 | 0/40 | 11/40 | 5/40 | 36/40 |
| sympy | 1/40 | 0/40 | 8/40 | 9/40 | 29/40 |
| xarray | 0/40 | 0/40 | 23/40 | 14/40 | 30/40 |

| question kind | baseline | paged | paged+fade | cortex | cortex+fade |
|---|---|---|---|---|---|
| default | 0/78 | 1/78 | 49/78 | 26/78 | 60/78 |
| const | 15/64 | 2/64 | 42/64 | 12/64 | 40/64 |
| raises | 1/66 | 0/66 | 16/66 | 6/66 | 12/66 |
| base | 1/112 | 0/112 | 52/112 | 27/112 | 91/112 |

