# Held-out session stress test

`bench/heldout.py`: 320 machine-written questions over 8 package subtrees of SWE-bench Lite repositories (none used to build or tune CortexNET), one conversation per subtree and arm (up to 40 questions in a row).

## llama-3.1-8b, context 8,192 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged | 4/320 (1%) | 313 | 0/8 | 2 | 454 | 0.1 | 264 | 10 |
| paged+fade | 113/320 (35%) | 42 | 4/8 | 39 | 16,103 | 1.9 | 5,942 | 31 |
| cortex | 30/320 (9%) | 249 | 1/8 | 6 | 1,240 | 0.1 | 864 | 6 |
| cortexna+fade | 174/320 (54%) | 15 | 7/8 | 41 | 10,040 | 1.0 | 5,219 | 13 |
| cortex+fade | 212/320 (66%) | 0 | 8/8 | never | 10,505 | 1.0 | 5,456 | 8 |

- cortex minus paged: +8.1% correct (95% CI +5.3% to +11.6%, paired over 320 questions)
- cortex+fade minus paged+fade: +30.9% correct (95% CI +23.8% to +38.1%, paired over 320 questions)
- cortex+fade minus cortex: +56.9% correct (95% CI +51.6% to +62.2%, paired over 320 questions)
- paged+fade minus paged: +34.1% correct (95% CI +28.7% to +39.4%, paired over 320 questions)
- cortex+fade minus cortexna+fade: +11.9% correct (95% CI +6.9% to +16.9%, paired over 320 questions)

| subtree | paged | paged+fade | cortex | cortexna+fade | cortex+fade |
|---|---|---|---|---|---|
| axes | 0/40 | 11/40 | 0/40 | 0/40 | 0/40 |
| backends | 0/40 | 26/40 | 2/40 | 23/40 | 27/40 |
| fits | 1/40 | 14/40 | 3/40 | 15/40 | 12/40 |
| flask | 1/40 | 20/40 | 6/40 | 28/40 | 36/40 |
| forms | 0/40 | 10/40 | 3/40 | 26/40 | 33/40 |
| functions | 1/40 | 3/40 | 4/40 | 28/40 | 35/40 |
| preprocessing | 1/40 | 16/40 | 7/40 | 24/40 | 35/40 |
| sphinxutil | 0/40 | 13/40 | 5/40 | 30/40 | 34/40 |

| question kind | paged | paged+fade | cortex | cortexna+fade | cortex+fade |
|---|---|---|---|---|---|
| default | 3/99 | 39/99 | 11/99 | 61/99 | 58/99 |
| const | 0/31 | 15/31 | 0/31 | 23/31 | 23/31 |
| raises | 0/86 | 17/86 | 7/86 | 14/86 | 48/86 |
| base | 1/104 | 42/104 | 12/104 | 76/104 | 83/104 |

## llama-3.1-8b, context 16,384 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged | 4/320 (1%) | 313 | 0/8 | 2 | 920 | 0.1 | 535 | 12 |
| paged+fade | 138/320 (43%) | 0 | 8/8 | never | 27,601 | 1.7 | 11,902 | 35 |
| cortex | 48/320 (15%) | 224 | 1/8 | 9 | 3,622 | 0.2 | 2,155 | 10 |
| cortexna+fade | 186/320 (58%) | 0 | 8/8 | never | 17,814 | 1.0 | 9,526 | 7 |
| cortex+fade | 246/320 (77%) | 0 | 8/8 | never | 17,331 | 0.9 | 9,775 | 9 |

- cortex minus paged: +13.8% correct (95% CI +10.0% to +17.8%, paired over 320 questions)
- cortex+fade minus paged+fade: +33.8% correct (95% CI +27.5% to +39.7%, paired over 320 questions)
- cortex+fade minus cortex: +61.9% correct (95% CI +56.6% to +67.2%, paired over 320 questions)
- paged+fade minus paged: +41.9% correct (95% CI +36.6% to +47.5%, paired over 320 questions)
- cortex+fade minus cortexna+fade: +18.8% correct (95% CI +14.1% to +23.4%, paired over 320 questions)

| subtree | paged | paged+fade | cortex | cortexna+fade | cortex+fade |
|---|---|---|---|---|---|
| axes | 0/40 | 9/40 | 0/40 | 0/40 | 0/40 |
| backends | 0/40 | 17/40 | 3/40 | 25/40 | 29/40 |
| fits | 1/40 | 17/40 | 16/40 | 15/40 | 35/40 |
| flask | 1/40 | 24/40 | 10/40 | 31/40 | 39/40 |
| forms | 0/40 | 16/40 | 1/40 | 27/40 | 39/40 |
| functions | 1/40 | 24/40 | 5/40 | 30/40 | 36/40 |
| preprocessing | 1/40 | 19/40 | 8/40 | 26/40 | 36/40 |
| sphinxutil | 0/40 | 12/40 | 5/40 | 32/40 | 32/40 |

| question kind | paged | paged+fade | cortex | cortexna+fade | cortex+fade |
|---|---|---|---|---|---|
| default | 3/99 | 49/99 | 16/99 | 66/99 | 76/99 |
| const | 0/31 | 14/31 | 3/31 | 24/31 | 28/31 |
| raises | 0/86 | 17/86 | 15/86 | 15/86 | 50/86 |
| base | 1/104 | 58/104 | 14/104 | 81/104 | 92/104 |

## llama-3.1-8b, context 32,768 tokens (repeats [0])

| arm | correct | questions lost to an overflow | sessions alive at the end | first overflow (median question) | input tokens per question | tool calls per question | mean peak context | tool-call repairs |
|---|---|---|---|---|---|---|---|---|
| paged | 5/320 (2%) | 309 | 0/8 | 2 | 2,056 | 0.1 | 1,172 | 14 |
| paged+fade | 176/320 (55%) | 0 | 8/8 | never | 43,777 | 1.3 | 22,727 | 30 |
| cortex | 73/320 (23%) | 175 | 2/8 | 14 | 10,139 | 0.4 | 5,314 | 9 |
| cortexna+fade | 194/320 (61%) | 0 | 8/8 | never | 31,690 | 1.0 | 16,801 | 9 |
| cortex+fade | 243/320 (76%) | 0 | 8/8 | never | 35,412 | 1.1 | 17,732 | 9 |

- cortex minus paged: +21.2% correct (95% CI +16.9% to +25.9%, paired over 320 questions)
- cortex+fade minus paged+fade: +20.9% correct (95% CI +14.7% to +26.9%, paired over 320 questions)
- cortex+fade minus cortex: +53.1% correct (95% CI +47.8% to +58.8%, paired over 320 questions)
- paged+fade minus paged: +53.4% correct (95% CI +47.8% to +59.1%, paired over 320 questions)
- cortex+fade minus cortexna+fade: +15.3% correct (95% CI +10.6% to +19.7%, paired over 320 questions)

| subtree | paged | paged+fade | cortex | cortexna+fade | cortex+fade |
|---|---|---|---|---|---|
| axes | 0/40 | 9/40 | 0/40 | 0/40 | 0/40 |
| backends | 0/40 | 22/40 | 11/40 | 27/40 | 37/40 |
| fits | 1/40 | 21/40 | 11/40 | 11/40 | 22/40 |
| flask | 0/40 | 32/40 | 17/40 | 32/40 | 39/40 |
| forms | 1/40 | 22/40 | 6/40 | 36/40 | 34/40 |
| functions | 1/40 | 23/40 | 6/40 | 29/40 | 35/40 |
| preprocessing | 1/40 | 18/40 | 10/40 | 27/40 | 37/40 |
| sphinxutil | 1/40 | 29/40 | 12/40 | 32/40 | 39/40 |

| question kind | paged | paged+fade | cortex | cortexna+fade | cortex+fade |
|---|---|---|---|---|---|
| default | 3/99 | 66/99 | 21/99 | 65/99 | 74/99 |
| const | 0/31 | 19/31 | 6/31 | 24/31 | 23/31 |
| raises | 0/86 | 18/86 | 20/86 | 17/86 | 56/86 |
| base | 2/104 | 73/104 | 26/104 | 88/104 | 90/104 |

