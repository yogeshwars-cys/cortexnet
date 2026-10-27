# Replay of logged edits through the outline-first reader

Cases: 99 edits that followed a full Read of the same file (anchor found once). Costs in tokens (chars/4). Upper bound: assumes the model asks for the right symbol after a miss.

## Query: no query

- all cases: whole-file reads 581,279 tokens -> outline-first 110,920 (19%)
- files large enough for outline mode: 91 of 99; 578,831 -> 108,166 tokens (19%); median per-case ratio 0.12
- edited lines already shown by the read (no range call needed): 0% of outline-mode cases

## Query: prompt + agent text

- all cases: whole-file reads 581,279 tokens -> outline-first 123,832 (21%)
- files large enough for outline mode: 91 of 99; 578,831 -> 121,078 tokens (21%); median per-case ratio 0.15
- edited lines already shown by the read (no range call needed): 7% of outline-mode cases

File types: .md 44, .py 18, .dart 10, .html 10, .cpp 5, .tsx 4, .txt 3, .mjs 2, .js 2, .c 1
