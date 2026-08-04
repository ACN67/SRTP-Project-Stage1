# R4-Half Formal Results

## Formal protocol
R4-half uses benchmark splits under `data/benchmarks/r4_half/` and score rows indexed in `results/status/scores.csv`.

## Generation command
Regenerate with `python workflows/aggregate/build_formal_r4_table.py --write`.

## Filtering conditions
Rows must have `protocol=r4_half`, result completeness of complete or partial, non-invalid validity, no pilot status, no superseded run status, and empty `superseded_by` in the run registry.

## Baseline supersession
`qwen25c3b_r4_baseline_evalhalf_20260723_193503` and `qwen25c3b_r4_baseline_evalhalf_recheck_20260726_181426` are superseded by `qwen25c3b_official_evalhalf_20260727_135521`.

## Pilot exclusion
`pilot_keep80_official_all_20260727_174732` is retained in the full registry but excluded from this table.

## Partial definition
Partial rows keep actual task counts and do not claim full benchmark completion.

## Fallback marking
Fallback routes remain visible through `officiality` and registry notes.

## Quality gate
Quality gate failures can appear as formal evidence but should not be used as success claims.

## Not directly comparable
Auxiliary full-eval, smoke, diagnostic, and superseded rows are not directly comparable with this table.

## Current limits
The table supports reproducibility analysis and failure localization, not a final ranking of all pruning methods.
