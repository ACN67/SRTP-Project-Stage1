# Pruning Method Comparison
Local commit: `9117e9c340e1c56bc5bd6560c2418688da430853`

| Method | HumanEval | MBPP | LiveCodeBench | Actual keep | First break |
|---|---:|---:|---:|---:|---|
| flabpruner | 0.0 | 0.004464285714285714 | 0.0 | 0.8939371828221396 | not determined; S1-S5 missing |
| llmpruner | 0.024390243902439025 | 0.004464285714285714 | 0.0 | 0.8202547406077543 | not determined; S1-S5 missing |
| slicegpt | 0.0 | 0.008928571428571428 | 0.0 | 0.677403188718526 | not determined; S1-S5 missing |

Dense Qwen1.5B baseline: HumanEval 0.2195, MBPP 0.5134, LiveCodeBench 0.14. Existing final runs show collapse, but S1-S5 smoke checkpoints are missing, so algorithmic conclusions remain provisional.
