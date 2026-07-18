# LiveCodeBench/SWE-bench Lite Smoke Split Generation

Status: success

Method: benchmark split generation
Owner: 潘阔
Run ID: create_lcb_swebench_smoke_splits_20260718_190255

Purpose:

- Create deterministic smoke guide/eval splits for LiveCodeBench and SWE-bench Lite.
- Complete the four-benchmark guide/eval split surface required by the Stage 1 execution book.
- Avoid running model inference or benchmark evaluation.

Key output:

- LiveCodeBench loaded problems: 400
- LiveCodeBench guide: 4 tasks
- LiveCodeBench eval: 4 tasks
- LiveCodeBench guide SHA256: d610fd76a20a0a8dca49a8e1e1855044ccb7db6f27e7f43aa5c36f31770179c7
- LiveCodeBench eval SHA256: 8cc3b8d6aa98ecd8dd574d0a94c0d41166f9ea9553d9492b45ae1f79565932b1
- SWE-bench Lite guide: 4 tasks
- SWE-bench Lite eval: 4 tasks
- SWE-bench Lite guide SHA256: 6375db5faf6a905d919590a7c2c76d9dfa013a0eb998569726e6022ade27e5f5
- SWE-bench Lite eval SHA256: a743f1f9e1f4c7784446c0cf9fd86c7a8a9033e5fe2750a69c469cf9119009f4
- Duration: 1813.722 seconds

These smoke splits are for pipeline checks and benchmark-guided pruning input plumbing. They are not formal benchmark result splits.
