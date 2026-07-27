#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/eval/run_official_eval.sh --benchmark humaneval|mbpp_evalplus|livecodebench --model MODEL --split SPLIT --out-dir OUT_DIR [options]

Official benchmark policy:
  humaneval       EvalPlus HumanEval prompts + EvalPlus correctness checker.
  mbpp_evalplus   EvalPlus MBPP prompts + EvalPlus correctness checker.
  livecodebench   LiveCodeBench official prompt formatter, extract_code, and codegen_metrics.

Common options:
  --adapter PATH
  --max-new-tokens N
  --dtype fp16|bf16|fp32
  --device cuda:0|cpu
  --local-files-only
  --load-mode direct|device_map
  --device-map VALUE
  --max-memory-json JSON
  --offload-folder PATH
  --load-in-8bit
  --load-in-4bit
  --llm-int8-enable-fp32-cpu-offload

LiveCodeBench options:
  --lcb-release release_v1
  --lcb-config release_latest
  --lcb-lm-style CodeQwenInstruct
  --timeout N
  --num-process-evaluate N
USAGE
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

BENCHMARK=""
MODEL=""
SPLIT=""
OUT_DIR=""
ADAPTER=""
MAX_NEW_TOKENS=""
DTYPE="fp16"
DEVICE="cuda:0"
LOCAL_FILES_ONLY=0
LOAD_MODE="direct"
DEVICE_MAP="auto"
MAX_MEMORY_JSON=""
OFFLOAD_FOLDER=""
LOAD_IN_8BIT=0
LOAD_IN_4BIT=0
INT8_FP32_CPU_OFFLOAD=0
LCB_RELEASE="release_v1"
LCB_CONFIG="release_latest"
LCB_LM_STYLE="CodeQwenInstruct"
TIMEOUT=6
NUM_PROCESS_EVALUATE=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --split) SPLIT="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --adapter) ADAPTER="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --load-mode) LOAD_MODE="$2"; shift 2 ;;
    --device-map) DEVICE_MAP="$2"; shift 2 ;;
    --max-memory-json) MAX_MEMORY_JSON="$2"; shift 2 ;;
    --offload-folder) OFFLOAD_FOLDER="$2"; shift 2 ;;
    --load-in-8bit) LOAD_IN_8BIT=1; shift ;;
    --load-in-4bit) LOAD_IN_4BIT=1; shift ;;
    --llm-int8-enable-fp32-cpu-offload) INT8_FP32_CPU_OFFLOAD=1; shift ;;
    --lcb-release) LCB_RELEASE="$2"; shift 2 ;;
    --lcb-config) LCB_CONFIG="$2"; shift 2 ;;
    --lcb-lm-style) LCB_LM_STYLE="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --num-process-evaluate) NUM_PROCESS_EVALUATE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${BENCHMARK}" || -z "${MODEL}" || -z "${SPLIT}" || -z "${OUT_DIR}" ]]; then
  usage >&2
  exit 2
fi

case "${BENCHMARK}" in
  humaneval)
    PROMPT_MODE="humaneval_official"
    SCORE_SCRIPT="scripts/eval/score_humaneval_official.py"
    DEFAULT_MAX_NEW_TOKENS=512
    ;;
  mbpp_evalplus)
    PROMPT_MODE="mbpp_evalplus_official"
    SCORE_SCRIPT="scripts/eval/score_mbpp_evalplus_official.py"
    DEFAULT_MAX_NEW_TOKENS=512
    ;;
  livecodebench)
    PROMPT_MODE="livecodebench_official"
    SCORE_SCRIPT="scripts/eval/score_livecodebench_official.py"
    DEFAULT_MAX_NEW_TOKENS=1024
    ;;
  *)
    echo "Unsupported benchmark: ${BENCHMARK}" >&2
    usage >&2
    exit 2
    ;;
esac

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-${DEFAULT_MAX_NEW_TOKENS}}"
mkdir -p "${OUT_DIR}"

GEN_ARGS=(
  python scripts/eval/generate_official_samples.py
  --model "${MODEL}"
  --split "${SPLIT}"
  --out-dir "${OUT_DIR}/generation"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --prompt-mode "${PROMPT_MODE}"
  --load-mode "${LOAD_MODE}"
)

[[ -n "${ADAPTER}" ]] && GEN_ARGS+=(--adapter "${ADAPTER}")
[[ "${LOCAL_FILES_ONLY}" -eq 1 ]] && GEN_ARGS+=(--local-files-only)
[[ "${LOAD_MODE}" == "device_map" ]] && GEN_ARGS+=(--device-map "${DEVICE_MAP}")
[[ -n "${MAX_MEMORY_JSON}" ]] && GEN_ARGS+=(--max-memory-json "${MAX_MEMORY_JSON}")
[[ -n "${OFFLOAD_FOLDER}" ]] && GEN_ARGS+=(--offload-folder "${OFFLOAD_FOLDER}")
[[ "${LOAD_IN_8BIT}" -eq 1 ]] && GEN_ARGS+=(--load-in-8bit)
[[ "${LOAD_IN_4BIT}" -eq 1 ]] && GEN_ARGS+=(--load-in-4bit)
[[ "${INT8_FP32_CPU_OFFLOAD}" -eq 1 ]] && GEN_ARGS+=(--llm-int8-enable-fp32-cpu-offload)

if [[ "${BENCHMARK}" == "livecodebench" ]]; then
  GEN_ARGS+=(
    --lcb-release "${LCB_RELEASE}"
    --lcb-config "${LCB_CONFIG}"
    --lcb-lm-style "${LCB_LM_STYLE}"
  )
fi

"${GEN_ARGS[@]}" 2>&1 | tee "${OUT_DIR}/generate.log"

if [[ "${BENCHMARK}" == "livecodebench" ]]; then
  python "${SCORE_SCRIPT}" \
    --split "${SPLIT}" \
    --generations "${OUT_DIR}/generation/generations.jsonl" \
    --out-dir "${OUT_DIR}/score" \
    --lcb-release "${LCB_RELEASE}" \
    --lcb-config "${LCB_CONFIG}" \
    --timeout "${TIMEOUT}" \
    --num-process-evaluate "${NUM_PROCESS_EVALUATE}" \
    2>&1 | tee "${OUT_DIR}/score.log"
else
  python "${SCORE_SCRIPT}" \
    --split "${SPLIT}" \
    --samples "${OUT_DIR}/generation/samples.jsonl" \
    --out-dir "${OUT_DIR}/score" \
    --base-only \
    2>&1 | tee "${OUT_DIR}/score.log"
fi
