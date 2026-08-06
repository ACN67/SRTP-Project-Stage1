#!/usr/bin/env bash
# Half-set align eval for pan Mag/Wanda vs Dense (HE82 / MBPP224 / LCB200).
set -euo pipefail
cd "$HOME/projects/srtp-code-llm-pruning"
source scripts/setup/env.sh 2>/dev/null || true

WIN=/mnt/c/Users/Xile/.vscode/.code/SRTP-Project-Stage1
PY="${PY:-.venv-common/bin/python}"
LCB_PY="${LCB_PY:-.venv-livecodebench/bin/python}"
FORMAL_ROOT="${FORMAL_ROOT:-results/raw/pan_formal_20260724_203248}"
TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
ROOT_OUT="${ROOT_OUT:-results/raw/pan_half_align_${TS}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

mkdir -p "$ROOT_OUT"
echo "ROOT_OUT=$ROOT_OUT" | tee "$ROOT_OUT/run.log"

# Sync half splits + scripts from Windows clone
for split in humaneval_half mbpp_evalplus_half livecodebench_half; do
  mkdir -p "data/splits/$split"
  cp -f "$WIN/data/splits/$split/"* "data/splits/$split/" 2>/dev/null || true
done
mkdir -p scripts/evaluate scripts/eval scripts/setup
cp -f "$WIN/scripts/evaluate/"*.py scripts/evaluate/
cp -f "$WIN/scripts/eval/score_humaneval_smoke.py" scripts/eval/
cp -f "$WIN/scripts/eval/score_mbpp_smoke.py" scripts/eval/
cp -f "$WIN/scripts/eval/score_livecodebench_split.py" scripts/eval/ 2>/dev/null || true
# strip CR
for f in scripts/evaluate/*.py scripts/eval/score_*.py; do
  [[ -f "$f" ]] || continue
  tr -d '\r' < "$f" > "$f.tmp" && mv "$f.tmp" "$f"
done

run_one() {
  local tag="$1"
  local model="$2"
  local out="$ROOT_OUT/$tag"
  mkdir -p "$out"

  # HumanEval half
  if [[ ! -f "$out/humaneval/score_summary.json" ]]; then
    echo "=== [$tag] generate HE half ===" | tee -a "$ROOT_OUT/run.log"
    mkdir -p "$out/humaneval"
    "$PY" scripts/evaluate/generate_full_benchmark.py \
      --model "$model" \
      --benchmark humaneval \
      --split data/splits/humaneval_half/eval.jsonl \
      --output-dir "$out/humaneval" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --device-map cuda:0
    "$PY" scripts/evaluate/predictions_to_samples.py \
      --predictions "$out/humaneval/predictions.jsonl" \
      --output "$out/humaneval/samples.jsonl" \
      --benchmark humaneval
    echo "=== [$tag] score HE half ===" | tee -a "$ROOT_OUT/run.log"
    "$PY" scripts/eval/score_humaneval_smoke.py \
      --split data/splits/humaneval_half/eval.jsonl \
      --samples "$out/humaneval/samples.jsonl" \
      --out-dir "$out/humaneval" \
      --base-only | tee "$out/humaneval/score_stdout.log"
  else
    echo "skip HE $tag" | tee -a "$ROOT_OUT/run.log"
  fi

  # MBPP half
  if [[ ! -f "$out/mbpp/score_summary.json" ]]; then
    echo "=== [$tag] generate MBPP half ===" | tee -a "$ROOT_OUT/run.log"
    mkdir -p "$out/mbpp"
    "$PY" scripts/evaluate/generate_full_benchmark.py \
      --model "$model" \
      --benchmark mbpp \
      --split data/splits/mbpp_evalplus_half/eval.jsonl \
      --output-dir "$out/mbpp" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --device-map cuda:0
    "$PY" scripts/evaluate/predictions_to_samples.py \
      --predictions "$out/mbpp/predictions.jsonl" \
      --output "$out/mbpp/samples.jsonl" \
      --benchmark mbpp
    echo "=== [$tag] score MBPP half ===" | tee -a "$ROOT_OUT/run.log"
    "$PY" scripts/eval/score_mbpp_smoke.py \
      --split data/splits/mbpp_evalplus_half/eval.jsonl \
      --samples "$out/mbpp/samples.jsonl" \
      --out-dir "$out/mbpp" \
      --base-only | tee "$out/mbpp/score_stdout.log"
  else
    echo "skip MBPP $tag" | tee -a "$ROOT_OUT/run.log"
  fi

  # LiveCodeBench half
  if [[ ! -f "$out/livecodebench/score_summary.json" ]]; then
    echo "=== [$tag] generate LCB half ===" | tee -a "$ROOT_OUT/run.log"
    mkdir -p "$out/livecodebench"
    "$PY" scripts/evaluate/generate_code_split.py \
      --model "$model" \
      --benchmark livecodebench \
      --split data/splits/livecodebench_half/eval.jsonl \
      --output-dir "$out/livecodebench" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --device-map cuda:0
    # score script expects generations.jsonl
    cp -f "$out/livecodebench/predictions.jsonl" "$out/livecodebench/generations.jsonl"
    echo "=== [$tag] score LCB half ===" | tee -a "$ROOT_OUT/run.log"
    "$LCB_PY" scripts/eval/score_livecodebench_split.py \
      --split data/splits/livecodebench_half/eval.jsonl \
      --generations "$out/livecodebench/generations.jsonl" \
      --out-dir "$out/livecodebench" \
      --lcb-release release_v1 | tee "$out/livecodebench/score_stdout.log"
  else
    echo "skip LCB $tag" | tee -a "$ROOT_OUT/run.log"
  fi

  echo "DONE_TAG=$tag" | tee -a "$ROOT_OUT/run.log"
}

MAG_MODEL="$FORMAL_ROOT/qwen15b_magnitude_he_s0p10/pruned/pruned_model"
WANDA_MODEL="$FORMAL_ROOT/qwen15b_wanda_he_s0p10/pruned/pruned_model"
if [[ ! -f "$MAG_MODEL/model.safetensors" ]]; then
  echo "Missing Mag pruned model: $MAG_MODEL" >&2
  exit 1
fi
if [[ ! -f "$WANDA_MODEL/model.safetensors" ]]; then
  echo "Missing Wanda pruned model: $WANDA_MODEL" >&2
  exit 1
fi

run_one dense_baseline "Qwen/Qwen2.5-Coder-1.5B-Instruct"
run_one magnitude_he_s0p10 "$MAG_MODEL"
run_one wanda_he_s0p10 "$WANDA_MODEL"

# Aggregate CSV
"$PY" - <<'PY'
import csv, json
from pathlib import Path
root = Path("ROOT_OUT_PLACEHOLDER")
rows = []
for tag_dir in sorted(root.glob("*")):
    if not tag_dir.is_dir() or tag_dir.name.startswith("."):
        continue
    tag = tag_dir.name
    method = "Dense" if tag.startswith("dense") else ("Magnitude" if tag.startswith("magnitude") else "Wanda")
    spars = "0.0" if method == "Dense" else "0.10"
    for bench, key in [("humaneval","he"), ("mbpp","mbpp"), ("livecodebench","lcb")]:
        summary = tag_dir / bench / "score_summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if bench == "livecodebench":
            value = data.get("pass_count")
            total = data.get("task_count") or data.get("n") or 200
            rate = data.get("pass_rate")
            metric = f"{value}/{total}" if value is not None else ""
            rows.append({"method": method, "sparsity": spars, "benchmark": bench, "metric": "pass_count", "value": metric, "pass_rate": rate or "", "run_id": str(tag_dir / bench)})
        else:
            pc = data.get("base_pass_count")
            tc = data.get("task_count")
            rate = data.get("base_pass_rate")
            rows.append({"method": method, "sparsity": spars, "benchmark": bench, "metric": "pass_count", "value": f"{pc}/{tc}", "pass_rate": rate, "run_id": str(tag_dir / bench)})
out = Path("results/tables/pan_half_comparison.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method","sparsity","benchmark","metric","value","pass_rate","run_id"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote", out, "rows", len(rows))
(root / "HALF_ALIGN_DONE").write_text("ok\n", encoding="utf-8")
print("HALF_ALIGN_DONE")
PY

# fix placeholder
sed -i "s|ROOT_OUT_PLACEHOLDER|$ROOT_OUT|g" /dev/null 2>/dev/null || true
"$PY" - "$ROOT_OUT" <<'PY'
import csv, json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for tag_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    tag = tag_dir.name
    method = "Dense" if tag.startswith("dense") else ("Magnitude" if tag.startswith("magnitude") else "Wanda")
    spars = "0.0" if method == "Dense" else "0.10"
    for bench in ("humaneval", "mbpp", "livecodebench"):
        summary = tag_dir / bench / "score_summary.json"
        if not summary.exists():
            continue
        data = json.loads(summary.read_text(encoding="utf-8"))
        if bench == "livecodebench":
            value = data.get("pass_count")
            total = data.get("task_count") or 200
            rate = data.get("pass_rate")
            metric_val = f"{value}/{total}" if value is not None else ""
            rows.append({"method": method, "sparsity": spars, "benchmark": bench, "metric": "pass_count", "value": metric_val, "pass_rate": rate or "", "run_id": str(tag_dir / bench)})
        else:
            pc = data.get("base_pass_count")
            tc = data.get("task_count")
            rate = data.get("base_pass_rate")
            rows.append({"method": method, "sparsity": spars, "benchmark": bench, "metric": "pass_count", "value": f"{pc}/{tc}", "pass_rate": rate, "run_id": str(tag_dir / bench)})
out = Path("results/tables/pan_half_comparison.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method","sparsity","benchmark","metric","value","pass_rate","run_id"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote", out, "rows", len(rows))
(root / "HALF_ALIGN_DONE").write_text("ok\n", encoding="utf-8")
print("HALF_ALIGN_DONE", root)
PY

echo "$ROOT_OUT" > "$ROOT_OUT/ROOT_OUT.txt"
cp -f results/tables/pan_half_comparison.csv "$WIN/results/tables/" 2>/dev/null || true
