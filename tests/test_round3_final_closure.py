
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run_cmd(args: list[str], cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def sha(path: Path) -> str:
    h = hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def rows(rel: str) -> list[dict[str, str]]:
    with (ROOT / rel).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def test_data_builders_have_dry_run_and_no_old_split_root(tmp_path: Path) -> None:
    for script in [
        'workflows/data/build_r4_half_splits.py',
        'workflows/data/build_auxiliary_full_splits.py',
        'workflows/data/build_mbpp_evalplus.py',
    ]:
        help_result = run_cmd([PY, script, '--help'])
        assert help_result.returncode == 0, help_result.stderr
        assert '--dry-run' in help_result.stdout
        assert '--output-root' in help_result.stdout
        dry_root = tmp_path / script.split('/')[-1]
        result = run_cmd([PY, script, '--dry-run', '--output-root', str(dry_root)])
        assert result.returncode == 0, result.stderr
        assert not (ROOT / 'data' / 'splits').exists()
        assert not (dry_root / 'data' / 'splits').exists()


def test_auxiliary_protocol_and_manifests_are_consistent() -> None:
    base = ROOT / 'data/benchmarks/auxiliary_full_eval'
    for dataset in ['humaneval', 'mbpp']:
        guide = read_jsonl(base / dataset / 'guide.jsonl')
        full = read_jsonl(base / dataset / 'eval.jsonl')
        heldout = read_jsonl(base / dataset / 'heldout_eval.jsonl')
        gids = {r['task_id'] for r in guide}
        fids = {r['task_id'] for r in full}
        hids = {r['task_id'] for r in heldout}
        assert gids <= fids
        assert gids.isdisjoint(hids)
        assert fids == gids | hids
    for manifest in (ROOT / 'data/benchmarks').rglob('manifest.json'):
        data = json.loads(manifest.read_text(encoding='utf-8'))
        protocol = manifest.relative_to(ROOT).parts[2]
        dataset = manifest.parent.name
        assert data.get('protocol', protocol) == protocol
        assert data.get('dataset', data.get('benchmark')) in {dataset, dataset.replace('_evalplus', ''), 'mbpp'}
        for key in ['guide', 'eval', 'heldout_eval']:
            path_key = f'{key}_path'
            hash_key = f'{key}_sha256'
            count_key = f'{key}_count'
            if path_key in data:
                path = ROOT / data[path_key]
                assert path.exists(), manifest
                assert 'data/splits' not in data[path_key]
                assert data.get(hash_key) == sha(path)
                assert data.get(count_key) == len(read_jsonl(path))
    aux_mbpp = json.loads((base / 'mbpp_evalplus/manifest.json').read_text(encoding='utf-8'))
    assert aux_mbpp['eval_path'] == 'data/benchmarks/auxiliary_full_eval/mbpp_evalplus/eval.jsonl'
    split_rows = rows('results/status/data_splits.csv')
    aux_policies = {r['role']: r['overlap_policy'] for r in split_rows if r['protocol'] == 'auxiliary_full_eval' and r['dataset'] in {'humaneval', 'mbpp'}}
    assert 'guide_subset_of_full_eval' in aux_policies.values()
    assert 'guide_disjoint_from_heldout' in aux_policies.values()


def test_registry_builders_write_check_and_are_idempotent(tmp_path: Path) -> None:
    work = tmp_path / 'repo'
    ignore = shutil.ignore_patterns('.git', 'third_party', 'results/evidence')
    shutil.copytree(ROOT, work, ignore=ignore)
    (work / 'results/evidence').mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / 'results/evidence', work / 'results/evidence', dirs_exist_ok=True)
    scripts = [
        'workflows/aggregate/build_data_split_registry.py',
        'workflows/aggregate/build_run_registry.py',
        'workflows/aggregate/build_score_registry.py',
        'workflows/aggregate/build_method_status.py',
        'workflows/aggregate/build_formal_r4_table.py',
    ]
    targets = [
        'results/status/data_splits.csv', 'results/status/runs.csv', 'results/status/scores.csv',
        'results/status/methods.csv', 'results/formal/r4_half/scores.csv'
    ]
    for rel in targets:
        (work / rel).unlink(missing_ok=True)
    for script in scripts:
        result = run_cmd([PY, script, '--write'], cwd=work)
        assert result.returncode == 0, (script, result.stderr, result.stdout)
    hashes = {rel: sha(work / rel) for rel in targets}
    for script in scripts:
        result = run_cmd([PY, script, '--check'], cwd=work)
        assert result.returncode == 0, (script, result.stderr, result.stdout)
    for script in scripts:
        result = run_cmd([PY, script, '--write'], cwd=work)
        assert result.returncode == 0
    assert hashes == {rel: sha(work / rel) for rel in targets}
    score_rows = list(csv.DictReader((work / 'results/status/scores.csv').open(encoding='utf-8-sig')))
    ids = [r['score_id'] for r in score_rows]
    assert len(ids) == len(set(ids))


def test_formal_table_filters_and_variant_uniqueness() -> None:
    run_rows = {r['run_id']: r for r in rows('results/status/runs.csv')}
    score_rows = {r['score_id']: r for r in rows('results/status/scores.csv')}
    formal = rows('results/formal/r4_half/scores.csv')
    banned = {
        'pilot_keep80_official_all_20260727_174732',
        'qwen25c3b_r4_baseline_evalhalf_20260723_193503',
        'qwen25c3b_r4_baseline_evalhalf_recheck_20260726_181426',
    }
    assert not (banned & {r['run_id'] for r in formal})
    assert all(r['protocol'] == 'r4_half' for r in formal)
    assert all(run_rows[r['run_id']]['superseded_by'] == '' for r in formal)
    assert all(r['result_completeness'] != 'pilot' for r in formal)
    keys = [(r['method'], r['model'], r['variant'], r['benchmark'], r['protocol'], r['split'], r['metric_name'], r['run_id']) for r in formal]
    assert len(keys) == len(set(keys))
    assert all(r['variant'] for r in formal)
    assert any(r['variant'] == 'default_keep80' for r in formal)
    assert any(r['variant'] == 'benchmark_guided_keep80' for r in formal)
    assert any(r['variant'] == 'default_keep80_lora_merged' for r in formal)
    assert any(r['variant'] == 'benchmark_guided_keep80_lora_merged' for r in formal)
    assert all(any(s['run_id'] == r['run_id'] and s['source_file'] == r['source_file'] for s in score_rows.values()) for r in formal)


def test_flab_modes_cli_order_and_dry_run(tmp_path: Path) -> None:
    path = ROOT / 'methods/flab_pruner/qwen_prune.py'
    text = path.read_text(encoding='utf-8')
    main_pos = text.index('if __name__ == "__main__"')
    for name in ['topk_index', 'mask_from_index', 'validate_zs', 'move_zs', 'build_benchmark_guided_zs', 'compute_benchmark_activation_importance']:
        assert text.index(f'def {name}') < main_pos
    help_result = run_cmd([PY, str(path), '--help'])
    assert help_result.returncode == 0
    assert 'structural' in help_result.stdout and 'benchmark_activation' in help_result.stdout
    guide = tmp_path / 'guide.jsonl'
    guide.write_text(json.dumps({'task_id': 'x', 'prompt': 'def f(): pass', 'contains_solution': False}) + '\n', encoding='utf-8')
    out = tmp_path / 'out'
    structural = run_cmd([PY, str(path), '--model', 'fake-model', '--guide-file', str(guide), '--save-dir', str(out), '--dry-run', '--importance-mode', 'structural'])
    assert structural.returncode == 0, structural.stderr
    assert 'local_official_adapter' in structural.stdout
    bench = run_cmd([PY, str(path), '--model', 'fake-model', '--guide-file', str(guide), '--save-dir', str(out), '--dry-run', '--importance-mode', 'benchmark'])
    assert bench.returncode == 0, bench.stderr
    assert 'benchmark_activation' in bench.stdout and 'experimental_extension' in bench.stdout


def test_plan_audits_and_completion_semantics(tmp_path: Path) -> None:
    plan = (ROOT / 'workflows/experiment/stage1_plan.yaml').read_text(encoding='utf-8')
    assert ('check_' + 'readiness.py') not in plan
    assert '|| true' not in plan
    assert 'results/stage1' not in (ROOT / 'workflows/audit/check_environment.py').read_text(encoding='utf-8')
    env_no_output = run_cmd([PY, 'workflows/audit/check_environment.py'])
    assert env_no_output.returncode != 0
    policy = (ROOT / 'workflows/audit/check_official_benchmark_policy.py').read_text(encoding='utf-8')
    assert ('ROOT / "' + 'scripts"') not in policy
    assert ('ROOT / "' + 'configs"') not in policy
    completion = run_cmd([PY, 'workflows/audit/check_stage1_completion.py'])
    assert completion.returncode == 0
    data = json.loads(completion.stdout)
    assert data['repository_integrity'] is True
    assert data['stage1_complete'] is False
    assert data['blocked_methods'] or data['pending_methods'] or data['partial_methods']


def test_environment_lock_capture_fake_venv(tmp_path: Path) -> None:
    venv_root = tmp_path / 'venvs'
    for name in ['.venv-a', '.venv-b']:
        bin_dir = venv_root / name / 'bin'
        bin_dir.mkdir(parents=True)
        py = bin_dir / 'python'
        py.write_text('#!/usr/bin/env bash\nif [ "$1" = "-m" ] && [ "$2" = "pip" ]; then printf "z==2\\na==1\\n"; fi\n', encoding='utf-8')
        py.chmod(0o755)
    method_map = tmp_path / 'method_env_map.csv'
    with method_map.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method','venv_name','lock_file','extra_install','notes'])
        writer.writeheader()
        writer.writerow({'method':'a','venv_name':'.venv-a','lock_file':'environment/locks/a.txt','extra_install':'','notes':''})
        writer.writerow({'method':'b','venv_name':'.venv-b','lock_file':'environment/locks/b.txt','extra_install':'','notes':''})
    out_map = tmp_path / 'out_map.csv'
    result = run_cmd([PY, 'environment/setup/capture_environment_locks.py', '--write', '--venv-root', str(venv_root), '--method-map', str(method_map), '--output-map', str(out_map), '--lock-root', str(tmp_path / 'locks')])
    assert result.returncode == 0, result.stderr
    out_rows = list(csv.DictReader(out_map.open(encoding='utf-8')))
    assert len({r['lock_file'] for r in out_rows}) == 1
    check = run_cmd([PY, 'environment/setup/capture_environment_locks.py', '--check', '--venv-root', str(venv_root), '--method-map', str(out_map), '--lock-root', str(tmp_path / 'locks')])
    assert check.returncode == 0, check.stdout + check.stderr


def test_documentation_has_required_closure_sections() -> None:
    required = {
        'docs/stage1_protocol.md': ['Stage 1 goals', 'Method scope', 'R0-R4', 'Model policy', 'Completion criteria'],
        'docs/benchmark_protocol.md': ['HumanEval', 'MBPP', 'LiveCodeBench', 'SWE-bench-lite', 'auxiliary_full_eval', 'Formal table filtering'],
        'docs/recovery_protocol.md': ['LoRA', 'distillation', 'adapter', 'merge', 'recovery status'],
        'results/reports/reproduction_status.md': ['Owner', 'Upstream', 'Missing work', 'Flab-Pruner', 'FLAP'],
        'results/reports/failure_audit.md': ['dense baseline', 'save/reload', 'Flab first break', 'LLM-Pruner first break', 'SliceGPT first break', 'raw completion failure taxonomy'],
        'results/reports/protocol_deviations.md': ['fallback', 'pilot', 'aggregate-only', 'experimental importance'],
    }
    for rel, needles in required.items():
        text = (ROOT / rel).read_text(encoding='utf-8').lower()
        for needle in needles:
            assert needle.lower() in text, (rel, needle)
