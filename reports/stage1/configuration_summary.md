# Stage 1 Local Configuration Summary

## Configured

- WSL project root: `~/projects/srtp-code-llm-pruning`
- Remote GitHub repository: `https://github.com/ACN67/SRTP-Project-Stage1.git`
- Git LFS initialized with model/data artifact patterns.
- Third-party upstream repositories added as Git submodules under `third_party/`.
- Upstream commit manifest written to `data/manifests/upstream_repos.csv`.
- User-level WSL tools installed:
  - `uv`
  - `micromamba`
  - `gh`
  - `ninja`
  - `git-lfs`
- Windows Docker Desktop installed and running.
- WSL `docker` wrapper installed at `~/.local/bin/docker`.
- Hugging Face, datasets, Transformers and Torch caches pinned to WSL `~/.cache`.
- Common Python environment: `.venv-common`.
- Method Python environments:
  - `.venv-magnitude`
  - `.venv-wanda`
  - `.venv-sparsegpt`
  - `.venv-llm_pruner`
  - `.venv-dsnot`
  - `.venv-maskllm`
  - `.venv-flap`
  - `.venv-slicegpt`
  - `.venv-laco`
  - `.venv-owl`
  - `.venv-pruner_zero`
  - `.venv-flab_pruner`
  - `.venv-livecodebench`
- Environment lock snapshots generated under `env/<name>/pip_freeze.txt`.
- Environment audit generated at `reports/stage1/environment_check.json`.

## Deliberately Deferred

- Ubuntu 22.04 LTS was not installed because the existing WSL distro is already functional and GPU-enabled. Current distro is Ubuntu 26.04, with Python pinned to 3.11 in every project venv.
- Heavy or fragile upstream packages are not installed globally: `apex`, `deepspeed`, `fairseq`, `mamba-ssm`, `vllm`. Install these only inside a method environment when a concrete reproduction command requires them.
- Native Windows MSVC `cl` is still absent. The primary execution path is WSL, so this is not blocking the current pruning workflow.
