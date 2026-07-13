# Environment Gap Analysis

## Already configured

- WSL2 Ubuntu is available and can access the NVIDIA GPU.
- Project workspace created at `~/projects/srtp-code-llm-pruning` inside the WSL Linux filesystem.
- Python 3.11 virtual environment created at `.venv-common` with `uv`.
- PyTorch CUDA stack verified: `torch==2.11.0+cu128`, CUDA runtime 12.8, RTX 5060 Laptop GPU visible.
- WSL-local Git LFS installed to `~/.local/bin/git-lfs` and initialized for this repository.
- Common benchmark Python packages installed: `evalplus`, `human-eval`, `swebench`.

## Remaining gaps

- Current WSL distribution is Ubuntu 26.04, while the execution book recommends Ubuntu 22.04 LTS. The Python environment is pinned to 3.11 to reduce compatibility risk, but strict team parity still requires an Ubuntu 22.04 distro.
- Docker Desktop is not installed. `winget install Docker.DockerDesktop` downloaded the installer but failed because the installer requested administrator elevation.
- LiveCodeBench is not available as a `livecodebench` PyPI package. It should be integrated by fixing the official repository URL and commit under `third_party/`.
- Native Windows MSVC `cl` is missing. This is not on the main WSL path, but may matter for Windows-side builds.
- Conda is not installed. The common environment currently uses `uv`; conda-compatible environment metadata is recorded in `env/common/environment.yml`.
