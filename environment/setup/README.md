# Stage 1 Environment Setup

Project root in WSL:

```bash
cd ~/projects/srtp-code-llm-pruning
source environment/setup/env.sh
```

Common environment:

```bash
source .venv-common/bin/activate
python workflows/audit/check_environment.py
```

Method environments:

```bash
source .venv-wanda/bin/activate
source .venv-sparsegpt/bin/activate
source .venv-llm_pruner/bin/activate
source .venv-dsnot/bin/activate
source .venv-maskllm/bin/activate
source .venv-flap/bin/activate
source .venv-slicegpt/bin/activate
source .venv-laco/bin/activate
source .venv-owl/bin/activate
source .venv-pruner_zero/bin/activate
source .venv-flab_pruner/bin/activate
source .venv-livecodebench/bin/activate
```

Recreate environments:

```bash
environment/setup/create_method_envs.sh
```

Docker:

```bash
environment/setup/install_docker_wrapper.sh
docker ps
```

Notes:

- Hugging Face, datasets, Transformers and Torch caches are kept under `~/.cache` inside WSL.
- Docker Desktop is installed on Windows. The WSL wrapper calls the Windows Docker CLI if Docker Desktop has not created a native WSL socket.
- Some upstream requirements are intentionally not installed globally, especially `apex`, `deepspeed`, `fairseq`, `mamba-ssm`, and `vllm`, because they are heavyweight and version-sensitive. Install them only in the specific method env when a reproduction command actually requires them.
