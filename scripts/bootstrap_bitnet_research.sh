#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="bitnet_research"
PREPARE_DATA=0
DEPLOY_AGENT=0
RUN_TESTS=1

while [[ $# -gt 0 ]]; do
	case "$1" in
		--env)
			ENV_NAME="$2"
			shift 2
			;;
		--prepare-data)
			PREPARE_DATA=1
			shift
			;;
		--deploy-agent)
			DEPLOY_AGENT=1
			shift
			;;
		--skip-tests)
			RUN_TESTS=0
			shift
			;;
		*)
			echo "Unknown option: $1" >&2
			exit 1
			;;
	esac
done

if ! command -v conda >/dev/null 2>&1; then
	echo "conda is required but was not found in PATH." >&2
	exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BASE="$(conda info --base)"
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
	conda env update -n "$ENV_NAME" -f "$REPO_ROOT/environment.yml" --prune
else
	conda env create -f "$REPO_ROOT/environment.yml"
fi

conda activate "$ENV_NAME"
export UV_LINK_MODE=copy

cd "$REPO_ROOT"

uv pip install --python "$(which python)" --index-url https://download.pytorch.org/whl/cpu torch==2.9.1
uv pip install --python "$(which python)" kernels matplotlib numpy pandas pyarrow requests rustbpe tiktoken
uv pip install --python "$(which python)" -e . --no-deps

if [[ "$RUN_TESTS" -eq 1 ]]; then
	python -m unittest tests/test_bitnet_cpu_poc.py tests/test_ternary_lab.py tests/test_deploy_cpu_agent.py
fi

if [[ "$PREPARE_DATA" -eq 1 ]]; then
	python prepare.py
fi

if [[ "$DEPLOY_AGENT" -eq 1 ]]; then
	DEPLOY_MODE="mock"
	if [[ "$PREPARE_DATA" -eq 1 ]]; then
		DEPLOY_MODE="auto"
	fi
	python deploy_cpu_agent.py --agent-name cpu-agent-1 --mode "$DEPLOY_MODE"
fi

echo "---"
echo "Environment:      $ENV_NAME"
echo "Python:           $(which python)"
echo "Tests:            $([[ "$RUN_TESTS" -eq 1 ]] && echo ran || echo skipped)"
echo "Data prep:        $([[ "$PREPARE_DATA" -eq 1 ]] && echo ran || echo skipped)"
if [[ "$DEPLOY_AGENT" -eq 1 ]]; then
	if [[ "$PREPARE_DATA" -eq 1 ]]; then
		echo "Agent deploy:     requested (auto mode)"
	else
		echo "Agent deploy:     requested (mock mode)"
	fi
else
	echo "Agent deploy:     skipped"
fi
echo "Next step:        conda activate $ENV_NAME"