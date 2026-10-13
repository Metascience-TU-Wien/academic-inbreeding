#!/usr/bin/env bash
#SBATCH -J acib_mobility
#SBATCH --account=default
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mobility_%j.out
#SBATCH --error=logs/mobility_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_DIR="${ACIB_CONFIG_DIR:-${REPO_DIR}/config}"
DATA_DIR="${ACIB_DATA_DIR:-${REPO_DIR}/data}"
IMAGE_PATH="${ACIB_APPTAINER_IMAGE:-${REPO_DIR}/academic-inbreeding.sif}"
CPUS="${SLURM_CPUS_PER_TASK:-3}"

cd "${REPO_DIR}"
mkdir -p logs "${DATA_DIR}"
test -f "${CONFIG_DIR}/pipeline.toml"
test -f "${IMAGE_PATH}"

export OMP_NUM_THREADS="${CPUS}"
export MKL_NUM_THREADS="${CPUS}"
export OPENBLAS_NUM_THREADS="${CPUS}"
export NUMEXPR_NUM_THREADS="${CPUS}"
export PYTHONUNBUFFERED=1

apptainer exec \
    --pwd /app \
    --bind "${CONFIG_DIR}:/app/config:ro" \
    --bind "${DATA_DIR}:/app/data" \
    "${IMAGE_PATH}" \
    python -u scripts/analyze_mobility.py --config /app/config/pipeline.toml "$@"
