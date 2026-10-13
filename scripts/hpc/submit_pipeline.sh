#!/usr/bin/env bash
#SBATCH -J acib_submit_pipeline
#SBATCH --account=default
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=logs/submit_pipeline_%j.out
#SBATCH --error=logs/submit_pipeline_%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
CONFIG_DIR="${ACIB_CONFIG_DIR:-${REPO_DIR}/config}"
IMAGE_PATH="${ACIB_APPTAINER_IMAGE:-${REPO_DIR}/academic-inbreeding.sif}"

require_command() {
    command -v "$1" >/dev/null 2>&1 || {
        echo "Missing required command: $1" >&2
        exit 1
    }
}

submit_job() {
    local output
    output="$(sbatch --parsable "$@")"
    echo "${output%%;*}"
}

require_command sbatch
test -f "${CONFIG_DIR}/pipeline.toml" || {
    echo "Missing pipeline config: ${CONFIG_DIR}/pipeline.toml" >&2
    exit 1
}
test -f "${IMAGE_PATH}" || {
    echo "Missing Apptainer image: ${IMAGE_PATH}" >&2
    exit 1
}
for script in \
    match_author_names.sh \
    disambiguate_author_names.sh \
    analyze_mobility.sh \
    regression_analysis.sh \
    statistical_matching_analysis.sh; do
    test -f "${REPO_DIR}/scripts/hpc/${script}" || {
        echo "Missing stage script: ${REPO_DIR}/scripts/hpc/${script}" >&2
        exit 1
    }
done

mkdir -p "${REPO_DIR}/logs"

matching_job_id="$(submit_job --chdir="${REPO_DIR}" "${REPO_DIR}/scripts/hpc/match_author_names.sh")"
disambiguation_job_id="$(submit_job --chdir="${REPO_DIR}" \
    --dependency="afterok:${matching_job_id}" \
    "${REPO_DIR}/scripts/hpc/disambiguate_author_names.sh")"
mobility_job_id="$(submit_job --chdir="${REPO_DIR}" \
    --dependency="afterok:${disambiguation_job_id}" \
    "${REPO_DIR}/scripts/hpc/analyze_mobility.sh")"
regression_job_id="$(submit_job --chdir="${REPO_DIR}" \
    --dependency="afterok:${mobility_job_id}" \
    "${REPO_DIR}/scripts/hpc/regression_analysis.sh")"
statistical_matching_job_id="$(submit_job --chdir="${REPO_DIR}" \
    --dependency="afterok:${regression_job_id}" \
    "${REPO_DIR}/scripts/hpc/statistical_matching_analysis.sh")"

echo "Submitted academic-inbreeding pipeline jobs:"
echo "  matching:             ${matching_job_id}"
echo "  disambiguation:       ${disambiguation_job_id} afterok:${matching_job_id}"
echo "  mobility:             ${mobility_job_id} afterok:${disambiguation_job_id}"
echo "  regression:           ${regression_job_id} afterok:${mobility_job_id}"
echo "  statistical matching: ${statistical_matching_job_id} afterok:${regression_job_id}"
