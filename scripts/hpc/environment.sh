# Shared defaults for the Apptainer/Slurm launch scripts.
# Set DATA_DIR_ACADEMIC_INBREEDING in the user's shell environment.

ACIB_CONFIG_DIR="${ACIB_CONFIG_DIR:-${REPO_DIR}/config}"
ACIB_APPTAINER_IMAGE="${ACIB_APPTAINER_IMAGE:-${REPO_DIR}/academic-inbreeding.sif}"

if [[ -n "${ACIB_DATA_DIR:-}" ]]; then
    DATA_DIR_ACADEMIC_INBREEDING="${ACIB_DATA_DIR}"
fi

if [[ -z "${DATA_DIR_ACADEMIC_INBREEDING:-}" ]]; then
    cat >&2 <<'EOF'
ERROR: DATA_DIR_ACADEMIC_INBREEDING is not set.
Set it in ~/.bashrc to the shared data directory containing input/, for example:
  export DATA_DIR_ACADEMIC_INBREEDING="/path/to/academic_inbreeding/data"
Then reload your shell with: source ~/.bashrc
EOF
    return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${DATA_DIR_ACADEMIC_INBREEDING}" ]]; then
    printf 'ERROR: DATA_DIR_ACADEMIC_INBREEDING does not exist: %s\n' \
        "${DATA_DIR_ACADEMIC_INBREEDING}" >&2
    return 1 2>/dev/null || exit 1
fi

if [[ ! -d "${DATA_DIR_ACADEMIC_INBREEDING}/input" ]]; then
    printf 'ERROR: expected input/ under DATA_DIR_ACADEMIC_INBREEDING, but not found: %s/input\n' \
        "${DATA_DIR_ACADEMIC_INBREEDING}" >&2
    return 1 2>/dev/null || exit 1
fi

ACIB_DATA_DIR="${DATA_DIR_ACADEMIC_INBREEDING}"
export DATA_DIR_ACADEMIC_INBREEDING
export ACIB_CONFIG_DIR ACIB_DATA_DIR ACIB_APPTAINER_IMAGE
