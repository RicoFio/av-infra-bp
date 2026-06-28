#!/bin/bash

# Job Flags
#SBATCH -p mit_preemptable
#SBATCH --time=10:00:00
#SBATCH -c 64
#SBATCH --mem=128G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ -n "${PROJECT_ROOT:-}" ]]; then
	BASEDIR="${PROJECT_ROOT}"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
	BASEDIR="${SLURM_SUBMIT_DIR}"
else
	BASEDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
fi

cd "${BASEDIR}"

CONTAINER="${RIYADH_TRANSPORTATION_CONTAINER:-${JLAB_CONTAINER:-}}"

if [[ -n "$CONTAINER" ]]; then
	exec apptainer exec \
		${RIYADH_TRANSPORTATION_APPTAINER_ARGS:-} \
		--bind "${BASEDIR}:/work" \
		--pwd /work \
		"$CONTAINER" \
		bash -lc 'set -euo pipefail; /usr/local/bin/uv sync --frozen; exec /usr/local/bin/uv run python -m experiments.all_gurobi "$@"' bash "$@"
fi

uv sync --frozen
exec uv run python -m 
