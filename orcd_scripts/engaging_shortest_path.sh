#!/bin/bash

# Job Flags
#SBATCH --job-name=shortest_path
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
mkdir -p logs

# Data directory: override with RIYADH_DATA_DIR env var on the cluster
# DATA_DIR="${RIYADH_DATA_DIR:-${BASEDIR}/data}"
# CONTAINER="${RIYADH_TRANSPORTATION_CONTAINER:-${JLAB_CONTAINER:-}}"

DATA_DIR="/home/fiorista/documents/riyadh-transportation/data"
CONTAINER="/home/fiorista/containers/riyadh-transportation.sif"

if [[ -n "$CONTAINER" ]]; then
	exec apptainer exec \
		${RIYADH_TRANSPORTATION_APPTAINER_ARGS:-} \
		--bind "${BASEDIR}:/work" \
		--bind "${DATA_DIR}:${DATA_DIR}" \
		--env RIYADH_DATA_DIR="${DATA_DIR}" \
		--pwd /work \
		"$CONTAINER" \
		bash -lc 'set -euo pipefail; exec /usr/local/bin/uv run --no-sync python -u -m experiments.shortest_path_calculation "$@"' bash "$@"
fi

export RIYADH_DATA_DIR="${DATA_DIR}"
uv sync --frozen
exec uv run python -u -m experiments.shortest_path_calculation
