#!/usr/bin/env bash
# Submit the V2 small_scaling campaign. Run from the repository root:
#   bash cluster_slurm_files_small_scaling/submit_all.sh
# Requires: git checkout experimental/v2-dataset
set -euo pipefail
mkdir -p logs
for f in cluster_slurm_files_small_scaling/pipeline_*.slurm; do
    name="$(basename "${f%.slurm}")"
    echo "submitting $name"; sbatch --job-name="$name" "$f"
done
