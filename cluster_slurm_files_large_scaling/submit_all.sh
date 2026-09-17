#!/usr/bin/env bash
# Submit the V2 large_scaling campaign. Run from the repository root:
#   bash cluster_slurm_files_large_scaling/submit_all.sh
# Requires: git checkout v2.0.0, or a later commit of branch feature/sector-schedule-column
set -euo pipefail
mkdir -p logs
for f in cluster_slurm_files_large_scaling/pipeline_*.slurm; do
    name="$(basename "${f%.slurm}")"
    echo "submitting $name"; sbatch --job-name="$name" "$f"
done
