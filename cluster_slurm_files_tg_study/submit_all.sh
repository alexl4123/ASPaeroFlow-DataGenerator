#!/usr/bin/env bash
# Submit the V2 tg_study campaign. Run from the repository root:
#   bash cluster_slurm_files_tg_study/submit_all.sh
# Requires: git checkout experimental/v2-dataset
set -euo pipefail
mkdir -p logs
for f in cluster_slurm_files_tg_study/pipeline_*.slurm; do
    name="$(basename "${f%.slurm}")"
    echo "submitting $name"; sbatch --job-name="$name" "$f"
done
