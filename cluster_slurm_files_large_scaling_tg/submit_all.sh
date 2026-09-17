#!/usr/bin/env bash
# V2 large-scaling granularity ladder: 8 regions x 4 granularities, each followed by the
# transform and the capacity sweep. Run from the repository root:
#   bash cluster_slurm_files_large_scaling_tg/submit_all.sh
# Requires: git checkout v2.0.0, or a later commit of branch feature/sector-schedule-column
set -euo pipefail
mkdir -p logs
for f in cluster_slurm_files_large_scaling_tg/pipeline_*.slurm; do
    name="$(basename "${f%.slurm}")"
    echo "submitting $name"; sbatch --job-name="$name" "$f"
done
