#!/usr/bin/env bash
# Submit the full V2 generation campaign. Run from the repository root:
#   bash cluster_slurm_files_V2/submit_all.sh
set -euo pipefail
mkdir -p logs
for f in cluster_slurm_files_V2/pipeline_*.slurm; do
    name="$(basename "${f%.slurm}")"
    echo "submitting $name"
    sbatch --job-name="$name" "$f"
done
