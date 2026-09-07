#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -t 720:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=22G
#SBATCH -o logs/DATA-%x-%j.out

set -euo pipefail
echo "Job: $SLURM_JOB_NAME  |  ID: $SLURM_JOB_ID"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco

# V2 large-scaling, granularity ladder. Requires branch experimental/v2-dataset.
# All parameters live in the config; no CLI overrides.
python run_pipeline.py --config=default_configs_large_scaling_tg/00_0_central_europe_7x7_TG04.json --out-root=unparsed_experiment_data_V2_large_scaling_TG4

python 05_transform_for_optimizer.py --in-exp-dir=unparsed_experiment_data_V2_large_scaling_TG4/00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1-V2-TG4 --out-root=experiment_data_V2_large_scaling_TG4

# Relative-capacity overlays. Emitted as sectors.csv per level rather than full instance
# copies: a PCAP level differs in that one file only, so overlays cost ~0.5 GB where copies
# would cost ~27 GB for this family. Reconstruct with --materialise (see the overlay README).
python 06_capacity_sweep.py --exp-dir=experiment_data_V2_large_scaling_TG4/00-0-CENTRAL-EUROPE-7x7-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-1-V2-TG4 --out-root=capacity_overlays_V2_large_scaling_TG4 --time-granularity=4
