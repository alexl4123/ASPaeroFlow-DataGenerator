#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -t 720:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=22G
#SBATCH -o logs/DATA-%x-%j.out    # %x = job name, %j = job id

set -euo pipefail

echo "Job: $SLURM_JOB_NAME  |  ID: $SLURM_JOB_ID"

# Activate potassco env
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco

python run_pipeline.py --config=default_configs_TG4/04_0_dach_20190601_30_cap_enroute_1200_clustersize_50_GABRIEL_GRAPH.json --out-root=unparsed_experiment_data_TG4

python 05_transform_for_optimizer.py --in-exp-dir=unparsed_experiment_data_TG4/04-0-DACH-2019-06-01--2019-06-30-CAP-ENROUTE-1200-CLUSTERSIZE-50-GABRIEL-GRAPH --out-root=experiment_data_TG4
