#!/usr/bin/env bash
#SBATCH -p sunnycove
#SBATCH -t 720:00:00
#SBATCH -N 1 -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=22G
#SBATCH -o logs/DATA-%x-%j.out    # %x = job name, %j = job id

set -euo pipefail

echo "Job: $SLURM_JOB_NAME  |  ID: $SLURM_JOB_ID"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate potassco

# V2 dataset -- requires the generator branch `experimental/v2-dataset`.
# timezone, airport-types and time-granularity all live IN the config, so this run is
# reproducible from the config alone and needs no CLI overrides.
python run_pipeline.py --config=default_configs_small_scaling/30_2_india_4x10.json --out-root=unparsed_experiment_data_V2_small_scaling

python 05_transform_for_optimizer.py --in-exp-dir=unparsed_experiment_data_V2_small_scaling/30-2-INDIA-4x10-V2 --out-root=experiment_data_V2_small_scaling
