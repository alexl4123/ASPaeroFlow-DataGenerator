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

# V2 dataset. Generator branch: experimental/v2-dataset
# Every V2 parameter (timezone, airport-types, time-granularity) lives IN the config,
# so no CLI overrides are needed and the run is reproducible from the config alone.
python run_pipeline.py --config=default_configs_V2/30_0_east_asia_3x3.json --out-root=unparsed_experiment_data_V2_TG1

python 05_transform_for_optimizer.py --in-exp-dir=unparsed_experiment_data_V2_TG1/30-0-EAST-ASIA-3x3-V2 --out-root=experiment_data_V2_TG1
