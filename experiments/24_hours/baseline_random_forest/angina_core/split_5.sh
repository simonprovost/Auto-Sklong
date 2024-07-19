#!/bin/bash

# ========================================
# SLURM Configuration
# ========================================
#SBATCH --job-name="24_HOURS_CORE_ANGINA_BASELINE_RF_split_5"
#SBATCH --output="24_HOURS_CORE_ANGINA_BASELINE_RF_split_5.log"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="<your_email_to_be_alerted>
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1GB
#SBATCH --partition=cpu

# ========================================
# Experiment Variables
# ========================================
# Variables that can be customized:
# - DATASET_PATH: Path to the dataset [MANDATORY]
# - TARGET_COLUMN: Target column in the dataset [MANDATORY]
# - FOLD_NUMBER: Outer-fold cross validation number (for report only) [MANDATORY]
# - EXPORT_NAME: Folder name for exporting results [MANDATORY]
# - N_INNER_JOBS: Number of inner jobs (for the search algorithm) [OPTIONAL]
# - RANDOM_STATE: Random seed for reproducibility [OPTIONAL]
# - MAX_TOTAL_TIME: Maximum total time for the experiment [OPTIONAL]
# - MAX_EVAL_TIME: Maximum evaluation time for one candidate cross-validation process [OPTIONAL]
# - SCORING: Scoring metric for evaluation and optimisation [OPTIONAL]
# - SEARCH_ALGORITHM: Search algorithm for hyperparameter optimization ("bayesian_optimisation", "random_search", "evolutionary_algorithm", "ASHA") [OPTIONAL]
# - SHUFFLING: Whether to shuffle the data (True/False) [OPTIONAL]
# - STORE: Store level (e.g., "all") [OPTIONAL]
# - VERBOSITY: Level of verbosity for output (integer) [OPTIONAL]
# - MAX_MEMORY_MB: Maximum memory in MB for the experiment [OPTIONAL]
# - N_OUTER_SPLITS: Number of outer splits for cross-validation [OPTIONAL]

DATASET_PATH="../scikit-longitudinal/data/elsa/core/csv/angina_dataset.csv"
TARGET_COLUMN="class_angina_w8"
FOLD_NUMBER=5
EXPORT_NAME="24_HOURS_CORE_ANGINA_BASELINE_RF"

# ========================================
# Environment Setup (NO NEED TO MODIFY)
# ========================================
#source ~/miniconda3/etc/profile.d/conda.sh
#export PATH="<your_home_directory_path?miniconda3/bin/:$PATH"
#export PATH="<your_home_directory_path?.local/bin/:$PATH"
#export PATH="<your_home_directory_path?.pyenv/bin/:$PATH"
#pyenv local 3.9.8
#pdm use 3.9
#export PDM_IN_ENV=in-project
#cd <your_home_directory_path?Auto-Sklong
#eval $(pdm venv activate $PDM_IN_ENV)

# ========================================
# Run Experiment (NO NEED TO MODIFY)
# ========================================
python ./experiments/experiment_launchers/random_forest.py --dataset_path $DATASET_PATH --target_column $TARGET_COLUMN --fold_number $FOLD_NUMBER --export_name $EXPORT_NAME
