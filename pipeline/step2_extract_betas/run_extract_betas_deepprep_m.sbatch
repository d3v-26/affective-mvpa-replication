#!/bin/bash

#SBATCH --job-name=extract_betas_deepprep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30gb
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log

module load matlab

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPM_PATH="/home/pateld3/spm"

matlab -nodisplay -nosplash -r \
    "addpath('${SPM_PATH}'); addpath('${SCRIPT_DIR}'); extract_betas_deepprep(); exit;"
