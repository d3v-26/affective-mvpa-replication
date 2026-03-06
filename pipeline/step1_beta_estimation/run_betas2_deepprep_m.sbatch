#!/bin/bash

#SBATCH --job-name=betas_deepprep
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

module load matlab

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPM_PATH="/home/pateld3/spm"

matlab -nodisplay -nosplash -r \
    "addpath('${SPM_PATH}'); run('${SCRIPT_DIR}/BetaS2_deepprep.m'); exit;"
