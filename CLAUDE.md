# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Neuroimaging MVPA (Multi-Voxel Pattern Analysis) pipeline for decoding affective scenes from retinotopic visual cortex. Based on **Bo et al. (2021) - "Decoding Neural Representations of Affective Scenes in Retinotopic Visual Cortex"**. Analyzes fMRI data from 20 subjects across 5 runs, classifying three emotional conditions: Pleasant (Pl), Neutral (Nt), and Unpleasant (Up) using SVM decoding.

## Pipeline Architecture

The analysis runs as a sequential pipeline, each step producing inputs for the next:

1. **`pipeline/step1_beta_estimation/BetaS2.m`** — SPM 1st-level GLM estimation. Builds design matrices with 60 condition regressors per run (20 Pl + 20 Nt + 20 Up) plus 6 motion regressors. Outputs single-trial beta images per subject. Run via `run_betas2_m.sbatch`.

2. **`pipeline/step2_extract_betas/extract_betas.m`** — Reads SPM beta images and reshapes them into per-condition matrices (`Pl#.mat`, `Nt#.mat`, `Up#.mat`), each shaped `[nVoxels x nTrials]`. Beta indexing: 66 regressors/run (60 conditions + 6 motion), sequential across 5 sessions.

3. **`pipeline/step3_roi_masks/make_roi_masks_mat.m`** — Converts ROI NIfTI masks (`*_in_EPI_bin.nii.gz`, expected shape 53x63x46) into a single `roi_masks.mat` struct. ROIs: V1v/d, V2v/d, V3v/d, hV4, VO1/2, PHC1/2, hMT, LO1/2, V3a/b, IPS.

4. **`pipeline/step4_decoding/SingleTrialDecodingv3.m`** — SVM decoding (linear kernel, standardized) with repeated k-fold cross-validation (k=5, 100 repetitions). Runs two binary classifications per ROI: Pleasant-vs-Neutral and Unpleasant-vs-Neutral. Also supports whole-brain decoding. Run via `run_singletd_m.sbatch`.

5. **`pipeline/step5_group_validation/group_level_validation.py`** — Python script for group-level permutation testing (10^5 permutations, p<0.001 threshold) with one-sample t-tests and Cohen's d. Supports SLURM checkpointing for job recovery. Run via `run_group_validation.sh`.

## Running on HiPerGator (SLURM)

MATLAB scripts require SPM on the path (`/home/pateld3/spm`):
```bash
sbatch pipeline/step1_beta_estimation/run_betas2_m.sbatch        # Step 1: GLM estimation (24h, 30GB)
sbatch pipeline/step4_decoding/run_singletd_m.sbatch             # Step 4: SVM decoding (4 days, 30GB)
```

Python group validation (uses `~/.venv` virtualenv, needs numpy/scipy/matplotlib/seaborn):
```bash
sbatch pipeline/step5_group_validation/run_group_validation.sh    # Step 5: permutation testing (4h, 20GB)
```

Steps 2-3 (`extract_betas.m`, `make_roi_masks_mat.m`) are run interactively in MATLAB.

## Key Data Paths (cluster)

- Preprocessed fMRI & betas: `/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/`
- Extracted matrices & ROI masks: `/blue/ruogu.fang/pateld3/neuroimaging/output_mats/`
- Decoding results: `/orange/.../single_mvpa_results/decoding_results_k5x100_v4.mat`
- Onset files: `/home/pateld3/NewStimuluesSetting/Sub##run#.mat` (100 total: 20 subs x 5 runs)

## Key Parameters

- TR = 1.98s, 206 scans/run, stimulus duration = 1.5152s
- 20 trials per condition per run (60 total/run, 300 total/subject)
- High-pass filter: 128s, AR(1) autocorrelation, global scaling
- SVM: linear kernel, standardized features, 5-fold x 100 reps
- Group statistics: 100,000 permutations, alpha = 0.001
