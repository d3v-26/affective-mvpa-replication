# Migrating from SPM Preprocessing to fMRIPrep / DeepPrep

This document describes what changes are needed to swap the SPM-preprocessed input data (`swars*.img`) with fMRIPrep or DeepPrep outputs, and which scripts need modification.

---

## Overview: What Changes and What Doesn't

| Pipeline Step | Script | Needs Changes? | Why |
|---|---|---|---|
| **Step 1** | `BetaS2.m` | **YES - Major** | Reads SPM-preprocessed `swars*.img` and `rp_*.txt`; both file formats and naming change entirely |
| **Step 2** | `extract_betas.m` | **Minor / None** | Reads beta images output by Step 1. If you still use SPM for the GLM, no change needed. If you switch the GLM tool (e.g., to Nilearn/NiBabel), this needs rewriting. |
| **Step 3** | `make_roi_masks_mat.m` | **YES - Moderate** | ROI masks must match the output space of fMRIPrep (different MNI template, different voxel grid) |
| **Step 4** | `SingleTrialDecodingv3.m` | **No** | Consumes `.mat` files from Steps 2-3; agnostic to preprocessing |
| **Step 5** | `group_level_validation.py` | **No** | Consumes `.mat` from Step 4; agnostic to preprocessing |
| **SLURM** | `run_betas2_m.sbatch` | **Minor** | Path updates, possibly module changes |
| **SLURM** | `run_singletd_m.sbatch` | **No** | No dependency on preprocessing |
| **SLURM** | `run_group_validation.sh` | **No** | No dependency on preprocessing |

---

## Required Input Files from fMRIPrep / DeepPrep

### 1. Preprocessed BOLD images (replaces `swars*.img`)

**SPM current format:**
```
Sub##/run#/swars*.img    (smoothed + warped + realigned + slice-timing corrected)
```

**fMRIPrep output (BIDS derivatives):**
```
sub-##/func/sub-##_task-emotion_run-#_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
```

**DeepPrep output (similar BIDS layout):**
```
sub-##/func/sub-##_task-emotion_run-#_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
```

Key differences from the current `swars*.img` files:

| Property | SPM (`swars*.img`) | fMRIPrep / DeepPrep |
|---|---|---|
| **Format** | Analyze `.img/.hdr` pairs | NIfTI `.nii.gz` |
| **MNI template** | SPM's MNI152 (linear, ICBM) | `MNI152NLin2009cAsym` (nonlinear, higher quality) |
| **Voxel size** | 3 x 3 x 3 mm (specified in normalization) | Varies by `res-` tag; commonly `res-2` (2mm) or native |
| **Smoothing** | Applied (FWHM = 8mm, the `s` prefix) | **Not applied** by default |
| **Slice timing** | Applied (the inner `s` in `swars`) | Applied if `--slice-timing-correction` flag used (default: auto-detect from BIDS metadata) |
| **Volume count** | 206 per run (first 5 already discarded) | All volumes present; initial volumes **not** discarded |
| **Naming** | `swars{original_name}.img` | BIDS entity naming |

### 2. Confounds / motion parameters (replaces `rp_*.txt`)

**SPM current format:**
```
Sub##/run#/rp_*.txt    (6 columns: transX, transY, transZ, rotX, rotY, rotZ)
```

**fMRIPrep output:**
```
sub-##/func/sub-##_task-emotion_run-#_desc-confounds_timeseries.tsv
```

This is a TSV file with **many** columns (50+), including:
- `trans_x`, `trans_y`, `trans_z` — translation in mm
- `rot_x`, `rot_y`, `rot_z` — rotation in radians
- `framewise_displacement` — summary motion metric
- `csf`, `white_matter` — tissue-based confounds
- `cosine00`, `cosine01`, ... — discrete cosine basis set (high-pass filtering)
- `a_comp_cor_00`, ... — aCompCor components
- Many derivatives and squared terms

You need to select which columns to use as nuisance regressors. The minimal equivalent of the current `rp_*.txt` is:
```
trans_x, trans_y, trans_z, rot_x, rot_y, rot_z
```

But fMRIPrep also offers richer denoising strategies (see Section "Choosing Confound Strategy" below).

### 3. Brain masks

**fMRIPrep output:**
```
sub-##/func/sub-##_task-emotion_run-#_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz
```

These can be used to constrain your GLM to brain voxels only (improving efficiency).

### 4. Event / onset files (NO CHANGE)

The `Sub##run#.mat` onset files are independent of preprocessing. They stay the same. However, if you switch to a Python-based GLM (e.g., Nilearn), you'll want to convert them to BIDS-style `_events.tsv` format:

```tsv
onset	duration	trial_type
0.0	3.0	Pl1
5.8	3.0	Nt3
...
```

---

## Script-by-Script Changes

### Step 1: `BetaS2.m` — Major Changes Required

This is the script most affected. There are **two approaches** to adapt it:

#### Approach A: Keep SPM for the GLM (recommended for minimal changes)

Keep using SPM's `spm_fmri_spm_ui` and `spm_spm` for beta estimation, but change how input files are loaded.

**Changes needed:**

1. **File selection** — Replace the `spm_select` call for `swars*.img`:

```matlab
% CURRENT (line 81):
tmp{j} = spm_select('fplist', runDir{j}, '^swars.*\.img');

% NEW: point to fMRIPrep output (gunzip first, SPM needs uncompressed .nii)
fmriprep_dir = '/path/to/fmriprep/output';
bold_gz = fullfile(fmriprep_dir, sprintf('sub-%02d', i), 'func', ...
    sprintf('sub-%02d_task-emotion_run-%d_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz', i, j));

% Gunzip to a working directory (SPM cannot read .nii.gz directly)
gunzip(bold_gz, working_dir);
bold_nii = fullfile(working_dir, strrep(basename(bold_gz), '.gz', ''));

% SPM needs individual volume paths (one per TR) via spm_select
tmp{j} = spm_select('ExtFPList', fileparts(bold_nii), spm_file(bold_nii,'basename'), Inf);
```

2. **Discard initial volumes** — fMRIPrep does NOT discard the first 5 volumes. Either:
   - Add `non_steady_state_outlier00`, etc. columns as nuisance regressors (fMRIPrep detects these automatically), OR
   - Manually discard the first N volumes and adjust `nscan` accordingly:
   ```matlab
   n_discard = 5;  % or check fMRIPrep's non_steady_state columns
   tmp{j} = tmp{j}(n_discard+1:end, :);
   nscan(j) = size(tmp{j}, 1);
   ```

3. **Motion parameters** — Replace `rp_*.txt` loading with confounds TSV:

```matlab
% CURRENT (lines 167-169):
fn = spm_select('list', rpDir, '^rp.*\.txt');
[r1,r2,r3,r4,r5,r6] = textread(fn, '%f%f%f%f%f%f');

% NEW: read fMRIPrep confounds TSV
confounds_file = fullfile(fmriprep_dir, sprintf('sub-%02d', i), 'func', ...
    sprintf('sub-%02d_task-emotion_run-%d_desc-confounds_timeseries.tsv', i, j));
T = readtable(confounds_file, 'FileType', 'text', 'Delimiter', '\t');

% Extract 6 motion parameters (same as before)
r1 = T.trans_x; r2 = T.trans_y; r3 = T.trans_z;
r4 = T.rot_x;   r5 = T.rot_y;   r6 = T.rot_z;

% If discarding initial volumes:
r1 = r1(n_discard+1:end); % ... same for r2-r6

SPM.Sess(j).C.C = [r1 r2 r3 r4 r5 r6];
SPM.Sess(j).C.name = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
```

4. **Smoothing** — fMRIPrep does not smooth. You must either:
   - Apply smoothing before the GLM using SPM's `spm_smooth`:
     ```matlab
     spm_smooth(bold_nii, smoothed_nii, [8 8 8]);  % 8mm FWHM to match paper
     ```
   - OR skip smoothing (common for MVPA — smoothing actually blurs spatial patterns and can *reduce* decoding accuracy; see note below)

5. **High-pass filtering** — fMRIPrep includes `cosine` basis regressors in the confounds TSV that implement high-pass filtering. You have two options:
   - Keep SPM's built-in high-pass filter (`SPM.xX.K(j).HParam = 128`) — this is fine and matches the current pipeline
   - OR remove SPM's filter and instead add the `cosine##` columns from the confounds TSV as additional nuisance regressors

6. **Voxel dimensions** — If fMRIPrep output is at 2mm resolution instead of 3mm, the volume dimensions will differ. This affects ROI mask alignment (see Step 3).

7. **Directory structure** — The current script expects `Sub##/run#/` directory layout. fMRIPrep uses BIDS: `sub-##/func/`. Update all path construction accordingly.

#### Approach B: Replace SPM GLM entirely with Nilearn (Python)

For a fully Python-based pipeline, replace `BetaS2.m` + `extract_betas.m` with a Python script using Nilearn's `FirstLevelModel`. This is a larger rewrite but removes the SPM/MATLAB dependency.

```python
# Pseudocode for Python-based beta series estimation
from nilearn.glm.first_level import FirstLevelModel
import nibabel as nib
import pandas as pd
import numpy as np

for sub_id in range(1, 21):
    # Build events DataFrame with one row per trial
    events = build_single_trial_events(sub_id)  # columns: onset, duration, trial_type

    # Load confounds
    confounds = load_fmriprep_confounds(sub_id, strategy='6motion')

    # Load fMRIPrep preprocessed BOLD
    bold_imgs = [f'sub-{sub_id:02d}_task-emotion_run-{r}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'
                 for r in range(1, 6)]

    # Fit GLM with single-trial regressors
    model = FirstLevelModel(
        t_r=1.98,
        hrf_model='spm',           # canonical HRF (same as current)
        drift_model='cosine',       # high-pass filter
        high_pass=1/128,
        noise_model='ar1',          # AR(1) autocorrelation
        standardize=False,
        smoothing_fwhm=8,           # or None for no smoothing
    )
    model.fit(bold_imgs, events=events, confounds=confounds)

    # Extract single-trial betas
    for trial_name in trial_names:
        beta_img = model.compute_contrast(trial_name, output_type='effect_size')
        # ... save or collect into Pl/Nt/Up matrices
```

This approach would combine Steps 1 and 2 into a single Python script and eliminate the MATLAB/SPM dependency for beta estimation.

---

### Step 2: `extract_betas.m` — Depends on Step 1 Approach

**If using Approach A (SPM GLM):** No changes needed. SPM still writes `beta_####.nii` files in the same format. `extract_betas.m` reads these identically.

**If using Approach B (Nilearn GLM):** This script is replaced entirely by the Python code above, which directly produces the `Pl`, `Nt`, `Up` matrices. Save them as `.mat` or `.npy` files:

```python
import scipy.io as sio
sio.savemat(f'Pl{sub_id}.mat', {'Pl': pl_matrix})  # [nVoxels x 100]
sio.savemat(f'Nt{sub_id}.mat', {'Nt': nt_matrix})
sio.savemat(f'Up{sub_id}.mat', {'Up': up_matrix})
```

---

### Step 3: `make_roi_masks_mat.m` — Moderate Changes Required

The ROI masks must be in the **same space and resolution** as the preprocessed BOLD data. Currently, masks are in SPM's EPI/MNI space at the same voxel grid as `swars*.img` (53 x 63 x 46 at 3mm).

**The problem:** fMRIPrep normalizes to `MNI152NLin2009cAsym` (a different, nonlinear MNI template) and may output at a different resolution (e.g., 2mm isotropic). The current `*_in_EPI_bin.nii.gz` masks will **not align** with fMRIPrep outputs.

**What you need to do:**

1. **Get the Wang et al. (2015) atlas in MNI152NLin2009cAsym space.** Options:
   - Use `templateflow` to get the atlas already in the correct space:
     ```python
     from templateflow import api as tflow
     # Check if Wang atlas is available in this space
     ```
   - OR resample the original atlas masks to the fMRIPrep output space using ANTs or FSL:
     ```bash
     antsApplyTransforms -d 3 \
       -i V1v_original.nii.gz \
       -r sub-01_task-emotion_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz \
       -o V1v_in_fmriprep_space.nii.gz \
       -n NearestNeighbor \
       -t identity  # if both are already in MNI, just resampling
     ```
   - OR use Nilearn's `resample_to_img`:
     ```python
     from nilearn.image import resample_to_img
     resampled_mask = resample_to_img(
         source_img='V1v_original.nii.gz',
         target_img='sub-01_..._desc-preproc_bold.nii.gz',
         interpolation='nearest'  # critical for binary masks
     )
     ```

2. **Update expected dimensions** in `make_roi_masks_mat.m`:
   ```matlab
   % CURRENT:
   expected_shape = [53 63 46];    % SPM 3mm MNI

   % NEW (for 2mm MNI152NLin2009cAsym, typical fMRIPrep output):
   expected_shape = [97 115 97];   % approximate — verify from your actual data
   ```

3. **Ensure the mask voxel grid matches the beta images.** After running Step 1, load one beta image and check its dimensions. The ROI masks must have the exact same shape.

**Changes to `make_roi_masks_mat.m`:**
- Update `roi_root` to point to resampled masks
- Update `expected_shape` to match fMRIPrep output dimensions
- Update `roi_suffix` if the naming convention changes
- The rest of the logic (loading, binarizing, saving) stays the same

---

### Steps 4-5: No Changes

`SingleTrialDecodingv3.m` and `group_level_validation.py` consume `.mat` files (`Pl#.mat`, `Nt#.mat`, `Up#.mat`, `roi_masks.mat`) that are agnostic to the preprocessing tool. As long as Steps 1-3 produce these files in the same format, Steps 4-5 work without modification.

---

## Choosing a Confound/Denoising Strategy

fMRIPrep provides many more confound options than SPM's 6 motion parameters. Common strategies:

| Strategy | Confound Columns | Notes |
|---|---|---|
| **6 motion** (current equivalent) | `trans_x/y/z`, `rot_x/y/z` | Minimal, matches current pipeline |
| **24 motion** | 6 motion + their temporal derivatives + squares of both | More aggressive motion correction |
| **6 motion + aCompCor** | 6 motion + top 5 `a_comp_cor_##` | Removes physiological noise (cardiac, respiratory) |
| **ICA-AROMA** | fMRIPrep can run ICA-AROMA; use `desc-smoothAROMAnonaggr_bold.nii.gz` | Aggressive denoising; already smoothed |
| **Scrubbing** | Use `framewise_displacement` to censor high-motion volumes | Removes contaminated timepoints entirely |

**Recommendation for MVPA replication:** Start with **6 motion parameters** to match the original paper. Then optionally test with 24-motion or aCompCor to see if decoding improves.

---

## Smoothing Considerations for MVPA

The current pipeline applies 8mm FWHM smoothing (done during SPM preprocessing). fMRIPrep does **not** smooth by default.

**Important for MVPA:** Smoothing blurs spatial patterns across voxels. For MVPA, many studies recommend:
- **No smoothing** or **minimal smoothing** (2-4mm FWHM) to preserve fine-grained spatial patterns
- The original paper used 8mm, but this is unusually aggressive for MVPA

**Options:**
1. Apply 8mm smoothing post-fMRIPrep (to match the paper exactly)
2. Apply lighter smoothing (e.g., 4mm) or skip it entirely (may improve decoding)
3. Run both and compare — this is a meaningful methodological extension

---

## Summary: Files You Need from fMRIPrep

For each subject and run, you need these files:

```
derivatives/fmriprep/
  sub-{id}/
    func/
      sub-{id}_task-emotion_run-{r}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz   # preprocessed BOLD
      sub-{id}_task-emotion_run-{r}_desc-confounds_timeseries.tsv                                # confounds
      sub-{id}_task-emotion_run-{r}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz      # brain mask (optional)
```

Plus resampled ROI masks in the fMRIPrep output space:
```
rois_in_fmriprep_space/
  V1v.nii.gz
  V1d.nii.gz
  ...
```

And the existing onset files (unchanged):
```
NewStimuluesSetting/
  Sub##run#.mat
```

---

## Checklist

- [ ] Run fMRIPrep / DeepPrep on raw BIDS data
- [ ] Verify output space and resolution (`space-MNI152NLin2009cAsym`, check `res-` tag)
- [ ] Decide on smoothing strategy (8mm to replicate, or reduce/skip for MVPA)
- [ ] Resample Wang et al. (2015) ROI masks to fMRIPrep output space
- [ ] Verify resampled mask dimensions match BOLD volume dimensions
- [ ] Update `BetaS2.m`:
  - [ ] New file paths (BIDS naming)
  - [ ] Gunzip `.nii.gz` for SPM (or switch to Nilearn)
  - [ ] Handle initial non-steady-state volumes
  - [ ] Load confounds from TSV instead of `rp_*.txt`
  - [ ] Apply smoothing if desired
- [ ] Update `make_roi_masks_mat.m`:
  - [ ] Point to resampled masks
  - [ ] Update `expected_shape`
- [ ] Run Steps 1-3 and verify outputs:
  - [ ] Beta images have correct dimensions
  - [ ] `Pl#.mat`, `Nt#.mat`, `Up#.mat` have expected shape `[nVoxels x 100]`
  - [ ] ROI masks have same number of voxels as beta matrices
- [ ] Run Steps 4-5 (no changes needed)

---

## fMRIPrep vs DeepPrep: Key Differences

Both produce BIDS-derivative outputs and are largely interchangeable from this pipeline's perspective. The main differences:

| Feature | fMRIPrep | DeepPrep |
|---|---|---|
| **Surface recon** | FreeSurfer (slow) | FastSurfer (faster) |
| **Speed** | ~8-16h per subject | ~2-4h per subject |
| **Registration** | ANTs SyN | SynthMorph (DL-based) |
| **Output format** | BIDS derivatives | BIDS derivatives (compatible) |
| **Confounds TSV** | Yes | Yes (same format) |
| **MNI space** | `MNI152NLin2009cAsym` | `MNI152NLin2009cAsym` (same default) |

The script changes described above apply equally to both tools. The only difference would be the exact path layout, which both follow BIDS conventions.
