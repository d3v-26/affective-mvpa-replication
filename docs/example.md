# Worked Example: From BOLD Signal to Decoding Accuracy

A complete numerical walkthrough of the Bo et al. (2021) pipeline. We follow **one subject**, **one run**, **one ROI (V1v with 4 voxels)**, and **6 trials** through every mathematical step — from raw scanner output to a final decoding accuracy.

---

## Setup: What We Start With

### The experiment

Subject S1 views 6 pictures in Run 1 (we shrink from 60 to 6 for clarity):
- 2 Pleasant trials (Pl1, Pl2)
- 2 Neutral trials (Nt1, Nt2)
- 2 Unpleasant trials (Up1, Up2)

Each picture is displayed for 3 seconds, followed by a fixation cross (2.8 or 4.3 s).

### The scanner

- TR = 1.98 s (one whole-brain volume every 1.98 seconds)
- We acquire T = 20 time points (shrunk from 206)
- Each volume has V = 4 voxels in our toy V1v ROI (real V1v has ~500)

### The raw data

After preprocessing (slice timing, realignment, normalization, smoothing), we have the BOLD time series — a matrix of measured signal intensity at each voxel at each time point:

```
y = measured BOLD signal, shape [T × V] = [20 × 4]

         Voxel 1   Voxel 2   Voxel 3   Voxel 4
t=1      1000.2    998.5     1001.3    999.8
t=2      1001.5    999.2     1000.8    1000.1
t=3      1003.8    997.1     1004.2    998.3      ← Pl1 onset (t=3)
t=4      1008.1    995.3     1009.5    996.7      ← HRF rising for Pl1
t=5      1012.4    993.8     1013.1    995.2      ← HRF peak for Pl1
t=6      1009.7    995.1     1010.8    996.5
t=7      1004.2    997.3     1005.1    998.8
t=8      1001.1    999.8     1002.3    1000.2     ← Nt1 onset (t=8)
t=9      1003.5    1003.2    1003.8    1004.1     ← HRF rising for Nt1
t=10     1005.2    1006.8    1005.1    1007.3     ← HRF peak for Nt1
t=11     1003.1    1004.5    1003.9    1005.2
t=12     1000.8    1000.2    1001.1    1000.5
t=13     999.5     1001.8    998.7     1002.1     ← Up1 onset (t=13)
t=14     1006.3    999.2     1010.2    997.8      ← HRF rising for Up1
t=15     1011.8    996.5     1015.3    994.1      ← HRF peak for Up1
t=16     1008.2    997.8     1011.7    996.3
t=17     1003.1    999.5     1004.8    999.1
t=18     1000.5    1000.1    1001.2    1000.3
t=19     1001.2    999.8     1000.5    1000.1
t=20     1000.8    1000.3    1000.9    1000.0
```

Notice the patterns in the raw signal:
- **Pl1** (onset t=3): Voxels 1 and 3 go up, voxels 2 and 4 go down
- **Nt1** (onset t=8): All voxels go up roughly equally
- **Up1** (onset t=13): Voxels 1 and 3 go up strongly, voxels 2 and 4 go down

These differences in spatial pattern are what MVPA will exploit.

### Trial onset times

```
Trial    Condition    Onset (scan #)
Pl1      Pleasant     3
Nt1      Neutral      8
Up1      Unpleasant   13
Pl2      Pleasant     ... (from Run 2 in the full experiment)
Nt2      Neutral      ...
Up2      Unpleasant   ...
```

### Motion parameters (from `rp_*.txt`)

```
         transX   transY   transZ   rotX     rotY     rotZ
t=1      0.012    0.003   -0.005    0.001    0.000    0.002
t=2      0.015    0.005   -0.003    0.001    0.001    0.002
...      ...      ...      ...      ...      ...      ...
t=20     0.021    0.008   -0.001    0.002    0.001    0.003
```

These 6 columns become nuisance regressors in the GLM.

---

## Step 1: Build the Design Matrix X

### 1.1 Create the Hemodynamic Response Function (HRF)

The HRF is the canonical double-gamma function that models how blood flow responds to a brief neural event:

```
HRF(t) = [t^(a1-1) * e^(-t/b1)] / [b1^a1 * Γ(a1)]  -  c * [t^(a2-1) * e^(-t/b2)] / [b2^a2 * Γ(a2)]

where:
  a1 = 6,  b1 = 1    → peak at ~6 seconds
  a2 = 16, b2 = 1    → undershoot at ~16 seconds
  c  = 1/6            → undershoot amplitude ratio
```

Sampled at TR = 1.98 s, the HRF looks like this (arbitrary units):

```
Time (s):   0     1.98   3.96   5.94   7.92   9.90   11.88  13.86  15.84
HRF value:  0.00  0.04   0.28   0.58   0.32   0.08  -0.02  -0.03  -0.01
            ↑                    ↑                           ↑
         stimulus             peak                       undershoot
```

### 1.2 Convolve each trial onset with the HRF

Each trial becomes a **regressor** — a column in the design matrix. We place a boxcar of duration 1.5152 scans (~3 seconds) at the trial's onset time, then convolve with the HRF.

**Regressor for Pl1** (onset at scan 3):

```
Step 1: Boxcar stimulus function s(t)
  s(t) = 1 for t ∈ [3, 4.5152],  0 otherwise

Step 2: Convolve with HRF → x_Pl1(t)
  x_Pl1(t) = (s * HRF)(t) = ∫ s(τ) · HRF(t - τ) dτ

Step 3: Sample at each TR:

Scan:    1     2     3     4     5     6     7     8     ...   20
x_Pl1:   0.00  0.00  0.04  0.28  0.58  0.32  0.08 -0.02  ...  0.00
                      ↑           ↑
                   onset        peak (~5s delay)
```

Similarly for Nt1 (onset scan 8) and Up1 (onset scan 13):

```
Scan:    1     2     3     4     5     6     7     8     9     10    11    12    13    14    15    ...  20
x_Pl1:   0.00  0.00  0.04  0.28  0.58  0.32  0.08 -0.02 -0.03  0.00  0.00  0.00  0.00  0.00  0.00  ... 0.00
x_Nt1:   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.04  0.28  0.58  0.32  0.08 -0.02 -0.03  0.00  ... 0.00
x_Up1:   0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.04  0.28  0.58  ... 0.00
```

### 1.3 Assemble the full design matrix

The complete design matrix X has **one column per regressor** — 3 trial regressors + 6 motion regressors + 1 constant = 10 columns:

```
X = [x_Pl1 | x_Nt1 | x_Up1 | motX | motY | motZ | rotX | rotY | rotZ | constant]

Shape: [T × P] = [20 × 10]

         Pl1    Nt1    Up1    motX   motY   motZ   rotX   rotY   rotZ   const
t=1    [ 0.00   0.00   0.00   0.012  0.003 -0.005  0.001  0.000  0.002  1.0 ]
t=2    [ 0.00   0.00   0.00   0.015  0.005 -0.003  0.001  0.001  0.002  1.0 ]
t=3    [ 0.04   0.00   0.00   0.018  0.004 -0.004  0.001  0.000  0.002  1.0 ]
t=4    [ 0.28   0.00   0.00   0.014  0.006 -0.002  0.001  0.001  0.002  1.0 ]
t=5    [ 0.58   0.00   0.00   0.016  0.005 -0.003  0.002  0.001  0.003  1.0 ]
  .        .      .      .      .      .      .      .      .      .     .
t=10   [ 0.00   0.58   0.00   0.019  0.007 -0.002  0.002  0.001  0.003  1.0 ]
  .        .      .      .      .      .      .      .      .      .     .
t=15   [ 0.00   0.00   0.58   0.020  0.008 -0.001  0.002  0.001  0.003  1.0 ]
  .        .      .      .      .      .      .      .      .      .     .
t=20   [ 0.00   0.00   0.00   0.021  0.008 -0.001  0.002  0.001  0.003  1.0 ]
```

In the real pipeline, there are **60 trial regressors** per run (not 3), so the design matrix is [206 × 66] with 60 condition columns + 6 motion columns. Across 5 sessions modeled jointly, the full matrix is [1030 × 335] (330 regressors + 5 session constants).

---

## Step 2: Solve the GLM — y = Xβ + ε

### 2.1 The model (for one voxel)

For **Voxel 1**, extract its time series as a column vector:

```
y₁ = [1000.2, 1001.5, 1003.8, 1008.1, 1012.4, 1009.7, 1004.2, 1001.1, 1003.5, 1005.2,
      1003.1, 1000.8, 999.5, 1006.3, 1011.8, 1008.2, 1003.1, 1000.5, 1001.2, 1000.8]ᵀ

Shape: [20 × 1]
```

The GLM says:

```
y₁ = X · β₁ + ε₁

where:
  y₁  = [20 × 1]  observed BOLD at voxel 1
  X   = [20 × 10] design matrix (same for all voxels)
  β₁  = [10 × 1]  unknown weights to solve for
  ε₁  = [20 × 1]  residual noise
```

### 2.2 Handle temporal autocorrelation — AR(1) prewhitening

BOLD data is temporally correlated (each time point resembles its neighbors). The AR(1) model estimates this:

```
ε(t) = ρ · ε(t-1) + white_noise(t)
```

SPM estimates ρ from the residuals (typically ρ ≈ 0.3–0.5). It then constructs a prewhitening matrix W such that W·ε has no autocorrelation.

```
         [  1    0    0    0   ...]
         [ -ρ    1    0    0   ...]
W =      [  0   -ρ    1    0   ...]    shape: [20 × 20]
         [  0    0   -ρ    1   ...]
         [ ...                     ]

Transform both sides:
  W·y₁ = W·X · β₁ + W·ε₁
  ỹ₁   = X̃   · β₁ + ε̃₁        where ε̃₁ is now white noise
```

### 2.3 Solve via weighted least squares

```
β̂₁ = (X̃ᵀ X̃)⁻¹ X̃ᵀ ỹ₁

Equivalently (in terms of the original data and autocorrelation matrix V = (WᵀW)⁻¹):

β̂₁ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y₁
```

### 2.4 Numerical result for Voxel 1

After solving, suppose we get:

```
β̂₁ = [β̂_Pl1, β̂_Nt1, β̂_Up1, β̂_motX, β̂_motY, β̂_motZ, β̂_rotX, β̂_rotY, β̂_rotZ, β̂_const]

β̂₁ = [ 2.50,   1.20,   3.10,   0.15,  -0.08,   0.03,   0.01,  -0.02,   0.01,  1000.5 ]
```

**Interpretation:**
- β̂_Pl1 = 2.50 → Voxel 1 responded with amplitude 2.50 (above baseline) to Pleasant picture 1
- β̂_Nt1 = 1.20 → Voxel 1 responded with amplitude 1.20 to Neutral picture 1
- β̂_Up1 = 3.10 → Voxel 1 responded with amplitude 3.10 to Unpleasant picture 1
- β̂_const = 1000.5 → baseline signal level at this voxel
- Motion betas absorb motion-correlated variance → we discard them

### 2.5 Verify the model fit

The predicted signal for Voxel 1:

```
ŷ₁ = X · β̂₁

At t=5 (peak of Pl1 response):
  ŷ₁(5) = 0.58 × 2.50  +  0.00 × 1.20  +  0.00 × 3.10  +  (motion terms)  +  1000.5
         = 1.45 + 0 + 0 + ~0 + 1000.5
         = 1001.95

Residual: ε̂₁(5) = y₁(5) - ŷ₁(5) = 1012.4 - 1001.95 = 10.45
(In reality, the fit would be tighter with more regressors and properly scaled data)
```

### 2.6 Repeat for all 4 voxels

SPM runs the identical regression **independently** at every voxel:

```
Voxel 1:  β̂₁ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y₁  →  β̂₁ = [2.50, 1.20, 3.10, ...]
Voxel 2:  β̂₂ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y₂  →  β̂₂ = [0.50, 1.80, 0.30, ...]
Voxel 3:  β̂₃ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y₃  →  β̂₃ = [2.80, 1.10, 3.50, ...]
Voxel 4:  β̂₄ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y₄  →  β̂₄ = [0.30, 1.90, 0.10, ...]
```

Note: X and V⁻¹ are the same for all voxels (same design, same whitening). Only y changes. The matrix `(Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹` is computed once and applied to every voxel's time series.

---

## Step 3: Extract Single-Trial Betas

### 3.1 What is a beta image?

The beta for regressor Pl1 across all voxels forms a **beta image** — one 3D brain volume (here simplified to 4 voxels):

```
beta_0001.nii (Pl1):  [2.50, 0.50, 2.80, 0.30]   ← one value per voxel
beta_0002.nii (Nt1):  [1.20, 1.80, 1.10, 1.90]
beta_0003.nii (Up1):  [3.10, 0.30, 3.50, 0.10]
```

### 3.2 Organize into condition matrices

Across 5 runs, each condition has 20 trials per run × 5 runs = 100 trials. We organize the betas into matrices:

```
Pl matrix: [nVoxels × nTrials] = [4 × 100]

              Trial 1   Trial 2   Trial 3  ...  Trial 100
              (Run 1)   (Run 1)   (Run 1)       (Run 5)
Voxel 1:    [  2.50      2.35      2.60   ...    2.45   ]
Voxel 2:    [  0.50      0.65      0.40   ...    0.55   ]
Voxel 3:    [  2.80      2.70      2.90   ...    2.75   ]
Voxel 4:    [  0.30      0.40      0.25   ...    0.35   ]

Nt matrix: [4 × 100]

              Trial 1   Trial 2  ...  Trial 100
Voxel 1:    [  1.20      1.15   ...    1.25   ]
Voxel 2:    [  1.80      1.85   ...    1.75   ]
Voxel 3:    [  1.10      1.20   ...    1.15   ]
Voxel 4:    [  1.90      1.85   ...    1.95   ]

Up matrix: [4 × 100]

              Trial 1   Trial 2  ...  Trial 100
Voxel 1:    [  3.10      3.05   ...    3.15   ]
Voxel 2:    [  0.30      0.35   ...    0.25   ]
Voxel 3:    [  3.50      3.40   ...    3.55   ]
Voxel 4:    [  0.10      0.15   ...    0.05   ]
```

These are saved as `Pl1.mat`, `Nt1.mat`, `Up1.mat` for Subject 1.

---

## Step 4: Apply ROI Mask

### 4.1 The ROI mask

The V1v mask from the Wang et al. (2015) atlas is a binary 3D volume (same grid as the betas). Vectorized:

```
roi_masks.V1v = [1, 1, 1, 1]    ← all 4 voxels are inside V1v (in our toy example)
```

In reality, the mask is shape [53 × 63 × 46] = 153,594 voxels, of which ~500 are inside V1v. The mask selects which rows of the Pl/Nt/Up matrices to keep:

```
Pl_roi = Pl(roi_mask, :)   →  [4 × 100]  (all kept in our toy case)
Nt_roi = Nt(roi_mask, :)   →  [4 × 100]
```

---

## Step 5: SVM Classification

### 5.1 Construct the classification problem

**Pleasant vs. Neutral:**

Stack the trial vectors as rows (each trial = one sample, each voxel = one feature):

```
Data matrix:  [nTrials × nVoxels] = [200 × 4]

              Voxel 1   Voxel 2   Voxel 3   Voxel 4     Label
Pl trial 1:  [  2.50      0.50      2.80      0.30  ]     +1
Pl trial 2:  [  2.35      0.65      2.70      0.40  ]     +1
  ...             ...       ...       ...       ...        +1
Pl trial 100:[  2.45      0.55      2.75      0.35  ]     +1
Nt trial 1:  [  1.20      1.80      1.10      1.90  ]     -1
Nt trial 2:  [  1.15      1.85      1.20      1.85  ]     -1
  ...             ...       ...       ...       ...        -1
Nt trial 100:[  1.25      1.75      1.15      1.95  ]     -1
```

### 5.2 Standardize features (z-score each voxel)

For each voxel (column), compute mean and std across all 200 trials, then z-score:

```
Voxel 1:  mean = 1.85,  std = 0.72
  z-scored: (2.50 - 1.85) / 0.72 = 0.90  (Pl trial 1)
            (1.20 - 1.85) / 0.72 = -0.90 (Nt trial 1)

Voxel 2:  mean = 1.15,  std = 0.68
  z-scored: (0.50 - 1.15) / 0.68 = -0.96 (Pl trial 1)
            (1.80 - 1.15) / 0.68 = 0.96  (Nt trial 1)

After standardization:

              Voxel 1   Voxel 2   Voxel 3   Voxel 4     Label
Pl trial 1:  [  0.90     -0.96      0.87     -0.94  ]     +1
Nt trial 1:  [ -0.90      0.96     -0.87      0.94  ]     -1
```

### 5.3 Train the linear SVM

The SVM finds a weight vector **w** and bias *b* that define a decision boundary:

```
f(x) = w · x + b

Optimization problem:
  minimize    ½ ||w||² + C · Σᵢ max(0, 1 - yᵢ · (w · xᵢ + b))
              ↑                    ↑
         margin width         hinge loss (penalty for errors)
```

**Geometric intuition in our 4D voxel space:**

```
                    Voxel 2
                      ↑
                      |     N N
                      |   N N N
                      |  N N
            ----------|--------→ Voxel 1
                      | P P
                      |  P P P
                      |   P P

P = Pleasant trials (high voxel 1 & 3, low voxel 2 & 4)
N = Neutral trials  (moderate on all voxels)

The SVM hyperplane separates the two clusters.
```

After training, suppose the SVM learns:

```
w = [0.45, -0.48, 0.43, -0.47]
b = 0.02
```

**Interpretation of w:**
- w₁ = +0.45 → higher Voxel 1 activation pushes prediction toward Pleasant (+1)
- w₂ = -0.48 → higher Voxel 2 activation pushes prediction toward Neutral (-1)
- w₃ = +0.43 → higher Voxel 3 activation pushes toward Pleasant
- w₄ = -0.47 → higher Voxel 4 activation pushes toward Neutral

The SVM has learned the spatial pattern: **Voxels 1,3 up + Voxels 2,4 down = Pleasant**.

### 5.4 Predict on a test trial

For a held-out Pleasant trial with standardized features x_test = [0.85, -0.90, 0.82, -0.88]:

```
f(x_test) = w · x_test + b
           = (0.45)(0.85) + (-0.48)(-0.90) + (0.43)(0.82) + (-0.47)(-0.88) + 0.02
           = 0.3825 + 0.432 + 0.3526 + 0.4136 + 0.02
           = 1.6007

f(x_test) = 1.60 > 0  →  predict +1 (Pleasant)  ✓ CORRECT
```

For a held-out Neutral trial with x_test = [-0.82, 0.91, -0.85, 0.90]:

```
f(x_test) = (0.45)(-0.82) + (-0.48)(0.91) + (0.43)(-0.85) + (-0.47)(0.90) + 0.02
           = -0.369 + -0.4368 + -0.3655 + -0.423 + 0.02
           = -1.5743

f(x_test) = -1.57 < 0  →  predict -1 (Neutral)  ✓ CORRECT
```

### 5.5 Cross-validation (5-fold, 100 repetitions)

We never evaluate the SVM on data it trained on. Instead:

```
Repetition 1:
  Randomly partition 200 trials into 5 folds of 40 trials each:
    Fold 1: trials {3, 17, 22, 45, ...}    (40 trials)
    Fold 2: trials {1, 8, 29, 51, ...}     (40 trials)
    Fold 3: trials {5, 11, 33, 67, ...}    (40 trials)
    Fold 4: trials {2, 14, 38, 72, ...}    (40 trials)
    Fold 5: trials {7, 19, 41, 88, ...}    (40 trials)

  Iteration 1: Train on Folds {2,3,4,5} (160 trials), Test on Fold 1 (40 trials)
    Predictions:  [+1, -1, +1, +1, -1, -1, +1, -1, ...] (40 predictions)
    True labels:  [+1, -1, +1, -1, -1, -1, +1, -1, ...]
    Correct:      [ ✓,  ✓,  ✓,  ✗,  ✓,  ✓,  ✓,  ✓, ...]
    Fold accuracy: 35/40 = 0.875

  Iteration 2: Train on Folds {1,3,4,5}, Test on Fold 2
    Fold accuracy: 32/40 = 0.800

  Iteration 3: Train on Folds {1,2,4,5}, Test on Fold 3
    Fold accuracy: 34/40 = 0.850

  Iteration 4: Train on Folds {1,2,3,5}, Test on Fold 4
    Fold accuracy: 33/40 = 0.825

  Iteration 5: Train on Folds {1,2,3,4}, Test on Fold 5
    Fold accuracy: 31/40 = 0.775

  Repetition 1 accuracy = mean(0.875, 0.800, 0.850, 0.825, 0.775) = 0.825

Repetition 2:
  New random partition into 5 folds...
  Repetition 2 accuracy = 0.810

...

Repetition 100:
  Repetition 100 accuracy = 0.835
```

**Final decoding accuracy for Subject 1, V1v, Pleasant vs. Neutral:**

```
accuracy = mean(rep₁, rep₂, ..., rep₁₀₀) = mean(0.825, 0.810, ..., 0.835) = 0.623
```

This value — **62.3%** — is what goes into the group analysis. (Chance = 50% for binary classification.)

---

## Step 6: Repeat Across Subjects and ROIs

The entire process (Steps 1–5) is repeated for:
- **All 20 subjects** → 20 accuracy values per ROI
- **All 17 ROIs** (V1v, V1d, V2v, V2d, V3v, V3d, hV4, VO1, VO2, PHC1, PHC2, hMT, LO1, LO2, V3a, V3b, IPS)
- **2 comparisons** (Pleasant vs. Neutral, Unpleasant vs. Neutral)

This produces two matrices:

```
PlNt_acc: [20 subjects × 17 ROIs]

              V1v    V1d    V2v    V2d    V3v    V3d    hV4   ...   IPS
Subject 1:  [ 0.623  0.601  0.645  0.618  0.657  0.634  0.671 ...  0.689 ]
Subject 2:  [ 0.589  0.612  0.624  0.605  0.631  0.648  0.655 ...  0.701 ]
  ...
Subject 20: [ 0.634  0.598  0.651  0.627  0.642  0.619  0.663 ...  0.678 ]


UpNt_acc: [20 subjects × 17 ROIs]

              V1v    V1d    V2v    ...
Subject 1:  [ 0.651  0.638  0.672  ... ]
  ...
```

---

## Step 7: Group-Level Permutation Test

### 7.1 The question

Is the group mean accuracy (e.g., 62.3% for V1v, Pleasant vs. Neutral) significantly above the 50% chance level? A simple t-test might not be appropriate because the accuracy distribution may not be normal. Instead, we use a nonparametric permutation test (Stelzer et al., 2013).

### 7.2 Build the null distribution

**For each of 100,000 permutations:**
1. For each of the 20 subjects, simulate a chance-level accuracy by drawing from a binomial distribution: `acc_chance = Binomial(n=100, p=0.5) / 100`
   - This simulates what accuracy you'd get if the classifier were guessing randomly on 100 test trials (across folds)
2. Average these 20 simulated accuracies to get one group-level chance mean

```
Permutation 1:
  Subject 1 chance accuracy:  Binomial(100, 0.5)/100 = 52/100 = 0.520
  Subject 2 chance accuracy:  Binomial(100, 0.5)/100 = 48/100 = 0.480
  Subject 3 chance accuracy:  Binomial(100, 0.5)/100 = 51/100 = 0.510
  ...
  Subject 20 chance accuracy: Binomial(100, 0.5)/100 = 49/100 = 0.490

  Group mean chance = mean(0.520, 0.480, 0.510, ..., 0.490) = 0.503

Permutation 2:
  Group mean chance = 0.497

...

Permutation 100,000:
  Group mean chance = 0.501
```

This gives us 100,000 values forming the **null distribution** — what group-level accuracy looks like when there is no real signal.

### 7.3 Visualize the null distribution

```
Null Distribution (100,000 group-level chance means)
                                                          Observed = 0.623
                                                               ↓
Count   |
  8000  |          ████
  7000  |        ████████
  6000  |      ████████████
  5000  |    ████████████████
  4000  |  ████████████████████
  3000  | ██████████████████████
  2000  |████████████████████████
  1000  |██████████████████████████                          |
     0  +------+------+------+------+------+------+------+---+--→
        0.46   0.48   0.50   0.52   0.54   0.56   0.58  0.60 0.62

                                     ↑
                               threshold at
                              p < 0.001 ≈ 0.54
```

### 7.4 Compute significance

```
p-value = (# of permuted means ≥ observed mean) / 100,000

Observed mean accuracy for V1v = 0.623
Number of permuted means ≥ 0.623 = 0 out of 100,000

p-value = 0/100,000 = 0.00000 < 0.001  →  SIGNIFICANT ✓
```

The threshold accuracy at p < 0.001 is approximately **54%** — any ROI with group mean above this is declared significant.

### 7.5 One-sample t-test (supplementary)

As a complementary parametric test:

```
H₀: μ = 0.50 (accuracy equals chance)
H₁: μ > 0.50

For V1v, Pleasant vs. Neutral:
  accuracies = [0.623, 0.589, 0.634, ..., 0.634]  (20 values)
  mean = 0.621
  std  = 0.042
  n    = 20

  t = (mean - 0.50) / (std / √n)
    = (0.621 - 0.50) / (0.042 / √20)
    = 0.121 / 0.00939
    = 12.88

  Cohen's d = (mean - 0.50) / std
            = 0.121 / 0.042
            = 2.88

  p < 0.0001  →  SIGNIFICANT ✓
```

---

## Step 8: Final Results Table

```
ROI          Pl vs Nt      p-value     Sig?     Up vs Nt      p-value     Sig?
─────────────────────────────────────────────────────────────────────────────────
V1v          62.1 ± 0.9%   < 0.001     ***      65.3 ± 1.1%   < 0.001     ***
V1d          60.5 ± 1.0%   < 0.001     ***      63.8 ± 0.9%   < 0.001     ***
V2v          63.2 ± 0.8%   < 0.001     ***      66.1 ± 1.0%   < 0.001     ***
V2d          61.8 ± 0.9%   < 0.001     ***      64.5 ± 0.8%   < 0.001     ***
V3v          64.1 ± 1.1%   < 0.001     ***      67.2 ± 0.9%   < 0.001     ***
V3d          63.5 ± 0.8%   < 0.001     ***      65.9 ± 1.0%   < 0.001     ***
hV4          65.8 ± 0.9%   < 0.001     ***      68.4 ± 1.1%   < 0.001     ***
VO1          66.2 ± 1.0%   < 0.001     ***      69.1 ± 0.8%   < 0.001     ***
VO2          67.5 ± 0.8%   < 0.001     ***      70.3 ± 0.9%   < 0.001     ***
PHC1         64.3 ± 1.1%   < 0.001     ***      67.8 ± 1.0%   < 0.001     ***
PHC2         63.9 ± 0.9%   < 0.001     ***      66.5 ± 0.8%   < 0.001     ***
hMT          64.8 ± 1.0%   < 0.001     ***      67.1 ± 1.1%   < 0.001     ***
LO1          65.1 ± 0.8%   < 0.001     ***      68.2 ± 0.9%   < 0.001     ***
LO2          66.3 ± 0.9%   < 0.001     ***      69.5 ± 1.0%   < 0.001     ***
V3a          65.5 ± 1.1%   < 0.001     ***      68.7 ± 0.8%   < 0.001     ***
V3b          64.7 ± 0.8%   < 0.001     ***      67.9 ± 1.1%   < 0.001     ***
IPS          67.1 ± 0.9%   < 0.001     ***      70.8 ± 0.9%   < 0.001     ***
─────────────────────────────────────────────────────────────────────────────────
Chance level: 50%    Significance threshold (p < 0.001): ~54%

Result: ALL 17 retinotopic ROIs show significant above-chance decoding
        for BOTH comparisons, including primary visual cortex V1.
```

(Values above are illustrative to match the paper's finding that all ROIs were significant with accuracies in the ~58–72% range.)

---

## Summary: The Complete Pipeline in One Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  INPUT: Raw BOLD time series y (per voxel) + trial onset times + motion    │
│         [206 timepoints × ~150,000 voxels] per run, 5 runs per subject     │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 1: Build design matrix X                                             │
│                                                                             │
│  For each trial, convolve onset with HRF → one column of X                 │
│  X = [x_Pl1 | x_Pl2 | ... | x_Up20 | motX | motY | motZ | rX | rY | rZ]  │
│  Shape: [206 × 66] per run                                                 │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 2: Solve GLM at each voxel                                           │
│                                                                             │
│  β̂ = (Xᵀ V⁻¹ X)⁻¹ Xᵀ V⁻¹ y                                              │
│                                                                             │
│  → β̂_Pl1 at voxel v = how strongly voxel v responded to Pleasant trial 1  │
│  → 300 beta images per subject (60 trials × 5 runs)                        │
│  → Discard motion betas, keep only condition betas                          │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 3: Organize into condition matrices                                   │
│                                                                             │
│  Pl = [nVoxels × 100]  ← 100 Pleasant trial patterns                      │
│  Nt = [nVoxels × 100]  ← 100 Neutral trial patterns                       │
│  Up = [nVoxels × 100]  ← 100 Unpleasant trial patterns                    │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 4: Apply ROI mask                                                     │
│                                                                             │
│  For V1v (~500 voxels out of ~150,000):                                     │
│  Pl_roi = Pl(mask, :)  →  [500 × 100]                                     │
│  Nt_roi = Nt(mask, :)  →  [500 × 100]                                     │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 5: SVM classification (Pleasant vs. Neutral)                          │
│                                                                             │
│  Data:   [200 trials × 500 voxels]                                          │
│  Labels: [+1, +1, ..., -1, -1, ...]  (100 each)                           │
│                                                                             │
│  Standardize each voxel (z-score across trials)                             │
│  Train linear SVM: f(x) = w · x + b                                        │
│  5-fold CV × 100 repetitions → mean accuracy                               │
│                                                                             │
│  Result: accuracy = 62.3% for this subject, this ROI                        │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 6: Repeat for all 20 subjects × 17 ROIs × 2 comparisons              │
│                                                                             │
│  → PlNt_acc: [20 × 17] matrix                                              │
│  → UpNt_acc: [20 × 17] matrix                                              │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STEP 7: Group-level permutation test                                       │
│                                                                             │
│  Null distribution: 100,000 simulated group means under chance (50%)        │
│  p-value = fraction of null ≥ observed group mean                           │
│  Threshold at p < 0.001 ≈ 54% accuracy                                     │
│                                                                             │
│  Conclusion: ALL 17 ROIs significant → retinotopic visual cortex            │
│              (including V1) encodes emotion-specific patterns               │
└─────────────────────────────────────────────────────────────────────────────┘
```
