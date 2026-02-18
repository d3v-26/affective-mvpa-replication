# Decoding Neural Representations of Affective Scenes in Retinotopic Visual Cortex

A replication and extension of **Bo et al. (2021)** — *"Decoding Neural Representations of Affective Scenes in Retinotopic Visual Cortex"*, published in *Cerebral Cortex*, 31(6), 3047–3063.

This repository implements the fMRI multivariate pattern analysis (MVPA) pipeline described in the paper, from single-trial beta estimation through SVM decoding and group-level permutation testing.

---

## Part 1: The Original Experiment (Bo et al., 2021)

### 1.1 Research Question

Can retinotopic visual cortex (including primary visual cortex V1) encode valence-specific information about emotional scenes? And if so, are these representations shaped by reentrant feedback signals from anterior brain structures like the amygdala?

Prior univariate fMRI studies had failed to find consistent emotion-specific activation in retinotopic visual areas. Bo et al. hypothesized that **multivariate pattern analysis (MVPA)** could reveal emotion-specific multivoxel patterns invisible to univariate methods.

### 1.2 Participants

- 26 healthy volunteers recruited; 6 excluded (2 withdrew, 4 excessive motion)
- **20 subjects analyzed** (10 women; mean age 20.4 ± 3.1 years)

### 1.3 Stimuli

60 grayscale pictures from the **International Affective Picture System (IAPS)**, divided into three categories of 20 each:

| Category      | Content                                    | Valence (1–9) | Arousal (1–9) |
|---------------|-------------------------------------------|---------------|---------------|
| **Pleasant**  | Sports scenes, romance, erotic couples     | 7.0 ± 0.45    | 5.8 ± 0.90    |
| **Neutral**   | Landscapes, neutral humans                 | 6.3 ± 0.99    | 4.2 ± 0.97    |
| **Unpleasant**| Threat/attack scenes, bodily mutilations   | 2.8 ± 0.88    | 6.2 ± 0.79    |

Pictures were matched across categories for composition, complexity, entropy (F=1.05, P=0.35), and pixel contrast (F=0.52, P=0.60).

### 1.4 Experimental Paradigm

```
[Picture 3s] → [Fixation 2.8 or 4.3s] → [Picture 3s] → [Fixation] → ...
```

- **5 sessions** (runs), each ~7 minutes
- Each session presented all 60 pictures (20 pleasant + 20 neutral + 20 unpleasant)
- Presentation order randomized across sessions
- Participants maintained central fixation throughout
- Same 60 pictures repeated across all 5 sessions

### 1.5 Data Acquisition

**fMRI** (3T Philips Achieva):
- TR = 1.98 s, TE = 30 ms, flip angle = 80°
- 36 slices, FOV = 224 mm, voxel size = 3.5 × 3.5 × 3.5 mm, matrix = 64 × 64
- Ascending acquisition, AC-PC aligned

**EEG** (simultaneous, 32-channel MR-compatible):
- Used to extract Late Positive Potential (LPP) as an index of reentrant processing
- *Note: Our replication focuses on the fMRI pipeline only*

### 1.6 fMRI Preprocessing (SPM)

The paper's preprocessing pipeline, in order:

1. **Discard** first 5 volumes per session (scanner transient instability)
2. **Slice timing correction** — interpolation to account for acquisition time differences
3. **Realignment** — spatial alignment to 6th image of each session (produces `rp_*.txt` motion parameters)
4. **Normalization** — registered to MNI template, resampled to 3 × 3 × 3 mm
5. **Smoothing** — Gaussian filter, FWHM = 8 mm
6. **High-pass filter** — cutoff = 1/128 Hz to remove low-frequency drift

The preprocessed output naming convention is `swars*.img` where:
- `s` = smoothed
- `wa` = normalized (warped)
- `r` = realigned
- `s` = slice-timing corrected (inner)

### 1.7 Single-Trial Beta Estimation (Beta Series Method)

This is the critical bridge between raw fMRI and MVPA. The paper uses the **beta series method** (Mumford et al., 2012):

**Concept:** Instead of a single regressor per condition (standard GLM), each individual trial gets its own regressor. This yields one beta image per trial per voxel — a single-trial BOLD activation map.

**How it works:**
1. For each trial of interest, create a GLM with:
   - **One regressor** for that specific trial
   - **One regressor** grouping all other trials
   - **Six motion regressors** (translation X,Y,Z + rotation x,y,z)
2. Estimate the model → the beta weight for the trial-of-interest regressor is that trial's activation pattern
3. Repeat for every trial across all sessions

**Result:** For each subject, 300 beta images total (60 trials × 5 sessions), each capturing the spatial pattern of BOLD response to a single picture presentation. These 300 maps are the input to MVPA.

### 1.8 ROI Definition

ROIs defined from the **Wang et al. (2015) probabilistic retinotopic atlas** — 17 ROIs:

| Region Group | ROIs |
|---|---|
| **Early Visual Cortex (EVC)** | V1v, V1d, V2v, V2d, V3v, V3d |
| **Ventral Visual Cortex (VVC)** | hV4, VO1, VO2, PHC1, PHC2 |
| **Dorsal Visual Cortex (DVC)** | V3a, V3b, hMT, LO1, LO2, IPS (IPS0–IPS5 combined) |

Homologous regions from both hemispheres were combined.

### 1.9 MVPA Decoding

**Classifier:** Linear SVM (LibSVM)

**Comparisons (pairwise binary classification):**
1. **Pleasant vs. Neutral** — tests if pleasant scenes have distinct representations
2. **Unpleasant vs. Neutral** — tests if unpleasant scenes have distinct representations
3. **Pleasant vs. Unpleasant** — controls for arousal (matched arousal between these categories)

**Cross-validation procedure:**
1. Pool all single-trial betas across 5 sessions (100 trials per condition: 20 trials × 5 sessions)
2. Apply 10-fold cross-validation: train on 9 folds, test on 1 fold
3. Repeat the 10-fold partition **100 times** with different random splits
4. Average all accuracies → one decoding accuracy per subject per ROI

### 1.10 Statistical Testing (Group Level)

**Permutation test** (Stelzer et al., 2013) to determine if decoding accuracy is above chance:

1. **Subject level:** Randomly shuffle class labels 100 times → 100 chance-level accuracies per subject
2. **Group level:** Randomly draw one chance accuracy from each subject, average across subjects. Repeat **10^5 (100,000) times** → empirical null distribution
3. **Threshold:** The accuracy at **p < 0.001** in the null distribution determines significance (found to be ~54%)

### 1.11 Key Findings

1. Decoding accuracy was **significantly above chance in all 17 retinotopic ROIs** for both pleasant vs. neutral and unpleasant vs. neutral — including primary visual cortex V1
2. **Ventral visual cortex (VVC)** decoding accuracy correlated with LPP amplitude (R=0.69 for unpleasant, R=0.50 for pleasant), supporting reentrant feedback
3. **Amygdala→VVC effective connectivity** predicted unpleasant vs. neutral decoding accuracy (r=0.66, p=0.001)
4. **Frontal regions (IFG, VLPFC)→VVC** connectivity predicted pleasant vs. neutral decoding accuracy
5. Univariate analysis missed many of these effects — MVPA revealed emotion signals in areas where standard activation maps showed nothing

---

## Part 2: Technical Deep Dive — From Raw BOLD Signal to Decoding Accuracy

This section explains the mathematics and logic behind every step of the pipeline, answering: what is a beta, how is it generated, why does the hemodynamic response matter, how does head motion get accounted for, why is there no baseline comparison, and how does SVM use voxel patterns to decode emotion.

### 2.1 The Fundamental Problem: fMRI Measures Blood, Not Neurons

An fMRI scanner does not directly measure neural activity. It measures the **Blood Oxygen Level-Dependent (BOLD) signal** — a proxy based on the fact that active neurons consume oxygen, which changes the local ratio of oxygenated to deoxygenated hemoglobin, which in turn changes the magnetic resonance signal.

The raw data is a 4D volume: a 3D brain image (many voxels) acquired at each time point (TR = 1.98 s). For our data, each run produces **206 volumes**, each volume being a 3D grid of voxels. The challenge is: given this noisy, indirect time series at each voxel, how do we estimate the brain's response to each individual picture?

### 2.2 The General Linear Model (GLM): y = Xβ + ε

The GLM is the mathematical framework that links experimental events to the observed BOLD signal. For a **single voxel**, the model is:

```
y = Xβ + ε
```

where:
- **y** is a column vector of length T (the number of time points, T=206 per run). This is the measured BOLD time series at that voxel.
- **X** is the **design matrix** of size T × P, where P is the number of regressors (predictors). Each column of X represents one regressor — a predicted time course.
- **β** is a column vector of length P — the **beta weights** (regression coefficients). These are what we solve for. Each β_i tells us how much regressor i contributed to the observed signal at this voxel.
- **ε** is the residual error vector (T × 1) — the part of the signal the model cannot explain (noise, physiological artifacts, etc.)

SPM solves for β using **weighted least squares**:

```
β̂ = (X'V⁻¹X)⁻¹ X'V⁻¹y
```

where V is the temporal autocorrelation matrix (estimated via the AR(1) model — see Section 2.5). In the simpler ordinary least squares case (no autocorrelation), this reduces to:

```
β̂ = (X'X)⁻¹ X'y
```

This is just multiple linear regression. Each β̂_i is the best-fit scaling factor for regressor i that minimizes the squared error between the predicted signal (Xβ) and the observed signal (y). **This is computed independently at every voxel in the brain**, producing a whole-brain β̂ map (a 3D image) for each regressor.

### 2.3 Standard GLM vs. Beta Series: Why We Need Single-Trial Betas

#### Standard GLM (not what we use)

In a typical fMRI activation study, you model all trials of the same condition with a **single regressor**:

```
Design matrix X has columns:
  [Pleasant_all | Neutral_all | Unpleasant_all | Motion1..6 | Constant]
```

Here, `Pleasant_all` is one regressor whose predicted time course has a bump at every pleasant trial onset. Solving the GLM gives you **one β per condition per voxel** — the average BOLD amplitude for that condition. This is used for univariate analysis (e.g., "is β_pleasant > β_neutral at this voxel?").

**Why this fails for MVPA:** MVPA needs many individual data points (samples) to train a classifier. The standard GLM gives only **one sample per condition per run** — far too few. With 5 runs and 2 conditions, you would have only 10 data points to train an SVM. That is not enough.

#### Beta Series Method (what we use — Mumford et al., 2012)

The beta series approach gives each trial its **own regressor**. In our implementation (`BetaS2.m`), a single run's design matrix has **66 columns**:

```
Design matrix X for one session (T=206 rows × 66 columns):

  [Pl1 | Pl2 | ... | Pl20 | Nt1 | Nt2 | ... | Nt20 | Up1 | Up2 | ... | Up20 | MotX | MotY | MotZ | Rotx | Roty | Rotz]
   ←— 20 Pleasant ——→  ←— 20 Neutral ——→   ←— 20 Unpleasant —→  ←——— 6 motion ———→
```

Each condition column (e.g., `Pl1`) is a predicted time course with a bump **only at that trial's onset**, convolved with the HRF. When SPM solves β̂ = (X'V⁻¹X)⁻¹ X'V⁻¹y, the resulting β̂_Pl1 captures how strongly that specific voxel responded to Pleasant picture #1 in that run, after accounting for every other trial and the motion regressors.

With 5 sessions modeled together, there are 60 condition regressors × 5 sessions + 6 motion × 5 sessions = **330 regressors** (plus session constants). This produces 300 condition beta images — **one per trial** — which become the samples for MVPA.

#### Why the beta series works mathematically

Consider the design matrix column for trial Pl1 in run 1. It looks like a vector of mostly zeros with a single HRF-shaped bump at the time of that trial's onset. Because the other 59 trials in that run each have their own separate regressors, the regression can isolate the BOLD variance unique to Pl1. The key insight: by giving every trial its own regressor, the GLM effectively "deconvolves" the overlapping BOLD responses from temporally adjacent trials, attributing the signal at each time point to the correct trial.

### 2.4 The Hemodynamic Response Function (HRF): Why It Matters

Neural activity is fast (milliseconds), but the BOLD response is slow (seconds). When neurons fire, the hemodynamic response peaks roughly **5–6 seconds later** and takes ~15–20 seconds to return to baseline. The HRF is the shape of this sluggish blood-flow response to a brief neural event.

```
Neural event (stimulus at t=0):     __|__
                                       ↓
BOLD response:          ____        ___/  \___
                            \      /          \____/  (undershoot)
                             \____/
                        0   2   4   6   8   10  12  14  16 sec
```

**Why it matters for the GLM:** Each regressor in X is not just a spike at the trial onset — it is the onset convolved with the HRF. This is how the model predicts what the BOLD signal *should* look like if that trial activated the voxel. Without HRF convolution, the model would try to find signal at the moment of stimulus onset, but the actual BOLD peak is delayed by ~5 seconds. The convolution aligns the model to the hemodynamics.

In `BetaS2.m`, the HRF is configured as:
```matlab
SPM.xBF.name = 'hrf';          % Canonical HRF (double-gamma function)
SPM.xBF.length = 32.0513;      % Model the response for ~32 seconds
SPM.xBF.order = 1;             % Just the canonical shape (no derivatives)
SPM.xBF.T = 36;                % 36 microtime bins per TR for upsampling
SPM.xBF.T0 = 18;               % Onset aligned to middle of TR
SPM.xBF.UNITS = 'scans';       % Onset times are in scan units
SPM.xBF.Volterra = 1;          % No interaction terms
```

The stimulus duration is set to **1.5152 scan units** (= 1.5152 × 1.98s ≈ 3.0s, matching the 3-second picture display). SPM convolves a boxcar of that duration with the HRF to create each regressor's predicted time course. The duration tells the model how long the neural event lasted, which affects the shape and amplitude of the predicted BOLD response.

### 2.5 How Head Motion Is Accounted For

Head motion is one of the largest sources of artifact in fMRI. Even sub-millimeter movement can shift voxels relative to brain anatomy, creating spurious signal changes. Motion is handled at **two stages**:

#### Stage 1: Realignment (during preprocessing)

Before any analysis, SPM's realignment step physically corrects each volume by estimating 6 rigid-body parameters — 3 translations (X, Y, Z in mm) and 3 rotations (pitch, roll, yaw in radians) — relative to a reference image. Each volume is spatially transformed to undo the estimated movement. This produces the `r` prefix in `swars*.img` and the `rp_*.txt` files containing the motion parameters.

**However, realignment is imperfect.** It corrects for bulk motion but cannot fix:
- Signal changes caused by the movement itself (spin-history effects)
- Interactions between motion and magnetic field inhomogeneities
- Residual misalignment at sub-voxel scales

#### Stage 2: Motion regressors in the GLM (our pipeline)

To handle residual motion effects, the 6 motion parameter time courses are included as **nuisance regressors** in the design matrix:

```matlab
SPM.Sess(j).C.C = [r1 r2 r3 r4 r5 r6];   % 6 motion columns added to X
SPM.Sess(j).C.name = {'X','Y','Z','x','y','z'};
```

These 6 columns are added to the design matrix X alongside the 60 condition regressors. When SPM solves for β̂, any variance in the BOLD signal that correlates with head motion gets absorbed by the motion regressors' β weights, **not** by the condition betas. This "regresses out" motion-related signal from the trial-specific estimates.

Mathematically, the full model for one session is:

```
y(t) = β₁·HRF_Pl1(t) + β₂·HRF_Pl2(t) + ... + β₆₀·HRF_Up20(t)
       + β₆₁·MotionX(t) + β₆₂·MotionY(t) + ... + β₆₆·MotionZ_rot(t)
       + ε(t)
```

The motion betas (β₆₁–β₆₆) act as a "sponge" for motion-correlated variance. After estimation, we discard these betas and keep only β₁–β₆₀ (the trial betas) for MVPA. The trial betas are therefore **motion-corrected** — they reflect neural activation with motion contamination removed to the extent that a linear model allows.

### 2.6 Temporal Autocorrelation and the AR(1) Model

BOLD data has **temporal autocorrelation** — each time point is correlated with its neighbors because the hemodynamic response is smooth and slow. Standard OLS regression assumes independent errors, which is violated here. If ignored, the estimated β weights would still be unbiased, but their **standard errors** would be wrong (usually too small), and the model fit would be inefficient.

`BetaS2.m` handles this with:
```matlab
SPM.xVi.form = 'AR(1)';   % First-order autoregressive noise model
```

This tells SPM to estimate a first-order autoregressive coefficient (ρ) from the residuals: `ε(t) = ρ·ε(t-1) + white_noise(t)`. SPM then "prewhitens" both y and X — transforming them so that the effective noise becomes uncorrelated — before re-estimating β. The result is the weighted least squares solution β̂ = (X'V⁻¹X)⁻¹ X'V⁻¹y, where V encodes the AR(1) autocorrelation structure. This gives more efficient (lower variance) beta estimates.

### 2.7 High-Pass Filtering and Global Scaling

#### High-pass filter (128 s cutoff)

fMRI data contains slow signal drifts caused by scanner heating, physiological cycles (breathing, cardiac pulsation), and subject alertness changes. These drifts are low-frequency noise that can confound the beta estimates if a trial's onset happens to coincide with a drift.

```matlab
SPM.xX.K(j).HParam = 128;   % Remove frequencies below 1/128 Hz
```

SPM implements this by adding **discrete cosine basis functions** to the design matrix as additional nuisance regressors, effectively projecting out any signal component with period >128 seconds. This is applied to both the data y and the design matrix X simultaneously, ensuring the beta estimates are not affected by slow drifts.

#### Global scaling

```matlab
SPM.xGX.iGXcalc = 'Scaling';
```

Global scaling normalizes each volume's overall intensity to a common mean, removing global signal fluctuations (e.g., from breathing or scanner drift) that affect all voxels uniformly. Each time point's data is divided by its global mean and multiplied by a grand mean. This makes beta values comparable across subjects and sessions.

### 2.8 Why There Is No Baseline Comparison (And Why It Doesn't Matter)

In many fMRI studies, researchers compare each condition to a "rest" or "fixation" baseline: *"Is this voxel more active during Pleasant pictures than during fixation?"* This yields an activation map.

**MVPA does not need a baseline comparison.** Here's why:

#### What MVPA asks vs. what univariate analysis asks

| | Univariate | Multivariate (MVPA) |
|---|---|---|
| **Question** | "Is this voxel *more active* for condition A than B?" | "Is the *pattern across many voxels* different for A vs B?" |
| **Unit of analysis** | Single voxel amplitude | Vector of amplitudes across voxels |
| **Needs baseline?** | Yes — to define "activation" | No — only needs to distinguish two patterns |

In MVPA, we never ask "is voxel 4523 *activated* by pleasant pictures?" We ask: "given the full pattern of beta values across all voxels in V1, can a classifier learn to distinguish pleasant trials from neutral trials?" The classifier operates on **relative differences within the pattern**, not on absolute activation levels.

#### What the beta represents without a baseline

Each single-trial beta (β̂_i) represents the estimated BOLD amplitude for that trial at that voxel, **relative to the implicit baseline in the GLM**. The implicit baseline is everything not modeled by any regressor — primarily the inter-trial fixation intervals. So β̂ = 2.5 at some voxel means "this voxel's BOLD signal during this trial was 2.5 units above the fixation/rest level."

But for MVPA, the absolute value doesn't matter. What matters is the **pattern**:
- Trial Pl1 might produce pattern [2.5, 1.3, 0.8, 3.1, ...] across voxels in V1
- Trial Nt1 might produce pattern [1.8, 1.9, 2.1, 0.9, ...] across those same voxels

Even if both are above baseline, their *shapes* differ — and that's what the SVM learns. The fixation baseline is implicitly absorbed by the GLM's constant term and never enters the classification.

#### Why baseline subtraction would actually be harmful

If we subtracted a "baseline beta" from each condition, we would remove shared variance across voxels. But some of that shared variance carries information about the spatial pattern. MVPA thrives on the full, rich pattern — including both condition-specific and condition-general components — because the SVM can learn which dimensions of the pattern are discriminative.

### 2.9 What Exactly Is a "Beta Image"? Concrete Walkthrough

Let's trace the life of a single beta — say, `beta_0003.nii` for Subject 1, which corresponds to Pleasant picture #3 in Run 1.

**Step 1:** Subject 1 sees Pleasant picture #3 at scan number 45 (approximately) in Run 1.

**Step 2:** The HRF-convolved regressor for this trial is a vector of 206 values, mostly near zero except for a bump spanning roughly scans 45–55 (the ~5s HRF delay means the predicted BOLD peaks around scan 48).

**Step 3:** This regressor becomes column 3 of the design matrix X. The other 65 columns (59 other trials + 6 motion regressors) are also present.

**Step 4:** At voxel (27, 31, 22), the time series y = [100.2, 99.8, 101.5, ...] (206 values). SPM solves:
```
β̂ = (X'V⁻¹X)⁻¹ X'V⁻¹y
```
and finds that β̂₃ = 1.87 for this voxel. This means: after accounting for all other trials and motion, voxel (27,31,22) responded with amplitude 1.87 to Pleasant picture #3.

**Step 5:** SPM repeats this for every voxel in the brain. The collection of all these β̂₃ values across voxels forms the 3D image `beta_0003.nii`.

**Step 6:** `extract_betas.m` reads `beta_0003.nii`, flattens it to a 1D vector of ~153,594 voxel values, and stores it as the 3rd column of the `Pl` matrix.

**Result:** The `Pl` matrix for Subject 1 has shape [153,594 voxels × 100 trials]. Each column is one trial's whole-brain activation pattern. Each value is one voxel's estimated BOLD amplitude for that trial.

### 2.10 Univariate Pattern Analysis (UPA) vs. Multivariate Pattern Analysis (MVPA)

This distinction is central to the paper's contribution. Both start from the same beta images, but they ask fundamentally different questions.

#### Univariate Pattern Analysis (UPA)

Univariate analysis examines **one voxel at a time**:

```
For each voxel v:
    Average β across all Pleasant trials  →  mean_Pl(v)
    Average β across all Neutral trials   →  mean_Nt(v)
    Test: is mean_Pl(v) > mean_Nt(v)?     →  t-statistic, p-value
```

This produces a statistical map (SPM{t}) where each voxel is colored by how strongly it differentiates the conditions. Voxels passing a threshold (e.g., p < 0.05, FDR-corrected) are declared "activated."

**Key limitation:** A voxel is only detected if it *consistently* responds more to one condition than another *across subjects*. But emotional processing might activate different sub-patterns within V1 in different people. If subject A has higher V1 activity in the upper-left quadrant for pleasant images while subject B has it in the lower-right, the average cancels out and univariate analysis declares V1 uninvolved.

This is exactly what happened in the paper: univariate analysis found emotion effects in only 6 of 17 ROIs for pleasant vs. neutral, and 11 of 17 for unpleasant vs. neutral.

#### Multivariate Pattern Analysis (MVPA)

MVPA examines the **joint pattern across many voxels simultaneously**:

```
For each subject, for each ROI:
    Collect all single-trial beta patterns:
        Pleasant trials:    each is a vector of [V voxels]  →  100 such vectors
        Neutral trials:     each is a vector of [V voxels]  →  100 such vectors
    Train a classifier to distinguish Pleasant from Neutral patterns
    Test accuracy via cross-validation
```

The critical difference: MVPA does not require voxels to individually show a consistent direction of effect. It can exploit **distributed patterns** where some voxels go up, some go down, and the informative signal is in the combinatorial pattern across voxels.

**Visual comparison for a hypothetical 4-voxel ROI:**

```
                  Voxel A   Voxel B   Voxel C   Voxel D
Pleasant trial 1:   2.1       0.5       1.8       0.3
Pleasant trial 2:   1.9       0.7       1.6       0.4
Neutral trial 1:    0.8       1.7       0.4       1.9
Neutral trial 2:    0.6       1.5       0.5       1.8

Univariate test per voxel:
  Voxel A: Pl > Nt ✓    Voxel B: Nt > Pl    Voxel C: Pl > Nt ✓    Voxel D: Nt > Pl
  After averaging across subjects (with individual differences), possibly none survive

MVPA sees:
  Pleasant pattern ≈ [high, low, high, low]
  Neutral pattern  ≈ [low, high, low, high]
  → Clearly separable! The PATTERN is consistently different even though individual voxels
    might not show consistent effects across subjects.
```

MVPA found that **all 17 retinotopic ROIs** — including V1 — contained emotion-discriminative patterns, where univariate analysis had missed many of them.

### 2.11 How the SVM Uses Beta Patterns to Decode Emotion

#### The data format entering the SVM

After beta extraction and ROI masking, each trial becomes a **feature vector**. For a given subject and ROI:

```
Training data X_train: [n_trials × n_voxels] matrix
Labels y_train:        [n_trials × 1] vector of +1 (Pleasant) or -1 (Neutral)
```

For example, in V1v with ~500 voxels and 200 trials (100 Pleasant + 100 Neutral):
- X_train is [200 × 500] — each row is one trial's spatial pattern across V1v voxels
- y_train is [200 × 1] — the class label for each trial

#### What the linear SVM learns

A linear SVM finds a **hyperplane** in voxel space that best separates the two classes. Mathematically, it learns a weight vector **w** (length = n_voxels) and a bias term *b* such that:

```
Decision:  f(x) = w · x + b

If f(x) > 0  →  predict Pleasant (+1)
If f(x) < 0  →  predict Neutral (-1)
```

where **x** is a trial's beta pattern vector and **w · x** is the dot product (sum of element-wise products).

The SVM finds w and b by solving:

```
minimize    ½ ||w||² + C Σ max(0, 1 - yᵢ(w · xᵢ + b))

where:
  - ½||w||² encourages a wide margin between classes
  - The sum is the "hinge loss" — penalty for misclassified or margin-violating samples
  - C is the regularization parameter (default in fitcsvm)
```

The solution depends only on the **support vectors** — the training samples closest to the decision boundary. The weight vector w is a linear combination of these support vectors:

```
w = Σ αᵢ yᵢ xᵢ    (sum over support vectors)
```

Each element of **w** tells us how much that voxel contributes to the classification. Voxels with large positive w_j push the decision toward Pleasant; voxels with large negative w_j push toward Neutral.

#### Standardization

Before training, `fitcsvm` with `'Standardize', true` z-scores each voxel (feature) across trials:

```
x_voxel_j_standardized = (x_voxel_j - mean_j) / std_j
```

This prevents voxels with large raw beta variance from dominating the classifier. After standardization, every voxel contributes on an equal footing, and the SVM can focus on *which voxels carry discriminative information* rather than *which voxels have the largest variance*.

#### Cross-validation: measuring generalization

A high training accuracy is meaningless — the SVM might be memorizing noise. Cross-validation estimates how well the classifier **generalizes** to new, unseen trials:

```
For each of 100 repetitions:
    Randomly partition 200 trials into 5 folds (40 trials each)
    For each fold f:
        Train SVM on folds 1..5 \ {f}  (160 trials)
        Test on fold f                  (40 trials)
        fold_accuracy = correct_predictions / 40
    repetition_accuracy = mean of 5 fold accuracies

Final accuracy = mean of 100 repetition accuracies
```

Chance level is 50% (two classes, balanced). Accuracy significantly above 50% means the ROI contains information that distinguishes the two conditions. The paper found the significance threshold to be ~54% at p < 0.001 via permutation testing.

#### Connecting it all together: from stimulus to accuracy

```
IAPS picture displayed (3s)
        ↓
Neural response in visual cortex (milliseconds)
        ↓
Hemodynamic response: blood flow changes (peaks ~5s later)
        ↓
BOLD signal captured by fMRI scanner (TR = 1.98s sampling)
        ↓
Preprocessing: correct for slice timing, motion, normalize, smooth
        ↓  [swars*.img files]
GLM with single-trial regressors (HRF-convolved) + motion regressors
        ↓  [beta_####.nii files — one per trial]
Extract betas, vectorize, organize by condition
        ↓  [Pl.mat, Nt.mat, Up.mat — voxels × trials matrices]
Apply ROI mask → select subset of voxels
        ↓  [e.g., 500 V1v voxels × 200 trials]
SVM classification with cross-validation
        ↓
Decoding accuracy per subject per ROI (e.g., 62.3%)
        ↓
Group-level permutation test: is 62.3% > chance?
        ↓
Conclusion: V1v carries emotion-discriminative information (p < 0.001)
```

### 2.12 Why MVPA Reveals What Univariate Analysis Misses

The paper's most striking result is the discrepancy:

| Method | ROIs with significant emotion effects |
|---|---|
| **Univariate** (Pleasant vs Neutral) | 6 / 17 ROIs (V3d, hMT, LO1, LO2, V3a, IPS) |
| **Univariate** (Unpleasant vs Neutral) | 11 / 17 ROIs |
| **MVPA** (both comparisons) | **17 / 17 ROIs** — all retinotopic areas including V1 |

Why does MVPA succeed where univariate fails?

1. **Individual differences in spatial patterns.** Subject A's V1 might encode "pleasant" in a different sub-pattern of voxels than Subject B. Univariate analysis averages across subjects at each voxel, canceling out these idiosyncratic patterns. MVPA decodes each subject separately and then averages accuracies — preserving subject-specific information.

2. **Distributed, sub-threshold codes.** No single voxel in V1 may robustly distinguish pleasant from neutral. But *collectively*, many voxels each carrying a tiny signal can form a pattern that a classifier can learn. MVPA exploits these weak, distributed signals that fall below the univariate detection threshold.

3. **Canceling activations.** If some V1 voxels increase and others decrease for pleasant stimuli, the mean activation might be zero (no univariate effect). But the pattern of increases and decreases is itself informative — MVPA captures this.

This is why the paper's finding that "all 17 retinotopic ROIs, including V1, contain emotion-specific patterns" was a significant advance: it demonstrated that emotional content is represented even in the earliest stages of the visual hierarchy, contradicting prior univariate null results.

---

## Part 3: Our Replication Pipeline

### 3.1 Available Data

We begin with:
- **20 subjects** of SPM-preprocessed fMRI data (files named `swars*.img`)
- **5 runs per subject**, each with 206 scans (after discarding initial volumes)
- **Onset timing files** (`Sub##run#.mat`) containing trial onset times for all 60 conditions per run
- **Motion parameters** (`rp_*.txt`) from the realignment step
- **ROI masks** in EPI space (`*_in_EPI_bin.nii.gz`), derived from the Wang et al. (2015) atlas

The preprocessed `swars*` files have already undergone slice timing → realignment → normalization → smoothing, matching Steps 1–5 of Section 1.6.

### 3.2 Pipeline Overview

```
Step 1                    Step 2                  Step 3              Step 4                Step 5
BetaS2.m          →   extract_betas.m    →  make_roi_masks_mat.m  →  SingleTrialDecodingv3.m  →  group_level_validation.py
(SPM GLM:              (Extract beta          (Build ROI mask        (SVM decoding per         (Permutation test,
 single-trial           images into            struct from            ROI: Pl vs Nt,            t-tests, group
 beta estimation)       .mat matrices)         NIfTI files)           Up vs Nt)                 statistics)
```

### 3.3 Step 1 — Single-Trial Beta Estimation (`BetaS2.m`)

**What it does:** Implements the beta series method (Section 1.7) using SPM's GLM framework.

**Relation to paper:** This is the "Single-Trial Estimation of fMRI-BOLD" step (Mumford et al., 2012). Each individual picture presentation gets its own regressor so that MVPA can decode single-trial activation patterns. See Section 2.3 for the mathematical rationale.

**How it works in detail:**

For each subject (1–20), the script builds an SPM design matrix with 5 sessions. Within each session, it creates **60 condition regressors** — one per trial:

| Regressors 1–20  | Regressors 21–40  | Regressors 41–60  | Regressors 61–66   |
|-------------------|-------------------|-------------------|--------------------|
| Pl1, Pl2, ..., Pl20 | Nt1, Nt2, ..., Nt20 | Up1, Up2, ..., Up20 | Motion (X,Y,Z,x,y,z) |

Each condition regressor is convolved with the canonical hemodynamic response function (HRF) with these parameters:
- HRF model: `'hrf'` (canonical)
- Stimulus duration: **1.5152 seconds** (scan-time units)
- HRF length: 32.0513 s
- Microtime resolution: T=36 bins, onset at T0=18
- Units: scans

Additional GLM settings:
- High-pass filter: 128 s cutoff
- Autocorrelation: AR(1) model
- Global normalization: scaling

After estimation (`spm_spm`), SPM writes one **beta image per regressor** to disk. Across 5 sessions with 66 regressors each, this produces 330 beta images per subject (plus session constants):

```
beta_0001.nii ... beta_0066.nii   → Session 1 (60 conditions + 6 motion)
beta_0067.nii ... beta_0132.nii   → Session 2
beta_0133.nii ... beta_0198.nii   → Session 3
beta_0199.nii ... beta_0264.nii   → Session 4
beta_0265.nii ... beta_0330.nii   → Session 5
```

**Output:** `beta_series_#/beta_####.nii` (or `.img`) per subject, stored under the betas directory.

**Run command:**
```bash
sbatch pipeline/step1_beta_estimation/run_betas2_m.sbatch   # 24h walltime, 30GB RAM
```

### 3.4 Step 2 — Beta Extraction (`extract_betas.m`)

**What it does:** Reads the SPM beta images from Step 1 and organizes them into per-condition MATLAB matrices for efficient loading during decoding.

**Relation to paper:** This is a data-wrangling step that converts SPM's beta image format into the `[nVoxels × nTrials]` matrix format needed for SVM classification.

**How it works:**

For each subject, the script loops through 5 runs and uses the known regressor ordering (66 per run) to extract condition-specific betas:

```
Run r, offset = (r-1) × 66:
  Pleasant  → beta images at offset + [1:20]
  Neutral   → beta images at offset + [21:40]
  Unpleasant→ beta images at offset + [41:60]
  (Motion regressors at offset + [61:66] are skipped)
```

Each beta image is read via `spm_read_vols`, vectorized (3D volume → 1D column), and concatenated across runs.

**Output per subject:**
- `Pl#.mat` — Pleasant betas, shape `[nVoxels × 100]` (20 trials × 5 runs)
- `Nt#.mat` — Neutral betas, shape `[nVoxels × 100]`
- `Up#.mat` — Unpleasant betas, shape `[nVoxels × 100]`

**Run:** Interactively in MATLAB (requires SPM on path).

### 3.5 Step 3 — ROI Mask Construction (`make_roi_masks_mat.m`)

**What it does:** Converts individual ROI NIfTI mask files into a single MATLAB struct for fast ROI-based voxel selection during decoding.

**Relation to paper:** Implements the ROI definition step (Section 1.8). The masks correspond to the Wang et al. (2015) probabilistic retinotopic atlas regions, transformed to EPI space.

**How it works:**

1. For each ROI name (V1v, V1d, V2v, etc.), loads the corresponding `{ROI}_in_EPI_bin.nii.gz` file
2. Reads the NIfTI volume (expected shape: 53 × 63 × 46) via gunzip → `niftiread`
3. Converts to binary logical mask (nonzero voxels = inside ROI)
4. Stores as a field in the `roi_masks` struct

The IPS ROI is created by combining IPS0–IPS5 sub-regions (matching the paper's approach of combining intraparietal sulcus subdivisions).

**Output:** `roi_masks.mat` with fields for each ROI (e.g., `roi_masks.V1v`, `roi_masks.PHC2`, etc.)

**Run:** Interactively in MATLAB.

### 3.6 Step 4 — MVPA Decoding (`SingleTrialDecodingv3.m`)

**What it does:** Performs SVM-based binary classification for each ROI and each subject, producing per-subject decoding accuracies.

**Relation to paper:** This is the core MVPA step (Section 1.9). Implements the linear SVM classification with repeated k-fold cross-validation.

**How it works:**

For each ROI and each subject:

1. **Load** the Pl, Nt, and Up beta matrices from Step 2
2. **Apply ROI mask** — select only voxels within the current ROI (or use all voxels for whole-brain)
3. **Remove invalid voxels** — drop voxels that are NaN across all conditions
4. **Construct classification problems:**
   - **Pleasant vs. Neutral:** data = `[Pl, Nt]'`, labels = `[+1, ..., -1, ...]`
   - **Unpleasant vs. Neutral:** data = `[Up, Nt]'`, labels = `[+1, ..., -1, ...]`
5. **Classify** using `svm_kfold_repeated`:
   - Partition data into **k=5 folds** (using `cvpartition`)
   - Train linear SVM (`fitcsvm`, linear kernel, standardized features) on training folds
   - Predict on test fold, compute fold accuracy
   - Repeat with **100 different random partitions**
   - Return mean accuracy across all folds and repetitions

**Note on our implementation vs. the paper:** The paper uses 10-fold CV; our implementation uses 5-fold CV. The paper uses LibSVM; we use MATLAB's `fitcsvm`. Both use linear kernels with standardization.

**Output:** `decoding_results_k5x100_v4.mat` containing:
- `PlNt_acc` — `[20 × 18]` matrix (subjects × ROIs) for Pleasant vs. Neutral accuracy
- `UpNt_acc` — `[20 × 18]` matrix for Unpleasant vs. Neutral accuracy
- Summary statistics (mean, SEM per ROI)

**Run command:**
```bash
sbatch pipeline/step4_decoding/run_singletd_m.sbatch   # 4 day walltime, 30GB RAM
```

### 3.7 Step 5 — Group-Level Statistical Validation (`group_level_validation.py`)

**What it does:** Tests whether decoding accuracies are significantly above chance at the group level using permutation testing.

**Relation to paper:** Implements Section 1.10 — the nonparametric permutation test (Stelzer et al., 2013) to establish significance thresholds.

**How it works:**

1. **Load** the decoding results `.mat` file from Step 4
2. **Permutation test** (per ROI, per comparison):
   - Generate 100,000 simulated group-level chance accuracies by sampling from a binomial distribution (50% chance for binary classification, n=100 trials) for each subject and averaging across subjects
   - Compare observed group mean accuracy against this null distribution
   - **p-value** = proportion of permuted means ≥ observed mean
   - **Significance threshold** at p < 0.001
3. **One-sample t-test** against chance level (0.5) with Cohen's d effect size
4. **Visualization:** Box plots and bar charts with significance annotations

**Checkpointing:** Designed for SLURM environments where jobs may be interrupted. Saves progress every 5,000 permutations and resumes from last checkpoint.

**Output:**
- `group_validation_results.mat` — p-values, significance flags, thresholds, Cohen's d for all ROIs
- `PlNt_group_validation.png` — Pleasant vs. Neutral group results figure
- `UpNt_group_validation.png` — Unpleasant vs. Neutral group results figure

**Run command:**
```bash
sbatch pipeline/step5_group_validation/run_group_validation.sh   # 4h walltime, 20GB RAM, 4 CPUs
```

**CLI arguments:**
```bash
python pipeline/step5_group_validation/group_level_validation.py \
    --results-file <path_to_decoding_results.mat> \
    --output-dir <output_path> \
    --checkpoint-dir <checkpoint_path> \
    --n-permutations 100000 \
    --checkpoint-interval 5000 \
    --alpha 0.001 \
    --comparison both          # Options: both, PlNt, UpNt
    --no-resume                # Start fresh, ignore checkpoints
```

---

## Part 4: Repository Structure

```
├── README.md
├── CLAUDE.md
├── FMRIPREP_MIGRATION.md
├── .gitignore
├── pipeline/
│   ├── step1_beta_estimation/
│   │   ├── BetaS2.m                    # SPM first-level GLM — single-trial beta estimation
│   │   └── run_betas2_m.sbatch         # SLURM job script for Step 1
│   ├── step2_extract_betas/
│   │   └── extract_betas.m             # Extract beta images into per-condition .mat files
│   ├── step3_roi_masks/
│   │   └── make_roi_masks_mat.m        # Build ROI mask struct from NIfTI files
│   ├── step4_decoding/
│   │   ├── SingleTrialDecodingv3.m     # SVM decoding per ROI (k=5, 100 reps)
│   │   └── run_singletd_m.sbatch       # SLURM job script for Step 4
│   └── step5_group_validation/
│       ├── group_level_validation.py   # Group-level permutation testing and statistics
│       └── run_group_validation.sh     # SLURM job script for Step 5
└── docs/
    ├── figures/                         # Pipeline explanation figures
    └── bold_pipeline_visualization.py   # Script to generate pipeline diagram
```

| File | Language | Pipeline Step | Description |
|---|---|---|---|
| `pipeline/step1_beta_estimation/BetaS2.m` | MATLAB | Step 1 | SPM first-level GLM — single-trial beta estimation |
| `pipeline/step2_extract_betas/extract_betas.m` | MATLAB | Step 2 | Extract beta images into per-condition .mat files |
| `pipeline/step3_roi_masks/make_roi_masks_mat.m` | MATLAB | Step 3 | Build ROI mask struct from NIfTI files |
| `pipeline/step4_decoding/SingleTrialDecodingv3.m` | MATLAB | Step 4 | SVM decoding per ROI (k=5, 100 reps) |
| `pipeline/step5_group_validation/group_level_validation.py` | Python | Step 5 | Group-level permutation testing and statistics |
| `pipeline/step1_beta_estimation/run_betas2_m.sbatch` | Bash/SLURM | — | SLURM job script for Step 1 |
| `pipeline/step4_decoding/run_singletd_m.sbatch` | Bash/SLURM | — | SLURM job script for Step 4 |
| `pipeline/step5_group_validation/run_group_validation.sh` | Bash/SLURM | — | SLURM job script for Step 5 |

---

## Dependencies

**MATLAB** (with toolboxes):
- SPM12 (path: `/path/to/spm` on HiPerGator)
- Statistics and Machine Learning Toolbox (for `fitcsvm`, `cvpartition`)
- Image Processing Toolbox (for `niftiread`, `niftiinfo`)

**Python 3.10+:**
- numpy, scipy, matplotlib, seaborn

---

## References

- Bo, K., Yin, S., Liu, Y., Hu, Z., Meyyappan, S., Kim, S., Keil, A., & Ding, M. (2021). Decoding Neural Representations of Affective Scenes in Retinotopic Visual Cortex. *Cerebral Cortex*, 31(6), 3047–3063. https://doi.org/10.1093/cercor/bhaa411
- Mumford, J. A., Turner, B. O., Ashby, F. G., & Poldrack, R. A. (2012). Deconvolving BOLD activation in event-related designs for multivoxel pattern classification analyses. *NeuroImage*, 59(3), 2636–2643.
- Wang, L., Mruczek, R. E. B., Arcaro, M. J., & Bhatt, S. S. (2015). Probabilistic maps of visual topography in human cortex. *Cerebral Cortex*, 25(10), 3911–3931.
- Stelzer, J., Chen, Y., & Turner, R. (2013). Statistical inference and multiple testing correction in classification-based multi-voxel pattern analysis (MVPA). *NeuroImage*, 84, 764–775.
