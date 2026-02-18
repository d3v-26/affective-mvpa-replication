"""
BOLD Signal Pipeline Visualization
===================================
Generates an educational figure showing the complete journey from
raw BOLD signal → HRF convolution → GLM design matrix → beta estimation
→ beta patterns → SVM classification.

Based on the methodology in Bo et al. (2021).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy.stats import norm

np.random.seed(42)

# ── Simulation parameters (matching our experiment) ──
TR = 1.98          # seconds
n_scans = 206      # per run
n_trials = 60      # per run (20 Pl + 20 Nt + 20 Up)
stim_dur = 3.0     # seconds
time = np.arange(n_scans) * TR  # time axis in seconds

# ── Figure setup ──
fig = plt.figure(figsize=(22, 32))
fig.patch.set_facecolor('white')

# Main grid: 8 rows
gs_main = gridspec.GridSpec(8, 1, hspace=0.45, top=0.96, bottom=0.03,
                            left=0.08, right=0.95,
                            height_ratios=[1, 1.1, 1.3, 1.0, 1.0, 1.0, 1.2, 1.0])

colors = {
    'Pl': '#E74C3C',   # red
    'Nt': '#7F8C8D',   # gray
    'Up': '#2980B9',   # blue
    'hrf': '#8E44AD',  # purple
    'motion': '#F39C12', # orange
    'noise': '#BDC3C7',
    'fit': '#27AE60',  # green
    'arrow': '#2C3E50',
}

panel_label_props = dict(fontsize=16, fontweight='bold', va='top', ha='left',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='#ECF0F1',
                                   edgecolor='#2C3E50', linewidth=1.5))

# ════════════════════════════════════════════════════════════
# PANEL A: Raw BOLD signal at a single voxel
# ════════════════════════════════════════════════════════════
ax_a = fig.add_subplot(gs_main[0])

# Generate realistic-looking BOLD signal
# slow drift + neural responses + noise
drift = 2 * np.sin(2 * np.pi * time / 300)  # slow scanner drift
noise = np.random.randn(n_scans) * 3

# Generate random trial onsets (in scans) for one run
iti_choices = [2800, 4300]  # ms
onset_times_ms = []
t_ms = 0
for _ in range(n_trials):
    onset_times_ms.append(t_ms)
    t_ms += stim_dur * 1000 + np.random.choice(iti_choices)
onset_scans = np.array(onset_times_ms) / 1000.0 / TR
onset_scans = onset_scans[onset_scans < n_scans - 5]

# Simple HRF for simulation
def hrf(t):
    """Double-gamma HRF"""
    from scipy.stats import gamma as gamma_dist
    h = gamma_dist.pdf(t, 6, scale=1) - 0.35 * gamma_dist.pdf(t, 16, scale=1)
    return h / np.max(h)

hrf_t = np.arange(0, 25, TR)
hrf_kernel = hrf(hrf_t)

# Create neural signal
neural = np.zeros(n_scans)
for ons in onset_scans:
    idx = int(round(ons))
    if idx < n_scans:
        neural[idx] = 1

# Convolve with HRF
bold_response = np.convolve(neural, hrf_kernel)[:n_scans] * 8
raw_bold = 1000 + bold_response + drift + noise  # baseline ~1000

ax_a.plot(time, raw_bold, color='#2C3E50', linewidth=0.8, alpha=0.9)
ax_a.fill_between(time, 990, raw_bold, alpha=0.15, color='#3498DB')

# Mark a few trial onsets
for i, ons in enumerate(onset_scans[:8]):
    t_onset = ons * TR
    ax_a.axvline(t_onset, color=colors['Pl'] if i < 3 else colors['Nt'] if i < 6 else colors['Up'],
                 alpha=0.4, linewidth=1, linestyle='--')

ax_a.set_xlabel('Time (seconds)', fontsize=11)
ax_a.set_ylabel('BOLD Signal\n(arbitrary units)', fontsize=11)
ax_a.set_xlim(0, 120)  # show first ~60s for clarity
ax_a.set_ylim(988, 1020)
ax_a.text(0.01, 0.95, 'A', transform=ax_a.transAxes, **panel_label_props)
ax_a.set_title('Raw BOLD Time Series at a Single Voxel  —  What the Scanner Measures',
               fontsize=13, fontweight='bold', pad=10)
ax_a.annotate('stimulus onsets\n(dashed lines)', xy=(onset_scans[1]*TR, 1017),
              fontsize=9, ha='center', color='#7F8C8D',
              arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2),
              xytext=(onset_scans[1]*TR + 15, 1019))
ax_a.annotate('slow drift + noise\ncorrupts the signal', xy=(80, 997),
              fontsize=9, ha='center', color='#95A5A6',
              arrowprops=dict(arrowstyle='->', color='#95A5A6', lw=1.2),
              xytext=(95, 1002))
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
# PANEL B: The Hemodynamic Response Function (HRF)
# ════════════════════════════════════════════════════════════
gs_b = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_main[1],
                                         width_ratios=[1, 0.15, 1.2], wspace=0.05)

# Left: neural event
ax_b1 = fig.add_subplot(gs_b[0])
t_neural = np.arange(0, 20, 0.01)
boxcar = np.zeros_like(t_neural)
boxcar[(t_neural >= 0) & (t_neural <= 3)] = 1

ax_b1.fill_between(t_neural, 0, boxcar, color=colors['Pl'], alpha=0.7, label='Neural activity\n(3s stimulus)')
ax_b1.set_xlabel('Time (seconds)', fontsize=11)
ax_b1.set_ylabel('Neural Activity', fontsize=11)
ax_b1.set_ylim(-0.15, 1.4)
ax_b1.set_xlim(-1, 20)
ax_b1.text(1.5, 1.1, 'Picture\ndisplayed\n(3 sec)', ha='center', fontsize=10, color=colors['Pl'],
           fontweight='bold')
ax_b1.axhline(0, color='black', linewidth=0.5)
ax_b1.text(0.01, 0.95, 'B', transform=ax_b1.transAxes, **panel_label_props)
ax_b1.set_title('Neural Event (Stimulus)', fontsize=12, fontweight='bold')
ax_b1.spines['top'].set_visible(False)
ax_b1.spines['right'].set_visible(False)

# Middle: convolution arrow
ax_arrow = fig.add_subplot(gs_b[1])
ax_arrow.set_xlim(0, 1)
ax_arrow.set_ylim(0, 1)
ax_arrow.text(0.5, 0.6, '   *   ', fontsize=28, ha='center', va='center',
              fontweight='bold', color=colors['hrf'],
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#F4ECF7', edgecolor=colors['hrf']))
ax_arrow.text(0.5, 0.35, 'convolution\nwith HRF', fontsize=10, ha='center', va='center',
              color=colors['hrf'], fontstyle='italic')
ax_arrow.axis('off')

# Right: HRF and resulting predicted BOLD
ax_b2 = fig.add_subplot(gs_b[2])
t_hrf = np.arange(0, 25, 0.01)
hrf_vals = hrf(t_hrf)

# Also show the convolution result (boxcar * HRF)
t_conv = np.arange(0, 35, 0.01)
boxcar_fine = np.zeros_like(t_conv)
boxcar_fine[(t_conv >= 0) & (t_conv <= 3)] = 1
hrf_fine = hrf(t_conv)
convolved = np.convolve(boxcar_fine, hrf_fine)[:len(t_conv)] * 0.01  # scale by dt
convolved = convolved / np.max(convolved)  # normalize

ax_b2.plot(t_hrf, hrf_vals, color=colors['hrf'], linewidth=2, linestyle='--',
           alpha=0.5, label='HRF (impulse response)')
ax_b2.plot(t_conv, convolved, color=colors['hrf'], linewidth=2.5,
           label='Predicted BOLD\n(boxcar * HRF)')
ax_b2.fill_between(t_conv, 0, convolved, alpha=0.15, color=colors['hrf'])
ax_b2.axhline(0, color='black', linewidth=0.5)

ax_b2.annotate('Peak ~5-6s after\nstimulus onset', xy=(6.5, convolved[650]),
               fontsize=9, ha='left',
               arrowprops=dict(arrowstyle='->', color=colors['hrf'], lw=1.5),
               xytext=(10, 0.85), color=colors['hrf'])
ax_b2.annotate('Undershoot', xy=(16, convolved[1600]),
               fontsize=9, ha='center',
               arrowprops=dict(arrowstyle='->', color=colors['hrf'], lw=1.2),
               xytext=(20, -0.15), color=colors['hrf'])

ax_b2.set_xlabel('Time (seconds)', fontsize=11)
ax_b2.set_ylabel('Predicted BOLD Response', fontsize=11)
ax_b2.set_xlim(-1, 30)
ax_b2.set_ylim(-0.25, 1.15)
ax_b2.legend(fontsize=9, loc='upper right')
ax_b2.set_title('Hemodynamic Response Function (HRF)', fontsize=12, fontweight='bold')
ax_b2.spines['top'].set_visible(False)
ax_b2.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
# PANEL C: Design Matrix — Standard GLM vs Beta Series
# ════════════════════════════════════════════════════════════
gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[2], wspace=0.35)

n_show = 80  # time points to show

# ── Standard GLM design matrix (3 condition regressors + motion) ──
ax_c1 = fig.add_subplot(gs_c[0])

# Create simplified design matrix for standard GLM
X_standard = np.zeros((n_show, 10))  # 3 conditions + 6 motion + constant
trial_indices_pl = np.sort(np.random.choice(range(5, n_show-10), 6, replace=False))
trial_indices_nt = np.sort(np.random.choice([x for x in range(5, n_show-10)
                                              if x not in trial_indices_pl], 6, replace=False))
trial_indices_up = np.sort(np.random.choice([x for x in range(5, n_show-10)
                                              if x not in trial_indices_pl
                                              and x not in trial_indices_nt], 6, replace=False))

for idx in trial_indices_pl:
    X_standard[idx:min(idx+8, n_show), 0] = hrf(np.arange(min(8, n_show-idx)))
for idx in trial_indices_nt:
    X_standard[idx:min(idx+8, n_show), 1] = hrf(np.arange(min(8, n_show-idx)))
for idx in trial_indices_up:
    X_standard[idx:min(idx+8, n_show), 2] = hrf(np.arange(min(8, n_show-idx)))

# motion regressors (smooth curves)
for m in range(6):
    X_standard[:, 3+m] = np.cumsum(np.random.randn(n_show) * 0.05)
X_standard[:, 9] = 1  # constant

# Normalize for display
X_disp = X_standard.copy()
for col in range(X_disp.shape[1]):
    if np.ptp(X_disp[:, col]) > 0:
        X_disp[:, col] = (X_disp[:, col] - X_disp[:, col].min()) / np.ptp(X_disp[:, col])

im1 = ax_c1.imshow(X_disp, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax_c1.set_xlabel('Regressors', fontsize=11)
ax_c1.set_ylabel('Time (scans) →', fontsize=11)
ax_c1.set_xticks(range(10))
ax_c1.set_xticklabels(['Pl\n(all)', 'Nt\n(all)', 'Up\n(all)',
                         'X', 'Y', 'Z', 'rx', 'ry', 'rz', 'const'], fontsize=8)
ax_c1.set_title('Standard GLM\n3 condition regressors\n→ 3 betas per voxel',
                fontsize=11, fontweight='bold', color='#E74C3C')

# Bracket annotations
ax_c1.annotate('', xy=(-0.5, -4), xytext=(2.5, -4),
               arrowprops=dict(arrowstyle='-', color=colors['Pl'], lw=2))
ax_c1.text(1, -6, '3 conditions', ha='center', fontsize=9, color=colors['Pl'], fontweight='bold')

ax_c1.annotate('', xy=(2.7, -4), xytext=(8.5, -4),
               arrowprops=dict(arrowstyle='-', color=colors['motion'], lw=2))
ax_c1.text(5.5, -6, '6 motion + const', ha='center', fontsize=9, color=colors['motion'])

ax_c1.text(0.01, 0.98, 'C', transform=ax_c1.transAxes, **panel_label_props)

# ── Beta Series design matrix (60 trial regressors + motion) ──
ax_c2 = fig.add_subplot(gs_c[1])

n_regressors_beta = 30  # show 30 of 60 for visual clarity
X_beta = np.zeros((n_show, n_regressors_beta + 7))  # trials + motion + constant

# Each trial gets one regressor with one HRF bump
all_onsets = np.sort(np.concatenate([trial_indices_pl, trial_indices_nt, trial_indices_up]))
for i, idx in enumerate(all_onsets[:n_regressors_beta]):
    X_beta[idx:min(idx+8, n_show), i] = hrf(np.arange(min(8, n_show-idx)))

# motion regressors
for m in range(6):
    X_beta[:, n_regressors_beta+m] = X_standard[:, 3+m]
X_beta[:, n_regressors_beta+6] = 1

# Normalize
X_beta_disp = X_beta.copy()
for col in range(X_beta_disp.shape[1]):
    if np.ptp(X_beta_disp[:, col]) > 0:
        X_beta_disp[:, col] = (X_beta_disp[:, col] - X_beta_disp[:, col].min()) / np.ptp(X_beta_disp[:, col])

im2 = ax_c2.imshow(X_beta_disp, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax_c2.set_xlabel('Regressors', fontsize=11)
ax_c2.set_ylabel('Time (scans) →', fontsize=11)

# Simplified x-labels
xtick_pos = [0, 5, 10, 15, 20, 25, n_regressors_beta, n_regressors_beta+3, n_regressors_beta+6]
xtick_lab = ['Pl1', 'Pl6', 'Nt1', 'Nt6', 'Up1', 'Up6', 'X', 'rx', 'const']
ax_c2.set_xticks(xtick_pos)
ax_c2.set_xticklabels(xtick_lab, fontsize=8)
ax_c2.set_title('Beta Series (Our Method)\n60 single-trial regressors\n→ 60 betas per voxel per run',
                fontsize=11, fontweight='bold', color='#27AE60')

ax_c2.annotate('', xy=(-0.5, -4), xytext=(n_regressors_beta-0.5, -4),
               arrowprops=dict(arrowstyle='-', color=colors['fit'], lw=2))
ax_c2.text(n_regressors_beta/2, -6, '60 individual trials (one column each)',
           ha='center', fontsize=9, color=colors['fit'], fontweight='bold')

# ════════════════════════════════════════════════════════════
# PANEL D: GLM fitting — how one beta is estimated
# ════════════════════════════════════════════════════════════
ax_d = fig.add_subplot(gs_main[3])

# Show: observed signal, model fit, one trial's contribution
t_short = np.arange(60) * TR
trial_onset = 15  # scan number for the trial of interest

# Build a realistic signal
signal_components = np.zeros((60, 4))  # trial_of_interest, other_trials, motion, noise
# Trial of interest
hrf_response = np.zeros(60)
hrf_t_short = np.arange(0, min(15, 60-trial_onset)) * 1.0
hrf_vals_short = hrf(hrf_t_short)
hrf_response[trial_onset:trial_onset+len(hrf_vals_short)] = hrf_vals_short * 3.5
signal_components[:, 0] = hrf_response

# Other trials
for ons in [5, 25, 38, 50]:
    ht = np.arange(0, min(12, 60-ons)) * 1.0
    hv = hrf(ht) * np.random.uniform(1.5, 4)
    signal_components[ons:ons+len(hv), 1] += hv

# Motion artifact
signal_components[:, 2] = np.cumsum(np.random.randn(60) * 0.15)
# Noise
signal_components[:, 3] = np.random.randn(60) * 0.8

observed = signal_components.sum(axis=1) + 1000
model_fit = signal_components[:, :3].sum(axis=1) + 1000

ax_d.plot(t_short, observed, color='#2C3E50', linewidth=1, alpha=0.6, label='Observed y(t)')
ax_d.plot(t_short, model_fit, color=colors['fit'], linewidth=2, label='Model fit Xβ̂')
ax_d.fill_between(t_short, 1000, 1000 + signal_components[:, 0],
                   color=colors['Pl'], alpha=0.4,
                   label=f'β̂₃ × HRF_trial3(t)  ← this trial\'s contribution')

# Mark the trial onset
ax_d.axvline(trial_onset * TR, color=colors['Pl'], linewidth=1.5, linestyle='--', alpha=0.7)
ax_d.annotate('Trial onset\n(Pl3)', xy=(trial_onset*TR, 1005.5), fontsize=9,
              ha='center', va='bottom', color=colors['Pl'], fontweight='bold')

# Show the beta value
ax_d.annotate('β̂₃ = 3.5\n(this voxel\'s\nresponse to\nPleasant #3)',
              xy=(trial_onset*TR + 10, 1002.8),
              fontsize=10, ha='left', va='center',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='#FADBD8', edgecolor=colors['Pl']),
              color=colors['Pl'], fontweight='bold')

# Show residual
residual_region = slice(40, 55)
ax_d.fill_between(t_short[residual_region],
                   model_fit[residual_region], observed[residual_region],
                   color=colors['noise'], alpha=0.5, hatch='///')
ax_d.annotate('ε (residual)', xy=(47*TR, (model_fit[47]+observed[47])/2),
              fontsize=9, color='#7F8C8D', ha='left',
              arrowprops=dict(arrowstyle='->', color='#7F8C8D'),
              xytext=(52*TR, 1003))

ax_d.set_xlabel('Time (seconds)', fontsize=11)
ax_d.set_ylabel('BOLD Signal', fontsize=11)
ax_d.set_xlim(0, t_short[-1])
ax_d.legend(fontsize=9, loc='upper right', ncol=3)
ax_d.text(0.01, 0.95, 'D', transform=ax_d.transAxes, **panel_label_props)
ax_d.set_title('GLM Estimation at One Voxel  —  y = Xβ + ε  →  solve for β̂ = (X\'V⁻¹X)⁻¹X\'V⁻¹y',
               fontsize=13, fontweight='bold', pad=10)
ax_d.spines['top'].set_visible(False)
ax_d.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
# PANEL E: Motion regressors absorbing variance
# ════════════════════════════════════════════════════════════
ax_e = fig.add_subplot(gs_main[4])

# Show signal with and without motion regression
motion_artifact = np.cumsum(np.random.randn(80) * 0.4) + np.sin(np.arange(80) * 0.15) * 2
clean_neural = np.zeros(80)
for ons in [8, 22, 38, 55, 68]:
    ht = np.arange(0, min(12, 80-ons)) * 1.0
    hv = hrf(ht) * 3
    clean_neural[ons:ons+len(hv)] += hv

contaminated = clean_neural + motion_artifact + np.random.randn(80) * 0.5
t_e = np.arange(80) * TR

ax_e.plot(t_e, contaminated, color='#E74C3C', linewidth=1, alpha=0.7,
          label='Signal WITH motion artifact')
ax_e.plot(t_e, clean_neural + np.random.randn(80) * 0.5, color=colors['fit'], linewidth=1.8,
          label='Signal AFTER motion regression (β̂_motion removed)')
ax_e.plot(t_e, motion_artifact, color=colors['motion'], linewidth=1.5, linestyle='--',
          alpha=0.8, label='Motion artifact (absorbed by motion β\'s)')

ax_e.axhline(0, color='black', linewidth=0.5)
ax_e.set_xlabel('Time (seconds)', fontsize=11)
ax_e.set_ylabel('Signal', fontsize=11)
ax_e.legend(fontsize=9, loc='upper right', ncol=1)
ax_e.text(0.01, 0.95, 'E', transform=ax_e.transAxes, **panel_label_props)
ax_e.set_title('Head Motion Correction in the GLM  —  Motion Regressors Absorb Movement-Related Variance',
               fontsize=13, fontweight='bold', pad=10)
ax_e.spines['top'].set_visible(False)
ax_e.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
# PANEL F: Beta images → voxel patterns
# ════════════════════════════════════════════════════════════
gs_f = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[5],
                                         width_ratios=[1.5, 1], wspace=0.3)

ax_f1 = fig.add_subplot(gs_f[0])

# Simulate a small "brain slice" beta image for 3 conditions
np.random.seed(7)
sz = 12
base_pattern = np.random.randn(sz, sz) * 0.3

# Pleasant pattern: higher in top-left region
pl_pattern = base_pattern.copy()
pl_pattern[:6, :6] += 1.5
pl_pattern[6:, 6:] -= 0.5

# Neutral pattern: uniform
nt_pattern = base_pattern.copy() + 0.2

# Unpleasant pattern: higher in bottom-right region
up_pattern = base_pattern.copy()
up_pattern[:6, :6] -= 0.5
up_pattern[6:, 6:] += 1.5

# Create brain mask (circular)
y_grid, x_grid = np.mgrid[:sz, :sz]
mask = ((x_grid - sz/2)**2 + (y_grid - sz/2)**2) < (sz/2.2)**2
pl_pattern[~mask] = np.nan
nt_pattern[~mask] = np.nan
up_pattern[~mask] = np.nan

# Show 3 beta images side by side
patterns = [pl_pattern, nt_pattern, up_pattern]
titles = ['β̂ for Pleasant trial #1', 'β̂ for Neutral trial #1', 'β̂ for Unpleasant trial #1']
title_colors = [colors['Pl'], colors['Nt'], colors['Up']]

for i, (pat, ttl, tc) in enumerate(zip(patterns, titles, title_colors)):
    extent = [i * (sz + 2), i * (sz + 2) + sz, 0, sz]
    ax_f1.imshow(pat, cmap='RdBu_r', vmin=-2, vmax=2,
                 extent=extent, interpolation='nearest')
    ax_f1.text(extent[0] + sz/2, sz + 0.8, ttl, ha='center', fontsize=9,
               fontweight='bold', color=tc)

ax_f1.set_xlim(-1, 3 * (sz + 2) - 1)
ax_f1.set_ylim(-1.5, sz + 2)
ax_f1.set_aspect('equal')
ax_f1.axis('off')
ax_f1.text(0.01, 0.95, 'F', transform=ax_f1.transAxes, **panel_label_props)
ax_f1.set_title('Beta Images (One "Brain Slice")  —  Each Trial Produces a Spatial Pattern',
                fontsize=13, fontweight='bold', pad=10, loc='left')

# Show vectorized patterns
ax_f2 = fig.add_subplot(gs_f[1])
n_vox_show = 20

pl_vec = pl_pattern[mask][:n_vox_show]
nt_vec = nt_pattern[mask][:n_vox_show]
up_vec = up_pattern[mask][:n_vox_show]

x_vox = np.arange(n_vox_show)
width = 0.25
ax_f2.bar(x_vox - width, pl_vec, width, color=colors['Pl'], alpha=0.8, label='Pleasant')
ax_f2.bar(x_vox, nt_vec, width, color=colors['Nt'], alpha=0.8, label='Neutral')
ax_f2.bar(x_vox + width, up_vec, width, color=colors['Up'], alpha=0.8, label='Unpleasant')
ax_f2.axhline(0, color='black', linewidth=0.5)
ax_f2.set_xlabel('Voxel index', fontsize=10)
ax_f2.set_ylabel('Beta value (β̂)', fontsize=10)
ax_f2.legend(fontsize=8, loc='upper right')
ax_f2.set_title('Vectorized Beta Patterns\n(same data as bar chart)', fontsize=11, fontweight='bold')
ax_f2.spines['top'].set_visible(False)
ax_f2.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
# PANEL G: SVM classification — finding the separating hyperplane
# ════════════════════════════════════════════════════════════
gs_g = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[6],
                                         width_ratios=[1, 1], wspace=0.25)

# Left: 2D projection of Pl vs Nt trials
ax_g1 = fig.add_subplot(gs_g[0])

# Simulate 2D projected trial data
np.random.seed(21)
n_trials_per = 30

# Pleasant cluster
pl_x = np.random.randn(n_trials_per) * 0.6 + 2.0
pl_y = np.random.randn(n_trials_per) * 0.5 + 1.5

# Neutral cluster
nt_x = np.random.randn(n_trials_per) * 0.6 - 0.5
nt_y = np.random.randn(n_trials_per) * 0.5 - 0.3

ax_g1.scatter(pl_x, pl_y, c=colors['Pl'], s=50, alpha=0.7, edgecolors='white',
              linewidth=0.5, label='Pleasant trials', zorder=3)
ax_g1.scatter(nt_x, nt_y, c=colors['Nt'], s=50, alpha=0.7, edgecolors='white',
              linewidth=0.5, label='Neutral trials', zorder=3)

# Decision boundary (hyperplane)
x_boundary = np.linspace(-2.5, 4.5, 100)
y_boundary = -0.8 * x_boundary + 1.5  # slope and intercept for visual effect

ax_g1.plot(x_boundary, y_boundary, 'k-', linewidth=2.5, label='SVM hyperplane\n(w · x + b = 0)')
ax_g1.plot(x_boundary, y_boundary + 0.7, 'k--', linewidth=1, alpha=0.4)
ax_g1.plot(x_boundary, y_boundary - 0.7, 'k--', linewidth=1, alpha=0.4)
ax_g1.fill_between(x_boundary, y_boundary + 0.7, y_boundary - 0.7,
                    alpha=0.08, color='black')

# Mark support vectors
sv_pl = [(1.2, 0.9), (1.5, 1.3)]
sv_nt = [(0.3, 0.3), (0.0, 0.6)]
for sv in sv_pl:
    ax_g1.scatter(*sv, c=colors['Pl'], s=150, edgecolors='black', linewidth=2, zorder=5)
for sv in sv_nt:
    ax_g1.scatter(*sv, c=colors['Nt'], s=150, edgecolors='black', linewidth=2, zorder=5)

ax_g1.annotate('margin', xy=(2.5, -0.1), fontsize=9, ha='center', color='#555',
               arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5),
               xytext=(2.5, -0.1))
# draw margin arrows manually
ax_g1.annotate('', xy=(2.8, y_boundary[65] + 0.7 - 0.7),
               xytext=(2.8, y_boundary[65] + 0.7 + 0.0),
               arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
ax_g1.text(3.2, y_boundary[65] + 0.35, 'margin', fontsize=9, color='#555', rotation=52)

ax_g1.annotate('support\nvectors', xy=(sv_pl[0][0], sv_pl[0][1]),
               xytext=(-1.5, 2.5), fontsize=9, color='#2C3E50',
               arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=1.2))

ax_g1.set_xlabel('Voxel pattern dimension 1\n(e.g., PC1 of beta patterns)', fontsize=10)
ax_g1.set_ylabel('Voxel pattern dimension 2', fontsize=10)
ax_g1.set_xlim(-2.5, 4.5)
ax_g1.set_ylim(-2, 3.5)
ax_g1.legend(fontsize=9, loc='lower right')
ax_g1.text(0.01, 0.98, 'G', transform=ax_g1.transAxes, **panel_label_props)
ax_g1.set_title('SVM Classification  —  Pleasant vs. Neutral\n(each dot = one trial\'s beta pattern)',
                fontsize=12, fontweight='bold')
ax_g1.spines['top'].set_visible(False)
ax_g1.spines['right'].set_visible(False)

# Right: cross-validation accuracy
ax_g2 = fig.add_subplot(gs_g[1])

# Simulate 100 repetitions of 5-fold CV
np.random.seed(99)
rep_accuracies = np.random.beta(12, 6, 100) * 0.3 + 0.5  # centered ~65%
fold_accs = np.random.beta(10, 7, (5, 3)) * 0.35 + 0.45  # one rep's 5 folds

# Show one repetition's folds as a bar chart
bar_x = np.arange(5)
bar_colors = ['#3498DB', '#E74C3C', '#27AE60', '#9B59B6', '#F39C12']
bars = ax_g2.bar(bar_x, fold_accs[:, 0], color=bar_colors, alpha=0.7,
                  edgecolor='white', linewidth=1.5, width=0.6)

# Add mean line for this rep
rep_mean = fold_accs[:, 0].mean()
ax_g2.axhline(rep_mean, color='#2C3E50', linewidth=2, linestyle='-',
              label=f'Rep. mean = {rep_mean:.1%}')

# Add chance line
ax_g2.axhline(0.5, color='#E74C3C', linewidth=1.5, linestyle='--',
              label='Chance (50%)')

# Add significance threshold
ax_g2.axhline(0.54, color='#F39C12', linewidth=1.5, linestyle='-.',
              label='Significance threshold (54%)\n[p < 0.001, permutation test]')

for i, b in enumerate(bars):
    ax_g2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
               f'{fold_accs[i,0]:.0%}', ha='center', fontsize=9, fontweight='bold')

ax_g2.set_xticks(bar_x)
ax_g2.set_xticklabels([f'Fold {i+1}' for i in range(5)], fontsize=10)
ax_g2.set_ylabel('Decoding Accuracy', fontsize=11)
ax_g2.set_ylim(0.35, 0.85)
ax_g2.legend(fontsize=9, loc='upper right')
ax_g2.set_title('5-Fold Cross-Validation (1 of 100 Reps)\nTrain on 4 folds → Test on held-out fold',
                fontsize=12, fontweight='bold')
ax_g2.spines['top'].set_visible(False)
ax_g2.spines['right'].set_visible(False)

# ════════════════════════════════════════════════════════════
# PANEL H: Univariate vs Multivariate — the key insight
# ════════════════════════════════════════════════════════════
gs_h = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[7], wspace=0.3)

# Left: Univariate — average activation per voxel
ax_h1 = fig.add_subplot(gs_h[0])

np.random.seed(15)
n_vox = 8
# Create a pattern where mean is similar but pattern differs
pl_mean_vals = np.array([1.2, 0.4, 1.1, 0.3, 1.0, 0.5, 0.9, 0.6])
nt_mean_vals = np.array([0.5, 1.1, 0.4, 1.0, 0.6, 0.9, 0.5, 1.1])

x_pos = np.arange(n_vox)
width_h = 0.35

bars_pl = ax_h1.bar(x_pos - width_h/2, pl_mean_vals, width_h, color=colors['Pl'],
                     alpha=0.8, label='Pleasant (mean β)')
bars_nt = ax_h1.bar(x_pos + width_h/2, nt_mean_vals, width_h, color=colors['Nt'],
                     alpha=0.8, label='Neutral (mean β)')

# Show that the ROI average is nearly identical
roi_avg_pl = np.mean(pl_mean_vals)
roi_avg_nt = np.mean(nt_mean_vals)
ax_h1.axhline(roi_avg_pl, color=colors['Pl'], linewidth=2, linestyle='--', alpha=0.6)
ax_h1.axhline(roi_avg_nt, color=colors['Nt'], linewidth=2, linestyle='--', alpha=0.6)

ax_h1.annotate(f'ROI mean Pl = {roi_avg_pl:.2f}', xy=(7.5, roi_avg_pl),
               fontsize=9, color=colors['Pl'], ha='right', va='bottom')
ax_h1.annotate(f'ROI mean Nt = {roi_avg_nt:.2f}', xy=(7.5, roi_avg_nt),
               fontsize=9, color=colors['Nt'], ha='right', va='top')

ax_h1.text(4, 1.45, 'UNIVARIATE RESULT:\nROI means nearly equal → "no effect"\n(t-test: p = 0.72, n.s.)',
           fontsize=10, ha='center', color='#E74C3C', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#FADBD8', alpha=0.9))

ax_h1.set_xticks(x_pos)
ax_h1.set_xticklabels([f'V{i+1}' for i in range(n_vox)], fontsize=9)
ax_h1.set_xlabel('Voxels in ROI', fontsize=11)
ax_h1.set_ylabel('Mean Beta (β̂)', fontsize=11)
ax_h1.legend(fontsize=9, loc='lower right')
ax_h1.text(0.01, 0.95, 'H', transform=ax_h1.transAxes, **panel_label_props)
ax_h1.set_title('Univariate Analysis\nCompares mean activation per voxel',
                fontsize=12, fontweight='bold')
ax_h1.set_ylim(0, 1.8)
ax_h1.spines['top'].set_visible(False)
ax_h1.spines['right'].set_visible(False)

# Right: Multivariate — pattern is clearly different
ax_h2 = fig.add_subplot(gs_h[1])

ax_h2.plot(x_pos, pl_mean_vals, 'o-', color=colors['Pl'], linewidth=2.5, markersize=10,
           label='Pleasant pattern', zorder=3)
ax_h2.plot(x_pos, nt_mean_vals, 's-', color=colors['Nt'], linewidth=2.5, markersize=10,
           label='Neutral pattern', zorder=3)

# Fill between to highlight pattern difference
ax_h2.fill_between(x_pos, pl_mean_vals, nt_mean_vals, alpha=0.15, color='#8E44AD')

# Add arrows showing the "flip" pattern
for i in range(n_vox):
    if abs(pl_mean_vals[i] - nt_mean_vals[i]) > 0.3:
        direction = '↑' if pl_mean_vals[i] > nt_mean_vals[i] else '↓'

ax_h2.text(4, 1.45, 'MVPA RESULT:\nPatterns clearly differ → Accuracy = 72%\n'
                     '(p < 0.001, well above 54% threshold)',
           fontsize=10, ha='center', color='#27AE60', fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='#D5F5E3', alpha=0.9))

ax_h2.set_xticks(x_pos)
ax_h2.set_xticklabels([f'V{i+1}' for i in range(n_vox)], fontsize=9)
ax_h2.set_xlabel('Voxels in ROI', fontsize=11)
ax_h2.set_ylabel('Mean Beta (β̂)', fontsize=11)
ax_h2.legend(fontsize=9, loc='lower right')
ax_h2.set_title('Multivariate (MVPA)\nExamines the PATTERN across voxels',
                fontsize=12, fontweight='bold')
ax_h2.set_ylim(0, 1.8)
ax_h2.spines['top'].set_visible(False)
ax_h2.spines['right'].set_visible(False)

# ── Save ──
fig.suptitle('BOLD Signal Pipeline: From Scanner to Decoding Accuracy',
             fontsize=18, fontweight='bold', y=0.99, color='#2C3E50')

output_path = '/Users/nik/Desktop/smile-lab/beta/bold_pipeline_visualization.png'
fig.savefig(output_path, dpi=200, facecolor='white', bbox_inches='tight')
print(f"Saved to {output_path}")
plt.close()
