"""
Educational Figure Series: BOLD Signal Pipeline
================================================
Generates a series of detailed, standalone figures explaining each step
of the fMRI→beta→MVPA pipeline from Bo et al. (2021).

Each figure is self-contained and heavily annotated so that someone
with no ML or neuroimaging background can follow along.

Usage:
    source .venv/bin/activate
    python generate_figures.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path('/Users/nik/Desktop/smile-lab/beta/figures')
OUT.mkdir(exist_ok=True)

np.random.seed(42)

# ── Shared helpers ──
def hrf(t):
    """Canonical double-gamma hemodynamic response function."""
    from scipy.stats import gamma as gd
    h = gd.pdf(t, 6, scale=1) - 0.35 * gd.pdf(t, 16, scale=1)
    h[t < 0] = 0
    return h / np.max(h) if np.max(h) > 0 else h

C = {
    'pl': '#E74C3C', 'nt': '#95A5A6', 'up': '#2980B9',
    'hrf': '#8E44AD', 'motion': '#E67E22', 'fit': '#27AE60',
    'dark': '#2C3E50', 'light_bg': '#F8F9FA', 'accent': '#F39C12',
    'good': '#27AE60', 'bad': '#E74C3C',
}

def savefig(fig, name):
    path = OUT / name
    fig.savefig(path, dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {path.name}")

def annotate_box(ax, text, xy, width=0.35, fontsize=10, color='#2C3E50', bg='#F8F9FA'):
    ax.annotate(text, xy=xy, fontsize=fontsize, ha='center', va='center',
                color=color, fontweight='bold',
                bbox=dict(boxstyle=f'round,pad=0.5', facecolor=bg,
                          edgecolor=color, linewidth=1.5))


# ════════════════════════════════════════════════════════════════
# FIGURE 1: What the Scanner Actually Sees
# ════════════════════════════════════════════════════════════════
def fig01():
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('1. What the fMRI Scanner Actually Measures',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])

    gs = gridspec.GridSpec(3, 1, hspace=0.4, top=0.93, bottom=0.06,
                           left=0.1, right=0.92, height_ratios=[1.2, 0.8, 1.0])

    # ── Top: brain grid with one highlighted voxel ──
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 7)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Draw a simplified brain slice as a grid
    grid_x, grid_y = 8, 5
    offset_x, offset_y = 1, 1
    for i in range(grid_x):
        for j in range(grid_y):
            # rough brain shape mask
            cx, cy = (i - grid_x/2 + 0.5), (j - grid_y/2 + 0.5)
            if cx**2/(grid_x/2)**2 + cy**2/(grid_y/2)**2 < 0.85:
                color = '#D5DBDB'
                alpha = 0.6
                if i == 4 and j == 3:
                    color = C['pl']
                    alpha = 0.9
                rect = Rectangle((offset_x + i*0.9, offset_y + j*0.9), 0.85, 0.85,
                                  facecolor=color, edgecolor='white', linewidth=1, alpha=alpha)
                ax1.add_patch(rect)

    # Highlight voxel
    ax1.annotate('This is ONE voxel\n(a tiny cube of brain tissue,\n~3mm × 3mm × 3mm)',
                 xy=(offset_x + 4*0.9 + 0.42, offset_y + 3*0.9 + 0.42),
                 xytext=(9, 5.5), fontsize=11, ha='left', color=C['pl'], fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=C['pl'], lw=2,
                                 connectionstyle='arc3,rad=0.2'))

    ax1.text(offset_x + grid_x*0.9/2, 0.3,
             'One brain "slice" — the scanner images ~36 of these stacked slices\n'
             'every TR = 1.98 seconds, creating a full 3D brain volume',
             ha='center', fontsize=11, color=C['dark'])

    ax1.text(offset_x + grid_x*0.9/2, 6.5,
             'The brain is divided into thousands of tiny cubes called VOXELS',
             ha='center', fontsize=14, fontweight='bold', color=C['dark'])

    # ── Middle: explanation text ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')
    explanation = (
        "The fMRI scanner does NOT directly measure brain activity (neurons firing).\n\n"
        "Instead, it measures changes in BLOOD FLOW. When neurons become active,\n"
        "they consume oxygen. The body responds by sending more oxygenated blood\n"
        "to that area. This changes the magnetic properties of the tissue, which\n"
        "the scanner detects as the BOLD signal (Blood Oxygen Level-Dependent).\n\n"
        "Think of it like measuring how hard a factory is working by looking at\n"
        "its power consumption — indirect, delayed, but informative."
    )
    ax2.text(0.5, 0.5, explanation, transform=ax2.transAxes, fontsize=12,
             ha='center', va='center', color=C['dark'], linespacing=1.6,
             bbox=dict(boxstyle='round,pad=0.8', facecolor='#EBF5FB',
                       edgecolor='#2980B9', linewidth=1.5))

    # ── Bottom: raw time series ──
    ax3 = fig.add_subplot(gs[2])
    TR = 1.98
    n_scans = 120
    time = np.arange(n_scans) * TR

    # Generate realistic BOLD
    drift = 1.5 * np.sin(2*np.pi*time/250)
    noise = np.random.randn(n_scans) * 2.5
    neural = np.zeros(n_scans)
    trial_onsets = [8, 16, 24, 33, 41, 50, 59, 67, 76, 85, 93, 102, 110]
    trial_types = ['Pl','Nt','Up','Pl','Up','Nt','Pl','Nt','Up','Pl','Nt','Up','Pl']
    for ons in trial_onsets:
        if ons < n_scans:
            neural[ons] = 1
    hrf_t = np.arange(0, 20, TR)
    hrf_k = hrf(hrf_t)
    bold_resp = np.convolve(neural, hrf_k)[:n_scans] * 6
    raw = 1000 + bold_resp + drift + noise

    ax3.plot(time, raw, color=C['dark'], linewidth=1)
    ax3.fill_between(time, 988, raw, alpha=0.1, color='#3498DB')

    # Mark trials with colored ticks
    for ons, tt in zip(trial_onsets, trial_types):
        t_sec = ons * TR
        col = C['pl'] if tt == 'Pl' else C['nt'] if tt == 'Nt' else C['up']
        ax3.axvline(t_sec, ymin=0, ymax=0.08, color=col, linewidth=3)
        ax3.text(t_sec, 987.5, tt, ha='center', fontsize=7, color=col, fontweight='bold')

    ax3.set_xlabel('Time (seconds)', fontsize=12)
    ax3.set_ylabel('BOLD Signal\n(scanner units)', fontsize=12)
    ax3.set_title('The BOLD time series at our highlighted voxel — 120 scans over ~4 minutes',
                  fontsize=13, fontweight='bold', color=C['dark'])
    ax3.set_xlim(0, time[-1])

    # Annotations
    ax3.annotate('Each colored tick = one picture\nshown to the participant',
                 xy=(trial_onsets[2]*TR, 988), xytext=(trial_onsets[2]*TR + 30, 990),
                 fontsize=10, color=C['dark'],
                 arrowprops=dict(arrowstyle='->', color=C['dark'], lw=1.5))

    ax3.annotate('The signal is NOISY —\nwe need the GLM to\nextract each trial\'s\nresponse from this mess',
                 xy=(80*TR, raw[80]), xytext=(85*TR, 1013),
                 fontsize=10, color='#7F8C8D',
                 arrowprops=dict(arrowstyle='->', color='#7F8C8D', lw=1.2))

    # Legend
    legend_elements = [
        mpatches.Patch(color=C['pl'], label='Pleasant picture'),
        mpatches.Patch(color=C['nt'], label='Neutral picture'),
        mpatches.Patch(color=C['up'], label='Unpleasant picture'),
    ]
    ax3.legend(handles=legend_elements, fontsize=10, loc='upper left')

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    savefig(fig, '01_what_the_scanner_sees.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 2: The Hemodynamic Delay
# ════════════════════════════════════════════════════════════════
def fig02():
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [1, 1.2, 1.2]})
    fig.suptitle('2. The Hemodynamic Response — Why Brain Signals Are Delayed',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    plt.subplots_adjust(hspace=0.45, top=0.93, bottom=0.06, left=0.12, right=0.92)

    t = np.arange(0, 25, 0.05)

    # ── Panel A: Neural event (fast) ──
    ax = axes[0]
    neural = np.zeros_like(t)
    neural[(t >= 0) & (t <= 3)] = 1

    ax.fill_between(t, 0, neural, color=C['pl'], alpha=0.7, linewidth=0)
    ax.plot(t, neural, color=C['pl'], linewidth=2)
    ax.axhline(0, color='black', linewidth=0.5)

    ax.annotate('Picture appears\non screen', xy=(0, 1.05), fontsize=12,
                color=C['pl'], fontweight='bold', ha='left')
    ax.annotate('Picture disappears\nafter 3 seconds', xy=(3, 1.05), fontsize=12,
                color=C['pl'], fontweight='bold', ha='left')
    ax.annotate('', xy=(0, -0.15), xytext=(3, -0.15),
                arrowprops=dict(arrowstyle='<->', color=C['pl'], lw=2))
    ax.text(1.5, -0.25, '3 seconds', ha='center', fontsize=11, color=C['pl'], fontweight='bold')

    ax.set_ylabel('Neural Activity\n(neurons firing)', fontsize=12)
    ax.set_title('A.  What actually happens in the brain — FAST response', fontsize=14,
                 fontweight='bold', color=C['dark'], loc='left')
    ax.set_xlim(-1, 25)
    ax.set_ylim(-0.4, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Off', 'On'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(8, 0.8, 'Neurons respond almost instantly\nwhen the picture appears,\nand stop when it disappears.',
            fontsize=11, color='#555', fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor=C['light_bg'], alpha=0.9))

    # ── Panel B: HRF (the delay function) ──
    ax = axes[1]
    hrf_vals = hrf(t)

    ax.plot(t, hrf_vals, color=C['hrf'], linewidth=3)
    ax.fill_between(t, 0, hrf_vals, where=hrf_vals > 0, color=C['hrf'], alpha=0.15)
    ax.fill_between(t, 0, hrf_vals, where=hrf_vals < 0, color=C['hrf'], alpha=0.1)
    ax.axhline(0, color='black', linewidth=0.5)

    # Key annotations
    peak_idx = np.argmax(hrf_vals)
    peak_t = t[peak_idx]
    ax.annotate(f'Peak at ~{peak_t:.0f} seconds\nBOLD signal is STRONGEST here\n'
                f'(even though the picture is\n long gone!)',
                xy=(peak_t, hrf_vals[peak_idx]),
                xytext=(peak_t + 4, 0.85), fontsize=11, color=C['hrf'], fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['hrf'], lw=2))

    undershoot_idx = np.argmin(hrf_vals)
    ax.annotate('Undershoot\n(blood flow briefly\ndips below normal)',
                xy=(t[undershoot_idx], hrf_vals[undershoot_idx]),
                xytext=(t[undershoot_idx] + 3, -0.2), fontsize=10, color=C['hrf'],
                arrowprops=dict(arrowstyle='->', color=C['hrf'], lw=1.5))

    ax.annotate('Signal returns\nto baseline ~20s later', xy=(22, 0.02),
                fontsize=10, color='#777', ha='center')

    ax.axvline(0, color=C['pl'], linewidth=1.5, linestyle='--', alpha=0.5)
    ax.text(0.3, 0.6, 'Stimulus\nonset', fontsize=9, color=C['pl'], rotation=90)

    ax.set_ylabel('Blood Flow Change\n(BOLD response)', fontsize=12)
    ax.set_title('B.  The Hemodynamic Response Function (HRF) — the "shape" of a blood-flow response',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.set_xlim(-1, 25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(14, 0.55,
            'This is the key problem:\n'
            'neural activity is FAST (milliseconds)\n'
            'but blood flow is SLOW (seconds).\n\n'
            'The HRF is the "translator" — it tells\n'
            'us what shape the blood-flow response\n'
            'should have for a given neural event.',
            fontsize=11, color=C['dark'],
            bbox=dict(boxstyle='round', facecolor='#F4ECF7', edgecolor=C['hrf'], linewidth=1.5))

    # ── Panel C: Convolution result ──
    ax = axes[2]

    # Boxcar convolved with HRF
    t_long = np.arange(0, 30, 0.05)
    boxcar = np.zeros_like(t_long)
    boxcar[(t_long >= 0) & (t_long <= 3)] = 1
    hrf_fine = hrf(t_long)
    convolved = np.convolve(boxcar, hrf_fine)[:len(t_long)] * 0.05
    convolved = convolved / np.max(convolved) if np.max(convolved) > 0 else convolved

    ax.fill_between(t_long, 0, boxcar * 0.3, color=C['pl'], alpha=0.3, label='Neural event (3s picture)')
    ax.plot(t_long, convolved, color=C['hrf'], linewidth=3,
            label='Predicted BOLD response\n(neural event ∗ HRF)')
    ax.fill_between(t_long, 0, convolved, color=C['hrf'], alpha=0.12)
    ax.axhline(0, color='black', linewidth=0.5)

    # Show the delay
    ax.annotate('', xy=(0, -0.08), xytext=(5.5, -0.08),
                arrowprops=dict(arrowstyle='<->', color=C['bad'], lw=2.5))
    ax.text(2.75, -0.15, '~5–6 second delay!', ha='center', fontsize=12,
            color=C['bad'], fontweight='bold')

    ax.set_xlabel('Time after stimulus onset (seconds)', fontsize=12)
    ax.set_ylabel('Signal Amplitude', fontsize=12)
    ax.set_title('C.  The result: predicted BOLD signal for a 3-second picture',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.set_xlim(-1, 28)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(15, 0.65,
            'WHY THIS MATTERS:\n\n'
            'When we build our statistical model (the GLM),\n'
            'we can\'t just look at the brain signal at the exact\n'
            'moment a picture appears. We have to account for\n'
            'this ~5 second delay. The HRF tells the model\n'
            'WHERE in time to look for each trial\'s response.',
            fontsize=11, color=C['dark'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#FEF9E7',
                      edgecolor=C['accent'], linewidth=2))

    savefig(fig, '02_the_hemodynamic_delay.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 3: Building the Design Matrix
# ════════════════════════════════════════════════════════════════
def fig03():
    fig = plt.figure(figsize=(16, 18))
    fig.suptitle('3. From Picture Onsets to the Design Matrix — Building the Model',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(4, 1, hspace=0.5, top=0.93, bottom=0.04,
                           left=0.08, right=0.95, height_ratios=[0.8, 1, 1.3, 0.6])

    TR = 1.98
    n_scans = 60
    time = np.arange(n_scans) * TR

    # ── Panel A: Stimulus timeline ──
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, time[-1])
    ax.set_ylim(-0.5, 2)
    ax.axis('off')

    onsets = [5, 12, 20, 28, 35, 43]
    types =  ['Pl1','Nt1','Up1','Pl2','Nt2','Up2']
    type_colors = [C['pl'], C['nt'], C['up'], C['pl'], C['nt'], C['up']]

    for ons, name, col in zip(onsets, types, type_colors):
        t_sec = ons * TR
        rect = Rectangle((t_sec, 0.2), 3*TR, 1.2, facecolor=col, alpha=0.7,
                          edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(t_sec + 1.5*TR, 0.8, name, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

    ax.annotate('', xy=(0, -0.1), xytext=(time[-1], -0.1),
                arrowprops=dict(arrowstyle='->', color=C['dark'], lw=1.5))
    ax.text(time[-1]/2, -0.4, 'Time →', ha='center', fontsize=11, color=C['dark'])
    ax.set_title('A.  The experiment timeline: pictures shown to the participant',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left', pad=10)

    # ── Panel B: Individual onset stick functions ──
    ax = fig.add_subplot(gs[1])
    for idx, (ons, name, col) in enumerate(zip(onsets[:3], types[:3], type_colors[:3])):
        stick = np.zeros(n_scans)
        stick[ons] = 1
        offset = idx * 1.5
        ax.plot(time, stick + offset, color=col, linewidth=1.5)
        ax.fill_between(time, offset, stick + offset, color=col, alpha=0.4)
        ax.text(-5, offset + 0.5, name, fontsize=12, fontweight='bold', color=col, ha='right')

    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Event occurs →', fontsize=11)
    ax.set_title('B.  Each trial becomes a "stick" — a spike at its onset time',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.set_xlim(-10, time[-1])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(70, 3.5,
            'Each trial gets its OWN\n'
            'onset marker. This is the\n'
            '"beta series" approach —\n'
            'every single picture\n'
            'presentation is tracked\n'
            'individually.',
            fontsize=11, color=C['dark'],
            bbox=dict(boxstyle='round', facecolor=C['light_bg'], edgecolor=C['dark']))

    # ── Panel C: After HRF convolution → regressors ──
    ax = fig.add_subplot(gs[2])
    hrf_t = np.arange(0, 20, TR)
    hrf_k = hrf(hrf_t)

    for idx, (ons, name, col) in enumerate(zip(onsets[:3], types[:3], type_colors[:3])):
        stick = np.zeros(n_scans)
        stick[ons] = 1
        regressor = np.convolve(stick, hrf_k)[:n_scans]
        regressor = regressor / np.max(regressor) if np.max(regressor) > 0 else regressor
        offset = idx * 1.5
        ax.plot(time, regressor + offset, color=col, linewidth=2.5)
        ax.fill_between(time, offset, regressor + offset, color=col, alpha=0.2)
        ax.text(-5, offset + 0.5, name, fontsize=12, fontweight='bold', color=col, ha='right')
        ax.axhline(offset, color='gray', linewidth=0.3, linestyle=':')

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Predicted BOLD →', fontsize=12)
    ax.set_title('C.  After HRF convolution → each stick becomes a predicted BOLD response (a "regressor")',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.set_xlim(-10, time[-1])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('This bump is the HRF shape\n— the model\'s prediction of\n'
                'what the blood flow response\nshould look like for this trial',
                xy=(onsets[0]*TR + 10, 0.85), xytext=(50, 1.2),
                fontsize=10, color=C['dark'],
                arrowprops=dict(arrowstyle='->', color=C['dark'], lw=1.5))

    ax.text(70, 3.8,
            'Each regressor is a column\n'
            'in the design matrix X.\n\n'
            'With 60 trials per run,\n'
            'we get 60 columns like this\n'
            '+ 6 motion columns\n'
            '= 66 total columns.',
            fontsize=11, color=C['dark'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#D5F5E3',
                      edgecolor=C['good'], linewidth=1.5))

    # ── Panel D: Arrow + explanation ──
    ax = fig.add_subplot(gs[3])
    ax.axis('off')
    ax.text(0.5, 0.5,
            'NEXT STEP:  These regressors become the columns of the design matrix X.\n'
            'The GLM then finds the best "weight" (beta, β) for each regressor that,\n'
            'when summed together, best matches the observed BOLD signal at each voxel.',
            transform=ax.transAxes, fontsize=13, ha='center', va='center',
            color=C['dark'], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#EBF5FB',
                      edgecolor='#2980B9', linewidth=2))

    savefig(fig, '03_from_events_to_regressors.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 4: Standard GLM vs Beta Series
# ════════════════════════════════════════════════════════════════
def fig04():
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('4. Standard GLM vs. Beta Series — Why Each Trial Needs Its Own Regressor',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.3, top=0.92, bottom=0.05,
                           left=0.06, right=0.96)

    TR = 1.98
    n_scans = 80
    time = np.arange(n_scans)
    hrf_t = np.arange(0, 15, 1.0)
    hrf_k = hrf(hrf_t)

    # Trial onsets (in scan units)
    pl_onsets = [5, 22, 40, 58, 72]
    nt_onsets = [12, 30, 48, 65]
    up_onsets = [8, 18, 35, 52, 68]

    # ── Top Left: Standard GLM ──
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('STANDARD GLM\n(traditional approach)', fontsize=15,
                 fontweight='bold', color=C['bad'])

    # 3 condition regressors + 1 motion example
    regressors_std = np.zeros((n_scans, 4))
    for ons in pl_onsets:
        length = min(len(hrf_k), n_scans - ons)
        regressors_std[ons:ons+length, 0] += hrf_k[:length] * 0.8
    for ons in nt_onsets:
        length = min(len(hrf_k), n_scans - ons)
        regressors_std[ons:ons+length, 1] += hrf_k[:length] * 0.8
    for ons in up_onsets:
        length = min(len(hrf_k), n_scans - ons)
        regressors_std[ons:ons+length, 2] += hrf_k[:length] * 0.8
    regressors_std[:, 3] = np.cumsum(np.random.randn(n_scans) * 0.02)  # motion

    names = ['All\nPleasant', 'All\nNeutral', 'All\nUnpleasant', 'Motion\n(1 of 6)']
    cols = [C['pl'], C['nt'], C['up'], C['motion']]

    for i in range(4):
        for t_idx in range(n_scans):
            val = regressors_std[t_idx, i]
            if abs(val) > 0.01:
                ax.barh(n_scans - t_idx, val, height=0.8, left=i*1.2,
                        color=cols[i], alpha=min(abs(val) + 0.2, 1))

    for i, (name, col) in enumerate(zip(names, cols)):
        ax.text(i*1.2 + 0.2, n_scans + 3, name, ha='center', fontsize=9,
                fontweight='bold', color=col)

    ax.set_xlabel('← Regressors (columns of X) →', fontsize=11)
    ax.set_ylabel('Time (scans) ↓', fontsize=11)
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-2, n_scans + 8)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Top Right: Beta Series ──
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('BETA SERIES (our method)\n(each trial = its own regressor)', fontsize=15,
                 fontweight='bold', color=C['good'])

    all_onsets = sorted([(o, 'pl') for o in pl_onsets] +
                        [(o, 'nt') for o in nt_onsets] +
                        [(o, 'up') for o in up_onsets])

    n_trials = len(all_onsets)
    regressors_bs = np.zeros((n_scans, n_trials + 1))
    for i, (ons, _) in enumerate(all_onsets):
        length = min(len(hrf_k), n_scans - ons)
        regressors_bs[ons:ons+length, i] = hrf_k[:length] * 0.8
    regressors_bs[:, -1] = regressors_std[:, 3]  # same motion

    trial_cols = [C['pl'] if t[1]=='pl' else C['nt'] if t[1]=='nt' else C['up'] for t in all_onsets]
    trial_cols.append(C['motion'])

    for i in range(n_trials + 1):
        for t_idx in range(n_scans):
            val = regressors_bs[t_idx, i]
            if abs(val) > 0.01:
                ax.barh(n_scans - t_idx, val * 0.5, height=0.8,
                        left=i * 0.55,
                        color=trial_cols[i], alpha=min(abs(val) + 0.2, 1))

    # Labels for first few
    for i in range(min(3, n_trials)):
        tp = all_onsets[i][1].capitalize()
        ax.text(i*0.55 + 0.1, n_scans + 3, f'{tp}\n#{i+1}', ha='center',
                fontsize=7, color=trial_cols[i], fontweight='bold')
    ax.text(n_trials*0.55/2, n_scans + 8, f'← {n_trials} individual trial regressors →',
            ha='center', fontsize=10, color=C['good'], fontweight='bold')

    ax.set_xlabel('← Regressors (one per trial!) →', fontsize=11)
    ax.set_ylabel('Time (scans) ↓', fontsize=11)
    ax.set_xlim(-0.3, (n_trials+1)*0.55 + 0.5)
    ax.set_ylim(-2, n_scans + 12)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Bottom Left: Standard GLM output ──
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')

    ax.text(0.5, 0.85, 'OUTPUT: Only 3 beta values per voxel',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', color=C['bad'])

    box_text = (
        "β₁ = average response to ALL pleasant pictures\n"
        "β₂ = average response to ALL neutral pictures\n"
        "β₃ = average response to ALL unpleasant pictures\n\n"
        "───────────────────────────────────────\n\n"
        "PROBLEM FOR MVPA:\n"
        "With only 3 values per voxel, we have\n"
        "only 3 data points to train a classifier.\n\n"
        "That's like trying to learn someone's face\n"
        "from 3 photos — not enough information!\n"
        "The classifier would just memorize them."
    )
    ax.text(0.5, 0.35, box_text, transform=ax.transAxes, fontsize=12,
            ha='center', va='center', color=C['dark'], linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FADBD8',
                      edgecolor=C['bad'], linewidth=2))

    # ── Bottom Right: Beta Series output ──
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')

    ax.text(0.5, 0.85, f'OUTPUT: {n_trials} beta values per voxel (per run!)',
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', color=C['good'])

    box_text = (
        "β₁  = response to Pleasant picture #1\n"
        "β₂  = response to Pleasant picture #2\n"
        "...  (one β per individual picture shown)\n"
        f"β{n_trials} = response to Unpleasant picture #5\n\n"
        "───────────────────────────────────────\n\n"
        "With 5 runs × 60 trials = 300 betas per voxel!\n\n"
        "For MVPA, each trial's beta pattern across\n"
        "all voxels in an ROI becomes one \"sample\"\n"
        "for the classifier — like having 300 photos\n"
        "to learn from instead of 3!"
    )
    ax.text(0.5, 0.35, box_text, transform=ax.transAxes, fontsize=12,
            ha='center', va='center', color=C['dark'], linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#D5F5E3',
                      edgecolor=C['good'], linewidth=2))

    savefig(fig, '04_standard_vs_beta_series_glm.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 5: The GLM Equation Explained
# ════════════════════════════════════════════════════════════════
def fig05():
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('5. The GLM Equation — How Beta Values Are Calculated',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(4, 1, hspace=0.5, top=0.93, bottom=0.04,
                           left=0.06, right=0.96, height_ratios=[0.6, 1.2, 1.4, 0.8])

    # ── Panel A: The equation with everyday analogy ──
    ax = fig.add_subplot(gs[0])
    ax.axis('off')

    eq_text = (
        "The GLM equation for ONE voxel:\n\n"
        r"y  =  X · β  +  ε"
    )
    ax.text(0.5, 0.7, eq_text, transform=ax.transAxes, fontsize=22,
            ha='center', va='center', color=C['dark'], fontweight='bold',
            fontfamily='monospace')

    parts = (
        "y = what we measured              (the BOLD signal at this voxel over time — T values)\n"
        "X = what we think happened         (the design matrix — T rows × P regressor columns)\n"
        "β = how strongly it responded      (the beta weights — one per regressor — what we SOLVE for)\n"
        "ε = what we can't explain          (noise / error — the leftover)"
    )
    ax.text(0.5, 0.15, parts, transform=ax.transAxes, fontsize=12,
            ha='center', va='center', color=C['dark'], fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#EBF5FB', edgecolor='#2980B9'))

    # ── Panel B: Visual matrix multiplication ──
    ax = fig.add_subplot(gs[1])
    ax.axis('off')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8)

    # y vector
    y_vals = [1002, 1005, 1001, 1008, 1003, '...', 998]
    ax.text(1.5, 7.5, 'y', fontsize=18, fontweight='bold', color=C['dark'], ha='center')
    ax.text(1.5, 7.0, '(measured)', fontsize=9, color='#777', ha='center')
    for i, v in enumerate(y_vals):
        rect = Rectangle((0.5, 6.2 - i*0.75), 2, 0.65,
                          facecolor='#D6EAF8', edgecolor='#2980B9', linewidth=1)
        ax.add_patch(rect)
        ax.text(1.5, 6.2 - i*0.75 + 0.32, str(v), ha='center', va='center',
                fontsize=10, color=C['dark'])

    # = sign
    ax.text(3.5, 3.5, '=', fontsize=28, fontweight='bold', color=C['dark'],
            ha='center', va='center')

    # X matrix
    ax.text(7.5, 7.5, 'X', fontsize=18, fontweight='bold', color=C['dark'], ha='center')
    ax.text(7.5, 7.0, '(design matrix)', fontsize=9, color='#777', ha='center')

    col_names = ['Pl1', 'Nt1', 'Up1', '...', 'Mot_X']
    col_colors = [C['pl'], C['nt'], C['up'], '#555', C['motion']]
    for j, (cn, cc) in enumerate(zip(col_names, col_colors)):
        ax.text(5 + j*1.2 + 0.5, 6.6, cn, ha='center', fontsize=8,
                fontweight='bold', color=cc, rotation=45)

    np.random.seed(10)
    for i in range(7):
        for j in range(5):
            val = np.random.rand() * 0.5 if j < 3 else (0 if j == 3 else np.random.rand()*0.3)
            intensity = min(val * 2, 1)
            rect = Rectangle((5 + j*1.2, 6.2 - i*0.75), 1.1, 0.65,
                              facecolor=plt.cm.YlOrRd(intensity),
                              edgecolor='white', linewidth=1)
            ax.add_patch(rect)
            if j == 3:
                ax.text(5 + j*1.2 + 0.55, 6.2 - i*0.75 + 0.32, '...',
                        ha='center', va='center', fontsize=10)

    # × sign
    ax.text(11.5, 3.5, '×', fontsize=28, fontweight='bold', color=C['dark'],
            ha='center', va='center')

    # β vector
    ax.text(13.5, 7.5, 'β', fontsize=18, fontweight='bold', color=C['dark'], ha='center')
    ax.text(13.5, 7.0, '(solve for these!)', fontsize=9, color=C['good'], ha='center',
            fontweight='bold')

    beta_names = ['β_Pl1', 'β_Nt1', 'β_Up1', '...', 'β_Mot']
    beta_colors = [C['pl'], C['nt'], C['up'], '#555', C['motion']]
    for i, (bn, bc) in enumerate(zip(beta_names, beta_colors)):
        rect = Rectangle((12.5, 6.2 - i*1.1), 2, 0.9,
                          facecolor='#D5F5E3', edgecolor=C['good'], linewidth=1.5)
        ax.add_patch(rect)
        ax.text(13.5, 6.2 - i*1.1 + 0.45, bn, ha='center', va='center',
                fontsize=11, fontweight='bold', color=bc)

    # + ε
    ax.text(16, 3.5, '+  ε', fontsize=22, fontweight='bold', color='#95A5A6',
            ha='center', va='center')
    ax.text(16, 2.5, '(noise)', fontsize=10, color='#95A5A6', ha='center')

    # ── Panel C: How the math works for one voxel ──
    ax = fig.add_subplot(gs[2])
    ax.axis('off')

    explanation = (
        "HOW DOES THE COMPUTER FIND β?\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Think of it like mixing paint colors:\n\n"
        "  • You have a final color (y = the measured BOLD signal)\n"
        "  • You know what pure colors are available (X = the predicted response shapes)\n"
        "  • You need to figure out HOW MUCH of each color was mixed in (β = the weights)\n\n"
        "The computer finds the β values that make X·β match y as closely as possible.\n"
        "Mathematically:  β̂ = (X'X)⁻¹ X'y   (least squares — minimize the leftover error ε)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "CONCRETE EXAMPLE at one voxel:\n\n"
        "  y(t) = 3.5 × HRF_shape_Pl1(t)  +  1.2 × HRF_shape_Nt1(t)  +  4.1 × HRF_shape_Up1(t)  + ...\n"
        "         ╰── β_Pl1 = 3.5 ──╯       ╰── β_Nt1 = 1.2 ──╯       ╰── β_Up1 = 4.1 ──╯\n\n"
        "  This voxel responded STRONGLY to Unpleasant #1 (β=4.1),\n"
        "  MODERATELY to Pleasant #1 (β=3.5),\n"
        "  and WEAKLY to Neutral #1 (β=1.2).\n\n"
        "  ➜  β is the estimated BOLD AMPLITUDE for that trial at that voxel."
    )
    ax.text(0.5, 0.5, explanation, transform=ax.transAxes, fontsize=12,
            ha='center', va='center', color=C['dark'], fontfamily='monospace',
            linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FDFEFE',
                      edgecolor=C['dark'], linewidth=1.5))

    # ── Panel D: What about motion betas? ──
    ax = fig.add_subplot(gs[3])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "WHAT HAPPENS TO THE MOTION BETAS?\n\n"
            "The motion regressors (head movement over time) are in the design matrix too.\n"
            "The GLM gives them their own β weights, which \"absorb\" any signal caused by head motion.\n"
            "We simply THROW AWAY the motion β's — we only keep the trial β's (β_Pl1, β_Nt1, etc.).\n\n"
            "This way, the trial β's are CLEANED of motion contamination.\n"
            "It's like running a spell-checker that removes typos from your essay — you keep the clean text.",
            transform=ax.transAxes, fontsize=12, ha='center', va='center',
            color=C['dark'], linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.7', facecolor='#FDF2E9',
                      edgecolor=C['motion'], linewidth=2))

    savefig(fig, '05_the_glm_equation.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 6: What Is a Beta Image
# ════════════════════════════════════════════════════════════════
def fig06():
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle('6. What Is a Beta Image? — From Equation to Brain Map',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(2, 1, hspace=0.4, top=0.92, bottom=0.05,
                           left=0.05, right=0.97, height_ratios=[1.2, 1])

    # ── Top: solving GLM at many voxels ──
    ax = fig.add_subplot(gs[0])
    ax.axis('off')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)

    # Draw 5 "voxels" each with their own equation
    voxel_x = [0.5, 3.8, 7.1, 10.4, 13.7]
    beta_vals = [3.5, 0.2, -0.8, 2.1, 4.7]
    voxel_cols = ['#E74C3C', '#F5B041', '#AEB6BF', '#5DADE2', '#E74C3C']

    ax.text(9, 9.5, 'The GLM is solved INDEPENDENTLY at every single voxel in the brain',
            fontsize=14, fontweight='bold', ha='center', color=C['dark'])

    for i, (vx, bv, vc) in enumerate(zip(voxel_x, beta_vals, voxel_cols)):
        # Voxel box
        rect = FancyBboxPatch((vx, 5.5), 2.8, 3.2, boxstyle='round,pad=0.2',
                               facecolor=vc, alpha=0.15, edgecolor=vc, linewidth=2)
        ax.add_patch(rect)

        ax.text(vx+1.4, 8.3, f'Voxel #{i+1}', fontsize=10, ha='center',
                fontweight='bold', color=vc)
        ax.text(vx+1.4, 7.5, f'y = Xβ + ε', fontsize=9, ha='center',
                fontfamily='monospace', color=C['dark'])
        ax.text(vx+1.4, 6.8, f'solve → β̂ = {bv}', fontsize=10, ha='center',
                fontweight='bold', fontfamily='monospace',
                color=C['good'] if bv > 1 else C['bad'] if bv < 0 else '#777')
        ax.text(vx+1.4, 6.1, f'(for trial Pl1)', fontsize=8, ha='center', color='#777')

    # Arrow down
    ax.annotate('', xy=(9, 4.8), xytext=(9, 5.3),
                arrowprops=dict(arrowstyle='->', color=C['dark'], lw=3))
    ax.text(9, 4.5, 'Collect all β̂_Pl1 values across every voxel in the brain...',
            fontsize=12, ha='center', fontweight='bold', color=C['dark'])

    # Beta image
    np.random.seed(5)
    sz = 12
    beta_img = np.random.randn(sz, sz) * 1.5
    mask = np.zeros((sz, sz), dtype=bool)
    for yi in range(sz):
        for xi in range(sz):
            if ((xi-sz/2+0.5)**2 + (yi-sz/2+0.5)**2) < (sz/2.3)**2:
                mask[yi, xi] = True
    beta_img[~mask] = np.nan

    # Draw the beta image
    ax_img = fig.add_axes([0.3, 0.1, 0.15, 0.25])  # manual position
    im = ax_img.imshow(beta_img, cmap='RdBu_r', vmin=-4, vmax=4, interpolation='nearest')
    ax_img.set_title('beta_0001.nii\n(Pleasant trial #1)', fontsize=11,
                     fontweight='bold', color=C['pl'])
    ax_img.axis('off')
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04, shrink=0.8)
    cbar.set_label('β value', fontsize=10)

    # ── Bottom: explanation ──
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('off')

    text = (
        "WHAT DOES A BETA IMAGE TELL US?\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "Each beta image is a 3D brain map where every voxel's value is that\n"
        "voxel's estimated BOLD response amplitude for ONE specific trial.\n\n"
        "  • β = 3.5 at some voxel → that voxel responded STRONGLY to this picture\n"
        "  • β = 0.2 at another    → barely responded\n"
        "  • β = -0.8 somewhere    → responded LESS than the implicit baseline (fixation)\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "THE IMPLICIT BASELINE (why we don't compare to rest):\n\n"
        "The GLM's constant term captures the average signal level during\n"
        "\"nothing happening\" (fixation between trials). Each trial's β is\n"
        "RELATIVE TO THIS BASELINE automatically. We don't need to subtract\n"
        "a separate rest condition — it's built into the math.\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        "For MVPA, we don't care about the absolute β value at any voxel.\n"
        "We care about the PATTERN of β's across voxels — that's next!"
    )
    ax2.text(0.55, 0.5, text, transform=ax2.transAxes, fontsize=12,
             ha='center', va='center', color=C['dark'], fontfamily='monospace',
             linespacing=1.5,
             bbox=dict(boxstyle='round,pad=0.8', facecolor=C['light_bg'],
                       edgecolor=C['dark'], linewidth=1.5))

    savefig(fig, '06_what_is_a_beta_image.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 7: Motion Correction
# ════════════════════════════════════════════════════════════════
def fig07():
    fig, axes = plt.subplots(3, 1, figsize=(16, 16),
                              gridspec_kw={'height_ratios': [1, 1, 0.8]})
    fig.suptitle('7. How Head Motion Is Removed — Two Lines of Defense',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    plt.subplots_adjust(hspace=0.45, top=0.92, bottom=0.06, left=0.1, right=0.92)

    TR = 1.98
    n = 100
    time = np.arange(n) * TR

    np.random.seed(33)

    # ── Panel A: physical realignment ──
    ax = axes[0]
    # Simulated motion trace
    motion_x = np.cumsum(np.random.randn(n) * 0.04) + np.sin(np.arange(n)*0.08) * 0.3
    motion_y = np.cumsum(np.random.randn(n) * 0.03)
    motion_z = np.sin(np.arange(n)*0.05) * 0.2

    ax.plot(time, motion_x, label='X translation (mm)', linewidth=2, color='#E74C3C')
    ax.plot(time, motion_y, label='Y translation (mm)', linewidth=2, color='#27AE60')
    ax.plot(time, motion_z, label='Z translation (mm)', linewidth=2, color='#2980B9')
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_ylabel('Displacement (mm)', fontsize=12)
    ax.set_title('A.  Line of Defense #1 — Physical Realignment (during preprocessing)',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(time[-1]*0.5, max(motion_x)*1.2,
            'These are the rp_*.txt files — they record how much\n'
            'the head moved at each time point. SPM physically\n'
            'shifts each brain volume back to undo the movement.\n'
            'But this correction is IMPERFECT...',
            fontsize=11, ha='center', color=C['dark'],
            bbox=dict(boxstyle='round', facecolor='#FEF9E7', edgecolor=C['motion']))

    # ── Panel B: motion regressors in GLM ──
    ax = axes[1]

    # True neural signal
    true_signal = np.zeros(n)
    for ons in [10, 25, 42, 58, 75, 90]:
        ht = np.arange(0, min(12, n-ons))
        true_signal[ons:ons+len(ht)] += hrf(ht) * 4

    # Residual motion artifact (what realignment didn't fix)
    residual_motion = 0.5 * motion_x + 0.3 * np.gradient(motion_x) * 10
    noise = np.random.randn(n) * 1.0

    contaminated = true_signal + residual_motion + noise
    cleaned = true_signal + noise * 0.8  # after regression

    ax.plot(time, contaminated, color=C['bad'], linewidth=1.2, alpha=0.7,
            label='Signal BEFORE motion regression')
    ax.plot(time, cleaned, color=C['good'], linewidth=2,
            label='Signal AFTER motion regression (motion β\'s removed)')
    ax.plot(time, residual_motion, color=C['motion'], linewidth=1.5, linestyle='--',
            alpha=0.7, label='Residual motion artifact (absorbed by β_motion)')
    ax.axhline(0, color='black', linewidth=0.5)

    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Signal', fontsize=12)
    ax.set_title('B.  Line of Defense #2 — Motion Regressors in the GLM (during beta estimation)',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.legend(fontsize=10, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel C: How it works ──
    ax = axes[2]
    ax.axis('off')
    ax.text(0.5, 0.5,
            "HOW DO MOTION REGRESSORS WORK?\n\n"
            "The 6 motion traces (3 translations + 3 rotations) are added as extra\n"
            "columns in the design matrix X, alongside the trial regressors.\n\n"
            "When the GLM solves y = Xβ + ε, any signal that LOOKS LIKE head motion\n"
            "gets assigned to the motion β's, NOT to the trial β's.\n\n"
            "It's like having a translator who filters out background noise from a\n"
            "conversation — the motion regressors \"listen for\" movement patterns\n"
            "and remove them, leaving cleaner trial-specific signals.\n\n"
            "We then DISCARD the motion β's and keep only the trial β's for MVPA.",
            transform=ax.transAxes, fontsize=12, ha='center', va='center',
            color=C['dark'], linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FDF2E9',
                      edgecolor=C['motion'], linewidth=2))

    savefig(fig, '07_motion_correction.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 8: From Beta Images to Feature Vectors
# ════════════════════════════════════════════════════════════════
def fig08():
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle('8. From Beta Images to Feature Vectors — Preparing Data for the Classifier',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.92, bottom=0.04,
                           left=0.06, right=0.96, height_ratios=[1.2, 1.2, 0.8])

    np.random.seed(42)
    sz = 10

    # Create masks and patterns
    mask = np.zeros((sz, sz), dtype=bool)
    for yi in range(sz):
        for xi in range(sz):
            if ((xi-sz/2+0.5)**2 + (yi-sz/2+0.5)**2) < (sz/2.4)**2:
                mask[yi, xi] = True

    # ROI mask (smaller region)
    roi_mask = np.zeros((sz, sz), dtype=bool)
    for yi in range(sz):
        for xi in range(sz):
            if ((xi-3)**2 + (yi-4)**2) < 5:
                roi_mask[yi, xi] = True
    roi_mask = roi_mask & mask

    # ── Panel A: Beta image → ROI selection → vector ──
    ax = fig.add_subplot(gs[0])
    ax.axis('off')
    ax.set_xlim(0, 20)
    ax.set_ylim(-1, 8)

    # Full brain beta image
    beta_full = np.random.randn(sz, sz) * 2
    beta_full[~mask] = np.nan

    ax_img1 = fig.add_axes([0.07, 0.67, 0.14, 0.2])
    ax_img1.imshow(beta_full, cmap='RdBu_r', vmin=-4, vmax=4)
    ax_img1.set_title('Full brain\nbeta image\n(1 trial)', fontsize=10, fontweight='bold')
    ax_img1.axis('off')

    # Arrow
    ax_arr1 = fig.add_axes([0.22, 0.73, 0.06, 0.08])
    ax_arr1.axis('off')
    ax_arr1.annotate('', xy=(1, 0.5), xytext=(0, 0.5),
                     arrowprops=dict(arrowstyle='->', lw=3, color=C['dark']))
    ax_arr1.text(0.5, -0.1, 'Apply\nROI mask', ha='center', fontsize=9, color=C['dark'],
                 fontweight='bold')

    # ROI-masked image
    beta_roi = beta_full.copy()
    beta_roi[~roi_mask] = np.nan
    # Show ROI outline on full brain
    roi_display = np.full((sz, sz), np.nan)
    roi_display[roi_mask] = beta_full[roi_mask]

    ax_img2 = fig.add_axes([0.29, 0.67, 0.14, 0.2])
    ax_img2.imshow(np.where(mask, 0.95, np.nan) * np.ones((sz,sz)),
                   cmap='Greys', vmin=0, vmax=1, alpha=0.3)
    ax_img2.imshow(roi_display, cmap='RdBu_r', vmin=-4, vmax=4)
    ax_img2.set_title('Only voxels\ninside ROI\n(e.g., V1v)', fontsize=10, fontweight='bold')
    ax_img2.axis('off')

    # Arrow
    ax_arr2 = fig.add_axes([0.44, 0.73, 0.06, 0.08])
    ax_arr2.axis('off')
    ax_arr2.annotate('', xy=(1, 0.5), xytext=(0, 0.5),
                     arrowprops=dict(arrowstyle='->', lw=3, color=C['dark']))
    ax_arr2.text(0.5, -0.1, 'Flatten\nto vector', ha='center', fontsize=9, color=C['dark'],
                 fontweight='bold')

    # Feature vector
    n_roi_voxels = int(roi_mask.sum())
    roi_values = beta_full[roi_mask]

    ax_vec = fig.add_axes([0.52, 0.7, 0.43, 0.15])
    bars = ax_vec.bar(range(n_roi_voxels), roi_values, color=C['pl'], alpha=0.7,
                       edgecolor='white')
    ax_vec.axhline(0, color='black', linewidth=0.5)
    ax_vec.set_xlabel(f'Voxel index (1 to {n_roi_voxels})', fontsize=10)
    ax_vec.set_ylabel('β value', fontsize=10)
    ax_vec.set_title(f'Feature vector for this trial — [{n_roi_voxels} values]',
                     fontsize=11, fontweight='bold', color=C['pl'])
    ax_vec.spines['top'].set_visible(False)
    ax_vec.spines['right'].set_visible(False)

    # ── Panel B: Many trials → data matrix ──
    ax = fig.add_subplot(gs[1])
    ax.axis('off')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)

    ax.text(9, 9.5, 'Repeat for ALL trials → build the data matrix for classification',
            fontsize=14, fontweight='bold', ha='center', color=C['dark'])

    # Draw the matrix
    n_show_trials = 10
    n_show_voxels = 15

    matrix_data = np.random.randn(n_show_trials, n_show_voxels) * 2
    trial_labels = ['Pl #1','Pl #2','Pl #3','Pl #4','Pl #5',
                    'Nt #1','Nt #2','Nt #3','Nt #4','Nt #5']
    label_colors = [C['pl']]*5 + [C['nt']]*5

    cell_w, cell_h = 0.6, 0.6
    start_x, start_y = 3, 1

    # Column headers
    for j in range(n_show_voxels):
        ax.text(start_x + j*cell_w + cell_w/2, start_y + n_show_trials*cell_h + 0.3,
                f'V{j+1}', fontsize=7, ha='center', color='#555')
    ax.text(start_x + n_show_voxels*cell_w/2, start_y + n_show_trials*cell_h + 0.8,
            f'← {n_roi_voxels} voxels in ROI →', fontsize=11, ha='center',
            fontweight='bold', color=C['dark'])

    # Row labels and cells
    for i in range(n_show_trials):
        ax.text(start_x - 0.3, start_y + (n_show_trials - i - 0.5)*cell_h,
                trial_labels[i], fontsize=9, ha='right', color=label_colors[i],
                fontweight='bold')
        for j in range(n_show_voxels):
            val = matrix_data[i, j]
            intensity = np.clip((val + 4) / 8, 0, 1)
            rect = Rectangle((start_x + j*cell_w, start_y + (n_show_trials-i-1)*cell_h),
                              cell_w, cell_h, facecolor=plt.cm.RdBu_r(intensity),
                              edgecolor='white', linewidth=0.5)
            ax.add_patch(rect)

    # Bracket for labels
    ax.text(1.0, start_y + n_show_trials*cell_h/2,
            f'100 Pleasant\ntrials\n+\n100 Neutral\ntrials\n=\n200 samples',
            fontsize=10, ha='center', va='center', color=C['dark'],
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#2980B9'))

    # Arrow to labels column
    ax.text(start_x + n_show_voxels*cell_w + 0.8,
            start_y + n_show_trials*cell_h/2,
            'Label\n\n+1\n+1\n+1\n+1\n+1\n−1\n−1\n−1\n−1\n−1',
            fontsize=9, ha='center', va='center', fontfamily='monospace',
            fontweight='bold', color=C['dark'],
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor=C['good']))

    # ── Panel C: Summary ──
    ax = fig.add_subplot(gs[2])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "WHAT WE NOW HAVE FOR THE CLASSIFIER:\n\n"
            f"• A data matrix:  200 rows (trials) × {n_roi_voxels} columns (voxels in this ROI)\n"
            "• A label vector:   200 values of +1 (Pleasant) or −1 (Neutral)\n\n"
            "Each ROW is one trial's spatial pattern of brain activity.\n"
            "The classifier's job: learn which patterns belong to Pleasant and which to Neutral.\n\n"
            "This is repeated for each ROI (V1v, V1d, V2v, ... IPS) and each subject (1–20).",
            transform=ax.transAxes, fontsize=13, ha='center', va='center',
            color=C['dark'], linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#D5F5E3',
                      edgecolor=C['good'], linewidth=2))

    savefig(fig, '08_from_betas_to_patterns.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 9: SVM Classification Explained
# ════════════════════════════════════════════════════════════════
def fig09():
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle('9. SVM Classification — How the Classifier Learns to Decode Emotion',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.3, top=0.92, bottom=0.05,
                           left=0.06, right=0.96)

    np.random.seed(21)

    # ── Panel A: 2D intuition ──
    ax = fig.add_subplot(gs[0, 0])

    # Two clusters
    pl_x = np.random.randn(40) * 0.7 + 2.5
    pl_y = np.random.randn(40) * 0.6 + 2.0
    nt_x = np.random.randn(40) * 0.7 - 0.5
    nt_y = np.random.randn(40) * 0.6 + 0.0

    ax.scatter(pl_x, pl_y, c=C['pl'], s=60, alpha=0.6, edgecolors='white',
               linewidth=0.5, zorder=3, label='Pleasant trials')
    ax.scatter(nt_x, nt_y, c=C['nt'], s=60, alpha=0.6, edgecolors='white',
               linewidth=0.5, zorder=3, label='Neutral trials')

    # Hyperplane
    x_line = np.linspace(-3, 5.5, 100)
    y_line = -0.7 * x_line + 1.8
    ax.plot(x_line, y_line, 'k-', linewidth=3, label='Decision boundary', zorder=4)
    ax.fill_between(x_line, y_line, 5, alpha=0.05, color=C['pl'])
    ax.fill_between(x_line, y_line, -3, alpha=0.05, color=C['nt'])

    # Labels on sides
    ax.text(3.5, 3.5, 'PLEASANT\nSIDE', fontsize=14, fontweight='bold',
            color=C['pl'], ha='center', alpha=0.5)
    ax.text(-1, -1.5, 'NEUTRAL\nSIDE', fontsize=14, fontweight='bold',
            color=C['nt'], ha='center', alpha=0.5)

    # New test point
    ax.scatter([1.8], [1.5], c='gold', s=200, marker='*', edgecolors='black',
               linewidth=2, zorder=5)
    ax.annotate('New trial to classify:\nWhich side does it fall on?\n→ PLEASANT!',
                xy=(1.8, 1.5), xytext=(3.5, -0.5), fontsize=10, fontweight='bold',
                color=C['dark'],
                arrowprops=dict(arrowstyle='->', color=C['dark'], lw=2))

    ax.set_xlabel('Voxel pattern dimension 1', fontsize=11)
    ax.set_ylabel('Voxel pattern dimension 2', fontsize=11)
    ax.set_title('A.  The basic idea (simplified to 2 dimensions)',
                 fontsize=13, fontweight='bold', color=C['dark'], loc='left')
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-3, 5.5)
    ax.set_ylim(-2.5, 4.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: What it really is (high-dimensional) ──
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "WHAT'S REALLY HAPPENING\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "In our data, each trial is not a 2D point\n"
            "but a point in 500-DIMENSIONAL space\n"
            "(one dimension per voxel in the ROI).\n\n"
            "We can't visualize 500 dimensions, but\n"
            "the math works the same way:\n\n"
            "The SVM finds a FLAT SURFACE (hyperplane)\n"
            "in this 500D space that best separates\n"
            "Pleasant trials from Neutral trials.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "For a new trial, check which SIDE of the\n"
            "surface it falls on → that's the prediction.\n\n"
            "DECISION RULE:\n"
            "  f(x) = w · x + b\n"
            "  if f(x) > 0 → Pleasant\n"
            "  if f(x) < 0 → Neutral\n\n"
            "w = a vector of weights (one per voxel)\n"
            "    learned during training\n"
            "x = the new trial's beta pattern\n"
            "b = bias term",
            transform=ax.transAxes, fontsize=11, ha='center', va='center',
            color=C['dark'], fontfamily='monospace', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#EBF5FB',
                      edgecolor='#2980B9', linewidth=1.5))

    # ── Panel C: What "linear" and "standardize" mean ──
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "KEY SETTINGS IN OUR SVM\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "KERNEL = 'linear'\n"
            "─────────────────\n"
            "The decision boundary is a FLAT surface\n"
            "(not curved). This means the classifier\n"
            "looks for a simple weighted combination\n"
            "of voxels that separates the classes.\n\n"
            "Why linear? With ~500 voxels and only\n"
            "200 trials, a complex boundary would\n"
            "just memorize noise (overfitting).\n\n\n"
            "STANDARDIZE = true\n"
            "──────────────────\n"
            "Before training, each voxel's values are\n"
            "rescaled to have mean=0 and std=1.\n\n"
            "Why? Without this, a voxel with large\n"
            "raw values (e.g., near a blood vessel)\n"
            "would dominate the classifier, even if\n"
            "it carries no useful information.\n\n"
            "Standardization puts every voxel on\n"
            "an equal footing.",
            transform=ax.transAxes, fontsize=11, ha='center', va='center',
            color=C['dark'], fontfamily='monospace', linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FEF9E7',
                      edgecolor=C['accent'], linewidth=1.5))

    # ── Panel D: What the weight vector tells us ──
    ax = fig.add_subplot(gs[1, 1])

    np.random.seed(55)
    n_vox = 20
    weights = np.random.randn(n_vox) * 0.8
    weights[3] = 2.5   # strongly positive
    weights[7] = -2.0  # strongly negative
    weights[12] = 1.8

    bar_colors = [C['pl'] if w > 0.5 else C['nt'] if w < -0.5 else '#BDC3C7' for w in weights]
    ax.bar(range(n_vox), weights, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', linewidth=1)

    ax.annotate('This voxel strongly\n"votes" for Pleasant',
                xy=(3, 2.5), xytext=(6, 3.2), fontsize=10, color=C['pl'],
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['pl'], lw=2))

    ax.annotate('This voxel strongly\n"votes" for Neutral',
                xy=(7, -2.0), xytext=(10, -2.8), fontsize=10, color=C['nt'],
                fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=C['nt'], lw=2))

    ax.set_xlabel('Voxel index in ROI', fontsize=11)
    ax.set_ylabel('SVM weight (w)', fontsize=11)
    ax.set_title('D.  The learned weight vector w — which voxels matter?',
                 fontsize=13, fontweight='bold', color=C['dark'], loc='left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(10, 3.8,
            'Each voxel gets a weight.\n'
            'Positive = "evidence for Pleasant"\n'
            'Negative = "evidence for Neutral"\n'
            'Near zero = "not informative"',
            fontsize=10, ha='center', color=C['dark'],
            bbox=dict(boxstyle='round', facecolor=C['light_bg']))

    savefig(fig, '09_svm_classification.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 10: Cross-Validation
# ════════════════════════════════════════════════════════════════
def fig10():
    fig = plt.figure(figsize=(18, 18))
    fig.suptitle('10. Cross-Validation — Making Sure the Classifier Isn\'t Just Memorizing',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(3, 1, hspace=0.45, top=0.92, bottom=0.04,
                           left=0.06, right=0.96, height_ratios=[1.2, 1.0, 1.0])

    # ── Panel A: The overfitting problem ──
    ax = fig.add_subplot(gs[0])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "WHY CAN'T WE JUST TRAIN AND TEST ON THE SAME DATA?\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "Imagine a student who memorizes all the answers to practice problems.\n"
            "They score 100% on those exact problems — but fail the real exam\n"
            "because they never actually LEARNED the material.\n\n"
            "This is OVERFITTING. The classifier memorizes the training data\n"
            "(including its noise) instead of learning the real patterns.\n\n"
            "SOLUTION: Cross-validation — train on SOME data, test on HELD-OUT data.\n"
            "This way, accuracy measures the classifier's ability to generalize\n"
            "to NEW trials it has never seen before.\n\n"
            "If accuracy on held-out data is above 50% (chance for 2 classes),\n"
            "the ROI truly contains information that distinguishes the conditions.",
            transform=ax.transAxes, fontsize=13, ha='center', va='center',
            color=C['dark'], linespacing=1.6,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FADBD8',
                      edgecolor=C['bad'], linewidth=2))

    # ── Panel B: 5-fold CV diagram ──
    ax = fig.add_subplot(gs[1])
    ax.axis('off')
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 8)

    ax.text(9, 7.5, '5-FOLD CROSS-VALIDATION', fontsize=16, fontweight='bold',
            ha='center', color=C['dark'])
    ax.text(9, 6.9, '200 trials split into 5 equal groups (40 each). Each fold takes a turn as test set.',
            fontsize=11, ha='center', color='#555')

    fold_colors_train = '#D5F5E3'
    fold_color_test = '#FADBD8'

    for fold in range(5):
        y_pos = 5.5 - fold * 1.2
        ax.text(0.5, y_pos + 0.2, f'Fold {fold+1}:', fontsize=11, fontweight='bold',
                va='center', color=C['dark'])

        for block in range(5):
            x_pos = 2.5 + block * 2.8
            is_test = (block == fold)
            color = fold_color_test if is_test else fold_colors_train
            label = 'TEST' if is_test else 'TRAIN'
            edge = C['bad'] if is_test else C['good']

            rect = FancyBboxPatch((x_pos, y_pos - 0.1), 2.4, 0.65,
                                   boxstyle='round,pad=0.1',
                                   facecolor=color, edgecolor=edge, linewidth=2)
            ax.add_patch(rect)
            ax.text(x_pos + 1.2, y_pos + 0.22, label, fontsize=9, ha='center',
                    va='center', fontweight='bold',
                    color=C['bad'] if is_test else C['good'])

        # Accuracy
        acc = np.random.uniform(0.58, 0.72)
        ax.text(16.5, y_pos + 0.2, f'→ {acc:.0%}', fontsize=11, fontweight='bold',
                va='center', color=C['dark'])

    ax.text(9, -0.3, 'Each fold: train SVM on 160 trials (4 groups), test on 40 held-out trials',
            fontsize=11, ha='center', color='#555', fontstyle='italic')

    # ── Panel C: Repeated CV ──
    ax = fig.add_subplot(gs[2])

    np.random.seed(77)
    n_reps = 100
    rep_means = np.random.beta(14, 8, n_reps) * 0.3 + 0.5

    ax.plot(range(1, n_reps+1), rep_means, 'o-', markersize=3, linewidth=0.8,
            color='#3498DB', alpha=0.7)
    ax.axhline(np.mean(rep_means), color=C['good'], linewidth=2.5, linestyle='-',
               label=f'Grand mean = {np.mean(rep_means):.1%}')
    ax.axhline(0.5, color=C['bad'], linewidth=2, linestyle='--',
               label='Chance level (50%)')
    ax.axhline(0.54, color=C['accent'], linewidth=2, linestyle='-.',
               label='Significance threshold (54%)')

    ax.fill_between(range(1, n_reps+1), 0.5, rep_means, alpha=0.1, color=C['good'])

    ax.set_xlabel('Repetition number (1 to 100)', fontsize=12)
    ax.set_ylabel('Mean 5-fold CV accuracy', fontsize=12)
    ax.set_title('C.  Repeat the entire 5-fold process 100 times with different random splits',
                 fontsize=14, fontweight='bold', color=C['dark'], loc='left')
    ax.set_xlim(0, 101)
    ax.set_ylim(0.42, 0.78)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(50, 0.44,
            'Each dot = the average accuracy across 5 folds for one random partition.\n'
            'The GRAND MEAN across all 100 repetitions is the final decoding accuracy\n'
            'reported for this subject in this ROI.',
            fontsize=10, ha='center', color='#555', fontstyle='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    savefig(fig, '10_cross_validation.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 11: Univariate vs Multivariate
# ════════════════════════════════════════════════════════════════
def fig11():
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('11. The Key Insight — Why Multivariate Analysis Sees What Univariate Misses',
                 fontsize=20, fontweight='bold', y=0.98, color=C['dark'])
    gs = gridspec.GridSpec(3, 2, hspace=0.4, wspace=0.25, top=0.92, bottom=0.04,
                           left=0.06, right=0.96, height_ratios=[1, 1, 0.8])

    np.random.seed(42)
    n_vox = 8
    vox_labels = [f'V{i+1}' for i in range(n_vox)]

    # Key: patterns are OPPOSITE but averages are the SAME
    pl_pattern = np.array([2.1, 0.4, 1.8, 0.3, 2.0, 0.5, 1.7, 0.6])
    nt_pattern = np.array([0.5, 1.9, 0.4, 2.0, 0.6, 1.8, 0.5, 1.9])

    # ── Panel A: Univariate view (per voxel) ──
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(n_vox)
    w = 0.35

    ax.bar(x - w/2, pl_pattern, w, color=C['pl'], alpha=0.8, label='Pleasant (avg β)')
    ax.bar(x + w/2, nt_pattern, w, color=C['nt'], alpha=0.8, label='Neutral (avg β)')

    avg_pl = np.mean(pl_pattern)
    avg_nt = np.mean(nt_pattern)
    ax.axhline(avg_pl, color=C['pl'], linewidth=2, linestyle='--', alpha=0.5)
    ax.axhline(avg_nt, color=C['nt'], linewidth=2, linestyle='--', alpha=0.5)

    ax.text(7.2, avg_pl + 0.05, f'Mean = {avg_pl:.2f}', fontsize=10, color=C['pl'])
    ax.text(7.2, avg_nt - 0.15, f'Mean = {avg_nt:.2f}', fontsize=10, color=C['nt'])

    ax.set_xticks(x)
    ax.set_xticklabels(vox_labels, fontsize=10)
    ax.set_ylabel('Beta value', fontsize=12)
    ax.set_title('A.  UNIVARIATE: look at each voxel independently',
                 fontsize=13, fontweight='bold', color=C['dark'], loc='left')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 2.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # ── Panel B: Univariate verdict ──
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "UNIVARIATE VERDICT\n"
            "━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Average Pleasant β = {avg_pl:.2f}\n"
            f"Average Neutral  β = {avg_nt:.2f}\n"
            f"Difference         = {avg_pl - avg_nt:.2f}\n\n"
            "The overall activation is almost\n"
            "IDENTICAL for both conditions.\n\n"
            "Statistical test (t-test):\n"
            "  p = 0.72 → NOT significant\n\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "CONCLUSION: \"V1 does not\n"
            "distinguish emotion from neutral.\"\n\n"
            "This is what prior fMRI studies\n"
            "concluded — and they were WRONG!",
            transform=ax.transAxes, fontsize=12, ha='center', va='center',
            color=C['dark'], fontfamily='monospace', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FADBD8',
                      edgecolor=C['bad'], linewidth=2))

    # ── Panel C: Multivariate view (pattern) ──
    ax = fig.add_subplot(gs[1, 0])

    ax.plot(x, pl_pattern, 'o-', color=C['pl'], linewidth=3, markersize=12,
            label='Pleasant PATTERN', zorder=3)
    ax.plot(x, nt_pattern, 's-', color=C['nt'], linewidth=3, markersize=12,
            label='Neutral PATTERN', zorder=3)

    # Highlight the "flip" at each voxel
    for i in range(n_vox):
        ax.annotate('', xy=(i, pl_pattern[i]), xytext=(i, nt_pattern[i]),
                    arrowprops=dict(arrowstyle='<->', color='#8E44AD', lw=1.5, alpha=0.5))

    ax.fill_between(x, pl_pattern, nt_pattern, alpha=0.12, color='#8E44AD')

    ax.set_xticks(x)
    ax.set_xticklabels(vox_labels, fontsize=10)
    ax.set_ylabel('Beta value', fontsize=12)
    ax.set_xlabel('Voxels in ROI (e.g., V1)', fontsize=12)
    ax.set_title('C.  MULTIVARIATE: look at the PATTERN across all voxels together',
                 fontsize=13, fontweight='bold', color=C['dark'], loc='left')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim(0, 2.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.text(3.5, 2.5, 'The patterns are\nCOMPLETELY OPPOSITE\n— like a mirror image!',
            fontsize=12, ha='center', color='#8E44AD', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#F4ECF7', edgecolor='#8E44AD'))

    # ── Panel D: Multivariate verdict ──
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "MULTIVARIATE (MVPA) VERDICT\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "The SVM sees these as two very\n"
            "different patterns — even though\n"
            "the averages are the same!\n\n"
            "Voxels where Pleasant > Neutral get\n"
            "positive SVM weights.\n"
            "Voxels where Neutral > Pleasant get\n"
            "negative SVM weights.\n\n"
            "Decoding accuracy: 72%\n"
            "Threshold (p < 0.001): 54%\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "CONCLUSION: \"V1 DOES contain\n"
            "emotion-specific information —\n"
            "it's encoded in the PATTERN,\n"
            "not in the average activation.\"\n\n"
            "This is the paper's key finding!",
            transform=ax.transAxes, fontsize=12, ha='center', va='center',
            color=C['dark'], fontfamily='monospace', linespacing=1.4,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#D5F5E3',
                      edgecolor=C['good'], linewidth=2))

    # ── Bottom: Paper results ──
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    ax.text(0.5, 0.5,
            "RESULTS FROM BO ET AL. (2021)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            "                         Univariate (standard fMRI)    MVPA (our approach)\n"
            "                         ─────────────────────────    ────────────────────\n"
            "  Pleasant vs Neutral:    6 / 17 ROIs significant     17 / 17 ROIs significant\n"
            "  Unpleasant vs Neutral: 11 / 17 ROIs significant     17 / 17 ROIs significant\n\n"
            "MVPA detected emotion signals in ALL visual areas including primary visual cortex (V1)\n"
            "— regions that decades of univariate studies had dismissed as uninvolved in emotion processing.",
            transform=ax.transAxes, fontsize=13, ha='center', va='center',
            color=C['dark'], fontfamily='monospace', linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#EBF5FB',
                      edgecolor='#2980B9', linewidth=2))

    savefig(fig, '11_univariate_vs_multivariate.png')


# ════════════════════════════════════════════════════════════════
# FIGURE 12: Full Pipeline Summary
# ════════════════════════════════════════════════════════════════
def fig12():
    fig = plt.figure(figsize=(20, 24))
    fig.suptitle('12. Complete Pipeline Summary — From Scanner to Scientific Conclusion',
                 fontsize=22, fontweight='bold', y=0.99, color=C['dark'])
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 28)

    steps = [
        {
            'y': 26.5, 'title': 'STEP 0: Data Collection',
            'color': '#3498DB', 'bg': '#D6EAF8',
            'text': ('20 subjects view 60 IAPS pictures (20 pleasant, 20 neutral, 20 unpleasant)\n'
                     '5 runs per subject × 206 fMRI volumes per run\n'
                     'Scanner: 3T, TR = 1.98s, voxel = 3.5mm³')
        },
        {
            'y': 23.5, 'title': 'STEP 1: Preprocessing (already done → swars*.img)',
            'color': '#9B59B6', 'bg': '#F4ECF7',
            'text': ('Slice timing → Realignment (→ rp_*.txt motion files) → Normalization to MNI → Smoothing\n'
                     'Removes the worst artifacts. Output: swars*.img files ready for modeling.')
        },
        {
            'y': 20.5, 'title': 'STEP 2: Build Design Matrix + Estimate Betas (BetaS2.m)',
            'color': '#E74C3C', 'bg': '#FADBD8',
            'text': ('For each subject: build GLM with 60 trial regressors + 6 motion regressors per run\n'
                     'Each trial regressor = onset time convolved with HRF (hemodynamic response function)\n'
                     'Solve y = Xβ + ε at every voxel → 300 beta images per subject (one per trial)\n'
                     'Motion regressors absorb head-movement artifacts; high-pass filter removes slow drift')
        },
        {
            'y': 17.5, 'title': 'STEP 3: Extract & Organize Betas (extract_betas.m)',
            'color': '#E67E22', 'bg': '#FDF2E9',
            'text': ('Read 300 beta images per subject, sort by condition:\n'
                     '  Pl.mat = [nVoxels × 100 pleasant trials]    (20 per run × 5 runs)\n'
                     '  Nt.mat = [nVoxels × 100 neutral trials]\n'
                     '  Up.mat = [nVoxels × 100 unpleasant trials]')
        },
        {
            'y': 14.5, 'title': 'STEP 4: Build ROI Masks (make_roi_masks_mat.m)',
            'color': '#1ABC9C', 'bg': '#D1F2EB',
            'text': ('Convert 17 ROI NIfTI masks (from Wang et al. 2015 retinotopic atlas) to MATLAB struct\n'
                     'ROIs: V1v/d, V2v/d, V3v/d, hV4, VO1/2, PHC1/2, hMT, LO1/2, V3a/b, IPS\n'
                     'Each mask = binary: which voxels belong to this visual region?')
        },
        {
            'y': 11.5, 'title': 'STEP 5: SVM Decoding per ROI (SingleTrialDecodingv3.m)',
            'color': '#27AE60', 'bg': '#D5F5E3',
            'text': ('For each subject × each ROI:\n'
                     '  1. Select voxels inside ROI → feature vectors [trials × voxels_in_ROI]\n'
                     '  2. Train linear SVM: Pleasant (label +1) vs Neutral (label −1)\n'
                     '  3. 5-fold cross-validation × 100 random repetitions → mean accuracy\n'
                     '  4. Repeat for Unpleasant vs Neutral\n'
                     'Output: 20×18 accuracy matrix (subjects × ROIs) for each comparison')
        },
        {
            'y': 8.0, 'title': 'STEP 6: Group-Level Statistics (group_level_validation.py)',
            'color': '#2980B9', 'bg': '#D6EAF8',
            'text': ('For each ROI:\n'
                     '  1. Compute group mean accuracy across 20 subjects\n'
                     '  2. Permutation test: generate 100,000 chance-level group means\n'
                     '     (simulate random guessing for each subject, average, repeat)\n'
                     '  3. p-value = proportion of chance means ≥ observed mean\n'
                     '  4. Threshold at p < 0.001 (found to be ~54% accuracy)\n'
                     'Also: one-sample t-test vs 50% chance, Cohen\'s d effect size')
        },
        {
            'y': 4.5, 'title': 'CONCLUSION',
            'color': C['dark'], 'bg': '#FCF3CF',
            'text': ('All 17 retinotopic visual ROIs — including primary visual cortex V1 —\n'
                     'show decoding accuracy significantly above chance (p < 0.001).\n\n'
                     'Emotional scenes evoke VALENCE-SPECIFIC neural patterns in visual cortex.\n'
                     'These patterns are influenced by reentrant feedback from amygdala and frontal cortex.')
        },
    ]

    for step in steps:
        rect = FancyBboxPatch((0.8, step['y'] - 0.3), 18.4,
                               2.4 if step['title'] != 'CONCLUSION' else 2.0,
                               boxstyle='round,pad=0.3',
                               facecolor=step['bg'], edgecolor=step['color'],
                               linewidth=2.5)
        ax.add_patch(rect)
        ax.text(1.3, step['y'] + 1.7, step['title'],
                fontsize=14, fontweight='bold', color=step['color'], va='top')
        ax.text(1.3, step['y'] + 0.9, step['text'],
                fontsize=10.5, color=C['dark'], va='top', fontfamily='monospace',
                linespacing=1.5)

    # Arrows between steps
    for i in range(len(steps) - 1):
        y_from = steps[i]['y'] - 0.3
        y_to = steps[i+1]['y'] + 2.1
        mid_y = (y_from + y_to) / 2
        ax.annotate('', xy=(10, y_to + 0.2), xytext=(10, y_from - 0.1),
                    arrowprops=dict(arrowstyle='->', color=C['dark'], lw=2.5,
                                    connectionstyle='arc3,rad=0'))

    # Script labels on the right
    scripts = [None, None, 'BetaS2.m\n+ sbatch', 'extract_betas.m', 'make_roi_masks_mat.m',
               'SingleTrialDecoding\nv3.m + sbatch', 'group_level_\nvalidation.py\n+ sbatch', None]
    for step_info, script in zip(steps, scripts):
        if script:
            ax.text(19.8, step_info['y'] + 1.0, script, fontsize=9,
                    ha='right', va='center', color='#555', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='#BDC3C7', linewidth=1))

    savefig(fig, '12_full_pipeline_summary.png')


# ════════════════════════════════════════════════════════════════
# RUN ALL
# ════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating figures...\n")
    fig01()
    fig02()
    fig03()
    fig04()
    fig05()
    fig06()
    fig07()
    fig08()
    fig09()
    fig10()
    fig11()
    fig12()
    print(f"\nDone! All figures saved to {OUT}/")
