"""
plot_gvpr.py — Visualise group-level validation results from data/gvpr.mat

Produces three figures saved to data/:
  fig1_decoding_accuracy.png  — bar chart mirroring paper Fig 3B
  fig2_effect_sizes.png       — Cohen's d per ROI (n-independent effect size)
  fig3_summary.png            — whole-brain bars + PlNt vs UpNt scatter

Usage (from repo root):
    python pipeline/step5_group_validation/plot_gvpr.py
    python pipeline/step5_group_validation/plot_gvpr.py --mat data/gvpr.mat --out data/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import scipy.io as sio

# ── ROI ordering (matches paper Fig 3B, whole_brain handled separately) ───────
PAPER_ROIS = [
    'V1v', 'V1d',
    'V2v', 'V2d',
    'V3v', 'V3d',
    'hV4',
    'VO1', 'VO2',
    'PHC1', 'PHC2',
    'hMT',
    'LO1', 'LO2',
    'V3a', 'V3b',
    'IPS',
]

# ── Colours ───────────────────────────────────────────────────────────────────
COL_SIG   = '#E07070'   # coral — significant ROI
COL_NS    = '#AAAAAA'   # gray  — non-significant ROI
COL_PLNT  = '#4878CF'   # blue  — PlNt
COL_UPNT  = '#E07070'   # coral — UpNt
COL_CHANCE = 'black'
COL_THRESH = '#CC0000'


# ── Helpers ───────────────────────────────────────────────────────────────────

def load(mat_path):
    d = sio.loadmat(str(mat_path))
    roi_names = [r.flat[0] for r in d['roi_names'].flatten()]
    return d, roi_names


def idx(roi_names, names):
    """Return integer indices for a list of ROI names."""
    return [roi_names.index(n) for n in names]


def get(d, key, indices):
    """Extract a 1-D numpy array from a (1, N) mat variable at given indices."""
    return d[key].flatten()[indices]


# ── Figure 1: decoding accuracy (paper Fig 3B style) ─────────────────────────

def fig1_decoding_accuracy(d, roi_names, out_dir):
    ri = idx(roi_names, PAPER_ROIS)
    x  = np.arange(len(PAPER_ROIS))

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    panels = [
        ('PlNt', 'Pleasant vs Neutral', axes[0]),
        ('UpNt', 'Unpleasant vs Neutral', axes[1]),
    ]

    for prefix, title, ax in panels:
        means  = get(d, f'{prefix}_mean_accuracy',   ri)
        sems   = get(d, f'{prefix}_sem_accuracy',    ri)
        sig    = get(d, f'{prefix}_perm_significant', ri).astype(bool)
        thresh = get(d, f'{prefix}_threshold',        ri)

        colors = [COL_SIG if s else COL_NS for s in sig]

        bars = ax.bar(x, means, yerr=sems, capsize=4, width=0.65,
                      color=colors, edgecolor='black', linewidth=0.8,
                      error_kw=dict(elinewidth=1.2, ecolor='black'))

        # chance line
        ax.axhline(0.5, color=COL_CHANCE, linestyle='--', linewidth=1.2,
                   label='Chance (50%)', zorder=2)

        # per-ROI threshold markers (mean threshold as single dashed line is cleaner)
        mean_thresh = float(np.mean(thresh))
        ax.axhline(mean_thresh, color=COL_THRESH, linestyle='--', linewidth=1.2,
                   label=f'Threshold p<0.001 ({mean_thresh:.3f})', zorder=2)

        # *** markers
        y_top = means + sems + 0.012
        for i, s in enumerate(sig):
            if s:
                ax.text(i, y_top[i], '***', ha='center', va='bottom',
                        fontsize=9, fontweight='bold', color='black')

        ax.set_ylabel('Decoding accuracy', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=4)
        ax.set_ylim(0.45, 0.76)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
        ax.grid(axis='y', alpha=0.3, linewidth=0.6)
        ax.legend(fontsize=9, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(PAPER_ROIS, rotation=45, ha='right', fontsize=10)
    axes[1].set_xlabel('ROI', fontsize=11)

    # shared legend patches
    sig_patch = mpatches.Patch(color=COL_SIG,  label='Significant (p < 0.001)')
    ns_patch  = mpatches.Patch(color=COL_NS,   label='Non-significant')
    fig.legend(handles=[sig_patch, ns_patch], loc='upper right',
               bbox_to_anchor=(0.99, 0.99), fontsize=9, framealpha=0.9)

    fig.suptitle(f'Group-level MVPA decoding accuracy (n={int(d["PlNt_n_subjects"].flatten()[1])})',
                 fontsize=13, fontweight='bold', y=1.01)

    path = out_dir / 'fig1_decoding_accuracy.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── Figure 2: Cohen's d effect sizes ─────────────────────────────────────────

def fig2_effect_sizes(d, roi_names, out_dir):
    ri = idx(roi_names, PAPER_ROIS)
    y  = np.arange(len(PAPER_ROIS))

    pn_d = get(d, 'PlNt_cohen_d', ri)
    un_d = get(d, 'UpNt_cohen_d', ri)

    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(9, 8))

    ax.barh(y + bar_h / 2, pn_d, height=bar_h, color=COL_PLNT,
            alpha=0.85, edgecolor='black', linewidth=0.6, label="Pleasant vs Neutral")
    ax.barh(y - bar_h / 2, un_d, height=bar_h, color=COL_UPNT,
            alpha=0.85, edgecolor='black', linewidth=0.6, label="Unpleasant vs Neutral")

    # reference lines
    for xv, ls, lbl in [(0.2, ':', 'd = 0.2 (small)'),
                         (0.5, '--', 'd = 0.5 (medium)'),
                         (0.8, '-',  'd = 0.8 (large)')]:
        ax.axvline(xv, color='gray', linestyle=ls, linewidth=1.0, alpha=0.7, label=lbl)

    ax.set_yticks(y)
    ax.set_yticklabels(PAPER_ROIS, fontsize=10)
    ax.set_xlabel("Cohen's d", fontsize=11)
    ax.set_title("Effect sizes per ROI (n-independent comparison with paper)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.3, linewidth=0.6)

    path = out_dir / 'fig2_effect_sizes.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── Figure 3: whole-brain + PlNt vs UpNt scatter ─────────────────────────────

def fig3_summary(d, roi_names, out_dir):
    ri    = idx(roi_names, PAPER_ROIS)
    wb    = roi_names.index('whole_brain')
    x_wb  = [0, 1]
    n_sub = int(d['PlNt_n_subjects'].flatten()[wb])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(wspace=0.35)

    # ── Left: whole-brain bars ────────────────────────────────────────────────
    pn_wb_mean  = float(d['PlNt_mean_accuracy'].flatten()[wb])
    pn_wb_sem   = float(d['PlNt_sem_accuracy'].flatten()[wb])
    un_wb_mean  = float(d['UpNt_mean_accuracy'].flatten()[wb])
    un_wb_sem   = float(d['UpNt_sem_accuracy'].flatten()[wb])
    pn_wb_thresh = float(d['PlNt_threshold'].flatten()[wb])
    un_wb_thresh = float(d['UpNt_threshold'].flatten()[wb])

    wb_means  = [pn_wb_mean,  un_wb_mean]
    wb_sems   = [pn_wb_sem,   un_wb_sem]
    wb_thresh = [pn_wb_thresh, un_wb_thresh]
    wb_colors = [COL_PLNT, COL_UPNT]
    wb_labels = ['PlNt', 'UpNt']

    bars = ax1.bar(x_wb, wb_means, yerr=wb_sems, capsize=6, width=0.5,
                   color=wb_colors, edgecolor='black', linewidth=1.0, alpha=0.85,
                   error_kw=dict(elinewidth=1.5, ecolor='black'))

    for xi, (thr, mean, sem) in enumerate(zip(wb_thresh, wb_means, wb_sems)):
        ax1.hlines(thr, xi - 0.3, xi + 0.3, colors=COL_THRESH,
                   linewidth=1.5, linestyles='--')
        ax1.text(xi, mean + sem + 0.012, '***', ha='center', va='bottom',
                 fontsize=11, fontweight='bold')

    ax1.axhline(0.5, color=COL_CHANCE, linestyle='--', linewidth=1.2, label='Chance (50%)')
    ax1.set_xticks(x_wb)
    ax1.set_xticklabels(['Pleasant\nvs Neutral', 'Unpleasant\nvs Neutral'], fontsize=10)
    ax1.set_ylabel('Decoding accuracy', fontsize=11)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylim(0.45, 0.80)
    ax1.set_title(f'Whole-brain (n={n_sub})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.spines[['top', 'right']].set_visible(False)
    ax1.grid(axis='y', alpha=0.3, linewidth=0.6)

    # ── Right: PlNt vs UpNt scatter per ROI ──────────────────────────────────
    pn_means = get(d, 'PlNt_mean_accuracy', ri)
    un_means = get(d, 'UpNt_mean_accuracy', ri)
    pn_sig   = get(d, 'PlNt_perm_significant', ri).astype(bool)
    un_sig   = get(d, 'UpNt_perm_significant', ri).astype(bool)
    both_sig = pn_sig & un_sig

    scatter_colors = [COL_SIG if s else COL_NS for s in both_sig]
    ax2.scatter(pn_means, un_means, c=scatter_colors, s=70,
                edgecolors='black', linewidths=0.6, zorder=3)

    # ROI labels (offset slightly to avoid overlap)
    for i, name in enumerate(PAPER_ROIS):
        ax2.annotate(name, (pn_means[i], un_means[i]),
                     textcoords='offset points', xytext=(5, 3), fontsize=7.5)

    # UpNt = PlNt diagonal
    lo = min(pn_means.min(), un_means.min()) - 0.01
    hi = max(pn_means.max(), un_means.max()) + 0.01
    ax2.plot([lo, hi], [lo, hi], color='gray', linestyle='--',
             linewidth=1.0, label='UpNt = PlNt', zorder=1)

    ax2.set_xlabel('PlNt mean accuracy', fontsize=11)
    ax2.set_ylabel('UpNt mean accuracy', fontsize=11)
    ax2.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1, decimals=0))
    ax2.set_title('UpNt vs PlNt accuracy per ROI', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.spines[['top', 'right']].set_visible(False)
    ax2.grid(alpha=0.3, linewidth=0.6)

    path = out_dir / 'fig3_summary.png'
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Plot group-level validation results')
    parser.add_argument('--mat', default='data/gvpr.mat',
                        help='Path to gvpr.mat (default: data/gvpr.mat)')
    parser.add_argument('--out', default='data/',
                        help='Output directory (default: data/)')
    args = parser.parse_args()

    mat_path = Path(args.mat)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d, roi_names = load(mat_path)
    print(f'Loaded {mat_path} | ROIs: {roi_names}')

    fig1_decoding_accuracy(d, roi_names, out_dir)
    fig2_effect_sizes(d, roi_names, out_dir)
    fig3_summary(d, roi_names, out_dir)

    print('Done.')


if __name__ == '__main__':
    main()
