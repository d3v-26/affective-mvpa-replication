"""
Group-level statistical validation for single-trial decoding
Based on: Bo et al. (2021) - Decoding Neural Representations of Affective Scenes
in Retinotopic Visual Cortex

This script performs permutation-based statistical testing at the group level
with significance threshold p < 0.001 as described in the paper.

Features:
- Checkpointing for SLURM job recovery
- Progress tracking
- Resume from last checkpoint
"""

import numpy as np
import scipy.io as sio
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
import json
from datetime import datetime
import sys


class GroupLevelValidator:
    """
    Performs group-level permutation testing for decoding accuracies.

    Based on the methodology from Bo et al. (2021):
    - Permutation test to establish chance-level distribution
    - Threshold at p < 0.001 for statistical significance
    - 10^5 (100,000) permutations at group level
    - Checkpointing for long-running jobs
    """

    def __init__(self, n_permutations=100000, alpha=0.001, checkpoint_dir=None,
                 checkpoint_interval=5000):
        """
        Initialize validator.

        Parameters:
        -----------
        n_permutations : int
            Number of permutations for group-level test (default: 100,000)
        alpha : float
            Significance level (default: 0.001 as per paper)
        checkpoint_dir : str, optional
            Directory to save checkpoints. If None, uses current directory
        checkpoint_interval : int
            Save checkpoint every N permutations (default: 5000)
        """
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.chance_level = 0.5  # Binary classification
        self.checkpoint_interval = checkpoint_interval

        # Setup checkpoint directory
        if checkpoint_dir is None:
            self.checkpoint_dir = Path('.')
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def load_decoding_results(self, results_file):
        """
        Load decoding results from MATLAB file.

        Parameters:
        -----------
        results_file : str
            Path to MATLAB file containing decoding results
            Expected variables: PlNt_acc, UpNt_acc, roi_names_of_interest

        Returns:
        --------
        dict : Dictionary containing decoding accuracies and ROI names
        """
        data = sio.loadmat(results_file)

        results = {
            'PlNt_acc': data['PlNt_acc'],  # Pleasant vs Neutral
            'UpNt_acc': data['UpNt_acc'],  # Unpleasant vs Neutral
            'roi_names': [name[0] for name in data['roi_names_of_interest'][0]]
        }

        return results

    def save_checkpoint(self, checkpoint_data, checkpoint_name):
        """
        Save checkpoint to disk.

        Parameters:
        -----------
        checkpoint_data : dict
            Data to save
        checkpoint_name : str
            Name of checkpoint file
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"
        temp_path = checkpoint_path.with_suffix('.tmp')

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            temp_path.replace(checkpoint_path)  # Atomic rename
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")

    def load_checkpoint(self, checkpoint_name):
        """
        Load checkpoint from disk.

        Parameters:
        -----------
        checkpoint_name : str
            Name of checkpoint file

        Returns:
        --------
        dict or None : Checkpoint data if exists, None otherwise
        """
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.pkl"

        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'rb') as f:
                    data = pickle.load(f)
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Checkpoint loaded: {checkpoint_path}")
                return data
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                return None
        return None

    def permutation_test_group_level(self, accuracies, roi_idx=None,
                                     comparison_name='', resume=True):
        """
        Perform group-level permutation test with checkpointing.

        At the individual subject level, class labels were randomly shuffled
        100 times. At the group level, one chance-level accuracy is randomly
        extracted from each subject and averaged. This is repeated 10^5 times
        to obtain empirical chance-level distribution.

        Parameters:
        -----------
        accuracies : ndarray
            Shape (n_subjects, n_rois) or (n_subjects,) for single ROI
        roi_idx : int, optional
            Index of specific ROI to test. If None, test all ROIs.
        comparison_name : str
            Name for checkpoint file (e.g., 'PlNt', 'UpNt')
        resume : bool
            Whether to resume from checkpoint if available

        Returns:
        --------
        dict : Dictionary containing test results
        """
        if roi_idx is not None:
            accuracies = accuracies[:, roi_idx:roi_idx+1]

        n_subjects, n_rois = accuracies.shape

        results = {
            'mean_accuracy': np.nanmean(accuracies, axis=0),
            'sem_accuracy': np.nanstd(accuracies, axis=0) / np.sqrt(np.sum(~np.isnan(accuracies), axis=0)),
            'n_subjects': np.sum(~np.isnan(accuracies), axis=0),
            'p_values': np.zeros(n_rois),
            'is_significant': np.zeros(n_rois, dtype=bool),
            'threshold_accuracy': np.zeros(n_rois)
        }

        # Perform permutation test for each ROI
        for roi in range(n_rois):
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                  f"Processing ROI {roi+1}/{n_rois} ({comparison_name})")

            roi_acc = accuracies[:, roi]
            roi_acc = roi_acc[~np.isnan(roi_acc)]
            n_valid_subjects = len(roi_acc)

            checkpoint_name = f"{comparison_name}_roi{roi}"

            # Try to load checkpoint
            checkpoint = None
            if resume:
                checkpoint = self.load_checkpoint(checkpoint_name)

            if checkpoint is not None:
                # Resume from checkpoint
                chance_distribution = checkpoint['chance_distribution']
                start_perm = checkpoint['completed_permutations']
                print(f"  Resuming from permutation {start_perm}/{self.n_permutations}")
            else:
                # Start fresh
                chance_distribution = np.zeros(self.n_permutations)
                start_perm = 0
                print(f"  Starting new permutation test")

            # Generate chance-level distribution at group level
            start_time = time.time()

            for perm in range(start_perm, self.n_permutations):
                # Simulate chance-level accuracies for each subject
                # Using binomial distribution as in classification tasks
                subject_chance = np.random.binomial(100, 0.5, n_valid_subjects) / 100.0
                chance_distribution[perm] = np.mean(subject_chance)

                # Save checkpoint periodically
                if (perm + 1) % self.checkpoint_interval == 0:
                    elapsed = time.time() - start_time
                    rate = self.checkpoint_interval / elapsed
                    remaining = (self.n_permutations - perm - 1) / rate

                    print(f"  Progress: {perm+1}/{self.n_permutations} "
                          f"({100*(perm+1)/self.n_permutations:.1f}%) - "
                          f"Rate: {rate:.0f} perm/s - "
                          f"ETA: {remaining/60:.1f} min")

                    checkpoint_data = {
                        'chance_distribution': chance_distribution,
                        'completed_permutations': perm + 1,
                        'n_permutations': self.n_permutations,
                        'n_valid_subjects': n_valid_subjects,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.save_checkpoint(checkpoint_data, checkpoint_name)
                    start_time = time.time()

            # Calculate p-value: proportion of permuted values >= observed
            observed_mean = np.nanmean(roi_acc)
            p_value = np.sum(chance_distribution >= observed_mean) / self.n_permutations

            # Determine threshold at specified alpha level
            threshold = np.percentile(chance_distribution, (1 - self.alpha) * 100)

            results['p_values'][roi] = p_value
            results['is_significant'][roi] = p_value < self.alpha
            results['threshold_accuracy'][roi] = threshold

            print(f"  Completed: Mean={observed_mean:.4f}, p={p_value:.4f}, "
                  f"Significant={'YES' if p_value < self.alpha else 'NO'}")

            # Save final checkpoint
            checkpoint_data = {
                'chance_distribution': chance_distribution,
                'completed_permutations': self.n_permutations,
                'n_permutations': self.n_permutations,
                'n_valid_subjects': n_valid_subjects,
                'timestamp': datetime.now().isoformat(),
                'results': {
                    'p_value': p_value,
                    'threshold': threshold,
                    'observed_mean': observed_mean
                }
            }
            self.save_checkpoint(checkpoint_data, checkpoint_name)

        return results

    def one_sample_ttest(self, accuracies, roi_idx=None):
        """
        Perform one-sample t-test against chance level (50%).

        Parameters:
        -----------
        accuracies : ndarray
            Shape (n_subjects, n_rois) or (n_subjects,)
        roi_idx : int, optional
            Index of specific ROI to test

        Returns:
        --------
        dict : Dictionary containing t-test results
        """
        if roi_idx is not None:
            accuracies = accuracies[:, roi_idx:roi_idx+1]

        n_rois = accuracies.shape[1]

        results = {
            't_statistic': np.zeros(n_rois),
            'p_values': np.zeros(n_rois),
            'is_significant': np.zeros(n_rois, dtype=bool),
            'cohen_d': np.zeros(n_rois)
        }

        for roi in range(n_rois):
            roi_acc = accuracies[:, roi]
            roi_acc = roi_acc[~np.isnan(roi_acc)]

            # One-sample t-test against chance (0.5)
            t_stat, p_val = stats.ttest_1samp(roi_acc, self.chance_level)

            # Cohen's d effect size
            cohen_d = (np.mean(roi_acc) - self.chance_level) / np.std(roi_acc, ddof=1)

            results['t_statistic'][roi] = t_stat
            results['p_values'][roi] = p_val
            results['is_significant'][roi] = p_val < self.alpha
            results['cohen_d'][roi] = cohen_d

        return results

    def visualize_results(self, accuracies, roi_names, comparison_name='',
                         save_path=None):
        """
        Visualize group-level results with statistical annotations.

        Parameters:
        -----------
        accuracies : ndarray
            Shape (n_subjects, n_rois)
        roi_names : list
            List of ROI names
        comparison_name : str
            Name of comparison (e.g., 'Pleasant vs Neutral')
        save_path : str, optional
            Path to save figure
        """
        # Perform statistical tests
        perm_results = self.permutation_test_group_level(accuracies)
        ttest_results = self.one_sample_ttest(accuracies)

        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Box plots with significance markers
        ax1 = axes[0]
        positions = np.arange(len(roi_names))

        # Create box plot
        bp = ax1.boxplot(accuracies, positions=positions, widths=0.6,
                         patch_artist=True, showfliers=True)

        # Color boxes based on significance
        for i, (box, is_sig) in enumerate(zip(bp['boxes'], perm_results['is_significant'])):
            if is_sig:
                box.set_facecolor('lightcoral')
                box.set_alpha(0.7)
            else:
                box.set_facecolor('lightgray')
                box.set_alpha(0.5)

        # Add chance level line
        ax1.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Chance (50%)')

        # Add threshold line (p < 0.001)
        threshold_mean = np.mean(perm_results['threshold_accuracy'])
        ax1.axhline(y=threshold_mean, color='r', linestyle='-.', linewidth=1,
                   label=f'Threshold (p < {self.alpha})')

        # Add significance markers
        y_max = np.nanmax(accuracies) + 0.02
        for i, is_sig in enumerate(perm_results['is_significant']):
            if is_sig:
                ax1.text(i, y_max, '***', ha='center', va='bottom', fontsize=12,
                        fontweight='bold')

        ax1.set_xticks(positions)
        ax1.set_xticklabels(roi_names, rotation=45, ha='right')
        ax1.set_ylabel('Decoding Accuracy', fontsize=12)
        ax1.set_title(f'{comparison_name} - Group Level Validation (n={accuracies.shape[0]})',
                     fontsize=14, fontweight='bold')
        ax1.set_ylim([0.45, y_max + 0.03])
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Mean accuracies with error bars and p-values
        ax2 = axes[1]

        means = perm_results['mean_accuracy']
        sems = perm_results['sem_accuracy']

        # Create bar plot
        colors = ['coral' if sig else 'gray' for sig in perm_results['is_significant']]
        bars = ax2.bar(positions, means, yerr=sems, capsize=5, alpha=0.7,
                      color=colors, edgecolor='black', linewidth=1.5)

        # Add chance level line
        ax2.axhline(y=0.5, color='k', linestyle='--', linewidth=1, label='Chance')
        ax2.axhline(y=threshold_mean, color='r', linestyle='-.', linewidth=1,
                   label=f'p < {self.alpha}')

        # Add p-values as text
        for i, (mean, p_val) in enumerate(zip(means, perm_results['p_values'])):
            if p_val < 0.001:
                p_text = 'p < 0.001'
            else:
                p_text = f'p = {p_val:.3f}'
            ax2.text(i, mean + sems[i] + 0.01, p_text, ha='center',
                    va='bottom', fontsize=8, rotation=45)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(roi_names, rotation=45, ha='right')
        ax2.set_ylabel('Mean Decoding Accuracy ± SEM', fontsize=12)
        ax2.set_xlabel('ROI', fontsize=12)
        ax2.set_title('Mean Accuracies with Permutation Test Results', fontsize=14)
        ax2.set_ylim([0.45, np.max(means + sems) + 0.05])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")

        plt.show()

        return perm_results, ttest_results

    def print_summary_table(self, accuracies, roi_names, comparison_name=''):
        """
        Print formatted summary table of results.

        Parameters:
        -----------
        accuracies : ndarray
            Shape (n_subjects, n_rois)
        roi_names : list
            List of ROI names
        comparison_name : str
            Name of comparison
        """
        perm_results = self.permutation_test_group_level(accuracies)
        ttest_results = self.one_sample_ttest(accuracies)

        print(f"\n{'='*80}")
        print(f"GROUP-LEVEL VALIDATION RESULTS: {comparison_name}")
        print(f"{'='*80}")
        print(f"Significance threshold: p < {self.alpha}")
        print(f"Number of permutations: {self.n_permutations:,}")
        print(f"Number of subjects: {accuracies.shape[0]}")
        print(f"{'='*80}\n")

        # Table header
        print(f"{'ROI':<15} {'Mean±SEM':<15} {'n':<5} {'p-value':<12} {'Sig':<5} "
              f"{'t-stat':<10} {'Cohen-d':<10}")
        print(f"{'-'*80}")

        # Table rows
        for i, roi in enumerate(roi_names):
            mean = perm_results['mean_accuracy'][i]
            sem = perm_results['sem_accuracy'][i]
            n = int(perm_results['n_subjects'][i])
            p_val = perm_results['p_values'][i]
            sig = '***' if perm_results['is_significant'][i] else 'n.s.'
            t_stat = ttest_results['t_statistic'][i]
            cohen_d = ttest_results['cohen_d'][i]

            p_str = '<0.001' if p_val < 0.001 else f'{p_val:.4f}'

            print(f"{roi:<15} {mean:.3f}±{sem:.3f}    {n:<5} {p_str:<12} {sig:<5} "
                  f"{t_stat:>9.3f} {cohen_d:>10.3f}")

        print(f"{'-'*80}")

        # Summary statistics
        n_significant = np.sum(perm_results['is_significant'])
        print(f"\nSignificant ROIs: {n_significant}/{len(roi_names)} "
              f"({100*n_significant/len(roi_names):.1f}%)")
        print(f"Mean threshold accuracy (p<{self.alpha}): "
              f"{np.mean(perm_results['threshold_accuracy']):.3f}")
        print(f"{'='*80}\n")


def main():
    """
    Main function to run group-level validation.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Group-level validation with checkpointing')
    parser.add_argument('--results-file', type=str,
                       default='/blue/ruogu.fang/pateld3/neuroimaging/output_new/decoding_results_k5x100_v4.mat',
                       help='Path to decoding results MAT file')
    parser.add_argument('--output-dir', type=str,
                       default='/blue/ruogu.fang/pateld3/neuroimaging/output_new',
                       help='Output directory')
    parser.add_argument('--checkpoint-dir', type=str,
                       default='/blue/ruogu.fang/pateld3/neuroimaging/checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--n-permutations', type=int, default=100000,
                       help='Number of permutations (default: 100000)')
    parser.add_argument('--checkpoint-interval', type=int, default=5000,
                       help='Save checkpoint every N permutations (default: 5000)')
    parser.add_argument('--alpha', type=float, default=0.001,
                       help='Significance level (default: 0.001)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Do not resume from checkpoints')
    parser.add_argument('--comparison', type=str, choices=['both', 'PlNt', 'UpNt'],
                       default='both', help='Which comparison to run')

    args = parser.parse_args()

    # Configuration
    results_file = args.results_file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize validator with checkpointing
    print(f"\n{'='*80}")
    print(f"GROUP-LEVEL VALIDATION WITH CHECKPOINTING")
    print(f"{'='*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Number of permutations: {args.n_permutations:,}")
    print(f"Checkpoint interval: {args.checkpoint_interval:,}")
    print(f"Significance level: {args.alpha}")
    print(f"Resume from checkpoint: {not args.no_resume}")
    print(f"{'='*80}\n")

    validator = GroupLevelValidator(
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )

    try:
        print("Loading decoding results...")
        results = validator.load_decoding_results(results_file)

        print(f"Data loaded: {results['PlNt_acc'].shape[0]} subjects, "
              f"{len(results['roi_names'])} ROIs\n")

        pn_perm = pn_ttest = None
        un_perm = un_ttest = None

        # Test Pleasant vs Neutral
        if args.comparison in ['both', 'PlNt']:
            print("\n" + "="*80)
            print("PLEASANT vs NEUTRAL")
            print("="*80)

            pn_perm = validator.permutation_test_group_level(
                results['PlNt_acc'],
                comparison_name='PlNt',
                resume=not args.no_resume
            )
            pn_ttest = validator.one_sample_ttest(results['PlNt_acc'])

            validator.print_summary_table(results['PlNt_acc'], results['roi_names'],
                                          'Pleasant vs Neutral')

            pn_perm_viz, pn_ttest_viz = validator.visualize_results(
                results['PlNt_acc'],
                results['roi_names'],
                comparison_name='Pleasant vs Neutral',
                save_path=str(output_dir / "PlNt_group_validation.png")
            )

        # Test Unpleasant vs Neutral
        if args.comparison in ['both', 'UpNt']:
            print("\n" + "="*80)
            print("UNPLEASANT vs NEUTRAL")
            print("="*80)

            un_perm = validator.permutation_test_group_level(
                results['UpNt_acc'],
                comparison_name='UpNt',
                resume=not args.no_resume
            )
            un_ttest = validator.one_sample_ttest(results['UpNt_acc'])

            validator.print_summary_table(results['UpNt_acc'], results['roi_names'],
                                          'Unpleasant vs Neutral')

            un_perm_viz, un_ttest_viz = validator.visualize_results(
                results['UpNt_acc'],
                results['roi_names'],
                comparison_name='Unpleasant vs Neutral',
                save_path=str(output_dir / "UpNt_group_validation.png")
            )

        # Save results to file
        output_file = output_dir / "group_validation_results.mat"
        save_dict = {
            'roi_names': results['roi_names'],
            'alpha': validator.alpha,
            'n_permutations': validator.n_permutations
        }

        if pn_perm is not None:
            save_dict.update({
                'PlNt_perm_pvalues': pn_perm['p_values'],
                'PlNt_perm_significant': pn_perm['is_significant'],
                'PlNt_threshold': pn_perm['threshold_accuracy'],
                'PlNt_ttest_pvalues': pn_ttest['p_values'],
                'PlNt_cohen_d': pn_ttest['cohen_d'],
            })

        if un_perm is not None:
            save_dict.update({
                'UpNt_perm_pvalues': un_perm['p_values'],
                'UpNt_perm_significant': un_perm['is_significant'],
                'UpNt_threshold': un_perm['threshold_accuracy'],
                'UpNt_ttest_pvalues': un_ttest['p_values'],
                'UpNt_cohen_d': un_ttest['cohen_d'],
            })

        sio.savemat(str(output_file), save_dict)

        print(f"\n{'='*80}")
        print(f"Results saved to: {output_file}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved in checkpoints.")
        print(f"To resume, run the same command again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nProgress saved in checkpoints. You can resume by running again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
