"""
Stability Metrics for RL Algorithms

This module implements metrics to measure the stability and consistency
of RL algorithms during training, following best practices from the literature.

References:
- Henderson et al. (2018): "Deep RL that Matters"
- Cobbe et al. (2019): "Quantifying Generalization in RL"
- Agarwal et al. (2021): "Deep RL at the Edge of the Statistical Precipice"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
from scipy import stats, signal
import warnings


class StabilityMetrics:
    """
    Calculate stability metrics to measure consistency and robustness of RL algorithms.

    Stability is crucial for reliable deployment and reproducibility.
    """

    def __init__(self, reward_history: Union[np.ndarray, List[float]],
                 timesteps: Optional[np.ndarray] = None):
        """
        Initialize with reward history.

        Args:
            reward_history: Array of rewards over training (shape: [n_evaluations] or [n_seeds, n_evaluations])
            timesteps: Array of timesteps when evaluations occurred
        """
        self.rewards = np.array(reward_history)

        # Handle both single seed and multi-seed inputs
        if len(self.rewards.shape) == 1:
            self.rewards = self.rewards.reshape(1, -1)
            self.single_seed = True
        else:
            self.single_seed = False

        self.n_seeds = self.rewards.shape[0]
        self.n_evaluations = self.rewards.shape[1]

        if timesteps is None:
            self.timesteps = np.arange(self.n_evaluations)
        else:
            self.timesteps = np.array(timesteps)

    def running_variance(self, window: int = 10) -> np.ndarray:
        """
        Calculate running variance of rewards over time.

        Args:
            window: Window size for variance calculation

        Returns:
            Array of running variances
        """
        if self.single_seed:
            rewards = self.rewards[0]
            if len(rewards) < window:
                return np.array([np.var(rewards)])

            variances = []
            for i in range(len(rewards) - window + 1):
                variances.append(np.var(rewards[i:i+window]))

            return np.array(variances)
        else:
            # Calculate variance across seeds at each timestep
            return np.var(self.rewards, axis=0)

    def coefficient_of_variation_over_time(self, window: int = 10) -> np.ndarray:
        """
        Calculate coefficient of variation (CV) over time.

        CV = std/mean, measures relative variability.

        Args:
            window: Window size for CV calculation

        Returns:
            Array of CV values over time
        """
        if self.single_seed:
            rewards = self.rewards[0]
            if len(rewards) < window:
                mean = np.mean(rewards)
                std = np.std(rewards)
                return np.array([std / (abs(mean) + 1e-8)])

            cvs = []
            for i in range(len(rewards) - window + 1):
                window_rewards = rewards[i:i+window]
                mean = np.mean(window_rewards)
                std = np.std(window_rewards)
                cv = std / (abs(mean) + 1e-8)
                cvs.append(cv)

            return np.array(cvs)
        else:
            # CV across seeds at each timestep
            means = np.mean(self.rewards, axis=0)
            stds = np.std(self.rewards, axis=0)
            return stds / (np.abs(means) + 1e-8)

    def catastrophic_failures(self, threshold_percentile: float = 10) -> Dict:
        """
        Detect catastrophic failures (sudden performance drops).

        Args:
            threshold_percentile: Percentile below which is considered catastrophic

        Returns:
            Dictionary with failure statistics
        """
        results = {
            'n_failures': 0,
            'failure_timesteps': [],
            'failure_magnitudes': [],
            'recovery_times': []
        }

        for seed_idx in range(self.n_seeds):
            rewards = self.rewards[seed_idx]

            # Calculate threshold based on historical performance
            threshold = np.percentile(rewards, threshold_percentile)

            # Find drops below threshold
            failures = np.where(rewards < threshold)[0]

            if len(failures) > 0:
                # Group consecutive failures
                failure_groups = []
                current_group = [failures[0]]

                for i in range(1, len(failures)):
                    if failures[i] - failures[i-1] == 1:
                        current_group.append(failures[i])
                    else:
                        failure_groups.append(current_group)
                        current_group = [failures[i]]
                failure_groups.append(current_group)

                # Analyze each failure episode
                for group in failure_groups:
                    start_idx = group[0]
                    end_idx = group[-1]

                    # Skip if at the beginning (no baseline)
                    if start_idx == 0:
                        continue

                    # Calculate failure magnitude
                    pre_failure_reward = rewards[max(0, start_idx-5):start_idx].mean()
                    failure_reward = rewards[group].mean()
                    magnitude = pre_failure_reward - failure_reward

                    results['n_failures'] += 1
                    results['failure_timesteps'].append(int(start_idx))
                    results['failure_magnitudes'].append(float(magnitude))

                    # Calculate recovery time
                    recovery_idx = None
                    for i in range(end_idx + 1, len(rewards)):
                        if rewards[i] >= pre_failure_reward * 0.95:
                            recovery_idx = i
                            break

                    if recovery_idx is not None:
                        recovery_time = recovery_idx - end_idx
                        results['recovery_times'].append(int(recovery_time))
                    else:
                        results['recovery_times'].append(None)

        return results

    def monotonicity_score(self) -> float:
        """
        Calculate monotonicity score (how consistently performance improves).

        Score close to 1 means monotonic improvement.

        Returns:
            Monotonicity score between -1 and 1
        """
        all_scores = []

        for seed_idx in range(self.n_seeds):
            rewards = self.rewards[seed_idx]

            # Count positive vs negative changes
            diffs = np.diff(rewards)
            positive_changes = np.sum(diffs > 0)
            negative_changes = np.sum(diffs < 0)
            total_changes = len(diffs)

            if total_changes == 0:
                score = 0
            else:
                score = (positive_changes - negative_changes) / total_changes

            all_scores.append(score)

        return float(np.mean(all_scores))

    def smoothness_score(self, order: int = 2) -> float:
        """
        Calculate smoothness score using derivative analysis.

        Lower scores indicate smoother learning curves.

        Args:
            order: Order of derivative to analyze (1 or 2)

        Returns:
            Smoothness score (lower is smoother)
        """
        all_scores = []

        for seed_idx in range(self.n_seeds):
            rewards = self.rewards[seed_idx]

            # Calculate derivative
            if order == 1:
                derivative = np.diff(rewards)
            elif order == 2:
                derivative = np.diff(rewards, n=2)
            else:
                raise ValueError("Order must be 1 or 2")

            # Measure roughness as variance of derivative
            score = np.var(derivative)
            all_scores.append(score)

        return float(np.mean(all_scores))

    def stability_index(self) -> float:
        """
        Calculate overall stability index combining multiple factors.

        Higher scores indicate more stable training.

        Returns:
            Stability index (0-1, higher is better)
        """
        # Factor 1: Low variance
        variance_score = 1.0 / (1.0 + np.mean(self.running_variance()))

        # Factor 2: Monotonicity
        mono_score = (self.monotonicity_score() + 1) / 2  # Normalize to 0-1

        # Factor 3: Low catastrophic failures
        failures = self.catastrophic_failures()
        failure_score = 1.0 / (1.0 + failures['n_failures'])

        # Factor 4: Smoothness
        smooth_score = 1.0 / (1.0 + self.smoothness_score())

        # Combine factors (weighted average)
        weights = [0.25, 0.25, 0.25, 0.25]
        scores = [variance_score, mono_score, failure_score, smooth_score]
        index = np.dot(weights, scores)

        return float(index)

    def signal_to_noise_ratio(self) -> float:
        """
        Calculate signal-to-noise ratio of learning curve.

        Higher values indicate clearer learning signal.

        Returns:
            Signal-to-noise ratio
        """
        snr_values = []

        for seed_idx in range(self.n_seeds):
            rewards = self.rewards[seed_idx]

            # Fit polynomial trend (signal)
            if len(rewards) > 3:
                coeffs = np.polyfit(range(len(rewards)), rewards, deg=3)
                trend = np.polyval(coeffs, range(len(rewards)))

                # Calculate residuals (noise)
                residuals = rewards - trend

                # SNR = variance of signal / variance of noise
                signal_var = np.var(trend)
                noise_var = np.var(residuals)

                if noise_var > 0:
                    snr = signal_var / noise_var
                else:
                    snr = float('inf')

                snr_values.append(snr)

        return float(np.mean(snr_values)) if snr_values else 0.0

    def plateau_detection(self, window: int = 20, tolerance: float = 0.05) -> Dict:
        """
        Detect performance plateaus (periods of no improvement).

        Args:
            window: Window size to check for plateaus
            tolerance: Maximum relative change to consider as plateau

        Returns:
            Dictionary with plateau information
        """
        plateaus = {
            'n_plateaus': 0,
            'plateau_starts': [],
            'plateau_lengths': [],
            'total_plateau_time': 0
        }

        for seed_idx in range(self.n_seeds):
            rewards = self.rewards[seed_idx]

            if len(rewards) < window:
                continue

            in_plateau = False
            plateau_start = None

            for i in range(len(rewards) - window):
                window_rewards = rewards[i:i+window]
                relative_change = (np.max(window_rewards) - np.min(window_rewards)) / (np.mean(window_rewards) + 1e-8)

                if relative_change < tolerance:
                    if not in_plateau:
                        in_plateau = True
                        plateau_start = i
                else:
                    if in_plateau:
                        plateau_length = i - plateau_start
                        plateaus['n_plateaus'] += 1
                        plateaus['plateau_starts'].append(int(plateau_start))
                        plateaus['plateau_lengths'].append(int(plateau_length))
                        plateaus['total_plateau_time'] += plateau_length
                        in_plateau = False

            # Handle plateau that extends to the end
            if in_plateau:
                plateau_length = len(rewards) - plateau_start
                plateaus['n_plateaus'] += 1
                plateaus['plateau_starts'].append(int(plateau_start))
                plateaus['plateau_lengths'].append(int(plateau_length))
                plateaus['total_plateau_time'] += plateau_length

        return plateaus

    def cross_seed_correlation(self) -> Optional[float]:
        """
        Calculate correlation between different seed performances.

        High correlation suggests environment factors dominate over random initialization.

        Returns:
            Mean correlation coefficient, or None for single seed
        """
        if self.n_seeds < 2:
            return None

        correlations = []
        for i in range(self.n_seeds):
            for j in range(i+1, self.n_seeds):
                corr, _ = stats.pearsonr(self.rewards[i], self.rewards[j])
                if not np.isnan(corr):
                    correlations.append(corr)

        return float(np.mean(correlations)) if correlations else None

    def oscillation_metric(self) -> float:
        """
        Measure oscillation in performance (rapid up-down changes).

        Lower values indicate less oscillation.

        Returns:
            Oscillation metric
        """
        oscillations = []

        for seed_idx in range(self.n_seeds):
            rewards = self.rewards[seed_idx]

            # Count sign changes in derivative
            diffs = np.diff(rewards)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

            # Normalize by length
            oscillation = sign_changes / (len(rewards) - 2)
            oscillations.append(oscillation)

        return float(np.mean(oscillations))

    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of stability metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        failures = self.catastrophic_failures()
        plateaus = self.plateau_detection()

        summary = {
            'stability_index': self.stability_index(),
            'monotonicity_score': self.monotonicity_score(),
            'smoothness_score': self.smoothness_score(),
            'signal_to_noise_ratio': self.signal_to_noise_ratio(),
            'oscillation_metric': self.oscillation_metric(),
            'mean_variance': float(np.mean(self.running_variance())),
            'catastrophic_failures': {
                'count': failures['n_failures'],
                'mean_magnitude': float(np.mean(failures['failure_magnitudes'])) if failures['failure_magnitudes'] else 0,
                'mean_recovery_time': float(np.nanmean([r for r in failures['recovery_times'] if r is not None])) if any(r is not None for r in failures['recovery_times']) else None
            },
            'plateaus': {
                'count': plateaus['n_plateaus'],
                'total_time_ratio': plateaus['total_plateau_time'] / (self.n_seeds * self.n_evaluations),
                'mean_length': float(np.mean(plateaus['plateau_lengths'])) if plateaus['plateau_lengths'] else 0
            }
        }

        if not self.single_seed:
            summary['cross_seed_correlation'] = self.cross_seed_correlation()

        return summary

    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


def compare_stability(algorithms: Dict[str, np.ndarray]) -> Dict:
    """
    Compare stability across multiple algorithms.

    Args:
        algorithms: Dict mapping algorithm names to reward histories

    Returns:
        Comparison results dictionary
    """
    results = {}

    # Calculate metrics for each algorithm
    for name, rewards in algorithms.items():
        metric = StabilityMetrics(rewards)
        results[name] = metric.get_summary()

    # Create comparison
    comparison = {
        'individual_metrics': results,
        'rankings': {}
    }

    # Rank by stability index
    stability_scores = {name: metrics['stability_index']
                       for name, metrics in results.items()}
    sorted_by_stability = sorted(stability_scores.items(),
                                key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_stability'] = [name for name, _ in sorted_by_stability]

    # Rank by monotonicity
    mono_scores = {name: metrics['monotonicity_score']
                  for name, metrics in results.items()}
    sorted_by_mono = sorted(mono_scores.items(),
                           key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_monotonicity'] = [name for name, _ in sorted_by_mono]

    # Rank by failure rate (lower is better)
    failure_rates = {name: metrics['catastrophic_failures']['count']
                    for name, metrics in results.items()}
    sorted_by_failures = sorted(failure_rates.items(),
                               key=lambda x: x[1])
    comparison['rankings']['by_robustness'] = [name for name, _ in sorted_by_failures]

    return comparison