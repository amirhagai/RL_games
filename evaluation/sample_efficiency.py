"""
Sample Efficiency Metrics for RL Algorithms

This module implements various sample efficiency metrics to measure how quickly
an RL algorithm learns, based on the literature.

References:
- Henderson et al. (2018): "Deep Reinforcement Learning that Matters"
- Agarwal et al. (2021): "Deep RL at the Edge of the Statistical Precipice"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path


class SampleEfficiencyMetrics:
    """
    Calculate various sample efficiency metrics for RL algorithms.

    Sample efficiency measures how quickly an algorithm learns to achieve
    certain performance thresholds, critical for practical applications.
    """

    def __init__(self, reward_history: Union[np.ndarray, List[float]],
                 timesteps: Optional[np.ndarray] = None):
        """
        Initialize with reward history.

        Args:
            reward_history: Array of rewards over training (shape: [n_evaluations])
            timesteps: Array of timesteps when evaluations occurred
        """
        self.rewards = np.array(reward_history)

        if timesteps is None:
            self.timesteps = np.arange(len(self.rewards))
        else:
            self.timesteps = np.array(timesteps)

        # Calculate performance thresholds
        self.max_reward = np.max(self.rewards)
        self.min_reward = np.min(self.rewards)
        self.final_reward = self.rewards[-1]

    def time_to_threshold(self, threshold: float) -> Optional[int]:
        """
        Calculate timesteps needed to reach a performance threshold.

        Args:
            threshold: Target reward threshold

        Returns:
            Number of timesteps to reach threshold, or None if never reached
        """
        indices = np.where(self.rewards >= threshold)[0]
        if len(indices) == 0:
            return None
        return int(self.timesteps[indices[0]])

    def time_to_percentage_of_max(self, percentage: float = 0.9) -> Optional[int]:
        """
        Time to reach a percentage of maximum observed performance.

        Args:
            percentage: Target percentage (0-1) of max performance

        Returns:
            Timesteps to reach percentage of max, or None if never reached
        """
        threshold = self.min_reward + (self.max_reward - self.min_reward) * percentage
        return self.time_to_threshold(threshold)

    def jumpstart_performance(self) -> float:
        """
        Initial performance of the algorithm (first evaluation).

        Higher jumpstart indicates better initialization or transfer learning.

        Returns:
            Initial reward value
        """
        return float(self.rewards[0])

    def asymptotic_performance(self, last_n: int = 10) -> Tuple[float, float]:
        """
        Final performance averaged over last n evaluations.

        Args:
            last_n: Number of final evaluations to average

        Returns:
            Tuple of (mean, std) of final performance
        """
        final_rewards = self.rewards[-last_n:]
        return float(np.mean(final_rewards)), float(np.std(final_rewards))

    def learning_rate_metric(self, window: int = 10) -> np.ndarray:
        """
        Calculate learning rate over time using moving average gradient.

        Args:
            window: Window size for moving average

        Returns:
            Array of learning rates over time
        """
        if len(self.rewards) < window + 1:
            return np.array([0.0])

        # Smooth rewards
        smoothed = np.convolve(self.rewards, np.ones(window)/window, mode='valid')

        # Calculate gradient
        learning_rates = np.gradient(smoothed)

        return learning_rates

    def sample_efficiency_score(self, target_reward: float) -> float:
        """
        Calculate efficiency score based on area above learning curve.

        Lower scores indicate better sample efficiency (faster learning).

        Args:
            target_reward: Target performance level

        Returns:
            Efficiency score (lower is better)
        """
        # Calculate area between target and actual performance
        deficits = np.maximum(0, target_reward - self.rewards)

        # Integrate over timesteps
        if len(self.timesteps) > 1:
            score = np.trapz(deficits, self.timesteps)
        else:
            score = deficits[0]

        return float(score)

    def performance_percentiles(self, percentiles: List[int] = [25, 50, 75, 90]) -> Dict[int, Optional[int]]:
        """
        Calculate time to reach various performance percentiles.

        Args:
            percentiles: List of percentiles to calculate (0-100)

        Returns:
            Dictionary mapping percentile to timesteps needed
        """
        results = {}

        for p in percentiles:
            percentage = p / 100.0
            timesteps = self.time_to_percentage_of_max(percentage)
            results[p] = timesteps

        return results

    def convergence_timestep(self, tolerance: float = 0.05,
                           window: int = 20) -> Optional[int]:
        """
        Detect when algorithm has converged (performance plateaus).

        Args:
            tolerance: Maximum std dev to consider converged
            window: Window size to check for convergence

        Returns:
            Timestep when converged, or None if not converged
        """
        if len(self.rewards) < window:
            return None

        for i in range(len(self.rewards) - window):
            window_rewards = self.rewards[i:i+window]
            if np.std(window_rewards) / (np.mean(window_rewards) + 1e-8) < tolerance:
                return int(self.timesteps[i])

        return None

    def relative_efficiency(self, baseline_rewards: np.ndarray) -> float:
        """
        Calculate relative efficiency compared to baseline algorithm.

        Args:
            baseline_rewards: Reward history of baseline algorithm

        Returns:
            Relative efficiency ratio (>1 means more efficient than baseline)
        """
        # Calculate AUC for both
        our_auc = np.trapz(self.rewards, self.timesteps) if len(self.timesteps) > 1 else self.rewards[0]

        baseline_timesteps = self.timesteps[:len(baseline_rewards)]
        baseline_auc = np.trapz(baseline_rewards[:len(baseline_timesteps)],
                               baseline_timesteps) if len(baseline_timesteps) > 1 else baseline_rewards[0]

        return float(our_auc / (baseline_auc + 1e-8))

    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of all sample efficiency metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        asymptotic_mean, asymptotic_std = self.asymptotic_performance()

        summary = {
            'jumpstart_performance': self.jumpstart_performance(),
            'asymptotic_performance': {
                'mean': asymptotic_mean,
                'std': asymptotic_std
            },
            'max_reward': float(self.max_reward),
            'final_reward': float(self.final_reward),
            'time_to_90_percent': self.time_to_percentage_of_max(0.9),
            'time_to_95_percent': self.time_to_percentage_of_max(0.95),
            'convergence_timestep': self.convergence_timestep(),
            'performance_percentiles': self.performance_percentiles(),
            'mean_learning_rate': float(np.mean(np.abs(self.learning_rate_metric())))
        }

        return summary

    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


def compare_sample_efficiency(algorithms: Dict[str, np.ndarray],
                             timesteps: Optional[np.ndarray] = None,
                             target_percentile: float = 0.9) -> Dict:
    """
    Compare sample efficiency across multiple algorithms.

    Args:
        algorithms: Dictionary mapping algorithm names to reward histories
        timesteps: Common timestep array for all algorithms
        target_percentile: Performance percentile for comparison

    Returns:
        Comparison results dictionary
    """
    results = {}

    # Calculate metrics for each algorithm
    for name, rewards in algorithms.items():
        metric = SampleEfficiencyMetrics(rewards, timesteps)
        results[name] = metric.get_summary()

    # Add relative comparisons
    comparison = {
        'individual_metrics': results,
        'rankings': {}
    }

    # Rank by time to target percentile
    times_to_target = {}
    for name, metrics in results.items():
        time = metrics[f'time_to_{int(target_percentile*100)}_percent']
        if time is not None:
            times_to_target[name] = time

    if times_to_target:
        sorted_by_time = sorted(times_to_target.items(), key=lambda x: x[1])
        comparison['rankings']['by_time_to_target'] = [name for name, _ in sorted_by_time]

    # Rank by final performance
    final_performances = {name: metrics['final_reward']
                         for name, metrics in results.items()}
    sorted_by_final = sorted(final_performances.items(),
                            key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_final_performance'] = [name for name, _ in sorted_by_final]

    # Rank by asymptotic performance
    asymptotic_performances = {name: metrics['asymptotic_performance']['mean']
                              for name, metrics in results.items()}
    sorted_by_asymptotic = sorted(asymptotic_performances.items(),
                                 key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_asymptotic_performance'] = [name for name, _ in sorted_by_asymptotic]

    return comparison


def load_benchmark_results(results_dir: str) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load reward histories from benchmark results.

    Args:
        results_dir: Path to benchmark results directory

    Returns:
        Tuple of (algorithm_rewards_dict, timesteps_array)
    """
    results_path = Path(results_dir)

    algorithm_rewards = {}
    timesteps = None

    # Load evaluation results
    for eval_file in results_path.glob('*/evaluations.json'):
        algorithm_name = eval_file.parent.name

        with open(eval_file, 'r') as f:
            data = json.load(f)

        if 'results' in data:
            rewards = [r['mean_reward'] for r in data['results']]
            algorithm_rewards[algorithm_name] = np.array(rewards)

            if timesteps is None and 'timesteps' in data['results'][0]:
                timesteps = np.array([r['timesteps'] for r in data['results']])

    if timesteps is None:
        # Create default timesteps if not found
        max_len = max(len(r) for r in algorithm_rewards.values())
        timesteps = np.arange(max_len) * 1000  # Assume evaluations every 1000 steps

    return algorithm_rewards, timesteps