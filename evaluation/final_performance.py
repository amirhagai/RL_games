"""
Final Performance Metrics for RL Algorithms

This module implements metrics focused on the final/asymptotic performance
of RL algorithms, following best practices from the literature.

References:
- Machado et al. (2018): "Revisiting the Arcade Learning Environment"
- Henderson et al. (2018): "Deep Reinforcement Learning that Matters"
- Agarwal et al. (2021): "Deep RL at the Edge of the Statistical Precipice"
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import json
from pathlib import Path
from scipy import stats


class FinalPerformanceMetrics:
    """
    Calculate final/asymptotic performance metrics for RL algorithms.

    Final performance is critical for determining the quality of the learned policy,
    especially in deployment scenarios.
    """

    def __init__(self, reward_histories: Union[np.ndarray, List[List[float]]],
                 eval_window: int = 10):
        """
        Initialize with reward histories from multiple seeds.

        Args:
            reward_histories: Array of shape [n_seeds, n_evaluations] or list of lists
            eval_window: Number of final evaluations to consider for final performance
        """
        self.rewards = np.array(reward_histories)

        if len(self.rewards.shape) == 1:
            # Single seed, reshape to [1, n_evaluations]
            self.rewards = self.rewards.reshape(1, -1)

        self.n_seeds = self.rewards.shape[0]
        self.n_evaluations = self.rewards.shape[1]
        self.eval_window = min(eval_window, self.n_evaluations)

    def final_performance(self) -> Tuple[float, float, float]:
        """
        Calculate mean and confidence interval of final performance.

        Returns:
            Tuple of (mean, std, stderr) of final performance across seeds
        """
        # Take last eval_window evaluations for each seed
        final_rewards = self.rewards[:, -self.eval_window:]

        # Average over evaluation window for each seed
        seed_finals = np.mean(final_rewards, axis=1)

        mean = float(np.mean(seed_finals))
        std = float(np.std(seed_finals))
        stderr = float(std / np.sqrt(self.n_seeds))

        return mean, std, stderr

    def confidence_interval(self, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval for final performance.

        Args:
            confidence: Confidence level (default 0.95 for 95% CI)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)

        if self.n_seeds < 2:
            mean = np.mean(seed_finals)
            return float(mean), float(mean)

        # Calculate confidence interval using t-distribution
        mean = np.mean(seed_finals)
        sem = stats.sem(seed_finals)
        ci = stats.t.interval(confidence, self.n_seeds - 1, loc=mean, scale=sem)

        return float(ci[0]), float(ci[1])

    def best_seed_performance(self) -> Tuple[float, int]:
        """
        Get the best performing seed's final performance.

        Returns:
            Tuple of (best_performance, seed_index)
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)

        best_idx = np.argmax(seed_finals)
        return float(seed_finals[best_idx]), int(best_idx)

    def worst_seed_performance(self) -> Tuple[float, int]:
        """
        Get the worst performing seed's final performance.

        Returns:
            Tuple of (worst_performance, seed_index)
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)

        worst_idx = np.argmin(seed_finals)
        return float(seed_finals[worst_idx]), int(worst_idx)

    def performance_range(self) -> float:
        """
        Calculate the range of final performances across seeds.

        High range indicates high variance/instability.

        Returns:
            Range of final performances
        """
        best, _ = self.best_seed_performance()
        worst, _ = self.worst_seed_performance()
        return float(best - worst)

    def median_performance(self) -> float:
        """
        Calculate median final performance across seeds.

        Median is more robust to outliers than mean.

        Returns:
            Median final performance
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)
        return float(np.median(seed_finals))

    def performance_quantiles(self, quantiles: List[float] = [0.25, 0.5, 0.75]) -> Dict[float, float]:
        """
        Calculate quantiles of final performance.

        Args:
            quantiles: List of quantiles to calculate (0-1)

        Returns:
            Dictionary mapping quantile to value
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)

        results = {}
        for q in quantiles:
            results[q] = float(np.quantile(seed_finals, q))

        return results

    def coefficient_of_variation(self) -> float:
        """
        Calculate coefficient of variation (CV) for final performance.

        CV = std / mean, normalized measure of dispersion.

        Returns:
            Coefficient of variation
        """
        mean, std, _ = self.final_performance()

        if abs(mean) < 1e-8:
            return float('inf')

        return float(std / abs(mean))

    def success_rate(self, threshold: float) -> float:
        """
        Calculate percentage of seeds that achieve a performance threshold.

        Args:
            threshold: Performance threshold

        Returns:
            Success rate (0-1)
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)

        successes = np.sum(seed_finals >= threshold)
        return float(successes / self.n_seeds)

    def optimality_gap(self, optimal_reward: float) -> Tuple[float, float]:
        """
        Calculate gap from optimal performance.

        Args:
            optimal_reward: Known optimal reward for the task

        Returns:
            Tuple of (mean_gap, std_gap)
        """
        final_rewards = self.rewards[:, -self.eval_window:]
        seed_finals = np.mean(final_rewards, axis=1)

        gaps = optimal_reward - seed_finals
        return float(np.mean(gaps)), float(np.std(gaps))

    def normalized_score(self, min_reward: float, max_reward: float) -> float:
        """
        Calculate normalized score (0-1) based on known reward bounds.

        Args:
            min_reward: Minimum possible reward (e.g., random policy)
            max_reward: Maximum possible reward (e.g., expert/optimal)

        Returns:
            Normalized score between 0 and 1
        """
        mean, _, _ = self.final_performance()

        if max_reward - min_reward == 0:
            return 0.0

        score = (mean - min_reward) / (max_reward - min_reward)
        return float(np.clip(score, 0, 1))

    def human_normalized_score(self, random_score: float, human_score: float) -> float:
        """
        Calculate human-normalized score (common in Atari benchmarks).

        Score = (Agent - Random) / (Human - Random)

        Args:
            random_score: Score of random policy
            human_score: Score of human expert

        Returns:
            Human-normalized score (0 = random, 1 = human level)
        """
        mean, _, _ = self.final_performance()
        return self._normalize_score(mean, random_score, human_score)

    def _normalize_score(self, score: float, min_score: float, max_score: float) -> float:
        """Helper for score normalization."""
        if max_score - min_score == 0:
            return 0.0
        return float((score - min_score) / (max_score - min_score))

    def get_summary(self) -> Dict:
        """
        Get comprehensive summary of final performance metrics.

        Returns:
            Dictionary containing all calculated metrics
        """
        mean, std, stderr = self.final_performance()
        ci_low, ci_high = self.confidence_interval()
        best, best_idx = self.best_seed_performance()
        worst, worst_idx = self.worst_seed_performance()

        summary = {
            'final_performance': {
                'mean': mean,
                'std': std,
                'stderr': stderr,
                'median': self.median_performance()
            },
            'confidence_interval_95': {
                'lower': ci_low,
                'upper': ci_high
            },
            'best_seed': {
                'performance': best,
                'index': best_idx
            },
            'worst_seed': {
                'performance': worst,
                'index': worst_idx
            },
            'performance_range': self.performance_range(),
            'coefficient_of_variation': self.coefficient_of_variation(),
            'quantiles': self.performance_quantiles(),
            'n_seeds': self.n_seeds,
            'eval_window': self.eval_window
        }

        return summary

    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)


def compare_final_performance(algorithms: Dict[str, np.ndarray],
                             confidence: float = 0.95) -> Dict:
    """
    Compare final performance across multiple algorithms.

    Args:
        algorithms: Dict mapping algorithm names to reward histories [n_seeds, n_evals]
        confidence: Confidence level for intervals

    Returns:
        Comparison results dictionary
    """
    results = {}

    # Calculate metrics for each algorithm
    for name, rewards in algorithms.items():
        metric = FinalPerformanceMetrics(rewards)
        results[name] = metric.get_summary()

    # Statistical comparisons
    comparison = {
        'individual_metrics': results,
        'rankings': {},
        'statistical_tests': {}
    }

    # Rank by mean final performance
    mean_performances = {
        name: metrics['final_performance']['mean']
        for name, metrics in results.items()
    }
    sorted_by_mean = sorted(mean_performances.items(),
                           key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_mean'] = [name for name, _ in sorted_by_mean]

    # Rank by median (more robust)
    median_performances = {
        name: metrics['final_performance']['median']
        for name, metrics in results.items()
    }
    sorted_by_median = sorted(median_performances.items(),
                             key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_median'] = [name for name, _ in sorted_by_median]

    # Rank by lower confidence bound (conservative ranking)
    lcb_performances = {
        name: metrics['confidence_interval_95']['lower']
        for name, metrics in results.items()
    }
    sorted_by_lcb = sorted(lcb_performances.items(),
                          key=lambda x: x[1], reverse=True)
    comparison['rankings']['by_lower_confidence_bound'] = [name for name, _ in sorted_by_lcb]

    # Pairwise statistical tests
    if len(algorithms) > 1:
        algo_names = list(algorithms.keys())
        for i, algo1 in enumerate(algo_names[:-1]):
            for algo2 in algo_names[i+1:]:
                # Get final performances for each seed
                rewards1 = algorithms[algo1][:, -10:].mean(axis=1)
                rewards2 = algorithms[algo2][:, -10:].mean(axis=1)

                # Welch's t-test (doesn't assume equal variance)
                if len(rewards1) > 1 and len(rewards2) > 1:
                    t_stat, p_value = stats.ttest_ind(rewards1, rewards2,
                                                      equal_var=False)
                    comparison['statistical_tests'][f'{algo1}_vs_{algo2}'] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant_at_0.05': p_value < 0.05
                    }

    return comparison


def aggregate_across_environments(env_results: Dict[str, Dict[str, np.ndarray]]) -> Dict:
    """
    Aggregate final performance across multiple environments.

    Args:
        env_results: Nested dict {env_name: {algo_name: reward_history}}

    Returns:
        Aggregated metrics across environments
    """
    # Calculate normalized scores for each env/algo combination
    normalized_scores = {}

    for env, algo_rewards in env_results.items():
        # Find min/max for normalization (across all algorithms)
        all_finals = []
        for rewards in algo_rewards.values():
            finals = np.mean(rewards[:, -10:], axis=1)
            all_finals.extend(finals)

        env_min = np.min(all_finals)
        env_max = np.max(all_finals)

        # Normalize each algorithm's score
        for algo, rewards in algo_rewards.items():
            metric = FinalPerformanceMetrics(rewards)
            score = metric.normalized_score(env_min, env_max)

            if algo not in normalized_scores:
                normalized_scores[algo] = []
            normalized_scores[algo].append(score)

    # Aggregate normalized scores
    aggregate = {}
    for algo, scores in normalized_scores.items():
        aggregate[algo] = {
            'mean_normalized_score': float(np.mean(scores)),
            'median_normalized_score': float(np.median(scores)),
            'std_normalized_score': float(np.std(scores)),
            'n_environments': len(scores)
        }

    return aggregate