"""
Interquartile Mean (IQM) and Robust Statistics for RL Evaluation

This module implements IQM and other robust statistical measures recommended
for RL evaluation, particularly useful when dealing with outliers.

References:
- Agarwal et al. (2021): "Deep RL at the Edge of the Statistical Precipice"
- Statistical Rethinking of Evaluation in RL
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
import json
from pathlib import Path
from scipy import stats
import warnings


class InterquartileMean:
    """
    Calculate Interquartile Mean (IQM) and other robust statistics for RL evaluation.

    IQM is the mean of the middle 50% of data, more robust to outliers than regular mean.
    """

    def __init__(self, scores: Union[np.ndarray, List[float], Dict[str, np.ndarray]]):
        """
        Initialize with performance scores.

        Args:
            scores: Can be:
                - 1D array/list of scores
                - 2D array of shape [n_seeds, n_envs] or [n_runs, n_evaluations]
                - Dict mapping algorithm names to score arrays
        """
        if isinstance(scores, dict):
            self.multi_algorithm = True
            self.scores_dict = {name: np.array(s) for name, s in scores.items()}
            self.scores = None
        else:
            self.multi_algorithm = False
            self.scores = np.array(scores)
            self.scores_dict = None

    def compute_iqm(self, data: Optional[np.ndarray] = None) -> float:
        """
        Compute Interquartile Mean (mean of middle 50% of data).

        Args:
            data: Optional data array, uses self.scores if not provided

        Returns:
            IQM value
        """
        if data is None:
            data = self.scores

        if data is None:
            raise ValueError("No data provided for IQM calculation")

        # Flatten if multidimensional
        flat_data = data.flatten()

        # Get quartiles
        q1 = np.percentile(flat_data, 25)
        q3 = np.percentile(flat_data, 75)

        # Filter data between Q1 and Q3
        iqr_data = flat_data[(flat_data >= q1) & (flat_data <= q3)]

        if len(iqr_data) == 0:
            return float(np.median(flat_data))

        return float(np.mean(iqr_data))

    def compute_trimmed_mean(self, data: Optional[np.ndarray] = None,
                           trim_percentage: float = 0.1) -> float:
        """
        Compute trimmed mean (remove top and bottom percentage).

        Args:
            data: Optional data array
            trim_percentage: Fraction to trim from each end (0.1 = 10%)

        Returns:
            Trimmed mean value
        """
        if data is None:
            data = self.scores

        if data is None:
            raise ValueError("No data provided")

        flat_data = data.flatten()
        return float(stats.trim_mean(flat_data, trim_percentage))

    def compute_winsorized_mean(self, data: Optional[np.ndarray] = None,
                              limits: Tuple[float, float] = (0.05, 0.05)) -> float:
        """
        Compute Winsorized mean (cap extreme values instead of removing).

        Args:
            data: Optional data array
            limits: (lower, upper) fraction of values to cap

        Returns:
            Winsorized mean value
        """
        if data is None:
            data = self.scores

        if data is None:
            raise ValueError("No data provided")

        flat_data = data.flatten()
        winsorized = stats.mstats.winsorize(flat_data, limits=limits)
        return float(np.mean(winsorized))

    def compute_median_absolute_deviation(self, data: Optional[np.ndarray] = None) -> float:
        """
        Compute Median Absolute Deviation (MAD), robust measure of variability.

        Args:
            data: Optional data array

        Returns:
            MAD value
        """
        if data is None:
            data = self.scores

        if data is None:
            raise ValueError("No data provided")

        flat_data = data.flatten()
        median = np.median(flat_data)
        mad = np.median(np.abs(flat_data - median))

        return float(mad)

    def compute_robust_statistics(self, data: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Compute all robust statistics.

        Args:
            data: Optional data array

        Returns:
            Dictionary with all robust statistics
        """
        if data is None:
            data = self.scores

        results = {
            'iqm': self.compute_iqm(data),
            'median': float(np.median(data)),
            'mean': float(np.mean(data)),
            'trimmed_mean_10': self.compute_trimmed_mean(data, 0.1),
            'winsorized_mean': self.compute_winsorized_mean(data),
            'mad': self.compute_median_absolute_deviation(data),
            'std': float(np.std(data)),
            'q1': float(np.percentile(data, 25)),
            'q3': float(np.percentile(data, 75)),
            'min': float(np.min(data)),
            'max': float(np.max(data))
        }

        # Add IQR
        results['iqr'] = results['q3'] - results['q1']

        # Add outlier detection
        outliers = self.detect_outliers(data)
        results['n_outliers'] = outliers['n_outliers']
        results['outlier_percentage'] = outliers['percentage']

        return results

    def detect_outliers(self, data: Optional[np.ndarray] = None,
                       method: str = 'iqr', threshold: float = 1.5) -> Dict[str, Any]:
        """
        Detect outliers using various methods.

        Args:
            data: Optional data array
            method: 'iqr', 'zscore', or 'mad'
            threshold: Threshold for outlier detection

        Returns:
            Dictionary with outlier information
        """
        if data is None:
            data = self.scores

        flat_data = data.flatten()

        if method == 'iqr':
            q1 = np.percentile(flat_data, 25)
            q3 = np.percentile(flat_data, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers_mask = (flat_data < lower_bound) | (flat_data > upper_bound)

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(flat_data))
            outliers_mask = z_scores > threshold

        elif method == 'mad':
            median = np.median(flat_data)
            mad = self.compute_median_absolute_deviation(data)
            if mad == 0:
                outliers_mask = np.zeros(len(flat_data), dtype=bool)
            else:
                modified_z_scores = 0.6745 * (flat_data - median) / mad
                outliers_mask = np.abs(modified_z_scores) > threshold

        else:
            raise ValueError(f"Unknown method: {method}")

        outliers = flat_data[outliers_mask]

        return {
            'n_outliers': int(np.sum(outliers_mask)),
            'percentage': float(np.mean(outliers_mask) * 100),
            'outlier_values': outliers.tolist(),
            'outlier_indices': np.where(outliers_mask)[0].tolist()
        }

    def compare_algorithms_iqm(self) -> Dict[str, Any]:
        """
        Compare multiple algorithms using IQM and robust statistics.

        Returns:
            Comparison results dictionary
        """
        if not self.multi_algorithm:
            raise ValueError("This method requires multiple algorithms (dict input)")

        results = {}

        # Calculate robust stats for each algorithm
        for name, scores in self.scores_dict.items():
            results[name] = self.compute_robust_statistics(scores)

        # Create comparison
        comparison = {
            'individual_stats': results,
            'rankings': {},
            'robustness_analysis': {}
        }

        # Rank by IQM
        iqm_scores = {name: stats['iqm'] for name, stats in results.items()}
        sorted_by_iqm = sorted(iqm_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings']['by_iqm'] = [name for name, _ in sorted_by_iqm]

        # Rank by median
        median_scores = {name: stats['median'] for name, stats in results.items()}
        sorted_by_median = sorted(median_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings']['by_median'] = [name for name, _ in sorted_by_median]

        # Rank by mean (for comparison)
        mean_scores = {name: stats['mean'] for name, stats in results.items()}
        sorted_by_mean = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)
        comparison['rankings']['by_mean'] = [name for name, _ in sorted_by_mean]

        # Analyze robustness (algorithms with smaller IQM-mean difference are more robust)
        for name, stats in results.items():
            robustness = abs(stats['iqm'] - stats['mean'])
            comparison['robustness_analysis'][name] = {
                'iqm_mean_difference': robustness,
                'coefficient_of_variation': stats['std'] / (abs(stats['mean']) + 1e-8),
                'iqr_to_range_ratio': stats['iqr'] / (stats['max'] - stats['min'] + 1e-8),
                'outlier_sensitivity': stats['outlier_percentage']
            }

        return comparison

    def bootstrap_iqm_confidence_interval(self, data: Optional[np.ndarray] = None,
                                         n_bootstrap: int = 10000,
                                         confidence: float = 0.95) -> Tuple[float, float, float]:
        """
        Calculate bootstrap confidence interval for IQM.

        Args:
            data: Optional data array
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level

        Returns:
            Tuple of (iqm, lower_ci, upper_ci)
        """
        if data is None:
            data = self.scores

        flat_data = data.flatten()
        n = len(flat_data)

        # Bootstrap
        iqm_samples = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(flat_data, n, replace=True)
            iqm_samples.append(self.compute_iqm(sample))

        iqm_samples = np.array(iqm_samples)

        # Calculate confidence interval
        alpha = 1 - confidence
        lower = np.percentile(iqm_samples, 100 * alpha / 2)
        upper = np.percentile(iqm_samples, 100 * (1 - alpha / 2))
        iqm = self.compute_iqm(data)

        return float(iqm), float(lower), float(upper)

    def performance_profile_iqm(self, threshold: float) -> Dict[str, float]:
        """
        Calculate performance profiles using IQM (fraction of runs above threshold).

        Args:
            threshold: Performance threshold

        Returns:
            Dict mapping algorithm names to success rates
        """
        if not self.multi_algorithm:
            raise ValueError("This method requires multiple algorithms")

        profiles = {}

        for name, scores in self.scores_dict.items():
            # Calculate IQM for each run/seed if 2D
            if len(scores.shape) == 2:
                run_iqms = [self.compute_iqm(scores[i]) for i in range(scores.shape[0])]
                success_rate = np.mean([iqm >= threshold for iqm in run_iqms])
            else:
                # For 1D, just check if IQM exceeds threshold
                iqm = self.compute_iqm(scores)
                success_rate = float(iqm >= threshold)

            profiles[name] = float(success_rate)

        return profiles

    def aggregate_iqm_across_environments(self, env_scores: Dict[str, np.ndarray]) -> float:
        """
        Calculate aggregate IQM across multiple environments.

        Args:
            env_scores: Dict mapping environment names to score arrays

        Returns:
            Aggregate IQM value
        """
        # Normalize scores per environment first
        normalized_scores = []

        for env, scores in env_scores.items():
            # Normalize to 0-1 within environment
            min_score = np.min(scores)
            max_score = np.max(scores)

            if max_score - min_score > 0:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = scores

            normalized_scores.append(normalized)

        # Concatenate all normalized scores
        all_scores = np.concatenate([s.flatten() for s in normalized_scores])

        # Calculate IQM on normalized scores
        return self.compute_iqm(all_scores)

    def get_summary(self, data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Get comprehensive summary including IQM and robust statistics.

        Args:
            data: Optional data array

        Returns:
            Summary dictionary
        """
        if data is None:
            if self.multi_algorithm:
                return self.compare_algorithms_iqm()
            else:
                data = self.scores

        robust_stats = self.compute_robust_statistics(data)
        iqm, ci_lower, ci_upper = self.bootstrap_iqm_confidence_interval(data)

        summary = {
            'robust_statistics': robust_stats,
            'iqm_confidence_interval': {
                'iqm': iqm,
                'lower_95': ci_lower,
                'upper_95': ci_upper
            },
            'outlier_analysis': self.detect_outliers(data),
            'robustness_measures': {
                'iqm_to_mean_ratio': robust_stats['iqm'] / (robust_stats['mean'] + 1e-8),
                'iqm_to_median_ratio': robust_stats['iqm'] / (robust_stats['median'] + 1e-8),
                'relative_mad': robust_stats['mad'] / (robust_stats['median'] + 1e-8)
            }
        }

        return summary

    def save_metrics(self, filepath: str):
        """Save metrics to JSON file."""
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)


def stratified_iqm(scores_by_category: Dict[str, np.ndarray],
                   weights: Optional[Dict[str, float]] = None) -> float:
    """
    Calculate stratified IQM across different categories (e.g., environment types).

    Args:
        scores_by_category: Dict mapping category names to score arrays
        weights: Optional weights for each category

    Returns:
        Stratified IQM value
    """
    if weights is None:
        weights = {cat: 1.0 for cat in scores_by_category}

    # Normalize weights
    total_weight = sum(weights.values())
    weights = {cat: w/total_weight for cat, w in weights.items()}

    # Calculate IQM for each category
    category_iqms = {}
    iqm_calculator = InterquartileMean({})

    for category, scores in scores_by_category.items():
        category_iqms[category] = iqm_calculator.compute_iqm(scores)

    # Weighted average of IQMs
    stratified = sum(category_iqms[cat] * weights[cat]
                    for cat in category_iqms)

    return float(stratified)