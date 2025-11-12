"""
Usage examples for Interquartile Mean (IQM) and Robust Statistics

This script demonstrates how to use IQM for robust evaluation of RL algorithms,
as recommended by Agarwal et al. (2021).
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from iqm import InterquartileMean, stratified_iqm


def example_basic_iqm():
    """Example: Basic IQM calculation and comparison with mean."""
    print("=" * 80)
    print("Example 1: Basic IQM vs Mean vs Median")
    print("=" * 80)

    # Create data with outliers
    np.random.seed(42)

    # Normal performance scores
    normal_scores = np.random.normal(100, 15, 80)

    # Add some outliers (both high and low)
    outlier_low = np.array([-50, -30, -20])  # Failed runs
    outlier_high = np.array([300, 350, 400])  # Lucky runs

    all_scores = np.concatenate([normal_scores, outlier_low, outlier_high])
    np.random.shuffle(all_scores)

    # Calculate statistics
    iqm_calc = InterquartileMean(all_scores)
    robust_stats = iqm_calc.compute_robust_statistics()

    print("\n1. Data Summary:")
    print(f"   Total samples: {len(all_scores)}")
    print(f"   Contains {len(outlier_low)} low outliers and {len(outlier_high)} high outliers")

    print("\n2. Central Tendency Measures:")
    print(f"   Mean:          {robust_stats['mean']:.2f} (sensitive to outliers)")
    print(f"   Median:        {robust_stats['median']:.2f} (robust)")
    print(f"   IQM:           {robust_stats['iqm']:.2f} (robust, uses middle 50%)")
    print(f"   Trimmed Mean:  {robust_stats['trimmed_mean_10']:.2f} (removes 10% each end)")

    print("\n3. Variability Measures:")
    print(f"   Std Dev:       {robust_stats['std']:.2f} (sensitive to outliers)")
    print(f"   MAD:           {robust_stats['mad']:.2f} (robust)")
    print(f"   IQR:           {robust_stats['iqr']:.2f} (Q3 - Q1)")

    print("\n4. Outlier Detection:")
    outliers = iqm_calc.detect_outliers()
    print(f"   Outliers detected: {outliers['n_outliers']}")
    print(f"   Outlier percentage: {outliers['percentage']:.1f}%")

    print("\n→ IQM provides a robust estimate less affected by outliers")


def example_algorithm_comparison():
    """Example: Compare algorithms using IQM vs traditional mean."""
    print("\n" + "=" * 80)
    print("Example 2: Algorithm Comparison with IQM")
    print("=" * 80)

    np.random.seed(42)

    # Simulate different algorithms with varying robustness
    algorithms = {}

    # Algorithm A: Consistent but moderate performance
    algo_a = np.random.normal(80, 10, 100)
    algorithms['Algo_A_Consistent'] = algo_a

    # Algorithm B: Usually better but occasional failures
    algo_b = np.concatenate([
        np.random.normal(95, 8, 85),  # Good performance 85% of time
        np.random.normal(-20, 10, 15)  # Catastrophic failures 15% of time
    ])
    np.random.shuffle(algo_b)
    algorithms['Algo_B_Unstable'] = algo_b

    # Algorithm C: High variance
    algo_c = np.random.normal(85, 35, 100)
    algorithms['Algo_C_HighVar'] = algo_c

    # Compare using IQM
    iqm_calc = InterquartileMean(algorithms)
    comparison = iqm_calc.compare_algorithms_iqm()

    print("\n1. Individual Algorithm Statistics:")
    for algo, stats in comparison['individual_stats'].items():
        print(f"\n   {algo}:")
        print(f"   - Mean:  {stats['mean']:6.2f}")
        print(f"   - IQM:   {stats['iqm']:6.2f}")
        print(f"   - Median:{stats['median']:6.2f}")
        print(f"   - Std:   {stats['std']:6.2f}")
        print(f"   - Outliers: {stats['n_outliers']} ({stats['outlier_percentage']:.1f}%)")

    print("\n2. Rankings Comparison:")
    print("\n   By Mean (traditional):")
    for i, algo in enumerate(comparison['rankings']['by_mean'], 1):
        mean = comparison['individual_stats'][algo]['mean']
        print(f"   {i}. {algo}: {mean:.2f}")

    print("\n   By IQM (robust):")
    for i, algo in enumerate(comparison['rankings']['by_iqm'], 1):
        iqm = comparison['individual_stats'][algo]['iqm']
        print(f"   {i}. {algo}: {iqm:.2f}")

    print("\n   → Rankings differ! IQM favors consistent performance")

    print("\n3. Robustness Analysis:")
    for algo, robust in comparison['robustness_analysis'].items():
        print(f"\n   {algo}:")
        print(f"   - IQM-Mean difference: {robust['iqm_mean_difference']:.2f}")
        print(f"   - Outlier sensitivity: {robust['outlier_sensitivity']:.1f}%")


def example_bootstrap_confidence():
    """Example: Bootstrap confidence intervals for IQM."""
    print("\n" + "=" * 80)
    print("Example 3: Bootstrap Confidence Intervals")
    print("=" * 80)

    np.random.seed(42)

    # Simulate performance scores with limited samples
    n_samples = 30  # Small sample size
    scores = np.random.normal(100, 20, n_samples)

    # Add a few outliers
    scores[0] = 200  # High outlier
    scores[1] = 10   # Low outlier

    iqm_calc = InterquartileMean(scores)

    # Calculate bootstrap CI
    iqm, ci_lower, ci_upper = iqm_calc.bootstrap_iqm_confidence_interval(
        n_bootstrap=5000,
        confidence=0.95
    )

    print(f"\n1. Sample Information:")
    print(f"   Number of samples: {n_samples}")
    print(f"   Mean: {np.mean(scores):.2f}")
    print(f"   Std: {np.std(scores):.2f}")

    print(f"\n2. IQM with 95% Confidence Interval:")
    print(f"   IQM: {iqm:.2f}")
    print(f"   95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print(f"   CI Width: {ci_upper - ci_lower:.2f}")

    # Compare with mean CI (using bootstrap for fairness)
    mean_samples = []
    for _ in range(5000):
        sample = np.random.choice(scores, n_samples, replace=True)
        mean_samples.append(np.mean(sample))

    mean_ci_lower = np.percentile(mean_samples, 2.5)
    mean_ci_upper = np.percentile(mean_samples, 97.5)

    print(f"\n3. Mean with 95% CI (for comparison):")
    print(f"   Mean: {np.mean(scores):.2f}")
    print(f"   95% CI: [{mean_ci_lower:.2f}, {mean_ci_upper:.2f}]")
    print(f"   CI Width: {mean_ci_upper - mean_ci_lower:.2f}")

    print("\n→ IQM often has tighter confidence intervals due to robustness")


def example_multi_environment():
    """Example: Aggregate IQM across multiple environments."""
    print("\n" + "=" * 80)
    print("Example 4: Multi-Environment Aggregation")
    print("=" * 80)

    np.random.seed(42)

    # Simulate scores across different environments
    env_scores = {}

    # Easy environment (high scores)
    env_scores['CartPole'] = np.random.normal(450, 30, 50)

    # Medium environment (moderate scores)
    env_scores['Acrobot'] = np.random.normal(-100, 20, 50)

    # Hard environment (low scores)
    env_scores['MountainCar'] = np.random.normal(-150, 15, 50)

    # Very different scale
    env_scores['Humanoid'] = np.random.normal(5000, 500, 50)

    iqm_calc = InterquartileMean({})

    print("\n1. Individual Environment IQMs (raw scores):")
    for env, scores in env_scores.items():
        iqm = iqm_calc.compute_iqm(scores)
        print(f"   {env:15s}: {iqm:8.2f}")

    print("\n2. Normalized Aggregate IQM:")
    aggregate_iqm = iqm_calc.aggregate_iqm_across_environments(env_scores)
    print(f"   Aggregate IQM (0-1 scale): {aggregate_iqm:.3f}")

    print("\n3. Stratified IQM (weighted by difficulty):")
    # Weight harder environments more
    weights = {
        'CartPole': 0.1,      # Easy
        'Acrobot': 0.2,       # Medium
        'MountainCar': 0.3,   # Hard
        'Humanoid': 0.4       # Very hard
    }

    stratified = stratified_iqm(env_scores, weights)
    print(f"   Stratified IQM: {stratified:.2f}")

    print("\n→ Different aggregation methods for different evaluation needs")


def example_performance_profiles():
    """Example: Performance profiles using IQM."""
    print("\n" + "=" * 80)
    print("Example 5: Performance Profiles with IQM")
    print("=" * 80)

    np.random.seed(42)

    # Simulate multiple runs for different algorithms
    n_runs = 20
    n_envs = 10

    algorithms = {}

    # Strong algorithm
    strong_scores = np.random.normal(80, 10, (n_runs, n_envs))
    algorithms['Strong'] = strong_scores

    # Medium algorithm
    medium_scores = np.random.normal(60, 15, (n_runs, n_envs))
    algorithms['Medium'] = medium_scores

    # Weak algorithm
    weak_scores = np.random.normal(40, 12, (n_runs, n_envs))
    algorithms['Weak'] = weak_scores

    # Inconsistent algorithm (sometimes great, sometimes terrible)
    inconsistent_scores = np.concatenate([
        np.random.normal(90, 5, (n_runs//2, n_envs)),
        np.random.normal(20, 10, (n_runs//2, n_envs))
    ])
    np.random.shuffle(inconsistent_scores)
    algorithms['Inconsistent'] = inconsistent_scores.reshape(n_runs, n_envs)

    iqm_calc = InterquartileMean(algorithms)

    print("\n1. Algorithm IQMs:")
    for name, scores in algorithms.items():
        iqm = iqm_calc.compute_iqm(scores)
        mean = np.mean(scores)
        print(f"   {name:12s}: IQM={iqm:.2f}, Mean={mean:.2f}")

    print("\n2. Performance Profiles (% runs with IQM ≥ threshold):")
    thresholds = [30, 50, 70, 90]

    for threshold in thresholds:
        print(f"\n   Threshold ≥ {threshold}:")
        profiles = iqm_calc.performance_profile_iqm(threshold)

        for name, success_rate in sorted(profiles.items(),
                                        key=lambda x: x[1], reverse=True):
            print(f"   - {name:12s}: {success_rate*100:.1f}%")


def example_outlier_robustness():
    """Example: Demonstrate IQM's robustness to different outlier types."""
    print("\n" + "=" * 80)
    print("Example 6: Robustness to Different Outlier Types")
    print("=" * 80)

    np.random.seed(42)
    base_scores = np.random.normal(100, 10, 95)

    scenarios = {
        'No_Outliers': base_scores.copy(),
        'Low_Outliers': np.concatenate([base_scores, np.array([-100, -80, -60, -40, -20])]),
        'High_Outliers': np.concatenate([base_scores, np.array([200, 250, 300, 350, 400])]),
        'Both_Outliers': np.concatenate([base_scores, np.array([-100, -50, 0, 300, 400])]),
        'Many_Outliers': np.concatenate([base_scores, np.random.uniform(-200, 500, 20)])
    }

    print("\nImpact of Outliers on Different Statistics:")
    print("\n{:15s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Scenario", "Mean", "Median", "IQM", "Trimmed", "Outliers"
    ))
    print("-" * 70)

    for scenario_name, scores in scenarios.items():
        iqm_calc = InterquartileMean(scores)
        stats = iqm_calc.compute_robust_statistics()

        print("{:15s} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8d}".format(
            scenario_name,
            stats['mean'],
            stats['median'],
            stats['iqm'],
            stats['trimmed_mean_10'],
            stats['n_outliers']
        ))

    print("\n→ IQM remains stable across different outlier scenarios")


def example_statistical_comparison():
    """Example: Compare IQM with other robust estimators."""
    print("\n" + "=" * 80)
    print("Example 7: Comparison of Robust Estimators")
    print("=" * 80)

    np.random.seed(42)

    # Create different distributions
    distributions = {
        'Normal': np.random.normal(100, 15, 1000),
        'Skewed': np.random.gamma(2, 20, 1000),
        'Heavy_Tailed': np.random.standard_t(3, 1000) * 20 + 100,
        'Bimodal': np.concatenate([
            np.random.normal(80, 10, 500),
            np.random.normal(120, 10, 500)
        ])
    }

    print("\nRobust Estimators Across Different Distributions:")
    print("\n{:15s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}".format(
        "Distribution", "Mean", "Median", "IQM", "Trimmed", "Winsor"
    ))
    print("-" * 70)

    for dist_name, scores in distributions.items():
        iqm_calc = InterquartileMean(scores)
        stats = iqm_calc.compute_robust_statistics()

        print("{:15s} {:8.2f} {:8.2f} {:8.2f} {:8.2f} {:8.2f}".format(
            dist_name,
            stats['mean'],
            stats['median'],
            stats['iqm'],
            stats['trimmed_mean_10'],
            stats['winsorized_mean']
        ))

    print("\n→ Different estimators perform differently on different distributions")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("INTERQUARTILE MEAN (IQM) - USAGE EXAMPLES")
    print("=" * 80)

    # Run all examples
    example_basic_iqm()
    example_algorithm_comparison()
    example_bootstrap_confidence()
    example_multi_environment()
    example_performance_profiles()
    example_outlier_robustness()
    example_statistical_comparison()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• IQM is more robust to outliers than mean")
    print("• IQM uses the middle 50% of data (between Q1 and Q3)")
    print("• Recommended by Agarwal et al. (2021) for RL evaluation")
    print("• Particularly useful when comparing algorithms with occasional failures")


if __name__ == "__main__":
    main()