"""
Usage examples for Final Performance Metrics

This script demonstrates how to use the final_performance module to evaluate
the asymptotic/final performance of RL algorithms across multiple seeds.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from final_performance import (
    FinalPerformanceMetrics,
    compare_final_performance,
    aggregate_across_environments
)


def example_single_algorithm():
    """Example: Analyze final performance of a single algorithm with multiple seeds."""
    print("=" * 80)
    print("Example 1: Single Algorithm Analysis (Multiple Seeds)")
    print("=" * 80)

    # Simulate reward histories from 5 different seeds
    # Shape: [n_seeds, n_evaluations]
    n_seeds = 5
    n_evals = 100

    # Create different learning curves for each seed
    np.random.seed(42)
    reward_histories = []

    for seed in range(n_seeds):
        # Each seed has slightly different performance
        base_curve = -100 + 200 * (1 - np.exp(-np.arange(n_evals) / 20))
        noise = np.random.normal(0, 15, n_evals)
        seed_variance = np.random.normal(0, 20)  # Seed-specific bias

        rewards = base_curve + noise + seed_variance
        reward_histories.append(rewards)

    reward_histories = np.array(reward_histories)

    # Create metrics calculator
    metrics = FinalPerformanceMetrics(reward_histories, eval_window=10)

    # Calculate various metrics
    print("\n1. Final Performance (last 10 evaluations):")
    mean, std, stderr = metrics.final_performance()
    print(f"   Mean: {mean:.2f}")
    print(f"   Std: {std:.2f}")
    print(f"   Standard Error: {stderr:.2f}")

    print("\n2. 95% Confidence Interval:")
    ci_low, ci_high = metrics.confidence_interval(0.95)
    print(f"   [{ci_low:.2f}, {ci_high:.2f}]")

    print("\n3. Best and Worst Seeds:")
    best, best_idx = metrics.best_seed_performance()
    worst, worst_idx = metrics.worst_seed_performance()
    print(f"   Best: {best:.2f} (seed {best_idx})")
    print(f"   Worst: {worst:.2f} (seed {worst_idx})")
    print(f"   Range: {metrics.performance_range():.2f}")

    print("\n4. Performance Quantiles:")
    quantiles = metrics.performance_quantiles([0.25, 0.5, 0.75])
    for q, val in quantiles.items():
        print(f"   {int(q*100)}th percentile: {val:.2f}")

    print("\n5. Coefficient of Variation:")
    cv = metrics.coefficient_of_variation()
    print(f"   CV: {cv:.4f} (lower is more stable)")

    print("\n6. Success Rate (threshold = 80):")
    success = metrics.success_rate(threshold=80)
    print(f"   {success*100:.1f}% of seeds achieved reward ≥ 80")

    # Save metrics
    metrics.save_metrics('final_performance_results.json')
    print("\n✓ Metrics saved to final_performance_results.json")


def example_algorithm_comparison():
    """Example: Compare final performance across algorithms."""
    print("\n" + "=" * 80)
    print("Example 2: Algorithm Comparison")
    print("=" * 80)

    # Simulate multiple algorithms, each with multiple seeds
    n_seeds = 10
    n_evals = 50
    np.random.seed(42)

    algorithms = {}

    # PPO: High mean, low variance
    ppo_rewards = []
    for seed in range(n_seeds):
        rewards = 180 + np.random.normal(0, 10, n_evals)
        rewards[:10] = np.linspace(-100, 100, 10) + np.random.normal(0, 20, 10)
        ppo_rewards.append(rewards)
    algorithms['PPO'] = np.array(ppo_rewards)

    # A2C: Medium mean, medium variance
    a2c_rewards = []
    for seed in range(n_seeds):
        rewards = 150 + np.random.normal(0, 20, n_evals)
        rewards[:10] = np.linspace(-100, 80, 10) + np.random.normal(0, 25, 10)
        a2c_rewards.append(rewards)
    algorithms['A2C'] = np.array(a2c_rewards)

    # DQN: High mean but high variance
    dqn_rewards = []
    for seed in range(n_seeds):
        rewards = 170 + np.random.normal(0, 35, n_evals)
        rewards[:10] = np.linspace(-100, 90, 10) + np.random.normal(0, 30, 10)
        dqn_rewards.append(rewards)
    algorithms['DQN'] = np.array(dqn_rewards)

    # Random: Low performance
    random_rewards = []
    for seed in range(n_seeds):
        rewards = -50 + np.random.normal(0, 15, n_evals)
        random_rewards.append(rewards)
    algorithms['Random'] = np.array(random_rewards)

    # Compare algorithms
    comparison = compare_final_performance(algorithms)

    print("\n1. Individual Algorithm Metrics:")
    for algo, metrics in comparison['individual_metrics'].items():
        perf = metrics['final_performance']
        ci = metrics['confidence_interval_95']
        print(f"\n   {algo}:")
        print(f"   - Mean ± Std: {perf['mean']:.2f} ± {perf['std']:.2f}")
        print(f"   - 95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
        print(f"   - CV: {metrics['coefficient_of_variation']:.4f}")

    print("\n2. Rankings:")
    print("\n   By Mean Performance:")
    for i, algo in enumerate(comparison['rankings']['by_mean'], 1):
        mean = comparison['individual_metrics'][algo]['final_performance']['mean']
        print(f"   {i}. {algo}: {mean:.2f}")

    print("\n   By Lower Confidence Bound (conservative):")
    for i, algo in enumerate(comparison['rankings']['by_lower_confidence_bound'], 1):
        lcb = comparison['individual_metrics'][algo]['confidence_interval_95']['lower']
        print(f"   {i}. {algo}: {lcb:.2f}")

    print("\n3. Statistical Tests (pairwise comparisons):")
    for test, results in comparison['statistical_tests'].items():
        algos = test.replace('_vs_', ' vs ')
        significant = "✓" if results['significant_at_0.05'] else "✗"
        print(f"   {algos}: p={results['p_value']:.4f} {significant}")


def example_normalized_scores():
    """Example: Calculate normalized and human-normalized scores."""
    print("\n" + "=" * 80)
    print("Example 3: Normalized Scores (Atari-style)")
    print("=" * 80)

    # Simulate Breakout scores (example from Atari)
    n_seeds = 5
    n_evals = 30

    # Agent scores
    agent_rewards = []
    for seed in range(n_seeds):
        # Final scores around 300-400
        rewards = np.linspace(0, 350, n_evals) + np.random.normal(0, 30, n_evals)
        agent_rewards.append(rewards)

    metrics = FinalPerformanceMetrics(np.array(agent_rewards))

    # Known baselines (from literature)
    random_score = 1.7  # Random policy average
    human_score = 31.8  # Human expert average

    print("\n1. Raw Performance:")
    mean, std, _ = metrics.final_performance()
    print(f"   Agent: {mean:.2f} ± {std:.2f}")
    print(f"   Random baseline: {random_score:.2f}")
    print(f"   Human baseline: {human_score:.2f}")

    print("\n2. Human-Normalized Score:")
    hns = metrics.human_normalized_score(random_score, human_score)
    print(f"   Score: {hns:.2%}")
    print(f"   Interpretation: Agent achieves {hns*100:.1f}% of human performance")
    print(f"   (0% = random, 100% = human level)")

    print("\n3. Simple Normalized Score (0-1):")
    # Assume max possible score is 500
    normalized = metrics.normalized_score(min_reward=0, max_reward=500)
    print(f"   Score: {normalized:.2%} of maximum possible")


def example_optimality_gap():
    """Example: Calculate optimality gap from known optimal performance."""
    print("\n" + "=" * 80)
    print("Example 4: Optimality Gap Analysis")
    print("=" * 80)

    # Simulate CartPole with known optimal (500 max reward)
    n_seeds = 8
    n_evals = 40
    optimal_reward = 500

    # Create rewards approaching but not reaching optimal
    rewards = []
    for seed in range(n_seeds):
        # Different seeds reach different levels
        final_performance = 400 + np.random.uniform(-50, 80)
        curve = np.linspace(0, final_performance, n_evals)
        curve += np.random.normal(0, 20, n_evals)
        rewards.append(np.minimum(curve, optimal_reward))  # Cap at optimal

    metrics = FinalPerformanceMetrics(np.array(rewards))

    print(f"\n1. Known Optimal Performance: {optimal_reward}")

    mean, std, _ = metrics.final_performance()
    print(f"\n2. Agent Performance: {mean:.2f} ± {std:.2f}")

    gap_mean, gap_std = metrics.optimality_gap(optimal_reward)
    print(f"\n3. Optimality Gap: {gap_mean:.2f} ± {gap_std:.2f}")
    print(f"   (Agent is {gap_mean:.2f} points below optimal)")

    print(f"\n4. Percentage of Optimal: {(mean/optimal_reward)*100:.1f}%")

    success = metrics.success_rate(threshold=450)
    print(f"\n5. Success Rate (≥450): {success*100:.1f}% of seeds")


def example_aggregate_environments():
    """Example: Aggregate performance across multiple environments."""
    print("\n" + "=" * 80)
    print("Example 5: Aggregate Across Environments")
    print("=" * 80)

    # Simulate results from multiple environments
    np.random.seed(42)
    env_results = {}

    environments = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    algorithms = ['PPO', 'A2C', 'DQN']

    for env in environments:
        env_results[env] = {}

        for algo in algorithms:
            # Create fake reward histories
            n_seeds = 5
            n_evals = 30

            # Different performance scales for different environments
            if env == 'CartPole-v1':
                base_reward = {'PPO': 450, 'A2C': 400, 'DQN': 420}[algo]
                scale = 50
            elif env == 'Acrobot-v1':
                base_reward = {'PPO': -100, 'A2C': -120, 'DQN': -110}[algo]
                scale = 30
            else:  # MountainCar
                base_reward = {'PPO': -120, 'A2C': -140, 'DQN': -130}[algo]
                scale = 20

            rewards = []
            for seed in range(n_seeds):
                seed_rewards = base_reward + np.random.normal(0, scale, n_evals)
                rewards.append(seed_rewards)

            env_results[env][algo] = np.array(rewards)

    # Aggregate across environments
    aggregate = aggregate_across_environments(env_results)

    print("\nNormalized Scores Across All Environments:")
    print("(0 = worst observed, 1 = best observed for each environment)")

    for algo, scores in aggregate.items():
        print(f"\n{algo}:")
        print(f"  Mean normalized: {scores['mean_normalized_score']:.3f}")
        print(f"  Median normalized: {scores['median_normalized_score']:.3f}")
        print(f"  Std normalized: {scores['std_normalized_score']:.3f}")

    # Rank algorithms by mean normalized score
    ranked = sorted(aggregate.items(),
                   key=lambda x: x[1]['mean_normalized_score'],
                   reverse=True)

    print("\nOverall Ranking (by mean normalized score):")
    for i, (algo, scores) in enumerate(ranked, 1):
        print(f"  {i}. {algo}: {scores['mean_normalized_score']:.3f}")


def example_robustness_analysis():
    """Example: Analyze robustness of final performance."""
    print("\n" + "=" * 80)
    print("Example 6: Robustness Analysis")
    print("=" * 80)

    # Create two algorithms: one stable, one unstable
    n_seeds = 20
    n_evals = 50

    # Stable algorithm: consistent across seeds
    stable_rewards = []
    for seed in range(n_seeds):
        rewards = 100 + np.random.normal(0, 5, n_evals)  # Low variance
        stable_rewards.append(rewards)

    # Unstable algorithm: high variance across seeds
    unstable_rewards = []
    for seed in range(n_seeds):
        # Some seeds work great, others fail
        if seed % 4 == 0:  # 25% failure rate
            rewards = -50 + np.random.normal(0, 10, n_evals)
        else:
            rewards = 120 + np.random.normal(0, 15, n_evals)
        unstable_rewards.append(rewards)

    # Analyze both
    stable_metrics = FinalPerformanceMetrics(np.array(stable_rewards))
    unstable_metrics = FinalPerformanceMetrics(np.array(unstable_rewards))

    print("\n1. Stable Algorithm:")
    mean, std, _ = stable_metrics.final_performance()
    print(f"   Mean ± Std: {mean:.2f} ± {std:.2f}")
    print(f"   CV: {stable_metrics.coefficient_of_variation():.4f}")
    print(f"   Range: {stable_metrics.performance_range():.2f}")
    quantiles = stable_metrics.performance_quantiles([0.05, 0.95])
    print(f"   5th-95th percentile: [{quantiles[0.05]:.2f}, {quantiles[0.95]:.2f}]")

    print("\n2. Unstable Algorithm:")
    mean, std, _ = unstable_metrics.final_performance()
    print(f"   Mean ± Std: {mean:.2f} ± {std:.2f}")
    print(f"   CV: {unstable_metrics.coefficient_of_variation():.4f}")
    print(f"   Range: {unstable_metrics.performance_range():.2f}")
    quantiles = unstable_metrics.performance_quantiles([0.05, 0.95])
    print(f"   5th-95th percentile: [{quantiles[0.05]:.2f}, {quantiles[0.95]:.2f}]")

    print("\n3. Robustness Comparison:")
    print(f"   Stable algorithm CV: {stable_metrics.coefficient_of_variation():.4f}")
    print(f"   Unstable algorithm CV: {unstable_metrics.coefficient_of_variation():.4f}")
    print(f"   → Stable algorithm is {unstable_metrics.coefficient_of_variation() / stable_metrics.coefficient_of_variation():.1f}x more consistent")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("FINAL PERFORMANCE METRICS - USAGE EXAMPLES")
    print("=" * 80)

    # Run all examples
    example_single_algorithm()
    example_algorithm_comparison()
    example_normalized_scores()
    example_optimality_gap()
    example_aggregate_environments()
    example_robustness_analysis()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()