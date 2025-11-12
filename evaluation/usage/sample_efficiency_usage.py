"""
Usage examples for Sample Efficiency Metrics

This script demonstrates how to use the sample_efficiency module to evaluate
and compare RL algorithms based on their learning speed and efficiency.
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from sample_efficiency import SampleEfficiencyMetrics, compare_sample_efficiency


def example_single_algorithm():
    """Example: Analyze sample efficiency of a single algorithm."""
    print("=" * 80)
    print("Example 1: Single Algorithm Analysis")
    print("=" * 80)

    # Simulate reward history (e.g., from PPO on CartPole)
    # In practice, load this from your benchmark results
    timesteps = np.linspace(0, 100000, 100)
    rewards = -200 + 400 * (1 - np.exp(-timesteps / 20000))  # Exponential learning curve
    rewards += np.random.normal(0, 10, size=len(rewards))  # Add noise

    # Create metrics calculator
    metrics = SampleEfficiencyMetrics(rewards, timesteps)

    # Calculate various metrics
    print("\n1. Jumpstart Performance (initial reward):")
    print(f"   {metrics.jumpstart_performance():.2f}")

    print("\n2. Asymptotic Performance (final 10 evaluations):")
    mean, std = metrics.asymptotic_performance()
    print(f"   Mean: {mean:.2f} ± {std:.2f}")

    print("\n3. Time to reach 90% of max performance:")
    time_90 = metrics.time_to_percentage_of_max(0.9)
    if time_90:
        print(f"   {time_90:,} timesteps")
    else:
        print("   Not reached")

    print("\n4. Convergence timestep:")
    convergence = metrics.convergence_timestep()
    if convergence:
        print(f"   Converged at {convergence:,} timesteps")
    else:
        print("   Not converged")

    print("\n5. Performance percentiles:")
    percentiles = metrics.performance_percentiles()
    for p, t in percentiles.items():
        if t is not None:
            print(f"   {p}th percentile reached at: {t:,} timesteps")
        else:
            print(f"   {p}th percentile: Not reached")

    print("\n6. Sample Efficiency Score (lower is better):")
    score = metrics.sample_efficiency_score(target_reward=180)
    print(f"   {score:.2f}")

    # Save metrics
    metrics.save_metrics('sample_efficiency_results.json')
    print("\n✓ Metrics saved to sample_efficiency_results.json")


def example_algorithm_comparison():
    """Example: Compare sample efficiency across multiple algorithms."""
    print("\n" + "=" * 80)
    print("Example 2: Algorithm Comparison")
    print("=" * 80)

    # Simulate reward histories for different algorithms
    timesteps = np.linspace(0, 100000, 100)

    # PPO: Fast initial learning, good final performance
    ppo_rewards = -200 + 400 * (1 - np.exp(-timesteps / 15000))
    ppo_rewards += np.random.normal(0, 8, size=len(ppo_rewards))

    # A2C: Moderate speed, moderate performance
    a2c_rewards = -200 + 380 * (1 - np.exp(-timesteps / 25000))
    a2c_rewards += np.random.normal(0, 12, size=len(a2c_rewards))

    # DQN: Slow start, eventual good performance
    dqn_rewards = -200 + 390 * (1 - np.exp(-timesteps / 30000))
    dqn_rewards += np.random.normal(0, 15, size=len(dqn_rewards))

    # Random: No learning
    random_rewards = np.random.normal(-150, 20, size=len(timesteps))

    algorithms = {
        'PPO': ppo_rewards,
        'A2C': a2c_rewards,
        'DQN': dqn_rewards,
        'Random': random_rewards
    }

    # Compare algorithms
    comparison = compare_sample_efficiency(algorithms, timesteps)

    print("\n1. Individual Metrics Summary:")
    for algo, metrics in comparison['individual_metrics'].items():
        print(f"\n   {algo}:")
        print(f"   - Jumpstart: {metrics['jumpstart_performance']:.2f}")
        print(f"   - Final: {metrics['final_reward']:.2f}")
        print(f"   - Time to 90%: {metrics['time_to_90_percent']}")
        print(f"   - Converged: {metrics['convergence_timestep']}")

    print("\n2. Rankings:")
    print(f"\n   By time to 90% of max (fastest first):")
    for i, algo in enumerate(comparison['rankings'].get('by_time_to_target', []), 1):
        print(f"   {i}. {algo}")

    print(f"\n   By final performance (best first):")
    for i, algo in enumerate(comparison['rankings']['by_final_performance'], 1):
        print(f"   {i}. {algo}")

    print(f"\n   By asymptotic performance (best first):")
    for i, algo in enumerate(comparison['rankings']['by_asymptotic_performance'], 1):
        print(f"   {i}. {algo}")


def example_relative_efficiency():
    """Example: Calculate relative efficiency between algorithms."""
    print("\n" + "=" * 80)
    print("Example 3: Relative Efficiency")
    print("=" * 80)

    # Create two algorithms with different learning speeds
    timesteps = np.linspace(0, 50000, 50)

    # Efficient algorithm
    efficient_rewards = -100 + 200 * (1 - np.exp(-timesteps / 10000))

    # Inefficient algorithm
    inefficient_rewards = -100 + 200 * (1 - np.exp(-timesteps / 30000))

    # Calculate relative efficiency
    efficient_metrics = SampleEfficiencyMetrics(efficient_rewards, timesteps)
    relative_eff = efficient_metrics.relative_efficiency(inefficient_rewards)

    print(f"\nRelative efficiency of efficient vs inefficient algorithm: {relative_eff:.2f}x")
    print("(Value > 1 means more efficient than baseline)")

    # Also calculate reverse
    inefficient_metrics = SampleEfficiencyMetrics(inefficient_rewards, timesteps)
    relative_eff_reverse = inefficient_metrics.relative_efficiency(efficient_rewards)

    print(f"Relative efficiency of inefficient vs efficient algorithm: {relative_eff_reverse:.2f}x")


def example_with_real_data():
    """Example: Load and analyze real benchmark results."""
    print("\n" + "=" * 80)
    print("Example 4: Real Benchmark Data (Simulated)")
    print("=" * 80)

    # Simulate loading real benchmark results
    # In practice, replace this with actual data loading
    results_dir = Path('results/20240101_comprehensive_benchmark')

    print(f"\nLoading results from: {results_dir}")
    print("(In practice, this would load actual benchmark results)")

    # Simulate realistic learning curves for different environments
    envs = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0']
    timesteps = np.linspace(0, 50000, 50)

    for env in envs:
        print(f"\n{env}:")

        # Simulate different learning curves for each environment
        if env == 'CartPole-v1':
            rewards = np.minimum(500, 100 + 400 * (1 - np.exp(-timesteps / 5000)))
        elif env == 'Acrobot-v1':
            rewards = -500 + 400 * (1 - np.exp(-timesteps / 15000))
        else:  # MountainCar
            rewards = -200 + 100 * (1 - np.exp(-timesteps / 20000))

        rewards += np.random.normal(0, 10, size=len(rewards))

        metrics = SampleEfficiencyMetrics(rewards, timesteps)

        # Print key metrics
        print(f"  Time to 90% of max: {metrics.time_to_percentage_of_max(0.9)} timesteps")
        print(f"  Convergence: {metrics.convergence_timestep()} timesteps")
        mean, std = metrics.asymptotic_performance()
        print(f"  Final performance: {mean:.2f} ± {std:.2f}")


def example_visualize_efficiency():
    """Example: Visualize sample efficiency metrics."""
    print("\n" + "=" * 80)
    print("Example 5: Visualization")
    print("=" * 80)

    # Create sample data
    timesteps = np.linspace(0, 100000, 100)
    rewards_fast = -100 + 200 * (1 - np.exp(-timesteps / 15000))
    rewards_slow = -100 + 200 * (1 - np.exp(-timesteps / 40000))

    # Create metrics
    metrics_fast = SampleEfficiencyMetrics(rewards_fast, timesteps)
    metrics_slow = SampleEfficiencyMetrics(rewards_slow, timesteps)

    # Calculate learning rates
    lr_fast = metrics_fast.learning_rate_metric(window=10)
    lr_slow = metrics_slow.learning_rate_metric(window=10)

    print("\nVisualization metrics calculated:")
    print(f"  Fast learner - Time to 90%: {metrics_fast.time_to_percentage_of_max(0.9)}")
    print(f"  Slow learner - Time to 90%: {metrics_slow.time_to_percentage_of_max(0.9)}")
    print(f"  Relative efficiency (fast/slow): {metrics_fast.relative_efficiency(rewards_slow):.2f}x")

    print("\n(In practice, you would plot these with matplotlib)")
    print("Suggested plots:")
    print("  1. Learning curves comparison")
    print("  2. Learning rate over time")
    print("  3. Time to reach percentiles bar chart")
    print("  4. Sample efficiency scores comparison")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("SAMPLE EFFICIENCY METRICS - USAGE EXAMPLES")
    print("=" * 80)

    # Run all examples
    example_single_algorithm()
    example_algorithm_comparison()
    example_relative_efficiency()
    example_with_real_data()
    example_visualize_efficiency()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()