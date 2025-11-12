"""
Usage examples for Stability Metrics

This script demonstrates how to use the stability_metrics module to analyze
the training stability and consistency of RL algorithms.
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from stability_metrics import StabilityMetrics, compare_stability


def example_single_seed_stability():
    """Example: Analyze stability of a single training run."""
    print("=" * 80)
    print("Example 1: Single Seed Stability Analysis")
    print("=" * 80)

    # Simulate a training curve with various stability issues
    n_evals = 100
    timesteps = np.arange(n_evals) * 1000

    # Create reward history with:
    # - General upward trend
    # - Some oscillation
    # - A catastrophic failure
    # - A plateau
    rewards = []

    # Phase 1: Initial learning (0-30)
    phase1 = np.linspace(-100, 50, 30) + np.random.normal(0, 10, 30)
    rewards.extend(phase1)

    # Phase 2: Catastrophic failure (30-40)
    phase2 = np.full(10, -50) + np.random.normal(0, 15, 10)
    rewards.extend(phase2)

    # Phase 3: Recovery and improvement (40-60)
    phase3 = np.linspace(-30, 100, 20) + np.random.normal(0, 8, 20)
    rewards.extend(phase3)

    # Phase 4: Plateau (60-80)
    phase4 = np.full(20, 100) + np.random.normal(0, 5, 20)
    rewards.extend(phase4)

    # Phase 5: Final improvement (80-100)
    phase5 = np.linspace(100, 150, 20) + np.random.normal(0, 12, 20)
    rewards.extend(phase5)

    rewards = np.array(rewards)

    # Create metrics calculator
    metrics = StabilityMetrics(rewards, timesteps)

    # Calculate various metrics
    print("\n1. Overall Stability Index (0-1, higher is better):")
    print(f"   {metrics.stability_index():.3f}")

    print("\n2. Monotonicity Score (-1 to 1, 1 = perfect improvement):")
    print(f"   {metrics.monotonicity_score():.3f}")

    print("\n3. Smoothness Score (lower is smoother):")
    print(f"   {metrics.smoothness_score():.2f}")

    print("\n4. Signal-to-Noise Ratio:")
    print(f"   {metrics.signal_to_noise_ratio():.2f}")

    print("\n5. Oscillation Metric (lower is better):")
    print(f"   {metrics.oscillation_metric():.3f}")

    print("\n6. Catastrophic Failures:")
    failures = metrics.catastrophic_failures()
    print(f"   Number of failures: {failures['n_failures']}")
    if failures['failure_magnitudes']:
        print(f"   Mean magnitude: {np.mean(failures['failure_magnitudes']):.2f}")
        print(f"   Failure timesteps: {failures['failure_timesteps']}")

    print("\n7. Performance Plateaus:")
    plateaus = metrics.plateau_detection()
    print(f"   Number of plateaus: {plateaus['n_plateaus']}")
    if plateaus['plateau_lengths']:
        print(f"   Mean plateau length: {np.mean(plateaus['plateau_lengths']):.1f} evaluations")
        print(f"   Plateau starts: {plateaus['plateau_starts']}")

    # Save metrics
    metrics.save_metrics('stability_metrics_single.json')
    print("\n✓ Metrics saved to stability_metrics_single.json")


def example_multi_seed_stability():
    """Example: Analyze stability across multiple seeds."""
    print("\n" + "=" * 80)
    print("Example 2: Multi-Seed Stability Analysis")
    print("=" * 80)

    # Simulate multiple seeds with different stability characteristics
    n_seeds = 10
    n_evals = 50
    np.random.seed(42)

    # Create reward histories with varying stability
    reward_histories = []

    for seed in range(n_seeds):
        if seed < 7:  # 70% stable seeds
            # Stable learning curve
            rewards = np.linspace(-100, 100, n_evals)
            rewards += np.random.normal(0, 10, n_evals)  # Low noise
        else:  # 30% unstable seeds
            # Unstable learning curve
            rewards = np.linspace(-100, 100, n_evals)
            rewards += np.random.normal(0, 30, n_evals)  # High noise

            # Add random catastrophic failure
            if seed == 8:
                rewards[25:30] = -80

        reward_histories.append(rewards)

    reward_histories = np.array(reward_histories)

    # Create metrics calculator
    metrics = StabilityMetrics(reward_histories)

    # Analyze stability
    print("\n1. Cross-Seed Analysis:")
    print(f"   Number of seeds: {n_seeds}")
    print(f"   Cross-seed correlation: {metrics.cross_seed_correlation():.3f}")
    print("   (High correlation = consistent behavior across seeds)")

    print("\n2. Variance Analysis:")
    cv_over_time = metrics.coefficient_of_variation_over_time()
    print(f"   Mean CV over time: {np.mean(cv_over_time):.3f}")
    print(f"   Max CV: {np.max(cv_over_time):.3f}")
    print(f"   Min CV: {np.min(cv_over_time):.3f}")

    print("\n3. Stability Metrics:")
    summary = metrics.get_summary()
    print(f"   Overall stability index: {summary['stability_index']:.3f}")
    print(f"   Mean variance: {summary['mean_variance']:.2f}")
    print(f"   Catastrophic failures: {summary['catastrophic_failures']['count']}")
    print(f"   Plateau ratio: {summary['plateaus']['total_time_ratio']:.2%}")


def example_algorithm_comparison():
    """Example: Compare stability across different algorithms."""
    print("\n" + "=" * 80)
    print("Example 3: Algorithm Stability Comparison")
    print("=" * 80)

    # Simulate different algorithms with distinct stability profiles
    n_seeds = 5
    n_evals = 60
    np.random.seed(42)

    algorithms = {}

    # Stable algorithm: PPO-style (smooth, consistent)
    ppo_rewards = []
    for seed in range(n_seeds):
        rewards = 100 * (1 - np.exp(-np.arange(n_evals) / 15))
        rewards += np.random.normal(0, 5, n_evals)  # Low noise
        ppo_rewards.append(rewards)
    algorithms['PPO (Stable)'] = np.array(ppo_rewards)

    # Moderately stable: A2C-style
    a2c_rewards = []
    for seed in range(n_seeds):
        rewards = 100 * (1 - np.exp(-np.arange(n_evals) / 20))
        rewards += np.random.normal(0, 12, n_evals)  # Medium noise
        # Occasional dips
        if seed % 2 == 0:
            rewards[30:33] -= 30
        a2c_rewards.append(rewards)
    algorithms['A2C (Moderate)'] = np.array(a2c_rewards)

    # Unstable: DQN-style (high variance, occasional failures)
    dqn_rewards = []
    for seed in range(n_seeds):
        rewards = 100 * (1 - np.exp(-np.arange(n_evals) / 18))
        rewards += np.random.normal(0, 20, n_evals)  # High noise
        # Random catastrophic failures
        if seed in [1, 3]:
            failure_point = np.random.randint(20, 40)
            rewards[failure_point:failure_point+5] -= 60
        dqn_rewards.append(rewards)
    algorithms['DQN (Unstable)'] = np.array(dqn_rewards)

    # Compare algorithms
    comparison = compare_stability(algorithms)

    print("\n1. Individual Algorithm Stability:")
    for algo, metrics in comparison['individual_metrics'].items():
        print(f"\n   {algo}:")
        print(f"   - Stability index: {metrics['stability_index']:.3f}")
        print(f"   - Monotonicity: {metrics['monotonicity_score']:.3f}")
        print(f"   - Smoothness: {metrics['smoothness_score']:.2f}")
        print(f"   - Failures: {metrics['catastrophic_failures']['count']}")
        print(f"   - SNR: {metrics['signal_to_noise_ratio']:.2f}")

    print("\n2. Rankings:")
    print("\n   Most Stable (by stability index):")
    for i, algo in enumerate(comparison['rankings']['by_stability'], 1):
        score = comparison['individual_metrics'][algo]['stability_index']
        print(f"   {i}. {algo}: {score:.3f}")

    print("\n   Most Monotonic (consistent improvement):")
    for i, algo in enumerate(comparison['rankings']['by_monotonicity'], 1):
        score = comparison['individual_metrics'][algo]['monotonicity_score']
        print(f"   {i}. {algo}: {score:.3f}")

    print("\n   Most Robust (fewest failures):")
    for i, algo in enumerate(comparison['rankings']['by_robustness'], 1):
        failures = comparison['individual_metrics'][algo]['catastrophic_failures']['count']
        print(f"   {i}. {algo}: {failures} failures")


def example_detect_training_issues():
    """Example: Detect specific training issues."""
    print("\n" + "=" * 80)
    print("Example 4: Detecting Training Issues")
    print("=" * 80)

    # Create specific problematic patterns
    n_evals = 100

    print("\n1. Oscillating Training (High Variance):")
    oscillating = 50 + 30 * np.sin(np.arange(n_evals) * 0.3)
    oscillating += np.random.normal(0, 10, n_evals)
    metrics_osc = StabilityMetrics(oscillating)
    print(f"   Oscillation metric: {metrics_osc.oscillation_metric():.3f}")
    print(f"   Smoothness score: {metrics_osc.smoothness_score():.2f}")
    print(f"   Stability index: {metrics_osc.stability_index():.3f}")

    print("\n2. Plateaued Training (No Progress):")
    plateaued = np.concatenate([
        np.linspace(-100, 50, 30),  # Initial learning
        np.full(70, 50)  # Long plateau
    ])
    plateaued += np.random.normal(0, 5, n_evals)
    metrics_plat = StabilityMetrics(plateaued)
    plateaus = metrics_plat.plateau_detection()
    print(f"   Plateaus detected: {plateaus['n_plateaus']}")
    print(f"   Total plateau time: {plateaus['total_plateau_time']} evaluations")
    print(f"   Monotonicity score: {metrics_plat.monotonicity_score():.3f}")

    print("\n3. Catastrophic Forgetting:")
    forgetting = np.concatenate([
        np.linspace(-100, 100, 40),  # Good learning
        np.full(20, -50),  # Sudden drop
        np.linspace(-50, 80, 40)  # Partial recovery
    ])
    forgetting += np.random.normal(0, 10, n_evals)
    metrics_forget = StabilityMetrics(forgetting)
    failures = metrics_forget.catastrophic_failures()
    print(f"   Catastrophic failures: {failures['n_failures']}")
    if failures['failure_magnitudes']:
        print(f"   Failure magnitude: {failures['failure_magnitudes'][0]:.2f}")
        print(f"   Recovery time: {failures['recovery_times'][0]} evaluations")

    print("\n4. Noisy Training (Poor SNR):")
    noisy = np.linspace(-100, 100, n_evals)
    noisy += np.random.normal(0, 50, n_evals)  # Very high noise
    metrics_noisy = StabilityMetrics(noisy)
    print(f"   Signal-to-noise ratio: {metrics_noisy.signal_to_noise_ratio():.3f}")
    print(f"   Mean variance: {np.mean(metrics_noisy.running_variance()):.2f}")
    print(f"   Stability index: {metrics_noisy.stability_index():.3f}")


def example_stability_over_time():
    """Example: Analyze how stability changes during training."""
    print("\n" + "=" * 80)
    print("Example 5: Stability Evolution During Training")
    print("=" * 80)

    # Simulate a training run that becomes more stable over time
    n_evals = 100

    # Early training: unstable
    early = np.linspace(-100, 0, 30) + np.random.normal(0, 30, 30)

    # Mid training: stabilizing
    mid = np.linspace(0, 80, 40) + np.random.normal(0, 15, 40)

    # Late training: stable
    late = np.linspace(80, 120, 30) + np.random.normal(0, 5, 30)

    full_training = np.concatenate([early, mid, late])

    # Analyze different phases
    print("\n1. Early Training (evaluations 0-30):")
    early_metrics = StabilityMetrics(early)
    print(f"   Stability index: {early_metrics.stability_index():.3f}")
    print(f"   CV: {np.mean(early_metrics.coefficient_of_variation_over_time()):.3f}")

    print("\n2. Mid Training (evaluations 30-70):")
    mid_metrics = StabilityMetrics(mid)
    print(f"   Stability index: {mid_metrics.stability_index():.3f}")
    print(f"   CV: {np.mean(mid_metrics.coefficient_of_variation_over_time()):.3f}")

    print("\n3. Late Training (evaluations 70-100):")
    late_metrics = StabilityMetrics(late)
    print(f"   Stability index: {late_metrics.stability_index():.3f}")
    print(f"   CV: {np.mean(late_metrics.coefficient_of_variation_over_time()):.3f}")

    print("\n4. Full Training:")
    full_metrics = StabilityMetrics(full_training)
    print(f"   Overall stability index: {full_metrics.stability_index():.3f}")
    print(f"   Monotonicity: {full_metrics.monotonicity_score():.3f}")

    # Running variance analysis
    running_var = full_metrics.running_variance(window=10)
    print(f"\n5. Variance Evolution:")
    print(f"   Early variance (mean): {np.mean(running_var[:20]):.2f}")
    print(f"   Late variance (mean): {np.mean(running_var[-20:]):.2f}")
    print(f"   Variance reduction: {(1 - np.mean(running_var[-20:]) / np.mean(running_var[:20])) * 100:.1f}%")


def example_hyperparameter_stability():
    """Example: Compare stability across hyperparameter settings."""
    print("\n" + "=" * 80)
    print("Example 6: Hyperparameter Impact on Stability")
    print("=" * 80)

    # Simulate different learning rates
    n_evals = 80
    n_seeds = 3
    np.random.seed(42)

    hyperparams = {}

    # Low learning rate: stable but slow
    low_lr = []
    for _ in range(n_seeds):
        rewards = 100 * (1 - np.exp(-np.arange(n_evals) / 40))  # Slow learning
        rewards += np.random.normal(0, 5, n_evals)  # Low noise
        low_lr.append(rewards)
    hyperparams['LR=0.0001 (Stable)'] = np.array(low_lr)

    # Medium learning rate: good balance
    med_lr = []
    for _ in range(n_seeds):
        rewards = 100 * (1 - np.exp(-np.arange(n_evals) / 20))  # Medium speed
        rewards += np.random.normal(0, 10, n_evals)  # Medium noise
        med_lr.append(rewards)
    hyperparams['LR=0.001 (Balanced)'] = np.array(med_lr)

    # High learning rate: fast but unstable
    high_lr = []
    for seed in range(n_seeds):
        rewards = 100 * (1 - np.exp(-np.arange(n_evals) / 10))  # Fast learning
        rewards += np.random.normal(0, 25, n_evals)  # High noise
        # Add oscillation
        rewards += 10 * np.sin(np.arange(n_evals) * 0.5)
        high_lr.append(rewards)
    hyperparams['LR=0.01 (Unstable)'] = np.array(high_lr)

    print("\nStability Analysis by Learning Rate:")
    for name, rewards in hyperparams.items():
        metrics = StabilityMetrics(rewards)
        summary = metrics.get_summary()

        print(f"\n{name}:")
        print(f"  Stability index: {summary['stability_index']:.3f}")
        print(f"  Oscillation: {summary['oscillation_metric']:.3f}")
        print(f"  Smoothness: {summary['smoothness_score']:.2f}")
        print(f"  Final performance: {np.mean(rewards[:, -10:]):.2f}")

    print("\n→ Trade-off: Higher learning rates learn faster but are less stable")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("STABILITY METRICS - USAGE EXAMPLES")
    print("=" * 80)

    # Run all examples
    example_single_seed_stability()
    example_multi_seed_stability()
    example_algorithm_comparison()
    example_detect_training_issues()
    example_stability_over_time()
    example_hyperparameter_stability()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()