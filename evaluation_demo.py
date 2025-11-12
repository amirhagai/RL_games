"""
Comprehensive RL Evaluation Demo

This script demonstrates how to use the complete evaluation suite to analyze
and compare RL algorithms following best practices from the literature.
"""

import numpy as np
import json
from pathlib import Path
import sys

# Import the evaluation package
from evaluation import (
    RLEvaluator,
    SampleEfficiencyMetrics,
    FinalPerformanceMetrics,
    StabilityMetrics,
    InterquartileMean,
    quick_evaluate
)


def simulate_algorithm_training(algo_type='stable', n_seeds=5, n_evaluations=100):
    """
    Simulate training curves for different algorithm types.

    Args:
        algo_type: 'stable', 'efficient', 'unstable', or 'random'
        n_seeds: Number of random seeds
        n_evaluations: Number of evaluation points

    Returns:
        Array of shape [n_seeds, n_evaluations]
    """
    np.random.seed(42)
    rewards = []

    for seed in range(n_seeds):
        if algo_type == 'stable':
            # Stable: Consistent improvement with low variance
            curve = 100 * (1 - np.exp(-np.arange(n_evaluations) / 20))
            noise = np.random.normal(0, 5, n_evaluations)
            seed_rewards = curve + noise

        elif algo_type == 'efficient':
            # Efficient: Fast initial learning
            curve = 100 * (1 - np.exp(-np.arange(n_evaluations) / 10))
            noise = np.random.normal(0, 8, n_evaluations)
            seed_rewards = curve + noise

        elif algo_type == 'unstable':
            # Unstable: High variance, occasional failures
            curve = 100 * (1 - np.exp(-np.arange(n_evaluations) / 25))
            noise = np.random.normal(0, 20, n_evaluations)
            seed_rewards = curve + noise

            # Add catastrophic failures
            if seed % 2 == 0:
                failure_point = np.random.randint(30, 70)
                seed_rewards[failure_point:failure_point+10] -= 50

        else:  # random
            # Random: No learning
            seed_rewards = np.random.normal(0, 20, n_evaluations)

        rewards.append(seed_rewards)

    return np.array(rewards)


def demo_single_algorithm():
    """Demo: Comprehensive evaluation of a single algorithm."""
    print("=" * 80)
    print("DEMO 1: Single Algorithm Comprehensive Evaluation")
    print("=" * 80)

    # Simulate training data
    rewards = simulate_algorithm_training('stable', n_seeds=10, n_evaluations=100)
    timesteps = np.arange(100) * 1000

    # Create evaluator
    evaluator = RLEvaluator(rewards, timesteps=timesteps)

    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation(target_reward=80, eval_window=10)

    # Display results
    print("\n1. SAMPLE EFFICIENCY METRICS:")
    print("-" * 40)
    efficiency = results['sample_efficiency']
    print(f"  Time to 90% of max: {efficiency['time_to_90_percent']} timesteps")
    print(f"  Convergence: {efficiency['convergence_timestep']} timesteps")
    print(f"  Jumpstart performance: {efficiency['jumpstart_performance']:.2f}")

    print("\n2. FINAL PERFORMANCE METRICS:")
    print("-" * 40)
    performance = results['final_performance']
    final = performance['final_performance']
    print(f"  Mean ± Std: {final['mean']:.2f} ± {final['std']:.2f}")
    ci = performance['confidence_interval_95']
    print(f"  95% CI: [{ci['lower']:.2f}, {ci['upper']:.2f}]")
    print(f"  Best seed: {performance['best_seed']['performance']:.2f}")
    print(f"  Worst seed: {performance['worst_seed']['performance']:.2f}")

    print("\n3. STABILITY METRICS:")
    print("-" * 40)
    stability = results['stability']
    print(f"  Stability index: {stability['stability_index']:.3f}")
    print(f"  Monotonicity: {stability['monotonicity_score']:.3f}")
    print(f"  Catastrophic failures: {stability['catastrophic_failures']['count']}")
    print(f"  Signal-to-noise ratio: {stability['signal_to_noise_ratio']:.2f}")

    print("\n4. ROBUST STATISTICS (IQM):")
    print("-" * 40)
    robust = results['robust_statistics']['robust_statistics']
    print(f"  IQM: {robust['iqm']:.2f}")
    print(f"  Mean: {robust['mean']:.2f}")
    print(f"  Median: {robust['median']:.2f}")
    print(f"  Outliers: {robust['n_outliers']} ({robust['outlier_percentage']:.1f}%)")


def demo_algorithm_comparison():
    """Demo: Compare multiple algorithms."""
    print("\n" + "=" * 80)
    print("DEMO 2: Multi-Algorithm Comparison")
    print("=" * 80)

    # Simulate different algorithms
    algorithms = {
        'PPO (Stable)': simulate_algorithm_training('stable', n_seeds=10),
        'SAC (Efficient)': simulate_algorithm_training('efficient', n_seeds=10),
        'DQN (Unstable)': simulate_algorithm_training('unstable', n_seeds=10),
        'Random': simulate_algorithm_training('random', n_seeds=10)
    }

    # Create evaluator
    evaluator = RLEvaluator(algorithms)

    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation()

    # Display comparison results
    print("\n1. SAMPLE EFFICIENCY COMPARISON:")
    print("-" * 40)
    if 'rankings' in results['sample_efficiency']:
        rankings = results['sample_efficiency']['rankings']
        if 'by_final_performance' in rankings:
            print("  Rankings by final performance:")
            for i, algo in enumerate(rankings['by_final_performance'], 1):
                individual = results['sample_efficiency']['individual_metrics'][algo]
                print(f"    {i}. {algo}: {individual['final_reward']:.2f}")

    print("\n2. FINAL PERFORMANCE COMPARISON:")
    print("-" * 40)
    perf_results = results['final_performance']
    print("  Mean ± Std:")
    for algo, metrics in perf_results['individual_metrics'].items():
        final = metrics['final_performance']
        print(f"    {algo:20s}: {final['mean']:6.2f} ± {final['std']:5.2f}")

    print("\n3. STABILITY COMPARISON:")
    print("-" * 40)
    stability_results = results['stability']
    print("  Stability indices:")
    for algo, metrics in stability_results['individual_metrics'].items():
        print(f"    {algo:20s}: {metrics['stability_index']:.3f}")

    print("\n4. IQM COMPARISON (Robust to outliers):")
    print("-" * 40)
    iqm_results = results['robust_statistics']
    print("  IQM vs Mean:")
    for algo, stats in iqm_results['individual_stats'].items():
        print(f"    {algo:20s}: IQM={stats['iqm']:6.2f}, Mean={stats['mean']:6.2f}")

    print("\n5. OVERALL RANKINGS:")
    print("-" * 40)
    if 'overall_rankings' in results:
        overall = results['overall_rankings']
        print("  Aggregate ranking (best to worst):")
        for i, algo in enumerate(overall['aggregate'], 1):
            score = overall['scores'][algo]
            print(f"    {i}. {algo:20s} (score: {score})")


def demo_multi_environment():
    """Demo: Evaluate across multiple environments."""
    print("\n" + "=" * 80)
    print("DEMO 3: Multi-Environment Evaluation")
    print("=" * 80)

    # Simulate performance across different environments
    environments = ['CartPole', 'Acrobot', 'MountainCar', 'LunarLander']
    algorithm_names = ['PPO', 'A2C', 'DQN']

    all_results = {}

    for env in environments:
        print(f"\n{env}:")
        print("-" * 40)

        # Simulate environment-specific performance
        env_results = {}
        for algo in algorithm_names:
            # Different scales for different environments
            if env == 'CartPole':
                base = 400 if algo == 'PPO' else 350
            elif env == 'Acrobot':
                base = -100 if algo == 'PPO' else -120
            elif env == 'MountainCar':
                base = -120 if algo == 'PPO' else -140
            else:  # LunarLander
                base = 200 if algo == 'PPO' else 150

            # Generate rewards
            rewards = simulate_algorithm_training('stable', n_seeds=5, n_evaluations=50)
            rewards = rewards + base  # Shift to environment-specific range

            env_results[algo] = rewards

        # Evaluate this environment
        evaluator = RLEvaluator(env_results)
        env_eval = evaluator.evaluate_with_iqm()

        # Display environment-specific results
        for algo, stats in env_eval['individual_stats'].items():
            print(f"  {algo}: IQM={stats['iqm']:.2f}, Mean={stats['mean']:.2f}")

        all_results[env] = env_results

    # Aggregate across environments
    print("\n" + "=" * 40)
    print("AGGREGATED RESULTS ACROSS ALL ENVIRONMENTS:")
    print("=" * 40)

    # Calculate normalized scores for each algorithm across environments
    algo_scores = {algo: [] for algo in algorithm_names}

    for env, env_results in all_results.items():
        # Normalize within environment
        all_values = np.concatenate([r.flatten() for r in env_results.values()])
        env_min, env_max = np.min(all_values), np.max(all_values)

        for algo, rewards in env_results.items():
            iqm_calc = InterquartileMean(rewards)
            iqm = iqm_calc.compute_iqm()
            normalized = (iqm - env_min) / (env_max - env_min + 1e-8)
            algo_scores[algo].append(normalized)

    # Display aggregate scores
    print("\nNormalized IQM scores (0-1) averaged across environments:")
    for algo, scores in algo_scores.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"  {algo}: {mean_score:.3f} ± {std_score:.3f}")


def demo_quick_evaluation():
    """Demo: Quick evaluation functions."""
    print("\n" + "=" * 80)
    print("DEMO 4: Quick Evaluation Functions")
    print("=" * 80)

    # Generate sample data
    rewards = simulate_algorithm_training('efficient', n_seeds=1, n_evaluations=50)[0]

    print("\n1. Quick efficiency check:")
    efficiency = quick_evaluate(rewards, metric='efficiency')
    print(f"   Time to 90%: {efficiency['time_to_90_percent']}")
    print(f"   Final reward: {efficiency['final_reward']:.2f}")

    print("\n2. Quick stability check:")
    stability = quick_evaluate(rewards, metric='stability')
    print(f"   Stability index: {stability['stability_index']:.3f}")
    print(f"   Monotonicity: {stability['monotonicity_score']:.3f}")

    print("\n3. Quick IQM check:")
    iqm_result = quick_evaluate(rewards, metric='iqm')
    print(f"   IQM: {iqm_result['robust_statistics']['iqm']:.2f}")
    print(f"   Outliers: {iqm_result['outlier_analysis']['n_outliers']}")


def demo_report_generation():
    """Demo: Automatic report generation."""
    print("\n" + "=" * 80)
    print("DEMO 5: Automatic Report Generation")
    print("=" * 80)

    # Create sample data for multiple algorithms
    algorithms = {
        'Algorithm_A': simulate_algorithm_training('stable', n_seeds=5),
        'Algorithm_B': simulate_algorithm_training('efficient', n_seeds=5),
        'Algorithm_C': simulate_algorithm_training('unstable', n_seeds=5)
    }

    # Create evaluator and generate report
    evaluator = RLEvaluator(algorithms)

    # Generate and save report
    report_path = 'evaluation_report.txt'
    report = evaluator.generate_report(report_path)

    print(f"\nGenerated comprehensive report:")
    print("-" * 40)
    print(report)
    print("-" * 40)
    print(f"\n✓ Report saved to: {report_path}")


def demo_real_benchmark_integration():
    """Demo: How to integrate with the benchmark system."""
    print("\n" + "=" * 80)
    print("DEMO 6: Integration with Benchmark System")
    print("=" * 80)

    print("\nTo integrate with your benchmark.py results:")
    print("-" * 40)

    print("""
    # Load results from benchmark run
    from evaluation import RLEvaluator, load_benchmark_results

    # Load benchmark results
    results_dir = 'results/20240101_comprehensive_benchmark'
    algorithm_rewards, timesteps = load_benchmark_results(results_dir)

    # Create evaluator
    evaluator = RLEvaluator(algorithm_rewards, timesteps=timesteps)

    # Run evaluation
    eval_results = evaluator.comprehensive_evaluation()

    # Generate report
    report = evaluator.generate_report('benchmark_evaluation.txt')

    # Access specific metrics
    iqm_rankings = eval_results['robust_statistics']['rankings']['by_iqm']
    stability_scores = eval_results['stability']['individual_metrics']
    """)

    print("\nExample with simulated benchmark data:")

    # Simulate loading benchmark results
    benchmark_results = {
        'PPO/CartPole-v1': np.random.normal(450, 30, (5, 50)),
        'PPO/Acrobot-v1': np.random.normal(-100, 20, (5, 50)),
        'A2C/CartPole-v1': np.random.normal(400, 40, (5, 50)),
        'A2C/Acrobot-v1': np.random.normal(-120, 25, (5, 50))
    }

    # Quick evaluation
    for exp_name, rewards in benchmark_results.items():
        iqm_calc = InterquartileMean(rewards)
        iqm = iqm_calc.compute_iqm()
        mean = np.mean(rewards)
        print(f"  {exp_name:20s}: IQM={iqm:6.2f}, Mean={mean:6.2f}")


def main():
    """Run all demos."""
    print("\n" + "=" * 80)
    print("RL EVALUATION METRICS - COMPREHENSIVE DEMO")
    print("=" * 80)
    print("\nThis demo shows how to use the evaluation suite for")
    print("comprehensive RL algorithm analysis following best practices.")

    # Run all demos
    demo_single_algorithm()
    demo_algorithm_comparison()
    demo_multi_environment()
    demo_quick_evaluation()
    demo_report_generation()
    demo_real_benchmark_integration()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("• Use IQM for robust evaluation (Agarwal et al. 2021)")
    print("• Consider sample efficiency, not just final performance")
    print("• Analyze stability to detect training issues")
    print("• Always use multiple seeds for statistical validity")
    print("• Report confidence intervals, not just means")
    print("\nFor more details, see the RL_EVALUATION_GUIDE.md")


if __name__ == "__main__":
    main()