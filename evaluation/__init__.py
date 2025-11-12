"""
RL Evaluation Metrics Package

A comprehensive suite of evaluation metrics for reinforcement learning algorithms,
implementing best practices from the literature.

References:
- Agarwal et al. (2021): "Deep RL at the Edge of the Statistical Precipice"
- Henderson et al. (2018): "Deep Reinforcement Learning that Matters"
- Machado et al. (2018): "Revisiting the Arcade Learning Environment"
"""

from .sample_efficiency import (
    SampleEfficiencyMetrics,
    compare_sample_efficiency,
    load_benchmark_results
)

from .final_performance import (
    FinalPerformanceMetrics,
    compare_final_performance,
    aggregate_across_environments
)

from .stability_metrics import (
    StabilityMetrics,
    compare_stability
)

from .iqm import (
    InterquartileMean,
    stratified_iqm
)

__version__ = "1.0.0"

__all__ = [
    # Sample Efficiency
    'SampleEfficiencyMetrics',
    'compare_sample_efficiency',
    'load_benchmark_results',

    # Final Performance
    'FinalPerformanceMetrics',
    'compare_final_performance',
    'aggregate_across_environments',

    # Stability
    'StabilityMetrics',
    'compare_stability',

    # IQM and Robust Statistics
    'InterquartileMean',
    'stratified_iqm',

    # Main Evaluator Class
    'RLEvaluator'
]


class RLEvaluator:
    """
    Master class for comprehensive RL algorithm evaluation.

    Combines all metrics for a complete evaluation pipeline.
    """

    def __init__(self, reward_histories, algorithm_names=None, timesteps=None):
        """
        Initialize the evaluator with reward histories.

        Args:
            reward_histories: Either:
                - Single array of shape [n_evaluations] or [n_seeds, n_evaluations]
                - Dict mapping algorithm names to reward arrays
            algorithm_names: Optional list of algorithm names (if reward_histories is a list)
            timesteps: Optional array of timesteps when evaluations occurred
        """
        # Handle different input formats
        if isinstance(reward_histories, dict):
            self.algorithms = reward_histories
            self.multi_algorithm = True
        elif algorithm_names is not None:
            self.algorithms = dict(zip(algorithm_names, reward_histories))
            self.multi_algorithm = True
        else:
            self.algorithms = {'algorithm': reward_histories}
            self.multi_algorithm = False

        self.timesteps = timesteps

    def evaluate_sample_efficiency(self, target_reward=None):
        """
        Evaluate sample efficiency for all algorithms.

        Args:
            target_reward: Optional target reward for efficiency calculation

        Returns:
            Dict with sample efficiency metrics for each algorithm
        """
        results = {}

        # Create averaged version for compare_sample_efficiency
        averaged_algorithms = {}
        for name, rewards in self.algorithms.items():
            # Handle multi-seed data
            if len(rewards.shape) == 2:
                # Average across seeds for sample efficiency
                avg_rewards = rewards.mean(axis=0)
            else:
                avg_rewards = rewards

            averaged_algorithms[name] = avg_rewards
            metrics = SampleEfficiencyMetrics(avg_rewards, self.timesteps)
            results[name] = metrics.get_summary()

            if target_reward is not None:
                results[name]['efficiency_score'] = metrics.sample_efficiency_score(target_reward)

        if self.multi_algorithm:
            return compare_sample_efficiency(averaged_algorithms, self.timesteps)
        else:
            return results['algorithm']

    def evaluate_final_performance(self, eval_window=10):
        """
        Evaluate final/asymptotic performance.

        Args:
            eval_window: Number of final evaluations to consider

        Returns:
            Dict with final performance metrics
        """
        results = {}

        for name, rewards in self.algorithms.items():
            metrics = FinalPerformanceMetrics(rewards, eval_window)
            results[name] = metrics.get_summary()

        if self.multi_algorithm:
            return compare_final_performance(self.algorithms)
        else:
            return results['algorithm']

    def evaluate_stability(self):
        """
        Evaluate training stability.

        Returns:
            Dict with stability metrics
        """
        results = {}

        for name, rewards in self.algorithms.items():
            metrics = StabilityMetrics(rewards, self.timesteps)
            results[name] = metrics.get_summary()

        if self.multi_algorithm:
            return compare_stability(self.algorithms)
        else:
            return results['algorithm']

    def evaluate_with_iqm(self):
        """
        Evaluate using IQM and robust statistics.

        Returns:
            Dict with IQM-based metrics
        """
        if self.multi_algorithm:
            iqm_calc = InterquartileMean(self.algorithms)
            return iqm_calc.compare_algorithms_iqm()
        else:
            rewards = list(self.algorithms.values())[0]
            iqm_calc = InterquartileMean(rewards)
            return iqm_calc.get_summary()

    def comprehensive_evaluation(self, target_reward=None, eval_window=10):
        """
        Run all evaluation metrics.

        Args:
            target_reward: Target reward for efficiency calculation
            eval_window: Window for final performance calculation

        Returns:
            Comprehensive evaluation results
        """
        results = {
            'sample_efficiency': self.evaluate_sample_efficiency(target_reward),
            'final_performance': self.evaluate_final_performance(eval_window),
            'stability': self.evaluate_stability(),
            'robust_statistics': self.evaluate_with_iqm()
        }

        # Add overall rankings if multiple algorithms
        if self.multi_algorithm:
            results['overall_rankings'] = self._compute_overall_rankings(results)

        return results

    def _compute_overall_rankings(self, results):
        """
        Compute overall rankings across all metrics.

        Args:
            results: Dict with all evaluation results

        Returns:
            Overall rankings dict
        """
        rankings = {}
        scores = {algo: 0 for algo in self.algorithms.keys()}

        # Aggregate rankings from different metrics
        ranking_sources = [
            ('sample_efficiency', 'rankings', 'by_final_performance'),
            ('final_performance', 'rankings', 'by_mean'),
            ('stability', 'rankings', 'by_stability'),
            ('robust_statistics', 'rankings', 'by_iqm')
        ]

        for source, key1, key2 in ranking_sources:
            if source in results and key1 in results[source]:
                if key2 in results[source][key1]:
                    ranking_list = results[source][key1][key2]
                    for i, algo in enumerate(ranking_list):
                        scores[algo] += len(ranking_list) - i

        # Sort by aggregate score
        sorted_algos = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings['aggregate'] = [algo for algo, _ in sorted_algos]
        rankings['scores'] = scores

        return rankings

    def generate_report(self, output_file=None):
        """
        Generate a comprehensive evaluation report.

        Args:
            output_file: Optional file path to save the report

        Returns:
            Report string
        """
        eval_results = self.comprehensive_evaluation()

        report = []
        report.append("=" * 80)
        report.append("RL ALGORITHM EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Sample Efficiency
        if 'sample_efficiency' in eval_results:
            report.append("SAMPLE EFFICIENCY")
            report.append("-" * 40)

            if self.multi_algorithm:
                rankings = eval_results['sample_efficiency'].get('rankings', {})
                if 'by_final_performance' in rankings:
                    report.append("Rankings by final performance:")
                    for i, algo in enumerate(rankings['by_final_performance'], 1):
                        report.append(f"  {i}. {algo}")
            report.append("")

        # Final Performance
        if 'final_performance' in eval_results:
            report.append("FINAL PERFORMANCE")
            report.append("-" * 40)

            if self.multi_algorithm:
                for algo, metrics in eval_results['final_performance']['individual_metrics'].items():
                    perf = metrics['final_performance']
                    report.append(f"{algo}:")
                    report.append(f"  Mean: {perf['mean']:.2f} ± {perf['std']:.2f}")
            report.append("")

        # Stability
        if 'stability' in eval_results:
            report.append("STABILITY ANALYSIS")
            report.append("-" * 40)

            if self.multi_algorithm:
                for algo, metrics in eval_results['stability']['individual_metrics'].items():
                    report.append(f"{algo}:")
                    report.append(f"  Stability Index: {metrics['stability_index']:.3f}")
                    report.append(f"  Catastrophic Failures: {metrics['catastrophic_failures']['count']}")
            report.append("")

        # Robust Statistics (IQM)
        if 'robust_statistics' in eval_results:
            report.append("ROBUST STATISTICS (IQM)")
            report.append("-" * 40)

            if self.multi_algorithm:
                for algo, stats in eval_results['robust_statistics']['individual_stats'].items():
                    report.append(f"{algo}:")
                    report.append(f"  IQM: {stats['iqm']:.2f}")
                    report.append(f"  Mean: {stats['mean']:.2f}")
                    report.append(f"  Outliers: {stats['n_outliers']}")
            report.append("")

        # Overall Rankings
        if self.multi_algorithm and 'overall_rankings' in eval_results:
            report.append("OVERALL RANKINGS")
            report.append("-" * 40)
            for i, algo in enumerate(eval_results['overall_rankings']['aggregate'], 1):
                score = eval_results['overall_rankings']['scores'][algo]
                report.append(f"{i}. {algo} (score: {score})")

        report_str = "\n".join(report)

        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_str)

        return report_str


# Convenience functions for quick evaluation
def quick_evaluate(reward_history, metric='all'):
    """
    Quick evaluation with a single metric.

    Args:
        reward_history: Array of rewards
        metric: Which metric to compute ('efficiency', 'performance', 'stability', 'iqm', 'all')

    Returns:
        Evaluation results
    """
    evaluator = RLEvaluator(reward_history)

    if metric == 'efficiency':
        return evaluator.evaluate_sample_efficiency()
    elif metric == 'performance':
        return evaluator.evaluate_final_performance()
    elif metric == 'stability':
        return evaluator.evaluate_stability()
    elif metric == 'iqm':
        return evaluator.evaluate_with_iqm()
    elif metric == 'all':
        return evaluator.comprehensive_evaluation()
    else:
        raise ValueError(f"Unknown metric: {metric}")