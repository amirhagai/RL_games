# Comprehensive Guide to RL Algorithm Evaluation

## Table of Contents
1. [Introduction](#introduction)
2. [Core Evaluation Metrics](#core-evaluation-metrics)
3. [Statistical Rigor and Reproducibility](#statistical-rigor-and-reproducibility)
4. [Evaluation Protocols](#evaluation-protocols)
5. [Comparison Methods](#comparison-methods)
6. [Visualization Standards](#visualization-standards)
7. [Implementation Examples](#implementation-examples)
8. [Literature References](#literature-references)

---

## Introduction

Evaluating RL algorithms properly is crucial for scientific validity and practical deployment. This guide synthesizes best practices from major RL papers and benchmarking studies.

### Key Principles (Henderson et al., 2018)
1. **Reproducibility**: Results must be reproducible with different random seeds
2. **Statistical Significance**: Claims require proper statistical testing
3. **Fair Comparison**: Same computational budget, hyperparameters tuning effort
4. **Comprehensive Reporting**: Report all metrics, not just favorable ones

---

## Core Evaluation Metrics

### 1. Sample Efficiency
**Definition**: How quickly an algorithm learns (data required to reach performance threshold)

```python
def measure_sample_efficiency(rewards, threshold=100, fraction=0.9):
    """
    Measure timesteps to reach X% of maximum performance.
    Based on Duan et al. (2016) benchmarking methodology.
    """
    max_reward = np.max(rewards)
    target = max_reward * fraction

    for t, reward in enumerate(rewards):
        if reward >= target:
            return t
    return len(rewards)  # Never reached
```

**Literature Standard**: Report timesteps to reach 90% of maximum observed performance

### 2. Final Performance
**Definition**: Average reward over last N episodes after convergence

```python
def measure_final_performance(rewards, last_n=100):
    """
    Standard: Average of last 100 episodes (Mnih et al., 2015)
    Alternative: Average of best 10% of episodes (Schulman et al., 2017)
    """
    if len(rewards) < last_n:
        return np.mean(rewards)
    return np.mean(rewards[-last_n:])
```

### 3. Stability Metrics
**Definition**: Consistency of learning across runs

```python
def measure_stability(all_runs_rewards):
    """
    Stability metrics from Cobbe et al. (2019)
    """
    # Inter-run variance
    final_rewards = [measure_final_performance(run) for run in all_runs_rewards]

    stability_metrics = {
        'mean': np.mean(final_rewards),
        'std': np.std(final_rewards),
        'min': np.min(final_rewards),
        'max': np.max(final_rewards),
        'coefficient_of_variation': np.std(final_rewards) / np.mean(final_rewards),
        'interquartile_range': np.percentile(final_rewards, 75) - np.percentile(final_rewards, 25)
    }
    return stability_metrics
```

### 4. Regret
**Definition**: Cumulative difference from optimal policy

```python
def calculate_regret(rewards, optimal_reward):
    """
    Cumulative regret (Lattimore & Szepesvári, 2020)
    Important for exploration-exploitation analysis
    """
    regret = []
    cumulative = 0
    for r in rewards:
        cumulative += (optimal_reward - r)
        regret.append(cumulative)
    return regret
```

### 5. Area Under Curve (AUC)
**Definition**: Total reward accumulated during training

```python
def calculate_auc(rewards, normalize=True):
    """
    Area under learning curve (Agarwal et al., 2021)
    Better captures overall learning efficiency
    """
    auc = np.trapz(rewards)

    if normalize:
        # Normalize by maximum possible area
        max_reward = np.max(rewards)
        max_auc = max_reward * len(rewards)
        auc = auc / max_auc

    return auc
```

---

## Statistical Rigor and Reproducibility

### 1. Multiple Random Seeds
**Literature Standard**: Minimum 5 seeds, preferably 10+ (Henderson et al., 2018)

```python
class StatisticalEvaluator:
    """
    Implements statistical best practices from RL literature.
    """

    def __init__(self, min_seeds=5, confidence_level=0.95):
        self.min_seeds = min_seeds
        self.confidence_level = confidence_level

    def evaluate_with_confidence(self, algorithm, env_id, seeds=None):
        """
        Standard evaluation with confidence intervals.
        Based on Machado et al. (2018) arcade learning environment paper.
        """
        if seeds is None:
            seeds = [42 + i * 1000 for i in range(self.min_seeds)]

        all_rewards = []

        for seed in seeds:
            rewards = self.run_experiment(algorithm, env_id, seed)
            all_rewards.append(rewards)

        # Calculate statistics
        final_performances = [rewards[-100:].mean() for rewards in all_rewards]

        # Bootstrap confidence interval (Efron & Tibshirani, 1993)
        ci_lower, ci_upper = self.bootstrap_ci(final_performances)

        # Inter-quartile mean (IQM) - robust to outliers (Agarwal et al., 2021)
        iqm = self.calculate_iqm(final_performances)

        return {
            'mean': np.mean(final_performances),
            'std': np.std(final_performances),
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'iqm': iqm,
            'all_seeds': final_performances
        }

    def bootstrap_ci(self, data, n_bootstrap=10000):
        """
        Bootstrap confidence intervals (recommended by Agarwal et al., 2021)
        """
        bootstrap_means = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))

        return ci_lower, ci_upper

    def calculate_iqm(self, data):
        """
        Inter-quartile mean: Mean of middle 50% of data.
        More robust than mean, more stable than median.
        """
        sorted_data = np.sort(data)
        q1_idx = len(sorted_data) // 4
        q3_idx = 3 * len(sorted_data) // 4
        return np.mean(sorted_data[q1_idx:q3_idx])
```

### 2. Statistical Tests
**Literature Standards for Comparing Algorithms**

```python
from scipy import stats

def compare_algorithms(results_a, results_b):
    """
    Statistical comparison following Colas et al. (2018) guidelines.
    """
    comparisons = {}

    # 1. Welch's t-test (assumes unequal variances)
    t_stat, p_value = stats.ttest_ind(results_a, results_b, equal_var=False)
    comparisons['welch_t_test'] = {'statistic': t_stat, 'p_value': p_value}

    # 2. Mann-Whitney U test (non-parametric, recommended by Henderson et al.)
    u_stat, p_value = stats.mannwhitneyu(results_a, results_b, alternative='two-sided')
    comparisons['mann_whitney'] = {'statistic': u_stat, 'p_value': p_value}

    # 3. Bootstrap test (Agarwal et al., 2021 recommendation)
    bootstrap_diff = bootstrap_difference_test(results_a, results_b)
    comparisons['bootstrap'] = bootstrap_diff

    # 4. Effect size (Cohen's d)
    cohens_d = (np.mean(results_a) - np.mean(results_b)) / np.sqrt((np.std(results_a)**2 + np.std(results_b)**2) / 2)
    comparisons['effect_size'] = cohens_d

    return comparisons

def bootstrap_difference_test(results_a, results_b, n_bootstrap=10000):
    """
    Bootstrap test for difference in means.
    """
    observed_diff = np.mean(results_a) - np.mean(results_b)

    # Combine and resample
    combined = np.concatenate([results_a, results_b])
    n_a = len(results_a)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        shuffled = np.random.permutation(combined)
        sample_a = shuffled[:n_a]
        sample_b = shuffled[n_a:]
        bootstrap_diffs.append(np.mean(sample_a) - np.mean(sample_b))

    # Calculate p-value
    p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))

    return {
        'observed_difference': observed_diff,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

---

## Evaluation Protocols

### 1. Training vs Evaluation Mode
**Literature Standard**: Always evaluate with deterministic policy

```python
class EvaluationProtocol:
    """
    Implements standard evaluation protocols from literature.
    """

    def __init__(self, eval_episodes=100, eval_freq=10000):
        """
        Standards:
        - eval_episodes: 100 (OpenAI Gym), 30 (Atari), 10 (MuJoCo)
        - eval_freq: Every 10k-100k steps depending on environment
        """
        self.eval_episodes = eval_episodes
        self.eval_freq = eval_freq

    def periodic_evaluation(self, agent, env, total_steps):
        """
        Periodic evaluation during training (Duan et al., 2016).
        """
        evaluations = []

        for step in range(0, total_steps, self.eval_freq):
            # Train for eval_freq steps
            agent.train(self.eval_freq)

            # Evaluate with deterministic policy
            eval_rewards = self.evaluate_agent(agent, env, deterministic=True)

            evaluations.append({
                'step': step,
                'mean_reward': np.mean(eval_rewards),
                'std_reward': np.std(eval_rewards),
                'min_reward': np.min(eval_rewards),
                'max_reward': np.max(eval_rewards)
            })

        return evaluations

    def evaluate_agent(self, agent, env, deterministic=True, render=False):
        """
        Standard evaluation procedure.
        """
        episode_rewards = []

        for episode in range(self.eval_episodes):
            obs = env.reset(seed=42 + episode)  # Fixed seeds for evaluation
            done = False
            episode_reward = 0

            while not done:
                action = agent.act(obs, deterministic=deterministic)
                obs, reward, done, info = env.step(action)
                episode_reward += reward

                if render:
                    env.render()

            episode_rewards.append(episode_reward)

        return episode_rewards
```

### 2. Environment-Specific Standards

```python
# From literature: Different environments need different evaluation settings

EVALUATION_STANDARDS = {
    # Atari (Machado et al., 2018)
    'atari': {
        'eval_episodes': 30,
        'sticky_actions': 0.25,  # Stochasticity
        'no_op_max': 30,  # Random no-ops at start
        'frame_skip': 4,
        'deterministic': True
    },

    # MuJoCo (Duan et al., 2016)
    'mujoco': {
        'eval_episodes': 10,
        'deterministic': True,
        'normalize_obs': True,
        'normalize_reward': False
    },

    # Classic Control (Brockman et al., 2016)
    'classic_control': {
        'eval_episodes': 100,
        'deterministic': True,
        'success_threshold': {
            'CartPole-v1': 195.0,
            'Acrobot-v1': -100.0,
            'MountainCar-v0': -110.0
        }
    }
}
```

---

## Comparison Methods

### 1. Performance Profiles (Dolan & Moré, 2002)
Used in Agarwal et al. (2021) for robust algorithm comparison

```python
def compute_performance_profile(algorithm_results, tau_max=10.0, n_points=100):
    """
    Performance profiles show probability of achieving within τ of best.
    More robust than point estimates.
    """
    # algorithm_results: dict of {algorithm_name: {env_name: score}}

    profiles = {}
    taus = np.linspace(1.0, tau_max, n_points)

    # For each algorithm
    for algo_name in algorithm_results:
        profile = []

        for tau in taus:
            # Count environments where algo is within tau of best
            count = 0
            total = 0

            for env in algorithm_results[algo_name]:
                # Get best score for this environment
                best_score = max(algorithm_results[a][env]
                               for a in algorithm_results
                               if env in algorithm_results[a])

                algo_score = algorithm_results[algo_name][env]

                # Check if within tau factor of best
                if algo_score >= best_score / tau:
                    count += 1
                total += 1

            profile.append(count / total)

        profiles[algo_name] = profile

    return taus, profiles
```

### 2. Aggregate Metrics (Agarwal et al., 2021)

```python
class AggregateMetrics:
    """
    Implements aggregate metrics from 'Deep RL at the Edge of the Statistical Precipice'.
    """

    @staticmethod
    def normalized_score(score, random_score, expert_score):
        """
        Normalize scores to [0, 1] range.
        Used in Atari and MuJoCo benchmarks.
        """
        return (score - random_score) / (expert_score - random_score)

    @staticmethod
    def optimality_gap(scores, optimal_score):
        """
        Mean optimality gap across tasks.
        """
        gaps = [(optimal_score - s) / optimal_score for s in scores]
        return np.mean(gaps)

    @staticmethod
    def probability_of_improvement(scores_a, scores_b, n_bootstrap=10000):
        """
        Probability that algorithm A is better than B.
        More interpretable than p-values.
        """
        improvements = 0

        for _ in range(n_bootstrap):
            # Bootstrap sample from both
            sample_a = np.random.choice(scores_a, size=len(scores_a), replace=True)
            sample_b = np.random.choice(scores_b, size=len(scores_b), replace=True)

            if np.mean(sample_a) > np.mean(sample_b):
                improvements += 1

        return improvements / n_bootstrap
```

### 3. Critical Difference Diagrams (Demšar, 2006)

```python
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon

def create_critical_difference_diagram(results_matrix):
    """
    Critical difference diagram for comparing multiple algorithms.
    results_matrix: algorithms x environments array

    Based on Demšar (2006) statistical comparison methodology.
    """
    n_algorithms, n_envs = results_matrix.shape

    # Friedman test
    stat, p_value = friedmanchisquare(*results_matrix)

    if p_value < 0.05:
        # Significant differences exist, do pairwise comparisons
        ranks = np.mean([stats.rankdata(-row) for row in results_matrix.T], axis=0)

        # Nemenyi post-hoc test
        cd = compute_critical_difference(n_algorithms, n_envs)

        # Plot
        plot_critical_difference(ranks, algorithm_names, cd)

    return ranks, p_value

def compute_critical_difference(n_algorithms, n_datasets, alpha=0.05):
    """
    Compute critical difference for Nemenyi test.
    """
    # Critical values from statistical tables
    q_alpha = {
        0.05: {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031},
        0.01: {2: 2.576, 3: 2.913, 4: 3.113, 5: 3.255, 6: 3.364, 7: 3.452, 8: 3.526}
    }

    q = q_alpha[alpha].get(n_algorithms, 3.0)  # Default if not in table
    cd = q * np.sqrt(n_algorithms * (n_algorithms + 1) / (6 * n_datasets))

    return cd
```

---

## Visualization Standards

### 1. Learning Curves with Confidence Intervals

```python
def plot_learning_curves(all_runs, algorithm_names, colors=None):
    """
    Standard learning curve visualization from literature.
    Shows mean ± confidence interval.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_runs)))

    for i, (algo_name, runs) in enumerate(all_runs.items()):
        # Convert to array for easier manipulation
        runs_array = np.array(runs)

        # Calculate statistics
        mean = np.mean(runs_array, axis=0)
        std = np.std(runs_array, axis=0)

        # Calculate confidence interval (95%)
        n = len(runs)
        ci = 1.96 * std / np.sqrt(n)

        # Alternative: Use IQM and percentiles (Agarwal et al., 2021)
        percentile_25 = np.percentile(runs_array, 25, axis=0)
        percentile_75 = np.percentile(runs_array, 75, axis=0)

        # Plot mean line
        steps = np.arange(len(mean))
        ax.plot(steps, mean, label=algo_name, color=colors[i], linewidth=2)

        # Add confidence band
        ax.fill_between(steps, mean - ci, mean + ci,
                        color=colors[i], alpha=0.2)

        # Alternative: Show IQR
        # ax.fill_between(steps, percentile_25, percentile_75,
        #                color=colors[i], alpha=0.2)

    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Learning Curves with 95% Confidence Intervals', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

### 2. Sample Efficiency Curves

```python
def plot_sample_efficiency(all_runs, thresholds=[0.5, 0.75, 0.9]):
    """
    Sample efficiency visualization showing time to reach performance thresholds.
    """
    fig, axes = plt.subplots(1, len(thresholds), figsize=(15, 5))

    for idx, threshold in enumerate(thresholds):
        ax = axes[idx]

        efficiency_data = {}
        for algo_name, runs in all_runs.items():
            steps_to_threshold = []

            for run in runs:
                max_reward = np.max(run)
                target = max_reward * threshold

                # Find first timestep reaching threshold
                for t, reward in enumerate(run):
                    if reward >= target:
                        steps_to_threshold.append(t)
                        break
                else:
                    steps_to_threshold.append(len(run))

            efficiency_data[algo_name] = steps_to_threshold

        # Box plot
        ax.boxplot(efficiency_data.values(), labels=efficiency_data.keys())
        ax.set_ylabel('Steps to Reach Threshold')
        ax.set_title(f'{int(threshold*100)}% of Max Performance')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Sample Efficiency Comparison', fontsize=14)
    plt.tight_layout()
    return fig
```

### 3. Performance Heatmap

```python
def plot_performance_heatmap(results_dict):
    """
    Heatmap showing algorithm performance across environments.
    Common in multi-task RL papers.
    """
    import seaborn as sns

    # Create matrix: algorithms x environments
    algorithms = list(results_dict.keys())
    environments = list(set(env for algo_results in results_dict.values()
                          for env in algo_results.keys()))

    matrix = np.zeros((len(algorithms), len(environments)))

    for i, algo in enumerate(algorithms):
        for j, env in enumerate(environments):
            if env in results_dict[algo]:
                matrix[i, j] = results_dict[algo][env]
            else:
                matrix[i, j] = np.nan

    # Normalize scores (0-1) per environment
    normalized_matrix = np.zeros_like(matrix)
    for j in range(len(environments)):
        col = matrix[:, j]
        if not np.all(np.isnan(col)):
            min_val = np.nanmin(col)
            max_val = np.nanmax(col)
            if max_val > min_val:
                normalized_matrix[:, j] = (col - min_val) / (max_val - min_val)

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(normalized_matrix,
                xticklabels=environments,
                yticklabels=algorithms,
                cmap='RdYlGn',
                center=0.5,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Normalized Score'})

    plt.title('Algorithm Performance Across Environments', fontsize=14)
    plt.xlabel('Environment', fontsize=12)
    plt.ylabel('Algorithm', fontsize=12)
    plt.tight_layout()

    return fig
```

---

## Implementation Examples

### Complete Evaluation Pipeline

```python
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

class RLEvaluationPipeline:
    """
    Complete evaluation pipeline following literature standards.
    """

    def __init__(self,
                 algorithms: List[str],
                 environments: List[str],
                 seeds: List[int] = None,
                 eval_episodes: int = 100,
                 eval_frequency: int = 10000):

        self.algorithms = algorithms
        self.environments = environments
        self.seeds = seeds or [0, 1337, 42, 2021, 9999]  # Standard seeds
        self.eval_episodes = eval_episodes
        self.eval_frequency = eval_frequency

        self.results = {}

    def run_complete_evaluation(self, total_timesteps: int = 1000000):
        """
        Run complete evaluation following best practices.
        """

        for algo in self.algorithms:
            self.results[algo] = {}

            for env in self.environments:
                print(f"Evaluating {algo} on {env}")

                # Run with multiple seeds
                seed_results = []
                for seed in self.seeds:
                    result = self.evaluate_single_run(
                        algo, env, seed, total_timesteps
                    )
                    seed_results.append(result)

                # Aggregate results
                self.results[algo][env] = self.aggregate_seed_results(seed_results)

        # Statistical comparisons
        self.statistical_analysis = self.run_statistical_comparisons()

        # Generate report
        self.generate_report()

        return self.results

    def evaluate_single_run(self, algorithm, env_id, seed, total_timesteps):
        """
        Single evaluation run with periodic evaluation.
        """
        # Initialize algorithm and environment
        env = gym.make(env_id)
        agent = self.create_agent(algorithm, env, seed)

        evaluations = []
        training_rewards = []

        for step in range(0, total_timesteps, self.eval_frequency):
            # Train
            train_rewards = agent.train(self.eval_frequency)
            training_rewards.extend(train_rewards)

            # Evaluate
            eval_rewards = []
            for _ in range(self.eval_episodes):
                reward = self.run_episode(agent, env, deterministic=True)
                eval_rewards.append(reward)

            evaluations.append({
                'step': step,
                'mean': np.mean(eval_rewards),
                'std': np.std(eval_rewards),
                'min': np.min(eval_rewards),
                'max': np.max(eval_rewards)
            })

        return {
            'seed': seed,
            'evaluations': evaluations,
            'training_curve': training_rewards,
            'final_performance': evaluations[-1]['mean'],
            'sample_efficiency': self.calculate_sample_efficiency(evaluations),
            'auc': np.trapz([e['mean'] for e in evaluations])
        }

    def aggregate_seed_results(self, seed_results):
        """
        Aggregate results across seeds with confidence intervals.
        """
        final_performances = [r['final_performance'] for r in seed_results]
        aucs = [r['auc'] for r in seed_results]
        sample_efficiencies = [r['sample_efficiency'] for r in seed_results]

        # Calculate IQM (Agarwal et al., 2021)
        iqm = self.calculate_iqm(final_performances)

        # Bootstrap confidence intervals
        ci_lower, ci_upper = self.bootstrap_ci(final_performances)

        return {
            'mean': np.mean(final_performances),
            'std': np.std(final_performances),
            'iqm': iqm,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'median': np.median(final_performances),
            'min': np.min(final_performances),
            'max': np.max(final_performances),
            'auc_mean': np.mean(aucs),
            'sample_efficiency_mean': np.mean(sample_efficiencies),
            'all_seeds': seed_results
        }

    def run_statistical_comparisons(self):
        """
        Run all statistical comparisons between algorithms.
        """
        comparisons = {}

        for i, algo1 in enumerate(self.algorithms):
            for algo2 in self.algorithms[i+1:]:
                key = f"{algo1}_vs_{algo2}"
                comparisons[key] = {}

                for env in self.environments:
                    if env in self.results[algo1] and env in self.results[algo2]:
                        scores1 = [r['final_performance']
                                  for r in self.results[algo1][env]['all_seeds']]
                        scores2 = [r['final_performance']
                                  for r in self.results[algo2][env]['all_seeds']]

                        # Multiple statistical tests
                        comparisons[key][env] = {
                            'welch_t': stats.ttest_ind(scores1, scores2, equal_var=False),
                            'mann_whitney': stats.mannwhitneyu(scores1, scores2),
                            'bootstrap_p': self.bootstrap_test(scores1, scores2),
                            'probability_of_improvement': self.probability_of_improvement(scores1, scores2),
                            'effect_size': self.cohens_d(scores1, scores2)
                        }

        return comparisons

    def generate_report(self, output_dir='evaluation_results'):
        """
        Generate comprehensive evaluation report.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save raw results
        with open(output_dir / 'raw_results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save statistical analysis
        with open(output_dir / 'statistical_analysis.json', 'w') as f:
            json.dump(self.statistical_analysis, f, indent=2, default=str)

        # Generate summary table
        summary = self.create_summary_table()
        summary.to_csv(output_dir / 'summary_table.csv')

        # Generate plots
        self.generate_all_plots(output_dir)

        # Generate LaTeX table for paper
        latex_table = self.create_latex_table()
        with open(output_dir / 'results_table.tex', 'w') as f:
            f.write(latex_table)

        print(f"Report saved to {output_dir}/")

    def create_summary_table(self):
        """
        Create summary table with key metrics.
        """
        import pandas as pd

        rows = []
        for algo in self.algorithms:
            for env in self.environments:
                if env in self.results[algo]:
                    result = self.results[algo][env]
                    rows.append({
                        'Algorithm': algo,
                        'Environment': env,
                        'Mean': f"{result['mean']:.2f}",
                        'Std': f"{result['std']:.2f}",
                        'IQM': f"{result['iqm']:.2f}",
                        'CI': f"[{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]",
                        'Median': f"{result['median']:.2f}",
                        'AUC': f"{result['auc_mean']:.0f}"
                    })

        return pd.DataFrame(rows)

    def create_latex_table(self):
        """
        Create publication-ready LaTeX table.
        """
        latex = "\\begin{table}[h]\n"
        latex += "\\centering\n"
        latex += "\\caption{Algorithm Performance Comparison}\n"
        latex += "\\begin{tabular}{l" + "c" * len(self.environments) + "}\n"
        latex += "\\toprule\n"
        latex += "Algorithm & " + " & ".join(self.environments) + " \\\\\n"
        latex += "\\midrule\n"

        for algo in self.algorithms:
            row = algo
            for env in self.environments:
                if env in self.results[algo]:
                    mean = self.results[algo][env]['mean']
                    ci_lower = self.results[algo][env]['ci_lower']
                    ci_upper = self.results[algo][env]['ci_upper']
                    row += f" & ${mean:.1f}_{{[{ci_lower:.1f}, {ci_upper:.1f}]}}$"
                else:
                    row += " & -"
            row += " \\\\\n"
            latex += row

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex

    # Helper methods
    def calculate_iqm(self, data):
        """Inter-quartile mean."""
        sorted_data = np.sort(data)
        q1_idx = len(sorted_data) // 4
        q3_idx = 3 * len(sorted_data) // 4
        return np.mean(sorted_data[q1_idx:q3_idx])

    def bootstrap_ci(self, data, n_bootstrap=10000, confidence=0.95):
        """Bootstrap confidence intervals."""
        bootstrap_means = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_means, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))

        return ci_lower, ci_upper

    def bootstrap_test(self, data1, data2, n_bootstrap=10000):
        """Bootstrap hypothesis test."""
        observed_diff = np.mean(data1) - np.mean(data2)
        combined = np.concatenate([data1, data2])
        n1 = len(data1)

        bootstrap_diffs = []
        for _ in range(n_bootstrap):
            shuffled = np.random.permutation(combined)
            sample1 = shuffled[:n1]
            sample2 = shuffled[n1:]
            bootstrap_diffs.append(np.mean(sample1) - np.mean(sample2))

        p_value = np.mean(np.abs(bootstrap_diffs) >= np.abs(observed_diff))
        return p_value

    def probability_of_improvement(self, scores1, scores2, n_bootstrap=10000):
        """Calculate probability that algorithm 1 is better than 2."""
        improvements = 0
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(scores1, size=len(scores1), replace=True)
            sample2 = np.random.choice(scores2, size=len(scores2), replace=True)
            if np.mean(sample1) > np.mean(sample2):
                improvements += 1
        return improvements / n_bootstrap

    def cohens_d(self, data1, data2):
        """Calculate Cohen's d effect size."""
        pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
        if pooled_std == 0:
            return 0
        return (np.mean(data1) - np.mean(data2)) / pooled_std

    def calculate_sample_efficiency(self, evaluations, threshold=0.9):
        """Calculate steps to reach 90% of max performance."""
        max_performance = max(e['mean'] for e in evaluations)
        target = max_performance * threshold

        for eval_point in evaluations:
            if eval_point['mean'] >= target:
                return eval_point['step']

        return evaluations[-1]['step']  # Never reached
```

### Usage Example with Your System

```python
# Integration with your existing benchmark.py

from RL_EVALUATION_GUIDE import RLEvaluationPipeline
from benchmark import BenchmarkRunner
from benchmark_config import BenchmarkConfig

# Configure comprehensive evaluation
config = BenchmarkConfig(
    mode='comprehensive',
    env_suite='publication',  # Standard benchmark environments
    algorithm='ppo',
    num_seeds=10,  # Literature standard: 10+ seeds
    total_timesteps=1000000,
    eval_freq=10000,
    eval_episodes=30
)

# Run with multiple algorithms for comparison
algorithms = ['ppo', 'a2c', 'sac']
environments = ['CartPole-v1', 'LunarLander-v3', 'Hopper-v5']

# Initialize evaluation pipeline
evaluator = RLEvaluationPipeline(
    algorithms=algorithms,
    environments=environments,
    seeds=[0, 1337, 42, 2021, 9999, 10000, 20000, 30000, 40000, 50000]  # 10 seeds
)

# Run complete evaluation
results = evaluator.run_complete_evaluation(total_timesteps=1000000)

# Results include:
# - Mean, std, IQM, confidence intervals
# - Statistical comparisons (t-test, Mann-Whitney, bootstrap)
# - Probability of improvement
# - Effect sizes
# - Performance profiles
# - Learning curves with confidence bands
# - LaTeX tables for publication

print("Evaluation complete! Check evaluation_results/ directory for full report.")
```

---

## Literature References

### Core Papers on RL Evaluation

1. **Henderson et al. (2018)** - "Deep Reinforcement Learning that Matters"
   - Established importance of seeds, hyperparameter reporting
   - Showed high variance in RL results
   - Guidelines for reproducible research

2. **Agarwal et al. (2021)** - "Deep Reinforcement Learning at the Edge of the Statistical Precipice"
   - Introduced Inter-quartile Mean (IQM)
   - Performance profiles for robust comparison
   - Bootstrap confidence intervals
   - Probability of improvement

3. **Machado et al. (2018)** - "Revisiting the Arcade Learning Environment"
   - Evaluation protocols for Atari
   - Sticky actions for stochasticity
   - Standardized evaluation settings

4. **Duan et al. (2016)** - "Benchmarking Deep Reinforcement Learning for Continuous Control"
   - MuJoCo benchmark standards
   - Sample efficiency metrics
   - Evaluation frequency guidelines

5. **Colas et al. (2018)** - "How Many Random Seeds?"
   - Statistical power analysis for RL
   - Minimum seed requirements
   - Effect of seed selection

6. **Islam et al. (2017)** - "Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control"
   - Showed implementation details matter
   - Importance of exact hyperparameters
   - Reproducibility challenges

7. **Cobbe et al. (2019)** - "Quantifying Generalization in Reinforcement Learning"
   - Procedural generation for generalization testing
   - Train/test split for RL
   - Overfitting in RL

8. **Jordan et al. (2020)** - "Evaluating the Performance of Reinforcement Learning Algorithms"
   - Comprehensive evaluation framework
   - Multiple metrics importance
   - Visualization standards

### Statistical Methods

1. **Demšar (2006)** - "Statistical Comparisons of Classifiers over Multiple Data Sets"
   - Friedman test for multiple comparisons
   - Nemenyi post-hoc test
   - Critical difference diagrams

2. **Dolan & Moré (2002)** - "Benchmarking optimization software with performance profiles"
   - Performance profiles methodology
   - Robust comparison across multiple problems

3. **Efron & Tibshirani (1993)** - "An Introduction to the Bootstrap"
   - Bootstrap confidence intervals
   - Bootstrap hypothesis testing

---

## Summary

### Essential Checklist for RL Evaluation

- [ ] **Multiple Seeds**: Minimum 5, preferably 10+
- [ ] **Confidence Intervals**: Report 95% CI, not just mean ± std
- [ ] **Statistical Tests**: Use appropriate tests (Mann-Whitney, bootstrap)
- [ ] **IQM Reporting**: Include Inter-quartile Mean for robustness
- [ ] **Deterministic Evaluation**: Always evaluate with deterministic policy
- [ ] **Fixed Evaluation Seeds**: Use same seeds across algorithms
- [ ] **Environment-Specific Settings**: Follow literature standards per domain
- [ ] **Sample Efficiency**: Report steps to 90% performance
- [ ] **AUC**: Include area under learning curve
- [ ] **Effect Size**: Report Cohen's d or similar
- [ ] **Visualization**: Learning curves with confidence bands
- [ ] **Performance Profiles**: For comparing across multiple environments
- [ ] **Probability of Improvement**: More interpretable than p-values
- [ ] **Hyperparameter Reporting**: Full configuration details
- [ ] **Computational Cost**: Report wall-clock time and resources

### Common Pitfalls to Avoid

1. **Single Seed Results**: Never report single seed performance
2. **Cherry-Picking**: Report all metrics, not just favorable ones
3. **Unfair Comparison**: Ensure equal computational budget
4. **Missing Confidence Intervals**: Always include uncertainty
5. **Wrong Statistical Tests**: Don't assume normality
6. **Evaluation During Training**: Keep training/evaluation separate
7. **Stochastic Evaluation**: Use deterministic policy for evaluation
8. **Incomplete Reporting**: Include all hyperparameters
9. **Visual Misrepresentation**: Don't hide variance in plots
10. **P-hacking**: Decide on tests before running experiments

This guide provides the foundation for scientifically rigorous RL evaluation following the highest standards from the literature.