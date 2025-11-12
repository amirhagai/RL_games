# RL Evaluation Metrics Library

A comprehensive suite of evaluation metrics for Reinforcement Learning algorithms, implementing best practices from recent literature.

## 📦 Installation

No additional installation required beyond the main project dependencies:
```bash
pip install numpy scipy
```

## 🚀 Quick Start

### Import the Library

```python
# Import main evaluator
from evaluation import RLEvaluator, quick_evaluate

# Import specific metrics
from evaluation import (
    SampleEfficiencyMetrics,
    FinalPerformanceMetrics,
    StabilityMetrics,
    InterquartileMean
)
```

## 📊 Basic Usage Examples

### 1. Quick Evaluation (Single Algorithm)

```python
import numpy as np
from evaluation import quick_evaluate

# Your reward history: [n_evaluations] or [n_seeds, n_evaluations]
rewards = np.array([...])

# Quick evaluation with all metrics
results = quick_evaluate(rewards, metric='all')

# Or specific metric only
efficiency = quick_evaluate(rewards, metric='efficiency')
stability = quick_evaluate(rewards, metric='stability')
iqm_result = quick_evaluate(rewards, metric='iqm')
```

### 2. Compare Multiple Algorithms

```python
from evaluation import RLEvaluator

# Prepare reward histories for each algorithm
# Shape: [n_seeds, n_evaluations] for each
algorithms = {
    'PPO': ppo_rewards,
    'A2C': a2c_rewards,
    'DQN': dqn_rewards
}

# Create evaluator
evaluator = RLEvaluator(algorithms)

# Run comprehensive evaluation
results = evaluator.comprehensive_evaluation()

# Generate human-readable report
report = evaluator.generate_report('evaluation_report.txt')
print(report)
```

### 3. Individual Metric Usage

#### Sample Efficiency
```python
from evaluation import SampleEfficiencyMetrics

# Analyze how quickly algorithm learns
metrics = SampleEfficiencyMetrics(reward_history, timesteps)
print(f"Time to 90% of max: {metrics.time_to_percentage_of_max(0.9)}")
print(f"Convergence at: {metrics.convergence_timestep()}")
summary = metrics.get_summary()
```

#### Final Performance
```python
from evaluation import FinalPerformanceMetrics

# Analyze final performance with confidence intervals
# Input shape: [n_seeds, n_evaluations]
metrics = FinalPerformanceMetrics(multi_seed_rewards)
mean, std, stderr = metrics.final_performance()
ci_lower, ci_upper = metrics.confidence_interval(0.95)
print(f"Final: {mean:.2f} ± {std:.2f}, 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

#### Stability Analysis
```python
from evaluation import StabilityMetrics

# Detect training issues
metrics = StabilityMetrics(reward_history)
print(f"Stability index: {metrics.stability_index():.3f}")
print(f"Catastrophic failures: {metrics.catastrophic_failures()['n_failures']}")
plateaus = metrics.plateau_detection()
```

#### IQM (Robust to Outliers)
```python
from evaluation import InterquartileMean

# Robust evaluation using IQM
iqm_calc = InterquartileMean(rewards)
iqm = iqm_calc.compute_iqm()
robust_stats = iqm_calc.compute_robust_statistics()
print(f"IQM: {robust_stats['iqm']:.2f} (vs Mean: {robust_stats['mean']:.2f})")
```

## 🔧 Integration with Benchmark System

### With Your Existing Benchmark Results

```python
from evaluation import RLEvaluator
import json
import numpy as np

# Load your benchmark results
def load_benchmark_results(results_dir):
    """Load results from benchmark.py output"""
    with open(f"{results_dir}/results.json", 'r') as f:
        results = json.load(f)

    # Organize by algorithm
    algorithms = {}
    for result in results:
        algo = result['algorithm']
        env = result['env_id']
        key = f"{algo}/{env}"

        # Load evaluation history
        eval_file = f"{results_dir}/{env}/seed_{result['seed']}/evaluations.json"
        with open(eval_file, 'r') as f:
            eval_data = json.load(f)
            rewards = [e['mean_reward'] for e in eval_data['results']]

        if key not in algorithms:
            algorithms[key] = []
        algorithms[key].append(rewards)

    # Convert to numpy arrays
    for key in algorithms:
        algorithms[key] = np.array(algorithms[key])

    return algorithms

# Evaluate benchmark results
results_dir = 'results/20240101_comprehensive_benchmark'
algorithms = load_benchmark_results(results_dir)

evaluator = RLEvaluator(algorithms)
evaluation = evaluator.comprehensive_evaluation()
```

### Direct Integration in Training Loop

```python
from evaluation import RLEvaluator

class TrainingManager:
    def __init__(self):
        self.reward_histories = {}

    def train_algorithm(self, algo_name, env):
        rewards = []
        for episode in range(n_episodes):
            # ... training code ...
            if episode % eval_freq == 0:
                eval_reward = self.evaluate(env)
                rewards.append(eval_reward)

        self.reward_histories[algo_name] = rewards

    def evaluate_all(self):
        evaluator = RLEvaluator(self.reward_histories)
        return evaluator.comprehensive_evaluation()
```

## 📈 Output Format

### Comprehensive Evaluation Returns:
```python
{
    'sample_efficiency': {
        'time_to_90_percent': 50000,
        'convergence_timestep': 80000,
        'jumpstart_performance': -100.5,
        'rankings': {...}
    },
    'final_performance': {
        'mean': 450.2,
        'std': 23.1,
        'confidence_interval_95': [420.1, 480.3],
        'rankings': {...}
    },
    'stability': {
        'stability_index': 0.823,
        'catastrophic_failures': {'count': 2, ...},
        'rankings': {...}
    },
    'robust_statistics': {
        'iqm': 445.6,
        'median': 448.2,
        'outliers': 3,
        'rankings': {...}
    },
    'overall_rankings': {
        'aggregate': ['PPO', 'A2C', 'DQN'],
        'scores': {'PPO': 12, 'A2C': 10, 'DQN': 6}
    }
}
```

## 🎯 When to Use Each Metric

| Use Case | Recommended Approach |
|----------|---------------------|
| **Quick Check** | `quick_evaluate(rewards, 'all')` |
| **Paper Results** | `RLEvaluator` with IQM and CI |
| **Debug Training** | `StabilityMetrics` for failure detection |
| **Compare Algorithms** | `RLEvaluator` with multiple seeds |
| **Sample Efficiency** | `SampleEfficiencyMetrics` for learning speed |

## 📝 Complete Examples

### Run Usage Demonstrations

```bash
# Individual metric examples
python evaluation/usage/sample_efficiency_usage.py
python evaluation/usage/final_performance_usage.py
python evaluation/usage/stability_metrics_usage.py
python evaluation/usage/iqm_usage.py

# Comprehensive demo (recommended to start)
python ../evaluation_demo.py
```

### Minimal Complete Example

```python
import numpy as np
from evaluation import RLEvaluator

# Simulate some training data
np.random.seed(42)
n_seeds, n_evals = 5, 100

# Create fake reward histories
algorithms = {
    'Algorithm_A': np.random.randn(n_seeds, n_evals).cumsum(axis=1),
    'Algorithm_B': np.random.randn(n_seeds, n_evals).cumsum(axis=1) + 10
}

# Evaluate
evaluator = RLEvaluator(algorithms)
results = evaluator.comprehensive_evaluation()

# Print key findings
print(f"Best algorithm (IQM): {results['robust_statistics']['rankings']['by_iqm'][0]}")
print(f"Most stable: {results['stability']['rankings']['by_stability'][0]}")
print(f"Overall winner: {results['overall_rankings']['aggregate'][0]}")

# Save detailed report
report = evaluator.generate_report('my_evaluation.txt')
```

## ⚠️ Important Notes

1. **Always use multiple seeds** (minimum 5, ideally 10+) for statistical validity
2. **IQM is preferred over mean** when outliers/failures are present (Agarwal et al. 2021)
3. **Report confidence intervals**, not just point estimates
4. **Normalize before aggregating** across different environments
5. **Consider both efficiency and final performance** for complete evaluation

## 📚 References

- Agarwal et al. (2021): "Deep RL at the Edge of the Statistical Precipice"
- Henderson et al. (2018): "Deep Reinforcement Learning that Matters"
- See `RL_EVALUATION_GUIDE.md` for detailed metric explanations

## 🐛 Troubleshooting

### Common Issues:

1. **Import Error**:
```python
# Make sure you're in the parent directory
import sys
sys.path.append('/home/amir/Desktop/RL')
from evaluation import RLEvaluator
```

2. **Shape Mismatch**:
```python
# Ensure correct shape
# Single seed: [n_evaluations]
# Multi seed: [n_seeds, n_evaluations]
rewards = np.array(rewards)
if len(rewards.shape) == 1:
    rewards = rewards.reshape(1, -1)  # Convert to multi-seed format
```

3. **Missing Dependencies**:
```bash
pip install numpy scipy
```

## 💡 Pro Tips

- Use `evaluation_demo.py` to see all features in action
- Start with `quick_evaluate()` for exploration
- Use `RLEvaluator` for publication-quality results
- Always save evaluation results to JSON for reproducibility
- Generate reports for human-readable summaries