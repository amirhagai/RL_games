# RL Benchmarking System - Complete Usage Guide

This guide provides comprehensive usage examples for every file in the RL benchmarking system, organized by difficulty level.

## Table of Contents
1. [extract_gym_metadata.py](#1-extract_gym_metadatapy)
2. [analyze_environments.py](#2-analyze_environmentspy)
3. [env_selector.py](#3-env_selectorpy)
4. [parallel_envs.py](#4-parallel_envspy)
5. [benchmark_config.py](#5-benchmark_configpy)
6. [algorithm_wrapper.py](#6-algorithm_wrapperpy)
7. [callbacks.py](#7-callbackspy)
8. [benchmark.py](#8-benchmarkpy)
9. [visualize_live.py](#9-visualize_livepy)
10. [check_gymnasium_envs.py](#10-check_gymnasium_envspy)

---

## 1. extract_gym_metadata.py
**Purpose**: Extracts raw metadata from all registered Gymnasium environments.

### 🟢 Easy Usage
```bash
# Basic extraction - creates gym_metadata_raw.json
python extract_gym_metadata.py
```

### 🟡 Medium Usage
```python
# Use as a module to extract specific environment metadata
from extract_gym_metadata import extract_env_metadata

# Get metadata for a single environment
metadata = extract_env_metadata('CartPole-v1')
print(f"Max steps: {metadata['spec']['max_episode_steps']}")
print(f"Action space: {metadata['action_space']['type']}")
```

### 🔴 Hard Usage
```python
# Custom metadata extraction with additional fields
import gymnasium as gym
from extract_gym_metadata import extract_env_metadata, convert_to_native_types

def extract_extended_metadata(env_id):
    """Extract metadata with custom fields."""
    base_metadata = extract_env_metadata(env_id)
    if base_metadata:
        env = gym.make(env_id)
        # Add custom fields
        base_metadata['custom'] = {
            'has_wrapper': hasattr(env, 'wrapper'),
            'metadata_keys': list(env.metadata.keys()) if hasattr(env, 'metadata') else [],
            'reward_range': env.reward_range,
            'spec_kwargs': env.spec.kwargs if env.spec else {}
        }
        env.close()
        return convert_to_native_types(base_metadata)
    return None

# Extract with custom fields for all MuJoCo environments
from gymnasium.envs.registration import registry
mujoco_envs = [spec.id for spec in registry.values() if 'mujoco' in spec.id.lower()]
extended_metadata = [extract_extended_metadata(env_id) for env_id in mujoco_envs[:5]]
```

---

## 2. analyze_environments.py
**Purpose**: Categorizes environments and estimates difficulty based on research.

### 🟢 Easy Usage
```bash
# Analyze all environments - creates env_metadata.json
python analyze_environments.py
```

### 🟡 Medium Usage
```python
# Use categorization functions programmatically
from analyze_environments import categorize_env, estimate_difficulty

# Categorize a specific environment
env_id = "ALE/MontezumaRevenge-v5"
category = categorize_env(env_id, obs_shape=(210, 160, 3))
difficulty = estimate_difficulty(env_id, category, max_steps=None, reward_threshold=None)

print(f"{env_id}: Category={category}, Difficulty={difficulty}")
```

### 🔴 Hard Usage
```python
# Custom difficulty estimation with your own criteria
import json
from analyze_environments import analyze_metadata

# Load raw metadata
with open('gym_metadata_raw.json', 'r') as f:
    raw_data = json.load(f)

def custom_difficulty_estimator(metadata):
    """Custom difficulty based on observation/action space complexity."""
    obs_size = 1
    if metadata['observation_space']['shape']:
        for dim in metadata['observation_space']['shape']:
            obs_size *= dim

    action_size = metadata['action_space'].get('size', 1)

    # Custom scoring
    complexity_score = obs_size * action_size

    if complexity_score > 1000000:
        return 'extreme'
    elif complexity_score > 100000:
        return 'very_hard'
    elif complexity_score > 10000:
        return 'hard'
    elif complexity_score > 1000:
        return 'medium'
    else:
        return 'easy'

# Apply custom analysis
for env_meta in raw_data['environments'][:10]:
    analyzed = analyze_metadata(env_meta)
    analyzed['custom_difficulty'] = custom_difficulty_estimator(env_meta)
    print(f"{analyzed['env_id']}: {analyzed['custom_difficulty']}")
```

---

## 3. env_selector.py
**Purpose**: Select environments based on criteria and predefined suites.

### 🟢 Easy Usage
```bash
# Run demo to see available suites
python env_selector.py

# Use in Python
from env_selector import BENCHMARK_SUITES

# Get a predefined suite
quick_envs = BENCHMARK_SUITES['quick']
atari_envs = BENCHMARK_SUITES['atari_dense']
```

### 🟡 Medium Usage
```python
from env_selector import EnvironmentSelector

selector = EnvironmentSelector()

# Select by various criteria
easy_envs = selector.by_difficulty('easy')
discrete_envs = selector.by_action_space('Discrete')
mujoco_envs = selector.by_category('mujoco')

# Get diverse sample with specific seed
diverse_10 = selector.diverse_sample(n=10, seed=42)

# Find similar environments
similar = selector.similar_to('CartPole-v1', n=5)
print(f"Environments similar to CartPole-v1: {similar}")

# Progressive sets for scaling experiments
prog_sets = selector.progressive_sets()
print(f"1 env: {prog_sets[1]}")
print(f"5 envs: {prog_sets[5]}")
print(f"10 envs: {prog_sets[10]}")
```

### 🔴 Hard Usage
```python
from env_selector import EnvironmentSelector
import json

class CustomSelector(EnvironmentSelector):
    """Extended selector with custom filtering."""

    def by_observation_size(self, min_size=None, max_size=None):
        """Select by observation space size."""
        results = []
        for env in self.environments:
            if env['observation_space']['shape']:
                size = 1
                for dim in env['observation_space']['shape']:
                    size *= dim
                if (min_size is None or size >= min_size) and \
                   (max_size is None or size <= max_size):
                    results.append(env['env_id'])
        return results

    def by_episode_length(self, min_steps=None, max_steps=None):
        """Select by episode length."""
        results = []
        for env in self.environments:
            steps = env['spec']['max_episode_steps']
            if steps is not None:
                if (min_steps is None or steps >= min_steps) and \
                   (max_steps is None or steps <= max_steps):
                    results.append(env['env_id'])
        return results

    def create_custom_suite(self, criteria):
        """Create suite based on multiple criteria."""
        # Start with all environments
        candidates = set(env['env_id'] for env in self.environments)

        # Apply filters
        if 'difficulty' in criteria:
            candidates &= set(self.by_difficulty(criteria['difficulty']))
        if 'category' in criteria:
            candidates &= set(self.by_category(criteria['category']))
        if 'min_obs_size' in criteria or 'max_obs_size' in criteria:
            candidates &= set(self.by_observation_size(
                criteria.get('min_obs_size'),
                criteria.get('max_obs_size')
            ))

        return list(candidates)

# Use custom selector
selector = CustomSelector()

# Complex multi-criteria selection
custom_suite = selector.create_custom_suite({
    'difficulty': 'medium',
    'category': 'atari',
    'min_obs_size': 10000,
    'max_obs_size': 100000
})

print(f"Custom suite: {custom_suite[:5]}")
```

---

## 4. parallel_envs.py
**Purpose**: Manage parallel and vectorized environment execution.

### 🟢 Easy Usage
```bash
# Run test demo
python parallel_envs.py

# Basic usage
from parallel_envs import create_vectorized_env

# Create 4 parallel CartPole environments
vec_env = create_vectorized_env('CartPole-v1', num_envs=4)
obs, info = vec_env.reset()
```

### 🟡 Medium Usage
```python
from parallel_envs import ParallelEnvManager, ResourceMonitor
import time

# Manage multiple environment types in parallel
manager = ParallelEnvManager(
    env_ids=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'],
    num_envs_per_id=4,  # 4 copies of each
    vectorization_mode='async'  # or 'sync'
)

# Create all vectorized environments
vec_envs = manager.get_all_vectorized_envs(seed=42)

# Run with resource monitoring
monitor = ResourceMonitor()
for _ in range(10):
    monitor.snapshot()
    # Your training code here
    time.sleep(0.5)

summary = monitor.summary()
print(f"Peak CPU usage: {summary['cpu_max']:.1f}%")
print(f"Peak memory: {summary['memory_max']:.1f}%")
```

### 🔴 Hard Usage
```python
from parallel_envs import ParallelEnvManager, ResourceMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def train_on_env_group(env_ids, config):
    """Train on a group of environments."""
    manager = ParallelEnvManager(env_ids, config['num_envs'], 'async')

    results = {}
    for env_id in env_ids:
        # Create vectorized env using SB3 (for compatibility)
        vec_env = make_vec_env(env_id, n_envs=config['num_envs'])

        # Train model
        model = PPO('MlpPolicy', vec_env, verbose=0)
        model.learn(total_timesteps=config['timesteps'])

        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
        results[env_id] = {'mean': mean_reward, 'std': std_reward}

        vec_env.close()

    return results

# Parallel training on multiple environment groups
env_groups = [
    ['CartPole-v1', 'Acrobot-v1'],
    ['MountainCar-v0', 'Pendulum-v1'],
    ['LunarLander-v3', 'BipedalWalker-v3']
]

config = {'num_envs': 4, 'timesteps': 50000}

with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(train_on_env_group, group, config)
               for group in env_groups]

    all_results = {}
    for future in futures:
        all_results.update(future.result())

print(f"Trained on {len(all_results)} environments in parallel")
```

---

## 5. benchmark_config.py
**Purpose**: Configuration management for benchmark runs.

### 🟢 Easy Usage
```bash
# Run demo
python benchmark_config.py

# Use presets
from benchmark_config import PRESET_CONFIGS

quick_config = PRESET_CONFIGS['quick']
standard_config = PRESET_CONFIGS['standard']
```

### 🟡 Medium Usage
```python
from benchmark_config import BenchmarkConfig
from pathlib import Path

# Create custom configuration
config = BenchmarkConfig(
    mode='custom',
    env_suite='atari_dense',
    algorithm='dqn',
    total_timesteps=500000,
    num_seeds=3,
    eval_freq=25000,
    vectorize=8,  # Use 8 parallel envs
    experiment_name='my_atari_experiment'
)

# Modify algorithm-specific parameters
config.algorithm_kwargs = {
    'learning_rate': 1e-4,
    'buffer_size': 100000,
    'learning_starts': 10000,
    'batch_size': 32,
    'gamma': 0.99
}

# Save configuration
config.save(Path('configs/my_config.json'))

# Load configuration
loaded_config = BenchmarkConfig.load(Path('configs/my_config.json'))
```

### 🔴 Hard Usage
```python
from benchmark_config import BenchmarkConfig
from dataclasses import dataclass, field
from typing import Dict, Any
import json

@dataclass
class AdvancedBenchmarkConfig(BenchmarkConfig):
    """Extended config with additional features."""

    # Hardware settings
    use_gpu: bool = True
    gpu_id: int = 0
    cpu_affinity: List[int] = field(default_factory=list)

    # Advanced training
    curriculum: Dict[str, Any] = field(default_factory=dict)
    hyperparameter_schedule: Dict[str, Any] = field(default_factory=dict)
    early_stopping: Dict[str, Any] = field(default_factory=dict)

    # Experiment tracking
    wandb_project: Optional[str] = None
    wandb_tags: List[str] = field(default_factory=list)

    def apply_curriculum(self, current_step: int):
        """Apply curriculum learning schedule."""
        if self.curriculum:
            for param, schedule in self.curriculum.items():
                if isinstance(schedule, dict):
                    # Step-based schedule
                    for step, value in sorted(schedule.items()):
                        if current_step >= int(step):
                            self.algorithm_kwargs[param] = value

    def should_early_stop(self, rewards: List[float]) -> bool:
        """Check early stopping criteria."""
        if not self.early_stopping:
            return False

        if 'min_reward' in self.early_stopping:
            if rewards[-1] >= self.early_stopping['min_reward']:
                return True

        if 'patience' in self.early_stopping:
            patience = self.early_stopping['patience']
            if len(rewards) > patience:
                recent = rewards[-patience:]
                if all(r >= self.early_stopping.get('threshold', 0) for r in recent):
                    return True

        return False

# Create advanced configuration
advanced_config = AdvancedBenchmarkConfig(
    mode='comprehensive',
    algorithm='ppo',
    total_timesteps=1000000,
    curriculum={
        'learning_rate': {
            '0': 3e-4,
            '500000': 1e-4,
            '750000': 3e-5
        }
    },
    early_stopping={
        'min_reward': 450,  # Stop if we reach this reward
        'patience': 10,     # Stop if stable for 10 evaluations
        'threshold': 400
    },
    wandb_project='rl-benchmarks',
    wandb_tags=['ppo', 'curriculum', 'advanced']
)

# Dynamic configuration based on environment
def create_env_specific_config(env_id: str) -> AdvancedBenchmarkConfig:
    """Create optimal config based on environment type."""

    base_config = AdvancedBenchmarkConfig(mode='standard')

    if 'Atari' in env_id or 'ALE/' in env_id:
        base_config.algorithm = 'dqn'
        base_config.algorithm_kwargs = {
            'buffer_size': 100000,
            'learning_starts': 10000,
            'train_freq': 4,
            'gradient_steps': 1
        }
        base_config.vectorize = 1  # DQN doesn't benefit from vectorization

    elif 'Humanoid' in env_id:
        base_config.algorithm = 'sac'
        base_config.total_timesteps = 2000000
        base_config.algorithm_kwargs = {
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'batch_size': 256
        }

    elif 'CartPole' in env_id:
        base_config.algorithm = 'ppo'
        base_config.total_timesteps = 50000
        base_config.early_stopping = {'min_reward': 450}

    return base_config
```

---

## 6. algorithm_wrapper.py
**Purpose**: Unified interface for different RL algorithms.

### 🟢 Easy Usage
```bash
# Run test
python algorithm_wrapper.py

# Basic usage
from algorithm_wrapper import create_algorithm
import gymnasium as gym

env = gym.make('CartPole-v1')
algo = create_algorithm(env, 'ppo', {}, seed=42)
algo.train(total_timesteps=10000)
```

### 🟡 Medium Usage
```python
from algorithm_wrapper import create_algorithm, StableBaselines3Wrapper
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# Create vectorized environment
vec_env = make_vec_env('LunarLander-v3', n_envs=4)

# Configure algorithm with specific hyperparameters
config = {
    'learning_rate': 1e-3,
    'n_steps': 1024,
    'batch_size': 64,
    'n_epochs': 4,
    'gamma': 0.999,
    'gae_lambda': 0.98,
    'ent_coef': 0.01,
    'verbose': 1
}

# Create and train
algo = create_algorithm(vec_env, 'ppo', config, seed=42)

# Train with evaluation callback
from callbacks import BenchmarkCallback
eval_env = gym.make('LunarLander-v3')
callback = BenchmarkCallback(
    env_id='LunarLander-v3',
    eval_env=eval_env,
    eval_freq=5000,
    eval_episodes=10,
    save_path=Path('lunar_lander_training'),
    checkpoint_freq=10000
)

algo.train(total_timesteps=100000, callback=callback)

# Save and load
algo.save(Path('models/lunar_lander_ppo'))
algo.load(Path('models/lunar_lander_ppo'))

# Evaluate
results = algo.evaluate(num_episodes=20)
print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
```

### 🔴 Hard Usage
```python
from algorithm_wrapper import AlgorithmWrapper, CustomAlgorithmWrapper
from typing import Dict, Any, Optional, Callable
import numpy as np
from pathlib import Path

class MyCustomAlgorithm:
    """Custom RL algorithm implementation."""

    def __init__(self, env, config: Dict[str, Any], seed: int = 0):
        self.env = env
        self.config = config
        np.random.seed(seed)
        self.policy = self._init_policy()

    def _init_policy(self):
        # Initialize your custom policy
        pass

    def train(self, total_timesteps: int, callback: Optional[Callable] = None):
        # Custom training loop
        for step in range(total_timesteps):
            # Your training logic
            if callback and step % 1000 == 0:
                callback(locals(), globals())

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.predict(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)

        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'num_episodes': num_episodes
        }

    def predict(self, observation):
        # Your prediction logic
        return self.env.action_space.sample()

    def save(self, path: Path):
        # Save your model
        pass

    def load(self, path: Path):
        # Load your model
        pass

# Use custom algorithm with the wrapper
custom_algo = CustomAlgorithmWrapper(
    env=gym.make('CartPole-v1'),
    algorithm_class=MyCustomAlgorithm,
    config={'custom_param': 42},
    seed=0
)

# Hybrid approach: switch algorithms based on performance
class AdaptiveAlgorithmWrapper(AlgorithmWrapper):
    """Switches between algorithms based on performance."""

    def __init__(self, env, algorithms: Dict[str, AlgorithmWrapper], switch_threshold: float):
        super().__init__(env, {}, 0)
        self.algorithms = algorithms
        self.current_algo = list(algorithms.keys())[0]
        self.switch_threshold = switch_threshold
        self.performance_history = {name: [] for name in algorithms}

    def train(self, total_timesteps: int, callback: Optional[Callable] = None):
        # Train with current algorithm
        self.algorithms[self.current_algo].train(total_timesteps, callback)

        # Evaluate and potentially switch
        eval_result = self.evaluate()
        self.performance_history[self.current_algo].append(eval_result['mean_reward'])

        # Check if we should switch
        for name, algo in self.algorithms.items():
            if name != self.current_algo:
                test_result = algo.evaluate(num_episodes=5)
                if test_result['mean_reward'] > eval_result['mean_reward'] + self.switch_threshold:
                    print(f"Switching from {self.current_algo} to {name}")
                    self.current_algo = name

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        return self.algorithms[self.current_algo].evaluate(num_episodes)

    def save(self, path: Path):
        for name, algo in self.algorithms.items():
            algo.save(path / f"{name}_model")

    def load(self, path: Path):
        for name, algo in self.algorithms.items():
            algo.load(path / f"{name}_model")

# Create adaptive wrapper with multiple algorithms
env = make_vec_env('CartPole-v1', n_envs=4)
adaptive_algo = AdaptiveAlgorithmWrapper(
    env=env,
    algorithms={
        'ppo': create_algorithm(env, 'ppo', {}, seed=42),
        'a2c': create_algorithm(env, 'a2c', {}, seed=42)
    },
    switch_threshold=50.0
)
```

---

## 7. callbacks.py
**Purpose**: Training callbacks for monitoring and checkpointing.

### 🟢 Easy Usage
```bash
# Run test
python callbacks.py

# Basic usage
from callbacks import BenchmarkCallback
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make('CartPole-v1')
model = PPO('MlpPolicy', env)

callback = BenchmarkCallback(
    env_id='CartPole-v1',
    eval_env=env,
    eval_freq=1000,
    eval_episodes=5,
    save_path=Path('training_results'),
    checkpoint_freq=5000
)

model.learn(total_timesteps=10000, callback=callback)
```

### 🟡 Medium Usage
```python
from callbacks import BenchmarkCallback
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from pathlib import Path
import json

# Chain multiple callbacks
class CustomMetricsCallback(BenchmarkCallback):
    """Extended callback with custom metrics."""

    def _evaluate(self) -> Dict[str, Any]:
        # Get standard evaluation
        results = super()._evaluate()

        # Add custom metrics
        results['episode_length'] = []
        results['success_rate'] = 0

        for _ in range(self.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            steps = 0
            while not done and steps < 1000:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                done = terminated or truncated
                steps += 1

            results['episode_length'].append(steps)
            if steps < 500:  # Custom success criterion
                results['success_rate'] += 1

        results['success_rate'] /= self.eval_episodes
        results['avg_episode_length'] = np.mean(results['episode_length'])

        return results

# Use with multiple callbacks
benchmark_callback = CustomMetricsCallback(
    env_id='CartPole-v1',
    eval_env=eval_env,
    eval_freq=2000,
    eval_episodes=10,
    save_path=Path('advanced_training'),
    checkpoint_freq=10000
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./best_model/',
    log_path='./logs/',
    eval_freq=2000,
    deterministic=True
)

callback_list = CallbackList([benchmark_callback, eval_callback])
model.learn(total_timesteps=50000, callback=callback_list)
```

### 🔴 Hard Usage
```python
from callbacks import BenchmarkCallback
from stable_baselines3.common.callbacks import BaseCallback
import wandb
import matplotlib.pyplot as plt
from typing import Dict, Any
import numpy as np

class AdvancedBenchmarkCallback(BenchmarkCallback):
    """Advanced callback with Weights & Biases, plotting, and early stopping."""

    def __init__(self,
                 env_id: str,
                 eval_env,
                 eval_freq: int,
                 eval_episodes: int,
                 save_path: Path,
                 checkpoint_freq: int,
                 wandb_config: Dict[str, Any] = None,
                 early_stopping_patience: int = 10,
                 early_stopping_threshold: float = None,
                 plot_freq: int = 10000,
                 verbose: int = 0):

        super().__init__(env_id, eval_env, eval_freq, eval_episodes,
                        save_path, checkpoint_freq, verbose)

        self.wandb_config = wandb_config
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.plot_freq = plot_freq

        self.best_mean_reward = -float('inf')
        self.patience_counter = 0

        if wandb_config:
            wandb.init(**wandb_config)

    def _on_training_start(self):
        super()._on_training_start()

        # Log hyperparameters to wandb
        if self.wandb_config:
            wandb.config.update({
                'env_id': self.env_id,
                'algorithm': self.model.__class__.__name__,
                'total_timesteps': self.locals.get('total_timesteps', 0)
            })

    def _on_step(self) -> bool:
        # Regular evaluation and checkpointing
        continue_training = super()._on_step()

        # Plot learning curves periodically
        if self.n_calls % self.plot_freq == 0 and len(self.evaluations) > 1:
            self._plot_learning_curves()

        # Log to wandb
        if self.wandb_config and self.n_calls % self.eval_freq == 0:
            latest_eval = self.evaluations[-1]
            wandb.log({
                'timestep': self.num_timesteps,
                'mean_reward': latest_eval['mean_reward'],
                'std_reward': latest_eval['std_reward'],
                'wall_time': latest_eval['wall_time']
            })

            # Early stopping check
            if self.early_stopping_threshold:
                if latest_eval['mean_reward'] > self.best_mean_reward:
                    self.best_mean_reward = latest_eval['mean_reward']
                    self.patience_counter = 0
                    # Save best model
                    self.model.save(self.save_path / 'best_model')
                else:
                    self.patience_counter += 1

                if latest_eval['mean_reward'] >= self.early_stopping_threshold:
                    print(f"\n🎉 Reached target reward {self.early_stopping_threshold}!")
                    return False

                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n⚠️ Early stopping: No improvement for {self.early_stopping_patience} evaluations")
                    return False

        return continue_training

    def _plot_learning_curves(self):
        """Generate and save learning curve plots."""
        if len(self.evaluations) < 2:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        timesteps = [e['timestep'] for e in self.evaluations]
        rewards = [e['mean_reward'] for e in self.evaluations]
        stds = [e['std_reward'] for e in self.evaluations]
        wall_times = [e['wall_time'] for e in self.evaluations]

        # Reward over timesteps
        axes[0, 0].plot(timesteps, rewards, 'b-', label='Mean')
        axes[0, 0].fill_between(timesteps,
                                [r - s for r, s in zip(rewards, stds)],
                                [r + s for r, s in zip(rewards, stds)],
                                alpha=0.3)
        axes[0, 0].set_xlabel('Timesteps')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Learning Curve')
        axes[0, 0].grid(True)

        # Reward over wall time
        axes[0, 1].plot(wall_times, rewards, 'g-')
        axes[0, 1].set_xlabel('Wall Time (s)')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].set_title('Reward vs Time')
        axes[0, 1].grid(True)

        # Reward variance
        axes[1, 0].plot(timesteps, stds, 'r-')
        axes[1, 0].set_xlabel('Timesteps')
        axes[1, 0].set_ylabel('Std Deviation')
        axes[1, 0].set_title('Reward Variance')
        axes[1, 0].grid(True)

        # Sample efficiency
        if len(rewards) > 1:
            efficiency = np.diff(rewards) / np.diff(timesteps)
            axes[1, 1].plot(timesteps[1:], efficiency, 'purple')
            axes[1, 1].set_xlabel('Timesteps')
            axes[1, 1].set_ylabel('Δ Reward / Δ Timesteps')
            axes[1, 1].set_title('Sample Efficiency')
            axes[1, 1].grid(True)

        plt.suptitle(f'{self.env_id} Training Progress')
        plt.tight_layout()

        # Save plot
        plot_path = self.save_path / f'learning_curves_{self.num_timesteps}.png'
        plt.savefig(plot_path, dpi=100)
        plt.close()

        if self.wandb_config:
            wandb.log({'learning_curves': wandb.Image(str(plot_path))})

    def _on_training_end(self):
        super()._on_training_end()

        # Final plots
        self._plot_learning_curves()

        # Close wandb
        if self.wandb_config:
            wandb.finish()

# Use advanced callback
advanced_callback = AdvancedBenchmarkCallback(
    env_id='LunarLander-v3',
    eval_env=eval_env,
    eval_freq=5000,
    eval_episodes=20,
    save_path=Path('advanced_results'),
    checkpoint_freq=25000,
    wandb_config={
        'project': 'rl-benchmarks',
        'name': 'lunar-lander-advanced',
        'tags': ['ppo', 'advanced-callback']
    },
    early_stopping_patience=15,
    early_stopping_threshold=200.0,
    plot_freq=10000,
    verbose=1
)
```

---

## 8. benchmark.py
**Purpose**: Main benchmark runner orchestrating experiments across environments.

### 🟢 Easy Usage
```bash
# Quick benchmark (5 environments, 10k steps each)
python benchmark.py --mode quick

# Test specific algorithm
python benchmark.py --mode quick --algorithm a2c

# Test single environment
python benchmark.py --env-ids CartPole-v1 --timesteps 20000
```

### 🟡 Medium Usage
```bash
# Test on predefined suite with multiple seeds
python benchmark.py --env-suite atari_dense --algorithm dqn --seeds 3 --mode standard

# Progressive scaling
python benchmark.py --num-envs 1 --experiment-name scale_1
python benchmark.py --num-envs 5 --experiment-name scale_5
python benchmark.py --num-envs 10 --experiment-name scale_10
python benchmark.py --num-envs 20 --experiment-name scale_20

# Custom configuration
python benchmark.py \
    --env-suite mujoco_easy \
    --algorithm sac \
    --timesteps 100000 \
    --seeds 5 \
    --experiment-name mujoco_sac_test
```

### 🔴 Hard Usage
```python
# Programmatic usage for complex experiments
from benchmark import BenchmarkRunner, run_single_env_experiment
from benchmark_config import BenchmarkConfig
from pathlib import Path
import json

# Create custom runner with specific configuration
class CustomBenchmarkRunner(BenchmarkRunner):
    """Extended runner with custom features."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__(config)
        self.baseline_results = {}

    def load_baseline(self, baseline_path: Path):
        """Load baseline results for comparison."""
        with open(baseline_path) as f:
            self.baseline_results = json.load(f)

    def run_with_comparison(self):
        """Run benchmark and compare with baseline."""
        results = self.run()

        # Compare with baseline
        comparison = {}
        for result in results:
            if result['status'] == 'success':
                env_id = result['env_id']
                if env_id in self.baseline_results:
                    baseline = self.baseline_results[env_id]
                    improvement = result['final_reward'] - baseline['final_reward']
                    comparison[env_id] = {
                        'new': result['final_reward'],
                        'baseline': baseline['final_reward'],
                        'improvement': improvement,
                        'percent_change': (improvement / baseline['final_reward']) * 100
                    }

        # Save comparison
        comp_file = self.output_dir / 'comparison.json'
        with open(comp_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        print("\nComparison with baseline:")
        for env_id, comp in comparison.items():
            print(f"{env_id}: {comp['percent_change']:.1f}% change")

        return results, comparison

# Run A/B test between algorithms
def run_ab_test(env_ids, algorithm_a, algorithm_b, config_base):
    """Run A/B test between two algorithms."""

    results_a = []
    results_b = []

    for env_id in env_ids:
        # Test algorithm A
        config_a = BenchmarkConfig(**config_base)
        config_a.algorithm = algorithm_a
        result_a = run_single_env_experiment(
            env_id, config_a, seed=42,
            output_dir=Path(f'ab_test/{algorithm_a}')
        )
        results_a.append(result_a)

        # Test algorithm B
        config_b = BenchmarkConfig(**config_base)
        config_b.algorithm = algorithm_b
        result_b = run_single_env_experiment(
            env_id, config_b, seed=42,
            output_dir=Path(f'ab_test/{algorithm_b}')
        )
        results_b.append(result_b)

    # Statistical comparison
    from scipy import stats

    rewards_a = [r['final_reward'] for r in results_a if r['status'] == 'success']
    rewards_b = [r['final_reward'] for r in results_b if r['status'] == 'success']

    t_stat, p_value = stats.ttest_ind(rewards_a, rewards_b)

    print(f"\nA/B Test Results:")
    print(f"{algorithm_a} mean: {np.mean(rewards_a):.2f}")
    print(f"{algorithm_b} mean: {np.mean(rewards_b):.2f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < 0.05:
        winner = algorithm_a if np.mean(rewards_a) > np.mean(rewards_b) else algorithm_b
        print(f"✓ {winner} is significantly better")
    else:
        print("✗ No significant difference")

# Hyperparameter sweep
def hyperparameter_sweep(env_id, algorithm, param_grid):
    """Run hyperparameter sweep."""

    from itertools import product

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []
    for combo in combinations:
        params = dict(zip(param_names, combo))

        config = BenchmarkConfig(
            mode='quick',
            env_ids=[env_id],
            algorithm=algorithm,
            algorithm_kwargs=params,
            total_timesteps=50000
        )

        runner = BenchmarkRunner(config)
        run_results = runner.run()

        results.append({
            'params': params,
            'reward': run_results[0]['final_reward'] if run_results[0]['status'] == 'success' else -float('inf')
        })

    # Find best parameters
    best = max(results, key=lambda x: x['reward'])
    print(f"\nBest parameters for {env_id}:")
    print(f"Params: {best['params']}")
    print(f"Reward: {best['reward']:.2f}")

    return results

# Example usage
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'n_steps': [128, 256, 512],
    'batch_size': [32, 64, 128]
}

sweep_results = hyperparameter_sweep('CartPole-v1', 'ppo', param_grid)
```

---

## 9. visualize_live.py
**Purpose**: Live visualization of agents training and performing.

### 🟢 Easy Usage
```bash
# Quick visualization with default settings
python visualize_live.py --env CartPole-v1 --train 10000

# Skip training, just show random agent
python visualize_live.py --env Hopper-v5 --train 0
```

### 🟡 Medium Usage
```bash
# Train and visualize complex environment
python visualize_live.py --env Humanoid-v5 --algorithm sac --train 100000 --episodes 10

# Control visualization speed
python visualize_live.py --env Ant-v5 --train 50000 --fps 60  # Fast
python visualize_live.py --env Walker2d-v5 --train 50000 --fps 15  # Slow

# Load existing model
python visualize_live.py --env Humanoid-v5 --model models/Humanoid-v5_ppo_model.zip --episodes 5
```

### 🔴 Hard Usage
```python
# Custom visualization with recording and analysis
from visualize_live import train_and_visualize, evaluate_and_visualize
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt

class AdvancedVisualizer:
    """Advanced visualization with recording and analysis."""

    def __init__(self, env_id, algorithm='ppo', record=True):
        self.env_id = env_id
        self.algorithm = algorithm
        self.record = record
        self.frames = []
        self.rewards = []
        self.actions = []

    def train_and_record(self, timesteps=50000):
        """Train and record performance metrics."""

        # Create environment with recording
        if self.record:
            env = RecordVideo(
                gym.make(self.env_id, render_mode='rgb_array'),
                video_folder='videos/',
                episode_trigger=lambda x: x % 10 == 0  # Record every 10th episode
            )
        else:
            env = gym.make(self.env_id)

        # Train model
        if self.algorithm == 'ppo':
            model = PPO('MlpPolicy', env, verbose=1)
        elif self.algorithm == 'sac':
            model = SAC('MlpPolicy', env, verbose=1)

        model.learn(total_timesteps=timesteps)

        return model, env

    def analyze_episode(self, model, env, render=True):
        """Analyze a single episode in detail."""

        obs, _ = env.reset()
        done = False

        episode_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'action_probs': []
        }

        while not done:
            # Get action and probability
            action, _ = model.predict(obs, deterministic=False)

            # Store data
            episode_data['observations'].append(obs)
            episode_data['actions'].append(action)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_data['rewards'].append(reward)

            if render and hasattr(env, 'render'):
                frame = env.render()
                if frame is not None:
                    self.frames.append(frame)

        return episode_data

    def visualize_analysis(self, episode_data):
        """Create visualization of episode analysis."""

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        # Rewards over time
        axes[0].plot(episode_data['rewards'])
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Reward per Step')
        axes[0].grid(True)

        # Actions over time
        if len(episode_data['actions'][0].shape) == 0:  # Discrete
            axes[1].plot(episode_data['actions'])
            axes[1].set_ylabel('Action')
        else:  # Continuous
            actions_array = np.array(episode_data['actions'])
            for i in range(actions_array.shape[1]):
                axes[1].plot(actions_array[:, i], label=f'Action {i}')
            axes[1].legend()
        axes[1].set_ylabel('Action Value')
        axes[1].set_title('Actions Taken')
        axes[1].grid(True)

        # Cumulative reward
        cumulative = np.cumsum(episode_data['rewards'])
        axes[2].plot(cumulative)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Cumulative Reward')
        axes[2].set_title('Cumulative Reward')
        axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(f'{self.env_id}_episode_analysis.png')
        plt.show()

    def create_action_heatmap(self, model, env, num_samples=1000):
        """Create heatmap of action distribution."""

        if not isinstance(env.observation_space, gym.spaces.Box):
            print("Heatmap only works for Box observation spaces")
            return

        # Sample observations
        observations = []
        actions = []

        for _ in range(num_samples):
            obs = env.observation_space.sample()
            action, _ = model.predict(obs)
            observations.append(obs)
            actions.append(action)

        # Create heatmap (for 2D observation spaces)
        if len(env.observation_space.shape) == 1 and env.observation_space.shape[0] >= 2:
            obs_array = np.array(observations)

            plt.figure(figsize=(10, 8))

            if isinstance(env.action_space, gym.spaces.Discrete):
                # Discrete actions - use colors
                scatter = plt.scatter(obs_array[:, 0], obs_array[:, 1],
                                     c=actions, cmap='viridis', alpha=0.6)
                plt.colorbar(scatter, label='Action')
            else:
                # Continuous actions - use first action dimension
                actions_array = np.array(actions)
                scatter = plt.scatter(obs_array[:, 0], obs_array[:, 1],
                                     c=actions_array[:, 0], cmap='coolwarm', alpha=0.6)
                plt.colorbar(scatter, label='Action[0]')

            plt.xlabel('Observation[0]')
            plt.ylabel('Observation[1]')
            plt.title(f'Action Distribution for {self.env_id}')
            plt.savefig(f'{self.env_id}_action_heatmap.png')
            plt.show()

# Use advanced visualizer
visualizer = AdvancedVisualizer('LunarLander-v3', algorithm='ppo', record=True)

# Train and record
model, env = visualizer.train_and_record(timesteps=100000)

# Analyze episode
episode_data = visualizer.analyze_episode(model, env, render=True)
visualizer.visualize_analysis(episode_data)

# Create action heatmap
visualizer.create_action_heatmap(model, env, num_samples=2000)

# Save frames as gif (requires imageio)
if visualizer.frames:
    import imageio
    imageio.mimsave(f'{visualizer.env_id}_episode.gif', visualizer.frames, fps=30)
```

---

## 10. check_gymnasium_envs.py
**Purpose**: Test Gymnasium environments for compatibility.

### 🟢 Easy Usage
```bash
# Test all ALE/Atari environments
python check_gymnasium_envs.py

# Test with rendering (saves first frame)
python check_gymnasium_envs.py --render

# Test all registered environments
python check_gymnasium_envs.py --all
```

### 🟡 Medium Usage
```bash
# Filter by substring
python check_gymnasium_envs.py --filter Pong

# Render with human window
python check_gymnasium_envs.py --render --mode human

# Render multiple frames
python check_gymnasium_envs.py --render --frames 15 --outdir frames

# Combine filters
python check_gymnasium_envs.py --all --filter mujoco --render
```

### 🔴 Hard Usage
```python
# Extended environment testing
import check_gymnasium_envs as checker
from gymnasium.envs.registration import registry
import gymnasium as gym
import time
import numpy as np

class EnvironmentProfiler:
    """Profile environment performance and characteristics."""

    def __init__(self):
        self.results = {}

    def profile_env(self, env_id, num_episodes=10):
        """Profile environment performance."""

        try:
            env = gym.make(env_id)

            profile = {
                'env_id': env_id,
                'reset_times': [],
                'step_times': [],
                'episode_lengths': [],
                'rewards': [],
                'observation_stats': {},
                'action_stats': {}
            }

            for episode in range(num_episodes):
                # Time reset
                t0 = time.time()
                obs, _ = env.reset()
                profile['reset_times'].append(time.time() - t0)

                done = False
                episode_reward = 0
                episode_length = 0
                step_times = []

                while not done and episode_length < 1000:
                    # Time step
                    t0 = time.time()
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, _ = env.step(action)
                    step_times.append(time.time() - t0)

                    done = terminated or truncated
                    episode_reward += reward
                    episode_length += 1

                profile['step_times'].extend(step_times)
                profile['episode_lengths'].append(episode_length)
                profile['rewards'].append(episode_reward)

            # Calculate statistics
            profile['avg_reset_time'] = np.mean(profile['reset_times'])
            profile['avg_step_time'] = np.mean(profile['step_times'])
            profile['avg_episode_length'] = np.mean(profile['episode_lengths'])
            profile['avg_reward'] = np.mean(profile['rewards'])
            profile['steps_per_second'] = 1.0 / profile['avg_step_time']

            env.close()
            return profile

        except Exception as e:
            return {'env_id': env_id, 'error': str(e)}

    def profile_all(self, env_ids, num_episodes=5):
        """Profile multiple environments."""

        for env_id in env_ids:
            print(f"Profiling {env_id}...")
            self.results[env_id] = self.profile_env(env_id, num_episodes)

        return self.results

    def find_fastest_envs(self, category=None, top_n=10):
        """Find fastest environments."""

        candidates = []

        for env_id, profile in self.results.items():
            if 'error' not in profile:
                if category is None or category in env_id:
                    candidates.append((env_id, profile['steps_per_second']))

        candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"\nTop {top_n} fastest environments:")
        for i, (env_id, sps) in enumerate(candidates[:top_n], 1):
            print(f"{i}. {env_id}: {sps:.0f} steps/second")

        return candidates[:top_n]

    def find_stable_envs(self, max_std_reward=10.0):
        """Find environments with stable rewards."""

        stable = []

        for env_id, profile in self.results.items():
            if 'error' not in profile and 'rewards' in profile:
                std_reward = np.std(profile['rewards'])
                if std_reward <= max_std_reward:
                    stable.append((env_id, std_reward))

        stable.sort(key=lambda x: x[1])

        print(f"\nMost stable environments (std <= {max_std_reward}):")
        for env_id, std in stable[:10]:
            print(f"  {env_id}: std={std:.2f}")

        return stable

# Profile environments
profiler = EnvironmentProfiler()

# Get sample of environments to profile
env_ids = checker.list_env_ids(test_all=False, substring='v5')[:20]

# Profile them
results = profiler.profile_all(env_ids, num_episodes=10)

# Find fastest
fastest = profiler.find_fastest_envs()

# Find most stable
stable = profiler.find_stable_envs(max_std_reward=5.0)

# Save profiling results
import json
with open('environment_profiles.json', 'w') as f:
    json.dump(results, f, indent=2)
```

---

## Quick Reference Commands

### Essential Commands
```bash
# Setup
python extract_gym_metadata.py      # Extract metadata (one-time)
python analyze_environments.py      # Categorize environments

# Quick tests
python benchmark.py --mode quick    # 5 envs, quick test
python visualize_live.py --env Hopper-v5 --train 20000  # Live visualization

# Full benchmark
python benchmark.py --env-suite atari_dense --mode comprehensive --seeds 5
```

### Testing Components
```bash
python parallel_envs.py    # Test vectorization
python callbacks.py        # Test monitoring
python env_selector.py     # Test selection
python algorithm_wrapper.py # Test algorithms
```

### Common Workflows
```bash
# Compare algorithms
python benchmark.py --env-suite quick --algorithm ppo --experiment-name ppo_test
python benchmark.py --env-suite quick --algorithm a2c --experiment-name a2c_test

# Scale testing
for n in 1 5 10 20; do
    python benchmark.py --num-envs $n --experiment-name scale_$n
done

# Visualize specific environment
python visualize_live.py --env Humanoid-v5 --algorithm sac --train 100000
```

## Tips
- Start with `--mode quick` for testing
- Use `--env-suite` for predefined benchmark sets
- Add `--seeds` for statistical robustness
- Check `results/` directory for outputs
- Each file can be run standalone for testing