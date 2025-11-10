"""
Unified interface for different RL algorithms.
Supports Stable-Baselines3 out of the box, extensible for custom algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np
from pathlib import Path
import gymnasium as gym

class AlgorithmWrapper(ABC):
    """Abstract base class for algorithm wrappers."""

    def __init__(self, env, config: Dict[str, Any], seed: int = 0):
        self.env = env
        self.config = config
        self.seed = seed
        self.total_timesteps = 0

    @abstractmethod
    def train(self, total_timesteps: int, callback: Optional[Callable] = None):
        """Train the algorithm for total_timesteps."""
        pass

    @abstractmethod
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the current policy."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save the model."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load a saved model."""
        pass


class StableBaselines3Wrapper(AlgorithmWrapper):
    """Wrapper for Stable-Baselines3 algorithms."""

    def __init__(self, env, algorithm_name: str, config: Dict[str, Any], seed: int = 0):
        super().__init__(env, config, seed)

        # Import the algorithm
        from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
        from stable_baselines3.common.vec_env import VecNormalize

        algo_map = {
            'ppo': PPO,
            'a2c': A2C,
            'dqn': DQN,
            'sac': SAC,
            'td3': TD3
        }

        if algorithm_name.lower() not in algo_map:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        AlgoClass = algo_map[algorithm_name.lower()]

        # Create the model
        self.model = AlgoClass(
            'MlpPolicy',  # Can be customized via config
            env,
            seed=seed,
            verbose=0,
            **config
        )

        self.algorithm_name = algorithm_name

    def train(self, total_timesteps: int, callback: Optional[Callable] = None):
        """Train using SB3's learn method."""
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False
        )
        self.total_timesteps += total_timesteps

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policy."""
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=num_episodes,
            deterministic=True
        )

        return {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'num_episodes': num_episodes
        }

    def save(self, path: Path):
        """Save the SB3 model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))

    def load(self, path: Path):
        """Load a saved SB3 model."""
        from stable_baselines3 import PPO, A2C, DQN, SAC, TD3

        algo_map = {
            'ppo': PPO,
            'a2c': A2C,
            'dqn': DQN,
            'sac': SAC,
            'td3': TD3
        }

        AlgoClass = algo_map[self.algorithm_name.lower()]
        self.model = AlgoClass.load(str(path), env=self.env)


class CustomAlgorithmWrapper(AlgorithmWrapper):
    """
    Wrapper for custom algorithm implementations.
    Users can extend this for their own algorithms.
    """

    def __init__(self, env, algorithm_class, config: Dict[str, Any], seed: int = 0):
        super().__init__(env, config, seed)
        self.algorithm = algorithm_class(env=env, config=config, seed=seed)

    def train(self, total_timesteps: int, callback: Optional[Callable] = None):
        """Delegate to custom algorithm's train method."""
        self.algorithm.train(total_timesteps, callback=callback)
        self.total_timesteps += total_timesteps

    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Delegate to custom algorithm's evaluate method."""
        return self.algorithm.evaluate(num_episodes=num_episodes)

    def save(self, path: Path):
        """Delegate to custom algorithm's save method."""
        self.algorithm.save(path)

    def load(self, path: Path):
        """Delegate to custom algorithm's load method."""
        self.algorithm.load(path)


def create_algorithm(env, algorithm_name: str, config: Dict[str, Any],
                     seed: int = 0) -> AlgorithmWrapper:
    """
    Factory function to create an algorithm wrapper.

    Args:
        env: Gymnasium environment or vectorized env
        algorithm_name: Name of algorithm ('ppo', 'dqn', etc.)
        config: Algorithm-specific configuration
        seed: Random seed

    Returns:
        Configured AlgorithmWrapper instance
    """
    # Check if it's a Stable-Baselines3 algorithm
    sb3_algorithms = ['ppo', 'a2c', 'dqn', 'sac', 'td3']

    if algorithm_name.lower() in sb3_algorithms:
        return StableBaselines3Wrapper(env, algorithm_name, config, seed)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}. "
                        f"Supported: {sb3_algorithms}")


if __name__ == '__main__':
    # Demo: Create and test algorithm wrapper
    print("Algorithm Wrapper Demo")
    print("=" * 80)

    # Test 1: Create PPO algorithm
    print("\nTest 1: Creating PPO algorithm wrapper")
    env = gym.make('CartPole-v1')

    config = {
        'learning_rate': 3e-4,
        'n_steps': 128,
        'batch_size': 64,
    }

    algo = create_algorithm(env, 'ppo', config, seed=42)
    print(f"  Created: {algo.algorithm_name.upper()} wrapper")
    print(f"  Environment: {env.spec.id}")
    print(f"  Config: {config}")

    # Test 2: Train for a few steps
    print("\nTest 2: Training for 1000 steps")
    algo.train(total_timesteps=1000)
    print(f"  Total timesteps trained: {algo.total_timesteps}")

    # Test 3: Evaluate
    print("\nTest 3: Evaluating policy (3 episodes)")
    results = algo.evaluate(num_episodes=3)
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Episodes: {results['num_episodes']}")

    # Test 4: Save and load
    print("\nTest 4: Save and load model")
    save_path = Path('test_model')
    algo.save(save_path)
    print(f"  Saved to: {save_path}.zip")

    # Create new wrapper and load
    algo2 = create_algorithm(env, 'ppo', config, seed=42)
    algo2.load(save_path)
    print(f"  Loaded successfully")

    # Evaluate loaded model
    results2 = algo2.evaluate(num_episodes=3)
    print(f"  Mean reward after loading: {results2['mean_reward']:.2f}")

    # Clean up
    import os
    if (save_path.parent / f"{save_path.name}.zip").exists():
        (save_path.parent / f"{save_path.name}.zip").unlink()
        print(f"  Cleaned up test files")

    env.close()

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
