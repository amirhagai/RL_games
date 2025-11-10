"""
Configuration system for different benchmark modes.
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any
import json
from pathlib import Path

@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Mode settings
    mode: str  # 'quick', 'standard', 'comprehensive'

    # Environment selection
    env_ids: Optional[List[str]] = None
    env_suite: Optional[str] = None  # e.g., 'atari_dense', 'mujoco_locomotion'
    num_envs: Optional[int] = None

    # Training parameters
    total_timesteps: int = 100_000
    num_seeds: int = 1  # Number of random seeds per env
    eval_freq: int = 10_000  # Evaluate every N steps
    eval_episodes: int = 10

    # Parallelization
    vectorize: int = 4  # Number of vectorized envs per env_id
    num_workers: Optional[int] = None  # Auto-detect if None

    # Algorithm
    algorithm: str = 'ppo'  # 'ppo', 'dqn', 'sac', 'a2c', etc.
    algorithm_kwargs: Dict[str, Any] = field(default_factory=dict)

    # Logging
    log_dir: str = 'logs'
    tensorboard: bool = True
    save_checkpoints: bool = True
    checkpoint_freq: int = 50_000

    # Output
    results_dir: str = 'results'
    experiment_name: Optional[str] = None

    def __post_init__(self):
        # Set defaults based on mode
        if self.mode == 'quick':
            self.total_timesteps = min(self.total_timesteps, 10_000)
            self.num_seeds = 1
            self.eval_freq = 5_000
            self.eval_episodes = 3
            self.save_checkpoints = False

        elif self.mode == 'standard':
            self.total_timesteps = max(self.total_timesteps, 100_000)
            self.num_seeds = max(self.num_seeds, 3)
            self.eval_freq = 10_000
            self.eval_episodes = 10

        elif self.mode == 'comprehensive':
            self.total_timesteps = max(self.total_timesteps, 1_000_000)
            self.num_seeds = max(self.num_seeds, 5)
            self.eval_freq = 50_000
            self.eval_episodes = 20

    def save(self, path: Path):
        """Save configuration to JSON."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        """Load configuration from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Predefined configurations
PRESET_CONFIGS = {
    'quick': BenchmarkConfig(
        mode='quick',
        total_timesteps=10_000,
        num_seeds=1,
        vectorize=2,
        eval_freq=5_000,
        eval_episodes=3
    ),

    'standard': BenchmarkConfig(
        mode='standard',
        total_timesteps=100_000,
        num_seeds=3,
        vectorize=4,
        eval_freq=10_000,
        eval_episodes=10
    ),

    'comprehensive': BenchmarkConfig(
        mode='comprehensive',
        total_timesteps=1_000_000,
        num_seeds=5,
        vectorize=8,
        eval_freq=50_000,
        eval_episodes=20
    )
}


if __name__ == '__main__':
    # Demo: Create and save/load configurations
    print("Benchmark Configuration Demo")
    print("=" * 80)

    # Test 1: Quick mode
    print("\nTest 1: Quick mode configuration")
    quick_config = PRESET_CONFIGS['quick']
    print(f"  Mode: {quick_config.mode}")
    print(f"  Total timesteps: {quick_config.total_timesteps:,}")
    print(f"  Num seeds: {quick_config.num_seeds}")
    print(f"  Vectorize: {quick_config.vectorize}")
    print(f"  Eval freq: {quick_config.eval_freq:,}")
    print(f"  Save checkpoints: {quick_config.save_checkpoints}")

    # Test 2: Standard mode
    print("\nTest 2: Standard mode configuration")
    standard_config = PRESET_CONFIGS['standard']
    print(f"  Mode: {standard_config.mode}")
    print(f"  Total timesteps: {standard_config.total_timesteps:,}")
    print(f"  Num seeds: {standard_config.num_seeds}")
    print(f"  Vectorize: {standard_config.vectorize}")
    print(f"  Eval freq: {standard_config.eval_freq:,}")

    # Test 3: Comprehensive mode
    print("\nTest 3: Comprehensive mode configuration")
    comp_config = PRESET_CONFIGS['comprehensive']
    print(f"  Mode: {comp_config.mode}")
    print(f"  Total timesteps: {comp_config.total_timesteps:,}")
    print(f"  Num seeds: {comp_config.num_seeds}")
    print(f"  Vectorize: {comp_config.vectorize}")
    print(f"  Eval freq: {comp_config.eval_freq:,}")

    # Test 4: Custom configuration
    print("\nTest 4: Custom configuration")
    custom_config = BenchmarkConfig(
        mode='standard',
        env_suite='atari_dense',
        algorithm='dqn',
        total_timesteps=500_000,
        num_seeds=5,
        experiment_name='atari_dqn_test'
    )
    print(f"  Mode: {custom_config.mode}")
    print(f"  Env suite: {custom_config.env_suite}")
    print(f"  Algorithm: {custom_config.algorithm}")
    print(f"  Total timesteps: {custom_config.total_timesteps:,}")
    print(f"  Experiment name: {custom_config.experiment_name}")

    # Test 5: Save and load
    print("\nTest 5: Save and load configuration")
    test_path = Path('test_config.json')
    custom_config.save(test_path)
    print(f"  Saved to: {test_path}")

    loaded_config = BenchmarkConfig.load(test_path)
    print(f"  Loaded successfully")
    print(f"  Mode: {loaded_config.mode}")
    print(f"  Algorithm: {loaded_config.algorithm}")
    print(f"  Total timesteps: {loaded_config.total_timesteps:,}")

    # Clean up
    test_path.unlink()
    print(f"  Cleaned up test file")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
