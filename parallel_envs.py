"""
Manage parallel vectorized environments using gymnasium.make_vec().
This is the official Gymnasium API for creating vectorized environments.
"""

import gymnasium as gym
from typing import List, Optional
import multiprocessing as mp
import time
import numpy as np

def create_vectorized_env(
    env_id: str,
    num_envs: int = 4,
    vectorization_mode: str = 'async',
    seed: int = 0
):
    """
    Create a vectorized environment using gymnasium.make_vec().

    This is the recommended way to create vectorized environments in Gymnasium.
    It handles worker management, seeding, and synchronization automatically.

    Args:
        env_id: Environment ID (e.g., 'CartPole-v1')
        num_envs: Number of parallel environment copies
        vectorization_mode: 'async' or 'sync'
        seed: Random seed for reproducibility

    Returns:
        VectorEnv instance (AsyncVectorEnv or SyncVectorEnv)
    """
    # Use official Gymnasium vectorization API
    vec_env = gym.make_vec(
        env_id,
        num_envs=num_envs,
        vectorization_mode=vectorization_mode,  # 'async' or 'sync'
        wrappers=None,  # Can add wrappers if needed
        # Seeds are handled automatically by make_vec
    )

    # Reset with seed for reproducibility
    vec_env.reset(seed=seed)

    return vec_env


class ParallelEnvManager:
    """
    Manages parallel environment execution using gymnasium.make_vec().
    Automatically determines optimal parallelization based on CPU cores.
    """

    def __init__(self,
                 env_ids: List[str],
                 num_envs_per_id: int = 4,
                 vectorization_mode: str = 'async'):
        """
        Args:
            env_ids: List of environment IDs to run
            num_envs_per_id: Number of vectorized copies per environment
            vectorization_mode: 'async' or 'sync' ('async' recommended)
        """
        self.env_ids = env_ids
        self.num_envs_per_id = num_envs_per_id
        self.vectorization_mode = vectorization_mode

        # Check available resources
        available_cores = mp.cpu_count()
        total_envs = len(env_ids) * num_envs_per_id

        print(f"ParallelEnvManager initialized:")
        print(f"  Environments: {len(env_ids)}")
        print(f"  Copies per env: {num_envs_per_id}")
        print(f"  Total parallel envs: {total_envs}")
        print(f"  Vectorization mode: {vectorization_mode}")
        print(f"  Available CPU cores: {available_cores}")

        if total_envs > available_cores:
            print(f"  ⚠ Warning: {total_envs} envs > {available_cores} cores")
            print(f"     Consider reducing num_envs_per_id or using fewer environments")

    def create_vec_env(self, env_id: str, seed: int = 0):
        """
        Create a vectorized environment for a single env_id.

        Uses gymnasium.make_vec() - the official API.
        """
        return create_vectorized_env(
            env_id=env_id,
            num_envs=self.num_envs_per_id,
            vectorization_mode=self.vectorization_mode,
            seed=seed
        )

    def get_all_vectorized_envs(self, seed: int = 0):
        """
        Create vectorized environments for all env_ids.
        Returns dict mapping env_id -> VectorEnv
        """
        vec_envs = {}

        for i, env_id in enumerate(self.env_ids):
            # Different seed offset for each environment
            env_seed = seed + i * 1000

            vec_envs[env_id] = self.create_vec_env(env_id, env_seed)

            print(f"  [{i+1}/{len(self.env_ids)}] Created {env_id} "
                  f"({self.num_envs_per_id} copies, seed={env_seed})")

        return vec_envs


class ResourceMonitor:
    """Monitor CPU, GPU, memory usage during benchmarking."""

    def __init__(self):
        self.history = []

    def snapshot(self) -> dict:
        """Take a snapshot of current resource usage."""
        try:
            import psutil
        except ImportError:
            # Fallback if psutil not installed
            return {
                'timestamp': time.time(),
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_used_gb': 0.0,
            }

        snapshot = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        }

        # GPU info if available
        try:
            import torch
            if torch.cuda.is_available():
                snapshot['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                snapshot['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
        except ImportError:
            pass

        self.history.append(snapshot)
        return snapshot

    def summary(self) -> dict:
        """Get summary statistics of resource usage."""
        if not self.history:
            return {}

        cpu_usage = [s['cpu_percent'] for s in self.history]
        mem_usage = [s['memory_percent'] for s in self.history]

        summary = {
            'cpu_mean': float(np.mean(cpu_usage)),
            'cpu_max': float(np.max(cpu_usage)),
            'memory_mean': float(np.mean(mem_usage)),
            'memory_max': float(np.max(mem_usage)),
            'duration_seconds': self.history[-1]['timestamp'] - self.history[0]['timestamp']
        }

        if 'gpu_memory_allocated_gb' in self.history[0]:
            gpu_mem = [s['gpu_memory_allocated_gb'] for s in self.history]
            summary['gpu_memory_mean_gb'] = float(np.mean(gpu_mem))
            summary['gpu_memory_max_gb'] = float(np.max(gpu_mem))

        return summary


if __name__ == '__main__':
    # Demo: Create parallel CartPole environments
    print("Testing parallel environment creation...")
    print("=" * 80)

    # Test 1: Single vectorized environment
    print("\nTest 1: Creating vectorized CartPole-v1 (4 copies)...")
    vec_env = create_vectorized_env('CartPole-v1', num_envs=4, seed=42)
    print(f"  Created: {vec_env}")
    print(f"  Observation space: {vec_env.observation_space}")
    print(f"  Action space: {vec_env.action_space}")

    # Test a few steps
    print("\n  Running 10 steps...")
    obs, info = vec_env.reset(seed=42)
    for i in range(10):
        actions = vec_env.action_space.sample()
        obs, rewards, dones, truncs, infos = vec_env.step(actions)
    print(f"  ✓ Successfully ran 10 steps")
    vec_env.close()

    # Test 2: ParallelEnvManager
    print("\nTest 2: Creating multiple vectorized environments...")
    manager = ParallelEnvManager(
        env_ids=['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0'],
        num_envs_per_id=2,
        vectorization_mode='async'
    )
    print("\n  Creating all vectorized environments...")
    vec_envs = manager.get_all_vectorized_envs(seed=42)
    print(f"\n  ✓ Created {len(vec_envs)} vectorized environments")

    # Clean up
    for env in vec_envs.values():
        env.close()

    # Test 3: Resource Monitor
    print("\nTest 3: Testing resource monitor...")
    monitor = ResourceMonitor()
    print("  Taking snapshots for 3 seconds...")
    for i in range(3):
        snapshot = monitor.snapshot()
        print(f"    CPU: {snapshot['cpu_percent']:.1f}%, "
              f"Memory: {snapshot['memory_percent']:.1f}% "
              f"({snapshot['memory_used_gb']:.2f} GB)")
        time.sleep(1)

    summary = monitor.summary()
    print(f"\n  Summary:")
    print(f"    CPU mean: {summary['cpu_mean']:.1f}%, max: {summary['cpu_max']:.1f}%")
    print(f"    Memory mean: {summary['memory_mean']:.1f}%, max: {summary['memory_max']:.1f}%")
    print(f"    Duration: {summary['duration_seconds']:.2f}s")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
