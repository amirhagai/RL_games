"""
Utilities for selecting environments based on various criteria.
Includes research-backed benchmark suites.
"""

import json
import random
from pathlib import Path
from typing import List, Optional, Dict, Any

class EnvironmentSelector:
    """Select environments based on metadata criteria."""

    def __init__(self, metadata_path: str = 'env_metadata.json'):
        with open(metadata_path) as f:
            data = json.load(f)
        self.environments = data['environments']

    def by_category(self, category: str) -> List[str]:
        """Get all environments in a category."""
        return [env['env_id'] for env in self.environments
                if env['category'] == category]

    def by_difficulty(self, difficulty: str) -> List[str]:
        """Get environments by difficulty level."""
        return [env['env_id'] for env in self.environments
                if env['estimated_difficulty'] == difficulty]

    def by_action_space(self, action_type: str) -> List[str]:
        """Get environments with specific action space type."""
        return [env['env_id'] for env in self.environments
                if env['action_space']['type'] == action_type]

    def diverse_sample(self, n: int, seed: Optional[int] = None) -> List[str]:
        """
        Select n diverse environments (one from each category if possible).
        """
        if seed is not None:
            random.seed(seed)

        # Group by category
        by_cat = {}
        for env in self.environments:
            cat = env['category']
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(env['env_id'])

        # Sample evenly across categories
        selected = []
        categories = list(by_cat.keys())

        while len(selected) < n and categories:
            for cat in categories[:]:  # Use slice to allow modification during iteration
                if len(by_cat[cat]) > 0:
                    selected.append(by_cat[cat].pop(random.randrange(len(by_cat[cat]))))
                    if len(selected) >= n:
                        break
                else:
                    categories.remove(cat)

        return selected[:n]

    def similar_to(self, env_id: str, n: int = 5) -> List[str]:
        """Find n environments similar to the given one."""
        target = next((env for env in self.environments if env['env_id'] == env_id), None)
        if not target:
            raise ValueError(f"Environment {env_id} not found")

        # Score similarity based on category, spaces, difficulty
        def similarity_score(env: Dict[str, Any]) -> float:
            score = 0.0
            if env['category'] == target['category']:
                score += 10.0
            if env['action_space']['type'] == target['action_space']['type']:
                score += 5.0
            if env['observation_space']['type'] == target['observation_space']['type']:
                score += 5.0
            if env['estimated_difficulty'] == target['estimated_difficulty']:
                score += 3.0
            return score

        # Sort by similarity
        similar = sorted(
            [env for env in self.environments if env['env_id'] != env_id],
            key=similarity_score,
            reverse=True
        )

        return [env['env_id'] for env in similar[:n]]

    def progressive_sets(self) -> Dict[int, List[str]]:
        """
        Create progressive benchmark sets: 1, 5, 10, 20, 50, 100, all.
        Returns dict mapping size -> env_ids
        """
        sets = {}

        # Start with most representative environments
        all_envs = [env['env_id'] for env in self.environments]

        # 1 env: most iconic/commonly used
        sets[1] = ['CartPole-v1']  # Classic RL benchmark

        # 5 envs: diverse sample
        sets[5] = self.diverse_sample(5, seed=42)

        # Ensure CartPole is in the 5-env set
        if 'CartPole-v1' not in sets[5]:
            sets[5][0] = 'CartPole-v1'

        # 10, 20, 50, 100: increasingly diverse samples
        for n in [10, 20, 50, 100]:
            if n <= len(all_envs):
                sets[n] = self.diverse_sample(n, seed=42)
                # Ensure previous sets are subsets
                for prev_n in [1, 5, 10, 20, 50]:
                    if prev_n < n and prev_n in sets:
                        for env_id in sets[prev_n]:
                            if env_id not in sets[n]:
                                sets[n] = [env_id] + sets[n][:-1]

        # All environments
        sets['all'] = all_envs

        return sets

# Research-backed benchmark suites
# Based on RL literature and Gymnasium documentation
BENCHMARK_SUITES = {
    # Quick sanity check suite (5-10 minutes)
    'quick': [
        'CartPole-v1',           # Easy: Discrete, dense reward
        'Acrobot-v1',            # Easy: Discrete, moderate
        'MountainCar-v0',        # Medium: Sparse reward
        'Pendulum-v1',           # Easy: Continuous control
        'LunarLander-v3'         # Medium: Discrete, physics
    ],

    # Classic control - all environments
    'classic_control': [
        'CartPole-v1', 'Acrobot-v1',
        'MountainCar-v0', 'MountainCarContinuous-v0',
        'Pendulum-v1'
    ],

    # Box2D physics environments
    'box2d': [
        'LunarLander-v3',              # Easy: Discrete
        'LunarLanderContinuous-v3',    # Medium: Continuous
        'BipedalWalker-v3',            # Medium: Continuous locomotion
        'BipedalWalkerHardcore-v3',    # Hard: Obstacles
        'CarRacing-v3'                 # Hard: Vision-based continuous
    ],

    # ATARI - Dense reward (easy to learn)
    # These games provide frequent rewards, suitable for standard RL algorithms
    'atari_dense': [
        'ALE/Pong-v5',              # Easiest Atari benchmark
        'ALE/Breakout-v5',          # Dense reward, well-studied
        'ALE/SpaceInvaders-v5',     # Dense reward, action variety
        'ALE/Qbert-v5',             # Moderate complexity
        'ALE/Seaquest-v5',          # Multiple reward sources
        'ALE/MsPacman-v5',          # Dense navigation rewards
        'ALE/BeamRider-v5',         # Shooting with immediate feedback
        'ALE/Enduro-v5'             # Racing with frequent scoring
    ],

    # ATARI - Sparse reward (exploration challenge)
    # Requires sophisticated exploration strategies (RND, curiosity, etc.)
    'atari_sparse': [
        'ALE/MontezumaRevenge-v5',  # Hardest: Requires key collection, long horizon
        'ALE/Pitfall-v5',           # Hard: Sparse rewards, careful navigation
        'ALE/PrivateEye-v5',        # Hard: Item collection, memory required
        'ALE/Solaris-v5',           # Hard: Complex multi-phase game
        'ALE/Venture-v5',           # Hard: Room exploration
        'ALE/Gravitar-v5'           # Hard: Physics + exploration
    ],

    # MuJoCo - Locomotion (by difficulty)
    # Based on degrees of freedom and control complexity
    'mujoco_easy': [
        'InvertedPendulum-v5',         # 1 DOF pendulum
        'InvertedDoublePendulum-v5',   # 2 DOF double pendulum
        'Reacher-v5',                  # 2 DOF arm reaching
        'Swimmer-v5'                   # 2 DOF swimmer
    ],

    'mujoco_medium': [
        'Hopper-v5',        # 3 DOF biped hopping
        'Walker2d-v5',      # 6 DOF biped walking
        'HalfCheetah-v5'    # 6 DOF quadruped running
    ],

    'mujoco_hard': [
        'Ant-v5',           # 8 DOF quadruped
        'Pusher-v5',        # 7 DOF manipulation
        'Humanoid-v5',      # 17 DOF biped (hardest)
        'HumanoidStandup-v5'  # 17 DOF standing task
    ],

    # Combined MuJoCo suite (all locomotion)
    'mujoco_locomotion': [
        'Ant-v5', 'HalfCheetah-v5', 'Hopper-v5',
        'Humanoid-v5', 'Walker2d-v5', 'Swimmer-v5'
    ],

    # Diverse sample for generalization testing
    'diverse_sample': [
        'CartPole-v1',              # Classic control
        'LunarLander-v3',           # Box2D
        'ALE/Pong-v5',              # Atari dense
        'ALE/MontezumaRevenge-v5',  # Atari sparse
        'Hopper-v5',                # MuJoCo medium
        'Humanoid-v5'               # MuJoCo hard
    ],

    # Publication-quality benchmark (common in papers)
    'publication': [
        # Classic
        'CartPole-v1', 'Acrobot-v1', 'Pendulum-v1',
        # Box2D
        'LunarLander-v3', 'BipedalWalker-v3',
        # Atari (representative sample)
        'ALE/Pong-v5', 'ALE/Breakout-v5', 'ALE/SpaceInvaders-v5',
        'ALE/Qbert-v5', 'ALE/Seaquest-v5',
        # MuJoCo (all difficulties)
        'HalfCheetah-v5', 'Hopper-v5', 'Walker2d-v5', 'Ant-v5'
    ]
}

if __name__ == '__main__':
    # Demo usage
    selector = EnvironmentSelector()

    print("Environment Selector Demo")
    print("=" * 80)

    print(f"\nTotal environments: {len(selector.environments)}")

    print(f"\nAtari environments: {len(selector.by_category('atari'))}")
    print(f"MuJoCo environments: {len(selector.by_category('mujoco'))}")
    print(f"Classic Control environments: {len(selector.by_category('classic_control'))}")

    print(f"\nEasy environments: {len(selector.by_difficulty('easy'))}")
    print(f"Medium environments: {len(selector.by_difficulty('medium'))}")
    print(f"Hard environments: {len(selector.by_difficulty('hard'))}")
    print(f"Very Hard environments: {len(selector.by_difficulty('very_hard'))}")

    print(f"\nDiscrete action space: {len(selector.by_action_space('Discrete'))}")
    print(f"Continuous action space: {len(selector.by_action_space('Box'))}")

    print("\nDiverse sample (5 envs):")
    for env_id in selector.diverse_sample(5, seed=42):
        print(f"  - {env_id}")

    print("\nProgressive sets:")
    prog_sets = selector.progressive_sets()
    for size in sorted([k for k in prog_sets.keys() if isinstance(k, int)]):
        print(f"  {size:3d} envs: {prog_sets[size][:3]}... ({len(prog_sets[size])} total)")

    print("\nBenchmark suites available:")
    for suite_name in sorted(BENCHMARK_SUITES.keys()):
        print(f"  {suite_name:20s}: {len(BENCHMARK_SUITES[suite_name]):3d} envs")
