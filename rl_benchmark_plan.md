# RL Benchmarking System Implementation Plan

## Overview

Build a scalable, flexible RL benchmarking system that can test algorithms across 1 to 395 Gymnasium environments in parallel, with support for quick iteration, comprehensive evaluation, and everything in between.

## Current State Analysis

**Available Resources:**
- 395 working Gymnasium environments (all up-to-date versions)
- Single GPU + multicore CPU (ideal for parallel vectorized envs)
- Stable-Baselines3 for standard algorithms
- PyTorch and JAX for custom implementations
- Existing `check_gymnasium_envs.py` for environment validation
- **Access to Gymnasium documentation** for accurate API usage

**Key Discoveries from Gymnasium Documentation:**
- Official `gymnasium.make_vec()` API for vectorization
- `env.spec` provides `max_episode_steps`, `reward_threshold`, `nondeterministic` attributes
- Atari environments available through ALE (100+ games)
- MuJoCo environments organized by locomotion vs manipulation
- Research literature distinguishes Atari by sparse vs dense rewards

**Key Challenges:**
- Need to categorize 395+ diverse environments by type/difficulty (✓ solved with Phase 0)
- Balance between quick iteration and comprehensive benchmarking
- Efficient parallel execution without overwhelming system resources (✓ using `make_vec`)
- Consistent metrics collection across different env types
- Managing results from hundreds of experiment runs

## Desired End State

A modular benchmarking system that supports:

1. **Environment Organization**: Environments categorized by type, difficulty, and characteristics
2. **Flexible Modes**: Quick (minutes), Standard (hours), Comprehensive (days)
3. **Progressive Scaling**: Easy to run 1, 5, 10, 20, or all 395 environments
4. **Algorithm Agnostic**: Works with SB3, custom implementations, any RL algorithm
5. **Automated Analysis**: Collects metrics, generates plots, statistical comparisons
6. **Resumable**: Can pause/resume long benchmark runs

### Success Verification:
```bash
# Quick test: 5 simple envs, 10k steps each (~5 minutes)
python benchmark.py --mode quick --num-envs 5 --algorithm ppo

# Standard test: 20 diverse envs, 100k steps (~2 hours)
python benchmark.py --mode standard --num-envs 20 --algorithm ppo

# Comprehensive: All envs in category, full training (~1 day)
python benchmark.py --mode comprehensive --category atari --algorithm ppo

# Compare two algorithm variants
python benchmark.py --compare experiments/ppo_v1 experiments/ppo_v2
```

## What We're NOT Doing

- Not building a distributed training system (single machine only)
- Not implementing new RL algorithms (focus on benchmarking)
- Not handling multi-agent or offline RL (Gymnasium focus)
- Not building a web UI (command-line + notebooks)

## Implementation Phases

### Updates Based on Gymnasium Documentation Research

This plan has been enhanced with insights from official Gymnasium documentation:

**Phase 0 Added**: Extract real metadata using `env.spec`, `observation_space`, `action_space` APIs
- Provides ground truth data before categorization
- Uses official Gymnasium EnvSpec attributes
- Outputs `gym_metadata_raw.json` for Phase 1

**Research-Backed Difficulty Estimation**:
- **Atari**: Sparse vs Dense reward classification (based on Bellemare et al., 2016)
  - Hard exploration: Montezuma's Revenge, Pitfall, Private Eye (require RND/curiosity)
  - Dense reward: Pong, Breakout, SpaceInvaders (standard RL works well)
- **MuJoCo**: DOF-based difficulty (based on Todorov et al., 2012)
  - Very Hard: Humanoid (17 DOF), Hard: Ant (8 DOF), Medium: HalfCheetah/Walker (6 DOF), Easy: Pendulums
- **Classic Control**: Episode length and stability requirements

**Enhanced Benchmark Suites**:
- `atari_dense` - 8 dense reward games (easy to learn)
- `atari_sparse` - 6 sparse reward games (exploration challenge)
- `mujoco_easy`/`medium`/`hard` - Categorized by DOF and complexity
- `publication` - 13 envs commonly used in RL papers
- `diverse_sample` - 6 envs covering all categories and difficulties

**Official Gymnasium APIs**:
- Use `gymnasium.make_vec()` instead of manual vectorization (Phase 2)
- Leverage `env.spec.max_episode_steps`, `reward_threshold`, `nondeterministic`
- Proper handling of observation/action space metadata

---

## Phase 0: Gymnasium Metadata Extraction (Foundation)

### Overview
Extract real metadata directly from Gymnasium's registry using official API. This provides ground truth about all environments before categorization.

### Changes Required:

#### 1. Metadata Extractor Using Gymnasium API
**File**: `extract_gym_metadata.py`

```python
"""
Extract metadata directly from Gymnasium registry using official API.
This is the foundation - uses actual env.spec data rather than heuristics.
"""

import json
import gymnasium as gym
from gymnasium.envs.registration import registry
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

def extract_env_metadata(env_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract all available metadata for a single environment using Gymnasium API.

    Returns metadata from:
    - env.spec (id, max_episode_steps, reward_threshold, nondeterministic, etc.)
    - env.observation_space (type, shape, bounds)
    - env.action_space (type, n or shape)
    """
    try:
        # Create environment (no rendering needed for metadata)
        env = gym.make(env_id)

        # Extract spec metadata (official Gymnasium EnvSpec)
        spec = env.spec
        metadata = {
            'env_id': env_id,
            'spec': {
                'id': spec.id if spec else env_id,
                'entry_point': str(spec.entry_point) if spec else None,
                'max_episode_steps': spec.max_episode_steps if spec else None,
                'reward_threshold': spec.reward_threshold if spec else None,
                'nondeterministic': spec.nondeterministic if spec else False,
            },

            # Observation space info
            'observation_space': {
                'type': type(env.observation_space).__name__,
                'shape': env.observation_space.shape if hasattr(env.observation_space, 'shape') else None,
                'n': env.observation_space.n if hasattr(env.observation_space, 'n') else None,
            },

            # Action space info
            'action_space': {
                'type': type(env.action_space).__name__,
                'n': env.action_space.n if hasattr(env.action_space, 'n') else None,
                'shape': env.action_space.shape if hasattr(env.action_space, 'shape') else None,
            },

            # Additional useful info
            'is_vectorizable': True,  # Most Gymnasium envs support vectorization
        }

        # Add action space size (useful for algorithm selection)
        if hasattr(env.action_space, 'n'):
            metadata['action_space']['size'] = env.action_space.n
        elif hasattr(env.action_space, 'shape'):
            metadata['action_space']['size'] = int(np.prod(env.action_space.shape))
        else:
            metadata['action_space']['size'] = 1

        env.close()
        return metadata

    except Exception as e:
        print(f"  ✗ Failed to extract metadata for {env_id}: {e}")
        return None

def main():
    """Extract metadata for all registered Gymnasium environments."""

    # Get all registered environments from Gymnasium registry
    all_env_ids = sorted([spec.id for spec in registry.values()])

    print(f"Extracting metadata from {len(all_env_ids)} registered environments...")
    print(f"Using official Gymnasium API: env.spec, observation_space, action_space\n")

    metadata_list = []
    failed = []

    for i, env_id in enumerate(all_env_ids, 1):
        print(f"[{i:>3}/{len(all_env_ids)}] {env_id:<40}", end=' ')

        metadata = extract_env_metadata(env_id)

        if metadata:
            metadata_list.append(metadata)

            # Show key info
            max_steps = metadata['spec']['max_episode_steps']
            action_type = metadata['action_space']['type']
            print(f"✓ (steps={max_steps}, action={action_type})")
        else:
            failed.append(env_id)

    # Save results
    output = {
        'total_envs': len(all_env_ids),
        'successful': len(metadata_list),
        'failed': len(failed),
        'failed_envs': failed,
        'environments': metadata_list
    }

    output_path = Path('gym_metadata_raw.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Metadata extraction complete!")
    print(f"  Total: {len(all_env_ids)}")
    print(f"  Successful: {len(metadata_list)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Output: {output_path}")
    print(f"{'='*80}")

    # Quick summary statistics
    discrete_action = sum(1 for m in metadata_list if m['action_space']['type'] == 'Discrete')
    continuous_action = sum(1 for m in metadata_list if m['action_space']['type'] == 'Box')

    print(f"\nAction Space Types:")
    print(f"  Discrete: {discrete_action}")
    print(f"  Continuous (Box): {continuous_action}")
    print(f"  Other: {len(metadata_list) - discrete_action - continuous_action}")

if __name__ == '__main__':
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Script runs without errors: `python extract_gym_metadata.py`
- [ ] Generates valid JSON: `python -c "import json; json.load(open('gym_metadata_raw.json'))"`
- [ ] Extracts metadata for 395+ environments

#### Manual Verification:
- [ ] Output shows max_episode_steps for known environments (CartPole-v1 should be 500)
- [ ] Action space types are correctly identified (Discrete vs Box)
- [ ] Failed environments list matches known incompatibilities

**Implementation Note**: Run this FIRST before Phase 1. This provides the ground truth data that Phase 1 will categorize and enhance.

---

## Phase 1: Environment Categorization & Research-Backed Difficulty

### Overview
Use the raw metadata from Phase 0 and categorize environments based on research literature. Apply research-backed difficulty classifications for Atari (sparse vs dense reward) and MuJoCo (locomotion difficulty).

### Changes Required:

#### 1. Environment Analyzer Script
**File**: `analyze_environments.py`

```python
"""
Extract metadata from all working Gymnasium environments.
Categorizes by: observation space, action space, domain, difficulty estimate.
"""

import json
import gymnasium as gym
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

@dataclass
class EnvMetadata:
    env_id: str
    category: str  # 'atari', 'mujoco', 'classic_control', 'box2d', 'toy_text'
    observation_space_type: str  # 'box', 'discrete', 'multi_binary', etc.
    observation_shape: tuple
    action_space_type: str
    action_space_size: int
    max_episode_steps: Optional[int]
    reward_threshold: Optional[float]
    estimated_difficulty: str  # 'easy', 'medium', 'hard'
    is_vectorizable: bool
    requires_rendering: bool

def categorize_env(env_id: str) -> str:
    """Categorize environment by ID patterns."""
    if env_id.startswith('ALE/'):
        return 'atari'
    elif any(name in env_id for name in ['Ant', 'HalfCheetah', 'Hopper',
                                           'Humanoid', 'Pusher', 'Reacher',
                                           'Swimmer', 'Walker', 'InvertedPendulum']):
        return 'mujoco'
    elif env_id in ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0',
                     'MountainCarContinuous-v0', 'Pendulum-v1']:
        return 'classic_control'
    elif 'BipedalWalker' in env_id or 'LunarLander' in env_id or 'CarRacing' in env_id:
        return 'box2d'
    elif env_id in ['FrozenLake-v1', 'FrozenLake8x8-v1', 'CliffWalking-v1',
                     'Taxi-v3', 'Blackjack-v1']:
        return 'toy_text'
    elif env_id.startswith('phys2d/'):
        return 'physics2d'
    elif env_id.startswith('tabular/'):
        return 'tabular'
    else:
        return 'other'

def estimate_difficulty(env_id: str, category: str, max_steps: Optional[int],
                        reward_threshold: Optional[float]) -> str:
    """
    Research-backed difficulty estimation.

    Based on RL research literature:
    - Atari: Sparse vs Dense reward environments (Bellemare et al., 2016)
    - MuJoCo: Locomotion complexity (Todorov et al., 2012)
    - Classic Control: Episode length and stability requirements
    """

    # ATARI environments - based on exploration difficulty
    # Research: Montezuma's Revenge, Pitfall, Private Eye require sophisticated exploration
    if category == 'atari':
        # Hard exploration games (sparse rewards, long-horizon dependencies)
        hard_atari = [
            'MontezumaRevenge', 'Pitfall', 'PrivateEye', 'Solaris',
            'Venture', 'Gravitar', 'Skiing'
        ]
        # Easy dense reward games
        easy_atari = [
            'Pong', 'Breakout', 'SpaceInvaders', 'Enduro',
            'BeamRider', 'Qbert', 'Seaquest', 'MsPacman'
        ]

        env_game = env_id.split('/')[-1].split('-')[0]  # Extract game name from ALE/GameName-v5

        if any(game in env_game for game in hard_atari):
            return 'hard'
        elif any(game in env_game for game in easy_atari):
            return 'easy'
        else:
            return 'medium'

    # MUJOCO environments - based on DOF and locomotion complexity
    # Research: Humanoid is hardest due to 17 DOF, Ant (8 DOF), then bipeds
    elif category == 'mujoco':
        if 'Humanoid' in env_id:
            return 'very_hard'  # 17 DOF, most complex
        elif 'Ant' in env_id:
            return 'hard'  # 8 DOF quadruped
        elif any(x in env_id for x in ['HalfCheetah', 'Walker', 'Hopper']):
            return 'medium'  # Bipedal locomotion
        elif any(x in env_id for x in ['Swimmer', 'Reacher', 'Pusher', 'InvertedPendulum']):
            return 'easy'  # Simpler control
        else:
            return 'medium'

    # CLASSIC CONTROL - generally easy for modern algorithms
    elif category == 'classic_control':
        if env_id in ['CartPole-v1', 'Acrobot-v1']:
            return 'easy'  # Can solve in < 100k steps
        elif 'MountainCar' in env_id:
            return 'medium'  # Sparse reward but solvable
        else:
            return 'easy'

    # BOX2D environments
    elif category == 'box2d':
        if 'Hardcore' in env_id:
            return 'hard'
        elif 'BipedalWalker' in env_id or 'CarRacing' in env_id:
            return 'medium'
        else:
            return 'easy'  # LunarLander

    # TOY TEXT - generally easy
    elif category == 'toy_text':
        if 'FrozenLake8x8' in env_id:
            return 'medium'
        else:
            return 'easy'

    # Default
    else:
        return 'medium'

def extract_metadata(env_id: str) -> Optional[EnvMetadata]:
    """Extract all metadata for a single environment."""
    try:
        # Create env without rendering
        env = gym.make(env_id)

        # Get spaces info
        obs_space = env.observation_space
        action_space = env.action_space

        obs_type = type(obs_space).__name__
        obs_shape = obs_space.shape if hasattr(obs_space, 'shape') else (obs_space.n,)

        action_type = type(action_space).__name__
        if hasattr(action_space, 'n'):
            action_size = action_space.n
        elif hasattr(action_space, 'shape'):
            action_size = np.prod(action_space.shape)
        else:
            action_size = 1

        # Get spec info
        spec = env.spec
        max_steps = spec.max_episode_steps if spec else None
        reward_threshold = spec.reward_threshold if spec else None

        category = categorize_env(env_id)

        metadata = EnvMetadata(
            env_id=env_id,
            category=category,
            observation_space_type=obs_type,
            observation_shape=obs_shape,
            action_space_type=action_type,
            action_space_size=action_size,
            max_episode_steps=max_steps,
            reward_threshold=reward_threshold,
            estimated_difficulty=estimate_difficulty(env_id, category, max_steps, reward_threshold),
            is_vectorizable=True,  # Most Gymnasium envs are
            requires_rendering=False  # For benchmarking we don't need rendering
        )

        env.close()
        return metadata

    except Exception as e:
        print(f"Failed to extract metadata for {env_id}: {e}")
        return None

def main():
    """Analyze all environments and save metadata."""
    # Load list of working environments from previous test
    working_envs = []

    # Get all registered environments
    from gymnasium.envs.registration import registry
    all_env_ids = sorted([spec.id for spec in registry.values()])

    print(f"Analyzing {len(all_env_ids)} environments...")

    metadata_list = []
    failed = []

    for i, env_id in enumerate(all_env_ids, 1):
        print(f"[{i}/{len(all_env_ids)}] {env_id}...", end=' ')
        metadata = extract_metadata(env_id)

        if metadata:
            metadata_list.append(asdict(metadata))
            print("✓")
        else:
            failed.append(env_id)
            print("✗")

    # Save results
    output = {
        'total_envs': len(all_env_ids),
        'successful': len(metadata_list),
        'failed': len(failed),
        'failed_envs': failed,
        'environments': metadata_list
    }

    output_path = Path('env_metadata.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nMetadata saved to {output_path}")
    print(f"Successful: {len(metadata_list)}, Failed: {len(failed)}")

    # Print category summary
    categories = {}
    for meta in metadata_list:
        cat = meta['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nEnvironments by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

if __name__ == '__main__':
    main()
```

#### 2. Environment Selection Utilities
**File**: `env_selector.py`

```python
"""
Utilities for selecting environments based on various criteria.
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
                if env['action_space_type'] == action_type]

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
            for cat in categories:
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
            if env['action_space_type'] == target['action_space_type']:
                score += 5.0
            if env['observation_space_type'] == target['observation_space_type']:
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
```

### Success Criteria:

#### Automated Verification:
- [ ] Script runs without errors: `python analyze_environments.py`
- [ ] Generates valid JSON file: `python -c "import json; json.load(open('env_metadata.json'))"`
- [ ] Can import selector: `python -c "from env_selector import EnvironmentSelector; s = EnvironmentSelector()"`

#### Manual Verification:
- [ ] `env_metadata.json` contains metadata for 395+ environments
- [ ] Environments correctly categorized (spot-check Atari, MuJoCo, Box2D)
- [ ] Progressive sets make sense (1->5->10->20 are proper subsets)
- [ ] Diverse sample includes different categories

**Implementation Note**: After automated verification passes, manually review the categorization to ensure it makes sense before proceeding.

---

## Phase 2: Parallel Benchmark Infrastructure

### Overview
Build the core infrastructure for running multiple environments in parallel with proper resource management.

### Changes Required:

#### 1. Vectorized Environment Manager (Using gymnasium.make_vec)
**File**: `parallel_envs.py`

```python
"""
Manage parallel vectorized environments using gymnasium.make_vec().
This is the official Gymnasium API for creating vectorized environments.
"""

import gymnasium as gym
from typing import List, Optional
import multiprocessing as mp
import time

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
        import psutil
        import torch

        snapshot = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3),
        }

        # GPU info if available
        if torch.cuda.is_available():
            snapshot['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
            snapshot['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)

        self.history.append(snapshot)
        return snapshot

    def summary(self) -> dict:
        """Get summary statistics of resource usage."""
        if not self.history:
            return {}

        cpu_usage = [s['cpu_percent'] for s in self.history]
        mem_usage = [s['memory_percent'] for s in self.history]

        summary = {
            'cpu_mean': np.mean(cpu_usage),
            'cpu_max': np.max(cpu_usage),
            'memory_mean': np.mean(mem_usage),
            'memory_max': np.max(mem_usage),
            'duration_seconds': self.history[-1]['timestamp'] - self.history[0]['timestamp']
        }

        if 'gpu_memory_allocated_gb' in self.history[0]:
            gpu_mem = [s['gpu_memory_allocated_gb'] for s in self.history]
            summary['gpu_memory_mean_gb'] = np.mean(gpu_mem)
            summary['gpu_memory_max_gb'] = np.max(gpu_mem)

        return summary
```

#### 2. Benchmark Configuration System
**File**: `benchmark_config.py`

```python
"""
Configuration system for different benchmark modes.
"""

from dataclasses import dataclass, asdict
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
    env_suite: Optional[str] = None  # e.g., 'atari_easy', 'mujoco_locomotion'
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
    algorithm_kwargs: Dict[str, Any] = None

    # Logging
    log_dir: str = 'logs'
    tensorboard: bool = True
    save_checkpoints: bool = True
    checkpoint_freq: int = 50_000

    # Output
    results_dir: str = 'results'
    experiment_name: Optional[str] = None

    def __post_init__(self):
        if self.algorithm_kwargs is None:
            self.algorithm_kwargs = {}

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
```

### Success Criteria:

#### Automated Verification:
- [ ] Import test passes: `python -c "from parallel_envs import ParallelEnvManager; from benchmark_config import BenchmarkConfig"`
- [ ] Can create vectorized envs: Test script that creates 5 parallel CartPole envs
- [ ] Resource monitor works: Test CPU/GPU monitoring for 10 seconds

#### Manual Verification:
- [ ] Parallel environments actually run faster than serial (benchmark with 10 envs)
- [ ] CPU usage stays reasonable (< 90% on average)
- [ ] No memory leaks when running 100+ episodes
- [ ] Configurations save/load correctly

**Implementation Note**: Test with a small number of environments first (e.g., 5 CartPole instances) before scaling up.

---

## Phase 3: Algorithm Integration Layer

### Overview
Create a unified interface for running any RL algorithm (Stable-Baselines3, custom, etc.) with consistent logging and evaluation.

### Changes Required:

#### 1. Algorithm Wrapper Interface
**File**: `algorithm_wrapper.py`

```python
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
```

#### 2. Callbacks for Logging and Checkpointing
**File**: `callbacks.py`

```python
"""
Custom callbacks for training monitoring and checkpointing.
"""

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from pathlib import Path
import json
import time
from typing import Dict, Any, List

class BenchmarkCallback(BaseCallback):
    """
    Unified callback for benchmarking that handles:
    - Periodic evaluation
    - Checkpointing
    - Metrics logging
    - Time tracking
    """

    def __init__(self,
                 env_id: str,
                 eval_env,
                 eval_freq: int,
                 eval_episodes: int,
                 save_path: Path,
                 checkpoint_freq: int,
                 verbose: int = 0):
        super().__init__(verbose)

        self.env_id = env_id
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_path = Path(save_path)
        self.checkpoint_freq = checkpoint_freq

        self.evaluations = []
        self.start_time = None

        # Create directories
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.save_path / 'checkpoints'
        self.checkpoint_dir.mkdir(exist_ok=True)

    def _on_training_start(self):
        """Called at the beginning of training."""
        self.start_time = time.time()

    def _on_step(self) -> bool:
        """Called at each step."""

        # Periodic evaluation
        if self.n_calls % self.eval_freq == 0:
            eval_results = self._evaluate()
            eval_results['timestep'] = self.num_timesteps
            eval_results['wall_time'] = time.time() - self.start_time

            self.evaluations.append(eval_results)

            # Save evaluation history
            with open(self.save_path / 'evaluations.json', 'w') as f:
                json.dump(self.evaluations, f, indent=2)

            if self.verbose > 0:
                print(f"[{self.env_id}] Step {self.num_timesteps}: "
                      f"mean_reward={eval_results['mean_reward']:.2f}")

        # Periodic checkpointing
        if self.checkpoint_freq > 0 and self.n_calls % self.checkpoint_freq == 0:
            checkpoint_path = self.checkpoint_dir / f'model_{self.num_timesteps}_steps'
            self.model.save(checkpoint_path)

            if self.verbose > 0:
                print(f"[{self.env_id}] Checkpoint saved: {checkpoint_path}")

        return True  # Continue training

    def _evaluate(self) -> Dict[str, Any]:
        """Run evaluation episodes."""
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_reward, std_reward = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.eval_episodes,
            deterministic=True
        )

        return {
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'num_episodes': self.eval_episodes
        }

    def _on_training_end(self):
        """Called at the end of training."""
        # Final evaluation
        final_eval = self._evaluate()
        final_eval['timestep'] = self.num_timesteps
        final_eval['wall_time'] = time.time() - self.start_time

        self.evaluations.append(final_eval)

        # Save final results
        with open(self.save_path / 'evaluations.json', 'w') as f:
            json.dump(self.evaluations, f, indent=2)

        # Save final model
        self.model.save(self.save_path / 'final_model')

        if self.verbose > 0:
            print(f"[{self.env_id}] Training complete. "
                  f"Final mean reward: {final_eval['mean_reward']:.2f}")
```

### Success Criteria:

#### Automated Verification:
- [ ] Import test: `python -c "from algorithm_wrapper import create_algorithm"`
- [ ] Can create PPO wrapper: Test creating PPO for CartPole
- [ ] Can train for 1000 steps: Verify training loop works
- [ ] Callbacks execute: Verify evaluation and checkpointing callbacks trigger

#### Manual Verification:
- [ ] Training actually improves reward on CartPole
- [ ] Checkpoints are saved at specified intervals
- [ ] Evaluation results are logged correctly
- [ ] Can load a saved model and continue training

**Implementation Note**: Start with a simple 10k step run on CartPole to verify the entire pipeline works before scaling up.

---

## Phase 4: Main Benchmark Runner

### Overview
Create the main script that orchestrates benchmark runs across multiple environments with all the pieces working together.

### Changes Required:

#### 1. Core Benchmark Runner
**File**: `benchmark.py`

```python
"""
Main benchmark runner script.
Orchestrates training across multiple environments in parallel.
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
from typing import List, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

import gymnasium as gym
from env_selector import EnvironmentSelector, BENCHMARK_SUITES
from benchmark_config import BenchmarkConfig, PRESET_CONFIGS
from parallel_envs import ParallelEnvManager, ResourceMonitor
from algorithm_wrapper import create_algorithm
from callbacks import BenchmarkCallback

def run_single_env_experiment(
    env_id: str,
    config: BenchmarkConfig,
    seed: int,
    output_dir: Path
) -> Dict[str, Any]:
    """
    Run a complete experiment on a single environment with one seed.

    This function is designed to run in a separate process.
    """
    try:
        print(f"[{env_id}][Seed {seed}] Starting experiment...")

        # Create train and eval environments
        train_env = gym.make_vec(env_id, num_envs=config.vectorize, vectorization_mode='async')
        eval_env = gym.make(env_id)

        # Create algorithm
        algorithm = create_algorithm(
            env=train_env,
            algorithm_name=config.algorithm,
            config=config.algorithm_kwargs,
            seed=seed
        )

        # Setup paths
        exp_dir = output_dir / env_id.replace('/', '_') / f'seed_{seed}'
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create callback
        callback = BenchmarkCallback(
            env_id=env_id,
            eval_env=eval_env,
            eval_freq=config.eval_freq,
            eval_episodes=config.eval_episodes,
            save_path=exp_dir,
            checkpoint_freq=config.checkpoint_freq if config.save_checkpoints else 0,
            verbose=1
        )

        # Train
        algorithm.train(
            total_timesteps=config.total_timesteps,
            callback=callback
        )

        # Save final model
        algorithm.save(exp_dir / 'final_model')

        # Get final evaluation
        final_eval = algorithm.evaluate(num_episodes=config.eval_episodes)

        # Cleanup
        train_env.close()
        eval_env.close()

        print(f"[{env_id}][Seed {seed}] ✓ Complete. "
              f"Final reward: {final_eval['mean_reward']:.2f}")

        return {
            'env_id': env_id,
            'seed': seed,
            'status': 'success',
            'final_reward': final_eval['mean_reward'],
            'final_std': final_eval['std_reward']
        }

    except Exception as e:
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        print(f"[{env_id}][Seed {seed}] ✗ Failed: {error_msg}")

        return {
            'env_id': env_id,
            'seed': seed,
            'status': 'failed',
            'error': error_msg
        }


class BenchmarkRunner:
    """Orchestrates benchmark runs across multiple environments."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.selector = EnvironmentSelector()

        # Determine which environments to run
        self.env_ids = self._get_env_ids()

        # Setup output directory
        self.output_dir = self._setup_output_dir()

        print(f"BenchmarkRunner initialized:")
        print(f"  Mode: {config.mode}")
        print(f"  Algorithm: {config.algorithm}")
        print(f"  Environments: {len(self.env_ids)}")
        print(f"  Seeds per env: {config.num_seeds}")
        print(f"  Total experiments: {len(self.env_ids) * config.num_seeds}")
        print(f"  Output: {self.output_dir}")

    def _get_env_ids(self) -> List[str]:
        """Determine which environments to benchmark."""
        if self.config.env_ids:
            return self.config.env_ids
        elif self.config.env_suite:
            if self.config.env_suite in BENCHMARK_SUITES:
                return BENCHMARK_SUITES[self.config.env_suite]
            else:
                raise ValueError(f"Unknown suite: {self.config.env_suite}")
        elif self.config.num_envs:
            return self.selector.diverse_sample(self.config.num_envs, seed=42)
        else:
            # Default: quick benchmark suite
            return BENCHMARK_SUITES['quick']

    def _setup_output_dir(self) -> Path:
        """Create output directory structure."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.config.experiment_name:
            dir_name = f"{timestamp}_{self.config.experiment_name}"
        else:
            dir_name = f"{timestamp}_{self.config.mode}_{self.config.algorithm}"

        output_dir = Path(self.config.results_dir) / dir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(output_dir / 'config.json')

        # Save environment list
        with open(output_dir / 'environments.json', 'w') as f:
            json.dump(self.env_ids, f, indent=2)

        return output_dir

    def run(self):
        """Execute the complete benchmark."""
        print(f"\n{'='*80}")
        print(f"Starting Benchmark Run")
        print(f"{'='*80}\n")

        # Create all (env, seed) pairs
        experiments = [
            (env_id, seed)
            for env_id in self.env_ids
            for seed in range(self.config.num_seeds)
        ]

        # Run experiments in parallel
        max_workers = min(mp.cpu_count() - 1, len(experiments))
        results = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all experiments
            future_to_exp = {
                executor.submit(
                    run_single_env_experiment,
                    env_id,
                    self.config,
                    seed,
                    self.output_dir
                ): (env_id, seed)
                for env_id, seed in experiments
            }

            # Collect results as they complete
            for future in as_completed(future_to_exp):
                env_id, seed = future_to_exp[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[{env_id}][Seed {seed}] Exception: {e}")
                    results.append({
                        'env_id': env_id,
                        'seed': seed,
                        'status': 'failed',
                        'error': str(e)
                    })

        # Save overall results
        self._save_results(results)

        # Print summary
        self._print_summary(results)

        return results

    def _save_results(self, results: List[Dict[str, Any]]):
        """Save aggregated results."""
        results_file = self.output_dir / 'results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {results_file}")

    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print summary of benchmark results."""
        print(f"\n{'='*80}")
        print(f"Benchmark Complete")
        print(f"{'='*80}\n")

        successes = [r for r in results if r['status'] == 'success']
        failures = [r for r in results if r['status'] == 'failed']

        print(f"Total experiments: {len(results)}")
        print(f"Successful: {len(successes)}")
        print(f"Failed: {len(failures)}")

        if successes:
            print(f"\nTop 5 environments by final reward:")
            sorted_results = sorted(successes,
                                   key=lambda x: x['final_reward'],
                                   reverse=True)
            for i, r in enumerate(sorted_results[:5], 1):
                print(f"  {i}. {r['env_id']}: {r['final_reward']:.2f}")

        if failures:
            print(f"\nFailed environments:")
            for r in failures:
                print(f"  - {r['env_id']} (seed {r['seed']})")


def main():
    parser = argparse.ArgumentParser(
        description="Run RL benchmarks across multiple Gymnasium environments"
    )

    # Mode selection
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'standard', 'comprehensive', 'custom'],
                       help='Benchmark mode (default: quick)')

    # Environment selection (mutually exclusive)
    env_group = parser.add_mutually_exclusive_group()
    env_group.add_argument('--num-envs', type=int,
                          help='Number of diverse environments to sample')
    env_group.add_argument('--env-suite', type=str,
                          help=f'Predefined suite: {list(BENCHMARK_SUITES.keys())}')
    env_group.add_argument('--env-ids', nargs='+',
                          help='Specific environment IDs to benchmark')

    # Algorithm
    parser.add_argument('--algorithm', type=str, default='ppo',
                       help='Algorithm to use (default: ppo)')

    # Training parameters
    parser.add_argument('--timesteps', type=int,
                       help='Total timesteps per environment (overrides mode default)')
    parser.add_argument('--seeds', type=int,
                       help='Number of seeds per environment (overrides mode default)')

    # Output
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory for results (default: results)')
    parser.add_argument('--experiment-name', type=str,
                       help='Name for this experiment')

    args = parser.parse_args()

    # Create config
    if args.mode in PRESET_CONFIGS:
        config = PRESET_CONFIGS[args.mode]
    else:
        config = BenchmarkConfig(mode=args.mode)

    # Override with command-line args
    if args.num_envs:
        config.num_envs = args.num_envs
    if args.env_suite:
        config.env_suite = args.env_suite
    if args.env_ids:
        config.env_ids = args.env_ids
    if args.algorithm:
        config.algorithm = args.algorithm
    if args.timesteps:
        config.total_timesteps = args.timesteps
    if args.seeds:
        config.num_seeds = args.seeds
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.experiment_name:
        config.experiment_name = args.experiment_name

    # Run benchmark
    runner = BenchmarkRunner(config)
    runner.run()


if __name__ == '__main__':
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Script runs without errors: `python benchmark.py --mode quick --num-envs 1 --timesteps 1000`
- [ ] Creates output directory with expected structure
- [ ] Generates results.json with valid data
- [ ] Can run with different modes: quick, standard

#### Manual Verification:
- [ ] Actually trains on CartPole for 10k steps in quick mode
- [ ] Parallel execution speeds up multi-env benchmarks
- [ ] Results directory contains all expected files (config, evaluations, models)
- [ ] Can run 5 environments in parallel successfully
- [ ] Progress is printed clearly during execution

**Implementation Note**: Start with `--mode quick --num-envs 1 --timesteps 1000` to verify the pipeline, then scale to `--num-envs 5` before attempting larger benchmarks.

---

## Phase 5: Results Analysis and Visualization

### Overview
Build tools to analyze and visualize benchmark results, compare algorithms, and generate publication-ready plots.

### Changes Required:

#### 1. Results Analyzer
**File**: `analyze_results.py`

```python
"""
Analyze and aggregate benchmark results.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

class BenchmarkAnalyzer:
    """Analyze results from benchmark runs."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.config = self._load_config()
        self.results = self._load_results()
        self.evaluations = self._load_all_evaluations()

    def _load_config(self) -> Dict[str, Any]:
        """Load benchmark configuration."""
        with open(self.results_dir / 'config.json') as f:
            return json.load(f)

    def _load_results(self) -> List[Dict[str, Any]]:
        """Load overall results."""
        with open(self.results_dir / 'results.json') as f:
            return json.load(f)

    def _load_all_evaluations(self) -> Dict[str, List[Dict]]:
        """Load evaluation histories for all environments."""
        evaluations = {}

        for env_result in self.results:
            if env_result['status'] != 'success':
                continue

            env_id = env_result['env_id']
            seed = env_result['seed']

            eval_file = (self.results_dir /
                        env_id.replace('/', '_') /
                        f'seed_{seed}' /
                        'evaluations.json')

            if eval_file.exists():
                with open(eval_file) as f:
                    evals = json.load(f)

                key = f"{env_id}_seed{seed}"
                evaluations[key] = evals

        return evaluations

    def aggregate_by_env(self) -> pd.DataFrame:
        """
        Aggregate results across seeds for each environment.
        Returns DataFrame with mean/std of final rewards.
        """
        env_results = defaultdict(list)

        for result in self.results:
            if result['status'] == 'success':
                env_results[result['env_id']].append(result['final_reward'])

        aggregated = []
        for env_id, rewards in env_results.items():
            aggregated.append({
                'env_id': env_id,
                'mean_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'min_reward': np.min(rewards),
                'max_reward': np.max(rewards),
                'num_seeds': len(rewards)
            })

        df = pd.DataFrame(aggregated)
        return df.sort_values('mean_reward', ascending=False)

    def get_learning_curves(self, env_id: str) -> pd.DataFrame:
        """
        Get learning curves (reward vs timestep) for a specific environment.
        Aggregates across seeds.
        """
        curves = []

        for key, evals in self.evaluations.items():
            if key.startswith(env_id):
                for eval_point in evals:
                    curves.append({
                        'timestep': eval_point['timestep'],
                        'mean_reward': eval_point['mean_reward'],
                        'std_reward': eval_point['std_reward'],
                        'seed': int(key.split('seed')[1])
                    })

        df = pd.DataFrame(curves)

        # Aggregate across seeds
        grouped = df.groupby('timestep').agg({
            'mean_reward': ['mean', 'std'],
            'std_reward': 'mean'
        }).reset_index()

        grouped.columns = ['timestep', 'mean_reward', 'std_across_seeds', 'std_reward']

        return grouped

    def summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the benchmark."""
        successful = [r for r in self.results if r['status'] == 'success']
        failed = [r for r in self.results if r['status'] == 'failed']

        rewards = [r['final_reward'] for r in successful]

        return {
            'total_experiments': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.results) if self.results else 0,
            'mean_reward_overall': np.mean(rewards) if rewards else None,
            'std_reward_overall': np.std(rewards) if rewards else None,
            'median_reward': np.median(rewards) if rewards else None,
            'num_environments': len(set(r['env_id'] for r in self.results)),
            'num_seeds': self.config.get('num_seeds', 1)
        }

    def save_summary(self, output_file: Optional[Path] = None):
        """Save analysis summary to file."""
        if output_file is None:
            output_file = self.results_dir / 'analysis_summary.json'

        summary = {
            'config': self.config,
            'statistics': self.summary_statistics(),
            'env_aggregates': self.aggregate_by_env().to_dict('records')
        }

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Analysis summary saved to {output_file}")
```

#### 2. Visualization Tools
**File**: `visualize.py`

```python
"""
Visualization tools for benchmark results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from analyze_results import BenchmarkAnalyzer

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class BenchmarkVisualizer:
    """Create visualizations for benchmark results."""

    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.analyzer = BenchmarkAnalyzer(results_dir)
        self.output_dir = results_dir / 'plots'
        self.output_dir.mkdir(exist_ok=True)

    def plot_final_rewards(self, save: bool = True):
        """Bar plot of final rewards by environment."""
        df = self.analyzer.aggregate_by_env()

        fig, ax = plt.subplots(figsize=(14, 6))

        x = range(len(df))
        ax.bar(x, df['mean_reward'], yerr=df['std_reward'],
               capsize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(df['env_id'], rotation=45, ha='right')
        ax.set_ylabel('Mean Final Reward')
        ax.set_title(f'Final Rewards Across Environments ({self.analyzer.config["algorithm"]})')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'final_rewards.png', dpi=150)
            print(f"Saved: {self.output_dir / 'final_rewards.png'}")
        else:
            plt.show()

        plt.close()

    def plot_learning_curve(self, env_id: str, save: bool = True):
        """Plot learning curve for a specific environment."""
        df = self.analyzer.get_learning_curves(env_id)

        fig, ax = plt.subplots()

        ax.plot(df['timestep'], df['mean_reward'], label='Mean Reward')
        ax.fill_between(
            df['timestep'],
            df['mean_reward'] - df['std_across_seeds'],
            df['mean_reward'] + df['std_across_seeds'],
            alpha=0.3,
            label='±1 Std Dev'
        )

        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Mean Reward')
        ax.set_title(f'Learning Curve: {env_id}')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()

        if save:
            safe_name = env_id.replace('/', '_')
            plt.savefig(self.output_dir / f'learning_curve_{safe_name}.png', dpi=150)
            print(f"Saved: {self.output_dir / f'learning_curve_{safe_name}.png'}")
        else:
            plt.show()

        plt.close()

    def plot_all_learning_curves(self, save: bool = True):
        """Plot learning curves for all environments in a grid."""
        env_ids = list(set(r['env_id'] for r in self.analyzer.results
                          if r['status'] == 'success'))

        n_envs = len(env_ids)
        n_cols = min(3, n_envs)
        n_rows = (n_envs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_envs == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows > 1 else axes

        for i, env_id in enumerate(env_ids):
            ax = axes[i]
            df = self.analyzer.get_learning_curves(env_id)

            ax.plot(df['timestep'], df['mean_reward'])
            ax.fill_between(
                df['timestep'],
                df['mean_reward'] - df['std_across_seeds'],
                df['mean_reward'] + df['std_across_seeds'],
                alpha=0.3
            )

            ax.set_xlabel('Timesteps')
            ax.set_ylabel('Mean Reward')
            ax.set_title(env_id, fontsize=10)
            ax.grid(alpha=0.3)

        # Hide unused subplots
        for i in range(n_envs, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'all_learning_curves.png', dpi=150)
            print(f"Saved: {self.output_dir / 'all_learning_curves.png'}")
        else:
            plt.show()

        plt.close()

    def plot_reward_distribution(self, save: bool = True):
        """Box plot of reward distributions."""
        # Collect all final rewards grouped by env
        data = []
        for result in self.analyzer.results:
            if result['status'] == 'success':
                data.append({
                    'env_id': result['env_id'],
                    'reward': result['final_reward']
                })

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.boxplot(data=df, x='env_id', y='reward', ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel('Final Reward')
        ax.set_title('Reward Distribution Across Seeds')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save:
            plt.savefig(self.output_dir / 'reward_distribution.png', dpi=150)
            print(f"Saved: {self.output_dir / 'reward_distribution.png'}")
        else:
            plt.show()

        plt.close()

    def generate_all_plots(self):
        """Generate all visualizations."""
        print("Generating visualizations...")

        self.plot_final_rewards()
        self.plot_all_learning_curves()
        self.plot_reward_distribution()

        print(f"\nAll plots saved to: {self.output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualize benchmark results")
    parser.add_argument('results_dir', type=str, help='Path to results directory')
    parser.add_argument('--plot-type', type=str,
                       choices=['final', 'learning', 'all_learning', 'distribution', 'all'],
                       default='all',
                       help='Type of plot to generate')
    parser.add_argument('--env-id', type=str, help='Environment ID for learning curve')

    args = parser.parse_args()

    viz = BenchmarkVisualizer(args.results_dir)

    if args.plot_type == 'final':
        viz.plot_final_rewards()
    elif args.plot_type == 'learning':
        if not args.env_id:
            print("Error: --env-id required for learning curve plot")
            return
        viz.plot_learning_curve(args.env_id)
    elif args.plot_type == 'all_learning':
        viz.plot_all_learning_curves()
    elif args.plot_type == 'distribution':
        viz.plot_reward_distribution()
    elif args.plot_type == 'all':
        viz.generate_all_plots()


if __name__ == '__main__':
    main()
```

#### 3. Comparison Tool
**File**: `compare_experiments.py`

```python
"""
Compare results from multiple benchmark runs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List
from analyze_results import BenchmarkAnalyzer

def compare_experiments(experiment_dirs: List[Path],
                       experiment_names: List[str] = None,
                       output_dir: Path = None):
    """
    Compare multiple benchmark experiments.

    Args:
        experiment_dirs: List of paths to experiment result directories
        experiment_names: Optional names for experiments (defaults to dir names)
        output_dir: Where to save comparison plots
    """
    if experiment_names is None:
        experiment_names = [d.name for d in experiment_dirs]

    if output_dir is None:
        output_dir = Path('comparisons')
    output_dir.mkdir(exist_ok=True)

    # Load all experiments
    analyzers = [BenchmarkAnalyzer(d) for d in experiment_dirs]

    # Get common environments
    env_sets = [set(a.aggregate_by_env()['env_id']) for a in analyzers]
    common_envs = set.intersection(*env_sets)

    print(f"Comparing {len(analyzers)} experiments on {len(common_envs)} common environments")

    # Aggregate results
    comparison_data = []
    for name, analyzer in zip(experiment_names, analyzers):
        df = analyzer.aggregate_by_env()
        df = df[df['env_id'].isin(common_envs)]
        df['experiment'] = name
        comparison_data.append(df)

    combined_df = pd.concat(comparison_data, ignore_index=True)

    # Plot comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    # Group by environment and plot side-by-side bars
    envs = sorted(common_envs)
    x = range(len(envs))
    width = 0.8 / len(experiment_names)

    for i, name in enumerate(experiment_names):
        exp_data = combined_df[combined_df['experiment'] == name]
        exp_data = exp_data.set_index('env_id').loc[envs]

        offset = (i - len(experiment_names)/2) * width + width/2
        ax.bar([xi + offset for xi in x],
               exp_data['mean_reward'],
               width=width,
               label=name,
               yerr=exp_data['std_reward'],
               capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, rotation=45, ha='right')
    ax.set_ylabel('Mean Final Reward')
    ax.set_title('Benchmark Comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison.png', dpi=150)
    print(f"Saved comparison plot: {output_dir / 'comparison.png'}")
    plt.close()

    # Save comparison table
    pivot = combined_df.pivot(index='env_id', columns='experiment', values='mean_reward')
    pivot.to_csv(output_dir / 'comparison_table.csv')
    print(f"Saved comparison table: {output_dir / 'comparison_table.csv'}")

    # Statistical summary
    summary = combined_df.groupby('experiment')['mean_reward'].agg(['mean', 'std', 'min', 'max'])
    summary.to_csv(output_dir / 'summary_stats.csv')
    print(f"Saved summary statistics: {output_dir / 'summary_stats.csv'}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple benchmark experiments")
    parser.add_argument('experiment_dirs', nargs='+', type=str,
                       help='Paths to experiment result directories')
    parser.add_argument('--names', nargs='+', type=str,
                       help='Names for experiments (optional)')
    parser.add_argument('--output-dir', type=str, default='comparisons',
                       help='Output directory for comparison results')

    args = parser.parse_args()

    experiment_dirs = [Path(d) for d in args.experiment_dirs]
    output_dir = Path(args.output_dir)

    compare_experiments(experiment_dirs, args.names, output_dir)


if __name__ == '__main__':
    main()
```

### Success Criteria:

#### Automated Verification:
- [ ] Analysis script runs: `python analyze_results.py <results_dir>`
- [ ] Generates valid summary JSON
- [ ] Visualization script runs: `python visualize.py <results_dir>`
- [ ] Creates PNG files in plots directory

#### Manual Verification:
- [ ] Final rewards plot is readable and informative
- [ ] Learning curves show expected training progression
- [ ] Can compare two different algorithm runs
- [ ] Comparison plots clearly show differences
- [ ] CSV exports contain expected data

**Implementation Note**: Run a quick benchmark first, then test all visualization tools before implementing the comparison tool.

---

## Phase 6: Progressive Scaling Support

### Overview
Add utilities and presets to make progressive scaling (1->5->10->20+) effortless.

### Changes Required:

#### 1. Progressive Benchmark Script
**File**: `progressive_benchmark.py`

```python
"""
Run progressive benchmarks: automatically scale from 1 to N environments.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
from benchmark import BenchmarkRunner
from benchmark_config import BenchmarkConfig
from env_selector import EnvironmentSelector

def run_progressive_benchmark(
    algorithm: str,
    mode: str = 'quick',
    max_envs: int = 100,
    stages: list = None,
    base_config: BenchmarkConfig = None
):
    """
    Run progressive benchmark from 1 env to max_envs.

    Args:
        algorithm: Which algorithm to test
        mode: 'quick', 'standard', or 'comprehensive'
        max_envs: Maximum number of environments
        stages: List of env counts to test (default: [1, 5, 10, 20, 50, 100])
        base_config: Base configuration (optional)
    """
    if stages is None:
        stages = [n for n in [1, 5, 10, 20, 50, 100] if n <= max_envs]

    selector = EnvironmentSelector()
    progressive_sets = selector.progressive_sets()

    # Create timestamped experiment directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_dir = Path('results') / f'{timestamp}_progressive_{algorithm}_{mode}'
    experiment_dir.mkdir(parents=True, exist_ok=True)

    results_summary = []

    for n_envs in stages:
        print(f"\n{'='*80}")
        print(f"Stage: {n_envs} environments")
        print(f"{'='*80}\n")

        # Get environment set for this stage
        env_ids = progressive_sets.get(n_envs, progressive_sets['all'][:n_envs])

        # Create config
        if base_config:
            config = base_config
        else:
            from benchmark_config import PRESET_CONFIGS
            config = PRESET_CONFIGS[mode]

        config.env_ids = env_ids
        config.algorithm = algorithm
        config.experiment_name = f'stage_{n_envs}_envs'
        config.results_dir = str(experiment_dir)

        # Run benchmark
        runner = BenchmarkRunner(config)
        stage_results = runner.run()

        # Aggregate stage results
        successful = [r for r in stage_results if r['status'] == 'success']

        stage_summary = {
            'n_envs': n_envs,
            'total_experiments': len(stage_results),
            'successful': len(successful),
            'mean_reward': sum(r['final_reward'] for r in successful) / len(successful) if successful else None,
            'env_ids': env_ids
        }

        results_summary.append(stage_summary)

        # Save progressive summary
        with open(experiment_dir / 'progressive_summary.json', 'w') as f:
            json.dump(results_summary, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Progressive Benchmark Complete")
    print(f"{'='*80}\n")
    print(f"Results saved to: {experiment_dir}")

    # Print summary table
    print(f"\n{'Envs':<10} {'Success Rate':<15} {'Mean Reward':<15}")
    print('-' * 40)
    for stage in results_summary:
        success_rate = stage['successful'] / stage['total_experiments']
        mean_reward = stage['mean_reward'] if stage['mean_reward'] else 0
        print(f"{stage['n_envs']:<10} {success_rate:<15.2%} {mean_reward:<15.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Run progressive benchmarks (1 -> 5 -> 10 -> ... environments)"
    )

    parser.add_argument('--algorithm', type=str, default='ppo',
                       help='Algorithm to benchmark (default: ppo)')
    parser.add_argument('--mode', type=str, default='quick',
                       choices=['quick', 'standard', 'comprehensive'],
                       help='Benchmark mode (default: quick)')
    parser.add_argument('--max-envs', type=int, default=100,
                       help='Maximum number of environments (default: 100)')
    parser.add_argument('--stages', nargs='+', type=int,
                       help='Custom stages (e.g., --stages 1 3 5 10)')

    args = parser.parse_args()

    run_progressive_benchmark(
        algorithm=args.algorithm,
        mode=args.mode,
        max_envs=args.max_envs,
        stages=args.stages
    )


if __name__ == '__main__':
    main()
```

#### 2. Quick Test Script
**File**: `quick_test.py`

```python
"""
Quick test script to verify an algorithm works before full benchmark.
Runs on CartPole for 10k steps.
"""

import gymnasium as gym
from algorithm_wrapper import create_algorithm

def quick_test(algorithm_name: str = 'ppo', timesteps: int = 10_000):
    """
    Quick test to verify an algorithm works.

    Args:
        algorithm_name: Name of algorithm to test
        timesteps: Number of timesteps to train

    Returns:
        True if test passed, False otherwise
    """
    print(f"Quick test: {algorithm_name} on CartPole-v1 for {timesteps} steps")

    try:
        # Create environment
        env = gym.make('CartPole-v1')

        # Create algorithm
        algo = create_algorithm(
            env=env,
            algorithm_name=algorithm_name,
            config={},
            seed=0
        )

        # Initial evaluation
        print("Initial evaluation...", end=' ')
        initial_eval = algo.evaluate(num_episodes=5)
        print(f"Mean reward: {initial_eval['mean_reward']:.2f}")

        # Train
        print(f"Training for {timesteps} steps...", end=' ')
        algo.train(total_timesteps=timesteps)
        print("Done")

        # Final evaluation
        print("Final evaluation...", end=' ')
        final_eval = algo.evaluate(num_episodes=10)
        print(f"Mean reward: {final_eval['mean_reward']:.2f}")

        # Check if improved
        improved = final_eval['mean_reward'] > initial_eval['mean_reward']

        print(f"\nResult: {'✓ PASS' if improved else '✗ FAIL'}")
        print(f"Improvement: {final_eval['mean_reward'] - initial_eval['mean_reward']:.2f}")

        env.close()

        return improved

    except Exception as e:
        print(f"\n✗ FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Quick algorithm test")
    parser.add_argument('--algorithm', type=str, default='ppo',
                       help='Algorithm to test (default: ppo)')
    parser.add_argument('--timesteps', type=int, default=10_000,
                       help='Training timesteps (default: 10000)')

    args = parser.parse_args()

    success = quick_test(args.algorithm, args.timesteps)
    exit(0 if success else 1)
```

### Success Criteria:

#### Automated Verification:
- [ ] Quick test passes: `python quick_test.py --algorithm ppo`
- [ ] Progressive benchmark runs: `python progressive_benchmark.py --mode quick --max-envs 5`
- [ ] Generates progressive summary JSON

#### Manual Verification:
- [ ] Quick test shows improvement on CartPole
- [ ] Progressive benchmark completes 1, 5 env stages successfully
- [ ] Results are properly organized by stage
- [ ] Can easily verify if an idea works before full benchmark

**Implementation Note**: Always run `quick_test.py` before starting any large benchmark run to catch issues early.

---

## Testing Strategy

### Unit Tests
Create `tests/test_benchmark.py`:
- Test environment selector diverse sampling
- Test config save/load
- Test algorithm wrapper creation
- Test results analyzer loading

### Integration Tests
Create `tests/test_integration.py`:
- End-to-end test: 1 env, 1 seed, 1000 steps
- Verify all output files created
- Verify results can be loaded and analyzed
- Verify plots can be generated

### Manual Testing Steps

1. **Initial Setup**:
   ```bash
   python analyze_environments.py
   python -c "from env_selector import EnvironmentSelector; s = EnvironmentSelector(); print(len(s.diverse_sample(5)))"
   ```

2. **Quick Test**:
   ```bash
   python quick_test.py --algorithm ppo
   ```

3. **Single Environment Benchmark**:
   ```bash
   python benchmark.py --mode quick --env-ids CartPole-v1 --timesteps 10000
   ```

4. **Progressive Scaling**:
   ```bash
   python progressive_benchmark.py --mode quick --max-envs 5 --stages 1 5
   ```

5. **Visualization**:
   ```bash
   python visualize.py results/<latest_dir>
   ```

6. **Comparison**:
   Run two benchmarks with different algorithms, then:
   ```bash
   python compare_experiments.py results/<exp1> results/<exp2> --names "PPO" "A2C"
   ```

## Performance Considerations

### Memory Management
- Use vectorized environments efficiently (4-8 copies per env_id)
- Close environments properly after each experiment
- Monitor memory usage with ResourceMonitor
- For Atari: Consider frame stacking impact on RAM

### CPU Utilization
- Auto-detect CPU cores and reserve 1-2 for system
- Use async vectorization for better parallelism
- Process pool for running multiple env experiments in parallel
- Expected: 80-90% CPU usage during benchmark runs

### GPU Utilization
- PPO/A2C can utilize GPU for policy network
- Single GPU shared across all parallel processes
- DQN/SAC benefit more from GPU (larger replay buffers)
- Expected: 40-60% GPU usage with default configs

### Estimated Runtimes
- **Quick mode** (10k steps, 5 envs, 1 seed): ~5-10 minutes
- **Standard mode** (100k steps, 20 envs, 3 seeds): ~2-4 hours
- **Comprehensive mode** (1M steps, 100 envs, 5 seeds): ~1-2 days

## Example Usage Scenarios

### Scenario 1: Test a new hyperparameter
```bash
# Quick test
python quick_test.py --algorithm ppo

# If passes, run on 5 diverse envs
python benchmark.py --mode quick --num-envs 5 --algorithm ppo --experiment-name "ppo_new_lr"

# Visualize
python visualize.py results/<experiment_dir>
```

### Scenario 2: Compare two algorithm variants
```bash
# Run first variant
python benchmark.py --mode standard --env-suite classic_control --algorithm ppo --experiment-name "ppo_baseline"

# Run second variant (modify config in code or pass kwargs)
python benchmark.py --mode standard --env-suite classic_control --algorithm ppo --experiment-name "ppo_variant"

# Compare
python compare_experiments.py results/<exp1_dir> results/<exp2_dir> --names "Baseline" "Variant"
```

### Scenario 3: Full evaluation for publication
```bash
# Run comprehensive benchmark on major suites
python benchmark.py --mode comprehensive --env-suite classic_control --algorithm ppo --experiment-name "ppo_classic"
python benchmark.py --mode comprehensive --env-suite box2d --algorithm ppo --experiment-name "ppo_box2d"
python benchmark.py --mode comprehensive --env-suite atari_easy --algorithm ppo --experiment-name "ppo_atari"

# Generate all visualizations
python visualize.py results/<exp1_dir>
python visualize.py results/<exp2_dir>
python visualize.py results/<exp3_dir>

# Aggregate results for paper
python analyze_results.py results/<exp1_dir>
```

### Scenario 4: Progressive scaling test
```bash
# Test algorithm generalization
python progressive_benchmark.py --algorithm ppo --mode standard --max-envs 50
```

## Migration Notes

### From Existing Code
Your current `check_gymnasium_envs.py` can be repurposed:
- Use it to verify environments before adding to benchmark
- Integrate its validation logic into `analyze_environments.py`
- Keep it as a standalone diagnostic tool

### Adding Custom Algorithms
To add your own algorithm:

1. Create a class that inherits from an RL base (e.g., PyTorch nn.Module)
2. Implement methods: `train()`, `evaluate()`, `save()`, `load()`
3. Wrap it with `CustomAlgorithmWrapper`
4. Register in `algorithm_wrapper.py`

Example:
```python
class MyCustomAlgorithm:
    def __init__(self, env, config, seed):
        # Your init code
        pass

    def train(self, total_timesteps, callback=None):
        # Your training loop
        pass

    def evaluate(self, num_episodes=10):
        # Return dict with 'mean_reward', 'std_reward'
        pass

    def save(self, path):
        # Save model
        pass

    def load(self, path):
        # Load model
        pass

# Use it:
wrapper = CustomAlgorithmWrapper(env, MyCustomAlgorithm, config, seed)
```

## References

- Environment metadata: `env_metadata.json` (generated in Phase 1)
- Benchmark configurations: `benchmark_config.py`
- All results stored in: `results/<timestamp>_<experiment_name>/`
- Visualization outputs: `results/<experiment_dir>/plots/`

## Success Metrics for Complete System

1. **Functionality**: Can run 1, 5, 10, 20, 50, 100, and all 395 environments
2. **Speed**: 5 envs in quick mode completes in < 10 minutes
3. **Reliability**: < 5% failure rate on working environments
4. **Usability**: Single command to run any benchmark configuration
5. **Analysis**: Automated visualization and comparison tools work
6. **Scalability**: CPU/GPU usage stays reasonable at all scales

---

**Next Steps**: Begin with Phase 1 (Environment Categorization) and test each phase thoroughly before proceeding to the next.
