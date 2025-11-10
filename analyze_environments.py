"""
Analyze and categorize environments using Phase 0 metadata.
Adds categorization and research-backed difficulty estimation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

def categorize_env(env_id: str, obs_shape: Optional[tuple] = None) -> str:
    """
    Categorize environment by ID patterns and observation shape.

    Args:
        env_id: Environment ID
        obs_shape: Observation space shape (used to detect Atari games)
    """
    if env_id.startswith('ALE/'):
        return 'atari'
    # Atari games have (210, 160, 3) or similar shapes, or "NoFrameskip" in name
    elif obs_shape and len(obs_shape) == 3 and obs_shape[2] == 3 and obs_shape[0] > 100:
        return 'atari'
    elif 'NoFrameskip' in env_id or 'Deterministic' in env_id:
        return 'atari'
    elif any(name in env_id for name in ['Ant', 'HalfCheetah', 'Hopper',
                                           'Humanoid', 'Pusher', 'Reacher',
                                           'Swimmer', 'Walker', 'Inverted']):
        return 'mujoco'
    elif env_id in ['CartPole-v1', 'CartPole-v0', 'Acrobot-v1', 'MountainCar-v0',
                     'MountainCarContinuous-v0', 'Pendulum-v1']:
        return 'classic_control'
    elif 'BipedalWalker' in env_id or 'LunarLander' in env_id or 'CarRacing' in env_id:
        return 'box2d'
    elif 'FrozenLake' in env_id or 'CliffWalking' in env_id or env_id == 'Taxi-v3' or 'Blackjack' in env_id:
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

        # Extract game name from ALE/GameName-v5 or GameNameNoFrameskip-v4
        env_game = env_id.split('/')[-1].split('-')[0]
        # Remove NoFrameskip suffix
        for suffix in ['NoFrameskip', 'Deterministic']:
            if env_game.endswith(suffix):
                env_game = env_game[:-len(suffix)]

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
        if env_id in ['CartPole-v1', 'CartPole-v0', 'Acrobot-v1']:
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
    elif category in ['toy_text', 'tabular', 'physics2d']:
        if 'FrozenLake8x8' in env_id:
            return 'medium'
        else:
            return 'easy'

    # Default
    else:
        return 'medium'

def analyze_metadata(metadata: Dict) -> Dict:
    """Add categorization and difficulty to raw metadata."""
    env_id = metadata['env_id']
    max_steps = metadata['spec']['max_episode_steps']
    reward_threshold = metadata['spec']['reward_threshold']
    obs_shape = tuple(metadata['observation_space']['shape']) if metadata['observation_space']['shape'] else None

    category = categorize_env(env_id, obs_shape)
    difficulty = estimate_difficulty(env_id, category, max_steps, reward_threshold)

    # Add new fields
    metadata['category'] = category
    metadata['estimated_difficulty'] = difficulty

    return metadata

def main():
    """Load Phase 0 metadata, add categorization and difficulty, save results."""

    # Load raw metadata from Phase 0
    input_path = Path('gym_metadata_raw.json')
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run extract_gym_metadata.py first (Phase 0).")
        return 1

    print(f"Loading raw metadata from {input_path}...")
    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    print(f"Analyzing {raw_data['successful']} environments...")

    # Add categorization and difficulty to each environment
    analyzed_envs = []
    for env_meta in raw_data['environments']:
        analyzed = analyze_metadata(env_meta)
        analyzed_envs.append(analyzed)

    # Create output structure
    output = {
        'total_envs': raw_data['total_envs'],
        'successful': raw_data['successful'],
        'failed': raw_data['failed'],
        'failed_envs': raw_data['failed_envs'],
        'environments': analyzed_envs
    }

    # Save results
    output_path = Path('env_metadata.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"  Output: {output_path}")
    print(f"  Analyzed: {len(analyzed_envs)} environments")
    print(f"{'='*80}")

    # Print category summary
    categories = {}
    difficulties = {}
    action_types = {}

    for meta in analyzed_envs:
        cat = meta['category']
        categories[cat] = categories.get(cat, 0) + 1

        diff = meta['estimated_difficulty']
        difficulties[diff] = difficulties.get(diff, 0) + 1

        act_type = meta['action_space']['type']
        action_types[act_type] = action_types.get(act_type, 0) + 1

    print("\nEnvironments by category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat:20s}: {count:3d}")

    print("\nEnvironments by difficulty:")
    difficulty_order = ['easy', 'medium', 'hard', 'very_hard']
    for diff in difficulty_order:
        count = difficulties.get(diff, 0)
        if count > 0:
            print(f"  {diff:20s}: {count:3d}")

    print("\nEnvironments by action space:")
    for act_type, count in sorted(action_types.items(), key=lambda x: -x[1]):
        print(f"  {act_type:20s}: {count:3d}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
