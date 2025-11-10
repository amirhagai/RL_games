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

# Ensure ALE envs are registered even if sitecustomize.py wasn't installed
try:
    import ale_py.gym  # side-effect registers ALE/* with Gymnasium
except Exception:
    pass

def convert_to_native_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_native_types(item) for item in obj]
    return obj

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
        return convert_to_native_types(metadata)

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
