"""
Live visualization of RL agents running in Gymnasium environments.
Shows real-time rendering of the agent's behavior.
"""

import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
import argparse
import time
from pathlib import Path
import numpy as np

def train_and_visualize(env_id='Humanoid-v5', algorithm='ppo', train_timesteps=1,
                        render_during_training=False, fps=30):
    """
    Train an agent and visualize its performance.

    Args:
        env_id: Environment ID (e.g., 'Humanoid-v5', 'Ant-v5', 'HalfCheetah-v5')
        algorithm: Algorithm to use ('ppo', 'sac', 'a2c')
        train_timesteps: Number of timesteps to train (0 to skip training)
        render_during_training: Whether to render during training (slower)
        fps: Frames per second for visualization
    """
    print(f"{'='*80}")
    print(f"Live Visualization: {algorithm.upper()} on {env_id}")
    print(f"{'='*80}\n")

    # Create environment with rendering
    if render_during_training and train_timesteps > 0:
        # Render during training (much slower)
        env = gym.make(env_id, render_mode='human')
        train_env = make_vec_env(env_id, n_envs=1)  # Single env for training
    else:
        # Train without rendering, then show results
        train_env = make_vec_env(env_id, n_envs=4)  # Vectorized for faster training
        env = gym.make(env_id, render_mode='human')  # For visualization

    # Select algorithm
    algo_map = {
        'ppo': PPO,
        'sac': SAC,
        'a2c': A2C,
        'dqn': DQN
    }

    if algorithm not in algo_map:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    AlgoClass = algo_map[algorithm]

    # Check if continuous or discrete action space
    if isinstance(env.action_space, gym.spaces.Box):
        policy = 'MlpPolicy'  # Continuous actions
    else:
        policy = 'MlpPolicy'  # Discrete actions

    print(f"Environment: {env_id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Max episode steps: {env.spec.max_episode_steps if env.spec else 'Unknown'}")
    print()

    # Train or load model
    model_path = Path(f'models/{env_id}_{algorithm}_model.zip')

    if train_timesteps > 0:
        print(f"Training {algorithm.upper()} for {train_timesteps:,} timesteps...")
        print("(This may take a few minutes for complex environments like Humanoid)\n")

        # Create and train model
        if algorithm == 'sac':
            # SAC only works with continuous actions
            if not isinstance(env.action_space, gym.spaces.Box):
                print("SAC only works with continuous action spaces. Switching to PPO.")
                AlgoClass = PPO
                algorithm = 'ppo'

        model = AlgoClass(
            policy,
            train_env,
            verbose=1,
            device='cpu',  # Use CPU to avoid GPU warnings
            seed=42
        )

        # Train
        model.learn(total_timesteps=train_timesteps)

        # Save model
        model_path.parent.mkdir(exist_ok=True)
        model.save(model_path.stem)
        print(f"\nModel saved to: {model_path}")

    elif model_path.exists():
        print(f"Loading existing model from: {model_path}")
        model = AlgoClass.load(model_path.stem, env=train_env)

    else:
        print(f"No existing model found. Training for {10000} timesteps first...")
        model = AlgoClass(
            policy,
            train_env,
            verbose=1,
            device='cpu',
            seed=42
        )
        model.learn(total_timesteps=10000)
        model_path.parent.mkdir(exist_ok=True)
        model.save(model_path.stem)

    # Close training env
    train_env.close()

    # Visualize trained agent
    print(f"\n{'='*80}")
    print("Visualizing trained agent...")
    print("Press Ctrl+C to stop")
    print(f"{'='*80}\n")

    try:
        episode = 0
        while True:
            episode += 1
            obs, info = env.reset(seed=42 + episode)
            episode_reward = 0
            step_count = 0

            print(f"\nEpisode {episode}")
            print("-" * 40)

            done = False
            while not done:
                # Get action from trained model
                action, _ = model.predict(obs, deterministic=True)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                step_count += 1

                # Control frame rate
                time.sleep(1.0 / fps)

                # Print progress every 50 steps
                if step_count % 50 == 0:
                    print(f"  Step {step_count}: Reward = {episode_reward:.2f}")

            print(f"Episode {episode} finished:")
            print(f"  Total steps: {step_count}")
            print(f"  Total reward: {episode_reward:.2f}")
            print(f"  Average reward per step: {episode_reward/step_count:.3f}")

    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")
    finally:
        env.close()
        print("Environment closed.")


def evaluate_and_visualize(env_id='Humanoid-v5', model_path=None, num_episodes=5, fps=30):
    """
    Load a trained model and visualize its performance.

    Args:
        env_id: Environment ID
        model_path: Path to saved model (if None, trains a quick model)
        num_episodes: Number of episodes to visualize
        fps: Frames per second for visualization
    """
    print(f"{'='*80}")
    print(f"Evaluating and Visualizing: {env_id}")
    print(f"{'='*80}\n")

    # Create environment with rendering
    env = gym.make(env_id, render_mode='human')

    # Load or create model
    if model_path and Path(model_path).exists():
        print(f"Loading model from: {model_path}")
        # Detect algorithm from filename
        if 'ppo' in model_path.lower():
            model = PPO.load(model_path, env=env)
        elif 'sac' in model_path.lower():
            model = SAC.load(model_path, env=env)
        else:
            model = PPO.load(model_path, env=env)  # Default
    else:
        print("No model specified. Training PPO for 10,000 timesteps...")
        train_env = make_vec_env(env_id, n_envs=4)
        model = PPO('MlpPolicy', train_env, verbose=1, device='cpu')
        model.learn(total_timesteps=10000)
        train_env.close()

    # Evaluate with visualization
    print(f"\nVisualizing {num_episodes} episodes...")
    print("Press Ctrl+C to stop early\n")

    try:
        total_rewards = []

        for episode in range(num_episodes):
            obs, info = env.reset(seed=42 + episode)
            episode_reward = 0
            step_count = 0

            print(f"Episode {episode + 1}/{num_episodes}")
            print("-" * 40)

            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                episode_reward += reward
                step_count += 1

                # Control frame rate
                time.sleep(1.0 / fps)

                if step_count % 100 == 0:
                    print(f"  Step {step_count}: Reward = {episode_reward:.2f}")

            total_rewards.append(episode_reward)
            print(f"  Finished: {step_count} steps, Reward = {episode_reward:.2f}\n")

    except KeyboardInterrupt:
        print("\n\nVisualization stopped by user.")

    finally:
        env.close()

        if total_rewards:
            print(f"\nSummary of {len(total_rewards)} episodes:")
            print(f"  Mean reward: {np.mean(total_rewards):.2f}")
            print(f"  Std reward: {np.std(total_rewards):.2f}")
            print(f"  Min reward: {np.min(total_rewards):.2f}")
            print(f"  Max reward: {np.max(total_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Live visualization of RL agents")

    parser.add_argument('--env', type=str, default='Humanoid-v5',
                       help='Environment ID (default: Humanoid-v5)')
    parser.add_argument('--algorithm', type=str, default='ppo',
                       choices=['ppo', 'sac', 'a2c', 'dqn'],
                       help='Algorithm to use (default: ppo)')
    parser.add_argument('--train', type=int, default=5,
                       help='Training timesteps (0 to skip training, default: 50000)')
    parser.add_argument('--episodes', type=int, default=5,
                       help='Number of episodes to visualize (default: 5)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for visualization (default: 30)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to existing model to load')

    args = parser.parse_args()

    if args.model:
        # Load and visualize existing model
        evaluate_and_visualize(
            env_id=args.env,
            model_path=args.model,
            num_episodes=args.episodes,
            fps=args.fps
        )
    else:
        # Train and visualize
        train_and_visualize(
            env_id=args.env,
            algorithm=args.algorithm,
            train_timesteps=args.train,
            render_during_training=False,  # Set to True to see training (very slow)
            fps=args.fps
        )


if __name__ == '__main__':
    main()
