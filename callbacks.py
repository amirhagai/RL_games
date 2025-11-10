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


if __name__ == '__main__':
    # Demo: Test callback with CartPole
    print("Benchmark Callback Demo")
    print("=" * 80)

    import gymnasium as gym
    from stable_baselines3 import PPO
    import tempfile
    import shutil

    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Using temporary directory: {temp_dir}")

    try:
        # Create environments
        print("\nTest 1: Setting up environment and algorithm")
        train_env = gym.make('CartPole-v1')
        eval_env = gym.make('CartPole-v1')

        # Create algorithm
        model = PPO('MlpPolicy', train_env, verbose=0, seed=42)
        print("  ✓ Created PPO model")

        # Create callback
        print("\nTest 2: Creating benchmark callback")
        callback = BenchmarkCallback(
            env_id='CartPole-v1',
            eval_env=eval_env,
            eval_freq=500,  # Evaluate every 500 steps
            eval_episodes=3,
            save_path=temp_dir / 'test_run',
            checkpoint_freq=1000,  # Checkpoint every 1000 steps
            verbose=1
        )
        print("  ✓ Created callback")

        # Train with callback
        print("\nTest 3: Training for 2000 steps with callback")
        model.learn(total_timesteps=2000, callback=callback)
        print("  ✓ Training complete")

        # Check outputs
        print("\nTest 4: Verifying outputs")
        evaluations_file = temp_dir / 'test_run' / 'evaluations.json'
        assert evaluations_file.exists(), "Evaluations file not found"
        print(f"  ✓ Evaluations file exists: {evaluations_file}")

        with open(evaluations_file) as f:
            evaluations = json.load(f)
        print(f"  ✓ Number of evaluations: {len(evaluations)}")
        print(f"  ✓ Final mean reward: {evaluations[-1]['mean_reward']:.2f}")

        final_model = temp_dir / 'test_run' / 'final_model.zip'
        assert final_model.exists(), "Final model not found"
        print(f"  ✓ Final model saved: {final_model}")

        # Test loading the saved model
        print("\nTest 5: Loading saved model")
        loaded_model = PPO.load(temp_dir / 'test_run' / 'final_model', env=train_env)
        print("  ✓ Model loaded successfully")

        # Cleanup
        train_env.close()
        eval_env.close()

    finally:
        # Remove temporary directory
        shutil.rmtree(temp_dir)
        print(f"\nCleaned up temporary directory")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
