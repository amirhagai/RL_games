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

    # Create and run benchmark
    runner = BenchmarkRunner(config)
    results = runner.run()

    # Exit with appropriate code
    failed = sum(1 for r in results if r['status'] == 'failed')
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
