# RL Benchmarking System

A comprehensive, scalable reinforcement learning benchmarking system for testing algorithms across multiple Gymnasium environments.

## Status: Core Implementation Complete ✓

**Phases 0-4 (Complete)**: The core benchmarking infrastructure is fully implemented and functional.

**Phases 5-6 (Optional)**: Results analysis, visualization, and convenience utilities remain to be implemented but are not required for basic benchmarking.

## What's Been Implemented

### Phase 0: Gymnasium Metadata Extraction ✓
- `extract_gym_metadata.py` - Extracts metadata from all 395 working Gymnasium environments
- Uses official Gymnasium API (env.spec, observation_space, action_space)
- Outputs: `gym_metadata_raw.json`

### Phase 1: Environment Categorization ✓
- `analyze_environments.py` - Categorizes environments with research-backed difficulty estimation
  - **Categories**: atari (352), mujoco (23), classic_control (6), box2d (3), toy_text (8), etc.
  - **Difficulty**: Based on RL literature (Atari sparse/dense rewards, MuJoCo DOF complexity)
  - Outputs: `env_metadata.json`

- `env_selector.py` - Environment selection utilities with predefined benchmark suites:
  - `quick` - 5 envs (5-10 minutes)
  - `classic_control` - 5 envs
  - `atari_dense` - 8 envs (easy Atari games)
  - `atari_sparse` - 6 envs (hard exploration)
  - `mujoco_easy/medium/hard` - Categorized by DOF
  - `publication` - 14 commonly used envs
  - `diverse_sample` - 6 envs across all categories

### Phase 2: Parallel Infrastructure ✓
- `parallel_envs.py` - Vectorized environment management
  - Uses official `gymnasium.make_vec()` API
  - Supports async/sync vectorization
  - Resource monitoring (CPU, GPU, memory)

- `benchmark_config.py` - Configuration management
  - Preset configs: quick, standard, comprehensive
  - Customizable: timesteps, seeds, algorithm, parallelization

### Phase 3: Algorithm Integration ✓
- `algorithm_wrapper.py` - Unified interface for RL algorithms
  - Stable-Baselines3 support (PPO, A2C, DQN, SAC, TD3)
  - Extensible for custom algorithms
  - Factory pattern for easy instantiation

- `callbacks.py` - Training monitoring
  - Periodic evaluation
  - Checkpointing
  - Metrics logging
  - Time tracking

### Phase 4: Main Benchmark Runner ✓
- `benchmark.py` - Orchestrates benchmark runs
  - Parallel execution across multiple environments
  - Multiple seeds per environment
  - Comprehensive results tracking
  - Command-line interface

## Quick Start

### 1. Setup
Ensure you have the required dependencies:
```bash
pip install gymnasium stable-baselines3 numpy psutil
pip install "gymnasium[atari,accept-rom-license]"  # For Atari
pip install "gymnasium[mujoco]"  # For MuJoCo
```

### 2. Extract Environment Metadata (One-time)
```bash
python extract_gym_metadata.py
python analyze_environments.py
```

This creates:
- `gym_metadata_raw.json` - Raw metadata from 395 environments
- `env_metadata.json` - Categorized and analyzed metadata

### 3. Run a Quick Benchmark
```bash
# Test PPO on 5 quick environments (CartPole, Acrobot, MountainCar, Pendulum, LunarLander)
python benchmark.py --mode quick --algorithm ppo

# Test on specific environments
python benchmark.py --env-ids CartPole-v1 Acrobot-v1 --algorithm ppo --timesteps 50000

# Test on a predefined suite
python benchmark.py --env-suite atari_dense --algorithm dqn --mode standard

# Test with multiple seeds for reproducibility
python benchmark.py --env-suite quick --seeds 3 --algorithm ppo
```

### 4. Results
Results are saved to `results/TIMESTAMP_MODE_ALGORITHM/`:
- `config.json` - Configuration used
- `environments.json` - List of environments tested
- `results.json` - Aggregated results
- `ENV_NAME/seed_N/` - Per-environment results
  - `evaluations.json` - Evaluation history
  - `final_model.zip` - Trained model
  - `checkpoints/` - Intermediate checkpoints

## Usage Examples

### Example 1: Quick Sanity Check (5-10 minutes)
```bash
python benchmark.py --mode quick --algorithm ppo
```
Tests PPO on 5 diverse environments with 10k timesteps each.

### Example 2: Compare Algorithms
```bash
# Test PPO
python benchmark.py --env-suite classic_control --algorithm ppo --experiment-name ppo_test

# Test A2C
python benchmark.py --env-suite classic_control --algorithm a2c --experiment-name a2c_test

# Compare results in results/ directory
```

### Example 3: Comprehensive Atari Benchmark
```bash
python benchmark.py --env-suite atari_dense --mode comprehensive --seeds 5 --algorithm dqn
```
Tests DQN on 8 Atari games with 1M timesteps and 5 seeds each.

### Example 4: Progressive Testing (1→5→10→20 envs)
```bash
# Start small
python benchmark.py --num-envs 1 --algorithm ppo --experiment-name progressive_1

# Scale up
python benchmark.py --num-envs 5 --algorithm ppo --experiment-name progressive_5
python benchmark.py --num-envs 10 --algorithm ppo --experiment-name progressive_10
python benchmark.py --num-envs 20 --algorithm ppo --experiment-name progressive_20
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: Metadata Extraction                                │
│ • extract_gym_metadata.py → gym_metadata_raw.json          │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Phase 1: Environment Categorization                         │
│ • analyze_environments.py → env_metadata.json              │
│ • env_selector.py (benchmark suites)                        │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Phase 2: Parallel Infrastructure                            │
│ • parallel_envs.py (vectorization)                          │
│ • benchmark_config.py (configuration)                       │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Phase 3: Algorithm Integration                              │
│ • algorithm_wrapper.py (SB3 interface)                      │
│ • callbacks.py (monitoring)                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────┐
│ Phase 4: Main Runner                                        │
│ • benchmark.py (orchestration)                              │
│   → Parallel execution                                      │
│   → Multiple seeds                                          │
│   → Results aggregation                                     │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

✓ **Progressive Scaling**: Start with 1 env, scale to 5→10→20→100+
✓ **Research-Backed**: Difficulty based on RL literature (Bellemare et al., Todorov et al.)
✓ **Parallel Execution**: Runs multiple environments concurrently
✓ **Vectorization**: Uses official `gymnasium.make_vec()` for efficiency
✓ **Multi-Seed Support**: Statistical robustness with multiple random seeds
✓ **Comprehensive Logging**: Evaluation history, checkpoints, metrics
✓ **Flexible Configuration**: Quick/standard/comprehensive presets + custom configs
✓ **Algorithm Agnostic**: Supports all SB3 algorithms, extensible for custom ones

## Files Overview

| File | Purpose | Status |
|------|---------|--------|
| `extract_gym_metadata.py` | Extract metadata from Gymnasium | ✓ |
| `analyze_environments.py` | Categorize and analyze envs | ✓ |
| `env_selector.py` | Environment selection utilities | ✓ |
| `parallel_envs.py` | Vectorization and parallelization | ✓ |
| `benchmark_config.py` | Configuration management | ✓ |
| `algorithm_wrapper.py` | Algorithm interface | ✓ |
| `callbacks.py` | Training callbacks | ✓ |
| `benchmark.py` | Main benchmark runner | ✓ |
| `analyze_results.py` | Results analysis | ⏸ (Phase 5) |
| `visualize.py` | Visualization tools | ⏸ (Phase 5) |
| `compare_experiments.py` | Experiment comparison | ⏸ (Phase 5) |
| `progressive_benchmark.py` | Progressive scaling | ⏸ (Phase 6) |
| `quick_test.py` | Quick testing utility | ⏸ (Phase 6) |

## What You Can Do Now

1. **Test a single environment quickly**:
   ```bash
   python benchmark.py --env-ids CartPole-v1 --timesteps 10000
   ```

2. **Test your algorithm implementation**:
   ```bash
   python benchmark.py --mode quick --algorithm ppo
   ```

3. **Run a comprehensive benchmark**:
   ```bash
   python benchmark.py --env-suite atari_dense --mode comprehensive
   ```

4. **Compare different algorithms**:
   ```bash
   python benchmark.py --env-suite classic_control --algorithm ppo --experiment-name exp1
   python benchmark.py --env-suite classic_control --algorithm a2c --experiment-name exp2
   ```

## Next Steps (Optional)

The remaining phases would add:

- **Phase 5**: Results analysis and visualization tools
  - `analyze_results.py` - Statistical analysis
  - `visualize.py` - Learning curves, performance plots
  - `compare_experiments.py` - Side-by-side comparison

- **Phase 6**: Convenience utilities
  - `progressive_benchmark.py` - Automated progressive scaling
  - `quick_test.py` - Fast environment compatibility testing

These are helpful but not required - the core benchmarking system is fully functional!

## Benchmark Suites Reference

- **quick**: CartPole, Acrobot, MountainCar, Pendulum, LunarLander (5 envs)
- **classic_control**: All classic control environments (5 envs)
- **box2d**: LunarLander, BipedalWalker, CarRacing (5 envs)
- **atari_dense**: Pong, Breakout, SpaceInvaders, Qbert, etc. (8 envs)
- **atari_sparse**: MontezumaRevenge, Pitfall, PrivateEye, etc. (6 envs)
- **mujoco_easy**: InvertedPendulum, Reacher, Swimmer (4 envs)
- **mujoco_medium**: Hopper, Walker2d, HalfCheetah (3 envs)
- **mujoco_hard**: Ant, Pusher, Humanoid (4 envs)
- **publication**: 14 commonly cited environments across all categories
- **diverse_sample**: One from each category (6 envs)

## Testing the System

Each implemented file has a built-in test when run directly:

```bash
python extract_gym_metadata.py  # Test metadata extraction
python analyze_environments.py   # Test categorization
python env_selector.py          # Test environment selection
python parallel_envs.py         # Test parallel environments
python benchmark_config.py      # Test configuration
python algorithm_wrapper.py     # Test algorithm wrapper
python callbacks.py             # Test callbacks
```

## License

This benchmarking system was created to facilitate RL research and experimentation.
