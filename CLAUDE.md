# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
RL Benchmarking System - A scalable reinforcement learning benchmarking infrastructure for testing algorithms across 395+ Gymnasium environments. Implements Phases 0-4 (core functionality complete) and evaluation metrics module, with Phases 5-6 (analysis/visualization) pending.

## Key Commands

### Initial Setup (One-time)
```bash
# Activate environment
source ~/venvs/RL/bin/activate

# Extract and categorize all Gymnasium environments
python extract_gym_metadata.py  # Creates gym_metadata_raw.json (395 envs)
python analyze_environments.py  # Creates env_metadata.json with categories
```

### Running Benchmarks
```bash
# Quick test (5 envs, 10k steps, ~5 min)
python benchmark.py --mode quick --algorithm ppo

# Standard benchmark suites
python benchmark.py --env-suite atari_dense --algorithm dqn --mode standard
python benchmark.py --env-suite mujoco_easy --algorithm sac --seeds 3

# Progressive scaling
python benchmark.py --num-envs 1 --algorithm ppo
python benchmark.py --num-envs 5 --algorithm ppo
python benchmark.py --num-envs 20 --algorithm ppo

# Custom experiments
python benchmark.py --env-ids CartPole-v1 Acrobot-v1 --timesteps 50000 --seeds 5
```

### Live Visualization
```bash
# Watch PPO learn on Humanoid in real-time
python visualize_live.py --env Humanoid-v5 --algorithm ppo --train 50000

# Faster demo with simpler environment
python visualize_live.py --env Hopper-v5 --train 20000
```

### Evaluation Module
```bash
# Run comprehensive evaluation demo
python evaluation_demo.py

# Test individual evaluation components
python evaluation/usage/sample_efficiency_usage.py
python evaluation/usage/final_performance_usage.py
python evaluation/usage/stability_metrics_usage.py
python evaluation/usage/iqm_usage.py
```

### Testing Individual Components
```bash
# Each file has built-in tests when run directly
python parallel_envs.py    # Test vectorization
python algorithm_wrapper.py # Test SB3 integration
python callbacks.py        # Test monitoring
python env_selector.py     # Test environment selection
```

## Architecture

### Phase Structure (Implementation Status)
- **Phase 0-1** (✓): Environment discovery and categorization
- **Phase 2** (✓): Parallel infrastructure with vectorization
- **Phase 3** (✓): Algorithm abstraction layer for SB3
- **Phase 4** (✓): Main benchmark orchestration
- **Evaluation Module** (✓): Comprehensive metrics following Agarwal et al. 2021
- **Phase 5** (pending): Results analysis and visualization
- **Phase 6** (pending): Convenience utilities

### Critical Architecture Decisions

1. **Vectorization Compatibility Issue**
   - `parallel_envs.py` uses Gymnasium's `gym.make_vec()` for standalone use
   - `benchmark.py` MUST use SB3's `make_vec_env()` for algorithm compatibility
   - This is intentional - SB3 doesn't recognize Gymnasium's AsyncVectorEnv

2. **Environment Categorization**
   - 352 Atari envs categorized by sparse/dense rewards (Bellemare et al.)
   - 23 MuJoCo envs categorized by DOF complexity (Todorov et al.)
   - Research-backed difficulty estimation in `analyze_environments.py`

3. **Parallel Execution Model**
   - Uses ProcessPoolExecutor for environment-level parallelism
   - Each (env_id, seed) pair runs in separate process
   - Vectorization happens within each process (default: 4 envs)

4. **Evaluation Architecture**
   - Modular design: separate files for each metric type
   - Master `RLEvaluator` class in `evaluation/__init__.py`
   - Uses IQM (Interquartile Mean) for robust statistics
   - Bootstrap confidence intervals for all metrics

5. **Results Structure**
   ```
   results/
   └── TIMESTAMP_MODE_ALGORITHM/
       ├── config.json              # Full configuration
       ├── environments.json        # List of tested envs
       ├── results.json            # Aggregated results
       └── ENV_NAME/seed_N/       # Per-experiment data
           ├── evaluations.json    # Training history
           ├── final_model.zip     # Trained model
           └── checkpoints/        # Intermediate saves
   ```

## Evaluation Module (`evaluation/`)

Key metrics implemented following literature best practices:

### Sample Efficiency Metrics (`sample_efficiency.py`)
- Time to threshold, convergence detection, jumpstart performance
- Learning rate analysis, relative efficiency comparisons

### Final Performance (`final_performance.py`)
- Confidence intervals, best/worst seed analysis
- Human-normalized scores (Atari), optimality gap

### Stability Metrics (`stability_metrics.py`)
- Catastrophic failure detection, monotonicity score
- Signal-to-noise ratio, plateau detection, cross-seed correlation

### IQM and Robust Statistics (`iqm.py`)
- Interquartile Mean (robust to outliers)
- Trimmed/Winsorized means, MAD, outlier detection
- Bootstrap confidence intervals

Usage:
```python
from evaluation import RLEvaluator, quick_evaluate
rewards = your_training_rewards  # [n_evaluations] or [n_seeds, n_evaluations]
results = quick_evaluate(rewards, metric='all')
```

## Benchmark Suites (env_selector.py)

Predefined suites in BENCHMARK_SUITES dict:
- `quick`: 5 diverse envs for sanity checks
- `atari_dense`: 8 easy Atari games (Pong, Breakout, etc.)
- `atari_sparse`: 6 hard exploration games (MontezumaRevenge, Pitfall, etc.)
- `mujoco_easy`: Simple control (InvertedPendulum, Reacher)
- `mujoco_medium`: Locomotion (Hopper, Walker2d, HalfCheetah)
- `mujoco_hard`: Complex (Ant, Humanoid - 17 DOF)
- `publication`: 14 commonly cited environments

## Algorithm Support

Supported SB3 algorithms (algorithm_wrapper.py):
- PPO (default, good for most tasks)
- A2C (faster but less stable than PPO)
- DQN (discrete actions only)
- SAC (continuous actions, sample efficient)
- TD3 (continuous actions, more stable than SAC)

## Configuration Presets

Three modes in benchmark_config.py:
- `quick`: 10k timesteps, 1 seed, no checkpoints
- `standard`: 100k timesteps, 3 seeds, checkpoints every 10k
- `comprehensive`: 1M timesteps, 5 seeds, full logging

## Known Issues and Workarounds

1. **"Gym unmaintained" warning**: From SB3's compatibility layer, harmless
2. **GPU warnings for PPO**: PPO prefers CPU for simple envs, still works
3. **pygame deprecation warning**: From Box2D rendering, doesn't affect functionality

## Pending Implementation (Phases 5-6)

Files exist but are not implemented:
- `analyze_results.py` - Statistical analysis of benchmark results
- `visualize.py` - Learning curves and performance plots
- `compare_experiments.py` - Side-by-side algorithm comparison
- `progressive_benchmark.py` - Automated 1→5→10→20→100+ scaling
- `quick_test.py` - Fast compatibility testing

## Data Flow

1. `extract_gym_metadata.py` → `gym_metadata_raw.json`
2. `analyze_environments.py` → `env_metadata.json`
3. `env_selector.py` reads metadata, provides suite selection
4. `benchmark.py` orchestrates training using:
   - `benchmark_config.py` for configuration
   - `algorithm_wrapper.py` for SB3 interface
   - `callbacks.py` for monitoring
5. Results saved to `results/` directory structure
6. `evaluation/` module for post-hoc analysis of results