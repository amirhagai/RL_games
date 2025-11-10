#!/usr/bin/env bash
# Reinforcement Learning environment setup (Ubuntu/Debian)
# GPU target: CUDA 12.4 (per your nvidia-smi)
# Creates and populates venv: ~/venvs/RL

set -euo pipefail

ENV_NAME="RL"
ENV_DIR="${HOME}/venvs/${ENV_NAME}"
ACTIVATE_FILE="${ENV_DIR}/bin/activate"

echo "=== [RL setup] Installing required OS packages (sudo may prompt) ==="
sudo apt-get update
# swig/build-essential -> needed to compile box2d-py when wheels aren't available
# ffmpeg -> video; mesa/osmesa/glfw -> rendering deps for mujoco/atari
sudo apt-get install -y \
  ffmpeg swig build-essential patchelf libgl1-mesa-glx libosmesa6-dev libglfw3

echo "=== [RL setup] Creating virtual environment at ${ENV_DIR} ==="
python3 -m venv "${ENV_DIR}"
# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel

echo "=== [RL setup] Installing PyTorch (CUDA 12.4 wheels) ==="
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch torchvision torchaudio

python - <<'PY'
import torch
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "| Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

echo "=== [RL setup] Installing Gymnasium[all] + MuJoCo + dm-control ==="
# Gymnasium[all] pulls classic-control, box2d, atari, mujoco extras
pip install "gymnasium[all]"
pip install mujoco
pip install dm-control

echo "=== [RL setup] Default headless rendering for MuJoCo (MUJOCO_GL=egl) ==="
if ! grep -q 'MUJOCO_GL' "${ACTIVATE_FILE}"; then
  {
    echo ''
    echo '# ---- RL defaults ----'
    echo '# Headless rendering for MuJoCo by default; override with `export MUJOCO_GL=glfw` before activation for GUI.'
    echo 'export MUJOCO_GL="${MUJOCO_GL:-egl}"'
  } >> "${ACTIVATE_FILE}"
fi

echo "=== [RL setup] Installing Atari ROMs via AutoROM (with retries) ==="
pip install "autorom[accept-rom-license]"
set +e
MAX_TRIES=3
TRY=1
AUTO_SUCCESS=0
while [ $TRY -le $MAX_TRIES ]; do
  echo "AutoROM attempt ${TRY}/${MAX_TRIES} ..."
  AutoROM --accept-license && AUTO_SUCCESS=1 && break
  echo "AutoROM failed on attempt ${TRY}. Retrying in 5s..."
  sleep 5
  TRY=$((TRY+1))
done
set -e
if [ $AUTO_SUCCESS -eq 0 ]; then
  echo "⚠️  AutoROM could not import ROMs after ${MAX_TRIES} attempts."
  echo "    You can retry later inside the venv with:  AutoROM --accept-license"
else
  echo "✅ Atari ROMs installed."
fi

echo "=== [RL setup] Installing baselines & implementations ==="
# IMPORTANT: install Spinning Up WITHOUT its legacy TF1/seaborn pins
pip install --no-deps git+https://github.com/openai/spinningup.git
pip install stable-baselines3 sb3-contrib cleanrl tianshou

echo "=== [RL setup] Installing science / plotting / notebooks ==="
pip install numpy scipy pandas matplotlib seaborn tqdm jupyterlab ipykernel

echo "=== [RL setup] Installing video & images ==="
# Headless OpenCV to avoid GUI deps; switch to `opencv-python` if you need imshow windows
pip install opencv-python-headless imageio imageio-ffmpeg moviepy

echo "=== [RL setup] Installing logging / viz ==="
pip install tensorboard wandb rich

echo "=== [RL setup] Installing multi-agent & robotics extras ==="
pip install pettingzoo supersuit pybullet gymnasium-robotics

echo "=== [RL setup] Optional utilities ==="
pip install numba einops hydra-core tyro

# ---- OPTIONAL: Brax (commented). GPU JAX needs version-matched CUDA/jaxlib wheels. ----
# pip install brax

# ---- OPTIONAL: Isaac Lab (manual; not a one-liner). See official docs: https://isaac-sim.github.io/IsaacLab/ ----
# To leave a reminder in your venv activation, uncomment below:
# if ! grep -q 'ISAAC_LAB_INFO' "${ACTIVATE_FILE}"; then
#   {
#     echo 'export ISAAC_LAB_INFO="See https://isaac-sim.github.io/IsaacLab/ for setup instructions."'
#   } >> "${ACTIVATE_FILE}"
# fi

echo "=== [RL setup] Quick sanity checks ==="
python - <<'PY'
import gymnasium as gym
import mujoco
from dm_control import suite

print("Gymnasium:", gym.__version__)
print("MuJoCo:", mujoco.__version__)
print("dm-control OK:", bool(suite.BENCHMARKING))

# check a few envs
to_check = ["CartPole-v1", "LunarLander-v2", "HalfCheetah-v4"]
for env_id in to_check:
    try:
        env = gym.make(env_id)
        env.reset()
        env.close()
        print(f"{env_id} ✓")
    except Exception as ex:
        print(f"{env_id} ✗ -> {ex}")

# Atari
try:
    env = gym.make("ALE/Pong-v5")
    env.reset()
    env.close()
    print("Atari Pong ✓")
except Exception as e:
    print("Atari Pong ✗ ->", e)
PY

echo
echo "✅ All set."
echo "Activate this env anytime with:"
echo "  source ${ENV_DIR}/bin/activate"
echo "Headless rendering is default (MUJOCO_GL=egl). For GUI, run BEFORE activation:"
echo "  export MUJOCO_GL=glfw && source ${ENV_DIR}/bin/activate"

