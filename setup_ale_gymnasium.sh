    #!/usr/bin/env bash
    # setup_ale_gymnasium.sh
    # Fully reproducible setup for Gymnasium + ALE Atari on Linux/macOS.
    # Creates a virtualenv, installs gymnasium, ale-py, AutoROM with license acceptance,
    # and auto-registers ALE envs via sitecustomize.py so `gym.make("ALE/Pong-v5")` works
    # without extra imports in every Python process.
    #
    # Usage:
    #   chmod +x setup_ale_gymnasium.sh
    #   ./setup_ale_gymnasium.sh
    #
    # Optional env vars:
    #   ENV_DIR=./.venvs/rl     # location for the venv (default as below)
    set -euo pipefail

    echo "=== Gymnasium + ALE Atari setup ==="

    : "${ENV_DIR:=${PWD}/.venvs/rl}"
    PYTHON_BIN="${PYTHON_BIN:-python3}"

    echo "[1/6] Creating venv at: ${ENV_DIR}"
    mkdir -p "$(dirname "${ENV_DIR}")"
    if [ ! -d "${ENV_DIR}" ]; then
      "${PYTHON_BIN}" -m venv "${ENV_DIR}"
    fi

    # shellcheck disable=SC1091
    source "${ENV_DIR}/bin/activate"
    echo "Activated venv: ${VIRTUAL_ENV}"

    echo "[2/6] Upgrading pip/setuptools/wheel"
    python -m pip install --upgrade pip setuptools wheel

    echo "[3/6] Installing pinned packages"
    # Pin versions known to work together. Adjust pins if you know a newer combo works for you.
    python -m pip install       "gymnasium==1.2.2"       "ale-py==0.11.2"       "autorom==0.6.1"

    echo "[4/6] Auto-register ALE envs on interpreter startup (sitecustomize.py)"
    python - <<'PY'
import site, os, textwrap, sys
site_pkgs = site.getsitepackages()[0]
sc_path = os.path.join(site_pkgs, "sitecustomize.py")
content = textwrap.dedent('''
    # Auto-register ALE Gymnasium envs on interpreter startup
    try:
        import gymnasium as gym, ale_py
        gym.register_envs(ale_py)
    except Exception:
        # Do not fail interpreter startup if ALE isn't available
        pass
''')
with open(sc_path, "w", encoding="utf-8") as f:
    f.write(content)
print(f"[sitecustomize] wrote: {sc_path}")
PY

    echo "[5/6] Installing Atari ROMs via AutoROM (accepting license)"
    # This downloads and installs ROMs into the AutoROM package directory.
    python -m AutoROM --accept-license

    echo "[6/6] Sanity check: make, reset, one step in ALE/Pong-v5"
    python - <<'PY'
import gymnasium as gym
import numpy as np

# Thanks to sitecustomize, ALE envs are registered automatically.
env = gym.make("ALE/Pong-v5", frameskip=4)
obs, info = env.reset(seed=0)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
env.close()
print("OK: Pong observation shape:", getattr(obs, "shape", None), "reward:", reward)
PY

    echo "=== Setup complete ==="
    echo "To use this environment in a new shell:"
    echo "  source \"${ENV_DIR}/bin/activate\""
    echo "Then try: python -c 'import gymnasium as gym; env=gym.make("ALE/Breakout-v5"); env.reset(); env.close(); print("Breakout OK")'"
