# check_gymnasium_envs.py
# Iterate over Gymnasium envs (default: ALE/*), try make/reset/step, and optionally render.
# Usage:
#   python check_gymnasium_envs.py                      # test ALE/* only, no rendering
#   python check_gymnasium_envs.py --render             # render with default mode=rgb_array, save first frame
#   python check_gymnasium_envs.py --render --mode human  # open native window (if supported) and step a few frames
#   python check_gymnasium_envs.py --all                # test all registered envs
#   python check_gymnasium_envs.py --filter Pong        # substring filter on env-id
#   python check_gymnasium_envs.py --frames 15 --outdir frames  # render N frames, save to dir (rgb_array only)
#
# Exit code 0 if all tested envs succeeded, non-zero otherwise.

import argparse
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym

# Ensure ALE envs are registered even if sitecustomize.py wasn't installed.
try:
    import ale_py.gym  # side-effect registers ALE/* with Gymnasium
except Exception:
    pass

from gymnasium.envs.registration import registry

def list_env_ids(test_all: bool, substring: str | None):
    env_ids = sorted(spec.id for spec in registry.values())
    if not test_all:
        env_ids = [eid for eid in env_ids if eid.startswith("ALE/")]
    if substring:
        env_ids = [eid for eid in env_ids if substring.lower() in eid.lower()]
    return env_ids

def _make_env(eid: str, render_mode: str | None):
    if render_mode:
        try:
            return gym.make(eid, render_mode=render_mode)
        except TypeError:
            # Some envs use "render_mode" but don't accept it at make() time; try without
            return gym.make(eid)
        except Exception:
            # Fallback: try without render_mode
            return gym.make(eid)
    else:
        return gym.make(eid)

def _render_frame(env, render_mode: str | None):
    if not render_mode:
        return None
    try:
        frame = env.render()
        return frame
    except Exception:
        return None

def try_env(eid: str, seed: int, render_mode: str | None, frames: int, outdir: Path | None) -> tuple[bool, str]:
    env = None
    try:
        env = _make_env(eid, render_mode)
    except Exception as e:
        return False, f"make() failed: {e!r}"

    try:
        obs, info = env.reset(seed=seed)
    except Exception as e:
        try:
            env.close()
        except Exception:
            pass
        return False, f"reset() failed: {e!r}"

    # Optionally render multiple frames
    if render_mode == "human":
        try:
            # step a few frames to show movement in a native window
            for _ in range(frames):
                action = env.action_space.sample()
                step_out = env.step(action)
                if len(step_out) != 5:
                    raise RuntimeError(f"unexpected step tuple length {len(step_out)}")
                _ = _render_frame(env, render_mode)  # display on screen
            env.close()
            return True, "ok (rendered human)"
        except Exception as e:
            try:
                env.close()
            except Exception:
                pass
            return False, f"render(human) failed: {e!r}"

    # Standard single step + rgb_array save if requested
    try:
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) != 5:
            raise RuntimeError(f"unexpected step tuple length {len(step_out)}")
    except Exception as e:
        try:
            env.close()
        except Exception:
            pass
        return False, f"step() failed: {e!r}"

    # For rgb_array, save first frame
    if render_mode == "rgb_array":
        try:
            import numpy as np
            from PIL import Image  # pillow needed
            outdir.mkdir(parents=True, exist_ok=True)
            frame = _render_frame(env, render_mode)
            if frame is None:
                msg = "render() returned None"
            else:
                arr = frame
                # Convert to uint8 image
                if hasattr(arr, "dtype"):
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0, 1) * 255).astype("uint8") if arr.dtype.kind == "f" else arr.astype("uint8")
                img = Image.fromarray(arr)
                safe_id = eid.replace("/", "_").replace(":", "_")
                fp = outdir / f"{safe_id}_frame0.png"
                img.save(fp)
                msg = f"saved {fp}"
        except Exception as e:
            msg = f"rgb_array save failed: {e!r}"
    else:
        msg = "ok"

    try:
        env.close()
    except Exception as e:
        return False, f"close() failed: {e!r}"

    return True, msg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="check all registered envs (not only ALE/*)")
    ap.add_argument("--filter", dest="substr", default=None, help="substring to filter env IDs")
    ap.add_argument("--render", action="store_true", help="render environments")
    ap.add_argument("--mode", default="rgb_array", choices=["rgb_array", "human"], help="render mode (default: rgb_array)")
    ap.add_argument("--frames", type=int, default=5, help="frames to step when rendering (default: 5)")
    ap.add_argument("--outdir", type=str, default="renders", help="output dir for rgb_array frames")
    args = ap.parse_args()

    render_mode = args.mode if args.render else None
    outdir = Path(args.outdir) if render_mode == "rgb_array" else None

    env_ids = list_env_ids(args.all, args.substr)
    if not env_ids:
        print("No environments match the criteria.", file=sys.stderr)
        return 2

    print(f"Testing {len(env_ids)} environments... (render_mode={render_mode})")
    failures = []
    for i, eid in enumerate(env_ids, 1):
        t0 = time.time()
        ok, msg = try_env(eid, seed=0, render_mode=render_mode, frames=args.frames, outdir=outdir)
        dt = time.time() - t0
        status = "PASS" if ok else "FAIL"
        tail = ("" if ok else "- ") + msg
        print(f"[{i:>3}/{len(env_ids):>3}] {status:>4} {eid:<30} ({dt:.2f}s) {tail}")
        if not ok:
            failures.append((eid, msg))

    print()
    if failures:
        print(f"{len(failures)} / {len(env_ids)} failed:")
        for eid, msg in failures:
            print(f" - {eid}: {msg}")
        return 1
    else:
        print("All tested environments passed.")
        return 0

if __name__ == "__main__":
    sys.exit(main())
