# Robot Pick-and-Place Under Randomization

A simulated UR5e robot arm picks up a cube and drops it into a bowl. The project measures how a hand-scripted controller holds up when the scene stops being the one it was tuned for.

![Robot arm with the cube placed in the bowl](docs/demo.png)

Simulated in [MuJoCo](https://mujoco.org/) through [TorchRL](https://github.com/pytorch/rl)'s `CubeBowlEnv`. The robot and gripper models come from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

## Results

The scripted controller is perfect on the layouts it was written for and drops to about 50% once the scene is randomized.

| Level | What changes | Success |
|---|---|---|
| `easy` | 4 fixed, hand-picked cube/bowl layouts | **4 / 4 (100%)** |
| `medium` | cube and bowl placed at random in the reachable area | **15 / 30 (50%)** |
| `hard` | `medium` + finger-pad friction (1.0–5.0) and cube mass (20–80 g) randomized | **15 / 30 (50%)** |

Seed 0. Each run is checked stage by stage, and the first stage that fails is recorded as the reason:

| Failure reason (`hard`, 30 runs) | Count |
|---|---|
| Arm reached the wrong spot (inverse kinematics) | 9 |
| Missed grasp (hand near the cube but didn't close on it) | 4 |
| Arm knocked the cube away on approach | 2 |

**Takeaway:** most failures come from **reaching**, not from holding on. The inverse-kinematics solver sometimes picks a bad arm configuration and puts the hand up to 30 cm from the cube. Lower friction and heavier cubes barely change the success rate. So the weak point to improve first is the approach, not the grip.

## How it works

The robot follows a fixed script of 16 high-level actions ("macro actions"). TorchRL turns each one into hundreds of low-level physics steps using inverse kinematics:

1. Settle, then open the gripper
2. Move above the cube, then lower around it
3. Close the gripper slightly tighter than the cube's width
4. Lift
5. Carry toward the bowl in four short steps (25/50/75/100%), re-reading the cube's position each time so it doesn't slip out
6. Lower, release, wait for the cube to settle
7. Retreat and return home

A run counts as a success only if all of these hold:
- the cube is grasped and lifted at least 8 cm
- it ends up inside the bowl and stays there
- the arm returns to its home pose

## What I changed on top of the tutorial

The scripted controller comes from TorchRL's `CubeBowlEnv` tutorial. My additions:

- **Difficulty levels** with randomized scene layout and physics (friction and mass)
- **Failure diagnosis.** Each run gets a reason instead of an assertion crash, including separate checks for an IK reach error and for a cube knocked away before the grasp.
- **Evaluation summary:** success rate, failures grouped by reason, and success by friction bucket
- **Physics fixes.** When the cube's mass changes, its inertia is scaled to match and MuJoCo's cached constants (`mj_setConst`) are recomputed. Without this, heavier cubes made the simulation unstable.
- **Video export** for the first N runs and the first N failures. Rendering is skipped when no video is wanted, which makes large evaluations faster.

## Setup

Requires Python 3.13 (PyTorch doesn't support 3.15 yet).

```bash
python3.13 -m venv .venv
.venv/bin/pip install torch torchrl tensordict mujoco imageio imageio-ffmpeg
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

The script looks for `mujoco_menagerie/` next to `ye.py`. To use a checkout somewhere else, set `TORCHRL_MUJOCO_MENAGERIE_PATH`.

## Usage

```bash
.venv/bin/python ye.py                                # easy, saves 4 videos
.venv/bin/python ye.py --level medium -n 50           # 50 random scenes
.venv/bin/python ye.py --level hard -n 50 --videos 2  # also save 2 runs + 2 failures
```

| Flag | Default | Meaning |
|---|---|---|
| `--level` | `easy` | `easy`, `medium` or `hard` |
| `-n`, `--trials` | 4 (easy), 20 (others) | number of runs |
| `--seed` | 0 | random seed for scene sampling |
| `--videos` | 4 (easy), 0 (others) | save the first N runs and up to N failures as `rollout_*.mp4` |

Example output (`--level hard -n 4`):

```
[  1/4] FAIL (arm reached wrong spot (IK)) cube=(0.43,-0.10) bowl=(0.43,0.11)  friction=2.82 mass=58g
[  2/4] ok                               cube=(0.44,-0.15) bowl=(0.36,0.00)  friction=2.18 mass=51g
[  3/4] FAIL (arm reached wrong spot (IK)) cube=(0.49,-0.04) bowl=(0.42,0.10)  friction=2.68 mass=53g
[  4/4] FAIL (arm reached wrong spot (IK)) cube=(0.53,-0.23) bowl=(0.39,0.03)  friction=2.22 mass=76g

Success rate: 1/4 = 25%   (10s total)
Failures by reason:
    3  arm reached wrong spot (IK)
Success by finger friction:
  2-3: 1/4
```

## Next steps

- **Learned policy:** use the successful scripted runs as demonstrations for behavior cloning, then compare against the 50% scripted baseline.
- **Better reaching:** seed IK from a pre-grasp joint pose, or move through a waypoint, to get rid of the main failure mode.
- **Harder physics:** randomize cube rotation, add observation noise, add pushes to the arm.
