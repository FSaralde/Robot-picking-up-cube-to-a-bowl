"""Robot arm picks up a cube and drops it in a bowl (TorchRL + MuJoCo).

Setup (one time):
    python3.13 -m venv .venv
    .venv/bin/pip install torch torchrl tensordict mujoco imageio imageio-ffmpeg
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git

Run:
    .venv/bin/python ye.py                          # easy: 4 fixed layouts
    .venv/bin/python ye.py --level medium -n 50     # random cube/bowl spots
    .venv/bin/python ye.py --level hard -n 50       # + slippery fingers, heavier cubes
    .venv/bin/python ye.py --level hard -n 50 --videos 3   # also save 3 videos

Levels:
    easy    the 4 hand-picked layouts below
    medium  cube and bowl placed anywhere in the reachable area
    hard    medium + random finger-pad friction and cube mass, so the cube
            can slip out of the grasp
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from tensordict import TensorDict, TensorDictBase
import mujoco
from torchrl.envs import CubeBowlEnv, RobotMacroAction, RobotMacroActionMode, step_mdp
from torchrl.record import VideoRecorder

if importlib.util.find_spec("mujoco") is None:
    raise ImportError("This script needs MuJoCo: pip install mujoco")

HERE = Path(__file__).resolve().parent
MENAGERIE_PATH = os.environ.get(CubeBowlEnv.MENAGERIE_ENV_VAR) or str(
    HERE / "mujoco_menagerie"
)
if not Path(MENAGERIE_PATH).is_dir():
    raise RuntimeError(
        f"MuJoCo Menagerie not found at {MENAGERIE_PATH}. Run:\n"
        "  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git\n"
        f"or set {CubeBowlEnv.MENAGERIE_ENV_VAR} to your checkout."
    )

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--level", choices=["easy", "medium", "hard"], default="easy")
parser.add_argument("-n", "--trials", type=int, default=None,
                    help="number of rollouts (default: 4 for easy, 20 otherwise)")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--videos", type=int, default=None,
                    help="save videos of the first N rollouts plus up to N "
                         "failures (default: 4 for easy, 0 otherwise)")
ARGS, _ = parser.parse_known_args()
if ARGS.trials is None:
    ARGS.trials = 4 if ARGS.level == "easy" else 20
if ARGS.videos is None:
    ARGS.videos = 4 if ARGS.level == "easy" else 0
RECORD = ARGS.videos > 0

TUTORIAL_FAST = os.environ.get("TORCHRL_TUTORIALS_FAST", "0") == "1"

MAX_EPISODE_STEPS = 12000
RENDER_WIDTH = 320 if TUTORIAL_FAST else 480
RENDER_HEIGHT = 240 if TUTORIAL_FAST else 360
VIDEO_FRAME_SKIP = 16
VIDEO_INTERVAL_MS = 55
IK_KWARGS = {
    "iterations": 220,
    "orientation_weight": 1.0,
    "step_size": 0.7,
    "damping": 1e-4,
}

# ``render_every`` matches the recorder's ``skip``: the recorder keeps one
# frame in ``VIDEO_FRAME_SKIP``, so rendering the frames it drops would only
# burn time (rendering dominates the cost of a step in this scene).
env = CubeBowlEnv(
    seed=0,
    max_episode_steps=MAX_EPISODE_STEPS,
    menagerie_path=MENAGERIE_PATH,
    # Rendering is the slowest part of a step, so skip it when no video is
    # wanted. That makes big --level hard evaluations much faster.
    **(
        dict(
            from_pixels=True,
            pixels_only=False,
            render_width=RENDER_WIDTH,
            render_height=RENDER_HEIGHT,
            render_every=VIDEO_FRAME_SKIP,
        )
        if RECORD
        else {}
    ),
)
assert env.action_spec.shape[-1] == 7
assert env.robot_home_qpos is not None

primitive_control = env.make_urscript_transform(
    execute=True,
    macro_steps=250,
    settle_steps=1600,
    ik_kwargs=IK_KWARGS,
    stack_rewards=True,
    stack_observations=False,
)
recorder = VideoRecorder(
    logger=None,
    tag="scripted_macro",
    skip=VIDEO_FRAME_SKIP,
    make_grid=False,
)
if RECORD:
    env = env.append_transform(recorder)
env = env.append_transform(primitive_control)

# Handles into the MuJoCo model so ``hard`` can change physics per rollout.
_mj_model = env.base_env._backend._m
_pad_geom_ids = [
    mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
    for name in (
        "gripper/left_pad1",
        "gripper/left_pad2",
        "gripper/right_pad1",
        "gripper/right_pad2",
    )
]
_cube_body_id = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, "cube")
DEFAULT_PAD_FRICTION = float(_mj_model.geom_friction[_pad_geom_ids[0], 0])
DEFAULT_CUBE_MASS = float(_mj_model.body_mass[_cube_body_id])
DEFAULT_CUBE_INERTIA = _mj_model.body_inertia[_cube_body_id].copy()

cube_width = 2 * env.OBJECT_HALF_SIZE
grasp_width = cube_width - 0.001
gripper_close_command = env.gripper_ctrl_for_width(grasp_width)

workspace_layouts = torch.tensor(
    [
        # Cube in front of the robot, bowl behind it and slightly to the right.
        [[0.44, -0.21], [0.48, 0.10]],
        # Cube on the robot's right-front side, bowl on the left-back side.
        [[0.51, -0.18], [0.39, 0.04]],
        # Cube in front of the robot, bowl on the left-back side.
        [[0.45, -0.20], [0.38, 0.02]],
        # Cube on the robot's right-front side, bowl near the center-back.
        [[0.52, -0.18], [0.45, 0.02]],
    ],
)


# Reachable table area used for random placement (meters, robot frame).
# It covers all four hand-picked layouts, plus some margin.
CUBE_X, CUBE_Y = (0.38, 0.54), (-0.24, -0.02)
BOWL_X, BOWL_Y = (0.36, 0.50), (-0.02, 0.12)
MIN_CUBE_BOWL_DIST = 0.15  # keep the cube from starting inside/against the bowl
PAD_FRICTION_RANGE = (1.0, 5.0)  # default 5: lower means more slippery
CUBE_MASS_RANGE = (0.02, 0.08)  # default 0.02 kg: higher means it slips more


def sample_scene(index: int, rng: torch.Generator) -> dict[str, float]:
    """Pick where the cube and bowl go (and the physics for ``hard``)."""
    if ARGS.level == "easy":
        (cx, cy), (bx, by) = workspace_layouts[index % len(workspace_layouts)].tolist()
    else:

        def uniform(lo_hi: tuple[float, float]) -> float:
            lo, hi = lo_hi
            return lo + (hi - lo) * torch.rand((), generator=rng).item()

        while True:
            cx, cy = uniform(CUBE_X), uniform(CUBE_Y)
            bx, by = uniform(BOWL_X), uniform(BOWL_Y)
            if math.dist((cx, cy), (bx, by)) >= MIN_CUBE_BOWL_DIST:
                break
    scene = {"cube_x": cx, "cube_y": cy, "bowl_x": bx, "bowl_y": by,
             "pad_friction": DEFAULT_PAD_FRICTION, "cube_mass": DEFAULT_CUBE_MASS}
    if ARGS.level == "hard":
        scene["pad_friction"] = uniform(PAD_FRICTION_RANGE)
        scene["cube_mass"] = uniform(CUBE_MASS_RANGE)
    return scene


def apply_scene(scene: dict[str, float]) -> TensorDict:
    """Set the physics for this scene and build the reset TensorDict."""
    for gid in _pad_geom_ids:
        _mj_model.geom_friction[gid, 0] = scene["pad_friction"]
    _mj_model.body_mass[_cube_body_id] = scene["cube_mass"]
    # Inertia must scale with mass, or a heavier cube spins like a light one
    # and the simulation can blow up.
    _mj_model.body_inertia[_cube_body_id] = DEFAULT_CUBE_INERTIA * (
        scene["cube_mass"] / DEFAULT_CUBE_MASS
    )
    # Recompute MuJoCo's cached mass-dependent constants (contact softness
    # is scaled by them); without this a heavier cube makes the sim explode.
    mujoco.mj_setConst(_mj_model, env.base_env._backend._d)

    cube_pos = torch.tensor([[scene["cube_x"], scene["cube_y"], env.cube_position[2]]])
    bowl_target_z = env.bowl_position[2] + env.MENAGERIE_BOWL_TARGET_OFFSET[2]
    bowl_pos = torch.tensor([[scene["bowl_x"], scene["bowl_y"], bowl_target_z]])
    return TensorDict(
        {
            "cube_pos": cube_pos.to(dtype=env.dtype, device=env.device),
            "bowl_pos": bowl_pos.to(dtype=env.dtype, device=env.device),
        },
        batch_size=[1],
        device=env.device,
    )


td = env.reset()
assert td["robot_qpos"].shape[-1] == 6
assert env.low_level_action(td["robot_qpos"]).shape[-1] == 7
if RECORD:
    assert td["pixels"].shape[-3:] == torch.Size([RENDER_HEIGHT, RENDER_WIDTH, 3])

home_qpos = torch.as_tensor(
    env.robot_home_qpos,
    dtype=td["robot_qpos"].dtype,
    device=td["robot_qpos"].device,
).view(1, 6)
policy_state: dict[str, TensorDictBase | torch.Tensor] = {
    "td": td,
    "gripper_quat": td["pinch_quat"].clone(),
}


def pose_at(
    xyz: torch.Tensor, quat: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if quat is None:
        quat = torch.zeros(xyz.shape[:-1] + (4,), dtype=xyz.dtype, device=xyz.device)
        quat[..., 0] = 1.0
    else:
        quat = quat.to(dtype=xyz.dtype, device=xyz.device).expand(xyz.shape[:-1] + (4,))
    return xyz, quat


def carry_action(alpha: float, bowl: torch.Tensor) -> RobotMacroAction:
    current_td = policy_state["td"]
    gripper_quat = policy_state["gripper_quat"]
    cube = current_td["cube_pos"].clone()
    carry_start_cube = policy_state["carry_start_cube"]
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    desired_xy = carry_start_cube[..., :2] + alpha * (
        bowl[..., :2] - carry_start_cube[..., :2]
    )
    desired_cube = torch.cat(
        [
            desired_xy,
            torch.full_like(bowl[..., 2:3], 0.24),
        ],
        dim=-1,
    )
    position, quaternion = pose_at(desired_cube + pinch_to_cube, gripper_quat)
    return RobotMacroAction.reach_pose(
        position=position,
        quaternion=quaternion,
        orientation_mask=(1.0, 1.0, 0.0),
        path="cartesian",
        gripper="closed",
        gripper_command=gripper_close_command,
        steps=80,
        settle_steps=20,
    )


def gen_actions(td: TensorDictBase) -> Iterator:
    cube = td["cube_pos"].clone()
    bowl = td["bowl_pos"].clone()
    quat = td["pinch_quat"].clone()
    grasp_offset = cube.new_tensor([[0.0, 0.0, -0.016]])

    # Action 0: Keep the arm at the reset joint target and let the scene settle.
    yield RobotMacroAction.wait(gripper="open", steps=1, settle_steps=19)

    # Action 1: Fully open the gripper before approaching the cube.
    yield RobotMacroAction.open_gripper(steps=100, settle_steps=20)

    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    bowl = current_td["bowl_pos"].clone()
    quat = current_td["pinch_quat"].clone()
    position, quaternion = pose_at(cube + cube.new_tensor([[0.0, 0.0, 0.18]]), quat)
    # Action 2: Move the open gripper above the cube.
    yield RobotMacroAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="open",
        steps=180,
        settle_steps=60,
    )

    position, quaternion = pose_at(cube + grasp_offset, quat)
    # Action 3: Lower the open gripper to the grasp pose around the cube.
    yield RobotMacroAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="open",
        steps=180,
        settle_steps=60,
    )

    # Action 4: Close the gripper to a grasp about one millimeter tighter than
    # the cube width.
    yield RobotMacroAction.close_gripper(
        command=gripper_close_command, steps=160, settle_steps=80
    )

    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(
        cube + pinch_to_cube + cube.new_tensor([[0.0, 0.0, 0.20]]),
        quat,
    )
    # Action 5: Lift the grasped cube above the table.
    yield RobotMacroAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        gripper_command=gripper_close_command,
        steps=120,
        settle_steps=60,
    )

    # The cube is not welded to the gripper. It is held by contact and
    # friction between the pads. If we ask for one large move directly to the
    # bowl, the fingers can drag faster than the cube can safely follow: the
    # cube may slide inside the grasp, get pushed out, or arrive shifted enough
    # that the final drop misses the bowl. We therefore carry it through a few
    # intermediate macro targets in the table plane. Within each macro,
    # ``path="cartesian"`` keeps the pinch site on the straight Cartesian line
    # and ``orientation_mask`` keeps the gripper level. After each target, the
    # next command uses the latest observed cube and finger positions.
    policy_state["carry_start_cube"] = policy_state["td"]["cube_pos"].clone()

    # Action 6: Carry the cube one quarter of the way toward the bowl.
    yield carry_action(0.25, bowl)

    # Action 7: Carry the cube halfway toward the bowl.
    yield carry_action(0.50, bowl)

    # Action 8: Carry the cube three quarters of the way toward the bowl.
    yield carry_action(0.75, bowl)

    # Action 9: Carry the cube above the bowl center.
    yield carry_action(1.00, bowl)

    drop_cube = torch.cat(
        [
            bowl[..., :1],
            bowl[..., 1:2],
            torch.full_like(bowl[..., 2:3], 0.13),
        ],
        dim=-1,
    )
    current_td = policy_state["td"]
    cube = current_td["cube_pos"].clone()
    pinch_to_cube = current_td["pinch_pos"].clone() - cube
    position, quaternion = pose_at(drop_cube + pinch_to_cube, quat)
    # Action 10: Lower the grasped cube into the bowl.
    yield RobotMacroAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="closed",
        gripper_command=gripper_close_command,
        steps=100,
        settle_steps=40,
    )

    # Action 11: Open the gripper to release the cube.
    yield RobotMacroAction.open_gripper(steps=100, settle_steps=20)

    # Action 12: Wait with the gripper open so the cube can settle in the bowl.
    yield RobotMacroAction.wait(gripper="open", steps=240)

    current_td = policy_state["td"]
    retreat_xyz = torch.cat(
        [
            current_td["pinch_pos"][..., :2],
            torch.full_like(current_td["pinch_pos"][..., 2:3], 0.26),
        ],
        dim=-1,
    )
    position, quaternion = pose_at(retreat_xyz, quat)
    # Action 13: Retreat upward and away from the released cube.
    yield RobotMacroAction.reach_pose(
        position=position,
        quaternion=quaternion,
        gripper="open",
        steps=120,
        settle_steps=60,
    )

    # Action 14: Return the arm to the environment's reset joint configuration.
    yield RobotMacroAction.RESET

    # Action 15: Hold the reset pose while the final reward is measured.
    yield RobotMacroAction.wait(gripper="open", steps=800)


def action_mode(action) -> int | None:
    if not hasattr(action, "mode"):
        return None
    return int(action.mode.reshape(-1)[0].item())


def action_gripper(action) -> int:
    if not hasattr(action, "gripper"):
        return RobotMacroAction.GRIPPER_OPEN
    return int(action.gripper.reshape(-1)[0].item())


def run_scripted_rollout(td: TensorDictBase) -> tuple[TensorDictBase, str | None]:
    """Run the scripted pick-and-place. Returns (final td, failure reason or None)."""
    policy_state["td"] = td
    policy_state["gripper_quat"] = td["pinch_quat"].clone()
    policy_state.pop("carry_start_cube", None)
    grasp_distance = torch.full_like(td["cube_pos"][..., :1], float("inf"))
    cube_motion_while_closed = torch.zeros_like(grasp_distance)
    cube_lift_while_closed = torch.zeros_like(grasp_distance)
    max_reward = torch.zeros_like(grasp_distance)
    last_reward = torch.zeros_like(grasp_distance)
    closed_reference_cube: torch.Tensor | None = None
    start_cube = td["cube_pos"].clone()
    pre_grasp: dict[str, float] | None = None

    script_actions = gen_actions(td)
    while True:
        try:
            td["action"] = next(script_actions)
        except StopIteration:
            break

        action = td["action"]
        if pre_grasp is None and action_mode(action) == int(
            RobotMacroActionMode.CLOSE_GRIPPER
        ):
            # Snapshot just before squeezing: is the hand where it should be,
            # and is the cube still where it started?
            pre_grasp = {
                "hand_xy_error": (td["pinch_pos"][..., :2] - start_cube[..., :2])
                .norm()
                .item(),
                # xy only: the cube always drops ~1.5 cm onto the table at spawn.
                "cube_shift": (td["cube_pos"][..., :2] - start_cube[..., :2])
                .norm()
                .item(),
            }
        transition = env.step(td)
        reward = transition["next", "reward"]
        max_reward = torch.maximum(max_reward, reward.amax(dim=-2))
        last_reward = reward[..., -1, :]
        td = step_mdp(transition)
        policy_state["td"] = td

        mode = action_mode(action)
        gripper = action_gripper(action)
        if mode == int(RobotMacroActionMode.CLOSE_GRIPPER):
            grasp_distance = torch.minimum(
                grasp_distance, env.gripper_cube_distance(td)
            )
            closed_reference_cube = td["cube_pos"].clone()
        if (
            closed_reference_cube is not None
            and gripper == RobotMacroAction.GRIPPER_CLOSED
        ):
            cube_motion = (td["cube_pos"] - closed_reference_cube).norm(
                dim=-1, keepdim=True
            )
            cube_lift = td["cube_pos"][..., 2:3] - closed_reference_cube[..., 2:3]
            cube_motion_while_closed = torch.maximum(
                cube_motion_while_closed, cube_motion
            )
            cube_lift_while_closed = torch.maximum(cube_lift_while_closed, cube_lift)

    robot_home_error = (td["robot_qpos"] - home_qpos).norm(dim=-1, keepdim=True)

    # Each check is a stage the robot has to get right, in order. The first
    # one that fails is the reason we report for the rollout.
    pre_grasp = pre_grasp or {"hand_xy_error": math.inf, "cube_shift": 0.0}
    checks = [
        ("arm reached wrong spot (IK)", pre_grasp["hand_xy_error"] <= 0.03
         or pre_grasp["cube_shift"] > 0.02),
        ("arm knocked cube away", pre_grasp["cube_shift"] <= 0.02),
        ("missed grasp", grasp_distance.item() <= 0.025),
        ("cube did not move", cube_motion_while_closed.item() >= 0.05),
        ("cube slipped / not lifted", cube_lift_while_closed.item() >= 0.08),
        ("never reached bowl", max_reward.item() == 1.0),
        ("cube left bowl", last_reward.item() == 1.0 and bool(td["success"].all())),
        ("arm did not return home", robot_home_error.item() <= 0.04),
    ]
    failure = next((name for name, ok in checks if not ok), None)
    return td, failure


def save_video(path: Path, fps: int = 1000 // VIDEO_INTERVAL_MS) -> None:
    """Write the frames the recorder collected to an mp4 file."""
    import imageio.v2 as imageio

    if not recorder.obs:
        print("No frames recorded, skipping video.")
        return
    frames = torch.stack(recorder.obs).cpu()
    if frames.shape[-1] != 3:  # recorder stores CHW; imageio wants HWC
        frames = frames.permute(0, 2, 3, 1)
    imageio.mimsave(
        path, frames.to(torch.uint8).numpy(), fps=fps, macro_block_size=1
    )
    # Clear in place: the macro transform holds a copy of the recorder that
    # shares this same list, so rebinding it would break recording.
    recorder.obs.clear()
    print(f"Saved video to {path}")


def main() -> None:
    rng = torch.Generator().manual_seed(ARGS.seed)
    num_rollouts = 1 if TUTORIAL_FAST else ARGS.trials
    results: list[tuple[dict[str, float], str | None]] = []
    failure_videos = 0
    t0 = time.time()
    print(f"Level: {ARGS.level}, rollouts: {num_rollouts}, seed: {ARGS.seed}\n")

    for i in range(num_rollouts):
        scene = sample_scene(i, rng)
        if RECORD:
            recorder.obs.clear()
        start = env.reset(apply_scene(scene), set_state=True)
        _, failure = run_scripted_rollout(start)
        results.append((scene, failure))

        status = "ok  " if failure is None else f"FAIL ({failure})"
        physics = (
            f"  friction={scene['pad_friction']:.2f} mass={scene['cube_mass'] * 1000:.0f}g"
            if ARGS.level == "hard"
            else ""
        )
        print(
            f"[{i + 1:3d}/{num_rollouts}] {status:32s} "
            f"cube=({scene['cube_x']:.2f},{scene['cube_y']:.2f}) "
            f"bowl=({scene['bowl_x']:.2f},{scene['bowl_y']:.2f}){physics}"
        )

        if RECORD:
            if i < ARGS.videos:
                save_video(HERE / f"rollout_{i}.mp4")
            elif failure is not None and failure_videos < ARGS.videos:
                save_video(HERE / f"rollout_{i}_FAIL.mp4")
                failure_videos += 1

    # ---- summary ----
    n = len(results)
    wins = sum(f is None for _, f in results)
    print(f"\nSuccess rate: {wins}/{n} = {100 * wins / n:.0f}%"
          f"   ({time.time() - t0:.0f}s total)")
    reasons: dict[str, int] = {}
    for _, f in results:
        if f is not None:
            reasons[f] = reasons.get(f, 0) + 1
    if reasons:
        print("Failures by reason:")
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:3d}  {reason}")
    if ARGS.level == "hard":
        # Bucket by finger friction to show where the scripted robot breaks down.
        print("Success by finger friction:")
        edges = [1.0, 2.0, 3.0, 4.0, 5.01]
        for lo, hi in zip(edges, edges[1:]):
            bucket = [f for s, f in results if lo <= s["pad_friction"] < hi]
            if bucket:
                ok = sum(f is None for f in bucket)
                print(f"  {lo:.0f}-{min(hi, 5):.0f}: {ok}/{len(bucket)}")


if __name__ == "__main__":
    main()
