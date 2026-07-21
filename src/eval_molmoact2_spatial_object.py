"""
Custom LIBERO eval(spatial / object) script for MolmoAct2
Saves:
  eval.log: timestamped run log (console + file)
  results.csv: per-episode success, steps, seeds, distractor fields
  scene_properties.csv: per-task BDDL distractor metrics + sim metadata
  distractor_density.csv: NLP subset (task_id, suite, total/target objects, density)
  nlp_analysis_table.csv: scene_properties merged with per-task success_rate
  frames/: initial scene frame per task (first episode only)
  frames/failures/: up to 3 failure frames per task
  videos/failures/: MP4 video of failed episodes (max 3 per task)
  videos/successes/: MP4 video of successful episodes (gated)

  - Grasp / close events from gripper state (not just action commands)
  - Nearest object at first close → wrong-object vs correct-object grounding
  - Pick / place distances, path length, EEF z range, timeout flag
  - "likely_recovery" successes (many grasps or long episodes that still succeed)
  - Gated success videos (only "suspicious" successes) + failure videos

Protocol constants (seeds, wait steps, max steps) match MolmoAct2's published
LIBERO eval so numbers are comparable to the official checkpoint card.
"""

import argparse
import csv
import logging
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path    
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--suite", type=str, required=True,
    choices=["libero_object", "libero_spatial"])
parser.add_argument("--n_episodes", type=int, default=50)
parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
parser.add_argument("--output_dir", type=str, default="outputs/custom_eval")
parser.add_argument("--save_fail_videos", action="store_true",
    help="Save fail MP4s (auto-on for spatial/object)")
parser.add_argument("--save_success_videos", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--max_success_videos_per_task", type=int, default=3)
parser.add_argument("--max_fail_videos_per_task", type=int, default=3)
parser.add_argument("--success_video_min_grasps", type=int, default=None,
    help="Default: 2")
parser.add_argument("--success_video_min_steps", type=int, default=None,
    help="Default: 150 spatial, 160 object")
parser.add_argument("--task_ids", type=str, default=None)
parser.add_argument("--num_steps_wait", type=int, default=50)
parser.add_argument("--eval_seed", type=int, default=1000)
parser.add_argument("--per_episode_seed", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use_init_states", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--use_egl", action=argparse.BooleanOptionalAction, default=None)
args = parser.parse_args()

# Zero motion + gripper open (last dim -1). Used only during settle after reset
# so physics can settle before the policy starts not a learned action.
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Episode length caps from LIBERO / MolmoAct2 suite defaults for Spatial & Object.
TASK_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
}

# likely_recovery video gates.
# Clean first-try successes are usually short (~70–120 steps) with 1 grasp.
# Longer episodes or >=2 grasps that still succeed are candidates for recovery
# (regrasp) analysis — we save those videos, not every success.
SUITE_SUCCESS_STEP_GATE = {
    "libero_spatial": 150,
    "libero_object": 160,
}
SUITE_SUCCESS_GRASP_GATE = {
    "libero_spatial": 2,
    "libero_object": 2,
}

# Empirically: mean(|robot0_gripper_qpos|) is ~0.04 when open and ~0.001 when
# closed on LIBERO's Franka parallel gripper.
GRIPPER_OPEN_THRESH = 0.015  # mean|qpos|: ~0.04 open, ~0.001 closed

# LIBERO BDDL goal_state predicates we treat as pick, place pairs, e.g.
# (On bowl_1 plate_1) or (In obj basket). Stored lowercase in parsed goals.
PLACE_PREDICATES = {"on", "in"}  # LIBERO goal_state uses lowercase


def _resolve_use_egl():
    #use EGL so MuJoCo/robosuite can render LIBERO camera images on a server without a display
    if args.use_egl is not None:
        return args.use_egl
    return sys.platform == "linux" and args.device == "cuda"


def setup_eval_environment():
    #Limit CPU thread pools
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if _resolve_use_egl():
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def episode_seed(episode_idx):
    return args.eval_seed + episode_idx if args.per_episode_seed else args.eval_seed


def reset_episode(env, init_states, episode_idx):
    seed = episode_seed(episode_idx)
    if hasattr(env, "seed"):
        env.seed(seed)
    env.reset()
    if args.use_init_states and init_states is not None:
        obs = env.set_init_state(init_states[episode_idx % len(init_states)])
    else:
        obs = env.env._get_observations()
    for _ in range(args.num_steps_wait):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def parse_task_ids(raw, n_tasks):
    if not raw:
        return list(range(n_tasks))
    ids = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    bad = [i for i in ids if i < 0 or i >= n_tasks]
    if bad:
        raise ValueError(f"--task_ids out of range for {n_tasks} tasks: {bad}")
    return ids


def eef_pos_from_obs(obs, env):
    """End-effector XYZ from observation, with MuJoCo grip site as fallback."""
    if "robot0_eef_pos" in obs:
        return np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
    return np.asarray(env.sim.data.get_site_xpos("gripper0_grip_site"), dtype=np.float64)


def gripper_open_amount(obs):
    """
    Scalar finger separation from gripper joint positions.

    LIBERO/robosuite exposes robot0_gripper_qpos as two joints that are often
    approximately [+x, -x]. Using mean(qpos) would cancel to ~0; mean(|qpos|)
    measures how open the gripper is. Compare to GRIPPER_OPEN_THRESH.
    """
    q = obs.get("robot0_gripper_qpos")
    if q is None:
        return None
    q = np.asarray(q, dtype=np.float64)
    return float(np.mean(np.abs(q)))


def _lookup_obj(env, name):
    inner = env.env
    for attr in ("objects_dict", "fixtures_dict"):
        d = getattr(inner, attr, None) or {}
        if name in d:
            return d[name]
    return None


def get_xpos(env, name):
    """Body or site position for objects / fixtures / contain regions."""
    if not name:
        return None
    # 1) object / fixture root body
    obj = _lookup_obj(env, name)
    if obj is not None:
        body = getattr(obj, "root_body", None) or name
        try:
            return np.asarray(env.sim.data.get_body_xpos(body), dtype=np.float64)
        except (ValueError, KeyError, TypeError):
            pass
    # 2) MuJoCo site (e.g. basket_1_contain_region)
    try:
        return np.asarray(env.sim.data.get_site_xpos(name), dtype=np.float64)
    except (ValueError, KeyError, TypeError):
        pass
    # 3) raw body name
    try:
        return np.asarray(env.sim.data.get_body_xpos(name), dtype=np.float64)
    except (ValueError, KeyError, TypeError):
        pass
    # 4) parent object before _contain_region / _region suffix
    for suf in ("_contain_region", "_region"):
        if name.endswith(suf):
            return get_xpos(env, name[: -len(suf)])
    return None


def body_xpos(env, name):
    pos = get_xpos(env, name)
    if pos is None:
        raise KeyError(name)
    return pos


def dist_eef_to(obs, env, name):
    """Euclidean distance from gripper to a named body/site (meters in sim)."""
    if not name:
        return None
    try:
        return float(np.linalg.norm(eef_pos_from_obs(obs, env) - get_xpos(env, name)))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def dist_between(env, a, b):
    """Euclidean distance between two named bodies/sites."""
    try:
        pa, pb = get_xpos(env, a), get_xpos(env, b)
        if pa is None or pb is None:
            return None
        return float(np.linalg.norm(pa - pb))
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def goal_states(env):
    return list(env.env.parsed_problem.get("goal_state") or [])


def on_in_pairs(env):
    """
    Extract (pick_object, place_receptacle) pairs from On/In goals.

    Example: (On akita_black_bowl_1 plate_1) → ("akita_black_bowl_1", "plate_1").
    Used for pick/place distance metrics without hard-coding task names.
    """
    pairs = []
    for state in goal_states(env):
        if len(state) == 3 and str(state[0]).lower() in PLACE_PREDICATES:
            pairs.append((state[1], state[2]))
    return pairs


def pick_names_from_goal(env):
    return list(dict.fromkeys(p for p, _ in on_in_pairs(env)))


def place_names_from_goal(env):
    return list(dict.fromkeys(pl for _, pl in on_in_pairs(env)))



def nearest_movable(obs, env):
    """
    Movable object whose body is closest to the gripper.

    Used at first gripper-close to label correct vs wrong object grounding
    (e.g. bowl vs ramekin). Fixtures are ignored; only objects_dict entries.
    """
    eef = eef_pos_from_obs(obs, env)
    objs = getattr(env.env, "objects_dict", None) or {}
    best, best_d = "", float("inf")
    for name in objs:
        pos = get_xpos(env, name)
        if pos is None:
            continue
        d = float(np.linalg.norm(eef - pos))
        if d < best_d:
            best, best_d = name, d
    return best, (best_d if best else None)


def mean_place_obj_dist(env):
    """Mean distance of each On/In object to its receptacle."""
    ds = []
    for obj, place in on_in_pairs(env):
        d = dist_between(env, obj, place)
        if d is not None:
            ds.append(d)
    return float(np.mean(ds)) if ds else None


def min_eef_place_dist(obs, env, place_names):
    """How close the gripper got to any place receptacle."""
    ds = [dist_eef_to(obs, env, p) for p in place_names]
    ds = [d for d in ds if d is not None]
    return min(ds) if ds else None


def min_eef_pick_dist(obs, env, pick_names):
     """How close the gripper got to any pick target."""
    ds = [dist_eef_to(obs, env, p) for p in pick_names]
    ds = [d for d in ds if d is not None]
    return min(ds) if ds else None


setup_eval_environment()

SAVE_FAIL_VIDEOS = args.save_fail_videos or args.suite in [
    "libero_spatial", "libero_object",
]
SAVE_SUCCESS_VIDEOS = args.save_success_videos
MAX_FAIL_VIDEOS = args.max_fail_videos_per_task
MAX_SUCCESS_VIDEOS = args.max_success_videos_per_task
SUCCESS_MIN_GRASPS = (
    args.success_video_min_grasps
    if args.success_video_min_grasps is not None
    else SUITE_SUCCESS_GRASP_GATE[args.suite]
)
SUCCESS_MIN_STEPS = (
    args.success_video_min_steps
    if args.success_video_min_steps is not None
    else SUITE_SUCCESS_STEP_GATE[args.suite]
)
BUFFER_VIDEOS = SAVE_FAIL_VIDEOS or SAVE_SUCCESS_VIDEOS
VIDEO_FPS = 10

output_dir = Path(args.output_dir) / args.suite
frames_dir = output_dir / "frames"
fail_dir = frames_dir / "failures"
video_fail_dir = output_dir / "videos" / "failures"
video_ok_dir = output_dir / "videos" / "successes"
for d in (output_dir, frames_dir, fail_dir):
    d.mkdir(parents=True, exist_ok=True)
if SAVE_FAIL_VIDEOS:
    video_fail_dir.mkdir(parents=True, exist_ok=True)
if SAVE_SUCCESS_VIDEOS:
    video_ok_dir.mkdir(parents=True, exist_ok=True)

results_path = output_dir / "results.csv"
scene_props_path = output_dir / "scene_properties.csv"
distractor_density_path = output_dir / "distractor_density.csv"
nlp_analysis_path = output_dir / "nlp_analysis_table.csv"
log_path = output_dir / "eval.log"

RESULT_FIELDS = [
    "task_id", "suite", "task_name", "instruction", "episode", "episode_seed",
    "success", "n_steps", "max_steps", "timeout",
    "distractor_density", "n_distractors", "n_pick_targets",
    "n_grasp_attempts", "n_gripper_close_events", "n_close_commands",
    "time_to_first_close",
    "frac_closed", "path_length",
    "min_target_dist", "final_target_dist", "time_to_min_pick_dist",
    "min_place_dist", "final_place_dist", "min_eef_place_dist", "final_eef_place_dist",
    "pick_displacement",
    "eef_z_min", "eef_z_max",
    "nearest_at_first_close", "correct_object_at_first_close",
    "likely_recovery", "success_video_gated",
    "frame_path", "fail_frame_path", "success_video_path",
]


def setup_logging():
    logger = logging.getLogger("eval_molmoact2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def append_result_row(path, row):
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows([row])


log = setup_logging()
if results_path.exists():
    results_path.unlink()
log.info("Eval run started (recovery monitor)")
log.info("Output dir: %s", output_dir)
log.info("Suite: %s | episodes=%s | device=%s",
         args.suite, args.n_episodes, args.device)
log.info("Fail videos: %s | Success videos (gated): %s", SAVE_FAIL_VIDEOS, SAVE_SUCCESS_VIDEOS)
log.info("Success gate: grasps>=%d OR steps>=%d | max ok vids=%d max fail vids=%d",
         SUCCESS_MIN_GRASPS, SUCCESS_MIN_STEPS, MAX_SUCCESS_VIDEOS, MAX_FAIL_VIDEOS)
log.info("task_ids: %s", args.task_ids or "all")

log.info("Loading MolmoAct2...")
from lerobot.policies import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

POLICY_PATH = "allenai/MolmoAct2-LIBERO-LeRobot"
policy = MolmoAct2Policy.from_pretrained(
    POLICY_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
).to(args.device)
policy.eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=policy.config,
    pretrained_path=POLICY_PATH,
    preprocessor_overrides={
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": {}},
    },
)
env_preprocessor = PolicyProcessorPipeline(steps=[LiberoProcessorStep()])
log.info("MolmoAct2 loaded")

log.info("Loading LIBERO suite: %s", args.suite)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

task_suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks = task_suite.n_tasks
task_id_list = parse_task_ids(args.task_ids, n_tasks)
log.info("Found %d tasks; running %d: %s", n_tasks, len(task_id_list), task_id_list)

# Camera mapping follows the official MolmoAct2-LIBERO-LeRobot checkpoint convention: https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot
CAMERA_MAP = {
    "agentview_image": "image",
    "robot0_eye_in_hand_image": "wrist_image",
}


def _find_bddl_block(text, name):
    """Extract a balanced (:name) S-expression block from a BDDL file."""
    marker = f"(:{name}"
    start = text.find(marker)
    if start == -1:
        return ""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + len(marker):i].strip()
    return ""


def _parse_object_instances(block):
    """Parse object instance names from a BDDL (:objects) block."""
    instances = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        left = line.split(" - ")[0].strip() if " - " in line else line
        instances.extend(left.split())
    return instances


def parse_bddl(bddl_path):
    """
    Read language + object lists from BDDL for scene-complexity metadata.

    distractor_density = n_distractors / total_objects, where distractors are
    objects not listed in obj_of_interest. Deterministic from ground truth
    (not estimated from RGB).
    """
    text = Path(bddl_path).read_text()
    instruction = _find_bddl_block(text, "language").strip()
    objects = _parse_object_instances(_find_bddl_block(text, "objects"))
    targets = [t for t in _find_bddl_block(text, "obj_of_interest").split() if t.strip()]
    target_set = set(targets)
    total_objects = len(objects)
    n_distractors = len([o for o in objects if o not in target_set])
    return {
        "instruction": instruction,
        "bddl_file": str(bddl_path),
        "total_objects": total_objects,
        "target_objects": len(targets),
        "target_names": targets,
        "n_distractors": n_distractors,
        "distractor_density": n_distractors / total_objects if total_objects else 0.0,
    }


def save_video(frames, path, fps=VIDEO_FPS):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()


def is_suspicious_success(n_steps, n_grasp_attempts):
    return n_grasp_attempts >= SUCCESS_MIN_GRASPS or n_steps >= SUCCESS_MIN_STEPS


all_results = []
all_scene_props = []

for task_id in task_id_list:
    task = task_suite.get_task(task_id)
    task_bddl_file = task_suite.get_task_bddl_file_path(task_id)
    task_instruction = task.language
    task_key = f"{args.suite}_{task_id}"
    bddl_meta = parse_bddl(task_bddl_file)

    log.info("Task %d/%d: %s", task_id, n_tasks - 1, task_instruction)
    log.info("  distractor_density=%.4f (%d/%d)",
             bddl_meta["distractor_density"], bddl_meta["n_distractors"],
             bddl_meta["total_objects"])

    # 256×256 + agentview/wrist matches MolmoAct2 LIBERO training / official eval
    env = OffScreenRenderEnv(**{
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
    })

    init_states = None
    if args.use_init_states:
        init_states = task_suite.get_task_init_states(task_id)

    obs = reset_episode(env, init_states, episode_idx=0)

    try:
        objects_dict = getattr(env.env, "objects_dict", None) or {}
        n_objects = len(objects_dict)
    except (AttributeError, TypeError) as e:
        log.warning("Could not read objects_dict: %s", e)
        objects_dict, n_objects = {}, -1

    pick_names = pick_names_from_goal(env) or list(bddl_meta.get("target_names") or [])[:1]
    place_names = place_names_from_goal(env)
    pick_name = pick_names[0] if pick_names else None
    n_pick_targets = len(pick_names)
    place_pairs = on_in_pairs(env)
    if not place_pairs:
        log.warning("  no On/In goal pairs found — place distances will be -1")
    else:
        log.info("  pick=%s place_pairs=%s", pick_names, place_pairs)

    # initial pick poses for displacement
    init_pick_pos = {}
    for pn in pick_names:
        try:
            init_pick_pos[pn] = body_xpos(env, pn).copy()
        except (ValueError, KeyError, TypeError, AttributeError):
            pass

    distance = dist_eef_to(obs, env, pick_name)
    if distance is None:
        distance = -1.0

    all_scene_props.append({
        "task_id": task_key,
        "suite": args.suite,
        "instruction": bddl_meta["instruction"],
        "bddl_file": bddl_meta["bddl_file"],
        "total_objects": bddl_meta["total_objects"],
        "target_objects": bddl_meta["target_objects"],
        "n_distractors": bddl_meta["n_distractors"],
        "distractor_density": bddl_meta["distractor_density"],
        "n_objects_sim": n_objects,
        "initial_distance": round(float(distance), 4),
        "n_pick_targets": n_pick_targets,
    })

    task_failures = 0
    task_success_vids = 0
    task_successes = 0
    max_steps = TASK_MAX_STEPS[args.suite]
    n_episodes = min(args.n_episodes, len(init_states)) if init_states is not None else args.n_episodes

    for episode_idx in range(n_episodes):
        obs = reset_episode(env, init_states, episode_idx)

        # Per-episode initial pick poses (init states differ by episode)
        init_pick_pos = {}
        for pn in pick_names:
            try:
                pos = get_xpos(env, pn)
                if pos is not None:
                    init_pick_pos[pn] = pos.copy()
            except (ValueError, KeyError, TypeError, AttributeError):
                pass

        if episode_idx == 0:
            frame = obs.get("agentview_image")
            if frame is not None:
                fp = frames_dir / f"{task_key}_initial.jpg"
                Image.fromarray(np.ascontiguousarray(frame[::-1, ::-1], dtype=np.uint8)).save(fp)
                frame_path = str(fp)
            else:
                frame_path = ""
        else:
            frame_path = ""

        episode_frames = [] if BUFFER_VIDEOS else None
        fail_frame_path = ""
        success_video_path = ""

        policy.reset()
        done = False
        success = False
        step = 0

        n_grasp_attempts = 0
        n_gripper_close_events = 0
        n_close_commands = 0
        time_to_first_close = ""
        closed_steps = 0
        path_length = 0.0
        #Starting at infinity guarantees the first real measurement always wins
        min_target_dist = float("inf")
        final_target_dist = -1.0
        time_to_min_pick_dist = ""
        min_place_dist = float("inf")
        final_place_dist = -1.0
        min_eef_place = float("inf")
        final_eef_place = -1.0
        eef_z_min = float("inf")
        eef_z_max = float("-inf")
        nearest_at_first_close = ""
        correct_object_at_first_close = ""
        prev_eef = eef_pos_from_obs(obs, env)
        prev_open = gripper_open_amount(obs)
        prev_closed = prev_open is not None and prev_open < GRIPPER_OPEN_THRESH
        prev_close_cmd = False

        while not done and step < max_steps:
            mapped_obs = {}
            for libero_key, model_key in CAMERA_MAP.items():
                if libero_key in obs:
                    img = obs[libero_key]
                    mapped_obs[f"observation.images.{model_key}"] = (
                        torch.from_numpy(img.astype(np.float32) / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )
            if "robot0_eef_pos" in obs:
                mapped_obs["observation.robot_state"] = {
                    "eef": {
                        "pos": torch.from_numpy(
                            np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
                        ).unsqueeze(0),
                        "quat": torch.from_numpy(
                            np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32)
                        ).unsqueeze(0),
                    },
                    "gripper": {
                        "qpos": torch.from_numpy(
                            np.asarray(obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
                        ).unsqueeze(0),
                    },
                }
            mapped_obs["task"] = task_instruction

            if episode_frames is not None:
                frame = obs.get("agentview_image")
                if frame is not None:
                    episode_frames.append(np.ascontiguousarray(frame[::-1, ::-1], dtype=np.uint8))

            with torch.inference_mode():
                observation = env_preprocessor(mapped_obs)
                #The preprocessor converts raw LIBERO observations (camera images, robot state) into whatever format the model was trained on
                observation = preprocessor(observation)
                action = policy.select_action(observation, inference_action_mode="continuous")
                #The postprocessor converts the model's raw output back into an actual 7D action 
                #[dx,dy,dz,droll,dpitch,dyaw,gripper] that the simulator understands.
                action = postprocessor(action)

            action_np = action.squeeze(0).cpu().float().numpy()
            # Close command edge (LIBERO: positive gripper cmd closes)
            close_cmd = bool(action_np[-1] > 0.3)
            if close_cmd and not prev_close_cmd:
                n_close_commands += 1
            prev_close_cmd = close_cmd

            obs, reward, done, info = env.step(action_np)
            step_idx = step + 1

            # Path length = sum of EEF displacements (proxy for inefficient motion)
            #longer path = more retries/hesitation
            eef = eef_pos_from_obs(obs, env)
            path_length += float(np.linalg.norm(eef - prev_eef))
            prev_eef = eef
            z = float(eef[2])
            eef_z_min = min(eef_z_min, z)
            eef_z_max = max(eef_z_max, z)

            # Pick progress: closest approach of gripper to any pick target
            d_pick = min_eef_pick_dist(obs, env, pick_names)
            if d_pick is not None:
                if d_pick < min_target_dist:
                    min_target_dist = d_pick
                    time_to_min_pick_dist = step_idx
                final_target_dist = d_pick

             # Place progress: object-receptacle distance from goal On/In pairs
            d_place_obj = mean_place_obj_dist(env)
            if d_place_obj is not None:
                min_place_dist = min(min_place_dist, d_place_obj)
                final_place_dist = d_place_obj

            d_eef_place = min_eef_place_dist(obs, env, place_names)
            if d_eef_place is not None:
                min_eef_place = min(min_eef_place, d_eef_place)
                final_eef_place = d_eef_place

            # Grasp attempt = open-closed transition on measured finger state
            open_amt = gripper_open_amount(obs)
            if open_amt is not None:
                closed = open_amt < GRIPPER_OPEN_THRESH
                if closed:
                    closed_steps += 1
                if closed and not prev_closed:
                    n_gripper_close_events += 1
                    n_grasp_attempts += 1
                    if time_to_first_close == "":
                        time_to_first_close = step_idx
                        nearest_at_first_close, _ = nearest_movable(obs, env)
                        correct_object_at_first_close = int(
                            nearest_at_first_close in set(pick_names)
                        ) if nearest_at_first_close else ""
                prev_closed = closed

            if reward > 0:
                success = True
                done = True
            step += 1

        # finalize shared
        if min_target_dist == float("inf"):
            min_target_dist = -1.0
        if min_place_dist == float("inf"):
            min_place_dist = -1.0
        if min_eef_place == float("inf"):
            min_eef_place = -1.0
        if eef_z_min == float("inf"):
            eef_z_min = -1.0
            eef_z_max = -1.0

        frac_closed = round(closed_steps / step, 4) if step else 0.0
        timeout = int((not success) and step >= max_steps)

        # How far the primary pick object moved from episode start (place progress)
        pick_displacement = -1.0
        if pick_name and pick_name in init_pick_pos:
            try:
                cur = get_xpos(env, pick_name)
                if cur is not None:
                    pick_displacement = float(np.linalg.norm(cur - init_pick_pos[pick_name]))
            except (ValueError, KeyError, TypeError, AttributeError):
                pass

        likely_recovery = int(success and is_suspicious_success(step, n_grasp_attempts))
        success_video_gated = 0

        if not success and task_failures < MAX_FAIL_VIDEOS:
            fail_frame = obs.get("agentview_image")
            if fail_frame is not None:
                fail_jpg = fail_dir / f"{task_key}_ep{episode_idx}_fail.jpg"
                Image.fromarray(np.ascontiguousarray(fail_frame[::-1, ::-1], dtype=np.uint8)).save(fail_jpg)
                fail_frame_path = str(fail_jpg)
            if SAVE_FAIL_VIDEOS and episode_frames:
                vid_path = video_fail_dir / f"{task_key}_ep{episode_idx}_fail.mp4"
                save_video(episode_frames, vid_path)
                log.info("  saved failure video: %s", vid_path.name)
            task_failures += 1

        if (
            success
            and SAVE_SUCCESS_VIDEOS
            and episode_frames
            and task_success_vids < MAX_SUCCESS_VIDEOS
            and is_suspicious_success(step, n_grasp_attempts)
        ):
            vid_path = video_ok_dir / f"{task_key}_ep{episode_idx}_ok.mp4"
            save_video(episode_frames, vid_path)
            success_video_path = str(vid_path)
            success_video_gated = 1
            task_success_vids += 1
            log.info("  saved gated success video: %s (steps=%d grasps=%d)",
                     vid_path.name, step, n_grasp_attempts)

        if success:
            task_successes += 1

        row = {
            "task_id": task_key,
            "suite": args.suite,
            "task_name": task_instruction,
            "instruction": bddl_meta["instruction"],
            "episode": episode_idx,
            "episode_seed": episode_seed(episode_idx),
            "success": int(success),
            "n_steps": step,
            "max_steps": max_steps,
            "timeout": timeout,
            "distractor_density": bddl_meta["distractor_density"],
            "n_distractors": bddl_meta["n_distractors"],
            "n_pick_targets": n_pick_targets,
            "n_grasp_attempts": n_grasp_attempts,
            "n_gripper_close_events": n_gripper_close_events,
            "n_close_commands": n_close_commands,
            "time_to_first_close": time_to_first_close,
            "frac_closed": frac_closed,
            "path_length": round(path_length, 4),
            "min_target_dist": round(float(min_target_dist), 4),
            "final_target_dist": round(float(final_target_dist), 4),
            "time_to_min_pick_dist": time_to_min_pick_dist,
            "min_place_dist": round(float(min_place_dist), 4),
            "final_place_dist": round(float(final_place_dist), 4),
            "min_eef_place_dist": round(float(min_eef_place), 4),
            "final_eef_place_dist": round(float(final_eef_place), 4),
            "pick_displacement": round(float(pick_displacement), 4),
            "eef_z_min": round(float(eef_z_min), 4),
            "eef_z_max": round(float(eef_z_max), 4),
            "nearest_at_first_close": nearest_at_first_close,
            "correct_object_at_first_close": correct_object_at_first_close,
            "likely_recovery": likely_recovery,
            "success_video_gated": success_video_gated,
            "frame_path": frame_path,
            "fail_frame_path": fail_frame_path,
            "success_video_path": success_video_path,
        }

        all_results.append(row)
        append_result_row(results_path, row)

        status = "SUCCESS" if success else "FAIL"
        extra = " | RECOVERY?" if likely_recovery else ""
        log.info(
            "  ep %3d/%d %s | steps=%d grasps=%d path=%.2f min_pick=%.3f | seed=%d | %d/%d%s",
            episode_idx, n_episodes - 1, status, step, n_grasp_attempts, path_length,
            min_target_dist if min_target_dist >= 0 else -1,
            episode_seed(episode_idx), task_successes, episode_idx + 1, extra,
        )

    sr = task_successes / n_episodes * 100
    log.info("Task %d done — %.1f%% (%d/%d) | fail vids=%d ok vids=%d",
             task_id, sr, task_successes, n_episodes, task_failures, task_success_vids)
    env.close()

log.info("Results CSV: %s (%d rows)", results_path, len(all_results))

SCENE_PROP_FIELDS = [
    "task_id", "suite", "instruction", "bddl_file",
    "total_objects", "target_objects", "n_distractors", "distractor_density",
    "n_objects_sim", "initial_distance", "n_pick_targets",
]
DISTRACTOR_DENSITY_FIELDS = [
    "task_id", "suite", "instruction",
    "total_objects", "target_objects", "n_distractors", "distractor_density",
]

with open(scene_props_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SCENE_PROP_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_scene_props)

with open(distractor_density_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=DISTRACTOR_DENSITY_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_scene_props)

import pandas as pd
df = pd.read_csv(results_path)
props_df = pd.read_csv(scene_props_path)
suite_sr = df.groupby("suite")["success"].mean() * 100
task_sr = df.groupby("task_id")["success"].mean() * 100

task_summary = df.groupby(["task_id", "suite"], as_index=False).agg(
    success_rate=("success", "mean"),
    n_episodes=("success", "count"),
    avg_steps=("n_steps", "mean"),
    avg_grasps=("n_grasp_attempts", "mean"),
    n_likely_recovery=("likely_recovery", "sum"),
    n_timeout=("timeout", "sum"),
)

nlp_table = task_summary.merge(props_df, on=["task_id", "suite"], how="left")
nlp_table.to_csv(nlp_analysis_path, index=False)

log.info("=" * 50)
log.info("SUMMARY — %s (recovery monitor)", args.suite)
log.info("=" * 50)
log.info("Overall success rate: %.1f%%", suite_sr.iloc[0])
log.info("likely_recovery=%d | gated success vids=%d | timeouts=%d",
         int(df["likely_recovery"].sum()),
         int(df["success_video_gated"].sum()),
         int(df["timeout"].sum()))
for tid, rate in task_sr.items():
    log.info("  %s: %.1f%%", tid, rate)
log.info("Results: %s", results_path)
log.info("Fail videos: %s", video_fail_dir if SAVE_FAIL_VIDEOS else "(off)")
log.info("Success videos (gated): %s", video_ok_dir if SAVE_SUCCESS_VIDEOS else "(off)")
log.info("=" * 50)
