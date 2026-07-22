"""
Custom LIBERO eval script for MolmoAct2
Saves:
  - eval.log: timestamped run log (console + file)
  - results.csv: per-episode success, steps, seeds, distractor fields (written live)
  - scene_properties.csv: per-task BDDL distractor metrics + sim metadata
  - distractor_density.csv: NLP subset (task_id, suite, total/target objects, density)
  - nlp_analysis_table.csv: scene_properties merged with per-task success_rate
  - frames/: initial scene frame per task (first episode only)
  - frames/failures/: up to 3 failure frames per task
  - videos/failures/: MP4 video of failed episodes (libero_spatial + libero_object only, max 3 per task)

MolmoAct2-aligned defaults: num_steps_wait=50, eval_seed=1000, per_episode_seed,
use_init_states, EGL on Linux+cuda. Override with --no-use-egl on Mac (mps).
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

# ── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--suite", type=str, required=True,
    choices=["libero_object", "libero_spatial", "libero_goal", "libero_10"],
    help="LIBERO task suite to evaluate")
parser.add_argument("--n_episodes", type=int, default=50,
    help="Number of episodes per task")
parser.add_argument("--device", type=str, default="cuda",
    choices=["cuda", "mps", "cpu"],
    help="Device: cuda (NVIDIA), mps (Apple Silicon), cpu")
parser.add_argument("--output_dir", type=str, default="outputs/custom_eval_500",
    help="Directory to save results")
parser.add_argument("--save_fail_videos", action="store_true",
    help="Save MP4 videos of failed episodes (auto-enabled)")
parser.add_argument("--num_steps_wait", type=int, default=50,
    help="Dummy steps after reset for sim to settle (MolmoAct2 uses 50)")
parser.add_argument("--eval_seed", type=int, default=1000,
    help="Base eval seed (MolmoAct2 uses 1000)")
parser.add_argument("--per_episode_seed", action=argparse.BooleanOptionalAction, default=True,
    help="Derive episode seed as eval_seed + episode_idx")
parser.add_argument("--use_init_states", action=argparse.BooleanOptionalAction, default=True,
    help="Use LIBERO fixed init states per episode")
parser.add_argument("--use_egl", action=argparse.BooleanOptionalAction, default=None,
    help="EGL headless rendering (auto-on for Linux+cuda if omitted)")
parser.add_argument("--task_ids", type=str, default=None,
    help="Comma-separated list of task IDs to evaluate (default: all tasks in suite)")
parser.add_argument("--episode_start", type=int, default=0,
    help="Episode index to start from within each task (for resuming/chunked runs)")
args = parser.parse_args()

# MolmoAct2 / LeRobot LIBERO protocol constants
LIBERO_DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]
TASK_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal":   300,
    "libero_10":     520,
}


def _resolve_use_egl():
    if args.use_egl is not None:
        return args.use_egl
    return sys.platform == "linux" and args.device == "cuda"


def setup_eval_environment():
    """Match MolmoAct2 lerobot-eval env setup (EGL + thread limits)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if _resolve_use_egl():
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"


def episode_seed(episode_idx):
    if args.per_episode_seed:
        return args.eval_seed + episode_idx
    return args.eval_seed


def reset_episode(env, init_states, episode_idx):
    """reset → set_init_state → num_steps_wait dummy steps (official LIBERO order)."""
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


setup_eval_environment()

# Auto-enable video saving for suites
SAVE_FAIL_VIDEOS = args.save_fail_videos or args.suite in ["libero_spatial", "libero_object","libero_goal","libero_10"]
MAX_FAIL_FRAMES_PER_TASK = 3   # max failure JPG frames to save per task
VIDEO_FPS = 10                  # fps for saved videos

# ── Setup output directories ──────────────────────────────────────────────────
output_dir    = Path(args.output_dir) / args.suite
frames_dir    = output_dir / "frames"
fail_dir      = output_dir / "frames" / "failures"
video_dir     = output_dir / "videos" / "failures"

output_dir.mkdir(parents=True, exist_ok=True)
frames_dir.mkdir(parents=True, exist_ok=True)
fail_dir.mkdir(parents=True, exist_ok=True)
if SAVE_FAIL_VIDEOS:
    video_dir.mkdir(parents=True, exist_ok=True)

results_path          = output_dir / "results.csv"
scene_props_path      = output_dir / "scene_properties.csv"
distractor_density_path = output_dir / "distractor_density.csv"
nlp_analysis_path     = output_dir / "nlp_analysis_table.csv"
log_path              = output_dir / "eval.log"

RESULT_FIELDS = [
    "task_id", "suite", "task_name", "instruction", "episode", "episode_seed",
    "success", "n_steps", "max_steps", "distractor_density", "n_distractors",
    "frame_path", "fail_frame_path",
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
    """Append one episode row immediately so progress survives crashes."""
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerows([row])


log = setup_logging()
if results_path.exists():
    results_path.unlink()
log.info("Eval run started")
log.info("Output dir: %s", output_dir)
log.info("Log file:   %s", log_path)

log.info("Suite:              %s", args.suite)
log.info("Episodes per task:  %s", args.n_episodes)
log.info("Device:             %s", args.device)
log.info("Save fail videos:   %s", SAVE_FAIL_VIDEOS)
log.info("num_steps_wait:     %s", args.num_steps_wait)
log.info("eval_seed:          %s", args.eval_seed)
log.info("per_episode_seed:   %s", args.per_episode_seed)
log.info("use_init_states:    %s", args.use_init_states)
log.info("use_egl:            %s", _resolve_use_egl())
log.info("max_policy_steps:   %s", TASK_MAX_STEPS[args.suite])

# ── Load MolmoAct2 policy ─────────────────────────────────────────────────────
log.info("Loading MolmoAct2...")
from lerobot.policies import make_pre_post_processors
from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy
from lerobot.processor import LiberoProcessorStep, PolicyProcessorPipeline

POLICY_PATH = "allenai/MolmoAct2-LIBERO-LeRobot"

policy = MolmoAct2Policy.from_pretrained(
    POLICY_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
policy = policy.to(args.device)
policy.eval()

# Same pre/post pipeline as lerobot-eval (tokenize + normalize + pack model inputs)
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

# ── Load LIBERO benchmark ─────────────────────────────────────────────────────
log.info("Loading LIBERO suite: %s", args.suite)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

task_suite = benchmark.get_benchmark_dict()[args.suite]()
n_tasks    = task_suite.n_tasks
log.info("Found %d tasks", n_tasks)

if args.task_ids:
    task_ids = [int(t.strip()) for t in args.task_ids.split(",") if t.strip()]
else:
    task_ids = list(range(n_tasks))
log.info("Evaluating task_ids: %s", task_ids)

# Camera mapping: LIBERO name → MolmoAct2 name
CAMERA_MAP = {
    "agentview_image":        "image",
    "robot0_eye_in_hand_image": "wrist_image",
}

# ── BDDL parser: distractor density (NLP) ────────────────────────────────────
def _find_bddl_block(text, name):
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
    instances = []
    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        left = line.split(" - ")[0].strip() if " - " in line else line
        instances.extend(left.split())
    return instances


def parse_bddl(bddl_path):
    """Parse LIBERO BDDL for instruction and distractor density metrics."""
    text = Path(bddl_path).read_text()

    instruction = _find_bddl_block(text, "language").strip()
    objects = _parse_object_instances(_find_bddl_block(text, "objects"))
    targets = [t for t in _find_bddl_block(text, "obj_of_interest").split() if t.strip()]

    target_set = set(targets)
    total_objects = len(objects)
    target_objects = len(targets)
    n_distractors = len([o for o in objects if o not in target_set])
    distractor_density = n_distractors / total_objects if total_objects > 0 else 0.0

    return {
        "instruction":        instruction,
        "bddl_file":          str(bddl_path),
        "total_objects":      total_objects,
        "target_objects":     target_objects,
        "target_names":       targets,
        "n_distractors":      n_distractors,
        "distractor_density": distractor_density,
    }


# ── Helper: write frames list to MP4 ─────────────────────────────────────────
def save_video(frames, path, fps=VIDEO_FPS):
    """Save list of HxWx3 uint8 numpy arrays as MP4."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    writer.release()

# ── Results tracking ──────────────────────────────────────────────────────────
all_results    = []
all_scene_props = []

# ── Main evaluation loop ──────────────────────────────────────────────────────
for task_id in task_ids:
    task             = task_suite.get_task(task_id)
    task_bddl_file   = task_suite.get_task_bddl_file_path(task_id)
    task_instruction = task.language
    task_key         = f"{args.suite}_{task_id}"
    bddl_meta        = parse_bddl(task_bddl_file)

    log.info("Task %d/%d: %s", task_id, n_tasks - 1, task_instruction)
    log.info("  distractor_density=%.4f (%d/%d objects)",
             bddl_meta["distractor_density"], bddl_meta["n_distractors"],
             bddl_meta["total_objects"])

    env = OffScreenRenderEnv(**{
        "bddl_file_name":  task_bddl_file,
        "camera_heights":  256,
        "camera_widths":   256,
        "camera_names":    ["agentview", "robot0_eye_in_hand"],
    })

    init_states = None
    if args.use_init_states:
        init_states = task_suite.get_task_init_states(task_id)

    obs = reset_episode(env, init_states, episode_idx=0)

    # ── Scene properties (once per task) ──────────────────────────────────────
    try:
        objects_dict = getattr(env.env, "objects_dict", None) or {}
        object_names = list(objects_dict.keys())
        n_objects = len(object_names)
    except (AttributeError, TypeError) as e:
        log.warning("Could not read objects_dict: %s", e)
        objects_dict, object_names, n_objects = {}, [], -1

    distance = -1.0
    try:
        if "robot0_eef_pos" in obs:
            gripper_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        else:
            gripper_pos = np.asarray(
                env.sim.data.get_site_xpos("gripper0_grip_site"), dtype=np.float64
            )
        target_names = bddl_meta.get("target_names") or list(getattr(env, "obj_of_interest", []) or [])
        for name in target_names:
            obj = objects_dict.get(name)
            body = getattr(obj, "root_body", None) if obj is not None else name
            if not body:
                body = name
            try:
                target_pos = env.sim.data.get_body_xpos(body)
            except (ValueError, KeyError, TypeError):
                continue
            distance = float(np.linalg.norm(gripper_pos - target_pos))
            break
    except (AttributeError, ValueError, TypeError) as e:
        log.warning("Could not compute initial_distance: %s", e)

    all_scene_props.append({
        "task_id":            task_key,
        "suite":              args.suite,
        "instruction":        bddl_meta["instruction"],
        "bddl_file":          bddl_meta["bddl_file"],
        "total_objects":      bddl_meta["total_objects"],
        "target_objects":     bddl_meta["target_objects"],
        "n_distractors":      bddl_meta["n_distractors"],
        "distractor_density": bddl_meta["distractor_density"],
        "n_objects_sim":      n_objects,
        "initial_distance":   round(distance, 4),
    })

    task_failures  = 0   # failure frames/videos saved so far this task
    task_successes = 0
    max_steps      = TASK_MAX_STEPS[args.suite]

    # ── Episode loop ──────────────────────────────────────────────────────────
    if args.episode_start < 0:
        raise ValueError("--episode_start must be >= 0")

    episode_end = (
        min(args.episode_start + args.n_episodes, len(init_states))
        if init_states is not None
        else args.episode_start + args.n_episodes
    )
    n_episodes = episode_end - args.episode_start
    if n_episodes <= 0:
        raise ValueError(
            f"No episodes to run: episode_start={args.episode_start}, episode_end={episode_end}"
        )

    log.info("  episode range: %d-%d (%d episodes)", args.episode_start, episode_end - 1, n_episodes)

    for episode_idx in range(args.episode_start, episode_end):
        obs = reset_episode(env, init_states, episode_idx)

        # Save initial scene frame — episode 0 only
        if episode_idx == 0:
            frame = obs.get("agentview_image")
            if frame is not None:
                fp = frames_dir / f"{task_key}_initial.jpg"
                flipped = np.ascontiguousarray(frame[::-1, ::-1], dtype=np.uint8)
                Image.fromarray(flipped).save(fp)
                frame_path = str(fp)
            else:
                frame_path = ""
        else:
            frame_path = ""

        # Buffer to collect frames for failure video
        episode_frames = [] if SAVE_FAIL_VIDEOS else None

        fail_frame_path = ""

        # ── Step loop ─────────────────────────────────────────────────────────
        policy.reset()
        done    = False
        success = False
        step    = 0

        while not done and step < max_steps:
            mapped_obs = {}

            # Vision inputs: (1,C,H,W) float32 — LiberoProcessor flips 180°, then MolmoAct2 packs tokens
            for libero_key, model_key in CAMERA_MAP.items():
                if libero_key in obs:
                    img = obs[libero_key]
                    mapped_obs[f"observation.images.{model_key}"] = (
                        torch.from_numpy(img.astype(np.float32) / 255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                    )

            # Nested robot_state (batched) so LiberoProcessorStep builds 8-D state
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

            # Language instruction (preprocessor batches str → list)
            mapped_obs["task"] = task_instruction

            # Collect frame for video buffer
            if episode_frames is not None:
                frame = obs.get("agentview_image")
                if frame is not None:
                    flipped = np.ascontiguousarray(frame[::-1, ::-1], dtype=np.uint8)
                    episode_frames.append(flipped)

            # Inference: LIBERO env preprocess → MolmoAct2 preprocess → policy → postprocess
            with torch.inference_mode():
                observation = env_preprocessor(mapped_obs)
                observation = preprocessor(observation)
                action = policy.select_action(
                    observation,
                    inference_action_mode="continuous",
                )
                action = postprocessor(action)

            action_np = action.squeeze(0).cpu().float().numpy()
            obs, reward, done, info = env.step(action_np)

            if reward > 0:
                success = True
                done    = True
            step += 1

        # ── Post-episode: save failure artifacts ──────────────────────────────
        if not success and task_failures < MAX_FAIL_FRAMES_PER_TASK:
            # Save failure JPG frame
            fail_frame = obs.get("agentview_image")
            if fail_frame is not None:
                fail_jpg = fail_dir / f"{task_key}_ep{episode_idx}_fail.jpg"
                flipped = np.ascontiguousarray(fail_frame[::-1, ::-1], dtype=np.uint8)
                Image.fromarray(flipped).save(fail_jpg)
                fail_frame_path = str(fail_jpg)

            # Save failure MP4 video (spatial + object only)
            if SAVE_FAIL_VIDEOS and episode_frames:
                vid_path = video_dir / f"{task_key}_ep{episode_idx}_fail.mp4"
                save_video(episode_frames, vid_path)
                log.info("  saved failure video: %s", vid_path.name)

            task_failures += 1

        if success:
            task_successes += 1

        # Log result (memory + incremental CSV)
        row = {
            "task_id":            task_key,
            "suite":              args.suite,
            "task_name":          task_instruction,
            "instruction":        bddl_meta["instruction"],
            "episode":            episode_idx,
            "episode_seed":       episode_seed(episode_idx),
            "success":            int(success),
            "n_steps":            step,
            "max_steps":          max_steps,
            "distractor_density": bddl_meta["distractor_density"],
            "n_distractors":      bddl_meta["n_distractors"],
            "frame_path":         frame_path,
            "fail_frame_path":    fail_frame_path,
        }
        all_results.append(row)
        append_result_row(results_path, row)

        status = "SUCCESS" if success else "FAIL"
        log.info("  ep %3d/%d %s | steps=%d | seed=%d | success %d/%d",
                 episode_idx, episode_end - 1, status, step,
                 episode_seed(episode_idx), task_successes, episode_idx - args.episode_start + 1)

    sr = task_successes / n_episodes * 100
    log.info("Task %d done — %.1f%% success (%d/%d)", task_id, sr, task_successes, n_episodes)
    env.close()

# ── Save CSVs (scene + NLP; results.csv already written incrementally) ───────
log.info("Results CSV:        %s (%d rows)", results_path, len(all_results))

SCENE_PROP_FIELDS = [
    "task_id", "suite", "instruction", "bddl_file",
    "total_objects", "target_objects", "n_distractors", "distractor_density",
    "n_objects_sim", "initial_distance",
]
DISTRACTOR_DENSITY_FIELDS = [
    "task_id", "suite", "instruction",
    "total_objects", "target_objects", "n_distractors", "distractor_density",
]

with open(scene_props_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=SCENE_PROP_FIELDS)
    w.writeheader(); w.writerows(all_scene_props)
log.info("Scene props saved:    %s", scene_props_path)

with open(distractor_density_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=DISTRACTOR_DENSITY_FIELDS, extrasaction="ignore")
    w.writeheader(); w.writerows(all_scene_props)
log.info("Distractor density:   %s", distractor_density_path)

# ── Summary + NLP analysis table ────────────────────────────────────────────
import pandas as pd
df           = pd.read_csv(results_path)
props_df     = pd.read_csv(scene_props_path)
suite_sr     = df.groupby("suite")["success"].mean() * 100
task_sr      = df.groupby("task_id")["success"].mean() * 100

task_summary = (
    df.groupby(["task_id", "suite"], as_index=False)
    .agg(success_rate=("success", "mean"), n_episodes=("success", "count"),
         avg_steps=("n_steps", "mean"))
)
nlp_table = task_summary.merge(props_df, on=["task_id", "suite"], how="left")
nlp_table.to_csv(nlp_analysis_path, index=False)
log.info("NLP analysis table:   %s", nlp_analysis_path)

log.info("=" * 50)
log.info("SUMMARY — %s", args.suite)
log.info("=" * 50)
log.info("Overall success rate: %.1f%%", suite_sr.iloc[0])
log.info("Per-task success rates:")
for tid, rate in task_sr.items():
    density = props_df.loc[props_df["task_id"] == tid, "distractor_density"]
    density_str = f", density={density.iloc[0]:.3f}" if len(density) else ""
    log.info("  %s: %.1f%%%s", tid, rate, density_str)
log.info("Initial frames:  %s", frames_dir)
log.info("Failure frames:  %s", fail_dir)
if SAVE_FAIL_VIDEOS:
    log.info("Failure videos:  %s", video_dir)
log.info("Full log:        %s", log_path)
log.info("=" * 50)
