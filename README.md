# MolmoAct2 × LIBERO Evaluation: Pick and Parse

**Analyzing the Impact of Visual Scene Complexity on Robot Manipulation with MolmoAct2**

Team: Pick and Parse (Priya, Poojitha, Mounika), CSE D 504

MolmoAct2 (Allen Institute for AI, 2026) is one of the most recent open Vision-Language-Action (VLA) models: it has a visual reasoning mechanism built into its architecture, full LIBERO evaluation support, and publicly available checkpoints, yet no systematic visual failure-mode analysis of it across all four LIBERO suites exists in the literature. This repo runs MolmoAct2 (`allenai/MolmoAct2-LIBERO-LeRobot`) on all four [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) task suites (Spatial, Object, Goal, Long; 40 tasks, 50 episodes/task, 2,000 episodes total) as an **inference-only** pipeline (no training happens here), extracting scene-complexity features directly from the simulator state (object count, gripper-to-target distance, BDDL distractor density) with zero extra models or manual annotation, so we can identify *which visual/linguistic scene properties predict where MolmoAct2 fails*.

## Research question

How does MolmoAct2's task success rate vary across LIBERO task suites, and which scene properties (object density, spatial layout, task length, distractor density: total objects / target objects named in the instruction) explain where the model succeeds vs. fails?

- **Minimal goal:** per-task success rates across all 2,000 episodes; a baseline profile of which suite is easiest/hardest.
- **Ambitious goal:** Spearman correlation between scene properties and success rate, a qualitative failure gallery, and (in the follow-on NLP phase) whether visual clutter and distractor density are correlated predictors of failure.
- **Success criterion:** suite-level success rates differ by >10 points and/or at least one scene property correlates with success at `|r| > 0.3, p < 0.05`. A null result (no correlation) is also a valid, reportable finding; it would suggest MolmoAct2 is robust to the complexity variation present in LIBERO.

## Motivation

As VLA models move closer to real-world robotic deployment, aggregate task success rates alone are insufficient to assess their reliability. A model that performs well overall may still fail in visually cluttered, spatially complex, or long-horizon environments. Characterizing these failure conditions gives deeper insight into model behavior, supporting the development of more robust VLA systems and their safe deployment.

## Use cases

This analysis can help VLA researchers improve model architectures, benchmark designers create more diagnostic evaluation tasks, and robotics practitioners identify conditions where additional safeguards may be needed before deployment in settings such as manufacturing, logistics, healthcare, and domestic assistance.

## Approach & rationale

Why this design, specifically:

- **Scene properties come from the simulator, not a second vision model.** Object count, distractor density, and gripper-to-target distance are read directly from LIBERO's own state (BDDL files, sim positions) rather than estimated by a detector. That removes a whole source of noise/confound from the analysis: any correlation we find is between MolmoAct2's behavior and *ground-truth* scene complexity, not between MolmoAct2 and some other model's guess at scene complexity. It's also free: zero extra GPU time, per the proposal's constraint.
- **LIBERO's four suites double as the complexity axis.** Rather than inventing a new complexity metric, we use Spatial/Object/Goal/Long as-is: LIBERO's own designers already ordered these by increasing difficulty, so suite label is a pre-validated proxy we don't have to justify from scratch.
- **Distractor density is defined from the BDDL ground truth** (`total objects / target objects`), not from counting objects in an image. Same reasoning as above: deterministic and reproducible, not dependent on a second model's accuracy.
- **Spearman, not Pearson, for the correlation analysis.** Success rate is bounded in [0, 1] and scene properties (object count, distance) aren't guaranteed to relate to it linearly; Spearman only assumes a monotonic relationship, which is the weaker, more defensible assumption here.
- **Fixed eval seed, per-episode seeding, and LIBERO's built-in init states** (`--eval_seed 1000`, `--per_episode_seed`, `--use_init_states`) match the protocol MolmoAct2's own published LIBERO numbers were evaluated under, so our success rates are comparable to prior reported results rather than an artifact of a different eval setup.

## Requirements

- **GPU:** MolmoAct2 inference at `bfloat16` uses ~14 GB VRAM. A full suite (50 episodes × 10 tasks) takes roughly 1–2 hours on CUDA; CPU/MPS is much slower and only recommended for a smoke test.
- **Python:** 3.12 (3.13 also confirmed working with the pinned `lerobot==0.6.0`).
- **Disk:** ~10 GB for the MolmoAct2 checkpoint (downloaded automatically, cached by `huggingface_hub`).
- **LIBERO** requires MuJoCo and a rendering backend (EGL headless on Linux+CUDA, native OpenGL elsewhere). See platform notes below.

## Setup

### 1. Create the environment (macOS / Linux / Windows)

```bash
conda create -n molmoact2-libero python=3.12
conda activate molmoact2-libero
```

### 2. Install LeRobot with LIBERO extras

```bash
pip install "lerobot[libero]"
```

**Linux only**: LeRobot's LIBERO extra needs these build deps first:

```bash
sudo apt-get install cmake build-essential python3-dev pkg-config \
  libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
  libswscale-dev libswresample-dev libavfilter-dev
```

### 3. Install ffmpeg (for the failure-video writer)

```bash
conda install ffmpeg -c conda-forge
```

macOS/Linux verify:
```bash
ffmpeg -encoders 2>/dev/null | grep libsvtav1
```
Windows (PowerShell) verify:
```powershell
ffmpeg -encoders 2>$null | Select-String libsvtav1
```

### 4. Clone and install LIBERO

`libero` isn't on PyPI, so clone the official repo and install it editable. This is what the `-e` line in `requirements.txt` originally pointed to on a contributor's machine, and what you need to reproduce locally:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```

If `import libero` still fails afterward, point site-packages at it directly:
```bash
python -c "import site; print(site.getsitepackages()[0])"
# then append your LIBERO path to a .pth file in that directory, e.g.:
#   echo "/path/to/LIBERO" >> <site-packages>/libero.pth      (macOS/Linux)
#   echo C:\path\to\LIBERO >> <site-packages>\libero.pth      (Windows)
```
When LIBERO prompts *"Do you want to specify a custom path for the dataset folder? (Y/N)"*, answer **N**.

### 5. Install remaining pinned dependencies

```bash
pip install -r requirements.txt
```

This installs `robosuite==1.4.1`, `bddl==1.0.1`, `gym==0.25.2`, `pandas`, `scipy`, `matplotlib`, etc., all version-pinned because LIBERO/robosuite/bddl are sensitive to version drift.

### 6. Rendering backend

- **Linux + CUDA:** `eval_molmoact2.py` auto-enables EGL headless rendering (`MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`); no action needed. Override with `--use_egl` / `--no-use_egl` if needed.
- **macOS / Windows:** EGL headless isn't available; the script leaves rendering on the platform default (a display/GL context is required; this is what `OffScreenRenderEnv` uses under the hood). If you hit MuJoCo/OpenGL errors on Windows, running under WSL2 (treated as Linux) is the most reliable path for unattended/headless runs.

### 7. Verify the install

```bash
python -c "from libero.libero import benchmark; print('libero OK')"
python -c "from robosuite.environments.manipulation.single_arm_env import SingleArmEnv; print('robosuite OK')"
python -c "import bddl; print('bddl OK')"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
nvidia-smi   # confirms GPU visibility if using --device cuda
```

The MolmoAct2 checkpoint (`allenai/MolmoAct2-LIBERO-LeRobot`, Apache 2.0, ~10 GB) downloads automatically on first run via `from_pretrained`, no manual download step.

## Usage

Run one suite at a time:

```bash
python eval_molmoact2.py --suite libero_spatial --n_episodes 50 --device cuda
python eval_molmoact2.py --suite libero_object  --n_episodes 50 --device cuda
python eval_molmoact2.py --suite libero_goal    --n_episodes 50 --device cuda
python eval_molmoact2.py --suite libero_10      --n_episodes 50 --device cuda
```

(`libero_10` is LIBERO's internal name for the Long suite.) For a quick smoke test before a full overnight run, use `--n_episodes 1` and `--device cpu`.

Key flags (see `python eval_molmoact2.py --help` for the full list):

| Flag | Default | Notes |
|---|---|---|
| `--suite` | *required* | `libero_spatial`, `libero_object`, `libero_goal`, or `libero_10` |
| `--n_episodes` | 50 | Episodes per task |
| `--device` | `cuda` | `cuda`, `mps`, or `cpu` |
| `--output_dir` | `outputs/custom_eval` | Root output directory |
| `--eval_seed` | 1000 | Matches MolmoAct2's published eval protocol |
| `--num_steps_wait` | 50 | Settle steps after reset, matches MolmoAct2 protocol |
| `--use_init_states` / `--no-use_init_states` | on | Use LIBERO's fixed per-episode init states (reproducibility) |
| `--per_episode_seed` / `--no-per_episode_seed` | on | Derive episode seed as `eval_seed + episode_idx`; off reuses `eval_seed` for every episode |
| `--use_egl` / `--no-use_egl` | auto | Force EGL headless rendering on/off; auto-detects Linux+CUDA if omitted |
| `--save_fail_videos` | off | Rarely needed manually; the script already auto-enables failure videos for all four suites ([eval_molmoact2.py:96](eval_molmoact2.py#L96)) |

Each suite takes roughly 1–2 hours on CUDA at `n_episodes=50`; total GPU time for all four suites is ~4–6 hours. Runs are independent per suite and can be split across teammates/machines and merged afterward (all outputs key on `task_id`).

### Team assignments (this run)

| Teammate | Suite(s) | Episodes | Device | Time |
|---|---|---|---|---|
| Priya | `libero_object`, `libero_spatial` | 500 + 500 = 1,000 | CUDA | ~30–45 min each |
| Poojitha | `libero_goal` | 500 | CUDA | ~30–45 min |
| Mounika | `libero_10` (Long) | 500 | CUDA | ~45–60 min |

```bash
# Priya
python eval_molmoact2.py --suite libero_object  --n_episodes 50 --device cuda
python eval_molmoact2.py --suite libero_spatial --n_episodes 50 --device cuda

# Poojitha
python eval_molmoact2.py --suite libero_goal --n_episodes 50 --device cuda

# Mounika
python eval_molmoact2.py --suite libero_10 --n_episodes 50 --device cuda
```

After all four suites finish, merge each suite's `nlp_analysis_table.csv` (see **Analysis**) before running the correlation analysis.

## Troubleshooting

- **LIBERO/MuJoCo/robosuite version mismatches** are the most common setup failure. Stick to the pinned versions in `requirements.txt` (`robosuite==1.4.1`, `bddl==1.0.1`, `gym==0.25.2`); LIBERO is sensitive to drift here.
- **`import libero` fails after `pip install -e .`:** confirm the `.pth` file (Setup step 4) points at the cloned LIBERO directory, and that you're in the same conda env you installed it into.
- **OOM on GPU:** MolmoAct2 needs ~14 GB VRAM at `bfloat16`; free other processes or fall back to `--device cpu` for a reduced-episode smoke test.
- **No visible GPU:** check with `nvidia-smi`; use `--device cpu` (slow) or `--device mps` on Apple Silicon.
- **Crashed suite mid-run:** `results.csv` is appended per-episode (survives crashes); re-running the same `--suite` will overwrite prior results for that suite, so back up partial CSVs first if you want to keep them.

## Data Requirements

- **LIBERO Simulation Environment**: open-source robot manipulation benchmark (MIT License), available via `pip install lerobot[libero]`. Provides 40 manipulation tasks across four task suites (Spatial, Object, Goal, and Long), including RGB observations, language instructions, and simulator state for evaluating VLA models. Benchmark repository: https://github.com/Lifelong-Robot-Learning/LIBERO. Requires Linux with the MuJoCo rendering backend (`MUJOCO_GL=egl`) for evaluation.

- **Scene State Data**: scene properties, including object count and initial gripper-to-target distance, are extracted directly from the LIBERO simulator state at the start of each episode. These features are used to analyze the relationship between visual scene complexity and task success.

- **MolmoAct2-LIBERO-LeRobot Checkpoint**: Apache 2.0 licensed, publicly available pretrained model hosted on Hugging Face (~10 GB). Fine-tuned specifically for all four LIBERO task suites; serves as the VLA model evaluated in this project. Model repository: https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot.

- **GPU Server (24 GB VRAM)**: MolmoAct2 inference requires approximately 14 GB of VRAM using bfloat16 precision. Running the planned 2,000 evaluation episodes is expected to require 4–6 hours of GPU time. Scene property extraction and subsequent data analysis are performed on the CPU using Python libraries such as pandas and matplotlib.

## Input

At every control step, `eval_molmoact2.py` assembles a `mapped_obs` dict ([eval_molmoact2.py:380-411](eval_molmoact2.py#L380-L411)) and runs it through `env_preprocessor` (`LiberoProcessorStep`) then `preprocessor` (MolmoAct2's tokenize/normalize/pack pipeline) before calling `policy.select_action()`. The model receives:

- **Two RGB camera views**, each `(1, C, H, W)` float32, normalized to `[0, 1]`:
  - `agentview_image` → `observation.images.image` (third-person view)
  - `robot0_eye_in_hand_image` → `observation.images.wrist_image` (wrist camera)
  - Camera mapping follows the official checkpoint's convention ([eval_molmoact2.py:203-206](eval_molmoact2.py#L203-L206)). Both are rendered at 256×256 by `OffScreenRenderEnv` and flipped 180° to correct MuJoCo/OpenGL's orientation convention.
- **Robot proprioceptive state**, nested and batched into an 8-D vector by `LiberoProcessorStep` ([eval_molmoact2.py:393-408](eval_molmoact2.py#L393-L408)):
  - end-effector position `robot0_eef_pos` (3-D)
  - end-effector orientation `robot0_eef_quat` (4-D)
  - gripper joint position `robot0_gripper_qpos` (2-D, only 1 dim used downstream)
- **Natural language instruction**: the task's `task.language` string from the LIBERO/BDDL task definition, passed as `task` ([eval_molmoact2.py:411](eval_molmoact2.py#L411)).

## Output

- **Per inference step:** a continuous 7-DoF action (`dx, dy, dz, droll, dpitch, dyaw, gripper`), predicted with `inference_action_mode="continuous"` ([eval_molmoact2.py:424-427](eval_molmoact2.py#L424-L427)), unnormalized by `postprocessor`, then applied to the simulator via `env.step(action_np)`.
- **Per episode:** a success flag, step count, seed, and (for episode 0 or on failure) a saved frame/video, written as one row to `results.csv` ([eval_molmoact2.py:459-475](eval_molmoact2.py#L459-L475)).
- **Per task:** aggregated success rate + scene properties (object counts, distractor density, initial gripper-to-target distance) written to `scene_properties.csv` / `distractor_density.csv`.
- **Per run:** all of the above merged into `nlp_analysis_table.csv`, plus `eval.log`, initial/failure frames, and failure videos; full list in **Repository contents** below.

## Metrics used for evaluation

All computed directly in `eval_molmoact2.py` / the CSVs it produces, no external eval harness:

- **Episode success (binary)**: an episode is marked successful the instant the environment reward exceeds zero (`reward > 0`), which ends the episode early ([eval_molmoact2.py:433-435](eval_molmoact2.py#L433-L435)).
- **Steps to completion (`n_steps`)**: number of environment steps taken before success or before hitting the suite's step cap, `TASK_MAX_STEPS` ([eval_molmoact2.py:54-59](eval_molmoact2.py#L54-L59)): 280 (Spatial/Object), 300 (Goal), 520 (Long).
- **Per-task success rate**: `task_successes / n_episodes * 100`, logged at the end of each task's episode loop ([eval_molmoact2.py:482-483](eval_molmoact2.py#L482-L483)).
- **Per-suite success rate**: `df.groupby("suite")["success"].mean() * 100` over all logged episodes, the top-line summary metric printed at the end of a run ([eval_molmoact2.py:511](eval_molmoact2.py#L511)).
- **Average steps per task (`avg_steps`)**: mean `n_steps` grouped by `task_id`/`suite`, included in `nlp_analysis_table.csv` ([eval_molmoact2.py:514-519](eval_molmoact2.py#L514-L519)).
- **Distractor density**: `n_distractors / total_objects` per task, parsed from the BDDL file's object/target lists ([eval_molmoact2.py:236-258](eval_molmoact2.py#L236-L258)); used as an independent variable against success rate, not a success metric itself.
- **Initial gripper-to-target distance**: Euclidean norm between the gripper site position and the target object's body position at episode start ([eval_molmoact2.py:309-330](eval_molmoact2.py#L309-L330)); the other independent variable for the correlation analysis (see **Analysis**).

## Repository contents

| File | Purpose |
|---|---|
| [eval_molmoact2.py](eval_molmoact2.py) | Runs MolmoAct2 inference over one LIBERO suite, logs results and scene properties, saves failure frames/videos. |
| [requirements.txt](requirements.txt) | Pinned Python dependencies (see note on LIBERO below; it isn't pip-installable from PyPI and must be cloned separately). |

Running `eval_molmoact2.py` (see **Usage** above) generates everything else needed for analysis under `outputs/custom_eval/<suite>/`:

| Output | Contents |
|---|---|
| `eval.log` | Timestamped run log (also printed to console). |
| `results.csv` | Per-episode success, step count, seed, distractor fields. |
| `scene_properties.csv` | Per-task BDDL/simulator metrics: object counts, distractor density, initial gripper-to-target distance. |
| `distractor_density.csv` | NLP-phase subset: `task_id, suite, total_objects, target_objects, distractor_density, n_distractors`. |
| `nlp_analysis_table.csv` | `scene_properties.csv` merged with per-task success rate: the table the correlation analysis reads. |
| `frames/` | One initial scene frame per task (episode 0). |
| `frames/failures/` | Up to 3 failure frames per task. |
| `videos/failures/` | MP4 of up to 3 failed episodes per task. |

## Analysis

Once one or more suites have been run, `nlp_analysis_table.csv` (per suite, under each suite's output folder) already contains success rate merged with scene properties. Concatenate across suites and run the correlation analysis from the project plan:

```python
import pandas as pd
from scipy.stats import spearmanr

df = pd.concat([
    pd.read_csv("outputs/custom_eval/libero_spatial/nlp_analysis_table.csv"),
    pd.read_csv("outputs/custom_eval/libero_object/nlp_analysis_table.csv"),
    pd.read_csv("outputs/custom_eval/libero_goal/nlp_analysis_table.csv"),
    pd.read_csv("outputs/custom_eval/libero_10/nlp_analysis_table.csv"),
])

# Primary CV result: suite-level success rate
suite_success = df.groupby("suite")["success_rate"].mean()

# Scene-property correlations (CV phase)
r_dist, p_dist = spearmanr(df["initial_distance"], df["success_rate"])
r_obj, p_obj   = spearmanr(df["n_objects_sim"], df["success_rate"])

# Distractor density correlation (NLP phase)
r_dd, p_dd = spearmanr(df["distractor_density"], df["success_rate"])
```

Suggested figures (per the project plan):
1. Suite-level success rate bar chart (primary CV result).
2. Task-level success rate heatmap within each suite.
3. Success rate vs. suite complexity scatter (Spatial=1, Object=2, Goal=3, Long=4).
4. Distractor density vs. success rate scatter, colored by suite (primary NLP result).
5. Qualitative failure gallery: hand-picked frames from `frames/failures/`, pairing easy (high-success) vs. hard (low-success) scenes.

## Results & conclusions

*Runs are in progress across the team; this section is a fill-in template so whoever merges the final CSVs can drop numbers straight in from the **Analysis** snippet above.*

### Suite-level success rates (primary CV result)

| Suite | Success rate | Episodes run | Notes |
|---|---|---|---|
| `libero_spatial` | `[XX.X]%` | `[N]` / 500 | |
| `libero_object` | `[XX.X]%` | `[N]` / 500 | |
| `libero_goal` | `[XX.X]%` | `[N]` / 500 | |
| `libero_10` (Long) | `[XX.X]%` | `[N]` / 500 | |

Best suite: `[suite]` at `[XX.X]%`. Worst suite: `[suite]` at `[XX.X]%`. Spread: `[XX.X]` points ( **meets** / **does not meet** the >10-point success criterion).

### Scene-property correlations (Spearman)

| Property | r | p | `|r| > 0.3, p < 0.05`? |
|---|---|---|---|
| `initial_distance` (`r_dist`, `p_dist`) | `[r]` | `[p]` | `[yes/no]` |
| `n_objects_sim` (`r_obj`, `p_obj`) | `[r]` | `[p]` | `[yes/no]` |
| `distractor_density` (`r_dd`, `p_dd`) | `[r]` | `[p]` | `[yes/no]` |

### Failure gallery

3–5 hand-picked frames from `frames/failures/`, pairing an easy (high-success) task against a hard (low-success) one, each with a one-line explanation of what likely went wrong (e.g., gripper missed target, wrong object grasped, task timed out at `max_steps`).

### Conclusion

State plainly whether the proposal's success criterion was met (suite spread >10 points and/or at least one `|r| > 0.3, p < 0.05`), and what that implies: e.g., which suite MolmoAct2 struggles with most, whether visual clutter (object count/distractor density) or spatial difficulty (initial distance) is the better predictor of failure, and whether this matches or contradicts the intuition that harder LIBERO suites should correlate with lower success. If a suite had to be skipped (e.g., a crash), say so explicitly and note that three completed suites still satisfies the proposal's risk-mitigation plan below.

## Risks & mitigations

Carried over from the project proposal, with the decision rules we committed to:

- **LIBERO/MuJoCo rendering setup challenges on a new machine.** Mitigation: allocate up to 3 days for environment setup and validation. Decision rule: if LIBERO/MolmoAct2 integration isn't running by Jul 5, switch to the MolmoAct v1 checkpoint to keep the schedule.
- **Limited variation in scene complexity across suites**, which could mute correlations. Mitigation: extend to LIBERO-90 (90 additional tasks) for more scene diversity and statistical power (~2 extra GPU hours) if needed.
- **High overall model performance leaving too few failures to analyze.** Mitigation: shift the failure-mode analysis toward the harder LIBERO-Long tasks, where prior work reports lower success rates.
- **A suite crashes mid-run.** Mitigation: skip it and analyze the remaining three suites, still a valid project per the proposal's success criteria.

## Ethical considerations

Only open-source models (MolmoAct2, Apache 2.0) and benchmarks (LIBERO, MIT License) are used; no proprietary data, human subjects, or personal information. Results objectively evaluate a public model on a public benchmark, purely to surface failure conditions that inform safer deployment (e.g., where added safeguards or human oversight may be needed), not to misrepresent the model. All experiments run in the LIBERO MuJoCo simulator against a simulated Franka arm, so there is no physical robot hardware and no risk of physical harm.

## Licenses & attribution

- **LIBERO**: MIT License. [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
- **MolmoAct2-LIBERO-LeRobot checkpoint**: Apache 2.0. [allenai/MolmoAct2-LIBERO-LeRobot](https://huggingface.co/allenai/MolmoAct2-LIBERO-LeRobot)
- **LeRobot**: evaluation framework used for the policy/processor pipeline. [huggingface/lerobot](https://github.com/huggingface/lerobot)
- No physical robot hardware, human subjects, or proprietary data are used; all experiments run in the LIBERO MuJoCo simulator.

## References

1. Wang et al. (2026). *MolmoAct2: Action Reasoning Models for Real-World Deployment.* Allen Institute for AI. arXiv:2605.02881.
2. Liu et al. (2023). *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning.* NeurIPS 2023.
3. Cadene et al. (2026). *LeRobot: An Open-Source Library for End-to-End Robot Learning.* ICLR 2026. arXiv:2602.22818.
4. Zhou et al. (2025). *LIBERO-PRO: Towards Robust and Fair Evaluation of VLA Models Beyond Memorization.* arXiv:2510.03827.
5. Kim et al. (2024). *OpenVLA: An Open-Source Vision-Language-Action Model.* CoRL 2024.
6. Shukor et al. (2025). *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.* Hugging Face.
7. Zhen et al. (2025). *TraceVLA: Visual Trace Prompting Enhances Spatial-Temporal Awareness for Generalist Robotic Policies.* arXiv:2412.10345.
8. Lifelong Robot Learning. *LIBERO Benchmark Repository.* GitHub. https://github.com/Lifelong-Robot-Learning/LIBERO
