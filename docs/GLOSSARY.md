# Glossary

Plain-language definitions for terms used in this repo’s MolmoAct2 / LIBERO evaluation scripts, logs, and CSV results.

---

## Core evaluation terms

**Suite**  
A LIBERO task family. This recovery eval uses `libero_object` (pick the right object among look-alikes) and `libero_spatial` (place relative to spatial relations). Other common suites: `libero_goal`, `libero_10` (long-horizon).

**Task**  
One language instruction / BDDL problem inside a suite (e.g. “pick up the alphabet soup and place it in the basket”). Tasks are indexed (`task_id` 0, 1, …).

**Episode**  
One full attempt of a task: reset the scene → run the policy until success or the step limit. We typically run 50 episodes per task.

**Step**  
One control cycle: the policy outputs an action, the simulator advances once. `n_steps` is how many policy steps an episode used.

**Max steps / timeout**  
Each suite has a step budget (280 for Object and Spatial here). If the episode hits that limit without success, `timeout = 1`.

**Success**  
The LIBERO environment’s sparse reward becomes > 0 (goal conditions satisfied). Recorded as `success = 1` or `0`.

**Success rate (SR)**  
Fraction of episodes that succeed, often shown as a percentage (e.g. 487/500 = 97.4%).

**Seed / episode seed**  
Random seed that controls reproducibility. Official MolmoAct2 LIBERO protocol uses base seed `1000` and `episode_seed = 1000 + episode_index`.

**Init state**  
A fixed starting configuration of objects/robot from LIBERO. Using init states makes episodes comparable across runs.

**Wait steps (`num_steps_wait`)**  
Dummy “settle” actions after reset (default 50) so physics settles before the policy starts. Not counted as policy skill.

---

## Robot / simulation terms

**Policy**  
The learned model that maps camera images (+ robot state) to actions. Here: `allenai/MolmoAct2-LIBERO-LeRobot`.

**Action**  
A 7D continuous command: 6 for arm motion + 1 for gripper. In LIBERO, a **positive** gripper value means “close.”

**Body**  
A physical, simulated object with mass that physics acts on (the arm, a bowl, the table, a drawer). Bodies can move, fall, collide.

**Site**  
A massless, non-colliding reference point attached to a body — used purely for measurement/logic, not physics. E.g. a site marking the exact center of a bowl's rim, so code can check "did the gripper pass through here?"

**Marker**  
Informal synonym for *site* — a labeled point of interest tracked in task logic. Not always a strict MuJoCo term, more a descriptive one used in task code/comments.

**Region**  
A defined zone in 3D space (not attached to any body) used for placement/success checks — e.g. "is the bowl's position currently inside this box?" Regions in LIBERO are typically implemented as MuJoCo *sites* under the hood.

**`basket_1_contain_region`**  
Example of a named *region*: the invisible zone representing "inside basket #1." Task success logic checks whether a target object's position falls inside it.

**`get_body_xpos(name)`**  
Returns the current world-frame `[x, y, z]` position (in meters) of a *body's* origin/center — e.g. `get_body_xpos("bowl_1")` → where the whole bowl currently is.

**`get_site_xpos(name)`**  
Returns the current world-frame `[x, y, z]` position (in meters) of a *site* — e.g. `get_site_xpos("basket_1_contain_region")` → where that invisible zone/marker currently is.

**`get_xpos(env, name)`**  
Generic resolver used in this repo's scripts: tries object → site → body → suffix-stripped retry, in that order, and returns whichever `[x, y, z]` position (meters) resolves first, or `None` if nothing matches. Handles the fact that goals sometimes refer to sites (like `basket_1_contain_region`) rather than an object's root body.

**`dist_eef_to(target)`**  
Euclidean distance (meters) between the end-effector's current position and a target body/site/region, computed as `norm(pa - pb)` on two `get_xpos` results. Used in reward shaping and success checks (e.g. "reached" if `dist_eef_to(target) < 0.02`).

**Observation**  
What the policy sees each step: usually third-person (`agentview`) and wrist (`eye_in_hand`) RGB images, plus end-effector pose.

**EEF (end-effector)**  
The robot’s “hand” / gripper tip in 3D space (`robot0_eef_pos`). Distances and path length are measured from the EEF.

**Gripper**  
The parallel fingers that open and close to grasp objects.

**Gripper open amount**  
How open the fingers are, from joint positions (`mean(|gripper_qpos|)`). Large ≈ open (~0.04); small ≈ closed (~0.001).

**Gripper open threshold (`GRIPPER_OPEN_THRESH = 0.015`)**  
Cutoff between “open” and “closed” for counting grasps. Empirically chosen for LIBERO’s Franka gripper—not a physics law.

**Close command**  
The policy *asks* to close (`action[-1] > 0.3`). Counted in `n_close_commands`. Different from the fingers actually being closed in sim.

**Grasp attempt**  
An open→closed transition on the *measured* finger state. Counted in `n_grasp_attempts` / `n_gripper_close_events`.

**Path length**  
Sum of EEF movements over the episode (meters in sim). Longer paths can mean inefficient or retry motion.

**Pick / place**  
**Pick** = object that must be moved; **place** = receptacle it should end on/in (from BDDL goals like `On` / `In`).

**Grounding (visual / object)**  
Whether the robot interacts with the *correct* object named in the instruction (vs a distractor).

**Nearest object at first close**  
Which movable object was closest to the gripper the first time the fingers closed. Used to label correct vs wrong-object grasps.

---

## Scene / language terms

**BDDL**  
Behavior Domain Definition Language — LIBERO’s task files that list objects, language instruction, and goal conditions.

**Instruction / language**  
The natural-language goal for the task (e.g. from BDDL `(:language ...)`).

**Target object(s)**  
Objects listed as “of interest” / required by the goal (what the instruction is about).

**Distractor**  
An object in the scene that is *not* a target. Can confuse object identity or spatial choice.

**Distractor density**  
`n_distractors / total_objects` from BDDL (deterministic scene metadata, not estimated from pixels).

**Initial distance**  
Distance from the EEF to the pick object right after reset (before the policy acts).

---

## Recovery metrics (this repo)

**Recovery**  
Informal: the policy fails a first attempt (e.g. miss / drop / wrong approach) but still succeeds later in the same episode (regrasp, retry).

**Likely recovery (`likely_recovery`)**  
Heuristic flag: success **and** (grasps ≥ 2 **or** steps ≥ suite gate: 150 Spatial / 160 Object). Not ground-truth recovery labels—candidates for video review.

**Suspicious success**  
Same gate as likely recovery; used to decide which success videos to save.

**Gated success video**  
We do **not** save every success. Only “suspicious” successes (up to a few per task) to save disk and focus on hard wins.

---

## Results CSV columns (quick reference)

| Column | Meaning |
|--------|---------|
| `task_id` | Suite + task index (e.g. `libero_spatial_5`) |
| `episode` | Episode index within the task |
| `success` | 1 if goal achieved, else 0 |
| `n_steps` | Policy steps used |
| `timeout` | 1 if failed by hitting max steps |
| `n_grasp_attempts` | Open→closed finger transitions |
| `n_close_commands` | Times the policy commanded close |
| `time_to_first_close` | Step index of first finger close |
| `frac_closed` | Fraction of steps with fingers closed |
| `path_length` | Total EEF travel distance |
| `min_target_dist` | Closest EEF↔pick-object distance during the episode |
| `final_target_dist` | EEF↔pick distance at episode end |
| `min_place_dist` | Closest object↔receptacle distance (place progress) |
| `pick_displacement` | How far the pick object moved from its start pose |
| `eef_z_min` / `eef_z_max` | Lowest / highest gripper height seen |
| `nearest_at_first_close` | Object name nearest gripper at first close |
| `correct_object_at_first_close` | 1 if that object was a pick target |
| `likely_recovery` | 1 if success looked like a retry/regrasp |
| `success_video_gated` | 1 if a gated success video was saved |

Values of `-1` usually mean “not available / never measured” for that metric.

---

## Acronyms

| Term | Expansion |
|------|-----------|
| **EEF** | End-effector (gripper tip) |
| **SR** | Success rate |
| **BDDL** | Behavior Domain Definition Language |
| **LIBERO** | Benchmark of robot manipulation tasks with language goals |
| **MolmoAct2** | Vision-language-action policy evaluated here |
| **RGB** | Color camera image (red/green/blue) |
| **CSV** | Comma-separated values (results tables) |
| **FPS** | Frames per second (saved videos use 10 FPS) |

---

*If a term appears in code or CSVs but not here, open an issue or add a one-line definition in this file.*