nlp_analysis_table_priya_format.csv

This file matches Priya's uploaded nlp_analysis_table schema:
task_id, suite, success_rate, n_episodes, avg_steps, instruction, bddl_file,
total_objects, target_objects, n_distractors, distractor_density, n_objects_sim, initial_distance.

Notes:
- success_rate is represented as a 0-1 fraction, matching Priya's example.
- bddl_file is stored as a relative LIBERO path because the absolute site-packages path depends on the machine.
- total_objects and n_objects_sim are reconstructed from n_distractors / distractor_density when available.
- initial_distance was not logged by our eval file, so it is set to -1.0 as an unavailable placeholder.
