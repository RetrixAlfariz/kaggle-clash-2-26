# Algo-5 — Stopped at user request

Status: stopped on 2026-09-15 before challenger training and before evaluation. All experiment worker processes were terminated. No 82% result has been established.

Completed: the continuation control trained for the fixed additional epochs 3 and 4. Its checkpoint is `output/bitrase-2/algo-5/run/control_epoch4.pt`; its EVAL2 logits were saved, but no accuracy evaluation was run.

Interrupted: the first FIT2 hard-negative mining pass for the challenger, after the log reported 6,400 of 52,582 documents. Its partial in-memory mining output was not saved. Challenger training had not begun, and no challenger checkpoint or completed-run marker exists.

The preceding algo-4 experiment is complete and verified: stable CE control MAP 80.9887%, MBR 81.1790%; ranking challenger MAP 26.7245%, MBR 26.6919%. Do not use the failed ranking challenger. See [algo-4 report](../algo-4/report.md).

Resume only upon a new user instruction. Preserve the saved continuation control and source/input hashes. The current algo-5 runner refuses an existing output directory; it does not yet implement partial-run resume. Add an explicit resume path for the uncompleted challenger stage rather than rerunning over existing artifacts. Finish both fixed heads before the planned paired development evaluation. EVAL2 remains adaptively reused development data.

No experiment remains scheduled or intentionally running. No Kaggle submission, holdout/test access, commit or push was performed.
