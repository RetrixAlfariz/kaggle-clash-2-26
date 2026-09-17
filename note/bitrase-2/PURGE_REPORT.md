# Post-P0 purge — blocked before deletion

Date: 2026-09-15. **Status: not executed. No experiment files were deleted.**

Vian explicitly confirmed the scope: purge experiments after P0, including their dedicated source code, tests, checkpoints, caches and outputs; retain their reports/protocols and retain P0 champion Kaggle **0.8058**. The score is user-reported. No further scope confirmation is needed.

## Audited scope

The planned output directories are `output/bitrase-2/algo-1` through `algo-5` and `output/bitrase-2/structure-diagnostic`. Dedicated implementation/test modules and their bytecode were also inventoried. Existing `src/run_algo1.py` and `src/slot_decoder.py` belong to earlier baseline work and are outside this deletion scope.

The [machine-readable manifest](PURGE_MANIFEST.json) lists exact relative paths, byte counts and SHA-256 hashes: **206 planned files, 16.273 GiB**, including **21 source files, 14 test files and 7 bytecode files**. It also records **139 protected files** covering the complete P0 artifact folder and P0 notes, retained source/tests, and P0's parent encoder/vocabulary run at `output/bitrase-1.algo-6/v1`.

P0's selected head is seed2026/epoch2. Its submission remains at `output/bitrase-2/p0/seed2026/submission.csv`, SHA-256 `587dd76b8cfc3d8b06b5c24cdea56b469a6e8affbb113300638612dccf8e0fc4`. All P0 seeds and caches remain retained under the confirmed scope.

## Execution block

The command execution tool rejected the combined audit/deletion command before execution. A second attempt separated the completed inventory from deletion and used explicit absolute paths for the six output directories with native PowerShell `Remove-Item -LiteralPath ... -Recurse -Force`. That command was also rejected before execution.

The tool supplied only **"rejected: blocked by policy"**; it did not provide a more specific reason. No deletion was attempted through a different shell or filesystem API afterward. The manifest remains a plan, not evidence of completed deletion.

## Remaining work

When deletion is permitted, verify the inventory against current files and preserve the protected hashes; delete only the confirmed post-P0 targets; mark retained experiment notes as historical reports whose executable artifacts have been removed; update the bitrase-2 research index; check remaining source imports and run the retained P0 span-head tests. Do not claim the purge completed or that archived commands remain runnable before those checks pass.

The previous experiment reports remain in place. Algo-5 remains stopped, and no training, evaluation, Kaggle submission, commit or push was initiated during this purge request.
