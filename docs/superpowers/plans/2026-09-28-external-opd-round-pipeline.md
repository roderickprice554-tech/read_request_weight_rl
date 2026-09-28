# External OPD Round Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train OPD from externally generated hindsight reflections in streaming rounds while every rollout in a round uses one frozen policy.

**Architecture:** Add a small round-protocol module that publishes trajectories, incrementally reads reflections, validates the external completion manifest, and tracks round state. Wire it into the recurrent trainer and gate rollout weight synchronization so actor minibatch updates do not change the rollout policy until the next round.

**Tech Stack:** Python, PyTorch, VERL DataProto/Ray workers, pytest, JSON/JSONL.

## Global Constraints

- One round uses one immutable `policy_version` and checkpoint.
- Each task has exactly eight stochastic trajectories.
- Full training batches contain 16 tasks; a smaller tail is allowed only after `round_manifest.json` reports completion.
- Rollout uses `temperature=1.0` and `top_p=0.98`.
- OPD uses same-token hindsight-conditioned minus original log probabilities, sign alignment, logarithmic clipping, and `A_i^t = A_i * w_i^t`.
- Do not change unrelated trainer behavior.

---

### Task 1: Incremental external reflection store

**Files:**
- Modify: `recurrent/external_reflection_store.py`
- Create: `tests/test_external_reflection_store.py`

**Interfaces:**
- `ExternalReflectionStore(path, *, poll_interval_seconds, timeout_seconds)`
- `refresh() -> None`
- `wait_for_many(trajectories) -> list[ReflectionResult]`

- [ ] Write tests proving a store sees records appended after construction, waits for all requested trajectory IDs, tolerates an incomplete final JSONL line, and rejects conflicting duplicate IDs or policy versions.
- [ ] Run `pytest -q tests/test_external_reflection_store.py` and verify the tests fail because refresh/wait behavior is absent.
- [ ] Implement incremental refresh and bounded polling with no API/model calls.
- [ ] Run `pytest -q tests/test_external_reflection_store.py` and verify it passes.
- [ ] Commit the store and tests.

### Task 2: Round file protocol

**Files:**
- Create: `recurrent/opd_round.py`
- Create: `tests/test_opd_round.py`

**Interfaces:**
- `OPDRoundFiles(root, policy_version)` resolves the isolated round paths.
- `start(checkpoint, expected_task_count, trajectories_per_task, sampling)` atomically writes `round_input.json`.
- `append_trajectories(trajectories)` appends reflection-ready JSONL records.
- `load_complete_manifest()` returns `None` until a valid manifest exists and validates version, checkpoint, task count, and eight trajectories per task.
- `ready_task_groups(trajectories, reflections, consumed)` returns deterministic complete groups only.
- `mark_consumed(trajectory_uids)` atomically updates `trainer_state.json`.

- [ ] Write tests for atomic round input, exact eight-trajectory grouping, deterministic groups, a full 16-task selection, no short batch before completion, a short tail after completion, manifest mismatch, and restart deduplication.
- [ ] Run `pytest -q tests/test_opd_round.py` and verify missing-module failure.
- [ ] Implement the minimal file protocol and grouping helpers.
- [ ] Run `pytest -q tests/test_opd_round.py` and verify it passes.
- [ ] Commit the protocol and tests.

### Task 3: Configuration and trainer integration

**Files:**
- Modify: `pipeline/rl_update.py`
- Modify: `verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `tests/test_run_pipeline.py`
- Create: `tests/test_external_opd_trainer_contract.py`

**Interfaces:**
- External mode config supplies `round_root`, `poll_interval_seconds`, and `timeout_seconds`.
- The trainer publishes each generated `ReflectionTrajectory`, waits for matching reflections, and calls the existing `build_opd_annotations` path.
- The outer epoch is one OPD round; each dataloader item is a 16-task batch and the final smaller dataloader item is allowed after completion.

- [ ] Extend command tests to require `data.train_batch_size=16`, `rollout.n=8`, `temperature=1.0`, and `top_p=0.98`.
- [ ] Add contract tests that inspect trainer helpers to prove external mode uses `wait_for_many`, publishes trajectories before waiting, and never calls `generate_reflections`.
- [ ] Run the focused tests and verify expected failures.
- [ ] Add config fields and minimally replace one-shot `ExternalReflectionStore(...).get_many(...)` with round publication plus incremental waiting.
- [ ] Validate exactly 16 tasks for ordinary batches and eight trajectories per task; permit a smaller batch only with a valid completion manifest.
- [ ] Run the focused tests and existing OPD tests.
- [ ] Commit the integration.

### Task 4: Frozen rollout synchronization and OPD formula regression

**Files:**
- Modify: `verl/workers/fsdp_workers.py`
- Modify: `verl/workers/sharding_manager/fsdp_vllm.py`
- Modify: `verl/workers/sharding_manager/fsdp_sglang.py`
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `tests/test_skill_opd_loss.py`
- Create: `tests/test_rollout_round_freeze.py`

**Interfaces:**
- Worker methods `begin_rollout_round(policy_version)` and `end_rollout_round(policy_version)` control weight-sync gating.
- Sharding managers receive a `sync_weights` flag; memory/KV-cache lifecycle remains unchanged when synchronization is skipped.

- [ ] Write tests proving the first generation in a round synchronizes weights, later generations skip synchronization after actor updates, and the next round synchronizes once again.
- [ ] Add numerical tests for the exact delta, sign-aligned clipping, bounded weight, sign preservation, and `token_advantage = advantage * weight` formula.
- [ ] Run the focused tests and verify failures for missing freeze behavior.
- [ ] Implement the smallest synchronization gate and call it only at round boundaries.
- [ ] Run focused tests, then `pytest -q tests`.
- [ ] Run `git diff --check` and inspect the final diff for unrelated changes.
- [ ] Commit the completed round-freeze implementation.
