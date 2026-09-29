# VST Question Timestamp Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Default a missing VST-RL `question_timestamp` to `extra_info.duration` and verify one real VST-3B rollout.

**Architecture:** Resolve the causal cutoff once at the `VideoMemoryDataset` boundary. All existing video decoding, caching, provenance, trajectory, reflection, and OPD code continues to consume that resolved value unchanged.

**Tech Stack:** Python, pytest, PyArrow, VERL/Ray, Qwen2.5-VL/VST-3B.

## Global Constraints

- Explicit top-level or `extra_info` question timestamps take priority.
- Missing timestamps fall back only to `extra_info.duration`.
- Missing timestamp and duration remains an error.
- Do not rewrite the source parquet.
- The smoke uses one dataset sample and `rollout.n=1`.

---

### Task 1: Timestamp fallback

**Files:**
- Modify: `recurrent/test/test_opd_trajectory.py`
- Modify: `recurrent/impls/video_memory.py`
- Modify: `docs/EXTERNAL_OPD_PIPELINE_HANDOFF.md`

**Interfaces:**
- Consumes: dataset row keys `question_timestamp`, `extra_info.question_timestamp`, and `extra_info.duration`.
- Produces: resolved positive `question_timestamp: float` used by `process_video(video_end=...)` and trajectory metadata.

- [ ] **Step 1: Write the failing test**

Add a dataset test whose row omits `question_timestamp`, contains
`extra_info={"duration": 12.5}`, and asserts the returned sample and video
processor both receive `12.5`. Retain explicit-timestamp and missing-all-fields
coverage.

- [ ] **Step 2: Run the focused test to verify RED**

Run:
`python -m pytest -q recurrent/test/test_opd_trajectory.py -k question_timestamp`

Expected: the duration-fallback case fails with `missing required 'question_timestamp'`.

- [ ] **Step 3: Implement the minimal fallback**

Use:

```python
question_timestamp = raw_row.get(
    configured_key,
    extra_info.get(configured_key, extra_info.get("duration")),
)
```

Keep the existing numeric and duration-bound validation unchanged.

- [ ] **Step 4: Update the handoff contract**

Document `extra_info.duration` as both the full duration and the fallback query
arrival time when an explicit timestamp is absent.

- [ ] **Step 5: Run focused and related tests**

Run:
`python -m pytest -q recurrent/test/test_opd_trajectory.py recurrent/test/test_reflection.py recurrent/test/test_skill_opd_trajectory.py`

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

Commit the test, implementation, and handoff update together.

### Task 2: Remote one-trajectory smoke

**Files:**
- No tracked files; use `/tmp/vst-opd-one-trajectory-smoke` for derived data and logs.

**Interfaces:**
- Consumes: VST-RL parquet, an existing video file, `/home/bujunru/vlm-repro/models/VST-3B`.
- Produces: one trajectory record plus trainer/worker logs and, if resources permit, one optimizer-step metric record.

- [ ] **Step 1: Create a one-row parquet without question_timestamp**

Select the first VST-RL row whose video exists and preserve its original
`extra_info.duration`; do not inject `question_timestamp`.

- [ ] **Step 2: Run with one rollout**

Set `data.train_batch_size=1`, `actor_rollout_ref.rollout.n=1`, stochastic
sampling, and one total training step.

- [ ] **Step 3: Verify artifacts**

Count complete lines in `trajectories.jsonl`, inspect the round input for
`expected_task_count=1` and `trajectories_per_task=1`, and confirm whether the
optimizer step completed. Report resource failures separately from code-flow
failures.

- [ ] **Step 4: Push the verified implementation**

Fast-forward GitHub `branch_opti` only after focused tests pass; do not include
temporary smoke data or credentials.
