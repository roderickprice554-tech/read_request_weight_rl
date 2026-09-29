# Shared Video Teacher Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make reflection group-aware, make OPD teacher scoring sparse and embedding-backed, measure real vLLM vision-cache behavior, and complete a one-task/eight-rollout LoRA smoke.

**Architecture:** Keep existing per-trajectory reflection envelopes downstream, but introduce a group request/response boundary upstream. Select valid OPD rows before distributed dispatch and scatter results afterward. Add opt-in instrumentation at the actual vLLM vision call boundary, and add minimal PEFT wrapping to the PPO actor so the real smoke trains adapters only.

**Tech Stack:** Python, PyTorch, VERL `DataProto`, Ray/FSDP, vLLM, PEFT, pytest, Hydra, Bash.

## Global Constraints

- Support arbitrary rollout group sizes; never hard-code eight.
- Preserve the existing actor-update visual embedding path.
- Both actor and external reflection sources use one request per group.
- LoRA smoke uses rank 16, alpha 32, dropout 0, and targets `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- Vision tower is frozen and only adapter parameters are optimized.
- Do not add rollout cache machinery if measurement proves one real vision forward per group/chunk.

---

### Task 1: Worktree-safe tests

**Files:**
- Modify: `recurrent/test/test_skill_opd_actor.py`

**Interfaces:**
- Produces: a repository root derived from `Path(__file__).resolve().parents[2]`.

- [ ] **Step 1: Change the path test to assert the current repository layout**

```python
REPO_ROOT = Path(__file__).resolve().parents[2]
actor_source = (REPO_ROOT / "verl" / "workers" / "actor" / "dp_actor.py").read_text()
```

- [ ] **Step 2: Run the focused test**

Run: `python -m pytest -q recurrent/test/test_skill_opd_actor.py`
Expected: both existing tests pass from an arbitrarily named worktree.

- [ ] **Step 3: Commit**

```bash
git add recurrent/test/test_skill_opd_actor.py
git commit -m "test: resolve OPD sources from worktree root"
```

### Task 2: Group-native reflection protocol

**Files:**
- Modify: `recurrent/skill_opd.py`
- Modify: `recurrent/reflection.py`
- Modify: `recurrent/reflection_sft.py`
- Modify: `recurrent/external_reflection_store.py`
- Modify: `pipeline/reflection_request.py`
- Modify: `scripts/run_external_reflection_worker.py`
- Modify: `recurrent/test/test_reflection.py`
- Modify: `recurrent/test/test_skill_opd_trajectory.py`
- Modify: `recurrent/test/test_external_reflection_store.py`
- Modify: `tests/test_deepseek_reflections.py`

**Interfaces:**
- Produces: `group_reflection_trajectories(trajectories) -> list[list[ReflectionTrajectory]]`.
- Produces: `build_group_reflection_prompt(group, ...) -> str`.
- Produces: `parse_and_validate_group_reflection(text, group, policy_version, ...) -> list[ReflectionEnvelope]`.
- Keeps: `build_opd_annotations(output, trajectories, reflections)` unchanged.

- [ ] **Step 1: Write failing grouping and membership tests**

```python
def test_group_reflection_supports_mixed_group_sizes():
    groups = group_reflection_trajectories([a0, a1, b0])
    assert [[x.trajectory_uid for x in group] for group in groups] == [[a0.trajectory_uid, a1.trajectory_uid], [b0.trajectory_uid]]

def test_group_response_requires_exact_trajectory_membership():
    with pytest.raises(ValueError, match="trajectory membership"):
        parse_and_validate_group_reflection(response_missing_a1, [a0, a1], a0.policy_version)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q recurrent/test/test_reflection.py recurrent/test/test_skill_opd_trajectory.py recurrent/test/test_external_reflection_store.py tests/test_deepseek_reflections.py`
Expected: FAIL because group APIs do not exist and current code sends per-trajectory requests.

- [ ] **Step 3: Implement stable grouping, one-video prompts, strict group parsing, and actor/external batching**

```python
def group_reflection_trajectories(trajectories):
    groups = {}
    for trajectory in trajectories:
        groups.setdefault(trajectory.group_uid, []).append(trajectory)
    return list(groups.values())
```

Return validated per-trajectory envelopes in original trajectory order. Reject group/policy mismatches and non-exact response membership.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q recurrent/test/test_reflection.py recurrent/test/test_skill_opd_trajectory.py recurrent/test/test_external_reflection_store.py tests/test_deepseek_reflections.py`
Expected: all selected tests pass.

- [ ] **Step 5: Commit**

```bash
git add recurrent pipeline scripts tests
git commit -m "feat: compare reflection trajectories by group"
```

### Task 3: Sparse teacher cache with precomputed visual embeddings

**Files:**
- Modify: `verl/trainer/ppo/ray_trainer.py`
- Modify: `verl/workers/actor/dp_actor.py`
- Modify: `recurrent/skill_opd.py`
- Modify: `recurrent/test/test_skill_opd_actor.py`
- Modify: `tests/test_portable_imports.py`

**Interfaces:**
- Produces: `select_valid_teacher_rows(data) -> tuple[DataProto | None, torch.Tensor]`.
- Produces: `scatter_teacher_cache(sparse_cache, valid_rows, full_size, response_length) -> DataProto`.
- Changes: `DataParallelPPOActor.compute_opd_teacher_cache` selects `_forward_micro_batch_embed` whenever `multi_modal_embeds` is present.

- [ ] **Step 1: Write failing sparse selection, all-invalid, scatter, and embed-path tests**

```python
valid_rows = torch.tensor([True, False, True])
sparse, selected = select_valid_teacher_rows(batch)
assert selected.tolist() == [0, 2]
assert sparse.non_tensor_batch["trajectory_uid"].tolist() == ["a", "c"]

full = scatter_teacher_cache(sparse_cache, valid_rows, 3, response_length=2)
assert full.batch["opd_teacher_log_probs"][1].eq(0).all()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q recurrent/test/test_skill_opd_actor.py tests/test_portable_imports.py`
Expected: FAIL because sparse helpers and embed dispatch do not exist.

- [ ] **Step 3: Implement filtering before RPC and scatter after RPC**

For an all-invalid batch, construct detached zero log probabilities and skip `compute_opd_teacher_cache`. For nonempty sparse batches, pad only the sparse batch to the actor world-size divisor, unpad its result, then scatter.

- [ ] **Step 4: Select the embedding forward path in the actor**

```python
forward = self._forward_micro_batch_embed if "multi_modal_embeds" in model_data else self._forward_micro_batch
_, log_probs = forward(model_data, temperature=1.0, calculate_entropy=False)
```

- [ ] **Step 5: Verify GREEN**

Run: `python -m pytest -q recurrent/test/test_skill_opd_actor.py tests/test_portable_imports.py`
Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add verl/trainer/ppo/ray_trainer.py verl/workers/actor/dp_actor.py recurrent/skill_opd.py recurrent/test/test_skill_opd_actor.py tests/test_portable_imports.py
git commit -m "feat: skip invalid OPD teacher rows"
```

### Task 4: Real vLLM vision-forward diagnostics

**Files:**
- Modify: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- Modify: the version-specific vLLM model runner/model wrapper selected by the installed runtime, only if required to reach the actual vision tower.
- Create: `tests/test_vllm_vision_counter.py`

**Interfaces:**
- Produces smoke metrics keyed by `group_uid` and chunk identity with `request_count` and `vision_forward_count`.

- [ ] **Step 1: Write a failing counter aggregation test**

```python
counter.record_request("g", "chunk-0", count=8)
counter.record_vision_forward("g", "chunk-0")
assert counter.snapshot()["g|chunk-0"] == {"request_count": 8, "vision_forward_count": 1}
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_vllm_vision_counter.py`
Expected: FAIL because the counter does not exist.

- [ ] **Step 3: Add opt-in instrumentation at the actual inference vision call**

Expose the snapshot through existing rollout metrics. Do not count the auxiliary actor-update encoder.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_vllm_vision_counter.py`
Expected: PASS.

- [ ] **Step 5: Commit measurement before any cache fix**

```bash
git add verl/workers/rollout tests/test_vllm_vision_counter.py
git commit -m "feat: measure vLLM vision cache calls"
```

### Task 5: PPO LoRA-only updates

**Files:**
- Modify: `verl/trainer/config/ppo_trainer.yaml`
- Modify: `verl/workers/fsdp_workers.py`
- Create: `tests/test_rl_lora.py`

**Interfaces:**
- Adds model config: `lora_rank`, `lora_alpha`, `lora_dropout`, and `lora_target_modules`.
- Guarantees optimizer inputs are exactly parameters with `requires_grad=True`.

- [ ] **Step 1: Write failing LoRA configuration and trainable-parameter tests**

```python
assert actor_config.model.lora_rank == 16
assert all("lora_" in name for name, parameter in model.named_parameters() if parameter.requires_grad)
assert any(not torch.equal(before[name], after[name]) for name in adapter_names)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest -q tests/test_rl_lora.py`
Expected: FAIL because PPO actor PEFT wrapping is absent.

- [ ] **Step 3: Wrap only the language modules with PEFT and build optimizer from adapters**

Use `LoraConfig(r=16, lora_alpha=32, lora_dropout=0, target_modules=[...])`, preserve frozen vision parameters, and log trainable/total parameter counts.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest -q tests/test_rl_lora.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add verl/trainer/config/ppo_trainer.yaml verl/workers/fsdp_workers.py tests/test_rl_lora.py
git commit -m "feat: support LoRA-only PPO updates"
```

### Task 6: Real one-by-eight remote smoke

**Files:**
- Modify: `scripts/run_skill_opd_smoke.sh`
- Modify: `scripts/audit_skill_opd_smoke.py`
- Modify: `verl/trainer/ppo/ray_trainer.py`

**Interfaces:**
- Produces: a JSON smoke report containing sample index, group size, timestamp, memory-step counts, vision counts, teacher row counts, reflection request count, finite loss, adapter update, and peak memory.

- [ ] **Step 1: Extend the smoke audit with exact real-run assertions**

```python
assert report["sample_index"] == 203
assert report["group_size"] == 8
assert report["question_timestamp"] == 237
assert report["memory_steps"] == [4] * 8
assert report["reflection_request_count"] == 1
assert report["adapter_max_change"] > 0
```

- [ ] **Step 2: Verify the audit fails on the old report schema**

Run: `python scripts/audit_skill_opd_smoke.py /tmp/old-report.json`
Expected: FAIL with missing real-smoke fields.

- [ ] **Step 3: Configure the smoke for VST sample 203, one task, eight rollouts, and LoRA-only update**

Set `data.train_batch_size=1`, `actor_rollout_ref.rollout.n=8`, the agreed LoRA values, four memory turns, embed-cache diagnostics, one optimizer step, and no validation/checkpoint save.

- [ ] **Step 4: Run unit regression before the GPU smoke**

Run: `python -m pytest -q recurrent/test tests`
Expected: all tests pass.

- [ ] **Step 5: Run the real smoke and inspect the measured cache result**

Run: `bash scripts/run_skill_opd_smoke.sh`
Expected: eight trajectories, four memory steps each, finite loss, nonzero adapter change, and recorded real vision-forward counts.

- [ ] **Step 6: Apply a minimal rollout cache fix only if measured count is not one**

First reuse normalized multimodal input/cache identity for all requests in the group/chunk. Re-run the same smoke and require `vision_forward_count == 1`; do not add this change when the first measurement is already one.

- [ ] **Step 7: Commit smoke evidence and any measurement-driven fix**

```bash
git add scripts verl tests docs/smoke
git commit -m "test: run one-by-eight LoRA OPD smoke"
```

### Task 7: Final verification and push

**Files:**
- Verify all modified files.

**Interfaces:**
- Produces: pushed `origin/share_video_teacher_mask` with a clean worktree.

- [ ] **Step 1: Run all relevant tests and syntax checks**

Run: `python -m pytest -q recurrent/test tests`
Run: `python -m py_compile recurrent/skill_opd.py recurrent/reflection.py verl/workers/actor/dp_actor.py verl/workers/fsdp_workers.py verl/trainer/ppo/ray_trainer.py`
Expected: all tests and compilation pass.

- [ ] **Step 2: Verify repository state and commits**

Run: `git diff --check && git status --short --branch && git log --oneline origin/share_video_teacher_mask..HEAD`
Expected: clean worktree and only task-related commits.

- [ ] **Step 3: Push after verification**

Run: `git push origin share_video_teacher_mask`
Expected: remote branch advances to the verified implementation commit.
