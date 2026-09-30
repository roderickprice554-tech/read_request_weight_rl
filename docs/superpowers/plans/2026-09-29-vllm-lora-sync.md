# vLLM LoRA Dynamic Sync Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every vLLM rollout use the latest PPO-trained LoRA adapter while keeping the frozen VST-3B base unchanged.

**Architecture:** The FSDP/vLLM sharding manager exports only LoRA tensors into a versioned PEFT adapter directory. The rollout owns the corresponding `LoRARequest` and supplies it to every generation call; full base-weight synchronization is skipped only in LoRA mode.

**Tech Stack:** PyTorch FSDP, PEFT, safetensors, vLLM 0.11, pytest

## Global Constraints

- CPU memory must remain within 32 GiB.
- Never construct or synchronize a second complete VST-3B text model.
- Preserve the existing non-LoRA path.
- Make only surgical changes required for dynamic adapter synchronization.

---

### Task 1: Define the adapter synchronization contract

**Files:**
- Create: `tests/test_vllm_lora_sync.py`
- Modify: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`

**Interfaces:**
- Produces: `vLLMRollout.set_lora_adapter(path: str, version: int) -> None`
- Produces: `vLLMRollout.lora_request`

- [ ] Write failing tests asserting vLLM is initialized with LoRA support and generation receives the active `LoRARequest`.
- [ ] Run `pytest -q tests/test_vllm_lora_sync.py` and verify failure is caused by missing LoRA wiring.
- [ ] Add the minimal constructor, setter, and generate-call changes.
- [ ] Re-run the test and verify it passes.

### Task 2: Export only current FSDP LoRA tensors

**Files:**
- Modify: `tests/test_vllm_lora_sync.py`
- Modify: `verl/workers/sharding_manager/fsdp_vllm.py`
- Modify: `verl/workers/fsdp_workers.py`

**Interfaces:**
- Produces: `select_lora_state_dict(state_dict: Mapping) -> dict`
- Consumes: `rollout.set_lora_adapter(path, version)`

- [ ] Add failing tests for adapter-only filtering, version increments, and full-weight-sync bypass.
- [ ] Run the focused test and verify the expected failures.
- [ ] Export a standard PEFT adapter on rank 0, barrier, activate its version on every rank, and retain the old full sync for `lora_rank == 0`.
- [ ] Re-run focused and existing LoRA/32 GiB tests.

### Task 3: Real two-version smoke and publish

**Files:**
- Modify only the smoke script if an explicit second rollout is required for verification.

**Interfaces:**
- Consumes: versioned adapter logs emitted by Tasks 1-2.
- Produces: remote commit on `share_video_teacher_mask`.

- [ ] Run the VST-RL index-203, 1x8, four-memory-step smoke under the 32 GiB cgroup.
- [ ] Verify two rollout entries use increasing adapter ids and no full VST-3B sync occurs.
- [ ] Run `git diff --check` and focused pytest suites.
- [ ] Commit and push `share_video_teacher_mask`.
