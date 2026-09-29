# 32 GiB Model Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the real two-GPU 1×8 LoRA smoke inside a 32 GiB cgroup without constructing duplicate full VST-3B models.

**Architecture:** Rank 0 loads actor checkpoint tensors while nonzero FSDP ranks construct on meta and receive parameters through `sync_module_states=True`. The rollout embedding cache owns a separate frozen vision module loaded only from `visual.*` safetensor entries.

**Tech Stack:** PyTorch FSDP, Transformers Qwen2.5-VL, safetensors, pytest, vLLM.

## Global Constraints

- Preserve two-rank FSDP and two-way vLLM tensor parallelism.
- Preserve tied input/output embeddings and existing training semantics.
- Never construct a second complete VST-3B to obtain the vision encoder.
- The acceptance test is the real sample-203, two-GPU, 1×8 LoRA smoke under the 32 GiB cgroup.

---

### Task 1: Rank-0-only actor checkpoint loading

**Files:**
- Modify: `verl/workers/fsdp_workers.py`
- Test: `tests/test_32g_model_loading.py`

**Interfaces:**
- Produces: `_actor_init_uses_meta(rank: int, world_size: int) -> bool`
- Consumes: existing `get_init_weight_context_manager`, `init_fn`, and FSDP `sync_module_states=True`.

- [ ] Write a failing test proving rank 0 uses real initialization and rank 1 uses meta initialization even when `tie_word_embeddings=True`.
- [ ] Run `python -m pytest -q tests/test_32g_model_loading.py` and confirm the new test fails for the missing selection helper.
- [ ] Implement the minimal rank-aware initialization selection and restore weight tying before FSDP wrapping.
- [ ] Run the focused test and existing LoRA tests.
- [ ] Commit with `feat: load FSDP actor checkpoint on rank zero`.

### Task 2: Visual-only checkpoint loader

**Files:**
- Modify: `verl/workers/fsdp_workers.py`
- Test: `tests/test_32g_model_loading.py`

**Interfaces:**
- Produces: `load_visual_encoder(model_path, vision_config, device) -> torch.nn.Module`
- Consumes: checkpoint `model.safetensors.index.json` and tensors under the exact `visual.` prefix.

- [ ] Write failing tests proving only `visual.*` tensors are selected, their prefix is stripped, and the full multimodal auto-model constructor is never called.
- [ ] Run the focused tests and confirm failure for the missing loader.
- [ ] Instantiate the Qwen2.5-VL vision transformer from `vision_config`, stream only referenced safetensor shards, load strictly, freeze, convert to bf16, move to the current CUDA device, and set eval mode.
- [ ] Replace the full-model extraction block with `load_visual_encoder`.
- [ ] Run focused, LoRA, and vLLM vision-counter tests.
- [ ] Commit with `feat: load standalone vision encoder weights`.

### Task 3: Full verification and real smoke

**Files:**
- Modify only if a failing test identifies a defect in Tasks 1 or 2.

- [ ] Run `python -m pytest -q` and require zero failures.
- [ ] Run Python compilation, `bash -n scripts/run_real_1x8_lora_smoke.sh`, and `git diff --check`.
- [ ] Run `scripts/run_real_1x8_lora_smoke.sh` under the existing 32 GiB cgroup.
- [ ] Verify sample index 203, group size 8, question timestamp 237, four memory steps per trajectory, one external reflection request, sparse teacher row counts, successful LoRA update, and recorded vision encoder counts.
- [ ] If the actual vLLM vision item count is one per group/chunk, keep the cache implementation unchanged; if it is eight, add the smallest sharing fix with a failing regression test first.
- [ ] Commit any verified smoke-only correction and push `share_video_teacher_mask`.
