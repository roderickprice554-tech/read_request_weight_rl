# Visual-only encoder loader design

## Goal

Keep the independent frozen vision encoder used by rollout embedding reuse, but
avoid constructing a second complete VST-3B model and avoid loading one complete
actor checkpoint per rank under the 32 GiB memory limit.

## Design

- Instantiate the Qwen2.5-VL vision transformer directly from the model's
  `vision_config`.
- Read only checkpoint tensors whose keys start with `visual.` and remove that
  prefix when loading the vision module.
- Preserve checkpoint dtype, freeze all vision parameters, switch the module to
  evaluation mode, and place it on the current CUDA device.
- Replace only the current full-model extraction block in
  `ActorRolloutRefWorker.init_model`; rollout and teacher-cache interfaces remain
  unchanged.
- Fail clearly when no `visual.*` weights exist or when the visual state dict is
  incomplete.
- Load the complete actor checkpoint only on rank 0. Other FSDP ranks construct
  the same model on the meta device and receive initialized parameters through
  the existing `sync_module_states=True` path.
- Preserve tied input/output embeddings. Any meta initialization workaround must
  restore the checkpoint's configured tying before training starts.

## Verification

- Unit test that the loader filters and strips only `visual.*` keys.
- Unit test that loading does not call a full multimodal model constructor.
- Unit test that rank 0 selects real checkpoint loading while nonzero ranks select
  meta initialization, including the tied-embedding model configuration.
- Run the existing LoRA, vLLM vision-counter, and full CPU test suites.
- Re-run the real 2-GPU, 1-task × 8-rollout smoke under the 32 GiB cgroup and
  record peak memory and vision encoder counts.
