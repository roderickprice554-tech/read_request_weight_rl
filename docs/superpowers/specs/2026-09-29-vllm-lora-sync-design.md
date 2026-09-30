# vLLM LoRA Dynamic Sync Design

## Goal

Keep the frozen VST-3B base model resident in vLLM and make every rollout use the latest PPO-trained LoRA adapter without synchronizing the full model.

## Design

- Enable vLLM LoRA support only when `model.lora_rank > 0`.
- At each rollout sharding-manager entry, gather only `lora_*` tensors from the FSDP sharded state dict, write a standard PEFT adapter directory on rank 0, and synchronize ranks.
- Give every export a monotonically increasing adapter name/id. `vLLMRollout.generate_sequences` passes that `LoRARequest` to `LLM.generate`, so vLLM cannot reuse an older adapter cache entry.
- In LoRA mode, skip the existing full-model `load_weights` path. The frozen base checkpoint never changes. Non-LoRA training retains the existing behavior.
- Keep only the current and immediately previous version until the rollout using the current version has completed; cleanup is scoped to files created by this manager.

## Verification

- Unit tests prove LoRA mode enables vLLM LoRA, passes a request to generation, exports only adapter tensors, increments versions, and bypasses full-model synchronization.
- Existing LoRA and 32 GiB loading tests remain green.
- The real sample-203 smoke must complete at least two rollout entries around one PPO update and log different adapter versions without exceeding the 32 GiB cgroup.
