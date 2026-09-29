# Shared Video, Sparse Teacher Cache, and Group Reflection Design

## Goal

Reduce redundant video and teacher computation in Skill OPD while allowing the
reflection model to compare every rollout from the same task. The implementation
must support arbitrary rollout group sizes and preserve row alignment for the
existing OPD loss.

## Scope

1. Measure the real vLLM vision-tower invocation count for each
   `(group_uid, chunk_identity)` before changing rollout caching.
2. Run OPD teacher scoring only for rows that contain at least one valid OPD
   token and reuse rollout-produced visual embeddings.
3. Submit one reflection request per `group_uid`, with one video and all group
   trajectories, for both actor and external reflection sources.
4. Add PPO actor LoRA support for the constrained-memory real smoke run.
5. Run one real VST-RL sample (index 203) with eight rollouts and one optimizer
   step on VST-3B.

The existing actor-update visual-embedding path remains unchanged.

## Rollout Vision-Encoder Measurement

The diagnostic counter must wrap the vision tower actually used by the vLLM
inference engine. Counting the auxiliary `self.model_vision_encoder` used to
produce actor-update embeddings is insufficient.

Counters are grouped by `(group_uid, chunk_identity)` and report both request
count and vision-forward count. The logic supports any group size:

- one vision forward means the existing vLLM multimodal cache satisfies the
  sharing requirement, so rollout code is not otherwise changed;
- one forward per request proves a miss and activates the smallest compatible
  fix, first sharing normalized multimodal input/cache identity and only using
  precomputed embeddings if that is insufficient;
- intermediate counts are reported as an unexpected state rather than treated
  as a cache hit.

The counter is enabled only for diagnostics/smoke so normal training does not
pay synchronization or logging overhead.

## Group Reflection Protocol

Trajectories are grouped in stable input order by `group_uid`. Every group must
have one policy version, one observed video/chunk identity, and unique
`trajectory_uid` values. Group size may be one or greater.

One request contains the observed video once plus every trajectory's memory
transitions, prediction, and boolean correctness. The prompt explicitly asks
the model to compare successes and failures within the group. The response is a
single JSON object:

```json
{
  "group_uid": "group-a",
  "policy_version": 9,
  "reflections": [
    {
      "trajectory_uid": "group-a:rollout-0",
      "apply_opd": true,
      "episode_skill": "query-independent skill",
      "key_transitions": []
    }
  ]
}
```

The parser rejects mismatched group or policy metadata and missing, duplicate,
extra, or cross-group trajectory IDs. Valid group results are split into the
existing per-trajectory `ReflectionEnvelope` objects so downstream OPD
annotation remains row-aligned. Actor and external reflection paths use the
same request and response contract.

## Sparse OPD Teacher Cache

The trainer computes:

```python
valid_rows = opd_valid_token_mask.any(dim=-1)
```

before the distributed teacher RPC. It selects tensor and non-tensor fields for
only those rows, pads the sparse batch only as required by the worker world
size, performs teacher scoring, removes dispatch padding, and scatters teacher
log probabilities back into the original row shape. Invalid positions contain
zero and remain excluded by the unchanged `opd_valid_token_mask`.

If the complete batch has no valid rows, the trainer skips the teacher RPC and
constructs the detached zero cache directly. Filtering before dispatch avoids
rank-local early exits that could deadlock FSDP or Ulysses collectives.

When `multi_modal_embeds` is present, teacher scoring calls the existing
precomputed-embedding forward path. It must not invoke the vision tower. Sparse
selection preserves alignment between input IDs, position IDs, responses, and
the corresponding visual embeddings.

## PPO LoRA Smoke Mode

PPO actor construction accepts an explicit positive LoRA rank. The smoke uses:

- rank 16;
- alpha 32;
- dropout 0;
- target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, and `down_proj`;
- frozen vision tower;
- only adapter parameters trainable.

The optimizer is constructed from trainable parameters only. Verification must
show that base-model parameters do not change and at least one adapter parameter
changes after the optimizer step. This smoke validates LoRA RL updating, not a
full-parameter RL update.

## Worktree-Safe Tests

Tests must derive the repository root from their own file path. They must not
assume a checkout directory named `VST-RL`.

New tests are written before implementation and cover:

- sparse selection excludes invalid rows from teacher forward;
- an all-invalid batch skips teacher RPC;
- sparse outputs scatter to the original rows;
- teacher scoring selects precomputed visual embeddings and performs no vision
  encoding;
- reflection grouping works for group sizes 1, 8, and mixed sizes;
- actor and external paths each issue one request per group with one video;
- invalid group response membership is rejected;
- the vLLM counter distinguishes one invocation from N invocations;
- LoRA leaves base parameters frozen and updates an adapter.

## Real Remote Smoke

The remote smoke uses:

- dataset: real VST-RL training set, sample index 203;
- model: VST-3B;
- one task and eight rollouts;
- 237-second video;
- parsed `question_timestamp` of 237 seconds;
- four streaming memory steps per trajectory;
- one LoRA-only optimizer step.

The smoke report records the resolved sample index, group size, timestamp,
memory-step counts, vLLM vision-forward counts per chunk, valid teacher row
count, skipped teacher row count, reflection request count, adapter parameter
change, finite loss, and peak GPU memory. A successful run requires all eight
trajectories, four memory steps per trajectory, one reflection request for the
group, no invalid-row teacher forward, finite loss, and a nonzero LoRA adapter
update.

## Non-goals

- Refactoring the existing actor-update multimodal path.
- Enabling full-parameter training in the constrained-memory smoke.
- Hard-coding a rollout group size of eight.
- Adding speculative cache layers when vLLM's existing cache is measured to
  work.
