# External OPD Pipeline Handoff

## Required external inputs

Each dataset row must provide:

| Field | Meaning |
| --- | --- |
| `video_context` | Video path, relative to `recurrent.video_memory.config.video_root` or absolute. |
| `question` / prompt fields | The question and the model-ready QA prompt. |
| `question_timestamp` | Required Q-arrival time in seconds on the source-video timeline. Frames after this time are forbidden. |
| answer/reward fields | Ground truth consumed by the configured environment reward function. Ground truth is not copied into the reflection skill text. |
| `extra_info.duration` | Full source-video duration; used to validate `0 < question_timestamp <= duration`. |

Runtime inputs are the local VST checkpoint, training/validation parquet files,
video root, reward function configuration, output directory, external reflection
API URL/model, and `DEEPSEEK_API_KEY`. The key is read from the environment and
is never written to resolved configuration or JSONL artifacts.

## Causal video policy

The dataset decoder receives `video_start=0` and
`video_end=question_timestamp`. The local model reads the resulting frames in
chronological chunks. Memory-generation prompts do not contain the question.
The question is exposed only on the final answer turn.

`final_chunk_mode` selects one of two paths:

- `raw_video`: generate memories for chunks `1..n-1`, then answer from those
  memories plus raw pre-question chunk `n` and Q.
- `memory_only`: generate memories for chunks `1..n`, then answer from the
  complete memory and Q without raw video.

No frame after `question_timestamp` may enter memory generation, the final
answer, or the external reflection request.

## Module inputs and outputs

### `VideoMemoryDataset`

- Input: dataset row, video root, `question_timestamp`, sampling limits.
- Output: sampled pre-question video tensor, source video path, Q-arrival time,
  tokenized final QA prompt, and video timing metadata.

### `VideoMemoryAgent`

- Input: pre-question video tensor and hidden final QA prompt.
- Output per memory turn: previous memory, current chunk boundary, generated
  memory tokens, updated memory, group/trajectory IDs, and `policy_version`.
- Final output: generated answer and final-turn marker, using the configured
  final-chunk path.

### Environment reward

- Input: generated final answer plus the dataset answer/reward metadata.
- Output: one scalar final reward per trajectory. The reward is propagated to
  all turns of that trajectory and converted to group advantages across the
  eight samples of the same task.

### External trajectory publisher

- Input: the complete trajectory after reward: pre-question video reference,
  Q, generated answer, correctness/reward outcome, memory transitions, chunk
  boundaries, `group_uid`, `trajectory_uid`, and `policy_version`.
- Output: one append-only JSONL record in `trajectories.jsonl`.

Exactly eight stochastic trajectories are generated per task with
`temperature=1.0` and `top_p=0.98`. A normal training batch contains 16 tasks,
therefore 128 trajectories. The final batch of a round may contain fewer than
16 tasks.

### External reflection process

- Input: one 16-task trajectory batch, generated answers, pre-question video
  frames, memory transitions, and reward/correctness. It also receives the
  frozen round checkpoint path and the external model API credentials.
- Output: append-only `reflections.jsonl` records keyed by
  (`policy_version`, `trajectory_uid`). A valid record contains a global
  `episode_skill` and transition-aligned `step_skill` entries. Each transition
  also identifies its correction/preservation kind, memory error attribute,
  and one to three evidence/key-frame locations. A skipped reflection records an explicit
  reason.
- End-of-round output: atomically written `round_manifest.json` containing the
  policy version, frozen checkpoint, expected task count, eight trajectories
  per task, and `complete=true`.

The trainer never calls the reflection model in external mode. It publishes
trajectories and waits until all matching reflections for the current training
batch are present.

### Hindsight-conditioned teacher

- Input: the exact sampled local-model tokens. Global `episode_skill` is added
  to every memory-generation prompt in the trajectory; a transition's
  `step_skill` is added only to that transition's memory prompt. Final-answer
  tokens are excluded from OPD.
- Output: detached hindsight-conditioned same-token log probabilities.

### OPD weighting and PPO/GRPO

For each sampled token:

```text
delta_i^t = log pi_theta(R_i^t | x, C_i, R_i^<t)
            - log pi_theta(R_i^t | x, R_i^<t)
aligned_i^t = clip(sign(A_i) * delta_i^t,
                   log(1 - epsilon_R), log(1 + epsilon_R))
w_i^t = 1 + lambda_R * (exp(aligned_i^t) - 1)
A_i^t = A_i * w_i^t
```

The existing PPO/GRPO loss consumes `A_i^t`; OPD is not a second loss. With
`epsilon_R=0.2` and `lambda_R=0.5`, weights are in `[0.9, 1.1]` and cannot
reverse the sign of the RL signal.

## Round boundary

At round start, freeze `theta_old`. All rollout batches in that round use this
same policy version even while the actor consumes ready 16-task batches. The
rollout side is refreshed only after the external completion manifest is
validated and every declared task, including a smaller final batch, has been
consumed. The updated actor then becomes the next round's `theta_old`.
