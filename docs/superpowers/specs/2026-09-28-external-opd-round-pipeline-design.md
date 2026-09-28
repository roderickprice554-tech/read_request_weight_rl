# External OPD Round Pipeline Design

## Goal

Support asynchronous OPD training rounds where rollout is produced by the
training process, hindsight reflections are produced by an external process,
and optimization can start before reflection generation finishes for the
entire round.

Each round freezes one policy checkpoint, `theta_old`. Every task is sampled
eight times from that same checkpoint with stochastic decoding. Training
consumes groups of 16 tasks (128 trajectories) as soon as all reflections for
those groups are available. A final smaller task batch is allowed after the
external process marks the round complete.

## Invariants

- A round has one immutable `policy_version` and one immutable checkpoint.
- Rollout uses `temperature=1.0` and `top_p=0.98`.
- Every task has exactly eight distinct trajectories.
- All trajectories and hindsight reflections in a round come from its frozen
  checkpoint.
- Actor optimization must not update the rollout worker during a round.
- A normal training batch contains 16 complete task groups, or 128
  trajectories.
- The final batch may contain fewer than 16 task groups only after the round is
  marked complete.
- A trajectory is never trained without a matching, valid reflection from the
  same `policy_version`.
- Each trajectory is consumed at most once, including after restart.

## Round Files

Each round uses a separate directory so readers never mix policy versions:

```text
<external_path>/round-<policy_version>/
  round_input.json
  trajectories.jsonl
  reflections.jsonl
  round_manifest.json
  trainer_state.json
```

The training process creates `round_input.json` before rollout. It contains the
policy version, immutable checkpoint path, expected eight trajectories per
task, rollout sampling settings, and stable round identifier.

The training process appends one JSON object per completed trajectory to
`trajectories.jsonl`. Existing trajectory identifiers remain stable and use the
current `group_uid` plus rollout index.

The external process tails `trajectories.jsonl`, loads the checkpoint named by
`round_input.json`, and appends one validated hindsight result per trajectory
to `reflections.jsonl`.

After it has handled every task in the round, the external process atomically
writes `round_manifest.json`. The manifest contains:

```json
{
  "policy_version": 12,
  "checkpoint": "/checkpoints/policy-12",
  "expected_task_count": 320,
  "trajectories_per_task": 8,
  "complete": true
}
```

The manifest is the only end-of-round signal. The trainer does not infer
completion from a temporarily idle JSONL file.

`trainer_state.json` records the trajectory identifiers already consumed. It
is replaced atomically after each successful training batch so a restart does
not repeat an update.

## Training Control Flow

At the start of a round, the trainer saves `theta_old`, assigns the next
`policy_version`, writes `round_input.json`, and freezes rollout-worker weight
synchronization.

The trainer generates eight stochastic trajectories per task using the frozen
rollout worker. It retains the train-ready tensors in memory and publishes the
reflection input record to `trajectories.jsonl`. The actor may change during
optimization, but the rollout worker remains on `theta_old` until the round is
finished.

The trainer refreshes `reflections.jsonl` while rollout and external hindsight
generation continue. It groups records by `group_uid`, rejects mismatched
policy versions and duplicate trajectory identifiers, and makes a task ready
only when all eight trajectories have valid reflections.

Whenever 16 task groups are ready, the trainer constructs a 128-trajectory
training batch. The batch may be divided into the existing PPO optimization
minibatches. No rollout is regenerated between those minibatches. Ready groups
may be consumed in deterministic rollout order to make restart behavior
reproducible.

When `round_manifest.json` appears, the trainer validates its policy version,
checkpoint, expected task count, and trajectory count. After all declared
tasks have matching reflections, it trains the remaining complete task groups
as one smaller final batch. Missing or malformed records are errors rather
than silently dropped samples.

After every declared task is consumed, the trainer completes the round,
publishes the updated actor as the next immutable checkpoint, and only then
synchronizes the rollout worker for the next round.

## OPD Advantage

GRPO computes the trajectory-level group advantage `A_i` across the eight
samples of one task. OPD computes a detached bounded token weight `w_i^t` from
the existing student and hindsight-conditioned teacher log probabilities.
The effective token advantage is:

```text
A_i^t = A_i * w_i^t
```

The existing PPO/GRPO loss consumes `A_i^t`. OPD is not added as a separate
loss term. Existing response masks continue to exclude non-action tokens and
final-answer exclusions remain unchanged.

## Failure and Restart Behavior

- Partial trailing JSONL lines are treated as not yet published and retried.
- Duplicate identical records are ignored; conflicting duplicates fail the
  round.
- Records for another `policy_version` are rejected.
- The trainer waits when fewer than 16 task groups are ready and no completion
  manifest exists.
- A completion manifest permits a smaller final batch but never permits an
  incomplete eight-trajectory task group.
- On restart, the trainer reloads `trainer_state.json`, skips consumed
  trajectories, and resumes the same frozen round.

## Configuration

External mode sets the required behavior directly:

```text
data.train_batch_size=16
actor_rollout_ref.rollout.n=8
actor_rollout_ref.rollout.temperature=1.0
actor_rollout_ref.rollout.top_p=0.98
skill_opd.reflection.source=external
```

Only the external round directory and polling interval need new configuration.
Batch size, sample count, and sampling parameters remain ordinary trainer
configuration values and are validated rather than duplicated in a second
configuration layer.

## Tests

Tests will cover:

- incremental JSONL refresh without reconstructing the store;
- grouping exactly eight trajectories per task;
- rejecting mixed policy versions, missing records, and conflicting
  duplicates;
- emitting a 16-task batch before the round is complete;
- waiting instead of consuming a short batch before completion;
- consuming a smaller final batch after a valid completion manifest;
- restart behavior that does not consume a trajectory twice;
- command/config defaults for 16 tasks, eight trajectories, temperature 1.0,
  and top-p 0.98;
- token advantage remains `A_i * w_i^t` in the existing OPD loss path.

GPU integration verification will confirm that rollout-worker weights remain
frozen for the full round and synchronize only at the next round boundary.
