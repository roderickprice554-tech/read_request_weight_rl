# VST question timestamp fallback design

## Goal

Allow the existing VST-RL training parquet to use the end of each sample video as
the query-arrival time when the row does not provide `question_timestamp`:

`t_q = T_sample_video = extra_info.duration`.

## Scope

Change only `VideoMemoryDataset` timestamp resolution. Do not rewrite the source
parquet and do not change rollout, reflection, OPD, or optimization semantics.

## Resolution order

For every dataset row, resolve the timestamp in this order:

1. top-level configured timestamp key (normally `question_timestamp`);
2. the same key inside `extra_info`;
3. `extra_info.duration` as the VST-RL fallback.

An explicitly supplied timestamp always wins. If neither a timestamp nor duration
is available, raise the existing missing-timestamp error. Continue validating that
the resolved value is positive and no later than the sample duration.

## Data flow

The resolved value remains the sole causal boundary used by video decoding,
cache keys, multimodal provenance, and trajectory metadata. Therefore fallback
samples stream the supplied sample video from time zero through its end, and the
question appears only on the final streaming turn.

## Verification

- A regression test must fail before implementation when only
  `extra_info.duration` is present.
- Tests must show explicit timestamps retain priority and missing timestamp plus
  missing duration still fails.
- A remote smoke must use one real VST-RL sample, VST-3B, batch size one, and one
  rollout. Evidence must include the trajectory count and whether the optimizer
  step completed; initialization-only or synthetic runs are reported separately.
