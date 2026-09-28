# Read → Request → Weight → RL

See `docs/EXTERNAL_OPD_PIPELINE_HANDOFF.md` for the causal video contract,
required external inputs, round protocol, and per-module inputs/outputs.

This directory is a portable copy of the VST/VERL trainer plus a thin pipeline for:

1. reading previously generated VST trajectories;
2. sending video frames and the trajectory to `deepseek-v4-flash`;
3. validating and loading the returned privileged reflection;
4. same-token teacher rescoring and detached bounded OPD weights;
5. the existing GRPO/PPO model update from the supplied SFT VST model.

No model, video, dataset, checkpoint, or API key is included.

## Inputs

- JSON or JSONL trajectories following `examples/minimal_trajectory.jsonl`;
- a directory containing the referenced videos;
- the SFT-stage VST model directory (this is the initial Actor and self-teacher);
- a DeepSeek-compatible API base URL;
- `DEEPSEEK_API_KEY` (recommended) or `--reflection-api-key`.

Install the same CUDA/PyTorch stack used by the target server, then install `requirements.txt`.

```bash
export DEEPSEEK_API_KEY='...'
python run_pipeline.py --config config.example.yaml --stage reflect
python run_pipeline.py --config config.example.yaml --stage train \
  --trainer-override data.train_files=/data/train.parquet \
  --trainer-override trainer.n_gpus_per_node=8
```

`--stage all` runs both stages. Accepted UIDs in `run-output/reflections.jsonl` are skipped on restart. The resolved config never contains the API key. Training refuses missing reflections or mismatched policy versions.

## OPD data flow

The existing trainer keeps task reward and group normalization unchanged. For `R` rollout trajectories and `T` generated tokens, student/teacher log probabilities, response mask, delta, weight, and modulated advantage are `[R,T]`; trajectory advantage starts as `[R]` and is broadcast over `T`.

```text
delta = (teacher_logprob - student_logprob).detach()
aligned = sign(trajectory_advantage) * delta
clipped = clamp(aligned, log(1-eps), log(1+eps))
weight = 1 + lambda * (exp(clipped) - 1)
token_advantage = trajectory_advantage * weight
```

Defaults are `eps=0.2` and `lambda=0.5`, so weights lie in `[0.9,1.1]`. Only existing Actor response/action tokens participate. Privileged reflection is training-only; it never enters Actor rollout or inference.

Use repeated `--trainer-override KEY=VALUE` arguments for the original VERL dataset, resource, rollout, and checkpoint settings. The repository intentionally preserves those settings instead of guessing server-specific values.
