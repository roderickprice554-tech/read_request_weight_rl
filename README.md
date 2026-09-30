# Read -> Request -> Weight -> RL

This repository provides one production entrypoint for:

```text
local model/data
  -> recurrent video-memory rollout (16 prompts x 8 samples)
  -> external multimodal group reflection
  -> teacher/student same-token scoring
  -> bounded OPD token weights
  -> PPO/GRPO update
  -> checkpoint
```

Models, datasets, videos, checkpoints, and API keys are not included.

## Requirements

Use the CUDA, PyTorch, vLLM, and Transformers versions required by the target
training server, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

The training data is the parquet format consumed by VERL. Set `video_root` to
the local directory used to resolve video paths in those rows, and `vst_model`
to the local SFT VST model directory.

## Configure training

Copy `config.example.yaml` and set at least:

- `train_files`, `val_files`, `video_root`, and `vst_model`;
- `output_dir` and the checkpoint/log/trajectory/reflection paths;
- `reflection_api_url` and `reflection_model`;
- the environment variable named by `reflection_api_key_env`.

The endpoint must implement an OpenAI-compatible `/chat/completions` API and
the selected model must accept `image_url` message content. Each logical
sample sends one shared set of sampled video frames plus all eight textual
trajectories. If the endpoint rejects image input, training stops and the
provider response is written to `reflection-worker.log`; there is no
text-only fallback.

```bash
export REFLECTION_API_KEY='replace-me'
python run_pipeline.py --config /path/to/train.yaml
```

The defaults are `train_batch_size=16` and `rollout_n=8`: all 128
trajectories belong to one logical RL batch. `ppo_mini_batch_size` and
`ppo_micro_batch_size_per_gpu` only split optimization work for memory use;
they do not change the rollout group or reward normalization batch.

Extra Hydra settings can be appended without editing the entrypoint:

```bash
python run_pipeline.py --config /path/to/train.yaml \
  --trainer-override actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  --trainer-override trainer.total_epochs=3
```

## LoRA or full-parameter training

A positive `lora_rank` enables LoRA and uses `lora_alpha` and
`lora_dropout`. Set the rank to zero to skip PEFT/LoRA construction and train
the base model parameters:

```yaml
lora_rank: 0
```

Full-parameter training needs substantially more GPU memory. FSDP/offload
settings remain normal VERL overrides and are not guessed by this wrapper.

## Final-chunk modes

Both recurrent policies are supported:

```yaml
# Last chunk + previous memory + question produces the multimodal final row.
final_chunk_write_memory: false
```

```yaml
# Last chunk first writes memory; completed memory + question produces a
# text-only final row.
final_chunk_write_memory: true
```

In both modes, `question_timestamp` limits decoding to the observation
cutoff; a missing timestamp decodes to video EOF. Chunk boundaries are
continuous and the cache identity includes the timestamp.

## Reflection-only mode

For existing JSON/JSONL trajectories, set `trajectories` and run:

```bash
python run_pipeline.py --config /path/to/train.yaml --stage reflect
```

Normal `stage: all` and `stage: train` both run the reflection worker
alongside the trainer so each newly published complete rollout group is
processed before OPD scoring.

## Artifacts

Paths are independently configurable. The example produces:

- `resolved_config.json` - resolved non-secret configuration;
- `trajectories.jsonl` - recurrent trajectories published by the trainer;
- `reflections.jsonl` - validated external reflections;
- `reflection_requests.jsonl` and `reflection_errors.jsonl` - audit trail;
- `logs/trainer.log` and `logs/reflection-worker.log`;
- `checkpoints/` - model/optimizer/trainer checkpoints; positive `save_freq`
  also saves on the final training step.

Accepted trajectory UIDs are skipped when reflection generation restarts.
API keys are passed through the environment and omitted from
`resolved_config.json`.

## OPD weighting

For rollout trajectories `R` and generated tokens `T`, student/teacher log
probabilities, masks, detached weights, and weighted advantages are
`[R, T]`:

```text
delta = (teacher_logprob - student_logprob).detach()
aligned = sign(trajectory_advantage) * delta
clipped = clamp(aligned, log(1-eps), log(1+eps))
weight = 1 + lambda * (exp(clipped) - 1)
token_advantage = trajectory_advantage * weight
```

With the defaults `eps=0.2` and `lambda=0.5`, weights are bounded to
`[0.9, 1.1]`. Reflection is training-only and is never inserted into actor
rollout or inference context.
