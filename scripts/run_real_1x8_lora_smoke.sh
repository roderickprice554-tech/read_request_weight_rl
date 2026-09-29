#!/usr/bin/env bash
set -euo pipefail

project=/home/bujunru/vlm-repro/read_request_weight_rl-share-video-teacher-mask
python=/home/bujunru/.conda/envs/vision-se/bin/python
source_data=/home/bujunru/vlm-repro/VST-SE/OVO_train_grok_skill/VST-Training-Data/vst_rl_data/train.parquet
video_root=/home/bujunru/vlm-repro/VST-SE/OVO_train_grok_skill/vst-stage1-videos
model=/home/bujunru/vlm-repro/models/VST-3B
output=/home/bujunru/vlm-repro/share-video-teacher-mask-smoke
sample_data=${output}/sample-203.parquet
metrics=${output}/metrics.json

mkdir -p "${output}"
rm -f "${output}/trajectories.jsonl" "${output}/reflections.jsonl" \
  "${output}/reflection_requests.jsonl" "${output}/reflection_errors.jsonl" \
  "${metrics}"

"${python}" -c "import pyarrow.parquet as pq; p='${source_data}'; pq.write_table(pq.read_table(p).slice(203, 1), '${sample_data}')"

cd "${project}"
export CUDA_VISIBLE_DEVICES=0,1
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${project}"

"${python}" scripts/run_external_reflection_worker.py \
  --trajectory-path "${output}/trajectories.jsonl" \
  --output-dir "${output}" \
  --expected-trajectories 8 \
  --api-url https://api.deepseek.com \
  --model deepseek-flash \
  >"${output}/reflection-worker.log" 2>&1 &
reflection_worker=$!
trap 'kill "${reflection_worker}" 2>/dev/null || true' EXIT

"${python}" -m verl.trainer.main_ppo \
  recurrent.enable=video_memory \
  recurrent.video_memory.path="${project}/recurrent/impls/video_memory.py" \
  recurrent.video_memory.config.video_clip_token_size=200 \
  recurrent.video_memory.config.max_video_frame=12 \
  recurrent.video_memory.config.max_video_clips=4 \
  recurrent.video_memory.config.video_root="${video_root}" \
  recurrent.video_memory.config.max_prompt_length=256 \
  recurrent.video_memory.config.max_memorization_length=128 \
  recurrent.video_memory.config.max_final_response_length=64 \
  recurrent.video_memory.config.prompt_type=type2 \
  skill_opd.enable=true \
  skill_opd.mode=global_episode \
  skill_opd.smoke_sample_index=203 \
  skill_opd.reflection.source=external \
  skill_opd.reflection.external_path="${output}/reflections.jsonl" \
  skill_opd.reflection.trajectory_path="${output}/trajectories.jsonl" \
  skill_opd.reflection.timeout_seconds=900 \
  skill_opd.smoke_metrics_path="${metrics}" \
  algorithm.adv_estimator=grpo \
  algorithm.grpo_use_adv=false \
  data.train_files="${sample_data}" \
  data.val_files="${sample_data}" \
  data.shuffle=false \
  data.filter_overlong_prompts=false \
  data.train_batch_size=1 \
  data.truncation=center \
  +data.context_key=context \
  data.max_prompt_length=600 \
  data.max_response_length=256 \
  reward_model.reward_manager=thread \
  actor_rollout_ref.model.path="${model}" \
  actor_rollout_ref.model.freeze_vision_tower=true \
  actor_rollout_ref.model.enable_embed_cache=true \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.lora_dropout=0 \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.vision_encoder_diagnostics=true \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.free_cache_engine=true \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  trainer.logger='[console]' \
  trainer.val_before_train=false \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_training_steps=1 \
  trainer.total_epochs=1 \
  trainer.default_local_dir="${output}/checkpoint" \
  2>&1 | tee "${output}/trainer.log"

wait "${reflection_worker}"
trap - EXIT
"${python}" scripts/audit_skill_opd_smoke.py --real-1x8 "${metrics}"
