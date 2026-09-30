import sys


def build_training_command(config, overrides=()):
    invalid = [value for value in overrides if "=" not in value]
    if invalid:
        raise ValueError(f"trainer overrides require KEY=VALUE: {invalid}")
    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"actor_rollout_ref.model.path={config.vst_model}",
        f"actor_rollout_ref.model.lora_rank={config.lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={config.lora_alpha}",
        f"actor_rollout_ref.model.lora_dropout={config.lora_dropout}",
        f"actor_rollout_ref.rollout.n={config.rollout_n}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={config.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_micro_batch_size=null",
        (
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="
            f"{config.ppo_micro_batch_size_per_gpu}"
        ),
        f"data.train_batch_size={config.train_batch_size}",
        "algorithm.adv_estimator=grpo",
        "algorithm.grpo_use_adv=true",
        "recurrent.enable=video_memory",
        f"recurrent.video_memory.config.video_root={config.video_root}",
        (
            "recurrent.video_memory.config.final_chunk_write_memory="
            f"{str(config.final_chunk_write_memory).lower()}"
        ),
        "skill_opd.enable=true",
        "skill_opd.reflection.source=external",
        f"skill_opd.reflection.trajectory_path={config.trajectory_path}",
        f"skill_opd.reflection.external_path={config.reflection_path}",
        (
            "skill_opd.reflection.poll_interval_seconds="
            f"{config.reflection_poll_interval_seconds}"
        ),
        f"skill_opd.reflection.timeout_seconds={config.reflection_timeout_seconds}",
        "skill_opd.opd_weight_eps=0.2",
        "skill_opd.opd_weight_lambda=0.5",
        f"trainer.n_gpus_per_node={config.n_gpus_per_node}",
        f"trainer.default_local_dir={config.checkpoint_dir}",
        f"trainer.save_freq={config.save_freq}",
        f"trainer.rollout_data_dir={config.log_dir}",
    ]
    if config.train_files is not None:
        command.append(f"data.train_files={config.train_files}")
    if config.val_files is not None:
        command.append(f"data.val_files={config.val_files}")
    return [*command, *overrides]
