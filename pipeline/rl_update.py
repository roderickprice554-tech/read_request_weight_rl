import sys


def build_training_command(config, overrides=()):
    invalid = [value for value in overrides if "=" not in value]
    if invalid:
        raise ValueError(f"trainer overrides require KEY=VALUE: {invalid}")
    return [
        sys.executable, "-m", "verl.trainer.main_ppo",
        f"actor_rollout_ref.model.path={config.vst_model}",
        "skill_opd.enable=true",
        "skill_opd.reflection.source=external",
        f"skill_opd.reflection.external_path={config.output_dir / 'reflections.jsonl'}",
        f"skill_opd.reflection.trajectory_path={config.output_dir / 'trajectories.jsonl'}",
        "skill_opd.opd_weight_eps=0.2",
        "skill_opd.opd_weight_lambda=0.5",
        "data.train_batch_size=16",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=0.98",
        *overrides,
    ]
