from pathlib import Path

from pipeline.config import PipelineConfig
from pipeline.rl_update import build_training_command


def test_train_command_uses_local_runtime_and_external_reflections(tmp_path):
    config = PipelineConfig(tmp_path / "t", tmp_path / "v", tmp_path / "model", "https://example", "key", output_dir=tmp_path / "out")
    command = build_training_command(config)
    assert command[1:3] == ["-m", "verl.trainer.main_ppo"]
    assert f"actor_rollout_ref.model.path={config.vst_model}" in command
    assert "skill_opd.reflection.source=external" in command
    assert "skill_opd.opd_weight_eps=0.2" in command
    assert "skill_opd.opd_weight_lambda=0.5" in command
    assert "data.train_batch_size=16" in command
    assert "actor_rollout_ref.rollout.n=8" in command
    assert "actor_rollout_ref.rollout.temperature=1.0" in command
    assert "actor_rollout_ref.rollout.top_p=0.98" in command
