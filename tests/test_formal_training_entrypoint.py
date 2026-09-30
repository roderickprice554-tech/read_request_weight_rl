import json
from pathlib import Path

import pytest
import requests

from conftest import trajectory_record, valid_skip_response
from pipeline.config import PipelineConfig
from pipeline.deepseek_v4_flash import (
    DeepSeekReflectionClient,
    complete_reflection_groups,
)
from pipeline.rl_update import build_training_command
from pipeline.reflection_request import _sample_frame_indices
from recurrent.skill_opd import observation_cutoff_seconds
from recurrent.reflection_sft import offline_record_to_trajectory


def _config(tmp_path, **overrides):
    values = {
        "trajectories": tmp_path / "offline.jsonl",
        "video_root": tmp_path / "videos",
        "vst_model": tmp_path / "model",
        "reflection_api_url": "https://vision.example/v1",
        "reflection_api_key": "secret",
        "reflection_model": "vision-model",
        "output_dir": tmp_path / "output",
        "train_files": tmp_path / "train.parquet",
        "val_files": tmp_path / "val.parquet",
    }
    values.update(overrides)
    return PipelineConfig(**values)


def test_formal_defaults_make_one_16_by_8_rl_batch(tmp_path):
    config = _config(tmp_path)

    assert config.train_batch_size == 16
    assert config.rollout_n == 8
    assert config.trajectories_per_batch == 128
    assert config.ppo_mini_batch_size <= config.train_batch_size


def test_training_command_covers_recurrent_opd_outputs_and_full_parameter_mode(tmp_path):
    config = _config(
        tmp_path,
        lora_rank=0,
        final_chunk_write_memory=True,
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        trajectory_path=tmp_path / "artifacts" / "trajectories.jsonl",
        reflection_path=tmp_path / "artifacts" / "reflections.jsonl",
    )

    command = build_training_command(config)

    expected = {
        f"data.train_files={config.train_files}",
        f"data.val_files={config.val_files}",
        "data.train_batch_size=16",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.model.lora_rank=0",
        "recurrent.enable=video_memory",
        f"recurrent.video_memory.config.video_root={config.video_root}",
        "recurrent.video_memory.config.final_chunk_write_memory=true",
        "skill_opd.enable=true",
        "trainer.save_freq=1",
        "skill_opd.reflection.source=external",
        f"skill_opd.reflection.trajectory_path={config.trajectory_path}",
        f"skill_opd.reflection.external_path={config.reflection_path}",
        f"trainer.default_local_dir={config.checkpoint_dir}",
        f"trainer.rollout_data_dir={config.log_dir}",
    }
    assert expected.issubset(command)


def test_complete_reflection_groups_waits_for_all_eight_rollouts(tmp_path):
    video = tmp_path / "a.mp4"
    video.touch()
    rows = []
    for index in range(8):
        record = trajectory_record(video)
        record["trajectory_uid"] = f"group-a:rollout-{index}"
        rows.append(offline_record_to_trajectory(record))
    partial = trajectory_record(video)
    partial["trajectory_uid"] = "group-b:rollout-0"
    rows.append(offline_record_to_trajectory(partial))

    groups = complete_reflection_groups(rows, rollout_n=8)

    assert [[item.trajectory_uid for item in group] for group in groups] == [
        [f"group-a:rollout-{index}" for index in range(8)]
    ]


class _ImageRejectingResponse:
    text = "image_url input is not supported"

    def raise_for_status(self):
        raise requests.HTTPError("400 Client Error", response=self)


class _ImageRejectingSession:
    def post(self, *args, **kwargs):
        return _ImageRejectingResponse()


def test_multimodal_api_rejection_names_required_capability(monkeypatch, tmp_path):
    video = tmp_path / "a.mp4"
    video.touch()
    trajectory = offline_record_to_trajectory(trajectory_record(video))
    monkeypatch.setattr(
        "pipeline.deepseek_v4_flash.build_multimodal_content",
        lambda value, count: [
            {"type": "text", "text": "prompt"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,eA=="}},
        ],
    )
    client = DeepSeekReflectionClient(
        "https://vision.example/v1",
        "secret",
        model="vision-model",
        session=_ImageRejectingSession(),
    )

    with pytest.raises(RuntimeError, match="must support image"):
        client.reflect_group([trajectory])

def test_observation_cutoff_ignores_text_only_final_boundary():
    boundaries = [
        {"seconds": [0.0, 120.0]},
        {"seconds": [120.0, 237.0]},
        None,
    ]

    assert observation_cutoff_seconds(boundaries) == 237.0
    trainer_source = Path("verl/trainer/ppo/ray_trainer.py").read_text(
        encoding="utf-8"
    )
    assert (
        'batch.non_tensor_batch["current_chunk_boundary"][row]["seconds"][1]'
        not in trainer_source
    )


def test_reflection_sampling_stops_at_observation_cutoff(tmp_path):
    video = tmp_path / "a.mp4"
    video.touch()
    record = trajectory_record(video)
    record["question_timestamp"] = 237.0

    trajectory = offline_record_to_trajectory(record)
    indices = _sample_frame_indices(
        frame_count=1000, max_frames=8, fps=2.0,
        end_seconds=trajectory.observation_cutoff_seconds,
    )

    assert trajectory.observation_cutoff_seconds == 237.0
    assert max(indices) == 474
    assert _sample_frame_indices(1000, 8, fps=2.0, end_seconds=None)[-1] == 999
