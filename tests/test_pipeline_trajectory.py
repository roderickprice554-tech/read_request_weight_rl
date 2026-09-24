import json

from conftest import trajectory_record
from pipeline.config import PipelineConfig
from pipeline.trajectory import load_trajectory_records


def test_load_trajectories_resolves_video_under_root(tmp_path):
    video = tmp_path / "videos" / "a.mp4"
    video.parent.mkdir()
    video.touch()
    record = trajectory_record("a.mp4")
    source = tmp_path / "trajectories.jsonl"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rows = load_trajectory_records(source, video.parent)
    assert rows[0]["observed_video"] == str(video.resolve())


def test_resolved_config_redacts_api_key(tmp_path):
    config = PipelineConfig(tmp_path / "t", tmp_path / "v", tmp_path / "m", "https://example", "secret")
    assert "reflection_api_key" not in config.public_dict()
    assert "secret" not in config.public_dict().values()
