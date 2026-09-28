from recurrent.reflection_sft import offline_record_to_trajectory
from pipeline.deepseek_v4_flash import DeepSeekReflectionClient
from pipeline import reflection_request
from conftest import trajectory_record, valid_skip_response
from types import SimpleNamespace


class Response:
    def raise_for_status(self): pass
    def json(self):
        return {"choices": [{"message": {"content": valid_skip_response()}}]}


class Session:
    def post(self, url, **kwargs):
        self.url, self.kwargs = url, kwargs
        return Response()


def test_client_uses_model_and_key_only_in_header(monkeypatch, tmp_path):
    video = tmp_path / "a.mp4"; video.touch()
    trajectory = offline_record_to_trajectory(trajectory_record(video))
    monkeypatch.setattr("pipeline.deepseek_v4_flash.build_multimodal_content", lambda value, count: [{"type": "text", "text": "prompt"}])
    session = Session()
    client = DeepSeekReflectionClient("https://api.example", "secret", session=session)
    result = client.reflect(trajectory)
    assert session.kwargs["json"]["model"] == "deepseek-v4-flash"
    assert session.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert "secret" not in str(result)


def test_external_video_frames_stop_at_question_timestamp(monkeypatch):
    calls = []
    monkeypatch.setattr(
        reflection_request,
        "build_reflection_prompt",
        lambda trajectory: "prompt",
    )
    monkeypatch.setattr(
        reflection_request,
        "_sample_video_frames",
        lambda path, count, end_seconds=None: calls.append(
            (path, count, end_seconds)
        ) or ["frame"],
    )
    trajectory = SimpleNamespace(
        observed_video={
            "source_paths": ["video.mp4"],
            "question_timestamp": 12.5,
        }
    )

    reflection_request.build_multimodal_content(trajectory, max_frames=4)

    assert calls == [("video.mp4", 4, 12.5)]
