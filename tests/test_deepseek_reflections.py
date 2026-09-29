from recurrent.reflection_sft import offline_record_to_trajectory
from pipeline.deepseek_v4_flash import DeepSeekReflectionClient
from conftest import trajectory_record, valid_skip_response


class Response:
    def raise_for_status(self): pass
    def json(self):
        return {"choices": [{"message": {"content": self.content}}]}

    content = ""


class Session:
    def post(self, url, **kwargs):
        self.url, self.kwargs = url, kwargs
        return Response()


def test_client_uses_model_and_key_only_in_header(monkeypatch, tmp_path):
    video = tmp_path / "a.mp4"; video.touch()
    trajectory = offline_record_to_trajectory(trajectory_record(video))
    Response.content = __import__("json").dumps({
        "group_uid": trajectory.group_uid,
        "policy_version": trajectory.policy_version,
        "reflections": [{
            "trajectory_uid": trajectory.trajectory_uid,
            **__import__("json").loads(valid_skip_response()),
        }],
    })
    monkeypatch.setattr("pipeline.deepseek_v4_flash.build_multimodal_content", lambda value, count: [{"type": "text", "text": "prompt"}])
    session = Session()
    client = DeepSeekReflectionClient("https://api.example", "secret", session=session)
    result = client.reflect(trajectory)
    assert session.kwargs["json"]["model"] == "deepseek-v4-flash"
    assert session.kwargs["headers"]["Authorization"] == "Bearer secret"
    assert "secret" not in str(result)


def test_client_sends_one_request_for_a_group(monkeypatch, tmp_path):
    video = tmp_path / "a.mp4"; video.touch()
    first = offline_record_to_trajectory(trajectory_record(video))
    second_record = trajectory_record(video)
    second_record["trajectory_uid"] = first.group_uid + ":rollout-1"
    second = offline_record_to_trajectory(second_record)
    Response.content = __import__("json").dumps({
        "group_uid": first.group_uid,
        "policy_version": first.policy_version,
        "reflections": [
            {"trajectory_uid": item.trajectory_uid, **__import__("json").loads(valid_skip_response())}
            for item in (first, second)
        ],
    })
    monkeypatch.setattr(
        "pipeline.deepseek_v4_flash.build_multimodal_content",
        lambda values, count: [{"type": "text", "text": str(len(values))}],
    )
    session = Session()

    result = DeepSeekReflectionClient("https://api.example", "secret", session=session).reflect_group([first, second])

    assert len(result) == 2
    assert session.kwargs["json"]["messages"][0]["content"][0]["text"] == "2"
