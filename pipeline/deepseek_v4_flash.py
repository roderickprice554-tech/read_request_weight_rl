import json
from dataclasses import asdict
from pathlib import Path

import requests

from recurrent.reflection import parse_and_validate_reflection
from pipeline.reflection_request import build_multimodal_content


class DeepSeekReflectionClient:
    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-v4-flash", *, session=None):
        if not api_key:
            raise ValueError("DeepSeek API key is required for reflection generation")
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.session = session or requests.Session()

    def reflect(self, trajectory, *, max_frames: int = 8):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_multimodal_content(trajectory, max_frames)}],
            "temperature": 0,
            "max_tokens": 2048,
            "thinking": {"type": "disabled"},
            "response_format": {"type": "json_object"},
        }
        response = self.session.post(
            self.base_url + "/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=body,
            timeout=300,
        )
        response.raise_for_status()
        raw = response.json()["choices"][0]["message"]["content"]
        parsed = parse_and_validate_reflection(raw, trajectory, trajectory.policy_version)
        if not parsed.reflection_valid:
            raise ValueError(parsed.rejection_reason or "invalid reflection")
        return {**asdict(parsed), "raw_response": raw}


def _append_jsonl(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def generate_reflections(trajectories, client, output_dir: str | Path, *, retries: int = 1):
    output_dir = Path(output_dir)
    accepted_path = output_dir / "reflections.jsonl"
    accepted = set()
    if accepted_path.exists():
        accepted = {json.loads(line)["trajectory_uid"] for line in accepted_path.read_text(encoding="utf-8").splitlines() if line.strip()}
    for trajectory in trajectories:
        if trajectory.trajectory_uid in accepted:
            continue
        _append_jsonl(output_dir / "reflection_requests.jsonl", {
            "trajectory_uid": trajectory.trajectory_uid,
            "policy_version": trajectory.policy_version,
            "model": client.model,
        })
        error = None
        for _ in range(retries + 1):
            try:
                result = client.reflect(trajectory)
                _append_jsonl(accepted_path, result)
                error = None
                break
            except Exception as exc:
                error = exc
        if error is not None:
            _append_jsonl(output_dir / "reflection_errors.jsonl", {
                "trajectory_uid": trajectory.trajectory_uid,
                "policy_version": trajectory.policy_version,
                "error": str(error),
            })
    return accepted_path
