import json
from dataclasses import asdict
from pathlib import Path

import requests

from recurrent.reflection import parse_and_validate_group_reflection
from recurrent.skill_opd import group_reflection_trajectories
from pipeline.reflection_request import build_multimodal_content


class DeepSeekReflectionClient:
    def __init__(self, base_url: str, api_key: str, model: str = "deepseek-v4-flash", *, session=None):
        if not api_key:
            raise ValueError("DeepSeek API key is required for reflection generation")
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.session = session or requests.Session()

    def reflect_group(self, trajectories, *, max_frames: int = 8):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": build_multimodal_content(trajectories, max_frames)}],
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
        parsed = parse_and_validate_group_reflection(
            raw, trajectories, trajectories[0].policy_version
        )
        return [{**asdict(item), "raw_response": json.dumps({
            key: value for key, value in next(
                row for row in json.loads(raw)["reflections"]
                if row["trajectory_uid"] == item.trajectory_uid
            ).items() if key != "trajectory_uid"
        })} for item in parsed]

    def reflect(self, trajectory, *, max_frames: int = 8):
        return self.reflect_group([trajectory], max_frames=max_frames)[0]


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
    for group in group_reflection_trajectories(trajectories):
        pending = [item for item in group if item.trajectory_uid not in accepted]
        if not pending:
            continue
        _append_jsonl(output_dir / "reflection_requests.jsonl", {
            "group_uid": group[0].group_uid,
            "trajectory_uids": [item.trajectory_uid for item in group],
            "policy_version": group[0].policy_version,
            "model": client.model,
        })
        error = None
        for _ in range(retries + 1):
            try:
                results = client.reflect_group(group)
                for result in results:
                    _append_jsonl(accepted_path, result)
                error = None
                break
            except Exception as exc:
                error = exc
        if error is not None:
            _append_jsonl(output_dir / "reflection_errors.jsonl", {
                "group_uid": group[0].group_uid,
                "trajectory_uids": [item.trajectory_uid for item in group],
                "policy_version": group[0].policy_version,
                "error": str(error),
            })
    return accepted_path
