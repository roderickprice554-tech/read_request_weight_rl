import json
import os
import time
from pathlib import Path


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def start_round(root, *, policy_version, checkpoint, expected_task_count, trajectories_per_task=8):
    root = Path(root)
    completion = root / "round_manifest.json"
    if completion.exists():
        completion.unlink()
    payload = {
        "policy_version": int(policy_version),
        "checkpoint": str(checkpoint),
        "expected_task_count": int(expected_task_count),
        "trajectories_per_task": int(trajectories_per_task),
        "temperature": 1.0,
        "top_p": 0.98,
    }
    _atomic_json(root / "round_input.json", payload)
    return payload


def load_complete_manifest(root, expected):
    path = Path(root) / "round_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "policy_version", "checkpoint", "expected_task_count", "trajectories_per_task"
    ):
        if manifest.get(key) != expected[key]:
            raise ValueError(f"round manifest {key} mismatch")
    if manifest.get("complete") is not True:
        return None
    return manifest


def wait_for_complete_manifest(root, expected, *, poll_interval_seconds=1.0, timeout_seconds=None):
    deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
    while True:
        manifest = load_complete_manifest(root, expected)
        if manifest is not None:
            return manifest
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for external round completion")
        time.sleep(poll_interval_seconds)
