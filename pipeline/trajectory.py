import json
from pathlib import Path
from typing import Any

from recurrent.reflection_sft import offline_record_to_trajectory


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    value = json.loads(text)
    if not isinstance(value, list):
        raise ValueError("trajectory JSON must contain a list")
    return value


def load_trajectory_records(path: str | Path, video_root: str | Path) -> list[dict[str, Any]]:
    path, video_root = Path(path), Path(video_root).resolve()
    records, seen = _read_records(path), set()
    for record in records:
        uid = str(record.get("trajectory_uid", ""))
        if not uid or uid in seen:
            raise ValueError(f"duplicate or empty trajectory_uid: {uid!r}")
        seen.add(uid)
        video = Path(record["observed_video"])
        video = video.resolve() if video.is_absolute() else (video_root / video).resolve()
        if not video.is_relative_to(video_root):
            raise ValueError(f"video escapes video_root: {video}")
        if not video.is_file():
            raise FileNotFoundError(f"video not found: {video}")
        record["observed_video"] = str(video)
        offline_record_to_trajectory(record)
    return records


def load_trajectories(path: str | Path, video_root: str | Path):
    return [offline_record_to_trajectory(row) for row in load_trajectory_records(path, video_root)]
