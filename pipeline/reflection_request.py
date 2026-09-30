import base64
import math
from pathlib import Path

from recurrent.reflection import build_group_reflection_prompt


def _sample_frame_indices(
    frame_count: int, max_frames: int, fps: float, end_seconds: float | None,
) -> list[int]:
    observed_count = frame_count
    if end_seconds is not None:
        if fps <= 0:
            raise ValueError("video FPS is required for timestamp-limited reflection")
        observed_count = min(
            frame_count, max(1, math.floor(float(end_seconds) * fps) + 1)
        )
    return sorted({
        round(i * max(observed_count - 1, 0) / max(max_frames - 1, 1))
        for i in range(min(max_frames, observed_count))
    })


def _sample_video_frames(
    path: str | Path, max_frames: int = 8, end_seconds: float | None = None,
) -> list[str]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    indices = _sample_frame_indices(count, max_frames, fps, end_seconds)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok:
            encoded, data = cv2.imencode(".jpg", frame)
            if encoded:
                frames.append("data:image/jpeg;base64," + base64.b64encode(data).decode("ascii"))
    capture.release()
    if not frames:
        raise ValueError(f"could not extract frames from {path}")
    return frames


def build_multimodal_content(trajectories, max_frames: int = 8) -> list[dict]:
    if not isinstance(trajectories, (list, tuple)):
        trajectories = [trajectories]
    content = [{"type": "text", "text": build_group_reflection_prompt(trajectories)}]
    cutoffs = {item.observation_cutoff_seconds for item in trajectories}
    if len(cutoffs) != 1:
        raise ValueError("reflection group has inconsistent observation cutoffs")
    observation_cutoff = cutoffs.pop()
    content.extend(
        {"type": "image_url", "image_url": {"url": frame}}
        for frame in _sample_video_frames(
            trajectories[0].observed_video, max_frames,
            end_seconds=observation_cutoff,
        )
    )
    return content
