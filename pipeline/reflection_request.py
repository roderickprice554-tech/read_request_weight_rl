import base64
from pathlib import Path

from recurrent.reflection import build_group_reflection_prompt


def _sample_video_frames(path: str | Path, max_frames: int = 8) -> list[str]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = {round(i * max(count - 1, 0) / max(max_frames - 1, 1)) for i in range(min(max_frames, count))}
    frames = []
    for index in sorted(indices):
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
    content.extend(
        {"type": "image_url", "image_url": {"url": frame}}
        for frame in _sample_video_frames(trajectories[0].observed_video, max_frames)
    )
    return content
