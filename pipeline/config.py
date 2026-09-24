import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    trajectories: Path
    video_root: Path
    vst_model: Path
    reflection_api_url: str
    reflection_api_key: str = ""
    reflection_model: str = "deepseek-v4-flash"
    output_dir: Path = Path("run-output")
    stage: str = "all"

    def __post_init__(self):
        if self.stage not in {"all", "reflect", "train"}:
            raise ValueError("stage must be all, reflect, or train")
        if self.reflection_model != "deepseek-v4-flash":
            raise ValueError("this pipeline requires reflection_model=deepseek-v4-flash")

    def public_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
            if key != "reflection_api_key"
        }

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PipelineConfig":
        values = dict(values)
        values["reflection_api_key"] = values.get("reflection_api_key") or os.getenv(
            "DEEPSEEK_API_KEY", ""
        )
        for key in ("trajectories", "video_root", "vst_model", "output_dir"):
            values[key] = Path(values[key]).expanduser()
        return cls(**values)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
