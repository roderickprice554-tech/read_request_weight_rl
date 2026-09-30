import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    trajectories: Path
    video_root: Path
    vst_model: Path
    reflection_api_url: str
    reflection_api_key: str = ""
    reflection_model: str = "multimodal-model"
    output_dir: Path = Path("run-output")
    stage: str = "all"
    train_files: Optional[Path] = None
    val_files: Optional[Path] = None
    train_batch_size: int = 16
    rollout_n: int = 8
    ppo_mini_batch_size: int = 16
    ppo_micro_batch_size_per_gpu: int = 1
    final_chunk_write_memory: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    n_gpus_per_node: int = 8
    checkpoint_dir: Optional[Path] = None
    save_freq: int = 1
    log_dir: Optional[Path] = None
    trajectory_path: Optional[Path] = None
    reflection_path: Optional[Path] = None
    reflection_api_key_env: str = "REFLECTION_API_KEY"
    reflection_max_frames: int = 8
    reflection_poll_interval_seconds: float = 1.0
    reflection_timeout_seconds: float = 900.0


    def __post_init__(self):
        if self.stage not in {"all", "reflect", "train"}:
            raise ValueError("stage must be all, reflect, or train")
        if self.train_batch_size <= 0 or self.rollout_n <= 0:
            raise ValueError("train_batch_size and rollout_n must be positive")
        if not 0 < self.ppo_mini_batch_size <= self.train_batch_size:
            raise ValueError("ppo_mini_batch_size must be in [1, train_batch_size]")
        if self.ppo_micro_batch_size_per_gpu <= 0:
            raise ValueError("ppo_micro_batch_size_per_gpu must be positive")
        if self.lora_rank < 0:
            raise ValueError("lora_rank must be non-negative; 0 disables LoRA")
        if self.reflection_max_frames <= 0:
            raise ValueError("reflection_max_frames must be positive")
        if self.save_freq <= 0:
            raise ValueError("save_freq must be positive")
        defaults = {
            "checkpoint_dir": self.output_dir / "checkpoints",
            "log_dir": self.output_dir / "logs",
            "trajectory_path": self.output_dir / "trajectories.jsonl",
            "reflection_path": self.output_dir / "reflections.jsonl",
        }
        for key, value in defaults.items():
            if getattr(self, key) is None:
                object.__setattr__(self, key, value)

    @property
    def trajectories_per_batch(self) -> int:
        return self.train_batch_size * self.rollout_n

    def public_dict(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
            if key != "reflection_api_key"
        }

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "PipelineConfig":
        values = dict(values)
        key_env = values.get("reflection_api_key_env", "REFLECTION_API_KEY")
        values["reflection_api_key"] = values.get("reflection_api_key") or os.getenv(
            key_env, ""
        ) or os.getenv("DEEPSEEK_API_KEY", "")
        for key in (
            "trajectories", "video_root", "vst_model", "output_dir",
            "train_files", "val_files", "checkpoint_dir", "log_dir",
            "trajectory_path", "reflection_path",
        ):
            if values.get(key) is not None:
                values[key] = Path(values[key]).expanduser()
        return cls(**values)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
