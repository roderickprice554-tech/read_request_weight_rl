import argparse
import json
import subprocess
from pathlib import Path

from pipeline.config import PipelineConfig, load_yaml
from pipeline.deepseek_v4_flash import DeepSeekReflectionClient, generate_reflections
from pipeline.rl_update import build_training_command
from pipeline.trajectory import load_trajectories


def parse_args():
    parser = argparse.ArgumentParser(description="VST trajectory -> reflection -> OPD-weighted GRPO")
    parser.add_argument("--config")
    for name in ("trajectories", "video-root", "vst-model", "reflection-api-key", "reflection-api-url", "reflection-model", "output-dir"):
        parser.add_argument("--" + name)
    parser.add_argument("--stage", choices=("all", "reflect", "train"))
    parser.add_argument("--trainer-override", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    values = load_yaml(args.config) if args.config else {}
    for key, value in vars(args).items():
        if key not in {"config", "trainer_override"} and value is not None:
            values[key] = value
    config = PipelineConfig.from_mapping(values)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.json").write_text(json.dumps(config.public_dict(), indent=2), encoding="utf-8")
    if config.stage in {"all", "reflect"}:
        trajectories = load_trajectories(config.trajectories, config.video_root)
        client = DeepSeekReflectionClient(config.reflection_api_url, config.reflection_api_key, config.reflection_model)
        generate_reflections(trajectories, client, config.output_dir)
    if config.stage in {"all", "train"}:
        subprocess.run(build_training_command(config, args.trainer_override), cwd=Path(__file__).parent, check=True)


if __name__ == "__main__":
    main()
