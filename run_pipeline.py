import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from pipeline.config import PipelineConfig, load_yaml
from pipeline.deepseek_v4_flash import DeepSeekReflectionClient, generate_reflections
from pipeline.rl_update import build_training_command
from pipeline.trajectory import load_trajectories


def parse_args():
    parser = argparse.ArgumentParser(
        description="recurrent video rollout -> multimodal reflection -> OPD-weighted GRPO"
    )
    parser.add_argument("--config")
    for name in (
        "trajectories", "video-root", "vst-model", "reflection-api-key",
        "reflection-api-url", "reflection-model", "output-dir",
        "train-files", "val-files", "checkpoint-dir", "log-dir",
        "trajectory-path", "reflection-path",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--stage", choices=("all", "reflect", "train"))
    parser.add_argument("--trainer-override", action="append", default=[])
    return parser.parse_args()


def build_reflection_worker_command(config):
    return [
        sys.executable,
        "scripts/run_external_reflection_worker.py",
        "--trajectory-path", str(config.trajectory_path),
        "--output-dir", str(config.output_dir),
        "--rollout-n", str(config.rollout_n),
        "--reflection-path", str(config.reflection_path),
        "--api-url", config.reflection_api_url,
        "--model", config.reflection_model,
        "--api-key-env", config.reflection_api_key_env,
        "--max-frames", str(config.reflection_max_frames),
        "--poll-interval", str(config.reflection_poll_interval_seconds),
        "--continuous",
    ]


def run_training(config, trainer_overrides=()):
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    config.reflection_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    if config.reflection_api_key:
        environment[config.reflection_api_key_env] = config.reflection_api_key
    worker_log_path = config.log_dir / "reflection-worker.log"
    trainer_log_path = config.log_dir / "trainer.log"
    with worker_log_path.open("a", encoding="utf-8") as worker_log, trainer_log_path.open(
        "a", encoding="utf-8"
    ) as trainer_log:
        worker = subprocess.Popen(
            build_reflection_worker_command(config),
            cwd=Path(__file__).parent,
            env=environment,
            stdout=worker_log,
            stderr=subprocess.STDOUT,
        )
        trainer = subprocess.Popen(
            build_training_command(config, trainer_overrides),
            cwd=Path(__file__).parent,
            env=environment,
            stdout=trainer_log,
            stderr=subprocess.STDOUT,
        )
        try:
            while trainer.poll() is None:
                if worker.poll() is not None:
                    trainer.terminate()
                    trainer.wait()
                    raise RuntimeError(
                        f"reflection worker failed; see {worker_log_path}"
                    )
                time.sleep(1.0)
            if trainer.returncode:
                raise subprocess.CalledProcessError(
                    trainer.returncode, build_training_command(config, trainer_overrides)
                )
        finally:
            if worker.poll() is None:
                worker.terminate()
            worker.wait()


def main():
    args = parse_args()
    values = load_yaml(args.config) if args.config else {}
    for key, value in vars(args).items():
        if key not in {"config", "trainer_override"} and value is not None:
            values[key] = value
    config = PipelineConfig.from_mapping(values)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "resolved_config.json").write_text(
        json.dumps(config.public_dict(), indent=2), encoding="utf-8"
    )
    if config.stage == "reflect":
        trajectories = load_trajectories(config.trajectories, config.video_root)
        client = DeepSeekReflectionClient(
            config.reflection_api_url,
            config.reflection_api_key,
            config.reflection_model,
            max_frames=config.reflection_max_frames,
        )
        generate_reflections(
            trajectories, client, config.output_dir, accepted_path=config.reflection_path
        )
    else:
        run_training(config, args.trainer_override)


if __name__ == "__main__":
    main()
