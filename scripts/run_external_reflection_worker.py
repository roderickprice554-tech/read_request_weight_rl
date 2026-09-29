#!/usr/bin/env python3
import argparse
import os
import time
from pathlib import Path

from pipeline.deepseek_v4_flash import DeepSeekReflectionClient, generate_reflections
from recurrent.reflection_sft import offline_record_to_trajectory


def _read_trajectories(path):
    import json

    if not path.exists():
        return []
    return [
        offline_record_to_trajectory(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-trajectories", required=True, type=int)
    parser.add_argument("--api-url", default="https://api.deepseek.com")
    parser.add_argument("--model", default="deepseek-flash")
    args = parser.parse_args()
    client = DeepSeekReflectionClient(
        args.api_url,
        os.environ.get("DEEPSEEK_API_KEY", ""),
        model=args.model,
    )
    while True:
        trajectories = _read_trajectories(args.trajectory_path)
        if len(trajectories) == args.expected_trajectories:
            generate_reflections(trajectories, client, args.output_dir)
            return
        if len(trajectories) > args.expected_trajectories:
            raise ValueError("received more trajectories than expected")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
