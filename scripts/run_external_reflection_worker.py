#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from pipeline.deepseek_v4_flash import (
    DeepSeekReflectionClient,
    complete_reflection_groups,
    generate_reflections,
)
from recurrent.reflection_sft import offline_record_to_trajectory


def _read_trajectories(path):
    if not path.exists():
        return []
    return [
        offline_record_to_trajectory(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _accepted_uids(path):
    if not path.exists():
        return set()
    return {
        json.loads(line)["trajectory_uid"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rollout-n", type=int)
    parser.add_argument("--reflection-path", type=Path)
    parser.add_argument("--expected-trajectories", type=int)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-env", default="REFLECTION_API_KEY")
    parser.add_argument("--max-frames", default=8, type=int)
    parser.add_argument("--poll-interval", default=1.0, type=float)
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()
    rollout_n = args.rollout_n or args.expected_trajectories
    if not rollout_n:
        parser.error("--rollout-n is required")
    client = DeepSeekReflectionClient(
        args.api_url,
        os.environ.get(args.api_key_env, ""),
        model=args.model,
        max_frames=args.max_frames,
    )
    reflection_path = args.reflection_path or args.output_dir / "reflections.jsonl"
    while True:
        trajectories = _read_trajectories(args.trajectory_path)
        groups = complete_reflection_groups(trajectories, rollout_n)
        if groups:
            generate_reflections(
                [item for group in groups for item in group],
                client,
                args.output_dir,
                accepted_path=reflection_path,
            )
        if (
            not args.continuous
            and args.expected_trajectories
            and len(_accepted_uids(reflection_path)) >= args.expected_trajectories
        ):
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
