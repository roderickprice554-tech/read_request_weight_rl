import argparse
from collections import Counter
import json
import os
import time
from pathlib import Path

from pipeline.deepseek_v4_flash import DeepSeekReflectionClient, generate_reflections
from recurrent.opd_round import _atomic_json
from recurrent.reflection_sft import offline_record_to_trajectory


def _read_complete_jsonl(path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index == len(text.splitlines()) - 1 and not text.endswith("\n"):
                break
            raise
    return rows


def run_round(root, client, *, poll_interval_seconds=1.0):
    root = Path(root)
    round_input = json.loads((root / "round_input.json").read_text(encoding="utf-8"))
    version = int(round_input["policy_version"])
    expected_tasks = int(round_input["expected_task_count"])
    trajectories_per_task = int(round_input["trajectories_per_task"])
    expected_trajectories = expected_tasks * trajectories_per_task

    while True:
        records = [
            row for row in _read_complete_jsonl(root / "trajectories.jsonl")
            if int(row["policy_version"]) == version
        ]
        trajectories = [offline_record_to_trajectory(row) for row in records]
        if trajectories:
            generate_reflections(trajectories, client, root)
        accepted = [
            row for row in _read_complete_jsonl(root / "reflections.jsonl")
            if int(row["policy_version"]) == version
        ]
        if len(records) == expected_trajectories and len(accepted) == expected_trajectories:
            group_counts = Counter(
                row["trajectory_uid"].split(":rollout-", 1)[0] for row in records
            )
            if len(group_counts) != expected_tasks or set(group_counts.values()) != {
                trajectories_per_task
            }:
                raise ValueError("round must contain exactly eight trajectories per task")
            manifest = {**round_input, "complete": True}
            _atomic_json(root / "round_manifest.json", manifest)
            return manifest
        if len(records) > expected_trajectories:
            raise ValueError("round contains more trajectories than declared")
        time.sleep(poll_interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Tail one external OPD reflection round")
    parser.add_argument("--round-root", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    args = parser.parse_args()
    client = DeepSeekReflectionClient(
        args.api_url,
        os.environ.get("DEEPSEEK_API_KEY", ""),
        args.model,
    )
    run_round(
        args.round_root,
        client,
        poll_interval_seconds=args.poll_interval_seconds,
    )


if __name__ == "__main__":
    main()
