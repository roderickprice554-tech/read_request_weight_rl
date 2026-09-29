import json
from pathlib import Path
import time

from recurrent.reflection import parse_and_validate_reflection


def publish_trajectories(path, trajectories):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if path.exists():
        existing = {
            json.loads(line)["trajectory_uid"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    with path.open("a", encoding="utf-8") as stream:
        for trajectory in trajectories:
            if trajectory.trajectory_uid in existing:
                continue
            stream.write(json.dumps(trajectory.to_analyzer_input(), ensure_ascii=False) + "\n")
            stream.flush()
            existing.add(trajectory.trajectory_uid)


class ExternalReflectionStore:
    def __init__(self, path, *, poll_interval_seconds=1.0, timeout_seconds=None):
        self.path = Path(path)
        self.rows = {}
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self.refresh()

    def refresh(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row["trajectory_uid"]
            if uid in self.rows:
                if self.rows[uid] != row:
                    raise ValueError(f"conflicting reflection trajectory_uid: {uid}")
                continue
            self.rows[uid] = row

    def get_many(self, trajectories):
        results = []
        for trajectory in trajectories:
            row = self.rows.get(trajectory.trajectory_uid)
            if row is None:
                raise ValueError(f"missing external reflection: {trajectory.trajectory_uid}")
            if int(row["policy_version"]) != trajectory.policy_version:
                raise ValueError(f"policy_version mismatch for {trajectory.trajectory_uid}")
            result = parse_and_validate_reflection(
                row["raw_response"], trajectory, trajectory.policy_version
            )
            if not result.reflection_valid:
                raise ValueError(result.rejection_reason or "invalid external reflection")
            results.append(result)
        return results

    def wait_for_many(self, trajectories):
        deadline = None if self.timeout_seconds is None else time.monotonic() + self.timeout_seconds
        while True:
            self.refresh()
            missing = [item.trajectory_uid for item in trajectories if item.trajectory_uid not in self.rows]
            if not missing:
                return self.get_many(trajectories)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for external reflections: {missing}")
            time.sleep(self.poll_interval_seconds)
