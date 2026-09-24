import json
from pathlib import Path

from recurrent.reflection import parse_and_validate_reflection


class ExternalReflectionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid = row["trajectory_uid"]
            if uid in self.rows:
                raise ValueError(f"duplicate reflection trajectory_uid: {uid}")
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
