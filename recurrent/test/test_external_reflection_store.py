import json
from dataclasses import asdict

import pytest

from recurrent.external_reflection_store import ExternalReflectionStore
from recurrent.reflection import parse_and_validate_reflection
from recurrent.reflection_sft import offline_record_to_trajectory


def _trajectory(uid, version=0):
    return offline_record_to_trajectory({
        "trajectory_uid": uid, "policy_version": version, "observed_video": "x.mp4",
        "transitions": [{"transition_index": 0, "previous_memory_tokens": [1], "current_chunk_boundary": [0, 1], "generated_y_t_tokens": [2], "updated_memory_tokens": [3]}],
        "query": "Question?", "prediction": "answer", "final_reward": 1,
    })


def _row(trajectory):
    raw = json.dumps({"apply_opd": False, "episode_skill": None, "key_transitions": [], "skip_reason": "memory_cause_uncertain"})
    return {**asdict(parse_and_validate_reflection(raw, trajectory, trajectory.policy_version)), "raw_response": raw}


def test_store_returns_reflections_in_trajectory_order(tmp_path):
    a, b = _trajectory("a"), _trajectory("b")
    path = tmp_path / "r.jsonl"
    path.write_text("\n".join(json.dumps(_row(x)) for x in (b, a)), encoding="utf-8")
    assert [x.trajectory_uid for x in ExternalReflectionStore(path).get_many([a, b])] == ["a", "b"]


def test_store_rejects_stale_policy_version(tmp_path):
    old, new = _trajectory("a", 1), _trajectory("a", 2)
    path = tmp_path / "r.jsonl"; path.write_text(json.dumps(_row(old)), encoding="utf-8")
    with pytest.raises(ValueError, match="policy_version"):
        ExternalReflectionStore(path).get_many([new])
