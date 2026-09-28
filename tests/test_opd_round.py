import json

import pytest

from recurrent.opd_round import load_complete_manifest, start_round


def test_round_input_and_completion_manifest_must_match(tmp_path):
    expected = start_round(
        tmp_path,
        policy_version=3,
        checkpoint="checkpoint-3",
        expected_task_count=18,
    )
    assert json.loads((tmp_path / "round_input.json").read_text())["trajectories_per_task"] == 8
    assert load_complete_manifest(tmp_path, expected) is None

    manifest = {**expected, "complete": True}
    (tmp_path / "round_manifest.json").write_text(json.dumps(manifest))
    assert load_complete_manifest(tmp_path, expected) == manifest

    manifest["policy_version"] = 4
    (tmp_path / "round_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="policy_version"):
        load_complete_manifest(tmp_path, expected)
