from pathlib import Path

import numpy as np
import torch

from recurrent.skill_opd import scatter_teacher_cache, select_valid_teacher_rows
from verl.protocol import DataProto

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_actor_uses_precomputed_modulated_advantage_without_auxiliary_opd_loss():
    actor_source = (
        REPO_ROOT / "verl" / "workers" / "actor" / "dp_actor.py"
    ).read_text(encoding="utf-8")

    assert "compute_policy_loss(" in actor_source
    assert "skill_conditioned_topk_opd_loss(" not in actor_source
    assert "combine_vst_rl_and_opd_loss(" not in actor_source
    assert "_forward_opd_student_selected" not in actor_source


def test_trainer_converts_reward_to_correctness_and_builds_episode_only_skills():
    trainer_source = (
        REPO_ROOT / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "reward_to_is_correct" in trainer_source
    assert "correctness_by_trajectory=" in trainer_source
    assert 'text = f"Episode skill: {episode_skill}"' in trainer_source
    assert 'text += f"\\nStep skill: {step_skill}"' in trainer_source
    assert 'metrics["skill_opd/non_key_episode_row_count"]' in trainer_source
    assert 'metrics["skill_opd/final_opd_token_count"]' in trainer_source


def _teacher_batch(valid_mask):
    rows = len(valid_mask)
    return DataProto.from_dict(
        tensors={
            "responses": torch.arange(rows * 2).reshape(rows, 2),
            "opd_valid_token_mask": torch.tensor(valid_mask, dtype=torch.bool),
        },
        non_tensors={
            "trajectory_uid": np.asarray([f"t-{index}" for index in range(rows)], dtype=object),
            "policy_version": np.asarray([3] * rows),
            "transition_index": np.asarray(list(range(rows))),
            "multi_modal_embeds": np.asarray(
                [{"video": torch.tensor([[index]])} for index in range(rows)], dtype=object
            ),
        },
    )


def test_teacher_rows_are_selected_with_tensor_and_visual_alignment():
    batch = _teacher_batch([[True, False], [False, False], [False, True]])

    sparse, valid_rows = select_valid_teacher_rows(batch)

    assert valid_rows.tolist() == [True, False, True]
    assert sparse.batch["responses"].tolist() == [[0, 1], [4, 5]]
    assert sparse.non_tensor_batch["trajectory_uid"].tolist() == ["t-0", "t-2"]
    assert [item["video"].item() for item in sparse.non_tensor_batch["multi_modal_embeds"]] == [0, 2]


def test_all_invalid_teacher_rows_return_no_sparse_batch():
    sparse, valid_rows = select_valid_teacher_rows(
        _teacher_batch([[False, False], [False, False]])
    )

    assert sparse is None
    assert not valid_rows.any()


def test_sparse_teacher_cache_scatter_restores_full_row_shape():
    rollout = _teacher_batch([[True, False], [False, False], [False, True]])
    sparse_cache = DataProto.from_dict(
        tensors={"opd_teacher_log_probs": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    )

    full = scatter_teacher_cache(sparse_cache, rollout)

    assert full.batch["opd_teacher_log_probs"].tolist() == [
        [1.0, 2.0], [0.0, 0.0], [3.0, 4.0]
    ]
    assert torch.equal(full.batch["opd_valid_token_mask"], rollout.batch["opd_valid_token_mask"])
    assert full.non_tensor_batch["trajectory_uid"].tolist() == ["t-0", "t-1", "t-2"]


def test_actor_teacher_cache_dispatches_precomputed_visual_embeddings():
    actor_source = (REPO_ROOT / "verl" / "workers" / "actor" / "dp_actor.py").read_text(
        encoding="utf-8"
    )

    assert 'if "multi_modal_embeds" in model_data' in actor_source
    assert "self._forward_micro_batch_embed(" in actor_source
