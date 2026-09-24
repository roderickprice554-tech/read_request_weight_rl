from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_actor_uses_precomputed_modulated_advantage_without_auxiliary_opd_loss():
    actor_source = (
        REPO_ROOT / "VST-RL" / "verl" / "workers" / "actor" / "dp_actor.py"
    ).read_text(encoding="utf-8")

    assert "compute_policy_loss(" in actor_source
    assert "skill_conditioned_topk_opd_loss(" not in actor_source
    assert "combine_vst_rl_and_opd_loss(" not in actor_source
    assert "_forward_opd_student_selected" not in actor_source


def test_trainer_converts_reward_to_correctness_and_builds_episode_only_skills():
    trainer_source = (
        REPO_ROOT / "VST-RL" / "verl" / "trainer" / "ppo" / "ray_trainer.py"
    ).read_text(encoding="utf-8")

    assert "reward_to_is_correct" in trainer_source
    assert "correctness_by_trajectory=" in trainer_source
    assert 'text = f"Episode skill: {episode_skill}"' in trainer_source
    assert 'text += f"\\nStep skill: {step_skill}"' in trainer_source
    assert 'metrics["skill_opd/non_key_episode_row_count"]' in trainer_source
    assert 'metrics["skill_opd/final_opd_token_count"]' in trainer_source
