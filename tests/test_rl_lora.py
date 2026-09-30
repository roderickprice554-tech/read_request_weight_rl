from pathlib import Path

import torch

from verl.workers.fsdp_workers import trainable_parameters


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_optimizer_parameter_filter_excludes_frozen_base_weights():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    model[0].requires_grad_(False)

    selected = list(trainable_parameters(model))

    assert selected == list(model[1].parameters())


def test_rl_actor_has_explicit_language_lora_configuration():
    source = (REPO_ROOT / "verl" / "workers" / "fsdp_workers.py").read_text(
        encoding="utf-8"
    )
    config = (REPO_ROOT / "verl" / "trainer" / "config" / "ppo_trainer.yaml").read_text(
        encoding="utf-8"
    )

    assert 'if role == "actor" and lora_rank > 0:' in source
    assert "get_peft_model(" in source
    assert 'exclude_modules=r".*visual(?:\\..*)?"' in source
    assert "trainable_parameters(actor_module_fsdp)" in source
    assert "lora_rank: 0" in config
    assert "lora_alpha: 32" in config
    for module in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert module in config
