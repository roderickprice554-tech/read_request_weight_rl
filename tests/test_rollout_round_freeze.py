import torch

from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager
from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


def _manager(manager_type):
    manager = manager_type.__new__(manager_type)
    manager.requested_policy_version = None
    manager.frozen_policy_version = None
    manager.frozen_params = None
    return manager


def _assert_round_freeze(manager):
    manager.set_policy_version(4)
    first = manager._params_for_rollout({"weight": torch.tensor([1.0])})
    same_round = manager._params_for_rollout({"weight": torch.tensor([2.0])})
    assert first["weight"].item() == 1.0
    assert same_round["weight"].item() == 1.0

    manager.set_policy_version(5)
    next_round = manager._params_for_rollout({"weight": torch.tensor([3.0])})
    assert next_round["weight"].item() == 3.0


def test_vllm_rollout_weights_are_frozen_until_policy_version_changes():
    _assert_round_freeze(_manager(FSDPVLLMShardingManager))


def test_sglang_rollout_weights_are_frozen_until_policy_version_changes():
    _assert_round_freeze(_manager(FSDPSGLangShardingManager))
