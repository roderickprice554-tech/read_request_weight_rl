from verl.workers.rollout.vllm_rollout import _uses_customized_vllm


def test_vllm_version_selection_uses_semantic_ordering():
    assert _uses_customized_vllm("0.6.3") is True
    assert _uses_customized_vllm("0.11.0") is False
