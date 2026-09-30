from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rollout_enables_and_passes_versioned_lora_request():
    source = (
        REPO_ROOT
        / "verl"
        / "workers"
        / "rollout"
        / "vllm_rollout"
        / "vllm_rollout_spmd.py"
    ).read_text(encoding="utf-8")

    assert "enable_lora=self.lora_rank > 0" in source
    assert "max_lora_rank=self.lora_rank" in source
    assert "def set_lora_adapter(self, path: str, version: int):" in source
    assert "lora_request=self.lora_request" in source


def test_sharding_manager_exports_adapter_instead_of_full_model():
    source = (
        REPO_ROOT / "verl" / "workers" / "sharding_manager" / "fsdp_vllm.py"
    ).read_text(encoding="utf-8")
    worker_source = (REPO_ROOT / "verl" / "workers" / "fsdp_workers.py").read_text(
        encoding="utf-8"
    )

    assert "def select_lora_state_dict(" in source
    assert "if self.lora_rank > 0:" in source
    assert "self.export_lora_adapter(params)" in source
    assert "rollout=rollout" in worker_source
    assert "lora_rank=lora_rank" in worker_source


def test_rollout_builder_reads_lora_rank_from_model_config():
    worker_source = (REPO_ROOT / "verl" / "workers" / "fsdp_workers.py").read_text(
        encoding="utf-8"
    )

    rollout_builder = worker_source.split("def _build_rollout", 1)[1]
    rollout_builder = rollout_builder.split("def ", 1)[0]
    assert 'lora_rank = int(self.config.model.get("lora_rank", 0))' in rollout_builder
