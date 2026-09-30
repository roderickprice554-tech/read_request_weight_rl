from verl.workers.fsdp_workers import (
    actor_init_plan,
    select_visual_checkpoint_tensors,
)


def test_only_nonzero_fsdp_ranks_use_meta_initialization():
    assert actor_init_plan(rank=0, world_size=2, tie_word_embeddings=True) == {
        "load_checkpoint": True,
        "tie_word_embeddings": False,
        "use_meta": False,
    }
    assert actor_init_plan(rank=1, world_size=2, tie_word_embeddings=True) == {
        "load_checkpoint": False,
        "tie_word_embeddings": False,
        "use_meta": True,
    }
    assert actor_init_plan(rank=0, world_size=1, tie_word_embeddings=True) == {
        "load_checkpoint": True,
        "tie_word_embeddings": True,
        "use_meta": False,
    }


def test_visual_checkpoint_selection_strips_prefix_and_groups_shards():
    weight_map = {
        "visual.blocks.0.weight": "model-00001-of-00002.safetensors",
        "visual.merger.weight": "model-00002-of-00002.safetensors",
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
    }

    assert select_visual_checkpoint_tensors(weight_map) == {
        "model-00001-of-00002.safetensors": {
            "visual.blocks.0.weight": "blocks.0.weight"
        },
        "model-00002-of-00002.safetensors": {
            "visual.merger.weight": "merger.weight"
        },
    }
