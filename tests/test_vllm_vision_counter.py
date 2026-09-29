from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import VisionEncoderCounter


def test_vision_counter_records_forward_calls_and_encoded_items():
    counter = VisionEncoderCounter()
    counter.begin(["group-a"], "chunk-0", request_count=8)
    counter.record_forward(encoded_items=1)

    assert counter.snapshot()["group-a|chunk-0"] == {
        "request_count": 8,
        "vision_forward_count": 1,
        "vision_item_count": 1,
    }


def test_vision_counter_supports_arbitrary_group_sizes():
    counter = VisionEncoderCounter()
    counter.begin(["group-a"], "chunk-0", request_count=3)
    counter.record_forward(encoded_items=3)

    assert counter.snapshot()["group-a|chunk-0"]["request_count"] == 3
    assert counter.snapshot()["group-a|chunk-0"]["vision_item_count"] == 3
