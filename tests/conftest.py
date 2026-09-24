import json


def trajectory_record(video):
    return {
        "trajectory_uid": "sample:rollout-0",
        "policy_version": 0,
        "observed_video": str(video),
        "transitions": [{
            "transition_index": 0,
            "previous_memory_tokens": [1],
            "current_chunk_boundary": [0.0, 1.0],
            "generated_y_t_tokens": [2],
            "updated_memory_tokens": [3],
        }],
        "query": "What happens?\nA. walking\nB. sitting",
        "prediction": "A",
        "final_reward": 1.0,
    }


def valid_skip_response():
    return json.dumps({
        "apply_opd": False,
        "episode_skill": None,
        "key_transitions": [],
        "skip_reason": "memory_cause_uncertain",
    })
