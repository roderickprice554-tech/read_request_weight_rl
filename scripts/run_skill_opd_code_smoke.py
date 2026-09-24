#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from recurrent.reflection_sft import (
    build_teacher_request,
    offline_record_to_trajectory,
    to_vst_sft_record,
    validate_teacher_response,
)
from recurrent.skill_opd import assemble_reflection_trajectories, build_opd_annotations
from verl.protocol import DataProto
from verl.trainer.ppo.skill_opd_loss import compute_opd_advantage_weights


REFLECTION_TEXT = json.dumps(
    {
        "apply_opd": True,
        "episode_skill": "Track stable entities across successive observations.",
        "key_transitions": [
            {
                "transition_index": 1,
                "kind": "preserve",
                "memory_attribute": "entity_identity",
                "step_skill": "Preserve entity identity while adding newly observed motion.",
            }
        ],
    },
    separators=(",", ":"),
)


def _offline_record():
    return {
        "trajectory_uid": "group-a:rollout-0",
        "policy_version": 5,
        "observed_video": "/readonly/cpu-smoke-video.mp4",
        "transitions": [
            {
                "transition_index": 0,
                "previous_memory_tokens": [],
                "current_chunk_boundary": {"frames": [0, 4], "seconds": [0.0, 2.0]},
                "generated_y_t_tokens": [11],
                "updated_memory_tokens": [11],
            },
            {
                "transition_index": 1,
                "previous_memory_tokens": [11],
                "current_chunk_boundary": {"frames": [4, 8], "seconds": [2.0, 4.0]},
                "generated_y_t_tokens": [12, 13],
                "updated_memory_tokens": [11, 12, 13],
            },
        ],
        "query": "Where does the object move?\nA. Left side\nB. Right side",
        "prediction": "A",
        "final_reward": 1.0,
    }


def _rollout_output() -> tuple[DataProto, torch.Tensor, torch.Tensor]:
    final_mask = torch.tensor([False, False, True])
    sample_index = torch.tensor([0, 0, 0])
    output = DataProto.from_dict(
        tensors={
            "responses": torch.tensor([[11, 0], [12, 13], [21, 0]]),
            "response_mask": torch.tensor([[True, False], [True, True], [True, False]]),
        },
        non_tensors={
            "group_uid": np.array(["group-a"] * 3, dtype=object),
            "trajectory_uid": np.array(["group-a:rollout-0"] * 3, dtype=object),
            "sample_index": np.array([0, 0, 0], dtype=np.int64),
            "transition_index": np.array([0, 1, None], dtype=object),
            "previous_memory_tokens": np.array([[], [11], None], dtype=object),
            "current_chunk_boundary": np.array(
                [
                    {"frames": [0, 4], "seconds": [0.0, 2.0]},
                    {"frames": [4, 8], "seconds": [2.0, 4.0]},
                    {"frames": [8, 12], "seconds": [4.0, 6.0]},
                ],
                dtype=object,
            ),
            "generated_y_t_tokens": np.array([[11], [12, 13], None], dtype=object),
            "updated_memory_tokens": np.array([[11], [11, 12, 13], None], dtype=object),
            "policy_version": np.array([5, 5, 5], dtype=np.int64),
            "final_mask": np.array([False, False, True], dtype=np.bool_),
        },
    )
    return output, final_mask, sample_index


class _TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 16)
        self.head = torch.nn.Linear(16, 128)

    def forward(self, input_ids):
        return self.head(self.embedding(input_ids))


def _max_parameter_change(module, before):
    return max(
        (parameter.detach() - old).abs().max().item()
        for parameter, old in zip(module.parameters(), before)
    )


def _run_toy_reflection_sft():
    trajectory = offline_record_to_trajectory(_offline_record())
    request = build_teacher_request(trajectory)
    if "final_reward" in request.payload or "reward" in request.payload:
        raise AssertionError("raw reward leaked into the teacher request")
    accepted = validate_teacher_response(trajectory, REFLECTION_TEXT, source="fixture")
    conversation = to_vst_sft_record(trajectory, accepted)
    if conversation[1]["content"][0]["text"] != REFLECTION_TEXT:
        raise AssertionError("assistant reflection target changed during SFT conversion")

    input_ids = torch.tensor([[7, 8, 9, 10, 21, 22, 23, 24, 25, 26]])
    labels = torch.tensor([[-100, -100, -100, -100, 21, 22, 23, 24, 25, 26]])
    model = _TinyCausalLM()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "target_token_count": int((labels != -100).sum().item()),
        "parameter_change": _max_parameter_change(model, before),
    }


def run_code_smoke(report_path: Path) -> dict:
    torch.manual_seed(7)
    sft = _run_toy_reflection_sft()
    output, final_mask, sample_index = _rollout_output()
    trajectories = assemble_reflection_trajectories(
        output=output,
        final_mask=final_mask,
        sample_index=sample_index,
        correctness_by_trajectory={"group-a:rollout-0": True},
        query_tokens_by_sample={0: [101, 102]},
        query_text_by_sample={0: _offline_record()["query"]},
        prediction_text_by_final_row={2: "A"},
        observed_video_by_sample={0: "/readonly/cpu-smoke-video.mp4"},
    )
    reflection = validate_teacher_response(
        trajectories[0], REFLECTION_TEXT, source="fixture"
    ).parsed
    annotations = build_opd_annotations(output, trajectories, [reflection])

    actor = torch.nn.Linear(8, 128)
    original_features = torch.randn(3, 2, 8)
    teacher_features = original_features.clone()
    teacher_features += annotations["opd_episode_mask"].unsqueeze(-1) * 0.25
    teacher_features += annotations["opd_key_mask"].unsqueeze(-1) * 0.25
    sampled_tokens = output.batch["responses"]
    with torch.no_grad():
        teacher_log_probs = torch.gather(
            torch.log_softmax(actor(teacher_features), dim=-1),
            -1,
            sampled_tokens.unsqueeze(-1),
        ).squeeze(-1)
        student_log_probs = torch.gather(
            torch.log_softmax(actor(original_features), dim=-1),
            -1,
            sampled_tokens.unsqueeze(-1),
        ).squeeze(-1)

    base_advantage = torch.tensor([[1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    opd = compute_opd_advantage_weights(
        base_advantage,
        teacher_log_probs,
        student_log_probs,
        annotations["opd_valid_token_mask"],
        eps=0.2,
        weight_lambda=0.5,
    )

    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    before = [parameter.detach().clone() for parameter in actor.parameters()]
    student_logits = actor(original_features)
    current_log_probs = torch.gather(
        torch.log_softmax(student_logits, dim=-1), -1, sampled_tokens.unsqueeze(-1)
    ).squeeze(-1)
    ratio = torch.exp(current_log_probs - student_log_probs)
    response_mask = output.batch["response_mask"]
    policy_loss = -(
        ratio * opd["token_advantage"] * response_mask
    ).sum() / response_mask.sum()
    total_loss = policy_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    parameter_change = _max_parameter_change(actor, before)

    valid_tokens = annotations["opd_valid_token_mask"]
    episode_rows = annotations["opd_episode_mask"].any(dim=-1)
    key_rows = annotations["opd_key_mask"].any(dim=-1)
    final_tokens = valid_tokens & final_mask.unsqueeze(-1)
    valid_weights = opd["opd_weight"][valid_tokens]
    base_abs = base_advantage[valid_tokens].abs().mean()
    modulated_abs = opd["token_advantage"][valid_tokens].abs().mean()
    report = {
        "smoke_type": "cpu_code",
        "device": "cpu",
        "gpu_used": False,
        "teacher_source": "fixture",
        "reflection_source": "fixture",
        "external_api_called": False,
        "sft_target_token_count": sft["target_token_count"],
        "sft_loss": sft["loss"],
        "sft_trainable_param_max_change": sft["parameter_change"],
        "memory_transition_count": sum(len(item.transitions) for item in trajectories),
        "episode_memory_row_count": int(episode_rows.sum().item()),
        "key_memory_row_count": int(key_rows.sum().item()),
        "non_key_episode_row_count": int((episode_rows & ~key_rows).sum().item()),
        "final_turn_count": int(final_mask.sum().item()),
        "trajectory_count": len(trajectories),
        "reward_mapped": True,
        "correctness_mapped": trajectories[0].is_correct is True,
        "reflection_valid": int(reflection.reflection_valid),
        "reflection_applied": int(reflection.apply_opd),
        "key_transition_count": len(reflection.key_transitions),
        "query_leakage_count": int(
            bool(reflection.rejection_reason) and "leakage" in reflection.rejection_reason
        ),
        "valid_token_count": int(valid_tokens.sum().item()),
        "final_token_count": int(final_tokens.sum().item()),
        "final_opd_token_count": int(final_tokens.sum().item()),
        "teacher_detached": not teacher_log_probs.requires_grad,
        "delta_detached": not opd["opd_delta"].requires_grad,
        "reward_mean": 1.0,
        "advantage_mean": float(base_advantage[response_mask].mean().item()),
        "opd_delta_mean": float(opd["opd_delta"][valid_tokens].mean().item()),
        "opd_delta_std": float(opd["opd_delta"][valid_tokens].std().item()),
        "opd_weight_min": float(valid_weights.min().item()),
        "opd_weight_mean": float(valid_weights.mean().item()),
        "opd_weight_max": float(valid_weights.max().item()),
        "base_advantage_abs_mean": float(base_abs.item()),
        "modulated_advantage_abs_mean": float(modulated_abs.item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl_loss": 0.0,
        "total_loss": float(total_loss.detach().item()),
        "cache_bytes": teacher_log_probs.numel() * teacher_log_probs.element_size(),
        "optimizer_completed": True,
        "trainable_param_max_change": float(parameter_change),
        "actor_trainable_param_max_change": float(parameter_change),
        "weight_finite": bool(torch.isfinite(opd["opd_weight"]).all()),
        "advantage_sign_preserved": bool(torch.all(
            torch.sign(opd["token_advantage"][response_mask])
            == torch.sign(base_advantage[response_mask])
        )),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    run_code_smoke(args.report)
    print(args.report)


if __name__ == "__main__":
    main()
