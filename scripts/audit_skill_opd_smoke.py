#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path


REQUIRED_FIELDS = {
    "memory_transition_count", "final_turn_count", "trajectory_count",
    "reward_mapped", "correctness_mapped", "reflection_valid",
    "reflection_applied", "key_transition_count", "query_leakage_count",
    "valid_token_count", "final_token_count", "final_opd_token_count",
    "teacher_detached", "delta_detached", "reward_mean", "advantage_mean",
    "opd_delta_mean", "opd_delta_std", "opd_weight_min", "opd_weight_mean",
    "opd_weight_max", "base_advantage_abs_mean",
    "modulated_advantage_abs_mean", "policy_loss", "kl_loss", "total_loss",
    "cache_bytes", "optimizer_completed", "trainable_param_max_change",
    "weight_finite", "advantage_sign_preserved",
}


def audit_enabled_smoke(report: dict) -> None:
    missing = REQUIRED_FIELDS - set(report)
    if missing:
        raise ValueError(f"missing smoke fields: {sorted(missing)}")
    if report["memory_transition_count"] < 2:
        raise ValueError("smoke needs at least two memory transitions")
    if report["final_turn_count"] != report["trajectory_count"] or report["final_turn_count"] < 1:
        raise ValueError("each trajectory must have exactly one final turn")
    for key in ("reward_mapped", "correctness_mapped", "teacher_detached",
                "delta_detached", "optimizer_completed", "weight_finite",
                "advantage_sign_preserved"):
        if report[key] is not True:
            raise ValueError(f"{key} was not verified")
    if report["reflection_valid"] < 1 or report["reflection_applied"] < 1:
        raise ValueError("smoke must contain a valid applied reflection")
    if report["key_transition_count"] < 1 or report["valid_token_count"] < 1:
        raise ValueError("OPD smoke has no valid reflected tokens")
    if report["query_leakage_count"] != 0:
        raise ValueError("query leakage was detected")
    if report["final_token_count"] != 0 or report["final_opd_token_count"] != 0:
        raise ValueError("final answer tokens entered OPD")
    for key in REQUIRED_FIELDS & {
        "reward_mean", "advantage_mean", "opd_delta_mean", "opd_delta_std",
        "opd_weight_min", "opd_weight_mean", "opd_weight_max",
        "base_advantage_abs_mean", "modulated_advantage_abs_mean",
        "policy_loss", "kl_loss", "total_loss", "trainable_param_max_change",
    }:
        if not math.isfinite(float(report[key])):
            raise ValueError(f"{key} is not finite")
    if float(report["opd_weight_min"]) < 0.9 - 1e-6:
        raise ValueError("OPD weight fell below its configured bound")
    if float(report["opd_weight_max"]) > 1.1 + 1e-6:
        raise ValueError("OPD weight exceeded its configured bound")
    if report["cache_bytes"] <= 0:
        raise ValueError("teacher cache byte count must be positive")
    if report["trainable_param_max_change"] <= 0:
        raise ValueError("no trainable Actor parameter changed")


def audit_disabled_smoke(report: dict) -> None:
    expected = {"skill_opd_enabled", "opd_branch_executed", "optimizer_completed", "rl_loss"}
    missing = expected - set(report)
    if missing:
        raise ValueError(f"missing disabled smoke fields: {sorted(missing)}")
    if report["skill_opd_enabled"] is not False or report["opd_branch_executed"] is not False:
        raise ValueError("disabled smoke executed the OPD branch")
    if report["optimizer_completed"] is not True or not math.isfinite(float(report["rl_loss"])):
        raise ValueError("disabled smoke did not complete with finite loss")


def audit_cpu_code_smoke(report: dict) -> None:
    audit_enabled_smoke(report)
    expected = {
        "smoke_type", "reflection_source", "teacher_source", "external_api_called",
        "device", "gpu_used", "episode_memory_row_count", "key_memory_row_count",
        "non_key_episode_row_count", "sft_target_token_count", "sft_loss",
        "sft_trainable_param_max_change", "actor_trainable_param_max_change",
    }
    missing = expected - set(report)
    if missing:
        raise ValueError(f"missing CPU code smoke fields: {sorted(missing)}")
    if report["smoke_type"] != "cpu_code" or report["device"] != "cpu" or report["gpu_used"] is not False:
        raise ValueError("report is not a CPU code smoke")
    if report["reflection_source"] != "fixture" or report["teacher_source"] != "fixture" or report["external_api_called"] is not False:
        raise ValueError("CPU code smoke must use local fixtures")
    if report["episode_memory_row_count"] != report["memory_transition_count"]:
        raise ValueError("episode skill must supervise every memory row")
    if report["key_memory_row_count"] < 1 or report["non_key_episode_row_count"] < 1:
        raise ValueError("CPU code smoke lacks required memory rows")
    for key in ("sft_loss", "sft_trainable_param_max_change", "actor_trainable_param_max_change"):
        if not math.isfinite(float(report[key])) or float(report[key]) <= 0:
            raise ValueError(f"{key} is not a positive finite value")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--cpu-code", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    if args.disabled and args.cpu_code:
        parser.error("--disabled and --cpu-code are mutually exclusive")
    if args.disabled:
        audit_disabled_smoke(report)
    elif args.cpu_code:
        audit_cpu_code_smoke(report)
    else:
        audit_enabled_smoke(report)
    print("Skill OPD smoke audit passed")


if __name__ == "__main__":
    main()
