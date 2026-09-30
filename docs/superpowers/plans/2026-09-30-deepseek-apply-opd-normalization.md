# DeepSeek `apply_opd` Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent PPO smoke training from being blocked when DeepSeek emits exact string representations of JSON booleans for `apply_opd`.

**Architecture:** Normalize only `apply_opd` values inside the DeepSeek client response boundary, immediately after JSON decoding. Pass the normalized JSON to the unchanged strict reflection validator and persist that same normalized interpretation as `raw_response`.

**Tech Stack:** Python, JSON, pytest

## Global Constraints

- Modify only the DeepSeek external reflection path and its focused tests.
- Accept only native booleans or string values equal to `true` or `false` after trimming and case folding.
- Preserve strict rejection of every other value and preserve the generic reflection parser unchanged.

---

### Task 1: Normalize DeepSeek string booleans

**Files:**
- Modify: `pipeline/deepseek_v4_flash.py`
- Modify: `tests/test_deepseek_v4_flash.py`

**Interfaces:**
- Produces: `_normalize_apply_opd(payload: dict) -> dict`
- Consumes: the decoded DeepSeek group response before `parse_and_validate_group_reflection`

- [ ] **Step 1: Write failing focused tests**

Add tests whose fake DeepSeek responses contain `"apply_opd": " TRUE "`, `"apply_opd": "false"`, a native boolean, and `"apply_opd": "yes"`. Assert the first three follow the existing validation semantics and the last still raises `ValueError("apply_opd must be a boolean")`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/test_deepseek_v4_flash.py`

Expected: string `true`/`false` cases fail with `apply_opd must be a boolean`.

- [ ] **Step 3: Implement the narrow normalization**

Decode the response once, copy each reflection row, and replace only exact trimmed/case-folded `true` and `false` strings. Serialize the normalized payload for both strict validation and `raw_response` extraction. Leave all other values untouched.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `python -m pytest -q tests/test_deepseek_v4_flash.py recurrent/test/test_reflection.py`

Expected: all tests pass.

- [ ] **Step 5: Commit and push**

Run `git diff --check`, commit only the plan, client, and focused test changes, then push `share_video_teacher_mask`.
