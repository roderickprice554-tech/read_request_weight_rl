# DeepSeek `apply_opd` Normalization Design

## Scope

Fix only the external DeepSeek reflection path when the model returns `apply_opd` as a JSON string instead of a JSON boolean.

## Behavior

Before calling the existing strict reflection validator, `DeepSeekReflectionClient` will normalize each reflection row as follows:

- `"true"`, with optional surrounding whitespace and any letter case, becomes `true`.
- `"false"`, with optional surrounding whitespace and any letter case, becomes `false`.
- Native JSON booleans remain unchanged.
- Every other value remains unchanged and is rejected by the existing validator.

The generic reflection parser, PPO update path, LoRA synchronization, and all other response fields remain unchanged. The normalized payload is also used when recording `raw_response`, so persisted evidence matches the validated interpretation.

## Verification

Add focused tests proving string booleans from the DeepSeek client are accepted, native booleans are unchanged, and unrelated strings still fail validation. Run the existing reflection tests as regression coverage.
