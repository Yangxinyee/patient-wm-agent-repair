All four edited tensors exactly match the GPU-validated checkpoint; no unvalidated changes remain. I’ll now review the identification and edit-design summaries to distinguish supported insulin conclusions from unresolved norepinephrine effects. Consistent record-based estimates would support the rationale; conflicting estimates would limit the causal claims.

```python
from pathlib import Path

print("=== RECORD-BASED IDENTIFICATION AND EDIT DESIGN ===")
for i in [5, 6, 9, 12, 13, 57, 81, 93, 98]:
    section = _sections295[i]
    print(f"\n--- NOTEBOOK SECTION {i} ---")
    print(section[:3200])
    if len(section) > 3200:
        print(f"[Section truncated: {len(section)} characters total]")

print("\n=== RECOVERED INSULIN DYNAMIC PARAMETERS ===")
for key in ["_target_beta", "_target_rho"]:
    if key in globals():
        print(key, globals()[key])
if "ins_target" in globals():
    print("Training-record dynamic-model insulin_x2 target:", float(ins_target[0]))

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Submission integrity audit; Python 296, forwards 50/50\n"
        "- Python295 verified exact equality of all four current tensors with "
        "amplitude093_checkpoint; no weights changed.\n"
        "- The three rollout-omitted tuning stays have no glucose_finger or "
        "map_arterial readings after hour12; their omission explains the "
        "397-stay rollout CSV coverage.\n"
        "- Reviewing identification and graft summaries before final submission. "
        "No further tuning or unvalidated edits planned.\n"
    )
```
