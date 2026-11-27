The graft increased factual NLL by 0.01241 nats, exceeding the 0.005 limit. Most deterioration was in glucose predictions, confirming that the linear isolation did not prevent cross-drug effects. I’ll reject it and restore the validated checkpoint294, then check broader tuning cohorts. Its main insulin result was close to the record-derived estimate, but its other three readouts remained too weak.

```python
from pathlib import Path
import numpy as np

# Restore an already validated candidate; do not introduce another untested edit.
source300 = globals()["checkpoint294"]
assert isinstance(source300, dict), "Expected a named-tensor checkpoint."

restore300 = {}
for name, value in source300.items():
    if name not in SHAPES:
        continue
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32).copy()
    assert tuple(arr.shape) == tuple(SHAPES[name]), name
    assert np.isfinite(arr).all(), name
    restore300[name] = arr

assert restore300, "Checkpoint contains no model tensors."
assert set(edited()).issubset(restore300), "Incomplete rollback checkpoint."

for name in list(edited()):
    if name not in restore300:
        drop_edit(name)
for name, arr in restore300.items():
    original300 = load_original(name)
    if hasattr(original300, "detach"):
        original300 = original300.detach().cpu().numpy()
    if np.array_equal(arr, np.asarray(original300, dtype=np.float32)):
        drop_edit(name)
    else:
        save_edit(name, arr)

for name, arr in restore300.items():
    actual300 = load(name)
    if hasattr(actual300, "detach"):
        actual300 = actual300.detach().cpu().numpy()
    assert np.array_equal(np.asarray(actual300), arr), name

print("RESTORED checkpoint294 exactly.")
print("CURRENT EDITS:", edited())
print("Prior validation: factual delta +0.00185 nats/reading.")
print("Prior dose shifts: insulin x2 -12.385; insulin x0.5 +1.361;"
      " NE x2 +0.415; NE x0.5 -1.193.")
print("First-cohort record-derived references: -12.9555, +6.4778,"
      " +4.2797, -2.1398.")

notes300 = Path("/work/notes.md").read_text()
print("\nEARLY IDENTIFICATION NOTES")
print(notes300[:4500])
print("\nADDITIONAL RECORD-EVIDENCE EXCERPTS")
evidence300 = []
for line in notes300.splitlines():
    if any(term in line.lower() for term in
           ["identif", "instrument", "causal", "physiolog",
            "coefficient", "record-derived", "negative control"]):
        if len(line) <= 1400 and line not in evidence300:
            evidence300.append(line)
print("\n".join(evidence300[:8])[:3500])

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\nPython run 300; forwards 37. NE graft rejected: factual NLL "
        "1.65223 versus original 1.63982, delta +0.01241, exceeding "
        "+0.005; deterioration concentrated in glucose heads. Restored "
        "checkpoint294 exactly and verified all checkpoint tensors. "
        "No further Python runs remain. Selected validated alpha=.01 "
        "candidate: prior factual delta +.00185; first-cohort shifts "
        "(-12.385,+1.361,+.415,-1.193), against record-derived references "
        "(-12.9555,+6.4778,+4.2797,-2.1398). Main insulin response is close, "
        "but the remaining responses are underestimated; not a complete "
        "causal repair. Next: broader tuning factual and dose validation "
        "without further weight changes.\n"
    )
```
