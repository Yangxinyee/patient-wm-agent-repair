The stress test returned zero for all four edited readouts, confirming that the edit removes dose-magnitude sensitivity. That does not establish the true effects. I’ll review the earlier record-based evidence and architecture before deciding whether to keep it; a stable, adjusted nonzero effect would argue against the null hypothesis.

```python
from pathlib import Path

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD DEFINITIONS ===")
print(Path("/data/README.md").read_text())

notes_path = Path("/work/notes.md")
notes = notes_path.read_text()
print("\n=== EARLIER NOTEBOOK EVIDENCE ===")
print(notes[:20000])
print("\n=== CURRENT EDITS ===")
print(edited())

entry = """
## Restart recovery and exposure stress test, Python run 36
Forward pass 7 (/work/fwd_8), 100 exposure-enriched tuning stays:
original insulin x2/half shifts +0.213/-0.279 over 885 reached stay-hours;
original NE x2/half shifts -0.002/+0.004 over 494 reached stay-hours.
Edited shifts are zero for all four interventions.
This confirms dose-magnitude invariance under the actual rollout, not the true causal effects.
Full tuning factual delta remains +0.0000816063 nats/reading (19,224 readings).
No weights changed. Reviewing earlier record-based evidence before final submission.
"""
notes_path.write_text(notes + entry)
```