The rollout refuted this probe: insulin doubling barely changed, while norepinephrine responses drifted unnecessarily. I’ll revert it and inspect the causal evidence and action pathway. The next edit needs demonstrable dose sensitivity, not just good factual fit.

```python
from pathlib import Path
import numpy as np

failed_insulin_probe = {
    n: load(n).copy() if isinstance(load(n), np.ndarray) else load(n).clone()
    for n in edited()
}
for n in list(edited()):
    drop_edit(n)

note = """
## Rank-one probe rejected; Python 60, forward 7/50
- Factual NLL penalty: +0.00012 on first 100 tuning stays.
- Insulin x2: original -1.847, edited -1.837; half: .777 -> .780.
- NE x2: -.498 -> -.542; half: -.600 -> -.628.
- No useful intervention leverage. Reverted both value-head edits.
- Need an explicitly dose-sensitive edit, with causal targets established from records.
"""
p = Path("/work/notes.md")
p.write_text(p.read_text() + note)

print("DATA README")
print(Path("/data/README.md").read_text())
print("\nEARLY CAUSAL ANALYSIS")
print(p.read_text()[:7200])
print("\nHOURLY COLUMNS", event_col)
print("\nACTION / STATE TENSORS")
for n in NAMES:
    if any(s in n for s in ["encode_action", "embed.numeric", "core.prelude",
                             "z_post", "core.b_in", "core.c_out"]):
        print(n, SHAPES[n])
print("\nRELEVANT ANALYSIS VARIABLES")
print([k for k in sorted(list(globals()))
       if any(s in k.lower() for s in
              ["rd", "causal", "instrument", "policy", "dynamic", "coef", "effect"])])
print("\nCURRENT EDITS", edited())
```
