The paired original and repaired states are aligned, but the last result does not yet validate the dose detector on repaired states. I’ll recover the record-derived targets and prior validation results before changing weights. Any next edit must improve dose responses while keeping factual NLL within the allowed margin.

```python
from pathlib import Path
import re, numpy as np

print("ARCHITECTURE")
print(Path("/model/ARCHITECTURE.md").read_text())

notes185 = Path("/work/notes.md").read_text()
print("\nDECISIVE NOTEBOOK LINES")
patterns185 = [
    r"true", r"causal", r"target", r"baseline77",
    r"fwd_", r"norepinephrine", r"NLL", r"insulin.*x2"
]
hits185 = [line for line in notes185.splitlines()
           if any(re.search(p, line, re.I) for p in patterns185)
           and not line.startswith("##")]
for line in hits185[:35] + hits185[-45:]:
    print(line[:700])

print("\nSAVED BASELINE")
with np.load("/work/baseline77.npz", allow_pickle=False) as a185:
    print([(k, a185[k].shape) for k in a185.files])
print("Currently edited:", edited())

with open("/work/notes.md", "a") as f185:
    f185.write("\n## Run 185: recover decision evidence after restart\n"
               "Read architecture first and recovered compact causal/validation evidence. "
               "Current insulin baseline retained; no weights changed. "
               "Next decision is a factual-budgeted correction, not an unvalidated detector refit. "
               "Python runs: 185; forwards: 22.\n")
```