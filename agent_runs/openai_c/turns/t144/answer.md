The rejected bin operations are removed, leaving the previously tested numeric gain. I’ll review the causal estimates and saved state directions before choosing a recurrent projection. A direction tied to treatment changes—not merely the condition that prompted treatment—would be the strongest candidate.

```python
from pathlib import Path
import re
import numpy as np

text = Path("/work/notes.md").read_text()
sections = re.split(r"(?m)(?=^## )", text)
wanted = (
    "Run 4", "Run 5", "Run 6", "Run 15", "Run 16",
    "Python run 21", "Python run 22", "Python run 23",
    "Python run 24", "Python run 90"
)
for section in sections:
    heading = section.splitlines()[0] if section.splitlines() else ""
    if any(heading.startswith("## " + prefix) for prefix in wanted):
        print(section[:1100])

print("\nRECURRENT TENSOR SHAPES")
for name in NAMES:
    if any(term in name for term in
           ("core.b_in", "core.c_out", "core.coda", "core.prelude_norm")):
        print(name, SHAPES[name])

print("\nSAVED DOSE DIRECTION NORMS")
for name in ("dose_gain4_72", "dose_state_coeff", "D_h", "D_out",
             "D72", "R72", "W72", "val90"):
    a = np.asarray(globals()[name])
    axis = 1 if a.shape[0] == 2 else 0
    print(name, np.linalg.norm(a, axis=axis))
    if a.shape == (2, 992):
        print("  out:", np.linalg.norm(a[:, :192], axis=1),
              "h:", np.linalg.norm(a[:, 192:960], axis=1),
              "z:", np.linalg.norm(a[:, 960:], axis=1))

print("\nRECORD-BASED TARGETS", target_probe)
print("BASELINE NLL", baseline_nll)

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Python run 102: selective recurrent design audit\n"
            "- Numeric gain 4 unchanged.\n"
            "- Reviewing treatment-direction provenance and recurrent tensor geometry.\n"
            "- No new candidate installed; forward passes used: 39.\n")
```