The weights are still untouched, so the reconstruction mismatch is in the local helper, not the model. I’ll recover the architecture and existing record-based effect estimates before editing. Consistent encoding and reproducible estimates would support proceeding; disagreement would require resolving those first.

```python
from pathlib import Path
import re, inspect, numpy as np, torch

arch81 = Path("/model/ARCHITECTURE.md").read_text()
print("=== ARCHITECTURE ===")
print(arch81[:6800])

notes81 = Path("/work/notes.md").read_text()
print("\n=== EARLY EVIDENCE ===")
early81 = notes81.split("## Python 35")[0]
key81 = [
    line for line in early81.splitlines()
    if re.search(r"effect|causal|placebo|threshold|discontin|impulse|lag|NLL|rollout|doubl|crossfit", line, re.I)
]
print("\n".join(key81)[:2300])

print("\n=== RETAINED HELPERS AND DATA ===")
for name81, obj81 in sorted(list(globals().items())):
    if inspect.isfunction(obj81) and obj81.__module__ == "__main__":
        if re.search(r"repair|posterior|atom|transition|flow|head|panel|impulse|cross|encoder", name81, re.I):
            print(name81, inspect.signature(obj81))
    elif isinstance(obj81, (np.ndarray, torch.Tensor)):
        if re.search(r"features|hidden|post|states|target|panel|impulse|rates", name81, re.I):
            print(name81, tuple(obj81.shape))
print("Submitted edits:", edited())
with open("/work/notes.md", "a") as f81:
    f81.write("\n## Python 81; forward 6 — restart recovery\n"
              "- No weights changed. Recovering architecture and existing identification evidence in one compact check.\n"
              "- Local encoding mismatch must not be mistaken for a defect in the original weights.\n")
```
