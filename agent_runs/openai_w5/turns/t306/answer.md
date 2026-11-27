The rollback stopped because the checkpoint omits `core.a_in`, which was unedited before the trial. The flow tensor was already restored. I’ll restore `core.a_in` from the original weights and then recover the evidence needed for the next decision. This is a bookkeeping correction, not evidence about drug effects.

```python
from pathlib import Path
import json, re, numpy as np, pandas as pd, torch
from safetensors.torch import load_file
BASE264 = load_file("/work/checkpoint_run264.safetensors")
for n in ("core.flow.a", "core.a_in"):
    if n in BASE264:
        save_edit(n, BASE264[n])
    else:
        drop_edit(n)
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 24–25 / runs 265–266: persistence trial rejected as-is\n"
            "- Insulin tau=1500 s: factual NLL 1.64604 vs 1.63933 (+0.00671), exceeds guardrail.\n"
            "- Rollouts: insulin x2 -8.110, half +1.952; NE x2 +4.362, half -1.332.\n"
            "- Magnitudes increased, asymmetry remains. Trial saved in checkpoint_trial264.safetensors.\n"
            "- Restored pre-trial flow; dropped a_in edit because checkpoint omitted this previously original tensor.\n")
print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== DATA README ===")
print(Path("/data/README.md").read_text())
notes266 = Path("/work/notes.md").read_text()
print("=== EARLY IDENTIFICATION NOTES ===")
print(notes266[:14000])
print("=== IDENTIFICATION / EDIT OBJECTS ===")
inventory266 = []
for k, v in list(globals().items()):
    if k.startswith("_") or k in ("notes266", "inventory266"):
        continue
    if re.search(r"(coef|effect|caus|target|center|beta|reg|expo|head|fit|sensor|theta|candidate)", k, re.I):
        shape = getattr(v, "shape", None)
        desc = str(shape) if shape is not None else (list(v.keys())[:20] if isinstance(v, dict) else "")
        inventory266.append((k, type(v).__name__, desc))
print(inventory266[:130])
print("=== PRIOR GPU SUMMARIES ===")
for k in (6, 18, 21, 22, 24):
    p = Path(f"/work/fwd_{k}/summary.json")
    if p.exists():
        print(k, p.read_text()[:4000])
print("Restored private flow:", np.asarray(load("core.flow.a"))[[15,21,61,66]])
print("core.a_in edited:", "core.a_in" in edited())
```