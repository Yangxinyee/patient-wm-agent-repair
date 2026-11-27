The last run failed its logit reconstruction check, so its reported loss increase is not yet a reliable basis for accepting or rejecting an edit. I’ll inspect the architecture, active weights, and recent variables without changing weights. My working hypothesis is a reconstruction mistake; matching direct logits would confirm a corrected calculation, while continued disagreement would refute it.

```python
from pathlib import Path
import hashlib
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

c179_names = list(globals())
print("=== RECENT VARIABLES ===")
for name in sorted(n for n in c179_names
                   if any(n.startswith(f"r{k}") for k in range(171, 179))):
    obj = globals()[name]
    description = f"{type(obj).__name__}"
    if hasattr(obj, "shape"):
        description += f" shape={obj.shape}"
    elif isinstance(obj, dict):
        description += f" keys={list(obj)[:35]}"
    elif isinstance(obj, (list, tuple)):
        description += f" length={len(obj)}"
    elif isinstance(obj, (int, float, str, np.number)):
        description += f" value={str(obj)[:250]}"
    print(name, description)

print("=== ACTIVE EDIT FINGERPRINTS ===")
for name in edited():
    tensor = load(name)
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    original = load_original(name)
    original = original.detach().cpu().numpy() if hasattr(original, "detach") else np.asarray(original)
    print(name, array.shape,
          "sha256", hashlib.sha256(array.tobytes()).hexdigest()[:16],
          "max_change", float(np.max(np.abs(array - original))),
          "finite", bool(np.isfinite(array).all()))

print("=== OTHER RELEVANT VARIABLE NAMES ===")
print(sorted(n for n in c179_names
             if not n.startswith("_")
             and any(s in n.lower() for s in
                     ["logit", "guard", "gate", "ne_", "nore", "cache", "head", "candidate", "target"])))

print("=== NOTEBOOK END ===")
print(Path("/work/notes.md").read_text()[-5000:])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 179: restart after failed reconstruction assertion\n"
            "Run 178 had a maximum logit reconstruction error of 1.640755 and "
            "cached NLL delta +0.013017. These diagnostics are not accepted as "
            "evidence about candidate performance until the reconstruction is "
            "audited. No weights changed in this audit run.\n")
```