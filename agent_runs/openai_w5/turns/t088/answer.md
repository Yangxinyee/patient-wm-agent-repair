The corrected readout is repeatable for identical states; the old one was not. Earlier response comparisons may therefore contain sampling noise. I’ll inspect the corrected routines and architecture before choosing an edit. Agreement with saved GPU predictions would support trusting the reconstruction.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA README ===")
print(Path("/data/README.md").read_text())

print("\n=== RELEVANT PERSISTENT OBJECTS ===")
keys = [k for k in globals() if not k.startswith("_")
        and any(s in k.lower() for s in
                ("correct", "readout", "response", "head", "recurr",
                 "latent", "candidate", "direction", "causal", "arx"))]
for k in sorted(keys):
    obj = globals()[k]
    shape = getattr(obj, "shape", None)
    if callable(shape):
        shape = None
    print(k, type(obj).__name__, shape)
    if callable(obj) and getattr(obj, "__module__", None) == "__main__":
        try:
            print(inspect.getsource(obj)[:14000])
        except (OSError, TypeError):
            print("signature:", inspect.signature(obj))

print("\n=== CURRENT EDITS ===")
print(edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after run 75\n"
            "- Identical-state test exposed randomness in the old readout; "
            "the corrected readout was exactly repeatable.\n"
            "- No edits retained. Rechecking architecture and corrected routines "
            "before drawing finite-dose conclusions or submitting an edit.\n")
```