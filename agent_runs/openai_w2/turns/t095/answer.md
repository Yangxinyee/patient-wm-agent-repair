The reconstruction is accurate for drug bundles but not observation bundles, so it is not yet safe to use for a weight edit. I’ll recover the architecture and existing helpers to isolate the mismatch. Accurate reconstruction across both bundle types would validate this route; persistent errors would rule it out.

```python
from pathlib import Path
import inspect

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== RECORD SCHEMA ===")
print(Path("/data/README.md").read_text())

print("\n=== KERNEL INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_"):
        continue
    if inspect.isfunction(obj):
        print("FUNCTION", name, str(inspect.signature(obj)),
              "names:", obj.__code__.co_names)
    elif hasattr(obj, "shape") and any(
        s in name.lower() for s in
        ["state", "feat", "post", "obs", "action", "train", "tuning", "bundle", "atom"]
    ):
        print("ARRAY", name, type(obj).__name__, obj.shape)
    elif isinstance(obj, dict) and any(
        s in name.lower() for s in ["vocab", "recon", "config", "unit", "bin", "cache"]
    ):
        print("DICT", name, "keys:", list(obj)[:30])

print("\nCurrent edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 81: context recovery\n"
            "No edits. Drug-only posterior reconstruction remains accurate (~4e-5), "
            "but observation reconstruction is inaccurate (~0.14 RMSE). "
            "Recovering architecture and helpers before using reconstructed features.\n")
```