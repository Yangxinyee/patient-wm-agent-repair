The local reconstruction still disagrees with the forward tool, so it is not yet reliable enough to guide weight edits. The gap and latent tests did not resolve this. I’ll inspect the exact architecture and retained helper code next; matching the original predictions would confirm that we have recovered the computation correctly.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("\n=== DATA DEFINITIONS ===")
print(Path("/data/README.md").read_text())

print("\n=== RETAINED OBJECTS ===")
for name in ["head_rows", "state_x", "head_weights", "reconstruct_head",
             "train", "tuning", "df", "wide", "NAMES", "VOCAB"]:
    if name in globals():
        obj = globals()[name]
        print(name, type(obj).__name__, getattr(obj, "shape", None))
        if isinstance(obj, dict):
            print(" keys:", list(obj)[:30])
        elif hasattr(obj, "columns"):
            print(" columns:", list(obj.columns))
            if name == "head_rows":
                print(obj.head(8).to_string(index=False))

print("\n=== HEAD HELPER SOURCE ===")
if "reconstruct_head" in globals():
    try:
        print(inspect.getsource(reconstruct_head))
    except (OSError, TypeError):
        ip = get_ipython() if "get_ipython" in globals() else None
        history = ip.history_manager.input_hist_raw if ip is not None else globals().get("In", [])
        for source in history:
            if "def reconstruct_head(" in source:
                print(source)

print("\n=== OTHER RETAINED HELPERS ===")
print([name for name, obj in list(globals().items())
       if callable(obj) and getattr(obj, "__module__", None) == "__main__"])
print("Current edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Reconstruction recovery — Python 17; forward 4\n"
            "- Gap, bin-boundary and latent tests did not reproduce original predictions.\n"
            "- Inspecting the exact architecture and retained helper code before editing weights.\n"
            "- No weights changed.\n")
```