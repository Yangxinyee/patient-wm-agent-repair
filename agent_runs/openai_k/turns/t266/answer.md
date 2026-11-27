The corrected joins validate dose encoding, but the local value predictions still disagree with the official tool. I’ll recover the exact architecture and replay interfaces before calibrating another edit. Matching official predictions would validate this route; continued disagreement would rule it out.

```python
from pathlib import Path
import inspect

print("=== AUTHORITATIVE ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== TENSOR INTERFACES ===")
for name in NAMES:
    if any(s in name.lower() for s in
           ["core", "head", "value", "read", "z_in", "z_post", "action", "dose"]):
        print(name, SHAPES[name])

print("\n=== RETAINED FUNCTION INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if inspect.isfunction(obj) and getattr(obj, "__module__", "") == "__main__":
        try:
            sig = str(inspect.signature(obj))
        except Exception:
            sig = "?"
        print(name, sig)

print("\n=== REPLAY / CALIBRATION OBJECT INVENTORY ===")
for name, obj in sorted(list(globals().items())):
    if any(s in name.lower() for s in
           ["replay", "capture", "context", "head", "candidate", "feature",
            "coef", "target", "backup", "fit", "code", "source"]):
        if isinstance(obj, str):
            print(name, "string", len(obj), repr(obj[:100]))
        elif isinstance(obj, dict):
            print(name, "dict", len(obj), list(obj.keys())[:15])
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, tuple(obj.shape))
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, len(obj))
print("\nCurrently edited:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 239: re-establish trustworthy edit interfaces\n"
            "- Corrected joins validate dose encoding and reveal factual-quiet E features.\n"
            "- Local value predictions remain unvalidated; do not use them to calibrate an edit yet.\n"
            "- Recovering authoritative architecture and retained helper inventory in one pass.\n"
            "- Submitted weights unchanged. Python runs: 239; forward passes: 17.\n")
```