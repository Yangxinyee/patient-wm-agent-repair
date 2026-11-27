The last check confirms that the restored weights match the validated edit. It also shows that bin rank is not proportional to glucose, so a simple mean-based calibration could be misleading. I’ll recover the architecture, causal evidence, and saved helpers before choosing another trial. Any new edit must improve the paired rollout while preserving factual fit.

```python
from pathlib import Path
import inspect, numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD DOCUMENTATION ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK ===")
print(notes_text[:16000])
print("=== NOTEBOOK HEADINGS ===")
print("\n".join(x for x in notes_text.splitlines() if x.startswith("#")))

print("=== CURRENT EDITS ===", edited())
print("=== RELEVANT KERNEL VARIABLES ===")
for name, obj in list(globals().items()):
    if name.startswith("_"):
        continue
    if inspect.isfunction(obj):
        try:
            print("FUNCTION", name, inspect.signature(obj))
        except Exception:
            pass
    elif any(s in name.lower() for s in
             ("checkpoint", "cache", "ridge", "causal", "effect", "target",
              "gate", "activity", "contrast", "train", "tuning", "rd_", "iv_")):
        if isinstance(obj, np.ndarray):
            print(name, "array", obj.shape)
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif isinstance(obj, dict):
            print(name, "dict keys", list(obj)[:20])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj), "first", obj[:5])
        elif isinstance(obj, (int, float, str, bool)):
            print(name, repr(obj)[:200])

with open("/work/notes.md", "a") as f:
    f.write("\n## Restart recovery; Python 256, forwards 32/50\n"
            "- Restored gated8 is still the only current accepted checkpoint.\n"
            "- Recovering architecture, identification evidence, and helpers before further edits.\n"
            "- Rank tilt is not a physical-unit tilt; GPU means alone do not determine its factual calibration.\n")
```
