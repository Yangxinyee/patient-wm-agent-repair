The architecture supports a targeted edit: remove dose magnitude while retaining drug identity, presence, and timing. That should eliminate dose-edit shifts without discarding treatment context. The output was truncated, so I’ll recover the edit details and validation results. Near-zero effects with a factual NLL increase below 0.005 would support keeping it; a stable nonzero response would justify refinement.

```python
from pathlib import Path
import types
import numpy as np

text = Path("/work/notes.md").read_text()
start = text.find("## Run 10")
end = max(start, len(text) - 8000)
print("=== EDIT CONSTRUCTION AND VALIDATION ===")
print(text[start:end][:4200])

print("\n=== WORKING OBJECTS ===")
lines = []
for key, obj in sorted(list(globals().items())):
    if key.startswith("_") or isinstance(obj, types.ModuleType):
        continue
    if key in {"text", "notes_text", "lines", "obj", "value"}:
        continue
    if hasattr(obj, "shape"):
        lines.append(f"{key}: {type(obj).__name__}, shape={obj.shape}")
    elif isinstance(obj, dict):
        lines.append(f"{key}: dict, keys={list(obj)[:8]}")
    elif callable(obj) and hasattr(obj, "__code__"):
        n = obj.__code__.co_argcount
        lines.append(f"{key}: function, args={obj.__code__.co_varnames[:n]}")
    elif isinstance(obj, (list, tuple, set)):
        lines.append(f"{key}: {type(obj).__name__}, len={len(obj)}")
print("\n".join(lines)[:4800])

print("\n=== FORWARD RESULT FILES ===")
for p in sorted(Path("/work").glob("fwd_*")):
    print(p.name, [q.name for q in p.iterdir()])

print("\n=== DOSE-INVARIANCE CHECK ===")
def to_numpy(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

for name in ("embed.bins.dose.weight", "embed.bins.rate.weight"):
    a = to_numpy(load(name))
    b = to_numpy(load_original(name))
    print(name,
          "absent_row_unchanged", bool(np.array_equal(a[0], b[0])),
          "max_positive_row_difference", float(np.max(np.abs(a[1:] - a[1]))))
a = to_numpy(load("embed.numeric.weight"))
b = to_numpy(load_original("embed.numeric.weight"))
print("numeric weight shape:", a.shape)
print("per-column max magnitude:", np.max(np.abs(a), axis=0).tolist())
print("per-column max edit:", np.max(np.abs(a - b), axis=0).tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 18 / forward passes 7: compact recovery\n"
            "- Checked the exact dose-invariant weight routes and recovered prior validation and working-object names.\n"
            "- No weight changes. Need broader factual validation and a final assessment of the small NE curvature signal.\n")
```