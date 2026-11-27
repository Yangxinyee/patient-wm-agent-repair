```python
from pathlib import Path
import json, types

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== DATA DICTIONARY ===")
print(Path("/data/README.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\n=== EARLIER NOTEBOOK (excluding the already supplied tail) ===")
print(notes_text[:-8000] if len(notes_text) > 8000 else notes_text)

print("\n=== PERSISTENT KERNEL VARIABLES ===")
for key, value in sorted(list(globals().items())):
    if key.startswith("_") or isinstance(value, types.ModuleType):
        continue
    if key in {"notes_text", "value"}:
        continue
    shape = getattr(value, "shape", None)
    if shape is not None:
        detail = f"shape={shape}"
    elif isinstance(value, dict):
        detail = f"len={len(value)}, keys={list(value)[:14]}"
    elif isinstance(value, (list, tuple, set)):
        detail = f"len={len(value)}"
    elif callable(value):
        detail = "callable"
    elif isinstance(value, (str, int, float, bool)):
        detail = repr(value)[:160]
    else:
        detail = ""
    print(f"{key}: {type(value).__name__} {detail}")

print("\n=== CURRENT EDIT MAGNITUDES ===")
for name in edited():
    a = load(name)
    b = load_original(name)
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
        b = b.detach().cpu().numpy()
    print(name, "shape", a.shape,
          "max_abs_delta", float(np.max(np.abs(a - b))),
          "rms_delta", float(np.sqrt(np.mean((a - b) ** 2))))

with open("/work/notes.md", "a") as f:
    f.write("\n## Python run 17 / forward passes 7: restart audit\n"
            "- Recovered architecture, data dictionary, earlier notebook, kernel variables, and current edits.\n"
            "- NE pooled curvature is counter-evidence to exact zero, but is small and weak in hours 12–23.\n"
            "- No weight changes in this run; inspect earlier rollout and factual validation before deciding whether refinement is justified.\n")
```