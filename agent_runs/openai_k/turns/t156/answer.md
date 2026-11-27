The records support drug-specific effects and persistence, but the current edit captures little of them. The likely issue is the channel routing, which relied on local readouts that failed validation. I’ll inspect the exact recipe so I can replace those directions using trustworthy state and prediction evidence.

```python
import inspect, re, torch
from pathlib import Path

notes = Path("/work/notes.md").read_text()
print("=== CHANNEL-CONSTRUCTION NOTES ===")
for section in re.split(r"(?=^## )", notes, flags=re.M):
    if any(f"Run {n}:" in section or f"Run {n} " in section
           for n in [66, 71, 82, 85, 86, 88, 89, 90]):
        print(section[:700])

print("\n=== EXACT CANDIDATE RECIPE ===")
for name, limit in [("build_candidate89", 3200), ("r132_channels", 1800)]:
    f = globals().get(name)
    if f is None:
        continue
    print("\nFUNCTION", name)
    try:
        print(inspect.getsource(f)[:limit])
    except Exception:
        print(compact_code95(f, limit=limit))

print("\n=== CANDIDATE TENSORS ===")
for name, x in candidates90[0.0].items():
    y = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    print(name, y.shape, "norm", float(np.linalg.norm(y)))
print("D88 candidates:", [(k, getattr(v, "shape", None))
                           for k, v in D88_candidates.items()])
print("target88:", target88)
print("\n=== CONSTRUCTION VARIABLES ===")
for name in sorted(list(globals())):
    if re.search(r"(66|71|82|85|88|89|90)$", name):
        v = globals()[name]
        if inspect.isfunction(v):
            continue
        detail = str(v.shape) if hasattr(v, "shape") else ""
        if isinstance(v, dict):
            detail = "keys=" + str(list(v)[:10])
        elif isinstance(v, (int, float, str)):
            detail = str(v)[:100]
        print(name, type(v).__name__, detail)

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 142: reconstruct channel recipe\n"
            "- Identification evidence supports insulin/glucose and NE/MAP effects with persistence.\n"
            "- Existing token directions were calibrated with unvalidated local value readouts.\n"
            "- Inspecting candidate construction to retain dose centering but replace response routing.\n"
            "- Python runs: 142; forward passes: 9; edits unchanged.\n")
```