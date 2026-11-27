The reconstruction left factual NLL unchanged to reported precision on 4,751 readings. That supports using the freed channels, but says nothing yet about causal accuracy. I’ll recover the architecture and fitted components before adding an effect correction; the next decisive test will require both improved dose responses and preserved factual fit.

```python
from pathlib import Path
import inspect, re, numpy as np, torch

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== NOTEBOOK: EARLY EVIDENCE ===")
notes_recovery = Path("/work/notes.md").read_text()
print(notes_recovery[:16000])

print("\n=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes_recovery.splitlines() if line.startswith("#")))

print("\n=== KERNEL FUNCTIONS ===")
for key, obj in list(globals().items()):
    if inspect.isfunction(obj) and not key.startswith("_"):
        try:
            print(key, str(inspect.signature(obj)))
        except Exception:
            print(key)

print("\n=== RECENT FITTED COMPONENTS ===")
for key, obj in list(globals().items()):
    if re.search(r"(118|119|12[0-9]|130|131)$", key):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            print(key, tuple(obj.shape), str(obj.dtype))
        elif isinstance(obj, dict):
            print(key, "dict keys:", list(obj)[:30])
        elif isinstance(obj, (list, tuple)):
            print(key, type(obj).__name__, "length", len(obj), "sample", str(obj[:3])[:500])
        else:
            print(key, type(obj).__name__, str(obj)[:500])

print("\n=== EXACT FACTUAL RECONSTRUCTION AUDIT ===")
import pandas as pd
f6o = pd.read_csv("/work/fwd_6/factual_original.csv")
f6e = pd.read_csv("/work/fwd_6/factual_edited.csv")
print("columns:", f6o.columns.tolist())
for col in f6o.select_dtypes(include=[np.number]).columns:
    delta = f6e[col].to_numpy() - f6o[col].to_numpy()
    if np.any(delta):
        print(col, "mean delta", float(delta.mean()), "max abs", float(np.abs(delta).max()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 133 / reconstruction factual audit\n"
            "Forward 5 (files fwd_6): 100 tuning stays, 4,751 values. "
            "Original and reconstruction-only edited NLL both 1.63164; "
            "all reported event NLL/MAE match. Reconstruction accepted provisionally. "
            "Recovering fitted controller/wire interfaces before causal injection. "
            "Forward passes 5/50; only attention kv/out tensors edited.\n")
```