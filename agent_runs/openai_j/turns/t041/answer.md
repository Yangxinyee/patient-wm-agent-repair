The architecture has separate action embeddings and value heads, so a targeted repair is possible in principle. The output was truncated before the rollout summary and relevant helpers appeared. I’ll inspect just those pieces next; a dose-sensitive feature with little factual variation would support a low-rank repair, while strong factual dependence would argue against it.

```python
import inspect
from pathlib import Path
import pandas as pd

print("=== RELEVANT TENSORS ===")
for name in NAMES:
    if any(s in name for s in [
        "embed.numeric", "encode_action", "last_numeric", "attr_heads.value",
        "query_norm", "type_emb", "z_head", "z_prior", "core.gap",
        "filter_", "embed_transition.value"
    ]):
        print(name, SHAPES[name])

print("\n=== AVAILABLE FUNCTION NAMES ===")
funcs = [(n, o) for n, o in list(globals().items())
         if not n.startswith("_") and inspect.isfunction(o)]
for n, o in funcs:
    if n not in ("load", "load_original", "save_edit", "drop_edit", "edited"):
        try:
            print(n, inspect.signature(o))
        except Exception:
            print(n)

print("\n=== FORWARD FILES AND BASELINE EFFECTS ===")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
    for p in folder.glob("dose_edit_original.csv"):
        df0 = pd.read_csv(p)
        ok = df0["reached"].astype(str).str.lower().eq("true")
        print(df0.loc[ok].groupby("edit")["shift"].agg(["mean", "count"]).to_string())

print("\n=== RECONSTRUCTION-RELATED ARRAYS ===")
for n, o in list(globals().items()):
    if any(s in n.lower() for s in
           ["state", "head", "query", "context", "ctx", "match", "with_code", "bin", "hist", "kernel"]):
        if hasattr(o, "shape") and not inspect.ismodule(o):
            print(n, type(o).__name__, o.shape)
        elif isinstance(o, dict):
            print(n, "keys:", list(o.keys())[:15])

print("\n=== EARLIEST NOTEBOOK ===")
print(Path("/work/notes.md").read_text()[:4200])
```
