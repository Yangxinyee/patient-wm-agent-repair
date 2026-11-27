The architecture has a separate action encoder, but its effects pass through shared state and prediction heads. That suggests a targeted edit may be possible, though changing dose inputs alone could harm factual fit. I’ll inspect the exported states and exact layer shapes next; reproducible treatment-sensitive features would support a low-rank correction.

```python
import numpy as np, pandas as pd, json, inspect
from pathlib import Path

print("=== CONFIG AND EVENT TYPES ===")
for k in ["config", "event_kinds", "atom_kinds", "value_bin_centres"]:
    print(k, json.dumps(VOCAB[k]))

print("\n=== CANDIDATE PATHWAY TENSORS ===")
for n in NAMES:
    if n.startswith(("encode_action.", "encode.", "core.prelude.",
                     "core.prelude_norm.", "z_post.", "embed_transition.",
                     "attr_heads.value_bin.", "last_numeric.")):
        print(n, SHAPES[n])

print("\n=== RETAINED OBJECT INVENTORY ===")
inventory21 = []
functions21 = []
for k, v in list(globals().items()):
    if k.startswith("_") or k in {"NAMES", "SHAPES", "VOCAB", "train", "tuning"}:
        continue
    if callable(v) and getattr(v, "__module__", "") == "__main__":
        try:
            functions21.append(f"{k}{inspect.signature(v)}")
        except Exception:
            functions21.append(k)
    elif isinstance(v, np.ndarray):
        inventory21.append(f"{k}: ndarray {v.shape}")
    elif isinstance(v, dict):
        inventory21.append(f"{k}: dict {list(v.keys())[:6]}")
    elif isinstance(v, pd.DataFrame):
        inventory21.append(f"{k}: DataFrame {v.shape}")
    elif isinstance(v, (list, tuple)) and len(v) > 5:
        inventory21.append(f"{k}: {type(v).__name__} len={len(v)}")
print("\n".join(inventory21)[:3500])
print("Functions:", "; ".join(functions21)[:1800])

print("\n=== STATES EXPORT ===")
for p in Path("/work/fwd_3").glob("*.npz"):
    states21 = np.load(p, allow_pickle=False)
    print(p.name)
    for k in states21.files:
        a21 = states21[k]
        print(k, a21.shape, a21.dtype,
              a21[:12].tolist() if a21.ndim == 1 else "")
print("Summary:", Path("/work/fwd_3/summary.json").read_text()[:1200])

print("\n=== ORIGINAL FORWARD CSV SCHEMAS ===")
for p in [Path("/work/fwd_1/dose_edit_original.csv"),
          Path("/work/fwd_2/factual_original.csv")]:
    d21 = pd.read_csv(p)
    print(p.name, d21.shape, list(d21.columns))
    print(d21.head(5).to_string(index=False))

print("\n=== ACTION/READING GRID ===")
for label21, obj21 in [("A", A), ("P", P)]:
    for k, v in obj21.items():
        print(label21, k, type(v).__name__, v.shape,
              list(v.columns)[:5] if isinstance(v, pd.DataFrame) else "")
        if label21 == "A":
            arr21 = np.asarray(v)
            print("positive-dose quantiles:",
                  np.quantile(arr21[np.isfinite(arr21) & (arr21 > 0)],
                              [0, .25, .5, .75, 1]).round(4).tolist())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 21\n"
            "Architecture: actions have a separate DeepSets encoder but enter shared posterior/state pathways. "
            "Values are a shared 192-hidden-unit bin head plus code/last-bin transition logits. "
            "Inspecting state-export schema and exact tensors for a low-rank, factual-preserving intervention correction. "
            "No edits.\n")
```
