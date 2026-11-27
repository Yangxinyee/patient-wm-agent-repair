The architecture confirms that a correction can be confined to the value head, leaving the recurrent dynamics unchanged. No weights have changed. I’ll inspect the recovered features, causal estimates, and original rollout results so the correction targets the model’s error rather than adding the drug effect twice.

```python
import inspect, numpy as np, pandas as pd
from pathlib import Path

print("=== RECOVERED HELPERS ===")
for k, v in sorted(list(globals().items())):
    if callable(v) and getattr(v, "__module__", None) == "__main__":
        try:
            print(k, inspect.signature(v))
        except Exception:
            print(k)

print("=== RECENT OBJECTS ===")
for k, v in sorted(list(globals().items())):
    if k.endswith(("97", "98", "99")):
        if isinstance(v, dict):
            desc = {str(a): (type(b).__name__, getattr(b, "shape", None),
                             list(b.keys())[:12] if isinstance(b, dict) else "")
                    for a, b in v.items()}
        elif hasattr(v, "shape"):
            desc = v.shape
        elif isinstance(v, (list, tuple)):
            desc = (len(v), repr(v[:2])[:250])
        else:
            desc = repr(v)[:160]
        print(k, type(v).__name__, desc)

print("=== HEAD METADATA ===")
print(head_meta.columns.tolist())
print(head_meta.head(4).to_string(index=False))
print("=== CAUSAL RESULT STRUCTURE ===")
for drug, obj in dml_results.items():
    print(drug, repr(obj)[:1600])
for drug, obj in dml_cache.items():
    print("cache", drug, {k: (type(v).__name__, getattr(v, "shape", None))
                        for k, v in obj.items()})
for drug, obj in record_targets.items():
    print("record_targets", drug, repr(obj)[:1200])

print("=== VALUE HEAD TENSORS ===")
for name in NAMES:
    if any(s in name for s in ("attr_heads.value_bin", "z_head", "last_numeric")):
        print(name, SHAPES[name])

print("=== ORIGINAL FORWARD SUMMARIES ===")
for d in sorted(Path("/work").glob("fwd_*")):
    for p in d.iterdir():
        if p.suffix == ".json":
            print(str(p), p.read_text()[:1800])
        elif p.name.startswith("dose_edit") and p.suffix == ".csv":
            df = pd.read_csv(p)
            print(str(p), df.shape, df.columns.tolist())
            print(df.head(5).to_string(index=False))

print("=== NOTEBOOK CAUSAL / BASELINE LINES ===")
for line in Path("/work/notes.md").read_text().splitlines():
    if any(s in line.lower() for s in
           ("effect", "rollout", "forward 1", "forward 2", "forward 3",
            "nll=", "nll:", "nll ", "causal", "confidence")):
        print(line[:700])
```
