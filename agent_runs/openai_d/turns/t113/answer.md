The output was truncated by a helper’s large default arguments, so it added no usable validation evidence. I’ll retrieve compact summaries only. The next step depends on confirming the decoder’s feature layout and the original dose responses; otherwise installing it would still be premature.

```python
import numpy as np, pandas as pd
from pathlib import Path

def summary102(x, depth=0):
    if hasattr(x, "shape"):
        return f"{type(x).__name__}{tuple(x.shape)}"
    if isinstance(x, dict):
        if depth >= 2:
            return "keys=" + repr(list(x.keys())[:15])
        return {str(k): summary102(v, depth+1) for k, v in list(x.items())[:18]}
    if isinstance(x, (list, tuple)):
        if len(x) <= 8 and all(np.isscalar(v) for v in x):
            return repr(x)
        return f"{type(x).__name__}[{len(x)}]"
    return repr(x)[:140]

print("HELPER NAMES:", [
    k for k, v in sorted(list(globals().items()))
    if callable(v) and getattr(v, "__module__", None) == "__main__"
])

print("\nRECENT OBJECTS:")
for k, v in sorted(list(globals().items())):
    if k.endswith(("97", "98", "99")):
        print(k, summary102(v))

print("\nHEAD METADATA:", head_meta.columns.tolist())
print(head_meta.head(2).to_string(index=False))
for nm in ("dml_results", "dml_cache", "record_targets"):
    print("\n", nm, summary102(globals()[nm]))

print("\nHEAD SHAPES:")
for name in NAMES:
    if any(s in name for s in ("attr_heads.value_bin", "z_head", "last_numeric")):
        print(name, SHAPES[name])

print("\nFORWARD FILES:")
for d in sorted(Path("/work").glob("fwd_*")):
    print(d.name, [p.name for p in d.iterdir()])
    for p in d.glob("dose_edit_original.csv"):
        df102 = pd.read_csv(p)
        print(df102.columns.tolist())
        print(df102.head(4).to_string(index=False))

print("\nNOTEBOOK EVIDENCE:")
lines102 = []
for line in Path("/work/notes.md").read_text().splitlines():
    if any(s in line.lower() for s in
           ("immediate", "target", "baseline", "dose_edit", "controlled",
            "nll=", "causal", "bootstrap")):
        lines102.append(line[:450])
print("\n".join(lines102)[:3600])
```
