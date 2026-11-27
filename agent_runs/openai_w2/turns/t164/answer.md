The architecture confirms that the current edit changes how dose information enters the recurrent state. The notes also show that dose thresholds provide evidence for insulin’s effect, but threshold-related changes in measurement frequency need care. The output was truncated before the factual diagnostics, so I’ll retrieve those next. A mostly insulin-specific NLL cost would support testing stronger norepinephrine steering separately.

```python
from pathlib import Path
import re, numpy as np, pandas as pd, torch

print("=== CURRENT EDIT AND COEFFICIENTS ===", edited())
for k143 in ["Wdose134", "V128", "fallback_recurrent139", "fit128",
             "repair_head_cfg", "raw_coef134", "affine_b134"]:
    if k143 in globals():
        v143 = globals()[k143]
        print(k143, type(v143).__name__, getattr(v143, "shape", None))
        if isinstance(v143, dict):
            print({k: (str(v)[:1800] if k in ["report", "family", "probe_weight"]
                       else getattr(v, "shape", type(v).__name__))
                   for k, v in v143.items()})

print("=== FACTUAL DIAGNOSTICS ===")
dfs143 = {}
for label143, f143 in [
    ("original", "/work/fwd_13/factual_original.csv"),
    ("fallback", "/work/fwd_13/factual_edited.csv"),
    ("stronger", "/work/fwd_15/factual_edited.csv")]:
    if Path(f143).exists():
        q143 = pd.read_csv(f143)
        dfs143[label143] = q143
        print(label143, q143.shape, q143.columns.tolist())
        print(q143.head(2).to_string(index=False))
        ncols143 = [c for c in q143 if "nll" in c.lower()]
        ecol143 = next((c for c in ["event", "code", "event_code", "event_name"]
                       if c in q143), None)
        if ncols143:
            print("NLL means:", q143[ncols143].mean().to_dict())
            if ecol143:
                print(q143.groupby(ecol143)[ncols143].agg(["mean", "size"]).to_string())

print("=== RESEARCH OBJECT NAMES ===")
print([k for k in globals() if any(s in k.lower() for s in
      ["panel", "target", "causal", "effect", "tuning", "dml", "latent_record", "iv_result"])])

print("=== RECORD-EFFECT NOTE SECTIONS ===")
sections143 = re.split(r"(?=^## )", Path("/work/notes.md").read_text(), flags=re.M)
selected143 = [s for s in sections143
               if any(w in s.lower() for w in ["persistence", "latent record", "selection bias",
                                               "accumulation", "measurement selection"])]
for s143 in selected143[:9]:
    print(s143[:1100])
```
