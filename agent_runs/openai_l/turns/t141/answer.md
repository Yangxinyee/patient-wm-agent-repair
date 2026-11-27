No newer run is reported, so I won’t assume the proposed edit has been applied. I’ll stage only the reversible attention reconstruction and retrieve valid tuning IDs. The next paired factual test must show negligible loss change before I add a causal correction.

```python
import numpy as np, pandas as pd

kv132 = np.asarray(load_original("core.prelude.0.attn.kv.weight"),
                   dtype=np.float32).copy()
kv132[192:] = np.asarray(Vnew125, dtype=np.float32)
out132 = np.asarray(Wrepair125, dtype=np.float32)
assert np.isfinite(kv132).all() and np.isfinite(out132).all()

save_edit("core.prelude.0.attn.kv.weight", kv132)
save_edit("core.prelude.0.attn.out.weight", out132)
print("Current edits:", edited())

cols132 = pd.read_csv("/data/tuning.csv", nrows=0).columns.tolist()
staycol132 = next(c for c in cols132
                 if "stay" in c.lower() and "hour" not in c.lower())
tuning_ids132 = pd.read_csv("/data/tuning.csv", usecols=[staycol132])[staycol132].drop_duplicates().tolist()
print("First 100 tuning stays:", tuning_ids132[:100])

with open("/work/notes.md", "a") as f:
    f.write("\n## Reconstruction-only attention edit staged\n"
            "Saved original keys plus Vnew125 and Wrepair125. "
            "No causal injection or value-head change. "
            "Next: paired factual audit; undo if reconstruction harms fit. "
            "CPU cache remains original.\n")
```
