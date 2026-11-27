The kernel contains several reconstruction attempts, but the failed factual check means none should be trusted without validation. I’ll recover the causal estimates and inspect the cached state files. Agreement across independent sensors and identification checks would support a repair target; disagreement would require revisiting it.

```python
import numpy as np
import pandas as pd
from pathlib import Path

for filename in ("dynamic_iv.csv", "sensor_iv.csv", "proxy_iv.csv",
                 "kernel_targets.csv", "local_projections.csv"):
    table = pd.read_csv("/work/" + filename)
    print("\n" + filename, "shape:", table.shape)
    print(table.head(16).to_string(index=False))

print("\nFORWARD FILES")
for folder in sorted(Path("/work").glob("fwd_*")):
    print(folder.name, [p.name for p in folder.iterdir()])
    for path in folder.glob("states_*.npz"):
        with np.load(path) as archive:
            print(" ", path.name,
                  {key: archive[key].shape for key in archive.files})

print("\nHEAD METADATA")
print(head_meta.head(4).to_string(index=False))
print("head_hidden range/SD:", np.nanmin(head_hidden),
      np.nanmax(head_hidden), np.nanstd(head_hidden))
print("\nRECONSTRUCTION FUNCTION REFERENCES")
for name in ("reconstruct_reading_head", "score_head_variant",
             "score_view_local", "ln", "mlp", "wt"):
    obj = globals()[name]
    print(name, "references:", obj.__code__.co_names,
          "file:", obj.__code__.co_filename)
```
