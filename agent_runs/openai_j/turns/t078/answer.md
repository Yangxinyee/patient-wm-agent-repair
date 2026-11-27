The posterior edit preserved factual fit but barely changed dose responses, so it did not repair the model. I’ll recover the record-based effect estimates and inspect the remaining action-to-state pathways. A useful next candidate must change dose sensitivity, not merely decode recorded doses.

```python
from pathlib import Path
import pandas as pd
import numpy as np

print("ARCHITECTURE:")
print(Path("/model/ARCHITECTURE.md").read_text())

notes_text = Path("/work/notes.md").read_text()
print("\nEARLIER NOTEBOOK:")
print(notes_text[:18000])

for filename in [
    "dml_results.csv", "dynamic_iv.csv", "local_projections.csv",
    "proxy_iv.csv", "sensor_iv.csv", "kernel_targets.csv",
    "posterior_contrast_cv.csv"
]:
    path = Path("/work") / filename
    if path.exists():
        table = pd.read_csv(path)
        print(f"\n{filename}: shape={table.shape}, columns={list(table.columns)}")
        print(table.head(12).to_string(index=False))

print("\nKERNEL OBJECTS:")
for name, obj in list(globals().items()):
    if name.startswith("_") or name in {"notes_text", "table", "obj"}:
        continue
    if isinstance(obj, (pd.DataFrame, np.ndarray)):
        print(name, type(obj).__name__, obj.shape)
    elif callable(obj) and getattr(obj, "__module__", None) == "__main__":
        print("FUNCTION", name)

print("\nACTIVE EDITS:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Restart after run 65\n"
            "- Posterior probe was rejected and all tensors restored.\n"
            "- Recovering the architecture and record-derived causal estimates before "
            "testing a more direct action/state or readout correction.\n")
```