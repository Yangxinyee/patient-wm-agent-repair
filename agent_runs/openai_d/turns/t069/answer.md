The failed edit was highly sensitive to individual latent draws, so its averaged-state checks did not justify installing it. All weights are back to the original. I’ll recover the architecture and earlier causal estimates before designing another edit; it must work per draw and pass the actual factual-fit test.

```python
from pathlib import Path
import json, numpy as np, pandas as pd, torch
torch.set_num_threads(8)

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())
print("=== RECORD FORMAT ===")
print(Path("/data/README.md").read_text())

notes61 = Path("/work/notes.md").read_text()
print("=== EARLY NOTEBOOK ===")
print(notes61[:16000])

print("=== PERSISTENT OBJECT INVENTORY ===")
for key61, obj61 in sorted(list(globals().items())):
    if key61.startswith("_") or key61 in {"obj61", "key61", "notes61"}:
        continue
    if isinstance(obj61, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        print(key61, type(obj61).__name__, tuple(obj61.shape))
    elif isinstance(obj61, dict):
        print(key61, "dict", len(obj61), "keys:", list(obj61)[:12])
    elif callable(obj61) and getattr(obj61, "__module__", "") == "__main__":
        print(key61, "function")
print("=== CURRENT EDITS ===", edited())

with open("/work/notes.md", "a") as f61:
    f61.write(
        "\n## Restart / Run 61\n"
        "No edits. Run 60 showed severe individual-latent sensitivity in the rejected "
        "head construction, with unreliable local reconstruction. Recovering the "
        "architecture, causal estimates, and persistent helpers before another edit.\n"
    )
```