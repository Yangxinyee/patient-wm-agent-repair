The last result resolves the scaling discrepancy: the extractor inputs are valid action vectors, and their factual-dose errors are smaller than previously reported. That supports testing a dose-sensitive correction, but does not yet establish that it will preserve factual fit. I’ll recover the architecture and retained helpers to identify a safe insertion point; inaccurate dose decoding or poor factual validation would refute this approach.

```python
from pathlib import Path
import inspect
import numpy as np

print("=== ARCHITECTURE ===")
print(Path("/model/ARCHITECTURE.md").read_text())

print("\n=== LIVE ARRAYS AND TABLES ===")
for key, value in list(globals().items()):
    if key.startswith("_"):
        continue
    if isinstance(value, np.ndarray) or (
        hasattr(value, "shape") and not callable(value)
        and type(value).__module__.split(".")[0] in ("torch", "pandas")
    ):
        print(key, type(value).__name__, tuple(value.shape))
    elif isinstance(value, dict):
        keys = list(value.keys())
        print(key, "dict", len(value), "keys:", [str(k)[:90] for k in keys[:8]])

print("\n=== LIVE HELPER SIGNATURES ===")
for key, value in list(globals().items()):
    if inspect.isfunction(value) and not key.startswith("_"):
        try:
            print(key, inspect.signature(value))
        except Exception:
            pass

print("\n=== CURRENT EDITS ===", edited())
with open("/work/notes.md", "a") as f:
    f.write(
        "\nPython run 49; forwards 3. Run 48 resolved the extractor scaling: "
        "X is the physical action vector, and earlier fitted coefficients require "
        "X/scale. On actual factual action bundles, active-dose RMSE is "
        ".321 insulin/.189 NE, with near-zero inactive presence. Recovering "
        "architecture and retained helpers before a bounded correction test. "
        "No weights edited.\n"
    )
```