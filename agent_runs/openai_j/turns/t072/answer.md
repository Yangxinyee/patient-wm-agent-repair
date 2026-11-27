The record-based checks support a glucose-lowering insulin effect and a MAP-raising norepinephrine effect, but insulin’s persistence remains uncertain. Broad input-gain edits already failed. I’ll inspect the posterior-contrast machinery for a more selective correction; it should change dose sensitivity without creating large factual-state shifts.

```python
import inspect
import numpy as np
import pandas as pd

items = list(globals().items())
arrays = [(k, tuple(v.shape)) for k, v in items
          if isinstance(v, (np.ndarray, pd.DataFrame, pd.Series))]
print("RECENT ARRAYS:", arrays[-70:])
print("\nDICTIONARIES:")
for k, v in items:
    if isinstance(v, dict) and k not in {"__builtins__", "SHAPES", "VOCAB"}:
        print(k, str(list(v.keys())[:12])[:260])

for fname in ["repair_atom_vectors", "repair_set_encode", "fit_post_contrast"]:
    print("\nFUNCTION:", fname)
    try:
        print(inspect.getsource(globals()[fname])[:2100])
    except Exception as exc:
        print(type(exc).__name__, str(exc)[:160])

print("\nSELECTED TENSORS:")
for n in NAMES:
    if any(s in n for s in ["z_post", "z_in", "z_head", "value_bin", "prelude"]):
        print(n, SHAPES[n])
```
