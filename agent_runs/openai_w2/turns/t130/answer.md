The architecture confirms that the prelude FFN can affect the persistent state, unlike the nearly ignored posterior pathway. The output was truncated before showing the saved helpers. I’ll retrieve those compactly and check whether the dose signal can drive a useful state direction without disturbing recorded predictions.

```python
import re, inspect
import numpy as np

rows114 = []
for k in sorted(list(globals())):
    if any(s in k.lower() for s in [
        "112", "111", "110", "prelude", "repair", "recur", "step", "ffn", "probe"
    ]):
        v = globals()[k]
        if isinstance(v, np.ndarray):
            desc = f"array {v.shape} {v.dtype}"
        elif callable(v):
            try:
                desc = f"function {inspect.signature(v)}"
            except Exception:
                desc = "function"
        elif isinstance(v, dict):
            desc = "dict " + str(list(v)[:14])
        elif isinstance(v, (list, tuple)):
            desc = f"{type(v).__name__} len={len(v)}"
        else:
            desc = f"{type(v).__name__} {repr(v)[:100]}"
        rows114.append(f"{k}: {desc}")
print("=== SAVED HELPERS AND FEATURES ===")
print("\n".join(rows114)[-6500:])

print("\n=== PRELUDE/RECURRENT WEIGHTS ===")
for k in NAMES:
    if ("prelude" in k or "b_in" in k or "c_out" in k or
        "delta_in" in k or "a_in" in k or "out_norm" in k):
        print(k, SHAPES[k])

print("\n=== NOTEBOOK SUMMARY LINES ===")
nt114 = Path("/work/notes.md").read_text()
for line in nt114.splitlines():
    if (line.startswith("## Run") or line.startswith("## Forward") or
        any(s in line.lower() for s in [
            "kalman", "persistence", "ar(1)", "causal estimate", "true effect",
            "slope", "identified", "confound", "clean lab"
        ])):
        print(line[:220])
print("\nHistory available:", [k for k in ["_ih", "In", "history"] if k in globals()])
```
