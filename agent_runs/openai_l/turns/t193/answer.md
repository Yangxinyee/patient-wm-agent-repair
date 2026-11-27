The architecture allows a narrow edit to the final value readout, leaving timing, counts, and action heads unchanged. The accepted targets come from record-based estimates, not the rejected state-space fit. I’ll locate the aligned factual and dose-edited features; the proposed edit is viable only if those features can separate dose effects from factual variation.
```python
import numpy as np
import pandas as pd
import inspect

print("FEATURE DICTIONARIES:")
for n176, o176 in sorted(list(globals().items())):
    if isinstance(o176, dict):
        if "hidden" in o176 and "rows" in o176:
            print(n176, {str(k): getattr(v, "shape", type(v).__name__)
                         for k, v in o176.items()})
        elif any(isinstance(v, dict) and "hidden" in v and "rows" in v
                 for v in list(o176.values())[:3]):
            print(n176, "nested feature dictionary", list(o176.keys()))

print("\nALIGNED TABLES:")
for n176, o176 in sorted(list(globals().items())):
    if isinstance(o176, pd.DataFrame):
        print(n176, o176.shape, list(o176.columns)[:16])

print("\n4828-ROW / MC ARRAYS:")
for n176, o176 in sorted(list(globals().items())):
    if hasattr(o176, "shape") and not isinstance(o176, pd.DataFrame):
        try:
            if 4828 in o176.shape and not n176.startswith(("o17", "obj", "v17")):
                print(n176, tuple(o176.shape))
        except Exception:
            pass

print("\nACCEPTED KERNELS:")
if "kernels19" in globals():
    print(repr(kernels19)[:2400])

print("\nHELPER INTERFACES:")
for f176 in (make_cf81, decode67, head_inputs96, contexts37, register67):
    print(f176.__name__, inspect.signature(f176))
    print("globals:", f176.__code__.co_names)
    print("locals:", f176.__code__.co_varnames)
    print("constants:", repr(f176.__code__.co_consts)[:1100])

print("\nVALUE BIN CENTRES:")
print(VOCAB["value_bin_centres"])
```