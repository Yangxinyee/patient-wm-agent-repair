The inventory failed because it encountered an object whose `shape` attribute was not an array shape. No weights changed, and that failure does not invalidate the earlier decoder test. I’ll restrict inspection to actual arrays and recover the cached inputs and metadata needed for an exact-dose probe.

```python
import torch
import pandas as pd
import inspect

print("=== NOTEBOOK HEADINGS ===")
print("\n".join(line for line in notes191.splitlines()
                if line.startswith("##"))[:2600])
print("=== PERSISTENCE NOTES ===")
print(notes191[6500:9700])

print("=== CACHED INPUT / METADATA ARRAYS ===")
arr195 = []
for n195, o195 in list(globals().items()):
    if isinstance(o195, (np.ndarray, torch.Tensor, pd.DataFrame, pd.Series)):
        sh195 = tuple(o195.shape)
        if 4510 in sh195 and (192 in sh195 or len(sh195) == 1):
            arr195.append((n195, type(o195).__name__, sh195))
print(sorted(arr195)[:55])

print("=== META96 ===")
if "meta96" in globals():
    print(type(meta96), getattr(meta96, "shape", None))
    print(meta96.head(3).to_string() if hasattr(meta96, "head")
          else repr(meta96)[:900])

print("=== CONTEXT / ACTION HELPERS ===")
for n195 in ("r124_context", "act96", "r124_lin", "r138_nll", "r144_nll"):
    o195 = globals().get(n195)
    if inspect.isfunction(o195):
        print(n195, inspect.signature(o195),
              "\nREFERENCES", o195.__code__.co_names,
              "\nLOCALS", o195.__code__.co_varnames,
              "\nCONSTANTS", repr(o195.__code__.co_consts)[:450])

print("=== STATE / PROBE OBJECT NAMES ===")
print(sorted(n for n in globals() if re.search(
    r"(state|probe|atom|bundle|context|query|register)", n,
    flags=re.I))[:110])
```