The inspection expanded a function’s default weight dictionary and obscured the useful output; it did not change any weights. I’ll print only parameter names and object shapes. This should expose the prepared contrast features without another oversized dump.

```python
import inspect, numpy as np, pandas as pd
snapshot = list(globals().items())
funcs = [(k, v) for k, v in snapshot
         if inspect.isfunction(v) and getattr(v, "__module__", "") == "__main__"]
print("FUNCTIONS")
for k, v in funcs:
    print(k, list(inspect.signature(v).parameters))
print("\nRECENT OBJECTS")
for k, v in snapshot[-105:]:
    if k.startswith("_"):
        continue
    if isinstance(v, np.ndarray):
        print(k, "array", v.shape, v.dtype)
    elif isinstance(v, pd.DataFrame):
        print(k, "dataframe", v.shape)
    elif isinstance(v, dict):
        print(k, "dict", list(v.keys())[:7])
    elif isinstance(v, (list, tuple)):
        print(k, type(v).__name__, "length", len(v))
    elif isinstance(v, (int, float, bool, np.number)):
        print(k, v)
    elif isinstance(v, str):
        print(k, repr(v[:100]))
```
