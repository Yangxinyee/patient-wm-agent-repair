The architecture confirms that changing the recurrent drug pathway could disturb every prediction head, whereas the current value-head edit is more contained. The output was truncated before the live inventory, so I’ll retrieve that and the causal-validation sections separately. This will tell us which existing estimates and candidates are safe to reuse.

```python
import pandas as pd
import inspect

snapshot166 = list(globals().items())
arr_inventory166 = [
    f"{n}:{tuple(o.shape)}"
    for n, o in snapshot166
    if not n.startswith("_") and isinstance(o, (np.ndarray, pd.DataFrame, pd.Series))
]
dict_inventory166 = [
    f"{n}:{len(o)}:{str(list(o.keys())[:5])[:110]}"
    for n, o in snapshot166
    if not n.startswith("_") and isinstance(o, dict) and n != "__builtins__"
]
func_inventory166 = [
    n for n, o in snapshot166
    if not n.startswith("_") and inspect.isfunction(o)
]
print("ARRAYS / DATAFRAMES:\n", "\n".join(arr_inventory166)[-5800:])
print("\nDICTIONARIES:\n", "\n".join(dict_inventory166)[-1700:])
print("\nFUNCTIONS:\n", ", ".join(func_inventory166)[-1800:])
print("\nEARLY NOTEBOOK HEADINGS:")
headings166 = [s for s in notes_text.splitlines() if s.startswith("##")]
print("\n".join(headings166[:38])[:1800])
print("\nEDIT SHAPES:", {n: SHAPES[n] for n in edited()})
```
