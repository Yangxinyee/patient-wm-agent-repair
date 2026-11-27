The saved objects include causal estimates and several local reconstructions, but the long inventory again hid their results. No weights changed. I’ll request only the effect estimates, original rollout shifts, and latest candidate variables. This will show whether the proposed correction has a defensible target and whether it addresses a measured model error.

```python
import numpy as np
import pandas as pd
import types

print("=== CAUSAL ESTIMATES ===")
for key in ("dyn_estimates", "dynamic_alternatives", "sensor_checks",
            "target_checks_A", "head_layout_spec"):
    value = globals().get(key)
    print(key, repr(value)[:2200])

print("\n=== PANEL ===")
print(panel.columns.tolist())
print(panel.head(3).to_string(index=False))

print("\n=== ORIGINAL FORWARDS ===")
for key in ("fA", "dA"):
    df = globals().get(key)
    if isinstance(df, pd.DataFrame):
        print(key, df.shape, df.columns.tolist())
        print(df.head(3).to_string(index=False))
        for column in ("edit", "intervention", "event", "drug"):
            if column in df.columns:
                numeric = [c for c in df.select_dtypes(include="number").columns
                           if c not in ("stay_id", "hour")]
                print(df.groupby(column)[numeric].mean().to_string())
                break

print("\n=== MOST RECENT OBJECTS ===")
recent = [(k, v) for k, v in list(globals().items())
          if not k.startswith("_") and k not in
          ("items", "fn_items", "source_keys", "key", "value", "df", "column", "numeric")]
for k, v in recent[-65:]:
    if isinstance(v, types.FunctionType):
        print("FUNCTION", k, v.__code__.co_varnames[:v.__code__.co_argcount])
    elif hasattr(v, "shape") and not callable(v.shape):
        print(k, type(v).__name__, tuple(v.shape))
    elif isinstance(v, dict):
        print(k, "dict", len(v), list(v)[:5])
    elif isinstance(v, (str, int, float, bool)):
        print(k, repr(v)[:120])
    elif isinstance(v, list):
        print(k, "list", len(v))

print("\n=== SOURCE ACCESS CHECK ===")
import inspect
for key in ("local_step67", "response_pair68", "latent_step66",
            "wm_attention", "action_atom_exact", "context_from_saved"):
    func = globals().get(key)
    if func is not None:
        try:
            text = inspect.getsource(func)
            print(key, "source available", len(text), func.__code__.co_filename)
        except (OSError, TypeError):
            print(key, "source unavailable", func.__code__.co_filename,
                  "references:", func.__code__.co_names)

print("Current edits:", edited())
```