The architecture specifies that the action vector is **zero when no action is present**. That may explain the observation-only mismatches. I’ll recover the helper and state-object names without printing tensor-valued defaults, then test that convention directly. If those mismatches disappear, it supports the diagnosis.

```python
import inspect
import numpy as np
import pandas as pd

snapshot75 = list(globals().items())

print("=== FUNCTIONS: PARAMETER NAMES ONLY ===")
for name, obj in sorted(snapshot75):
    if inspect.isfunction(obj) and obj.__module__ == "__main__":
        c = obj.__code__
        print(name, tuple(c.co_varnames[:c.co_argcount + c.co_kwonlyargcount]))

print("\n=== DATAFRAMES ===")
for name, obj in sorted(snapshot75):
    if isinstance(obj, pd.DataFrame) and len(obj) > 100:
        print(name, obj.shape, list(obj.columns)[:18])

print("\n=== NPZ STATE FILES ===")
for name, obj in sorted(snapshot75):
    if isinstance(obj, np.lib.npyio.NpzFile):
        print(name, {k: obj[k].shape for k in obj.files})

print("\n=== RELEVANT CONTAINERS / SOURCE STRINGS ===")
terms75 = ("state", "meta", "post", "recon", "context", "impulse",
           "effect", "causal", "transition", "source", "src")
for name, obj in sorted(snapshot75):
    if not any(t in name.lower() for t in terms75):
        continue
    if isinstance(obj, dict):
        print(name, "dict keys", list(obj)[:15])
    elif isinstance(obj, (np.ndarray, pd.Series)):
        print(name, type(obj).__name__, obj.shape)
    elif isinstance(obj, str) and len(obj) > 150:
        print(name, "string length", len(obj), "prefix", repr(obj[:90]))

print("\n=== POSTERIOR HELPER GLOBAL REFERENCES ===")
for name in ("latent_context", "latent_hidden", "latent_post_mean",
             "encode_grouped65", "complete_atom57", "head_exact"):
    obj = globals().get(name)
    if inspect.isfunction(obj):
        print(name, "file", obj.__code__.co_filename,
              "references", obj.__code__.co_names)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Python 75; forward 6 — empty-action convention\n"
        "- Architecture explicitly requires u=0 on bundles without actions. "
        "This is a candidate explanation for observation-only posterior errors.\n"
        "- Recovering helper names without printing large default arguments. "
        "No weights changed.\n"
    )
```