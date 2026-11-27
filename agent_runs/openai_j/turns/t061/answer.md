Both simple input-gain probes are now rejected, with the original weights restored. The posterior pathway can use both history and the current action, making it a better place to represent unexpected dosing. I’ll recover its embedding conventions and align action bundles with dose innovations; accurate reconstruction and held-out decoding are the next checks.

```python
import numpy as np
import pandas as pd
import torch

print("=== EMBEDDING TABLES ===")
print([(n, SHAPES[n]) for n in NAMES if n.startswith("embed.")])
print("Event kinds:", VOCAB["event_kinds"])
print("Atom kinds:", VOCAB["atom_kinds"])
print("Numeric keys:", list(VOCAB["numeric"]))
print("Numeric excerpt:", str(VOCAB["numeric"])[:3300])

print("\n=== CONVENTION VARIABLE NAMES ===")
conv_keys = [k for k in globals()
             if any(s in k.lower() for s in
                    ["cat_", "categor", "norm", "empty", "inn", "best", "unit"])]
print(conv_keys)
for k in conv_keys:
    v = globals()[k]
    if isinstance(v, (list, tuple, str, int, float)) and len(str(v)) < 500:
        print(k, v)

print("\n=== ACTION/INNOVATION ALIGNMENT ===")
print(raw_probe.head(6).to_string(index=False))
print(residual_meta[["stay_id", "hour", "event", "insulin_innov",
                     "ne_innov", "insulin_previous", "ne_previous"]]
      .head(8).to_string(index=False))
print(state_frame.head(5).to_string(index=False))
print("Bundle flag counts:",
      state_frame.groupby(["has_insulin", "has_norepinephrine", "n_atoms"])
      .size().to_dict())

print("\n=== HELPER DETAILS ===")
for helper_name in ["activation", "ln", "latent_context_at", "encode_single_debug"]:
    fun = globals()[helper_name]
    print(helper_name, "locals", fun.__code__.co_varnames,
          "constants", tuple(x for x in fun.__code__.co_consts
                             if isinstance(x, (str, int, float, tuple))))
print("\nNo active edits:", edited())
```
