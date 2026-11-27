Removing the latent contribution slightly improved factual NLL on these 100 training stays, from 1.63841 to 1.6374. That does not establish a useful repair, but it gives us a deterministic diagnostic. I’ll now check row alignment and register construction against these predictions. Numerical agreement—not merely a low average error—is the confirmation criterion.

```python
import dis
import inspect
import io
import numpy as np
import pandas as pd
from pathlib import Path

probe84_df = pd.read_csv("/work/fwd_8/factual_edited.csv")
probe84_keys = ["stay_id", "hour", "event"]
assert probe84_df[probe84_keys].equals(probe83_factual[probe84_keys])

print("=== ROW ALIGNMENT ===")
print("head_meta columns:", list(head_meta.columns))
print(head_meta.head(6).to_string(index=False))
check84 = probe84_df.head(10).copy()
check84["original_mean"] = probe83_factual["predicted_mean"].iloc[:10].values
check84["cached_local_mean"] = np.asarray(rep_means68)[:10]
print(check84.to_string(index=False))

print("\n=== VOCAB STRUCTURE ===")
for key in ("codes", "config", "numeric", "value_bin_centres"):
    value = VOCAB[key]
    print(key, type(value).__name__, str(value)[:1500])

print("\n=== REGISTER HELPERS ===")
for name in ("rh_register", "rh_eval"):
    function = globals()[name]
    print(name, inspect.signature(function))
    try:
        print(inspect.getsource(function)[:3500])
    except Exception:
        buffer84 = io.StringIO()
        dis.dis(function, file=buffer84, depth=0)
        print(buffer84.getvalue()[:3500])

print("\n=== CENTRE / ALIGNMENT ARRAYS ===")
for name, value in sorted(list(globals().items())):
    shape = getattr(value, "shape", None)
    if (shape is not None and
        (("cent" in name.lower()) or
         (tuple(shape) == (4510,) and not name.startswith("rep_")))):
        print(name, type(value).__name__, tuple(shape))

print("\n=== REGISTER CACHE KEYS ===")
for name in ("rep_registers67", "rep_regs67"):
    value = globals().get(name)
    if isinstance(value, dict):
        print(name, {
            str(k): (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__)
            for k, v in value.items()
        })

with Path("/work/notes.md").open("a") as f:
    f.write(
        "\n## Forward 8 / Run 84\n"
        "With only z_head.weight zeroed, train factual NLL is 1.63740 "
        "(original 1.63841 on the same 100 stays). Saved deterministic predictions "
        "in fwd_8 for exact head reconstruction. This diagnostic remains temporary; "
        "no causal repair has been validated.\n"
    )
```