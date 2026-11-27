The independently fitted kernels agree with the earlier estimates and are expressed per recorded rate unit. They support a multi-hour insulin response and a shorter norepinephrine response, conditional on the identification assumptions. I will now isolate the saved edit’s response terms from its structural compensation, so calibration need not scale—and damage—the whole repair.

```python
import inspect
import pandas as pd
import torch
import numpy as np

print("=== EXACT LATEST ROLLOUT FORMAT ===")
r242_roll = pd.read_csv("/work/fwd_23/dose_edit_edited.csv")
print(r242_roll.shape, list(r242_roll.columns))
print(r242_roll.head(2).to_string(index=False))

print("\n=== RECENT TARGET OBJECTS ===")
r242_g = dict(globals())
print(", ".join(sorted(n for n in r242_g if "238" in n and not n.startswith("r242"))))

print("\n=== REPLAY INTERFACES ===")
for name in ("rp_linear", "rp_mlp", "replay_factual125", "cpu_factual129",
             "factual_eval194", "transition_with_b110", "collect_selected187"):
    fn = r242_g.get(name)
    if callable(fn):
        print(name, inspect.signature(fn), "dependencies:", fn.__code__.co_names)

print("\n=== CHANNEL CONSTRUCTION OBJECTS ===")
for name in ("W0base94", "b0base94", "reserved94", "ins_direction187",
             "ne_direction194", "ne_directions206", "directions_ne187",
             "gate_fits180", "raw_base192", "test192", "active192"):
    v = r242_g.get(name)
    if isinstance(v, dict):
        print(name, "keys:", list(v))
        for k, a in list(v.items())[:5]:
            print(" ", k, type(a).__name__, getattr(a, "shape", None))
    elif isinstance(v, (list, tuple)):
        print(name, repr(v)[:200])
    else:
        print(name, type(v).__name__, getattr(v, "shape", None))

print("\n=== SAVED POSTERIOR EDIT ROWS ===")
for name in ("z_post.0.weight", "z_post.0.bias", "z_post.2.weight",
             "z_post.2.bias", "z_in.weight", "z_in.bias"):
    a = torch.as_tensor(load(name))
    b = torch.as_tensor(load_original(name))
    d = a - b
    changed_rows = torch.where(d.abs().reshape(d.shape[0], -1).max(dim=1).values > 1e-7)[0]
    print(name, tuple(a.shape), "changed rows:", changed_rows.tolist(),
          "difference norm:", round(float(d.norm()), 5))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 242\n"
            "The 12-lag kernels agree with earlier longitudinal estimates; causal "
            "interpretation remains conditional on sequential exogeneity and transferable "
            "additive dynamics. Inspecting exact replay and response/compensation separation "
            "before calibration. No weights changed.\n")
```