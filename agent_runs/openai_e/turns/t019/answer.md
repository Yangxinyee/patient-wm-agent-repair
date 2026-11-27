The architecture allows a posterior-only edit, but the relevant latent-layer names were missing from the filtered output. I’ll retrieve those names and the existing helper signatures, keeping the output compact. Before editing anything, I want a local reconstruction that matches the saved posterior states; a poor match would invalidate the proposed dose-residual repair.

```python
import inspect
import numpy as np
import torch

print("=== LATENT, TIME, AND INITIALIZATION TENSORS ===")
for name in NAMES:
    if any(s in name for s in ("z_post", "z_prior", "z_in", "z_head", "gap", "slots", "init", "age")):
        print(name, SHAPES[name])

print("\n=== RETAINED HELPERS ===")
for key in ("decoder_query", "head_calc", "activation0", "dose_covariates"):
    fn = globals().get(key)
    if callable(fn):
        print(key, inspect.signature(fn))
        print("references:", fn.__code__.co_names)
        print("constants:", repr(fn.__code__.co_consts)[:550])

print("\n=== RETAINED DATAFRAMES ===")
for key in sorted(list(globals())):
    obj = globals()[key]
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        print(key, obj.shape, list(obj.columns)[:18])
print("\nOther object names:",
      ", ".join(k for k in sorted(globals()) if not k.startswith("_"))[-2800:])

print("\n=== POLICY MODELS ===")
for key, obj in dose_policy.items():
    print(key, type(obj).__name__)
    if isinstance(obj, dict):
        print({k: (type(v).__name__, getattr(v, "shape", None)) for k, v in obj.items()})
print("dose-policy covariate count:", len(feature_names))
print("dose-policy covariates:", feature_names)

print("\n=== EMBEDDING DEFAULTS ===")
for name in NAMES:
    if name.startswith(("embed.categorical.", "embed.kind", "embed.ekind", "embed.bins.")):
        t = W0[name]
        if isinstance(t, torch.Tensor):
            t = t.numpy()
        print(name, "row norms:", np.round(np.linalg.norm(t, axis=1), 3).tolist())

print("\n=== FIRST ACTION BUNDLES ===")
idx = np.flatnonzero(S0["has_insulin"] | S0["has_norepinephrine"])[:7]
for j in idx:
    sid, hour = int(S0["stay_id"][j]), float(S0["hour"][j])
    print("state", j, sid, hour, "n_atoms", S0["n_atoms"][j],
          "previous hour", float(S0["hour"][j-1]))
    rows = df[(df.stay_id == sid) &
              ((np.abs(df.hour-hour) < 1e-5) | (np.abs(df.end_hour-hour) < 1e-5))]
    print(rows[["hour", "event", "rate", "dose", "rate_unit", "dose_unit", "end_hour"]].to_string(index=False))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 16: exact posterior reconstruction prerequisites\n"
        "Retrieve latent-layer names, helper signatures, policy covariates, categorical "
        "defaults, and start/END action-bundle composition. Require a match to saved "
        "posterior states before using local features for an edit. No weights edited.\n"
        "Python runs: 16. Forward requests: 3.\n"
    )
```