The action-only bundles contain one drug atom, so simultaneous measurements do not explain the mismatch. One possibility is that the “empty” measurement encoder contributes a nonzero vector through its biases. I’ll check that and recover the lag estimates. A large reduction in posterior error would support the empty-encoder explanation; little change would refute it.

```python
import numpy as np, pandas as pd, torch
from pathlib import Path

print("=== LONGITUDINAL EFFECT DIAGNOSTICS ===")
for drug in ["insulin", "norepinephrine"]:
    print("\n", drug)
    for name in ["impulse_tables_repair", "provisional_kernels", "_moment_inputs", "persistence_repair"]:
        obj = globals().get(name, {}).get(drug)
        if isinstance(obj, pd.DataFrame):
            print(name, obj.to_string(index=False)[:1800])
        else:
            print(name, str(obj)[:1000])

print("\n=== LATE ARRAY INVENTORY ===")
arrs = [(k, tuple(v.shape)) for k, v in list(globals().items())
        if isinstance(v, (np.ndarray, torch.Tensor))]
print("\n".join(f"{k}: {s}" for k, s in arrs[-70:]))

print("\n=== STATE FILES ===")
for p in sorted(Path("/work").glob("fwd_*/states_original.npz")):
    zz = np.load(p)
    print(str(p), {k: zz[k].shape for k in zz.files})
    print("first stays:", np.unique(zz["stay_id"])[:3])

print("\n=== ENCODER CONSTANTS ===")
for k in ["rp_linear", "rp_mlp", "rp_action_encoder", "head_features_fixed"]:
    fn = globals().get(k)
    if fn:
        print(k, "global names:", fn.__code__.co_names, "defaults:", fn.__defaults__)
for n in NAMES:
    if n.startswith(("encode.", "gap.")):
        print(n, SHAPES[n])
print("event kinds:", VOCAB["event_kinds"])
for n in NAMES:
    if n.startswith("embed.categorical"):
        ww = np.asarray(load(n))
        print(n, "row norms", np.round(np.linalg.norm(ww, axis=1), 3))

note("Run 37: recovered longitudinal diagnostics and action-reconstruction arrays. Testing whether the empty measurement encoder, rather than zero e, explains posterior mismatch. No edits.")
```
