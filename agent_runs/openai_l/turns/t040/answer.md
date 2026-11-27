We have state snapshots for about 100 tuning stays. I’ll check the available treatment-residual estimates and gather states from training stays as well. A residual-dose correction is promising only if the state exposes the dose while also capturing enough of the prescribing policy to preserve factual predictions.

```python
import numpy as np
import pandas as pd

print("=== CORE AND INPUT TENSOR NAMES ===")
for name in NAMES:
    if name.startswith(("core.", "embed.categorical.")):
        print(name, SHAPES[name])
print("event kinds:", VOCAB["event_kinds"])
print("atom kinds:", VOCAB["atom_kinds"])

print("\n=== EXISTING REGRESSION OBJECTS ===")
for name, obj in sorted(list(globals().items())):
    if name.startswith("_"):
        continue
    if any(s in name.lower() for s in ("resid", "model", "r_a", "r_y", "effect", "policy")):
        if isinstance(obj, dict):
            print(name, "dict keys:", list(obj)[:12])
        elif isinstance(obj, (list, tuple)):
            print(name, type(obj).__name__, "length", len(obj),
                  "first type", type(obj[0]).__name__ if obj else "")
        elif hasattr(obj, "shape"):
            print(name, type(obj).__name__, obj.shape)
        elif hasattr(obj, "predict"):
            print(name, type(obj).__name__)

print("\n=== TIME ALIGNMENT AND SAMPLE RECORD ===")
print("hours:", hours)
for collection_name, collection in (("P", P), ("A", A)):
    print(collection_name, {
        k: (type(v).__name__, getattr(v, "shape", None))
        for k, v in collection.items()
    })
first_train36 = int(train["stay_id"].iloc[0])
print(train.loc[train.stay_id.eq(first_train36)].head(22).to_string(index=False))

train_ids36 = np.sort(train.stay_id.unique())
rng36 = np.random.default_rng(3611)
state_train_ids36 = np.sort(rng36.choice(train_ids36, 100, replace=False)).astype(int)
print("\n=== TRAIN IDS FOR NEXT STATE REQUEST ===")
print(state_train_ids36.tolist())

print("\n=== CURRENT EFFECT AND KERNEL ESTIMATES ===")
for event in ("glucose_finger", "glucose_lab", "map_arterial", "map_cuff"):
    print(event, "kernel:", np.round(kernels19[event], 4),
          "SE:", np.round(kernel_se19[event], 4))
print("edits:", edited())

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 36 / consolidated recovery\n"
            "No edits; forward 3/50, Python 36/300. Original factual value NLL "
            "1.63164. Original reached-row rollout effects: insulin x2 -3.680, "
            "half +2.018; NE x2 +0.465, half -0.871. Record innovation/RD evidence "
            "supports immediate insulin about -1.95 per unit and NE about +3.1 "
            "per rate unit. Eight-lag kernels in kernels19; provisional hour-12 "
            "row-weighted targets -12.77,+6.38,+4.50,-2.25, with tail/linearity "
            "uncertainty. CPU decoder best r32_best has GELU gap/prior/head, "
            "seconds log1p /10, measurement type 1; last numeric has flags column0, "
            "age column3, value z column6, absent rate/status bin0. MC errors "
            "approximately expected sampling noise. Existing 100 tuning-stay "
            "states in /work/fwd_3/states_original.npz (x4356x992), matched "
            "4751 factual readings in d22; states are [out192,h768,z32]. "
            "Considering low-rank residual-action correction in z_post or "
            "value decoder, not an unconditional dose rescaling. Exact posterior "
            "z_post.0 is 192x576 [ctx,e,u], z_post.2 is 64x192. "
            "No gradients or optimizer. Training state IDs in state_train_ids36.\n")
```