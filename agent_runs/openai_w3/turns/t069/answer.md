Both state files are available. I’ll compare posterior and recurrent-state changes separately, then check the local posterior reconstruction against the stored original states. Agreement would validate the local diagnostics; disagreement would mean the encoding must be fixed before further weight edits.

```python
import numpy as np
import pandas as pd

states6_original = dict(np.load("/work/fwd_6/states_original.npz"))
states6_edited = dict(np.load("/work/fwd_6/states_edited.npz"))
state_delta6 = states6_edited["x"] - states6_original["x"]
drug_mask6 = (states6_original["has_insulin"].astype(bool)
              | states6_original["has_norepinephrine"].astype(bool))

print("AUTHORITATIVE PILOT STATE CHANGES")
for label, section in (("out", slice(0, 192)),
                       ("h", slice(192, 960)),
                       ("posterior mean", slice(960, 992))):
    for group, mask in (("all", np.ones(len(state_delta6), dtype=bool)),
                        ("drug", drug_mask6),
                        ("no drug", ~drug_mask6)):
        delta = state_delta6[mask, section]
        rms = float(np.sqrt(np.mean(delta ** 2)))
        maximum = float(np.max(np.abs(delta)))
        print(label, group, "rows", int(mask.sum()), "rms", rms, "max", maximum)

delta_norm6 = np.linalg.norm(state_delta6[:, 960:992], axis=1)
top6 = np.argsort(delta_norm6)[-8:][::-1]
print("\nLARGEST POSTERIOR CHANGES")
print(pd.DataFrame({
    "stay": states6_original["stay_id"][top6],
    "hour": states6_original["hour"][top6],
    "insulin": states6_original["has_insulin"][top6],
    "ne": states6_original["has_norepinephrine"][top6],
    "z_delta_norm": delta_norm6[top6],
    "out_delta_norm": np.linalg.norm(state_delta6[top6, :192], axis=1)
}).to_string(index=False))

print("\nLOCAL RECONSTRUCTION AGAINST TRAINING STATES")
target_post61 = diagnostic_array(S["x"])[:, 960:992]
local_error61 = _orig_post60[:, :32] - target_post61
masks61 = {
    "all": np.ones(len(target_post61), dtype=bool),
    "drug": diagnostic_array(S["has_insulin"]).astype(bool)
            | diagnostic_array(S["has_norepinephrine"]).astype(bool),
    "insulin": diagnostic_array(S["has_insulin"]).astype(bool),
    "ne": diagnostic_array(S["has_norepinephrine"]).astype(bool)
}
if np.asarray(r40_any_start).shape == (len(target_post61),):
    masks61["WORLD drug start"] = np.asarray(r40_any_start, dtype=bool)
for label, mask in masks61.items():
    error = local_error61[mask]
    print(label, "rows", int(mask.sum()),
          "MSE", float(np.mean(error ** 2)),
          "max", float(np.max(np.abs(error))))

action_input61 = diagnostic_array(r46_input)[:, 384:]
action_discrepancy61 = action_input61 - repair_actual
print("\nACTION INPUT VS FRESH ENCODING:",
      "rms", float(np.sqrt(np.mean(action_discrepancy61 ** 2))),
      "max", float(np.max(np.abs(action_discrepancy61))))

print("\nBUNDLE METADATA")
print("r40_info type:", type(r40_info).__name__)
print("first entry:", repr(r40_info[0])[:1000])
print("state_records columns:", state_records.columns.tolist())
print("training IDs:", list(train_ids[:5]))
for name in ("context_with_flow", "make_atom", "encode_bundles",
             "encode_one_set", "clean_empty_obs", "mlp", "layer_norm"):
    function = globals().get(name)
    if callable(function):
        try:
            print(name, inspect.signature(function))
        except Exception:
            print(name, "signature unavailable")

with open("/work/notes.md", "a") as f:
    f.write("\nPython run 61; forwards 6. Authoritative original/edited "
            "tuning states acquired. Comparing actual posterior and recurrent-state "
            "changes and auditing local posterior reconstruction against training states.\n")
```