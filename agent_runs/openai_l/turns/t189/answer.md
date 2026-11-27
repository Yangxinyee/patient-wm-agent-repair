The tool returned all 4,491 edited bundle states for the matching training sample. That enables fitting, but is not itself an accuracy result. I’ll align the states and validate the CPU decoder and one-step dose contrasts. Near-identical factual outputs at multiplier ×1 would confirm that these helpers are suitable for constructing the edit.

```python
import numpy as np
import pandas as pd
import torch
from pathlib import Path

torch.set_num_threads(8)

state_original173 = np.load("/work/fwd_4/states_original.npz")
state_edited173 = np.load("/work/fwd_10/states_edited.npz")
keys_original173 = pd.MultiIndex.from_arrays([
    state_original173["stay_id"], np.round(state_original173["hour"], 8)
])
keys_edited173 = pd.MultiIndex.from_arrays([
    state_edited173["stay_id"], np.round(state_edited173["hour"], 8)
])
order173 = keys_edited173.get_indexer(keys_original173)
assert np.all(order173 >= 0) and len(np.unique(order173)) == len(order173)

sx_original173 = sx63.clone()
cache_original173 = {
    name: w63_cache[name].clone() if name in w63_cache else None
    for name in edited()
}
for name in edited():
    w63_cache[name] = torch.as_tensor(load(name), dtype=torch.float32)
sx63 = torch.as_tensor(state_edited173["x"][order173], dtype=torch.float32)

ctx_original173, _ = context64(
    sx_original173[pi73, :192],
    sx_original173[pi73, 192:960].reshape(-1, 4, 192),
    dtread73
)
ctx_edited173, _ = context64(
    sx63[pi73, :192],
    sx63[pi73, 192:960].reshape(-1, 4, 192),
    dtread73
)
means_original173, probs_original173, hidden_original173, logits_original173 = decode67(
    ctx_original173, const_train67, trans_train67,
    centres_train67, epsread73, return_features=True
)
means_edited173, probs_edited173, hidden_edited173, logits_edited173 = decode67(
    ctx_edited173, const_train67, trans_train67,
    centres_train67, epsread73, return_features=True
)
input_edited173 = head_inputs96(ctx_edited173, const_train67, epsread73)

print("State alignment:", len(order173), "bundles")
print("Factual mean-change RMS:",
      float((means_edited173.mean(0) - means_original173.mean(0)).square().mean().sqrt()))
print("Factual logit-change RMS:",
      float((logits_edited173 - logits_original173).square().mean().sqrt()))

cf_features_original173 = cf_features81
cf_features81 = {}
for drug in ["insulin", "norepinephrine"]:
    for factor in [1.0, 2.0]:
        make_cf81(drug, factor, batch_size=128)
    base = cf_features81[(drug, 1.0)]
    changed = cf_features81[(drug, 2.0)]
    rows = np.asarray(base["rows"])
    bmean = np.asarray(base["means"].mean(0))
    cmean = np.asarray(changed["means"].mean(0))
    actual = np.asarray(means_edited173.mean(0))[rows]
    print("\n", drug, "rows:", len(rows),
          "x1 CPU/full-state mean RMS:", float(np.sqrt(np.mean((bmean - actual)**2))))
    dr = np.asarray(changed["delta_rate"]).reshape(-1)
    for event in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
        use = measure67.iloc[rows]["event"].to_numpy() == event
        if use.any():
            delta = cmean[use] - bmean[use]
            slope = np.dot(dr[use], delta) / np.dot(dr[use], dr[use])
            print(event, "n", int(use.sum()), "mean shift", round(float(delta.mean()), 5),
                  "per-rate slope", round(float(slope), 5))
    print("feature shapes:",
          {k: tuple(v.shape) for k, v in changed.items() if hasattr(v, "shape")})

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 173 / edited training-state decoder recovery\n"
            "Forward 9 supplied 4491 edited states on the original 100-stay "
            "training feature sample. Aligned by stay/hour; CPU cache now uses "
            "the three current edits. Recovering factual decoder features and "
            "x1/x2 one-step contrasts. No new weights saved.\n")
```