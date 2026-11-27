The hidden features predict ordinary doses, but not held-out dose innovations. That suggests they may encode treatment-selection patterns rather than the dose deviations we need. I’ll check whether the information survives earlier in the state. If it does, a selective readout repair remains plausible; if not, the action-to-state pathway needs repair.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from pathlib import Path

feature_sets = {
    "out": head_recovery["state"][:, :192],
    "mean_h": head_recovery["state"][:, 192:960].reshape(-1, 4, 192).mean(1),
    "all_slots": head_recovery["state"][:, 192:960],
    "posterior_z": head_recovery["state"][:, 960:],
    "context": head_recovery["context"],
    "world_view": head_recovery["view"],
    "value_hidden": head_recovery["hidden"],
}
state_decode_rows = []
for feat_name0, feat0 in feature_sets.items():
    if hasattr(feat0, "detach"):
        feat0 = feat0.detach().cpu().numpy()
    feat0 = np.asarray(feat0, dtype=np.float64)
    for dj0, drug0 in enumerate(DRUGS):
        active0 = eligible & (gates[:, dj0] > 0)
        x0 = feat0[active0]
        y0 = recent_innovation[active0, dj0]
        fold0 = stay_fold[active0]
        for alpha0 in [10.0, 100.0, 1000.0]:
            pred0 = np.zeros(len(y0))
            for held0 in [0, 1]:
                tr0 = fold0 != held0
                te0 = ~tr0
                sc0 = StandardScaler().fit(x0[tr0])
                rg0 = Ridge(alpha=alpha0).fit(sc0.transform(x0[tr0]), y0[tr0])
                pred0[te0] = rg0.predict(sc0.transform(x0[te0]))
            state_decode_rows.append({
                "features": feat_name0, "drug": drug0, "alpha": alpha0,
                "r2": r2_score(y0, pred0),
                "correlation": np.corrcoef(y0, pred0)[0, 1]
            })
state_decode_table = pd.DataFrame(state_decode_rows)
print("=== BEST HELD-OUT INNOVATION DECODING PER FEATURE SET ===")
best0 = state_decode_table.groupby(["features", "drug"])["r2"].idxmax()
print(state_decode_table.loc[best0].to_string(index=False))

print("\n=== EXISTING HISTORY DESIGN ARRAYS ===")
for n0, o0 in list(globals().items()):
    if any(s0 in n0.lower() for s0 in ["hist", "policy", "local", "moment"]):
        if isinstance(o0, np.ndarray):
            print(n0, o0.shape)
        elif isinstance(o0, list):
            print(n0, "list length", len(o0))
        elif isinstance(o0, dict):
            print(n0, "keys", list(o0.keys())[:8])

print("\n=== SENSOR CORRELATIONS WITH DOSE INNOVATIONS ===")
for dj0, drug0 in enumerate(DRUGS):
    primary0, secondary0 = [(0, 1), (2, 3)][dj0]
    current0 = PO[:, 1:25, primary0]
    paired0 = PO[:, 1:25, secondary0]
    pair_ok0 = np.isfinite(current0) & np.isfinite(paired0)
    sensor_difference0 = current0 - paired0
    r0 = innovation_panel[:, :, dj0]
    print(drug0,
          "paired n", int(pair_ok0.sum()),
          "corr(noise contrast, dose innovation)",
          np.corrcoef(sensor_difference0[pair_ok0], r0[pair_ok0])[0, 1])

with Path("/work/notes.md").open("a") as f:
    f.write("\n## Run 38: readout vs state information\n"
            "- Sensor-gated hidden-feature innovation decoding failed on held-out "
            "stays despite good raw-dose decoding (R2 ~0.79 insulin, ~0.70 NE).\n"
            "- Tested innovation decoding earlier in the state, separately by "
            "relevant sensor family.\n"
            + state_decode_table.loc[best0].to_string(index=False) + "\n"
            "- No edits; five forward passes used.\n")
```
