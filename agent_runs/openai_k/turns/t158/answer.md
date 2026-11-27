The official states are available for the same stays as the factual test. I’ll align each reading with the state immediately before it and inspect the head interfaces. This avoids using post-reading information; correct alignment should substantially improve agreement with official predictions.

```python
import numpy as np, pandas as pd, torch, inspect
from pathlib import Path

S143o = dict(np.load("/work/fwd_10/states_original.npz"))
S143e = dict(np.load("/work/fwd_10/states_edited.npz"))
F143o = pd.read_csv("/work/fwd_8/factual_original.csv")
F143e = pd.read_csv("/work/fwd_8/factual_edited.csv")
R143 = pd.read_csv("/data/tuning.csv")

M143 = R143[R143.event.isin(
    ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
)].sort_values(["stay_id", "event", "hour"]).copy()
M143["last_value"] = M143.groupby(["stay_id", "event"]).value.shift()
M143["last_hour"] = M143.groupby(["stay_id", "event"]).hour.shift()
F143 = F143o.merge(
    F143e[["stay_id", "hour", "event", "predicted_mean", "nll"]],
    on=["stay_id", "hour", "event"], suffixes=("_o", "_e"),
    validate="one_to_one"
).merge(
    M143[["stay_id", "hour", "event", "last_value", "last_hour"]],
    on=["stay_id", "hour", "event"], validate="one_to_one"
)
F143["state_index"] = -1
for sid, idx in F143.groupby("stay_id").groups.items():
    si = np.flatnonzero(S143o["stay_id"] == sid)
    si = si[np.argsort(S143o["hour"][si])]
    positions = np.searchsorted(
        S143o["hour"][si], F143.loc[idx, "hour"].to_numpy(), side="left"
    ) - 1
    F143.loc[idx, "state_index"] = np.where(
        positions >= 0, si[np.maximum(positions, 0)], -1
    )
F143 = F143[F143.state_index >= 0].copy().reset_index(drop=True)
ix143 = F143.state_index.to_numpy(dtype=int)
F143["dt_seconds"] = (
    F143.hour.to_numpy() - S143o["hour"][ix143]
) * 3600
print("Aligned readings:", len(F143))
print("dt seconds:", F143.dt_seconds.describe().to_dict())
print(F143[["stay_id", "hour", "event", "last_value",
            "last_hour", "dt_seconds"]].head(6).to_string(index=False))

print("\n=== HEAD INTERFACES ===")
for n in NAMES:
    if any(n.startswith(p) for p in
           ["last_numeric", "last_status", "query_norm", "type_emb",
            "z_prior", "z_head", "gap", "out_norm", "attr_heads.value_bin"]):
        print(n, SHAPES[n])
print("Config:", VOCAB.get("config"))
print("Categorical keys:", list(VOCAB.get("categorical", {})))

print("\n=== RETAINED FEATURE FUNCTIONS ===")
for name, limit in [("r59_features", 2200), ("prior_features74", 2200)]:
    fun143 = globals().get(name)
    print(name)
    if fun143 is not None:
        try:
            print(inspect.getsource(fun143)[:limit])
        except Exception:
            compact_code95(fun143, limit=limit)

print("\n=== OFFICIAL EDIT DELTAS ===")
F143["mean_delta"] = F143.predicted_mean_e - F143.predicted_mean_o
print(F143.groupby("event").mean_delta.agg(["mean", "std", "min", "max"]))
print("State delta RMS:",
      np.sqrt(np.mean((S143e["x"][:, :960] - S143o["x"][:, :960]) ** 2)))

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 143 / Forward 10: paired authoritative states\n"
            "- Loaded original/edited tuning-99 states and aligned factual readings to strict pre-bundle states.\n"
            "- Rebuilding readout calibration from official predictions, without post-reading leakage.\n"
            "- Python runs: 143; forward passes: 10; edits unchanged.\n")
```