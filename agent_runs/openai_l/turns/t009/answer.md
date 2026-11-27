The architecture shows that dose changes enter a shared recurrent state, so an embedding edit could also disturb unrelated predictions. Before editing, I’ll verify the dose-to-reading alignment and look for dose variation among otherwise similar histories. A sign reversal caused by timing or selection would invalidate the provisional targets.

```python
import numpy as np, pandas as pd
from pathlib import Path

print("=== EVENT TIMING AND DOSES ===")
print(train.groupby("event").agg(n=("stay_id","size"), min_hour=("hour","min"),
                                  max_hour=("hour","max"), mean_rate=("rate","mean")).round(4))
print(train.loc[train.event.isin(["insulin","norepinephrine"])].head(12).to_string(index=False))

rec_ids = np.sort(train.stay_id.unique())
rec_hours = np.arange(26)
rec_index = pd.MultiIndex.from_product([rec_ids, rec_hours], names=["stay_id","t"])
rec = train.loc[train.hour.ge(0)].copy()
rec["t"] = np.floor(rec.hour + 1e-5).astype(int)
rec_value = rec.pivot(index=["stay_id","t"], columns="event", values="value").reindex(rec_index)
rec_rate = rec.pivot(index=["stay_id","t"], columns="event", values="rate").reindex(rec_index)
P = {e: rec_value[e].to_numpy().reshape(-1,26)
     for e in ["glucose_finger","glucose_lab","map_arterial","map_cuff"]}
A = {e: rec_rate[e].fillna(0).to_numpy().reshape(-1,26)
     for e in ["insulin","norepinephrine"]}

print("\n=== PANEL AND POLICY CHECKS (hours 0..23) ===")
for drug, sensor in [("insulin","glucose_finger"), ("norepinephrine","map_arterial")]:
    a, y = A[drug][:,:24].ravel(), P[sensor][:,:24].ravel()
    print(drug, "active fraction", np.mean(a>0),
          "positive-rate quantiles", np.quantile(a[a>0], [0,.1,.5,.9,1]),
          "current sensor observed among active", np.mean(np.isfinite(y[a>0])))
    width = 10 if drug == "insulin" else 2
    q = pd.DataFrame({"reading": width*np.floor(y/width), "rate": a})
    tab = q.groupby("reading").agg(n=("rate","size"), active=("rate",lambda x: np.mean(x>0)),
                                   mean=("rate","mean"), sd=("rate","std"))
    print(tab.loc[tab.n.ge(100)].round(3).to_string())

print("\n=== PREVIOUS FIT STRUCTURES ===")
for key in [("glucose_finger","both sensors current",False),
            ("glucose_lab","both sensors current",False),
            ("map_arterial","both sensors current",False),
            ("map_cuff","both sensors current",False)]:
    v = rich_fits.get(key)
    print(key, repr(v)[:1600])
for k,v in policy_fits.items():
    print("policy", k, repr(v)[:1300])
for k,v in ssm_fits.items():
    print("SSM", k, repr(v)[:1800])

print("\n=== KALMAN FUNCTION CONSTANTS / VARIABLE NAMES ===")
print(kalman_nll.__code__.co_varnames)
print(kalman_nll.__code__.co_consts)

print("\n=== RELEVANT TENSORS ===")
for n in NAMES:
    if n.startswith(("embed.", "encode_action.", "last_", "attr_heads.value_bin", "query_norm", "z_in.", "z_head.")):
        print(n, SHAPES[n])
print("\n=== VOCAB STRUCTURE ===")
print("keys", list(VOCAB))
for k,v in VOCAB.items():
    if k in ["codes", "config"] or any(s in k.lower() for s in ["normal", "scale", "median"]):
        print(k, v)
    elif isinstance(v, dict):
        print(k, "keys", list(v)[:12])
    else:
        print(k, str(v)[:150])
with open("/work/notes.md","a") as f:
    f.write("\n## Run 7\nBuilt independently aligned P (readings at integer hour) and A (drug records floored to their start hour). Inspecting treatment support and previous fit implementation; no causal target or edit accepted yet.\n")
```