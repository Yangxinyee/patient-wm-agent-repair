The posterior takes a 576-dimensional input containing context, observations and actions. I’ll validate the reconstructed inputs and fit small, stay-held-out dose decoders. Accurate dose decoding plus a usable pre-dose policy prediction would support a residual-dose edit; poor validation would rule it out.

```python
from sklearn.linear_model import Ridge
import pandas as pd, numpy as np, torch
torch.set_num_threads(8)

print("STATE FILE COUNTS:")
ds_states = {}
for k in (2, 4, 7, 8):
    s = dict(np.load(f"/work/fwd_{k}/states_" + ("edited" if k == 8 else "original") + ".npz"))
    print(k, s["x"].shape, len(np.unique(s["stay_id"])))
    ds_states[k] = s

ds_in = np.asarray(rr118_input).copy()
ds_u = np.asarray(rr119_u_nonzero)
ds_in[:, 384:] = ds_u
ds_w0 = torch.as_tensor(load_original("z_post.0.weight"))
ds_b0 = torch.as_tensor(load_original("z_post.0.bias"))
ds_w2 = torch.as_tensor(load_original("z_post.2.weight"))
ds_b2 = torch.as_tensor(load_original("z_post.2.bias"))
ds_t = torch.from_numpy(ds_in)
ds_pre = ds_t @ ds_w0.T + ds_b0
print("\nPOSTERIOR RECONSTRUCTION:")
for label, activation in [("relu", torch.relu), ("gelu", torch.nn.functional.gelu), ("silu", torch.nn.functional.silu)]:
    zp = (activation(ds_pre) @ ds_w2.T + ds_b2)[:, :32].numpy()
    print(label, "RMSE against cached original:", float(np.sqrt(np.mean((zp - np.asarray(rr118_oldz))**2))))
    for k, s in ds_states.items():
        if len(s["x"]) == len(ds_in):
            print("  file", k, "RMSE:", float(np.sqrt(np.mean((zp - s["x"][:, -32:])**2))))
print("\nACTION HELPER REFERENCES:", repair_action_encoding.__code__.co_names)
print("RR118 component errors:", float(np.sqrt(np.mean((ds_in[:,192:384] - np.asarray(rr117_e))**2))), float(np.sqrt(np.mean((np.asarray(rr118_input)[:,384:] - np.asarray(rr117_u))**2))))

ds_matching = [s for k, s in ds_states.items() if k != 8 and len(s["x"]) == len(ds_in)]
if ds_matching:
    ds_s = ds_matching[0]
    ds_candidates = [v for v in list(globals().values()) if isinstance(v, pd.DataFrame)
                     and {"stay_id","hour","event","rate"}.issubset(v.columns)
                     and len(v) and int(v["stay_id"].iloc[0]) < 2000000000]
    ds_records = max(ds_candidates, key=len)
    ds_dr = ds_records[ds_records["event"].isin(["insulin","norepinephrine"])].copy()
    ds_dr["second"] = np.rint(ds_dr["hour"] * 3600).astype(np.int64)
    ds_lookup = ds_dr.pivot_table(index=["stay_id","second"], columns="event", values="rate", aggfunc="sum")
    ds_keys = pd.MultiIndex.from_arrays([ds_s["stay_id"], np.rint(ds_s["hour"]*3600).astype(np.int64)])
    ds_rates = ds_lookup.reindex(ds_keys).reindex(columns=["insulin","norepinephrine"]).fillna(0).to_numpy()
    ds_unique = np.unique(ds_s["stay_id"])
    ds_hold = np.isin(ds_s["stay_id"], ds_unique[::4])
    ds_decoders = {}
    print("\nDOSE DECODER VALIDATION:")
    for j, drug in enumerate(["insulin","norepinephrine"]):
        active = ds_rates[:,j] > 0
        for feature_name, feature in [("action",ds_u), ("context",ds_in[:,:192]), ("context_observation",ds_in[:,:384])]:
            fitmask = active & ~ds_hold if feature_name != "action" else ~ds_hold
            testmask = active & ds_hold
            reg = Ridge(alpha=1.0).fit(feature[fitmask], ds_rates[fitmask,j])
            pr = reg.predict(feature[testmask])
            yy = ds_rates[testmask,j]
            r2 = 1 - np.sum((pr-yy)**2)/max(np.sum((yy-yy.mean())**2),1e-12)
            print(drug, feature_name, "n:", int(testmask.sum()), "R2:", round(float(r2),4),
                  "RMSE:", round(float(np.sqrt(np.mean((pr-yy)**2))),4))
            ds_decoders[(drug,feature_name)] = reg
with open("/work/notes.md", "a") as f:
    f.write("\n## Run 153\nTesting exact posterior reconstruction and stay-held-out current-dose/policy decoders from action and pre-dose context. No edits. Counts: 153 Python runs, 8 forward passes.\n")
```