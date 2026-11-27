Doses occur five minutes after hourly readings, giving a clear before-and-after ordering. Insulin rates are discretized and norepinephrine rates are capped, which may provide useful comparisons near dosing thresholds. A tensor-inspection error interrupted the last run; I’ll fix it and examine hourly trajectories. Raw associations alone will not establish drug effects.

```python
import torch

def as_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

print("=== RELEVANT TENSORS ===")
prefixes = ("embed.", "encode_action.", "attr_heads.value", "embed_transition.value",
            "last_numeric", "last_status", "query_norm", "type_emb", "core.z_post")
for name in NAMES:
    if name.startswith(prefixes):
        w = as_numpy(load_original(name))
        print(name, tuple(w.shape), "rms =", round(float(np.sqrt(np.mean(w*w))), 5))

print("\n=== DRUG NORMALIZATION ===")
for entry in VOCAB["numeric"]:
    if entry["code"] in ["insulin", "norepinephrine"]:
        print(entry)
print("other vocab keys:", [k for k in VOCAB if k not in
                            ["numeric", "codes", "event_kinds", "atom_kinds", "config"]])
for key in ["value_bin_centres", "categorical"]:
    if key in VOCAB:
        print(key, str(VOCAB[key])[:2200])

print("\n=== HOURLY ARRAYS ===")
events6 = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff",
           "insulin", "norepinephrine"]
def make_arrays(df):
    ids = np.sort(df.stay_id.unique())
    sub = df[df.event.isin(events6)].copy()
    sub["t"] = np.floor(sub.hour + 1e-5).astype(int)
    sub["v"] = np.where(sub.event.isin(events6[-2:]), sub.rate, sub.value)
    wide = sub.pivot(index=["stay_id", "t"], columns="event", values="v")
    index = pd.MultiIndex.from_product([ids, range(-1, 25)], names=["stay_id", "t"])
    wide = wide.reindex(index)
    arr = {e: wide[e].to_numpy().reshape(len(ids), 26).astype(np.float64)
           for e in events6}
    for e in events6[-2:]:
        arr[e] = np.nan_to_num(arr[e], nan=0.0)
    return ids, arr

train_ids, A = make_arrays(train)
tuning_ids, AT = make_arrays(tuning)
for e in events6[:4]:
    counts = np.isfinite(A[e]).sum(axis=1)
    print(e, "readings per stay quantiles:",
          np.quantile(counts, [0, .25, .5, .75, 1]).tolist(),
          "never measured:", int((counts == 0).sum()))

print("\n=== RAW ADJACENT-HOUR ASSOCIATIONS (NOT CAUSAL) ===")
for drug, outcome in [("insulin", "glucose_finger"),
                      ("norepinephrine", "map_arterial")]:
    y0, y1, d = A[outcome][:, 1:-1], A[outcome][:, 2:], A[drug][:, 1:-1]
    ok = np.isfinite(y0) & np.isfinite(y1)
    x = np.column_stack([np.ones(ok.sum()), y0[ok], d[ok]])
    beta = np.linalg.lstsq(x, y1[ok], rcond=None)[0]
    print(drug, "n =", int(ok.sum()), "next-reading regression [intercept, current, dose]:",
          beta.round(6).tolist())
    print("mean next-minus-current by dose band:")
    tab = pd.DataFrame({"dose": d[ok], "current": y0[ok], "change": (y1-y0)[ok]})
    tab["band"] = pd.cut(tab.dose, [-1e-6, 0, .5, 1, 2, 4, 100])
    print(tab.groupby("band", observed=True).agg(
        n=("change", "size"), dose=("dose", "mean"),
        current=("current", "mean"), change=("change", "mean")
    ).round(4).to_string())

print("\n=== EXAMPLE TREATED TRAJECTORIES ===")
for drug in events6[-2:]:
    i = np.flatnonzero((A[drug][:, 1:] > 0).sum(axis=1) >= 20)[0]
    print("drug:", drug, "stay:", train_ids[i])
    show = pd.DataFrame({e: A[e][i] for e in events6}, index=range(-1, 25))
    print(show.loc[0:9].round(3).to_string())

with Path("/work/notes.md").open("a") as f:
    f.write("\nPython runs: 3; forward requests: 0. Loaded 1,998,495 training rows "
            "(30,000 stays), 400 tuning stays. Measurements are hourly; drug records "
            "start at hour+0.0833, last one hour. Insulin rates in 0.5 increments; "
            "norepinephrine rates in 0.01 increments, capped at 1.0. "
            "Run 2 tensor summary failed because load returns torch tensors; fixed "
            "with as_numpy. Constructed hourly arrays A/AT, times -1..24.\n")
```
