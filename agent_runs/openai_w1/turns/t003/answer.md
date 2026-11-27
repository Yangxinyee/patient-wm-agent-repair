Measurements occur on integer hours, and infusions start five minutes later. Doses vary substantially, so readings before each infusion can help separate dose assignment from subsequent response. I’ll build an aligned hourly panel and compare crude, history-adjusted, and within-stay associations. Agreement would strengthen a causal hypothesis; large changes across adjustments would warn of confounding.

```python
python_runs += 1
from scipy.linalg import lstsq

sensors = ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]
drugs = ["insulin", "norepinephrine"]
stay_ids = np.sort(train.stay_id.unique())
hours = np.arange(-1, 25)
ns, nt = len(stay_ids), len(hours)
P = {}
R = {}

for event in sensors + drugs:
    rows = train[train.event == event]
    arr = np.full((ns, nt), np.nan if event in sensors else 0.0)
    ii = np.searchsorted(stay_ids, rows.stay_id.to_numpy())
    hh = np.rint(rows.hour.to_numpy()).astype(int) if event in sensors else np.floor(rows.hour.to_numpy()).astype(int)
    valid = (hh >= -1) & (hh <= 24)
    arr[ii[valid], hh[valid] + 1] = rows["value" if event in sensors else "rate"].to_numpy()[valid]
    (P if event in sensors else R)[event] = arr

L, Age = {}, {}
for event, arr in P.items():
    obs = np.isfinite(arr)
    last_index = np.maximum.accumulate(
        np.where(obs, np.arange(nt)[None, :], -1), axis=1
    )
    last = np.take_along_axis(arr, np.maximum(last_index, 0), axis=1)
    last[last_index < 0] = np.nanmedian(arr)
    L[event] = last
    Age[event] = np.where(last_index >= 0,
                          np.arange(nt)[None, :] - last_index, 30)

print("=== HOURLY COVERAGE ===")
for event in sensors:
    print(event, "fraction observed at hours 0..24:",
          np.round(np.isfinite(P[event][:, 1:]).mean(axis=0), 3).tolist())
for drug in drugs:
    a = R[drug][:, 1:]
    print(drug, "fraction dosed:", round(float((a > 0).mean()), 4),
          "mean rate including zero:", round(float(a.mean()), 4))

# Transition from integer hour t to t+1; drug t starts at t+0.0833.
tgrid = np.broadcast_to(np.arange(24)[None, :], (ns, 24))
history_cols, history_names = [], []
for event in sensors:
    scale = 50.0 if "glucose" in event else 20.0
    x = (L[event][:, 1:25] - np.nanmedian(P[event])) / scale
    history_cols.extend([x, x*x, Age[event][:, 1:25] / 10.0])
    history_names.extend([event, event+"^2", event+"_age"])
history_cols.extend([tgrid / 24.0, (tgrid / 24.0)**2])
history_names.extend(["hour", "hour^2"])
for drug in drugs:
    scale = 1.0 if drug == "insulin" else 0.2
    history_cols.extend([R[drug][:, :24] / scale,
                         np.column_stack([np.zeros(ns), R[drug][:, :23]]) / scale])
    history_names.extend([drug+"_lag1", drug+"_lag2"])
H = np.stack(history_cols, axis=-1)

print("\n=== ASSOCIATIONS: NOT YET CAUSAL ESTIMATES ===")
association_results = []
for event, drug in [
    ("glucose_finger", "insulin"), ("glucose_lab", "insulin"),
    ("map_arterial", "norepinephrine"), ("map_cuff", "norepinephrine")
]:
    y0, y1 = P[event][:, 1:25], P[event][:, 2:26]
    valid = np.isfinite(y0) & np.isfinite(y1)
    outcome = (y1 - y0)[valid]
    dose = R[drug][:, 1:25][valid]
    other = R[drugs[1] if drug == drugs[0] else drugs[0]][:, 1:25][valid]
    sid_index = np.broadcast_to(np.arange(ns)[:, None], (ns, 24))[valid]
    controls = H[valid]
    crude = lstsq(np.column_stack([np.ones(len(outcome)), dose]), outcome)[0][1]
    X = np.column_stack([dose, other, controls])
    coef = lstsq(np.column_stack([np.ones(len(outcome)), X]), outcome)[0]
    counts = np.bincount(sid_index, minlength=ns)
    means_X = np.zeros((ns, X.shape[1]))
    np.add.at(means_X, sid_index, X)
    means_X /= np.maximum(counts[:, None], 1)
    means_y = np.bincount(sid_index, weights=outcome, minlength=ns) / np.maximum(counts, 1)
    within_coef = lstsq(X - means_X[sid_index], outcome - means_y[sid_index])[0]
    result = dict(event=event, drug=drug, n=len(outcome), crude=crude,
                  adjusted=coef[1], within_stay=within_coef[0])
    association_results.append(result)
print(pd.DataFrame(association_results).round(5).to_string(index=False))

rng = np.random.default_rng(713)
all_tuning_ids = np.sort(tuning.stay_id.unique())
tuning_batches = [x.tolist() for x in np.array_split(rng.permutation(all_tuning_ids), 4)]
eval_ids = tuning_batches[0]
print("\n=== BASELINE FORWARD STAYS ===")
print(json.dumps(eval_ids))

with notes_path.open("a") as f:
    f.write(
        "\n## Run 3\n"
        "Constructed hourly panels: measurements at integer hours, doses five minutes later. "
        "Missing drug records represent no infusion, not carried-forward rates. "
        "Compared crude, history-adjusted and within-stay one-step associations; "
        "these are diagnostics, not accepted causal estimates.\n"
        + pd.DataFrame(association_results).to_markdown(index=False)
        + f"\nCounts: {python_runs} Python runs, {forward_passes} forward passes.\n"
    )
```
