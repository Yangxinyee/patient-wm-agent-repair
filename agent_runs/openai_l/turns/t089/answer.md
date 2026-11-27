The original model’s next-hour insulin response is only about −0.58 per added unit on finger glucose and −0.39 on lab glucose, versus about −1.95 in the records. I’ll add halved-dose and norepinephrine examples, then fit against these contrasts while checking factual loss on held-out stays. This will test whether the final head has enough useful dose information.

```python
for drug, factor in [("insulin", 0.5),
                     ("norepinephrine", 2.0),
                     ("norepinephrine", 0.5)]:
    c = make_cf81(drug, factor)
    r = c["rows"]
    shift = (c["means"]-means_fact80[:, r]).mean(0).numpy()
    print("\n", drug, factor)
    for ev in np.unique(events74):
        m = events74[r] == ev
        if m.any():
            dr = c["delta_rate"][m]
            print(ev, "mean shift", float(shift[m].mean()),
                  "slope", float(np.dot(dr, shift[m])/np.dot(dr, dr)))

print("\n=== ALIGNED FACTUAL OBSERVATIONS ===")
records82 = pd.read_csv("/data/train.csv",
                        usecols=["stay_id", "hour", "event", "value"])
records82 = records82[
    records82.stay_id.isin(state_train_ids36) &
    records82.event.isin(["glucose_finger", "glucose_lab",
                         "map_arterial", "map_cuff"])
].copy()
records82 = records82.sort_values(["stay_id", "event", "hour"])
records82["last_value"] = records82.groupby(["stay_id", "event"]).value.shift()
records82["hour_key"] = records82.hour.round(4)
records82 = records82.set_index(["stay_id", "hour_key", "event"])

read_stays82 = sid63[pi73]
read_hours82 = np.round(hour63[pi73]+dtread73.numpy()/3600, 4)
read_keys82 = pd.MultiIndex.from_arrays(
    [read_stays82, read_hours82, events74],
    names=["stay_id", "hour_key", "event"])
aligned82 = records82.reindex(read_keys82)
assert aligned82.value.notna().all(), "Reading alignment failed"
observed82 = aligned82.value.to_numpy(np.float32)
last_values82 = aligned82.last_value.to_numpy(np.float32)
observed_bins82 = np.zeros(len(observed82), dtype=np.int64)
last_bins82 = np.zeros(len(observed82), dtype=np.int64)
for ev in np.unique(events74):
    m = events74 == ev
    edges = np.asarray(num42[(ev, "value")]["fine_bin_edges"], np.float32)
    observed_bins82[m] = np.searchsorted(edges, observed82[m], side="left")+1
    present = m & np.isfinite(last_values82)
    last_bins82[present] = np.searchsorted(
        edges, last_values82[present], side="left")+1
transition_rows82 = np.array([code42[e] for e in events74])*17+last_bins82
pf82 = probs_fact80.mean(0).numpy()
nll_base82 = -np.log(np.maximum(
    pf82[np.arange(len(pf82)), observed_bins82], 1e-30))
rng82 = np.random.default_rng(8201)
train_stays82 = rng82.permutation(state_train_ids36)[:75]
fit_mask82 = np.isin(read_stays82, train_stays82)
print("original CPU NLL:", nll_base82.mean(),
      "fit:", nll_base82[fit_mask82].mean(),
      "held-out:", nll_base82[~fit_mask82].mean())

print("\n=== FACTUAL GROUP FEATURE BASELINES ===")
group_means82 = np.zeros((load_original(
    "embed_transition.value_bin.weight").shape[0], 192), np.float64)
group_counts82 = np.zeros(len(group_means82), dtype=int)
hf_mean82 = hidden_fact80.mean(0).numpy()
for g in np.unique(transition_rows82):
    m = fit_mask82 & (transition_rows82 == g)
    group_counts82[g] = int(m.sum())
    if m.any():
        group_means82[g] = hf_mean82[m].mean(0)
for ev in np.unique(events74):
    code = code42[ev]
    groups = np.arange(code*17, (code+1)*17)
    known = groups[group_counts82[groups] > 0]
    for g in groups[group_counts82[groups] == 0]:
        group_means82[g] = group_means82[
            known[np.argmin(np.abs(known-g))]]
print("fit readings", int(fit_mask82.sum()),
      "held-out readings", int((~fit_mask82).sum()),
      "occupied register groups", int((group_counts82 > 0).sum()))
```