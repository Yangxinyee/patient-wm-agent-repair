The original rollouts barely respond to either drug, and the current edit only modestly improves insulin. I’ll compare an isolated dose change with the record-derived lag pattern. This replay keeps later readings factual, so it is only a mechanism test: rapid loss of the response would confirm weak state persistence, not establish the counterfactual effect itself.

```python
import numpy as np, pandas as pd, torch, time
from pathlib import Path

weights159 = {n: torch.as_tensor(load(n), dtype=torch.float32) for n in NAMES}
baseline_x159, baseline_q159 = replay_factual125(weights159)
dataset159 = head_train_data_repair.copy()

def centres_for159(ds):
    table = VOCAB["value_bin_centres"]
    rows = []
    for ev, code in zip(ds["event"], ds["code"]):
        if isinstance(table, dict):
            key = next((k for k in [ev, str(int(code)), int(code)] if k in table), None)
            if key is None:
                raise KeyError(("centre table keys", list(table)[:10], ev, code))
            row = table[key]
        else:
            row = table[int(code)]
        row = np.asarray(row, dtype=float)
        if row.size == 16:
            row = np.r_[0.0, row]
        assert row.size == 17, (ev, row.shape)
        rows.append(row)
    return np.asarray(rows)

centres159 = centres_for159(dataset159)
_, baseline_p159 = head_features_fixed({"x": baseline_x159}, dataset159, draws=4, seed=15901)
baseline_mean159 = (baseline_p159 * centres159).sum(axis=1)
impulse_results159 = {}

for drug159, event159 in [("insulin", "glucose_finger"),
                         ("norepinephrine", "map_arterial")]:
    rr159 = repair_records[
        (repair_records.event == drug159) &
        (repair_records.hour >= 10) & (repair_records.hour <= 15)
    ].sort_values(["stay_id", "hour"]).copy()
    if len(rr159) == 0:
        continue

    ix159 = rr159.bundle_index.to_numpy(dtype=int)
    rate159 = rr159.rate.to_numpy(dtype=float)
    dose159 = rr159.dose.to_numpy(dtype=float)
    best159 = None
    for mode159, values159 in [("nan", np.full(len(rr159), np.nan)),
                               ("zero", np.zeros(len(rr159)))]:
        try:
            atoms159 = atom_exact65(drug159, values159, rate159, dose159)
            encoded159 = singleton_action_repair(atoms159)
            err159 = torch.mean((encoded159 - U65[ix159])**2, dim=1).sqrt().numpy()
            good159 = np.isfinite(err159) & (err159 < 1e-4)
            print(drug159, mode159, "matching singleton bundles",
                  int(good159.sum()), "of", len(rr159),
                  "median RMSE", float(np.nanmedian(err159)))
            if best159 is None or good159.sum() > best159[0].sum():
                best159 = (good159, mode159)
        except Exception as exc159:
            print(drug159, mode159, type(exc159).__name__, str(exc159)[:150])

    if best159 is None or not best159[0].any():
        continue
    chosen159 = rr159.iloc[np.flatnonzero(best159[0])].drop_duplicates("stay_id").head(40)
    ix159 = chosen159.bundle_index.to_numpy(dtype=int)
    rate159 = chosen159.rate.to_numpy(dtype=float)
    dose159 = chosen159.dose.to_numpy(dtype=float)
    values159 = (np.full(len(ix159), np.nan) if best159[1] == "nan"
                 else np.zeros(len(ix159)))
    increment159 = 1.0 if drug159 == "insulin" else 0.2
    ratio159 = dose159 / rate159
    atoms159 = atom_exact65(drug159, values159, rate159 + increment159,
                           dose159 + increment159 * ratio159)
    new_u159 = singleton_action_repair(atoms159)
    saved_u159 = U65.clone()
    try:
        U65[ix159] = new_u159
        impulse_x159, impulse_q159 = replay_factual125(weights159)
    finally:
        U65[:] = saved_u159

    _, impulse_p159 = head_features_fixed({"x": impulse_x159}, dataset159,
                                          draws=4, seed=15901)
    impulse_mean159 = (impulse_p159 * centres159).sum(axis=1)
    impulse_time159 = dict(zip(chosen159.stay_id, chosen159.hour))
    lag159 = dataset159.hour.to_numpy() - dataset159.stay_id.map(impulse_time159).to_numpy()
    shifted159 = pd.DataFrame({
        "stay_id": dataset159.stay_id, "event": dataset159.event,
        "lag": np.ceil(lag159 - 1e-4),
        "response_per_rate": (impulse_mean159 - baseline_mean159) / increment159
    })
    select159 = shifted159[
        (shifted159.event == event159) &
        (shifted159.lag >= 1) & (shifted159.lag <= 8)
    ]
    impulse_results159[drug159] = {
        "chosen": chosen159, "states": impulse_x159, "responses": shifted159
    }
    print(drug159, "n_stays", len(chosen159), "response by lag:",
          select159.groupby("lag").response_per_rate.agg(["mean", "count"]).round(4).to_dict("index"))
    print("record kernel:", np.round(kernels137[drug159][:8], 4))

Path("/work/notes.md").open("a").write(
    "\n## Run 159\nTested isolated dose increments with exact action-encoding parity checks "
    "and complete factual replay. Later readings/registers remain factual, so this is a state-persistence "
    "diagnostic rather than a free-rollout or causal-effect estimate. Results retained in impulse_results159. "
    "No weights changed.\n")
```
