The correction uses a shared gate in two head neurons, so an activity offset can suppress it without changing its fitted dose direction. I will check where scale 7 incurred its factual loss and screen gated variants. Substantial loss during inactive hours would support this approach; loss confined to active hours would limit its benefit.

```python
import pandas as pd

w_ins = np.asarray(r3_candidates[(0.3, 1.0)]["w"], dtype=np.float32)
b_ins = float(r3_candidates[(0.3, 1.0)]["b"])
activity_fit = _activity_fits[(0.003, 100.0)]
print("Activity fit keys:", list(activity_fit.keys()))
activity_w = np.asarray(activity_fit["w"], dtype=np.float32)
activity_b = float(activity_fit["b"])
print("Graft nodes:", sc_nodes, "tensor order:", sc_names)

check6 = make_selective_contrast_graft(w_ins, b_ins, 6.0, 30.0)
print("Fallback reconstruction errors:", {
    name: float(np.max(np.abs(np.asarray(check6[name]) - as_numpy(load(name)))))
    for name in check6
})

print("\n=== ACTUAL SCALE-7 GPU COST ===")
for request in [24, 26]:
    original_df = pd.read_csv(f"/work/fwd_{request}/factual_original.csv")
    edit_df = pd.read_csv(f"/work/fwd_{request}/factual_edited.csv")
    print("Request", request, "columns:", list(original_df.columns))
    if request == 24:
        print(original_df.head(2).to_string(index=False))
    nll_cols = [c for c in original_df.columns if "nll" in c.lower()]
    if len(nll_cols) == 1 and {"stay_id", "hour", "event"}.issubset(original_df.columns):
        nll_col = nll_cols[0]
        assert original_df[["stay_id", "hour", "event"]].equals(
            edit_df[["stay_id", "hour", "event"]])
        costs = original_df[["stay_id", "hour", "event"]].copy()
        costs["delta"] = edit_df[nll_col].to_numpy() - original_df[nll_col].to_numpy()
        costs["drug_hour"] = np.floor(costs["hour"] + 1e-5).astype(int) - 1
        drugs = _gate_tune_drugs[["stay_id", "hour", "insulin"]].copy()
        drugs["drug_hour"] = np.floor(drugs["hour"] + 1e-5).astype(int)
        drugs = drugs.groupby(["stay_id", "drug_hour"], as_index=False)["insulin"].sum()
        costs = costs.merge(drugs.drop(columns="hour", errors="ignore"),
                            on=["stay_id", "drug_hour"], how="left", validate="many_to_one")
        costs["active"] = costs["insulin"].fillna(0) > 0
        print(costs.groupby(["event", "active"])["delta"].agg(["count", "mean"]).to_string())

print("\n=== CPU GATED SCREEN: DESIGN EVIDENCE ONLY ===")
rows_gate_trial = []
for strength in [120.0, 180.0, 240.0]:
    for scale in [6.0, 7.0, 8.0]:
        row = {"activity_strength": strength, "scale": scale}
        for label, cache0 in sc_fact.items():
            gated_cache = selective_cache(cache0, 30.0)
            score_activity = cache0["x"] @ activity_w + activity_b
            gated_cache["gate"] = gated_cache["gate"] + strength * (score_activity - 0.05)
            losses_new = np.asarray(cache_losses(gated_cache, w_ins, b_ins, scale))
            losses_old = np.asarray(cache_losses(selective_cache(cache0, 30.0), w_ins, b_ins, 6.0))
            row[label + "_vs6"] = float(np.mean(losses_new - losses_old))
        rows_gate_trial.append(row)
print(pd.DataFrame(rows_gate_trial).to_string(index=False))
```
