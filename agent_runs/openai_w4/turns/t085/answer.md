The reusable transition matches the exported output and recurrent state to numerical precision. I’ll now double and halve each drug in that transition, then compare the next-reading head features. This will show whether a head-only correction can recover the immediate effects; poorly distinguishable or unstable feature changes would argue against that approach.

```python
def intervention_bank74(drug, factor):
    rec = repair_records
    dm = rec["event"].eq(drug).to_numpy()
    am = rec["event"].isin(["insulin", "norepinephrine"]).to_numpy()
    atoms_cf = atoms65.clone()
    atoms_cf[dm] = atom_exact65(
        drug,
        rec.loc[dm, "value"].to_numpy(),
        rec.loc[dm, "rate"].to_numpy() * factor,
        rec.loc[dm, "dose"].to_numpy() * factor
    )
    u_cf = pooled_encoder58(
        atoms_cf[am], rec.loc[am, "bundle_index"].to_numpy(),
        counts_u58, "encode_action"
    )
    ix = np.unique(rec.loc[dm, "bundle_index"].to_numpy()).astype(int)
    b_cf = dict(repair_bank)
    b_cf["x"] = np.array(repair_bank["x"], copy=True)
    for j in range(0, len(ix), 192):
        ib = ix[j:j+192]
        b_cf["x"][ib] = transition73(ib, u_cf[ib]).numpy()
    rate_by_bundle = np.zeros(len(repair_bank["hour"]))
    np.add.at(rate_by_bundle, rec.loc[dm, "bundle_index"].to_numpy(),
              rec.loc[dm, "rate"].to_numpy())
    return b_cf, ix, rate_by_bundle

centres_by_event74 = {
    ev: np.asarray(fact_centres_repair)[
        np.flatnonzero(head_tuning_data_repair["event"].eq(ev).to_numpy())[0]]
    for ev in head_train_data_repair["event"].unique()
}
centres74 = np.stack([
    centres_by_event74[ev] for ev in head_train_data_repair["event"]
])
phi74, prob74 = head_features_fixed(
    repair_bank, head_train_data_repair, draws=8, seed=7401
)
phi74, prob74 = np.asarray(phi74), np.asarray(prob74)
mean74 = np.sum(prob74 * centres74, axis=1)
one_step74 = {}
for drug in ("insulin", "norepinephrine"):
    for factor in (0.5, 2.0):
        bc, ix, rate = intervention_bank74(drug, factor)
        ph, pr = head_features_fixed(
            bc, head_train_data_repair, draws=8, seed=7401
        )
        prev_idx = head_train_data_repair["prev"].to_numpy().astype(int)
        dr = rate[prev_idx] * (factor - 1)
        delta_mean = np.sum(np.asarray(pr) * centres74, axis=1) - mean74
        system = head_train_data_repair["event"].isin(
            ["glucose_finger", "glucose_lab"] if drug == "insulin"
            else ["map_arterial", "map_cuff"]
        ).to_numpy()
        eligible = (dr != 0) & system
        one_step74[(drug, factor)] = {
            "phi": np.asarray(ph), "prob": np.asarray(pr),
            "delta_rate": dr, "eligible": eligible,
            "delta_mean": delta_mean
        }
        print(drug, factor, "target readings", int(eligible.sum()),
              "baseline slope",
              float(np.dot(dr[eligible], delta_mean[eligible])
                    / np.dot(dr[eligible], dr[eligible])),
              "feature-difference RMS",
              float(np.sqrt(np.mean((np.asarray(ph)[eligible]
                                      - phi74[eligible])**2))))
print("Base feature/probability shapes:", phi74.shape, prob74.shape)
print("Current edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 69–74\nRecovered immediate estimates and six-lag kernels. Implemented reusable exact transition: out RMSE 1.4e-7, h RMSE 1.0e-5, posterior RMSE 5.9e-5. Constructed isolated half/double drug state changes and matched next-reading head features on 100 training stays. No weights edited; controlled feature changes are a repair design diagnostic, not causal evidence.\n")
```
