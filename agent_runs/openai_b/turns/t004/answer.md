The raw associations are large, but adjusting for the current reading nearly eliminates the insulin association. The remaining norepinephrine association could still reflect treatment selection. I’ll examine dose variation at identical readings and around insulin thresholds; consistent changes in the next reading at those thresholds would support a genuine effect.

```python
for drug, primary, secondary in [
    ("insulin", "glucose_finger", "glucose_lab"),
    ("norepinephrine", "map_arterial", "map_cuff")
]:
    p = panel.loc[panel.index.get_level_values("hour").between(0, 23)
                  if hasattr(panel.index.get_level_values("hour"), "between")
                  else ((panel.index.get_level_values("hour") >= 0) &
                        (panel.index.get_level_values("hour") <= 23))]
    active = p[drug] > 0
    print("\n===", drug, "policy diagnostics ===")
    print("Active records:", int(active.sum()),
          "primary missing:", round(p.loc[active, primary].isna().mean(),4),
          "both sensors missing:",
          round(p.loc[active, [primary,secondary]].isna().all(axis=1).mean(),4))
    print("Rate persistence:",
          round(p.loc[active, [drug,drug+"_lag"]].corr().iloc[0,1],4))
    q = p.dropna(subset=[primary,primary+"_next"]).copy()
    q["change"] = q[primary+"_next"] - q[primary]
    if drug == "insulin":
        vals = sorted(set(v for c in [150,180,220,260,300]
                          for v in range(c-2,c+3)))
    else:
        vals = list(range(63,83))
    print(q[q[primary].isin(vals)].groupby(primary).agg(
        n=(drug,"size"), rate=(drug,"mean"), rate_sd=(drug,"std"),
        change=("change","mean")
    ).round(4).to_string())

    # Saturated adjustment for the exact current primary reading.
    y = q[primary+"_next"].to_numpy()
    r = q[drug].to_numpy()
    means = q.groupby(primary)[[primary+"_next",drug]].transform("mean")
    yr = y - means[primary+"_next"].to_numpy()
    rr = r - means[drug].to_numpy()
    beta = rr @ yr / (rr @ rr)
    scores = pd.Series(rr*(yr-beta*rr), index=q.index).groupby(level=0).sum()
    se = np.sqrt(np.sum(scores.to_numpy()**2)) / (rr @ rr)
    print("Exact-current-reading FE slope:", round(beta,4),
          "stay-cluster SE:", round(se,4))

    # Richer pre-treatment controls, still an association diagnostic.
    all_p = p.copy()
    for e in measurement_events:
        filled = panel.groupby(level=0)[e].ffill()
        all_p[e+"_history"] = filled.reindex(p.index)
        all_p[e+"_previous"] = filled.groupby(level=0).shift(1).reindex(p.index)
    q2 = all_p.dropna(subset=[primary,primary+"_next"]).copy()
    Xparts = [q2[drug].to_numpy(), q2[drug+"_lag"].fillna(0).to_numpy()]
    labels = [drug, drug+"_lag"]
    for e in measurement_events:
        for suffix in ["_history","_previous"]:
            v = q2[e+suffix]
            missing = v.isna().to_numpy().astype(float)
            x = (v.fillna(v.median()).to_numpy()-v.median())/(v.std()+1e-8)
            Xparts += [x, x*x, x*x*x, missing]
            labels += [e+suffix, e+suffix+"^2", e+suffix+"^3", e+suffix+"_missing"]
    h = q2.index.get_level_values("hour").to_numpy()/24.
    Xparts += [h,h*h,q2[drug_events[drug != "insulin"]].to_numpy()]
    labels += ["hour","hour^2","other_drug"]
    fit = LinearRegression().fit(np.column_stack(Xparts),
                                q2[primary+"_next"].to_numpy())
    print("Rich history-adjusted slope:", round(fit.coef_[0],4),
          "lag-dose slope:", round(fit.coef_[1],4),
          "n:", len(q2))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 3: preliminary associations\n"
        "Naive next-reading slopes: insulin +24.48 per unit/hour, "
        "norepinephrine -35.58 per mcg/kg/min. Current-reading cubic "
        "adjustment gives +0.166 and -2.415 respectively; not causal. "
        "Insulin policy has large steps near glucose 150,180,220,260,300 "
        "with within-bucket rate variation. Investigating thresholds, "
        "sensor information and dose persistence before concluding.\n"
    )
```