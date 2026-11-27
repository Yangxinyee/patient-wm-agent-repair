The state export contains the information needed for probes, but it does not yet show which directions encode dose. Before editing, I’ll estimate how long treatment effects persist, separating physiological persistence from later clinician dose changes. A consistent impulse-response model would support rollout targets across dose multipliers and start times; poor fit would refute that simplification.

```python
sA = np.load("/work/fwd_3/states_original.npz")
fA = pd.read_csv("/work/fwd_2/factual_original.csv")
dA = pd.read_csv("/work/fwd_1/dose_edit_original.csv")
print("=== FORWARD FILE SCHEMAS ===")
print("states:", {k: sA[k].shape for k in sA.files})
print("factual columns:", fA.columns.tolist())
print(fA.head(3).to_string(index=False))
print("dose columns:", dA.columns.tolist())
print(dA.head(3).to_string(index=False))
print("state bundle fractions:",
      np.unique(np.round(sA["hour"] % 1, 6), return_counts=True))

import torch
torch.set_num_threads(8)
torch.set_grad_enabled(False)
W0 = {n: torch.as_tensor(load_original(n)).float() for n in NAMES}
print("\n=== EMBEDDING SCALE ===")
print("numeric column norms:", torch.linalg.vector_norm(W0["embed.numeric.weight"], dim=0).numpy())
for n in ["embed.bins.value.weight", "embed.bins.dose.weight", "embed.bins.rate.weight",
          "encode.norm.weight", "encode_action.norm.weight"]:
    w = W0[n]
    print(n, "norm", round(float(w.norm()),4),
          "mean row norm", round(float(w.norm(dim=-1).mean()) if w.ndim>1 else float(w.abs().mean()),4))

print("\n=== LOCAL PROJECTIONS, HELD-OUT STAYS ===")
lp = {}
for horizon in range(1, 9):
    for target, series in [
        ("G", G), ("M", M),
        ("I", panel.insulin), ("N", panel.norepinephrine)
    ]:
        lag = horizon if target in ["G", "M"] else horizon-1
        if lag == 0:
            coef = np.array([1., 0.]) if target == "I" else np.array([0., 1.])
            se = np.zeros(2)
            n = len(testsid)
        elif horizon == 1 and target in ["G", "M"]:
            rr = residuals[target+"_next"]
            b, ss, n = clustered_ols(Ar, rr, testsid)
            coef, se = b[1:], ss[1:]
        else:
            values = series.groupby(level=0).shift(-lag)
            validfit = fitmask & (hours+lag<=24) & values.notna().to_numpy()
            nu = HistGradientBoostingRegressor(
                max_iter=120, max_leaf_nodes=31, min_samples_leaf=100,
                l2_regularization=5., learning_rate=.08,
                early_stopping=False, random_state=17
            )
            nu.fit(X.loc[validfit], values.loc[validfit])
            rr = values.loc[testmask].to_numpy()-nu.predict(X.loc[testmask])
            rr[hours[testmask]+lag>24] = np.nan
            b, ss, n = clustered_ols(Ar, rr, testsid)
            coef, se = b[1:], ss[1:]
        lp[(target, lag)] = {"coef": coef, "se": se, "n": n}
    print("h", horizon,
          "G response to I:", round(lp[("G",horizon)]["coef"][0],4),
          "M response to N:", round(lp[("M",horizon)]["coef"][1],4),
          "future I response:", round(lp[("I",horizon-1)]["coef"][0],4),
          "future N response:", round(lp[("N",horizon-1)]["coef"][1],4))

dyn_estimates = {}
for outcome, drug, j in [("G","I",0),("M","N",1)]:
    response = np.array([0.] + [lp[(outcome,k)]["coef"][j] for k in range(1,9)])
    action = np.array([lp[(drug,k)]["coef"][j] for k in range(8)])
    design = np.column_stack([response[:-1], action])
    coef = np.linalg.lstsq(design, response[1:], rcond=None)[0]
    beta_one = response[1]
    phi_fixed = np.dot(response[:-1], response[1:]-beta_one*action)/np.dot(response[:-1],response[:-1])
    dyn_estimates[outcome] = {"phi": float(coef[0]), "beta": float(coef[1]),
                             "beta_one": float(beta_one), "phi_fixed": float(phi_fixed)}
    print("\n", outcome, "joint [persistence, rate effect]:", coef.round(5),
          "one-step rate effect:", round(beta_one,5),
          "persistence holding one-step effect:", round(phi_fixed,5))
    print("response:", response.round(3),
          "AR residual:", (response[1:]-design@coef).round(3))

def mechanistic_shifts(p, ids, drug, readout, phi, beta, start=12, multiplier=2.):
    sub = p.loc[(ids, slice(0,24)), :]
    shifts, rows = [], []
    for stay, group in sub.groupby(level=0):
        group = group.droplevel(0)
        effect = 0.
        reached = False
        for h in range(25):
            if h >= 13 and reached and pd.notna(group.loc[h,readout]):
                shifts.append(effect)
                rows.append((int(stay),h,effect))
            d = float(group.loc[h,drug])
            if h >= start and d>0:
                reached = True
            effect = phi*effect + beta*(multiplier-1)*d*(h>=start)
    return float(np.mean(shifts)), len(shifts), rows

print("\n=== PROVISIONAL FIXED-DOSE ROLLOUT TARGETS, GROUP A ===")
for outcome, drug, readout in [
    ("G","insulin","glucose_finger"),("M","norepinephrine","map_arterial")
]:
    est = dyn_estimates[outcome]
    for mult in [0., .5, 1.5, 2., 3.]:
        mean,n,_ = mechanistic_shifts(panel_tune,eval_ids_A,drug,readout,
                                     est["phi_fixed"],est["beta_one"],12,mult)
        print(drug, "x", mult, "mean", round(mean,4), "readout rows",n)

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Original baseline / persistence analysis (forwards 1–3; run 5)\n"
        "- Tuning A: original value NLL 1.63933; allowable ceiling 1.64433.\n"
        "- Original A rollout shifts: insulin x2 -0.454, half +0.098; NE x2 +0.150, half -0.232.\n"
        "- Original states A saved in /work/fwd_3/states_original.npz (4,141 rows).\n"
        "- Used residualized treatment innovations to estimate future outcome and future dose responses.\n"
        "- Deconvolved clinician feedback with an AR(1) response model; this requires causal-innovation and "
        "stable-linear-response assumptions, to be checked further.\n"
    )
    for key, val in dyn_estimates.items():
        f.write(f"- {key} dynamic estimates: {val}.\n")
    f.write("- Current counts: 5 Python runs, 3 forwards; no edits.\n")
```
