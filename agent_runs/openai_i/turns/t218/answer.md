The rate targets match the records, and changed doses produce measurable recurrent-input contrasts. Those contrasts are nonlinear, especially for insulin, so I won’t assume that doubling a feature doubles an effect. I’ll fit and cross-validate dose-innovation decoders across all five multipliers, checking both calibration and unwanted factual signals.

```python
fit194 = np.asarray(fit_mask168)[ix166].astype(bool)
hold194 = ~fit194
presence194 = np.asarray(presence168)[ix166]
rates194 = R193[ix166]
policy_pred194 = np.column_stack([
    policy168[d][1].predict(np.asarray(context))
    for d in ["insulin", "norepinephrine"]
])[ix166]
innov194 = (rates194 - policy_pred194) * presence194

Xf194 = base_E193.numpy().reshape(-1, 192).astype(np.float64)
Yf194 = np.repeat(innov194, 4, axis=0).astype(np.float64)
Mf194 = np.repeat(fit194, 4)
Fcov194 = Xf194[Mf194].T @ Xf194[Mf194] / Mf194.sum()
Fcross194 = Xf194[Mf194].T @ Yf194[Mf194] / Mf194.sum()
Dcov194 = np.zeros((192, 192), dtype=np.float64)
Dcross194 = np.zeros((192, 2), dtype=np.float64)
deltas194 = {}
targets194 = {}
for key194, E194 in ecore193.items():
    if key194 == (1.0, 1.0):
        continue
    delta194 = (E194 - base_E193).numpy()
    target194 = rates194 * (np.asarray(key194) - 1)
    deltas194[key194] = delta194
    targets194[key194] = target194
    xd194 = delta194[fit194].reshape(-1, 192).astype(np.float64)
    yd194 = np.repeat(target194[fit194], 4, axis=0)
    Dcov194 += xd194.T @ xd194 / len(xd194)
    Dcross194 += xd194.T @ yd194 / len(xd194)
Dcov194 /= len(deltas194)
Dcross194 /= len(deltas194)

decoders194 = {}
rows194 = []
for eta194 in [0.1, 1.0, 10.0, 100.0, 1000.0]:
    for ridge194 in [1e-5, 1e-3, 0.1]:
        coef194 = np.linalg.solve(
            Fcov194 + eta194 * Dcov194 + ridge194 * np.eye(192),
            Fcross194 + eta194 * Dcross194)
        decoders194[(eta194, ridge194)] = coef194.astype(np.float32)
        factual194 = (base_E193.numpy() @ coef194).mean(axis=1)
        result194 = {"eta": eta194, "ridge": ridge194}
        for col194, drug194 in enumerate(["ins", "ne"]):
            active194 = hold194 & (presence194[:, col194] > 0)
            inactive194 = hold194 & (presence194[:, col194] == 0)
            result194[drug194 + "_fact_rmse"] = np.sqrt(
                np.mean((factual194[hold194, col194] - innov194[hold194, col194]) ** 2))
            result194[drug194 + "_inactive_rms"] = np.sqrt(
                np.mean(factual194[inactive194, col194] ** 2))
            key194 = (2.0, 1.0) if col194 == 0 else (1.0, 2.0)
            pred194 = (deltas194[key194] @ coef194).mean(axis=1)[:, col194]
            truth194 = targets194[key194][:, col194]
            result194[drug194 + "_x2_slope"] = (
                pred194[active194] @ truth194[active194]
                / (truth194[active194] @ truth194[active194]))
        rows194.append(result194)
print(pd.DataFrame(rows194).round(4).to_string(index=False))
print("Held factual innovation RMS:",
      np.sqrt(np.mean(innov194[hold194] ** 2, axis=0)))
print("Mean slot deviation:",
      float((base_E193 - base_E193.mean(dim=1, keepdim=True)).square().mean().sqrt()))
with open("/work/notes.md", "a") as f194:
    f194.write("\n## Python 194 — held-stay recurrent dose decoders\n"
               "- Fitted closed-form linear decoders of rate minus pre-dose policy prediction.\n"
               "- Augmented fitting with paired dose contrasts; stay-separated held checks include inactive-bundle leakage.\n"
               "- No weights changed; a decoder alone is not a repair or causal validation.\n")
```