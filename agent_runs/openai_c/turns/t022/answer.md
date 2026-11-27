The metadata check succeeded, but the regression stopped because the CSV calls the recorded value `observed`, not `value`. No weights changed. The captured dose responses are small; I’ll correct the column name and finish the sensitivity estimates before deciding whether to amplify them.

```python
head_surrogates = {}
direction_diagnostics = {}

for event, drug in [("glucose_finger", "insulin"),
                    ("map_arterial", "norepinephrine")]:
    zf = pred_train[pred_train.event == event].copy()
    zf = zf.sort_values(["stay_id", "hour"])
    zf["last_value"] = zf.groupby("stay_id")["observed"].shift(1)
    zf["last_hour"] = zf.groupby("stay_id")["hour"].shift(1)
    zf = zf[state_for_reading[zf.index] >= 0]
    O = out_states[state_for_reading[zf.index]]
    lastv = zf.last_value.fillna(zf.observed.median()).to_numpy()
    lastv = (lastv - np.mean(lastv)) / np.std(lastv)
    age = (zf.hour - zf.last_hour).fillna(25).to_numpy()
    spline = SplineTransformer(n_knots=6, degree=3, include_bias=False)
    R = np.column_stack([
        spline.fit_transform(lastv[:, None]),
        np.log1p(age), np.log1p(age)**2,
        zf.last_value.isna().to_numpy().astype(float),
        lastv * np.log1p(age)
    ])
    H = np.column_stack([O, R])
    Y = zf[mean_col].to_numpy()
    G = zf.stay_id.to_numpy()
    cv_scores = {}
    for penalty in [0.1, 1.0, 10.0, 100.0]:
        heldout = np.zeros(len(Y))
        for it, iv in GroupKFold(4).split(H, Y, G):
            rr = Ridge(alpha=penalty).fit(H[it], Y[it])
            heldout[iv] = rr.predict(H[iv])
        cv_scores[penalty] = float(np.sqrt(np.mean((heldout - Y)**2)))
    penalty = min(cv_scores, key=cv_scores.get)
    rr = Ridge(alpha=penalty).fit(H, Y)
    w = rr.coef_[:192]
    head_surrogates[event] = {
        "w": w, "ridge": rr, "O": O, "R": R, "frame": zf,
        "cv_rmse": cv_scores, "penalty": penalty
    }
    dd = {}
    for test_drug, j in [("insulin", 0), ("norepinephrine", 1)]:
        delta_o = dose_state_coeff[j, :192]
        dd[test_drug] = {"linear_response": float(w @ delta_o)}
        for label, vec in [
            ("raw", dose_directions[test_drug]["out"]),
            ("white", whitened_dose_directions[test_drug])
        ]:
            dd[test_drug][label] = {
                "dose_projection": float(delta_o @ vec),
                "readout_projection": float(w @ vec),
                "rank1_response_per_alpha": float((delta_o @ vec) * (w @ vec)),
                "factual_projection_mean": float(np.mean(O @ vec)),
                "factual_projection_sd": float(np.std(O @ vec))
            }
    direction_diagnostics[event] = dd
    print(event, "n =", len(Y), "CV RMSE:", cv_scores)
    print(json.dumps(dd, indent=2))

print("Dose-direction cosine:",
      float(whitened_dose_directions["insulin"] @
            whitened_dose_directions["norepinephrine"]))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Runs 13–14 / forward 8: state-to-reading surrogates\n"
        "- Run 13 stopped on CSV column mismatch (observed versus value); "
        "run 14 corrected it. No weight edits.\n"
        "- Original training factual NLL: 1.63841 on the 100 captured stays.\n"
        f"- Stay-held-out RMSE: "
        f"{ {k: v['cv_rmse'] for k, v in head_surrogates.items()} }\n"
        f"- Dose/readout diagnostics: {direction_diagnostics}\n"
        "- These regressions approximate head sensitivity; forward tests remain decisive.\n"
        "- Python runs: 14; forward passes: 8.\n"
    )
```