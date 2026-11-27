The training baseline has similar factual fit to the tuning baseline. I’ll fit small, stay-held-out regressions linking the captured states to the model’s predictions, accounting for the last reading. A direction with measurable dose sensitivity but little factual variation could support a selective edit; poor regression accuracy or unstable sensitivity would argue against it.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

pred_train = pd.read_csv("/work/fwd_8/factual_original.csv")
print("Factual columns:", pred_train.columns.tolist())
print(pred_train.head(2).to_string(index=False))
print("Panel:", panel.index.names, panel.columns.tolist())
print("Controls:", controls.columns.tolist())
print("State samples:", list(zip(ss[:12], sh[:12],
    states_original["has_insulin"][:12],
    states_original["has_norepinephrine"][:12])))
print("Numeric tables:", str(VOCAB["numeric"])[:2300])

mean_col = next(c for c in pred_train.columns if "mean" in c.lower())
state_for_reading = np.full(len(pred_train), -1, dtype=int)
for stay, rows in pred_train.groupby("stay_id").groups.items():
    rows = np.asarray(list(rows))
    si = np.flatnonzero(ss == stay)
    order = np.argsort(sh[si], kind="stable")
    si = si[order]
    jj = np.searchsorted(sh[si], pred_train.loc[rows, "hour"], side="left") - 1
    valid = jj >= 0
    state_for_reading[rows[valid]] = si[jj[valid]]

out_states = sx[:, :192].astype(float)
head_surrogates = {}
direction_diagnostics = {}
all_read_out = out_states[state_for_reading[state_for_reading >= 0]]
C_out = all_read_out.T @ all_read_out / len(all_read_out)
ridge_cov = 0.03 * np.trace(C_out) / 192
whitened_dose_directions = {}

for drug, j in [("insulin", 0), ("norepinephrine", 1)]:
    raw = dose_state_coeff[j, :192].astype(float)
    white = np.linalg.solve(C_out + ridge_cov * np.eye(192), raw)
    white /= np.linalg.norm(white)
    whitened_dose_directions[drug] = white
    print(drug, "raw state response norm:", np.linalg.norm(raw),
          "raw/white cosine:", float(dose_directions[drug]["out"] @ white))

for event, drug in [("glucose_finger", "insulin"),
                    ("map_arterial", "norepinephrine")]:
    zf = pred_train[pred_train.event == event].copy()
    zf = zf.sort_values(["stay_id", "hour"])
    zf["last_value"] = zf.groupby("stay_id").value.shift(1)
    zf["last_hour"] = zf.groupby("stay_id").hour.shift(1)
    zf = zf[state_for_reading[zf.index] >= 0]
    O = out_states[state_for_reading[zf.index]]
    lastv = zf.last_value.fillna(zf.value.median()).to_numpy()
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
    print("\n", event, "n =", len(Y), "CV RMSE:", cv_scores)
    print(json.dumps(dd, indent=2))

with open("/work/notes.md", "a") as f:
    f.write(
        "\n## Run 13 / forward 8: state-to-reading surrogates\n"
        "- Original factual fit on 100 captured training stays: 1.63841 nats.\n"
        f"- Stay-held-out surrogate RMSE: "
        f"{ {k: v['cv_rmse'] for k, v in head_surrogates.items()} }\n"
        f"- Direction diagnostics: {direction_diagnostics}\n"
        "- No edits. Python runs: 13; forward passes: 8.\n"
    )
```