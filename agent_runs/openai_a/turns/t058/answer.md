The policies predict recorded doses well: residual variation is about 0.37 units/hour for insulin and 0.076 for norepinephrine. That may allow a small factual-preserving correction, but it does not establish causality. I’ll test whether the reading head decodes those residuals on held-out stays and estimate any missing factual dose response.

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

def head_dose_data(meta, panel, ids, predicted_policy):
    stay_index = np.searchsorted(ids, meta.stay_id.to_numpy())
    hours = np.rint(meta.hour.to_numpy()).astype(int)
    assert np.allclose(hours, meta.hour.to_numpy())
    valid = (hours >= 1) & (hours <= 24)
    prior_hour = np.clip(hours - 1, 0, 23)
    doses = np.nan_to_num(
        panel[stay_index, prior_hour + 1][:, drug_cols], nan=0.0)
    expected = predicted_policy[stay_index, prior_hour]
    residual = doses - expected
    residual[~valid] = 0.0
    glucose = meta.event.isin(["glucose_finger", "glucose_lab"]).to_numpy()
    pressure = meta.event.isin(["map_arterial", "map_cuff"]).to_numpy()
    return dict(stay_index=stay_index, prior_hour=prior_hour, valid=valid,
                doses=doses, residual=residual, groups=[glucose, pressure])

hd_train = head_dose_data(head_meta_train, P, train_ids, policy_predictions_panel)
hd_tune = head_dose_data(head_meta_tune, TP, tuning_ids, tpolicy_predictions_panel)

hidden_location = repair_train["hidden"].mean(axis=0)
hidden_scale = np.maximum(repair_train["hidden"].std(axis=0), 0.05)
HX = (repair_train["hidden"] - hidden_location) / hidden_scale
HTX = (repair_tune["hidden"] - hidden_location) / hidden_scale

dose_decoders = {}
decoder_diagnostics = {}
for j, drug in enumerate(("insulin", "norepinephrine")):
    group = hd_train["groups"][j]
    tgroup = hd_tune["groups"][j]
    c = np.maximum(repair_train["covariance"], 1.0)
    tc = np.maximum(repair_tune["covariance"], 1.0)
    sign = -1.0 if drug == "insulin" else 1.0
    target = sign * hd_train["residual"][:, j] * group / c
    ttarget = sign * hd_tune["residual"][:, j] * tgroup / tc
    scores = []
    for alpha in (1.0, 10.0, 100.0, 1000.0):
        model = Ridge(alpha=alpha).fit(HX, target)
        pred = model.predict(HTX)
        recovered = sign * pred * tc
        mask = tgroup & hd_tune["valid"]
        scores.append((float(np.mean((pred - ttarget)**2)), alpha, model))
        print(drug, "ridge", alpha,
              "residual R2", round(r2_score(hd_tune["residual"][mask, j],
                                           recovered[mask]), 4),
              "cross-event unit-effect RMS",
              round(float(np.sqrt(np.mean((pred[~tgroup] * tc[~tgroup])**2))), 4))
    _, alpha, model = min(scores, key=lambda item: item[0])
    raw_w = model.coef_ / hidden_scale
    raw_b = float(model.intercept_ - np.dot(raw_w, hidden_location))
    dose_decoders[drug] = dict(w=raw_w, b=raw_b, alpha=alpha)
    decoder_diagnostics[drug] = dict(
        chosen_alpha=alpha,
        train_target_sd=float(target.std()),
        tuning_target_sd=float(ttarget.std()),
        tuning_pred_sd=float(model.predict(HTX).std()))
    print("CHOSEN", drug, decoder_diagnostics[drug])

print("\n=== ORIGINAL FACTUAL DOSE RESPONSE ON TUNING ===")
for j, (drug, event) in enumerate(
        (("insulin", "glucose_finger"), ("norepinephrine", "map_arterial"))):
    mask = (head_meta_tune.event.to_numpy() == event) & hd_tune["valid"]
    rows = hd_tune["stay_index"][mask] * 24 + hd_tune["prior_hour"][mask]
    controls = tpolicy_X[rows].astype(np.float64)
    med = np.nanmedian(controls, axis=0)
    controls = np.where(np.isfinite(controls), controls, med)
    controls = (controls - controls.mean(0)) / np.maximum(controls.std(0), 1e-5)
    r = hd_tune["residual"][mask, j]
    design = np.column_stack([np.ones(len(r)), r, controls])
    y = head_meta_tune.value.to_numpy()[mask]
    p = head_gpu_mean[mask]
    coef = np.linalg.lstsq(design, np.column_stack([y, p, y - p]), rcond=1e-8)[0]
    print(drug, "rows", len(r), "observed slope", round(float(coef[1, 0]), 3),
          "model-mean slope", round(float(coef[1, 1]), 3),
          "missing-response slope", round(float(coef[1, 2]), 3))

print("\nCovariance quantiles:",
      np.quantile(repair_tune["covariance"], [0.05, 0.5, 0.95]).round(3))
print("Edits:", edited())
with open("/work/notes.md", "a") as f:
    f.write("\n## Residual-dose decoder; Python 52, forward 4\n"
            f"- Held-out feature-decoder diagnostics: {decoder_diagnostics}\n"
            "- Decoders use final value-head hidden features, with zero targets on other event types.\n"
            "- This is an algebraic low-rank edit candidate; no model gradients or optimizer used.\n")
```
