The run stopped before analysis because the vocabulary codes are structured entries, not plain strings. No weights changed. I’ll use the finger code already recorded in the cache and rerun the fit-and-check test. Agreement between the two halves would support the calibration; a sign reversal would refute it.

```python
import numpy as np, pandas as pd
from scipy.special import softmax as sp_softmax

print("=== BIN DEFINITIONS ===")
finger_code_recovery = int(sc_fact["tuning"]["meta"].loc[
    sc_fact["tuning"]["meta"].event.eq("glucose_finger"), "code"].iloc[0])
print("finger code:", finger_code_recovery)
print("code entry:", repr(VOCAB["codes"][finger_code_recovery])[:500])
print("numeric entry:", repr(VOCAB["numeric"][finger_code_recovery])[:1800])
print("finger centres:", VOCAB["value_bin_centres"][finger_code_recovery])

def fit_gpu_mean_calibration(frame):
    s = frame.dropna(subset=["last_observed"]).copy()
    X = np.column_stack([
        np.ones(len(s)), (s.last_observed.to_numpy() - 170.0) / 50.0])
    y = s.observed.to_numpy() - s.predicted_mean_edited.to_numpy()
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ coef
    scores = pd.DataFrame(X * resid[:, None]).groupby(
        s.stay_id.to_numpy()).sum().to_numpy()
    bread = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(bread @ (scores.T @ scores) @ bread))
    return coef, se

calib_fit_half = finger_gpu400[finger_gpu400.batch.isin([27, 28])]
calib_check_half = finger_gpu400[finger_gpu400.batch.isin([30, 31])]
for label, frame in (("fit-first200", calib_fit_half),
                     ("check-last200", calib_check_half),
                     ("all400", finger_gpu400)):
    coef, se = fit_gpu_mean_calibration(frame)
    print(label, "intercept / change per 50 mg/dL:", coef,
          "cluster SE:", se)

calib_coef_recovery, calib_se_recovery = fit_gpu_mean_calibration(calib_fit_half)
s = calib_check_half.dropna(subset=["last_observed"])
pred_correction = (calib_coef_recovery[0] +
                   calib_coef_recovery[1] * (s.last_observed.to_numpy() - 170) / 50)
e = s.observed.to_numpy() - s.predicted_mean_edited.to_numpy()
print("Held-out change in squared mean error:",
      float(np.mean((e - pred_correction)**2 - e**2)))

print("\n=== CPU PROBABILITY SLOPES, NOT ABSOLUTE CALIBRATION ===")
cc = sc_fact["tuning"]
prob = sp_softmax(cc["logits"], axis=-1).mean(axis=0)
cent = np.asarray(cc["centres"])
mr = (prob * sc_rank[None, :]).sum(axis=1)
mv = (prob * cent).sum(axis=1)
cov_v_rank = (prob * cent * sc_rank[None, :]).sum(axis=1) - mr * mv
cm = cc["meta"].copy()
cm["slope"] = cov_v_rank
print(cm[cm.event.eq("glucose_finger")].groupby("last_bin").agg(
    n=("slope", "size"), slope=("slope", "mean")).round(3).to_string())

print("\n=== NE RECORDED DOSE SCALE ===")
nr = tuning_records_recovery.loc[
    tuning_records_recovery.event.eq("norepinephrine"), "rate"]
print(nr.describe(percentiles=[.1, .5, .9]).to_string())

with open("/work/notes.md", "a") as f:
    f.write("\n## Actual-GPU calibration audit; Python 262, forwards 32/50\n"
            "- Python 261 failed before analysis or editing: vocabulary codes are structured entries.\n"
            "- Full400 gated8 NLL cost is confined to insulin-active finger readings.\n"
            "- Testing two-parameter previous-reading calibration with first200 fit / last200 check.\n"
            "- This is factual calibration, not additional causal evidence; no weights changed.\n")
```
