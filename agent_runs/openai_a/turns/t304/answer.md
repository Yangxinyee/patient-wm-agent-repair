Previous-reading errors have a consistent shape: low glucose is overpredicted, while several higher ranges are underpredicted. That suggests a small transition calibration may help factual fit and persistence, though it does not establish a drug effect. I’ll estimate a two-parameter correction on one half of tuning and check the other half before editing.

```python
import numpy as np, pandas as pd

print("=== BIN DEFINITIONS ===")
finger_code_recovery = list(VOCAB["codes"]).index("glucose_finger")
print("finger code:", finger_code_recovery)
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
prob = softmax(cc["logits"]).mean(axis=0)
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
    f.write("\n## Actual-GPU calibration audit; Python 261, forwards 32/50\n"
            "- Full400 gated8 NLL cost is confined to insulin-active finger readings.\n"
            "- Low previous glucose is consistently overpredicted; some high bins are underpredicted.\n"
            "- Testing a two-parameter previous-reading calibration using first200 fit / last200 check.\n"
            "- Such calibration is not additional causal evidence; no weights changed yet.\n")
```
