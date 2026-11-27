The graft is a gated difference between two GELU units. Shifting only the contrast unit’s intercept may correct factual bias while retaining dose sensitivity, but that needs direct testing. I’ll first measure bias in the actual GPU predictions and recover the norepinephrine estimates. A consistent factual bias would support calibration; mixed signs would argue against it.

```python
import pandas as pd, numpy as np
from pathlib import Path

audit_parts = []
for k in (27, 28, 30, 31):
    root = Path(f"/work/fwd_{k}")
    a = pd.read_csv(root / "factual_original.csv")
    b = pd.read_csv(root / "factual_edited.csv")
    m = a.merge(b, on=["stay_id", "hour", "event", "observed"],
                suffixes=("_original", "_edited"), validate="one_to_one")
    m["batch"] = k
    audit_parts.append(m)
gpu400 = pd.concat(audit_parts, ignore_index=True)

tuning_records_recovery = pd.read_csv("/data/tuning.csv")
for drug in ("insulin", "norepinephrine"):
    d = tuning_records_recovery.loc[
        tuning_records_recovery.event.eq(drug), ["stay_id", "hour", "rate"]].copy()
    d["hour"] = np.floor(d["hour"]) + 1
    d = d.groupby(["stay_id", "hour"], as_index=False)["rate"].sum()
    d = d.rename(columns={"rate": "prior_" + drug})
    gpu400 = gpu400.merge(d, on=["stay_id", "hour"], how="left")
    gpu400["prior_" + drug] = gpu400["prior_" + drug].fillna(0)

gpu400["delta_nll"] = gpu400.nll_edited - gpu400.nll_original
gpu400["bias_original"] = gpu400.predicted_mean_original - gpu400.observed
gpu400["bias_edited"] = gpu400.predicted_mean_edited - gpu400.observed
gpu400["insulin_active"] = gpu400.prior_insulin.gt(0)

print("=== GPU FACTUAL AUDIT, ALL 400 ===")
print(gpu400.groupby("event").agg(
    n=("delta_nll", "size"),
    delta_nll=("delta_nll", "mean"),
    original_bias=("bias_original", "mean"),
    edited_bias=("bias_edited", "mean")).round(5).to_string())
print("\n=== FINGER: ACTIVITY AND BATCH ===")
finger_gpu400 = gpu400[gpu400.event.eq("glucose_finger")].copy()
print(finger_gpu400.groupby(["batch", "insulin_active"]).agg(
    n=("delta_nll", "size"),
    delta_nll=("delta_nll", "mean"),
    original_bias=("bias_original", "mean"),
    edited_bias=("bias_edited", "mean"),
    mean_dose=("prior_insulin", "mean")).round(5).to_string())

print("\n=== NE DML CACHE STRUCTURE ===")
for k, v in ne_dml_cache.items():
    print("lags", k)
    if isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, (float, int, str, bool, np.floating)):
                print(" ", kk, vv)
            else:
                print(" ", kk, getattr(vv, "shape", type(vv).__name__))
    else:
        print(type(v).__name__)

print("\n=== NOTEBOOK SECTIONS ABOUT IDENTIFICATION ===")
text = Path("/work/notes.md").read_text()
secs = re.split(r"(?m)(?=^## )", text)
print("Notebook characters:", len(text), "sections:", len(secs))
selected = [s for s in secs if any(w in s.lower()
            for w in ("instrument", "fuzzy", "negative control", "rd beta",
                      "ne flexible", "sensor disagreement"))]
for s in selected[:4]:
    print(s[:650])
for s in selected[-2:]:
    if s not in selected[:4]:
        print(s[:650])
```
