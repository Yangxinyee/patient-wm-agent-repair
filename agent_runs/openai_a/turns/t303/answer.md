The factual cost is concentrated in insulin-active finger readings; other readings are essentially unchanged. Their average bias is small, so a single offset probably cannot recover much NLL margin. I’ll check for systematic error by the previous glucose bin. Consistent errors across batches would support a targeted calibration; inconsistent errors would not.

```python
import pandas as pd, numpy as np

finger_gpu400 = finger_gpu400.sort_values(["stay_id", "hour"]).copy()
finger_gpu400["last_observed"] = (
    finger_gpu400.groupby("stay_id")["observed"].shift())
finger_gpu400["last_group"] = pd.qcut(
    finger_gpu400.last_observed, 8, duplicates="drop")

print("=== FINGER BIAS BY PREVIOUS READING ===")
print(finger_gpu400.groupby("last_group", observed=True).agg(
    n=("observed", "size"),
    original_bias=("bias_original", "mean"),
    edited_bias=("bias_edited", "mean"),
    delta_nll=("delta_nll", "mean"),
    active=("insulin_active", "mean")).round(4).to_string())
print("\n=== BATCH CHECK OF PREVIOUS-READING BIAS ===")
print(finger_gpu400.pivot_table(
    index="last_group", columns="batch", values="bias_edited",
    observed=True).round(3).to_string())

print("\n=== CACHE SIZES ===")
for n in ("sc_fact", "sc_syn"):
    obj = globals().get(n)
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(n, k, {kk: getattr(vv, "shape", type(vv).__name__)
                          for kk, vv in v.items()
                          if kk in ("x", "logits", "meta", "centres", "gate")})
print("\n=== TRANSITION TENSORS ===")
print([(n, SHAPES[n]) for n in NAMES if "embed_transition.value" in n])
print("VOCAB keys:", list(VOCAB))
for k in VOCAB:
    if "edge" in k or k == "codes":
        v = VOCAB[k]
        print(k, type(v).__name__,
              list(v)[:12] if isinstance(v, dict) else
              np.asarray(v).shape)

print("\n=== SHORT IDENTIFICATION SECTIONS ===")
text = Path("/work/notes.md").read_text()
secs = re.split(r"(?m)(?=^## )", text)
for s in secs:
    title = s.splitlines()[0] if s.splitlines() else ""
    if any(q in title.lower() for q in ("threshold", "rd ", "flexible", "robust", "policy")):
        if "trial" not in title.lower():
            print(s[:600])
```
