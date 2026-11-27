Independent factual likelihood improved by 0.00083 nats, comfortably within the limit. The remaining issue is causal calibration, especially the rollout asymmetry. I’ll inspect the retained lag estimates and per-hour rollout shifts to distinguish a weak response from saturation or feedback; consistent lag evidence would justify a sustained target.

```python
import pandas as pd, numpy as np
print("--- Retained lag estimates ---")
for name131 in ["provisional_kernels", "kernel_models88", "dml_results",
                "dml_residuals", "dml_models"]:
    obj131 = globals().get(name131)
    print("\n", name131)
    if isinstance(obj131, pd.DataFrame):
        print(obj131.to_string(index=False))
    elif isinstance(obj131, dict):
        for key131, val131 in obj131.items():
            if isinstance(val131, dict):
                print(key131, {k: (np.round(v, 5).tolist() if isinstance(v, np.ndarray) and v.size <= 30
                                  else getattr(v, "shape", type(v).__name__))
                               for k, v in val131.items()})
            elif isinstance(val131, np.ndarray) and val131.size <= 30:
                print(key131, np.round(val131, 5))
            else:
                print(key131, getattr(val131, "shape", type(val131).__name__))

print("\n--- Hourly panels ---")
for name131, obj131 in list(globals().items()):
    if isinstance(obj131, pd.DataFrame) and len(obj131) > 500000:
        print(name131, obj131.shape, list(obj131.columns)[:45])

dose_original131 = pd.read_csv("/work/fwd_8/dose_edit_original.csv")
dose_edited131 = pd.read_csv("/work/fwd_8/dose_edit_edited.csv")
print("\n--- Dose rollout file ---")
print(dose_edited131.columns.tolist())
print(dose_edited131.head(8).to_string(index=False))
print("\n--- Schema ---")
print(open("/data/README.md").read()[:2400])
with open("/work/notes.md", "a") as f:
    f.write("\n## Forward 7–8 / Run 131\nScale-0.4 edit tuning rollouts: insulin x2 -1.712, half +2.202; NE x2 +0.095, half -0.299. Asymmetry remains unexplained. Independent tuning stays 100–199 factual NLL: original 1.59931, edited 1.59848 (delta -0.00083). Recovering lag estimates and rollout detail; these predictions are not truth.\n")
```