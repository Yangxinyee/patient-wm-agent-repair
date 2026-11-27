The candidate is staged; it changes only the value head on top of the previously factual-neutral dose wire. The offline results meet the loss limit, but they do not establish GPU performance. I’ll recover the exact audit stay IDs and sensor-level losses, then run the paired factual check.

```python
import pandas as pd
audit_ids236 = sorted(pd.read_csv("/data/tuning.csv", usecols=["stay_id"])
                     ["stay_id"].unique().tolist())[:100]
print("AUDIT STAY IDS:", json.dumps([int(x) for x in audit_ids236]))
nll235 = nll228(chosen235["Ftune"] @ T235 + offset234, B225)
delta235 = nll235 - base_nll_tune234
for ev in ["glucose_finger", "glucose_lab", "map_arterial", "map_cuff"]:
    use = np.asarray(events225) == ev
    print(ev, "readings", int(use.sum()),
          "offline NLL delta", round(float(delta235[use].mean()), 6))
print("Dose-feature factual ranges, tuning:",
      np.round(np.quantile(np.asarray(X225) @ rows235.T,
                           [0,.01,.5,.99,1], axis=(0,1)), 3))
```
