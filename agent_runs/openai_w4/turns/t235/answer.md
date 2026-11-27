Recorded-dose rollout baselines changed substantially, but their averages are not near the bin limits. Saturation alone therefore looks insufficient. I’ll compare those baselines with actual readings and inspect more regularized response directions. This will distinguish useful baseline correction from unstable feedback.

```python
import numpy as np
import pandas as pd
import inspect

actual205 = tuning204[tuning204["event"].isin(["glucose_finger", "map_arterial"])].copy()
actual205["hour"] = np.floor(actual205["hour"] + 1e-6).astype(int)
actual205 = actual205.groupby(["stay_id", "hour", "event"], as_index=False)["value"].mean()

print("MATCHED OBSERVED VERSUS GENERATED BASELINES")
for label, tab in rollout_tables204.items():
    for edit, event in [("insulin_x2", "glucose_finger"), ("ne_x2", "map_arterial")]:
        a = tab[tab["edit"] == edit].merge(
            actual205[actual205["event"] == event],
            on=["stay_id", "hour"], how="inner")
        print(label, event, {
            "matched_hours": len(a),
            "observed": float(a["value"].mean()),
            "generated": float(a["factual"].mean()),
            "bias": float((a["factual"] - a["value"]).mean()),
            "observed_by_hour": a.groupby("hour")["value"].mean().round(2).to_dict()
        })

print("\nACTION RATE DISTRIBUTIONS")
print(tuning204[tuning204["event"].isin(["insulin", "norepinephrine"])]
      .groupby("event")["rate"].describe().round(4).to_string())

print("\nNE DIRECTION ALTERNATIVES")
for ridge, direction in directions_ne187.items():
    d = as_np184(direction).reshape(-1)
    pred = np.asarray(J187) @ d
    current_d = as_np184(ne_direction194).reshape(-1)
    print(ridge, {
        "norm": float(np.linalg.norm(d)),
        "cosine_current": float(np.dot(d, current_d) /
                               (np.linalg.norm(d)*np.linalg.norm(current_d))),
        "J_response_quantiles": np.quantile(pred, [.05, .5, .95]).round(4).tolist()
    })

print("\nINSULIN CONNECTION TEST STRUCTURE")
for key, obj in connection_tests160.items():
    if isinstance(obj, dict):
        print(key, {
            str(k): {
                str(kk): list(vv.shape) if hasattr(vv, "shape") else vv
                for kk, vv in v.items()
                if hasattr(vv, "shape") or isinstance(vv, (float, int, str))
            } if isinstance(v, dict) else
            (list(v.shape) if hasattr(v, "shape") else str(type(v)))
            for k, v in obj.items()
        })
    else:
        print(key, list(obj.shape) if hasattr(obj, "shape") else str(type(obj)))

print("\nFUNCTION NAMES")
print(" ".join(sorted(n for n, v in dict(globals()).items() if inspect.isfunction(v))))
print("\nJACOBIAN DESIGN VARIABLE NAMES")
print(" ".join(sorted(n for n in globals()
                      if n.endswith(("160", "166", "187"))
                      and any(s in n.lower() for s in
                              ["j", "target", "response", "direction", "mask", "lag", "dose", "label"]))))

with open("/work/notes.md", "a") as f:
    f.write("\n\n## Run 205\nCompared recorded-dose generated baselines with matched actual tuning outcomes and inspected regularized response-direction alternatives. Mean generated values are not at global bin limits; large baseline shifts make recurrent feedback a plausible source of half/double asymmetry. No weights changed.\n")
```