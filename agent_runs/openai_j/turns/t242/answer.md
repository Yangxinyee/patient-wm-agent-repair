The unfinished candidate fails factual fit badly: tuning value NLL rises by about 0.047 nats per reading, so I’m rejecting it. The current edit remains installed. I’ll measure its dose-specific slopes, then try a minimal-change fit to average treatment contrasts rather than forcing every individual response to match.

```python
import numpy as np
import inspect

print("=== CURRENT LOCAL MAP DOSE RESPONSES ===")
local216 = {}
for label in ["calibration_only", "current"]:
    coef = checks215[label]
    w = head177_make(coef)
    base = score215[label][0]["mean"]
    for factor in [0.5, 2.0]:
        mc = wm_cf146[("norepinephrine", factor)]
        pred = head177_eval(mc, w, centres=cent208)
        local216[(label, factor)] = pred
        dr = np.asarray(mc["delta_rate"])
        active = np.asarray(mc["active"], bool)
        for split, splitmask in [("fit", fitall208), ("CV", cvall208)]:
            for event in ["map_arterial", "map_cuff"]:
                mask = splitmask & active & (events206 == event)
                shift = pred["mean"][mask] - base[mask]
                dose = dr[mask]
                slope = np.dot(dose, shift) / max(np.dot(dose, dose), 1e-20)
                print(label, factor, split, event,
                      "n", int(mask.sum()),
                      "mean_dose", round(float(dose.mean()), 5),
                      "mean_shift", round(float(shift.mean()), 5),
                      "slope", round(float(slope), 5))

print("\n=== RAW CACHE STRUCTURE ===")
for label, mc in [("train", wm_train145),
                  ("NE2", wm_cf146[("norepinephrine", 2.0)])]:
    print(label)
    for k, v in mc["raw"].items():
        print(" ", k, type(v).__name__, getattr(v, "shape", None),
              list(v)[:12] if isinstance(v, dict) else "")

print("\n=== AVAILABLE MODEL/COUNTERFACTUAL HELPERS ===")
for n in sorted(list(globals())):
    o = globals()[n]
    if callable(o) and (n.startswith("wm_") or
                        any(s in n for s in ("readout", "head_", "action_step"))):
        try:
            print(n, inspect.signature(o))
        except Exception:
            pass

with open("/work/notes.md", "a") as f:
    f.write("\n## Run 216: reject response candidate 208\n"
            "- Candidate208 tuning/CV value dNLL +0.04680/+0.03430: unacceptable; never installed.\n"
            "- Current edit remains active. Calibration-only/current local tuning dNLL +0.00227/+0.00317.\n"
            "- Next hypothesis: moment-constrained, factual-curvature-minimal correction can improve NE x2 more efficiently than pointwise target fitting.\n")
```