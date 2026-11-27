The data are split by stay into fitting and held-out groups, but the diagnostic stopped on an inspection error before scoring anything. No weights changed. I’ll fix that small error and compare the unfinished candidate with the current edit and calibration-only baseline.

```python
import numpy as np
import torch

print("=== CONTRAST CACHE STRUCTURE ===")
for k, v in stat208.items():
    if isinstance(v, (tuple, list)):
        print(k, [(type(x).__name__, getattr(x, "shape", None)) for x in v])
    else:
        print(k, type(v).__name__, getattr(v, "shape", None))
for n in sorted(list(globals())):
    o = globals()[n]
    if isinstance(o, dict) and o:
        keys = list(o)
        if any(isinstance(k, tuple) and k and k[0] == "norepinephrine"
               for k in keys):
            print(n, "keys:", keys[:10],
                  "value types:", [type(o[k]).__name__ for k in keys[:3]])
print("free neurons:", head177_free)
print("output direction:", np.round(np.asarray(wm_v151), 3))
print("calibration coefficient norm:", np.linalg.norm(map192_cal))
for k, c in map208_candidates.items():
    print("candidate", k, "norm", np.linalg.norm(c))

print("\n=== DIRECT FACTUAL SCORING ===")
score215 = {}
checks215 = {
    "calibration_only": map192_cal,
    "current": map192_candidates[(10.0, 6.0)]
}
for k, c in map208_candidates.items():
    checks215["candidate208_" + str(k)] = c
for label, coef in checks215.items():
    w = head177_make(coef)
    a = head177_eval(wm_train145, w)
    b = head177_eval(wm_tune145, w)
    score215[label] = (a, b)
    da = a["nll"] - orig_train209
    db = b["nll"] - orig_tune209
    print(label,
          "fit", float(da[fitall208].mean()),
          "CV", float(da[cvall208].mean()),
          "tuning", float(db.mean()))

with open("/work/notes.md", "a") as f:
    f.write("\n## Runs 212–215: recovered local fitting machinery\n"
            "- MAP uses two gated neurons for a raw-input linear logit tilt; insulin neurons are retained.\n"
            "- Training cached readings: 2,751 fitting / 1,759 held-out, separated by stay.\n"
            "- Runs 213–214 diagnostic output/errors did not alter weights.\n"
            "- Directly scoring unfinished response-weighted candidate before considering installation.\n")
```