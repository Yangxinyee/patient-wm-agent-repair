The MAP correction uses two neurons to apply a gated linear logit tilt, while retaining the insulin edit. The equation dump again exceeded the output limit. I’ll switch to compact diagnostics and score the unfinished candidate directly; if it improves only the fitting sample or disturbs insulin predictions, I’ll reject it.

```python
import numpy as np
import torch
import dis

print("=== COMPACT EQUATION RECOVERY ===")
for fn in [response_design208, raw_coef208, scalar_eval208]:
    ins = list(dis.get_instructions(fn))
    print(fn.__name__, " ".join(
        f"{i.opname}:{i.argrepr}" for i in ins
        if i.opname not in {"RESUME", "CACHE", "PRECALL", "EXTENDED_ARG"}
    )[:2400])

print("\n=== DATA AND SETTINGS ===")
for n in sorted(list(globals())):
    if "meta" in n and "145" in n:
        o = globals()[n]
        print(n, getattr(o, "shape", None),
              list(o.columns) if hasattr(o, "columns") else "")
for label, mc in [("train", wm_train145), ("tuning", wm_tune145)]:
    print(label, "raw type:", type(mc["raw"]).__name__)
    if hasattr(mc["raw"], "columns"):
        print("raw columns:", mc["raw"].columns.tolist())
print("event labels:", np.unique(events206))
for n in ["fitall208", "cvall208", "fitmap208", "tunemap208"]:
    a = np.asarray(globals()[n])
    print(n, a.dtype, a.shape, "sum", a.sum(), "unique", np.unique(a)[:8])
print("cf172 keys:", list(cf172))
print("bc208 keys:", list(bc208))
for k, v in stat208.items():
    print("stat208", k, type(v).__name__,
          list(v) if isinstance(v, dict) else np.shape(v))
print("free neurons:", head177_free)
print("output direction:", np.round(np.asarray(wm_v151), 3))
print("map192_cal norm:", np.linalg.norm(map192_cal))
for k, c in map208_candidates.items():
    print("candidate", k, "norm", np.linalg.norm(c))

print("\n=== DIRECT FACTUAL SCORING ===")
score214 = {}
checks214 = {
    "calibration_only": map192_cal,
    "current": map192_candidates[(10.0, 6.0)]
}
for k, c in map208_candidates.items():
    checks214["candidate208_" + str(k)] = c
for label, coef in checks214.items():
    w = head177_make(coef)
    a = head177_eval(wm_train145, w)
    b = head177_eval(wm_tune145, w)
    score214[label] = (a, b)
    da = a["nll"] - orig_train209
    db = b["nll"] - orig_tune209
    print(label,
          "fit", float(da[np.asarray(fitall208, bool)].mean()),
          "CV", float(da[np.asarray(cvall208, bool)].mean()),
          "tuning", float(db.mean()))
```