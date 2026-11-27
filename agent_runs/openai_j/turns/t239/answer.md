The cached features include separate doubled- and halved-dose contrasts, and a response-weighted fit was started before the restart. I’ll recover its equations and the weight-installation helper. The next candidate should improve the doubled-dose slope on held-out stays, not merely on the fitting sample.

```python
import dis
import inspect
import numpy as np

print("=== WEIGHT INSTALLATION HELPER ===")
try:
    print(inspect.getsource(head177_make))
except Exception:
    dis.dis(head177_make)

print("\n=== RESPONSE-WEIGHTED DESIGN ===")
try:
    print(inspect.getsource(response_design208))
except Exception:
    dis.dis(response_design208)

print("\n=== FIT SETTINGS AND DATA ===")
print("head names:", head177_names)
print("free neurons:", head177_free)
print("gate:", np.asarray(trial158_gate).shape,
      np.round(np.asarray(trial158_gate)[-8:], 4))
print("MAP output direction:", np.asarray(wm_v151))
print("training metadata columns:", wm_train_meta145.columns.tolist())
print("tuning metadata columns:", wm_tune_meta145.columns.tolist())
print("training metadata sample:")
print(wm_train_meta145.head(3).to_string(index=False))
print("candidate208 keys:", list(map208_candidates))
for k, c in map208_candidates.items():
    print(k, "shape", np.shape(c), "norm", np.linalg.norm(c),
          "range", np.min(c), np.max(c))
print("contrast containers:")
for n in ["cf172", "map172", "bc208", "stat208", "map192_nebase"]:
    o = globals().get(n)
    if isinstance(o, dict):
        print(n, list(o))
print("current coefficient norm:", np.linalg.norm(map192_candidates[(10.0, 6.0)]))
print("calibration coefficient norm:", np.linalg.norm(map192_cal))
print("split counts:",
      int(np.asarray(fitall208).sum()), int(np.asarray(cvall208).sum()),
      int(np.asarray(fitmap208).sum()), int(np.asarray(tunemap208).sum()))
```