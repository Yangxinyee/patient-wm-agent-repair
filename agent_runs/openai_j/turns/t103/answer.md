The record-based response-kernel targets are much larger than the original model’s rollout shifts, although the unstable insulin IV estimates warrant caution. Before using those targets, I’ll check exactly how the fitted decoder was converted into head weights. If the saved correction differs from the intended correction, that could explain the rejected candidate’s large factual loss.

```python
import inspect
import numpy as np
from pathlib import Path

print("=== IDENTIFICATION NOTES ===")
nt = Path("/work/notes.md").read_text()
for title in ("Causal identification checks", "Dose-innovation local projections",
              "Fixed-dose response-kernel estimates"):
    start = nt.find("## " + title)
    if start >= 0:
        end = nt.find("\n## ", start + 4)
        section = nt[start:end if end >= 0 else len(nt)]
        print(section[:1400])

print("\n=== STORED WEIGHT-CORRECTION AUDIT ===")
H = np.asarray(repair_fH, dtype=np.float64)
mu = np.asarray(repair_head_mean, dtype=np.float64)
sd = np.asarray(repair_head_sd, dtype=np.float64)
c = np.asarray(repair_head_coef, dtype=np.float64)
b = float(repair_head_intercept)
axis = np.asarray(repair_axis, dtype=np.float64)
j = int(np.argmax(np.abs(axis)))
saved_c = np.asarray(repair_head_dW[j], dtype=np.float64) / axis[j]
saved_b = float(repair_head_db[j]) / axis[j]
print("H mean/SD ranges:", np.quantile(H.mean(0), [0, .5, 1]),
      np.quantile(H.std(0), [0, .5, 1]))
print("stored coefficient norm/intercept:", np.linalg.norm(c), b)
print("saved correction norm/intercept:", np.linalg.norm(saved_c), saved_b)
print("saved coefficient distance to c and c/sd:",
      np.linalg.norm(saved_c-c), np.linalg.norm(saved_c-c/sd))
variants90 = {
    "H@c+b": H @ c + b,
    "standardized_H@c+b": ((H-mu)/sd) @ c + b,
    "saved_weight_tilt": H @ saved_c + saved_b,
    "H@decoder87+b": H @ np.asarray(decoder87) + b,
}
for label, vals in variants90.items():
    print(label, "mean/SD/RMS:", vals.mean(), vals.std(),
          np.sqrt(np.mean(vals**2)))
print("max saved rank-one error:",
      np.max(np.abs(repair_head_dW - np.outer(axis, saved_c))),
      np.max(np.abs(repair_head_db - axis*saved_b)))

print("\n=== OTHER COEFFICIENT-SIZED VARIABLES ===")
for k, v in sorted(dict(globals()).items()):
    if isinstance(v, np.ndarray) and v.shape in ((192,), (193,)):
        print(k, "norm", np.linalg.norm(v), "first", v[:3])
    elif any(s in k.lower() for s in ("tilt", "intercept")) and isinstance(v, (float, int, np.floating)):
        print(k, v)

print("\n=== RECONSTRUCTION HELPER ===")
for name in ("reconstruct_reading_head", "repair_tilt", "score_head_variant"):
    fn90 = globals()[name]
    print(name, inspect.signature(fn90))
    try:
        src90 = inspect.getsource(fn90)
        print(src90[:2500])
    except (OSError, TypeError):
        print("locals:", fn90.__code__.co_varnames)
        print("string constants:",
              [x for x in fn90.__code__.co_consts if isinstance(x, str)])

print("\nACTIVE EDITS:", edited())
```