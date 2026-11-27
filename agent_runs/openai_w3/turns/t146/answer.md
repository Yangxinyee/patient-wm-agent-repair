The retained coefficients failed this check: one greatly overstates dose contrasts, and the other returns almost zero. I won’t apply either. I’ll refit directly from the cached paired features, keeping whole stays held out and penalizing factual-score variance. Good held-out dose slopes with low factual variance would justify calibration.

```python
import numpy as np
import scipy.linalg as sla
from pathlib import Path

D133 = np.asarray(PD89, dtype=np.float64)
F133 = np.asarray(PF89, dtype=np.float64)
Y133 = np.asarray(DY88, dtype=np.float64)
tr133 = np.asarray(DT88, dtype=bool)
va133 = ~tr133
print("feature RMS:", np.sqrt(np.mean(D133**2, axis=(0,2))))
print("target RMS:", np.sqrt(np.mean(Y133**2, axis=0)))
print("metadata sample:", repr(globals().get("contrast_meta88"))[:700])

fits133 = {}
for head133 in range(4):
    d133 = D133[:, head133]
    f133 = F133[:, head133]
    mu133 = f133.mean(axis=0)
    fc133 = f133-mu133
    cc133 = fc133.T @ fc133 / len(fc133)
    cd133 = d133[tr133].T @ d133[tr133] / tr133.sum()
    rhs133 = d133[tr133].T @ Y133[tr133] / tr133.sum()
    for lam133 in (0.0, 0.001, 0.01, 0.1, 1.0):
        b133 = sla.solve(cd133 + lam133*cc133 + 1e-6*np.eye(192),
                         rhs133, assume_a="pos")
        pred133 = d133[va133] @ b133
        err133 = np.sqrt(np.mean((pred133-Y133[va133])**2, axis=0))
        slope133 = []
        for j133 in range(2):
            nz133 = np.abs(Y133[va133,j133]) > 1e-8
            slope133.append(np.sum(pred133[nz133,j133]*Y133[va133,j133][nz133]) /
                            np.sum(Y133[va133,j133][nz133]**2))
        fs133 = fc133 @ b133
        noise133 = np.sqrt(np.mean(fs133**2, axis=0))
        fits133[head133,lam133] = (b133, mu133, err133, noise133, slope133)
        print("head/lambda", head133, lam133,
              "held RMSE", np.round(err133,3),
              "slopes", np.round(slope133,3),
              "factual SD", np.round(noise133,3))

print("=== LAYER-NORM CONSTANT ===")
gamma133 = as_numpy(load_original("core.prelude.0.norm1.weight"))
beta133 = as_numpy(load_original("core.prelude.0.norm1.bias"))
print("min |gamma|", np.min(np.abs(gamma133)),
      "constant", np.sum(beta133/gamma133))

with Path("/work/notes.md").open("a") as f133:
    f133.write("\nPython run 133; forwards 7. Retained Bdelta88 and payload104 fail direct paired-contrast validation (large error versus near-zero response); neither applied. Refit attention-pooled dose contrasts directly with whole-stay holdout and factual-covariance regularization. Original weights remain unchanged.\n")
```
