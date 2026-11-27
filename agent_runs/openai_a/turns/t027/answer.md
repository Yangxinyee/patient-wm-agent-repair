Insulin rates show sharp jumps at glucose readings near 150 and 180, unlike the smooth norepinephrine relationship. Those jumps offer a stronger insulin identification strategy than the previous regressions. I’ll estimate their effects across several bandwidths and check pre-treatment outcomes; a stable post-treatment effect with no placebo jump would support causality.

```python
import scipy.linalg as sla
from scipy.optimize import least_squares
import json

n_stays, n_times, n_cols = P.shape
stay_ix = np.repeat(np.arange(n_stays), n_times)
time_ix = np.tile(np.arange(n_times), n_stays)
g = P[:, :, columns.index("glucose_finger")].reshape(-1).astype(np.float64)
d = P[:, :, columns.index("insulin")].reshape(-1).astype(np.float64)
thresholds = np.arange(150., 331., 30.)
nearest = np.argmin(np.abs(g[:, None] - thresholds[None, :]), axis=1)
distance = g - thresholds[nearest]
base_rd = np.isfinite(g) & (time_ix >= 1) & (time_ix < n_times - 1)

def shifted_col(col, horizon):
    a = P[:, :, columns.index(col)]
    result = np.full(a.shape, np.nan, dtype=np.float32)
    if horizon > 0:
        result[:, :-horizon] = a[:, horizon:]
    elif horizon < 0:
        result[:, -horizon:] = a[:, :horizon]
    else:
        result[:] = a
    return result.reshape(-1).astype(np.float64)

def insulin_rd(horizon=1, bandwidth=10, degree=1, history=False):
    y = shifted_col("glucose_finger", horizon)
    keep = base_rd & (np.abs(distance) <= bandwidth) & (distance != 0) & np.isfinite(y)
    idx = np.flatnonzero(keep)
    s = distance[idx] / bandwidth
    side = (s > 0).astype(float)
    group = nearest[idx]
    xs = [np.ones(len(idx)), (time_ix[idx] - 13) / 12]
    for k in range(1, len(thresholds)):
        xs.append((group == k).astype(float))
    for k in range(len(thresholds)):
        sel = (group == k).astype(float)
        for power in range(1, degree + 1):
            xs.extend([sel * s**power, sel * side * s**power])
    if history:
        for code, lag in [("glucose_lab", 0), ("glucose_finger", -1),
                          ("glucose_finger", -2), ("insulin", -1)]:
            val = shifted_col(code, lag)[idx]
            missing = ~np.isfinite(val)
            fill = np.nanmedian(val)
            val[missing] = fill
            scale = np.std(val) or 1
            xs.extend([(val - np.mean(val)) / scale, missing.astype(float)])
    X = np.column_stack(xs)
    zres = side - X @ sla.lstsq(X, side, lapack_driver="gelsy")[0]
    dd = d[idx]
    denom = zres @ dd
    beta = (zres @ y[idx]) / denom
    resid = y[idx] - beta * dd
    resid -= X @ sla.lstsq(X, resid, lapack_driver="gelsy")[0]
    scores = np.bincount(stay_ix[idx], weights=zres * resid, minlength=n_stays)
    se = np.sqrt(scores @ scores) / abs(denom)
    ds = []
    if horizon > 0:
        for lag in range(horizon):
            ds.append(float(zres @ shifted_col("insulin", lag)[idx] / denom))
    return {"n": len(idx), "beta": float(beta), "se": float(se),
            "first_stage_jump": float(denom / (zres @ side)),
            "dose_impulse": ds}

rd_checks = {}
for bandwidth in (5, 10, 14):
    for degree in (1, 2):
        for history in (False, True):
            rd_checks[str((bandwidth, degree, history))] = insulin_rd(
                1, bandwidth, degree, history)
print("=== INSULIN ONE-HOUR FUZZY RD ===")
print(json.dumps(rd_checks, indent=2))
rd_lp = {h: insulin_rd(h, 10, 1, False) for h in (-2, -1, 1, 2, 3, 4, 6, 8)}
print("\n=== PLACEBOS AND LOCAL PROJECTIONS ===")
print(json.dumps(rd_lp, indent=2))
hs = [h for h in rd_lp if h > 0]
def rd_dynamic_residual(par):
    beta, rho = par
    return np.array([
        (beta * np.dot(np.array(rd_lp[h]["dose_impulse"]),
                       rho ** np.arange(h - 1, -1, -1)) - rd_lp[h]["beta"])
        / rd_lp[h]["se"] for h in hs])
rd_dynamic_fit = least_squares(rd_dynamic_residual, [-2., .85],
                              bounds=([-15., 0.], [0., 1.05]))
print("RD dynamic fit [effect, persistence]:", rd_dynamic_fit.x)
print("Standardized moment errors:", rd_dynamic_fit.fun)

with open("/work/notes.md", "a") as f:
    f.write("\n## Recovery and threshold identification; Python 23, forward 4\n")
    f.write("- Architecture, panel, and 100-stay original states recovered; no edits.\n")
    f.write("- Insulin policy has sharp rate jumps near finger glucose 150, 180, etc.; "
            "testing fuzzy RD rather than assuming observed dose associations are causal.\n")
    f.write(json.dumps({"rd_checks": rd_checks, "rd_lp": rd_lp,
                        "dynamic_fit": rd_dynamic_fit.x.tolist(),
                        "moment_errors": rd_dynamic_fit.fun.tolist()}, indent=2) + "\n")
```
