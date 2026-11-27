The sensor-family gate is nearly exact, which is encouraging for avoiding cross-sensor effects. The pooled dose decoders may nevertheless carry unnecessary factual noise. I’ll refit each decoder using only its relevant sensor family and check every cached multiplier. Much lower factual variance with near-unit dose slopes would support a gated-head edit.

```python
from scipy.linalg import solve
from scipy.special import softmax
import torch.nn.functional as F

directions204 = {}
family204 = {
    "insulin": np.isin(events179, ["glucose_finger", "glucose_lab"]),
    "norepinephrine": np.isin(events179, ["map_arterial", "map_cuff"])
}
Xraw204 = np.asarray(X200, dtype=np.float64)
Wbase204 = np.asarray(load_original("attr_heads.value_bin.0.weight"))
bbase204 = np.asarray(load_original("attr_heads.value_bin.0.bias"))
Wfinal204 = np.asarray(load_original("attr_heads.value_bin.2.weight"))
bfinal204 = np.asarray(load_original("attr_heads.value_bin.2.bias"))
Hcheck204 = F.gelu(torch.tensor(
    Xraw204[..., :192] @ Wbase204.T + bbase204, dtype=torch.float32
)).numpy()
print("Hidden reconstruction RMS:",
      np.sqrt(np.mean((Hcheck204-H179)**2)))
offset204 = L179 - (H179 @ Wfinal204.T + bfinal204)
print("Offset variation over latent draws:",
      float(np.max(np.std(offset204, axis=0))))
print("Centres first rows:", C179[:2])
print("Original cached NLL:", float(np.mean(original_nll179)))

for drug in ("insulin", "norepinephrine"):
    own = family204[drug]
    FF = Xraw204[:, fitmask179 & own].reshape(-1, 193)
    FV = Xraw204[:, valmask179 & own].reshape(-1, 193)
    dc, yc = [], []
    for factor in (0.0, 0.5, 1.5, 2.0, 3.0):
        rows, xc, dx, dr = canon_cases200[(drug, factor)]
        assert np.max(np.abs(xc-Xraw204[:, rows]-dx)) < 2e-5
        use = fitmask179[rows] & own[rows]
        dc.append(dx[:, use].reshape(-1, 193))
        yc.append(np.broadcast_to(dr[use], (8, use.sum())).reshape(-1))
    DD = np.concatenate(dc).astype(np.float64)
    yy = np.concatenate(yc).astype(np.float64)
    cf = FF.T @ FF / len(FF)
    cd = DD.T @ DD / len(DD)
    rhs = DD.T @ yy / len(DD)
    reg = np.diag(np.maximum(np.diag(cf), 1e-4))
    print("\n", drug, "family-specific")
    for lam in (.001, .003, .01, .03, .1, .3, 1.):
        w = solve(cd + lam*cf + 1e-7*reg, rhs,
                  assume_a="pos", check_finite=False)
        directions204[(drug, lam)] = w
        slopes = {}
        for factor in (0.0, .5, 1.5, 2., 3.):
            rows, xc, dx, dr = canon_cases200[(drug, factor)]
            use = valmask179[rows] & own[rows]
            dd = dx[:, use] @ w
            rr = np.broadcast_to(dr[use], dd.shape)
            slopes[factor] = round(float(np.sum(dd*rr)/np.sum(rr*rr)), 3)
        print(lam, "held-out factual SD", round(float(np.std(FV@w)), 4),
              "mean", round(float(np.mean(FV@w)), 4), "slopes", slopes)

print("\nNo weights changed.")
```