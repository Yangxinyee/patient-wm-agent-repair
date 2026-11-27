The run identified the codes: 6 = glucose_finger, 10 = glucose_lab, 7 = map_arterial, 5 = map_cuff.

`gen_wc` generated `Dw` teacher-forced (`t_hide=1e9`), so all 8 arms share the recorded readings. The arm-to-arm differences in `wcm` are therefore the pure response of the head input to dose changes, with no feedback from drawn readings. That is exactly the data needed to test a head readout fitted on interventions. `Rx` now holds only 100 stays, while `Dw` used the 300 stays in `stays`, so I'll rebuild the rollout table for those 300 and check that rows line up.

The test:
- Target: the missing one-step effect on glucose, Δm = Σ gap(k)·Δdose(t−k) with gap = [−1.68, −1.37, −0.40, +0.21] (data minus model coefficients).
- Fit: ridge on arm differences of `wcm`, training-half stays.
- Check on held-out stays: R², the mean response per arm, and decodability of each dose lag separately.

```python
import numpy as np, pandas as pd, torch
big = sorted([(k, v) for k, v in globals().items() if isinstance(v, pd.DataFrame) and 'stay_id' in v.columns], key=lambda kv: -len(kv[1]))
trdf_ = big[0][1]; print("df:", big[0][0], len(trdf_))
R3 = prep_rollout(trdf_, stays)
A3 = R3['A']; hjD = Dw['hj'].astype(int)
print("align code", np.mean(A3.code.values[hjD] == Dw['code']), "t", np.mean(np.isclose(A3.t.values[hjD], Dw['t'])))
DG3 = dose_stack(R3, armsW, 'insulin'); print("DG3", DG3.shape)
armD = Dw['arm'].astype(int); siD = Dw['si'].astype(int); tD = Dw['t'].astype(float)
LG = np.asarray(lag_abs(DG3, armD, siD, tD, L=4), dtype=float); print("LG", LG.shape, np.round(LG.mean(0), 3))
gap = np.array([-1.68, -1.37, -0.40, 0.21])
mD = LG @ gap
gl = np.isin(Dw['code'], [6, 10])
X = np.asarray(Dw['wcm'], dtype=float)
key = pd.DataFrame(dict(si=siD, hj=hjD, arm=armD, i=np.arange(len(siD))))
k0_ = key[(key.arm == 0) & gl].set_index(['si', 'hj']).i
kd = key[(key.arm != 0) & gl].join(k0_.rename('i0'), on=['si', 'hj']).dropna()
i1, i0 = kd.i.values, kd.i0.values.astype(int)
dX = X[i1] - X[i0]; dL = LG[i1] - LG[i0]; dm = mD[i1] - mD[i0]
tr = siD[i1] < 150; te = ~tr
def ridge(Xa, y, lam=1e-3):
    return np.linalg.solve(Xa.T @ Xa + lam * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ y)
def r2(y, p): return 1 - ((y - p) ** 2).mean() / ((y - y.mean()) ** 2).mean()
for nm, y in [('dm', dm)] + [(f'lag{k+1}', dL[:, k]) for k in range(4)]:
    w = ridge(dX[tr], y[tr]); p = dX @ w
    print(f"{nm}: R2 tr {r2(y[tr], p[tr]):.3f} te {r2(y[te], p[te]):.3f}")
w_dm = ridge(dX[tr], dm[tr]); p = dX @ w_dm
print("mean true/pred dm by arm (test):", {int(a): (round(float(dm[te & (kd.arm.values == a)].mean()), 2), round(float(p[te & (kd.arm.values == a)].mean()), 2)) for a in range(1, 8)})
f0 = (armD == 0) & gl
print("factual sd: target m %.2f  readout %.2f" % (mD[f0].std(), (X[f0] @ w_dm).std()))
```