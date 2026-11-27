`lag_abs` indexes its dose grid with the time argument, so it needs integer hours. Readings fall on integer hours, so I'll pass rounded hours. I'll also print its short source to confirm what the lags mean, then run the same decodability test on the regenerated `D4`.

```python
import inspect
print(inspect.getsource(lag_abs) if hasattr(lag_abs, '__code__') and False else open('/work/dec.py').read().split('def wc_hidden')[0][-500:])
tI = np.round(tD).astype(int)
LG = np.asarray(lag_abs(DG, armD, siD, tI, L=4), dtype=float)
mD = LG @ gap
dL = LG[i1] - LG[i0]; dm = mD[i1] - mD[i0]
print("rows", len(X), "diff rows", len(kd), "nonzero dm", int((np.abs(dm) > 1e-9).sum()), "LG mean", np.round(LG.mean(0), 2))
chk = (armD == 1) & (tI >= 14) & gl
print("x2 check lag1 ratio:", round(float(LG[chk, 0].sum() / max(1e-9, LG[key.set_index(['si','hj']).index.isin([]) | False].sum() if False else 1)), 3) if False else "skip")
for nm, y in [('dm', dm)] + [(f'lag{k+1}', dL[:, k]) for k in range(4)]:
    w = ridge(dX[tr], y[tr]); p = dX @ w
    print(f"{nm}: R2 tr {r2(y[tr], p[tr]):.3f} te {r2(y[te], p[te]):.3f}")
w_dm = ridge(dX[tr], dm[tr]); p = dX @ w_dm
print("mean true/pred dm by arm (test):", {int(a): (round(float(dm[te & (kd.arm.values == a)].mean()), 2), round(float(p[te & (kd.arm.values == a)].mean()), 2)) for a in range(1, 8)})
f0 = (armD == 0) & gl
print("factual sd: target m %.2f  readout %.2f" % (mD[f0].std(), (X[f0] @ w_dm).std()))
```