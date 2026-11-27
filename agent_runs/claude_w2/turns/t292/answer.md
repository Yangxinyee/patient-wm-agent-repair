`gap` was undefined because run 277 crashed before defining everything after the `lag_abs` call. `lag_abs` itself is now clear: lag k for a reading at integer hour t is the infusion started at hour t−k (grid index t−k+3). I'll run the whole test self-contained.

```python
gap = np.array([-1.68, -1.37, -0.40, 0.21])
armD = np.asarray(D4['arm']).astype(int); siD = np.asarray(D4['si']).astype(int); tD = np.asarray(D4['t'], dtype=float); hjD = np.asarray(D4['hj']).astype(int)
tI = np.round(tD).astype(int)
LG = np.asarray(lag_abs(DG, armD, siD, tI, L=4), dtype=float); mD = LG @ gap
gl = np.isin(np.asarray(D4['code']), [6, 10]); X = np.asarray(D4['wcm'], dtype=float)
key = pd.DataFrame(dict(si=siD, hj=hjD, arm=armD, i=np.arange(len(siD))))
k0_ = key[(key.arm == 0) & gl].set_index(['si', 'hj']).i
kd = key[(key.arm != 0) & gl].join(k0_.rename('i0'), on=['si', 'hj']).dropna()
i1, i0 = kd.i.values, kd.i0.values.astype(int)
dX = X[i1] - X[i0]; dL = LG[i1] - LG[i0]; dm = mD[i1] - mD[i0]
tr = siD[i1] < 50; te = ~tr
def ridge(Xa, y, lam=1e-3): return np.linalg.solve(Xa.T @ Xa + lam * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ y)
def r2(y, p): return 1 - ((y - p) ** 2).mean() / ((y - y.mean()) ** 2).mean()
print("rows", len(X), "diff rows", len(kd), "nonzero dm", int((np.abs(dm) > 1e-9).sum()), "LG mean", np.round(LG.mean(0), 2))
for nm, y in [('dm', dm)] + [(f'lag{k+1}', dL[:, k]) for k in range(4)]:
    w = ridge(dX[tr], y[tr]); p = dX @ w
    print(f"{nm}: R2 tr {r2(y[tr], p[tr]):.3f} te {r2(y[te], p[te]):.3f}")
w_dm = ridge(dX[tr], dm[tr]); p = dX @ w_dm
print("mean true/pred dm by arm (test):", {int(a): (round(float(dm[te & (kd.arm.values == a)].mean()), 2), round(float(p[te & (kd.arm.values == a)].mean()), 2)) for a in range(1, 8)})
f0 = (armD == 0) & gl
print("factual sd: target m %.2f  readout %.2f" % (mD[f0].std(), (X[f0] @ w_dm).std()))
```