No rows matched in either file, so `stays` does not hold record stay ids, and the stays behind `Dw` cannot be recovered cheaply. I'll stop trying to reuse `Dw`.

Instead I'll regenerate the same teacher-forced arm data with `gen_wc` on the current `Rx` (100 stays). This time the original weights are loaded fresh, and `Rx` is the table whose dose grid I already have, so rows and doses line up by construction.

The test is the same: decodability, on held-out stays, of the missing insulin effect from head-input differences between arms.
- **Workable:** R² of the differences at or above about 0.5, with unbiased mean differences per arm. The edit then goes in the value head.
- **Not workable:** the dose information the head sees is too entangled.

```python
W0 = getW(load_original)
try:
    D4 = gen_wc(Rx, armsW, W0, None, None, None, seed=0); print("gen_wc ok", {k: np.shape(v) for k, v in list(D4.items())[:4]})
except Exception as e:
    print("gen_wc failed:", repr(e)[:300]); D4 = None
if D4 is not None:
    DG = dose_stack(Rx, armsW, 'insulin')
    armD = np.asarray(D4['arm']).astype(int); siD = np.asarray(D4['si']).astype(int); tD = np.asarray(D4['t'], dtype=float); hjD = np.asarray(D4['hj']).astype(int)
    LG = np.asarray(lag_abs(DG, armD, siD, tD, L=4), dtype=float)
    gap = np.array([-1.68, -1.37, -0.40, 0.21]); mD = LG @ gap
    gl = np.isin(D4['code'], [6, 10]); X = np.asarray(D4['wcm'], dtype=float)
    key = pd.DataFrame(dict(si=siD, hj=hjD, arm=armD, i=np.arange(len(siD))))
    k0_ = key[(key.arm == 0) & gl].set_index(['si', 'hj']).i
    kd = key[(key.arm != 0) & gl].join(k0_.rename('i0'), on=['si', 'hj']).dropna()
    i1, i0 = kd.i.values, kd.i0.values.astype(int)
    dX = X[i1] - X[i0]; dL = LG[i1] - LG[i0]; dm = mD[i1] - mD[i0]
    tr = siD[i1] < 50; te = ~tr
    def ridge(Xa, y, lam=1e-3): return np.linalg.solve(Xa.T @ Xa + lam * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ y)
    def r2(y, p): return 1 - ((y - p) ** 2).mean() / ((y - y.mean()) ** 2).mean()
    print("n diff rows", len(kd), "nonzero dm", int((np.abs(dm) > 1e-9).sum()))
    for nm, y in [('dm', dm)] + [(f'lag{k+1}', dL[:, k]) for k in range(4)]:
        w = ridge(dX[tr], y[tr]); p = dX @ w
        print(f"{nm}: R2 tr {r2(y[tr], p[tr]):.3f} te {r2(y[te], p[te]):.3f}")
    w_dm = ridge(dX[tr], dm[tr]); p = dX @ w_dm
    print("mean true/pred dm by arm (test):", {int(a): (round(float(dm[te & (kd.arm.values == a)].mean()), 2), round(float(p[te & (kd.arm.values == a)].mean()), 2)) for a in range(1, 8)})
    f0 = (armD == 0) & gl
    print("factual sd: target m %.2f  readout %.2f" % (mD[f0].std(), (X[f0] @ w_dm).std()))
```