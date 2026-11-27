The fallback table `dr_` does not contain the 300 `Dw` stays. I'll read the stays straight from /data/train.csv and /data/tuning.csv, filtered by id. I'll check alignment against `Dw` (codes and times of the referenced rows) before computing the targets.

The test is unchanged: can a ridge readout on head-input differences between dose arms reproduce the missing insulin effect, in held-out R² and the per-arm mean?

```python
ids_set = set(np.asarray(stays).tolist())
parts = []
for fn in ['/data/train.csv', '/data/tuning.csv']:
    for c in pd.read_csv(fn, chunksize=1000000):
        c = c[c.stay_id.isin(ids_set)]
        if len(c): parts.append(c.assign(_src=fn[6:11]))
rawdf = pd.concat(parts); print("rows", len(rawdf), "stays found", rawdf.stay_id.nunique(), rawdf._src.unique())
rawdf = rawdf.drop(columns='_src')
R3 = prep_rollout(rawdf, stays)
A3 = R3['A']; hjD = Dw['hj'].astype(int)
print("align code", np.mean(A3.code.values[hjD] == Dw['code']), "t", np.mean(np.isclose(A3.t.values[hjD], Dw['t'])))
DG3 = dose_stack(R3, armsW, 'insulin')
armD = Dw['arm'].astype(int); siD = Dw['si'].astype(int); tD = Dw['t'].astype(float)
LG = np.asarray(lag_abs(DG3, armD, siD, tD, L=4), dtype=float); print("LG", LG.shape, np.round(LG.mean(0), 3))
gap = np.array([-1.68, -1.37, -0.40, 0.21]); mD = LG @ gap
gl = np.isin(Dw['code'], [6, 10]); X = np.asarray(Dw['wcm'], dtype=float)
key = pd.DataFrame(dict(si=siD, hj=hjD, arm=armD, i=np.arange(len(siD))))
k0_ = key[(key.arm == 0) & gl].set_index(['si', 'hj']).i
kd = key[(key.arm != 0) & gl].join(k0_.rename('i0'), on=['si', 'hj']).dropna()
i1, i0 = kd.i.values, kd.i0.values.astype(int)
dX = X[i1] - X[i0]; dL = LG[i1] - LG[i0]; dm = mD[i1] - mD[i0]
tr = siD[i1] < 150; te = ~tr
def ridge(Xa, y, lam=1e-3): return np.linalg.solve(Xa.T @ Xa + lam * len(Xa) * np.eye(Xa.shape[1]), Xa.T @ y)
def r2(y, p): return 1 - ((y - p) ** 2).mean() / ((y - y.mean()) ** 2).mean()
for nm, y in [('dm', dm)] + [(f'lag{k+1}', dL[:, k]) for k in range(4)]:
    w = ridge(dX[tr], y[tr]); p = dX @ w
    print(f"{nm}: R2 tr {r2(y[tr], p[tr]):.3f} te {r2(y[te], p[te]):.3f}")
w_dm = ridge(dX[tr], dm[tr]); p = dX @ w_dm
print("mean true/pred dm by arm (test):", {int(a): (round(float(dm[te & (kd.arm.values == a)].mean()), 2), round(float(p[te & (kd.arm.values == a)].mean()), 2)) for a in range(1, 8)})
f0 = (armD == 0) & gl
print("factual sd: target m %.2f  readout %.2f" % (mD[f0].std(), (X[f0] @ w_dm).std()))
```