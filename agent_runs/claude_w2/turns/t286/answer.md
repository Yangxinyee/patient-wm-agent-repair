The run failed: the largest table I picked (`gg`) is my derived hourly grid, not the raw records that `prep_rollout` needs (columns hour, event, value, rate, dose, end_hour). I'll pick the in-memory table that has those columns, falling back to reading /data/train.csv filtered to the 300 stays. Then I'll run the same test.

The plan is unchanged: fit a ridge readout on arm differences of the head input to the missing insulin effect, and check held-out R² and mean response per arm.

```python
cands = [(k, v) for k, v in globals().items() if isinstance(v, pd.DataFrame) and {'stay_id','hour','event','value','rate','dose','end_hour'} <= set(v.columns)]
cands.sort(key=lambda kv: -len(kv[1]))
if cands: rawdf = cands[0][1]; print("raw:", cands[0][0], len(rawdf))
else:
    rawdf = pd.concat([c[c.stay_id.isin(set(stays))] for c in pd.read_csv('/data/train.csv', chunksize=500000)]); print("read csv", len(rawdf))
R3 = prep_rollout(rawdf, stays)
A3 = R3['A']; hjD = Dw['hj'].astype(int)
print("align code", np.mean(A3.code.values[hjD] == Dw['code']), "t", np.mean(np.isclose(A3.t.values[hjD], Dw['t'])))
DG3 = dose_stack(R3, armsW, 'insulin'); print("DG3", DG3.shape)
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