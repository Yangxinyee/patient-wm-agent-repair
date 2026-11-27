The state file holds 5,244 bundles for the 99 stays: 992 numbers each, plus stay, hour, action flags and atom counts. Action bundles sit at hour t + 0.083.

I'll take the action bundles and measure how well each part of the state, linearly and cross-validated by stay, recovers four rates: the current and previous action-A rate (the bundle holds the new record and the end of the previous one) and the same two for action B. I'll do this raw and residualised on readings history. The parts are the posterior z mean, out, and the slot mean of h.

If z or the slot mean keeps the residual rates at R² well above q's (0.31, and about 0.05 for B), the information is lost between the state and the head and can be recovered there. If all are low, the encoder or prelude is discarding it and the edit has to go further upstream.

```python
xs = S2['x']; hrS = S2['hour']; sidS = S2['stay_id']
isA = np.abs(hrS - np.floor(hrS) - 0.0833) < 0.01
tA = np.floor(hrS[isA]).astype(int); iA = np.array([sid2i[int(s)] for s in sidS[isA]])
def panA(P, t): c = t + 2; o = np.full(len(c), np.nan); ok = (c >= 0) & (c < P.shape[1]); o[ok] = P[iA[ok], c[ok]]; return np.nan_to_num(o)
tgt = {'I0': panA(ins, tA), 'I-1': panA(ins, tA - 1), 'N0': panA(ne, tA), 'N-1': panA(ne, tA - 1)}
cols = []
for l in (0, 1, 2):
    gl_ = np.nan_to_num(panA(glast, tA - l), nan=150); ml_ = np.nan_to_num(panA(mlast, tA - l), nan=75)
    cols += [np.clip(gl_, 60, 400), gl_ > 140, gl_ > 160, gl_ > 180, gl_ > 220, gl_ > 260, np.clip(ml_, 40, 120), ml_ < 65, ml_ > 75]
CA = np.column_stack(cols + [tA]).astype(float)
parts = {'z': xs[isA][:, 960:992], 'out': xs[isA][:, :192], 'slotmean': xs[isA][:, 192:960].reshape(-1, 4, 192).mean(1)}
grpA = sidS[isA]
for nm, y in tgt.items():
    yr = resid(y, CA)
    print(nm, ' '.join(f"{p}: raw {cvr2_(X, y, grpA):.2f} res {cvr2_(X, yr, grpA):.2f}" for p, X in parts.items()))
print('n action bundles', isA.sum())
```