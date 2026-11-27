The kernel keeps no input history, so I have to infer how fit_ro5 is used from its internals. It tiles head-input draws for target and off-target rows, fits a penalised linear readout restricted by a matrix Pp, and returns weights w, offset w0, gain and r2.

In both existing readouts the stored gain roughly equals R2 (0.50 vs 0.52, 0.19 vs 0.19). That points to least-squares readouts made unit-gain by dividing the output weight by the gain. On that reading the NE correction is about 3/0.19 ≈ 16 times a weak readout, which would explain its noise cost.

Next I'll check that by comparing the W2 columns of gated3 at two a_ne values with tm/gain. I'll also look for candidate Pp matrices (192x192 arrays) among the kernel globals.

```python
Ea = gated3(-1.9, 1.0); Eb = gated3(-1.9, 2.0)
W2a = Ea['W2'] if isinstance(Ea['W2'], np.ndarray) else npv(Ea['W2'])
W2b = Eb['W2'] if isinstance(Eb['W2'], np.ndarray) else npv(Eb['W2'])
col = W2b[:,u3]-W2a[:,u3]
tmv = np.asarray(tm, dtype=float).ravel(); tgv = np.asarray(tg, dtype=float).ravel()
print('per-unit-a NE col / tm: %.4f   1/gain %.4f  1/(gain*1.76) %.4f' % (col@tmv/(tmv@tmv), 1/ROn4['gain'], 1/(ROn4['gain']*1.76)))
coli = (gated3(-2.9,1.0)['W2'] if isinstance(Ea['W2'],np.ndarray) else npv(gated3(-2.9,1.0)['W2']))[:,u1] - W2a[:,u1]
print('per-unit-a INS col / tg: %.4f   1/gain %.4f' % (-(coli@tgv)/(tgv@tgv), 1/ROi5['gain']))
cands = [k for k,v in globals().items() if isinstance(v, np.ndarray) and v.shape==(192,192)]
print('192x192 arrays:', cands)
print('tm,tg shapes', np.shape(tm), np.shape(tg), 'gvn norm %.3f'%np.linalg.norm(gvn))
```