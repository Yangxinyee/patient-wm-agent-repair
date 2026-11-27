A single linear readout per drug across bundle types is not accurate enough:
- insulin change errors reach about 0.4 units/h;
- NE change errors in mixed bundles reach 0.57 at x2 and 3.7 at x3.

However, the evaluation readouts average over many bundles, so what matters most is the systematic error, i.e. the mean error in the readout's change over the real mix of recorded rates. Isolated large errors at rare rate combinations matter much less.

This run computes, over the tuning stays' actual drug bundles, the mean and rms error of the readout change for each multiplier, in units of the true mean rate change. The NE readout gets a mixed-only fit restricted to the recorded range. If the mean errors stay under a few percent, I can proceed with simple gated readouts. Otherwise I will fit separate readouts per bundle type.

```python
tdb = tu[tu.event.isin(['insulin','norepinephrine'])].copy()
piv = tdb.pivot_table(index=['stay_id','hour'], columns='event', values='rate', aggfunc='first')
pI = piv.get('insulin'); pN = piv.get('norepinephrine')
pairs = list(zip(pI.values, pN.values))
cacheU = {}
def Uc(a, b):
    k = (None if np.isnan(a) else round(float(a),4), None if np.isnan(b) else round(float(b),4))
    if k not in cacheU:
        at = ([ins_atom(k[0])] if k[0] is not None else []) + ([ne_atom(k[1])] if k[1] is not None else [])
        cacheU[k] = U_of(at)
    return cacheU[k]
for m in [0, 0.5, 2, 3]:
    eI_, dI_, eN_, dN_ = [], [], [], []
    for a, b in pairs:
        if not np.isnan(a):
            a2 = min(m*a, 11.5); d = fI.predict(np.stack([Uc(a2,b), Uc(a,b)]))
            eI_.append(d[0]-d[1]-(a2-a)); dI_.append(a2-a)
        if not np.isnan(b):
            d = fN.predict(np.stack([Uc(a, m*b), Uc(a,b)]))
            eN_.append(d[0]-d[1]-(m-1)*b); dN_.append((m-1)*b)
    eI_, dI_, eN_, dN_ = map(np.array, (eI_, dI_, eN_, dN_))
    print('m=%.1f  INS: mean err %.4f (rel %.3f) rms %.3f   NE: mean err %.4f (rel %.3f) rms %.4f  nI %d nN %d' % (m, eI_.mean(), eI_.mean()/dI_.mean(), np.sqrt((eI_**2).mean()), eN_.mean(), eN_.mean()/dN_.mean(), np.sqrt((eN_**2).mean()), len(eI_), len(eN_)))
```