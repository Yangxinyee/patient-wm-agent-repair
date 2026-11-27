This pass used the wrong stays. I typed IDs 2020000000–099 instead of loading sel100, so these 100 stays have only 60 with insulin and 33 with NE. The edited-vs-original comparison is still valid, but the sel100 truth (−12.96 etc.) does not apply here.

The result is surprising. With k = 1, the insulin channel has a large effect with the wrong sign: x2 gives +3.30 (original −0.38) and x1/2 gives −5.08 (original +0.21). It is also far from linear: x1/2 is 1.5 times larger than x2, where a linear response would give −0.5 times. NE barely responds: both edits move by about +0.13 regardless of direction.

A likely explanation for the sign: the value head takes the level from the "last value" register. It may read the state's news direction as "the last reading was surprisingly high/low", which predicts regression back. In that case injecting negative news raises predictions.

The asymmetry could come from the insulin-rate readout misbehaving at halved rates, since the action encoding is strongly non-monotonic near rate 1.

To tell these apart I'll check two things:
- The per-hour profile of the edited-minus-original shift. This shows whether the effect builds up over hours or appears at once.
- The replica's insulin readout at ×0.5 and ×2 rates against the true scaled rates.

```python
d8o = pd.read_csv('/work/fwd_8/dose_edit_original.csv'); d8e = pd.read_csv('/work/fwd_8/dose_edit_edited.csv')
m8 = d8e.merge(d8o, on=['stay_id','hour','edit'], suffixes=('_e','_o')); m8 = m8[m8.reached_e]
m8['d'] = m8.shift_e - m8.shift_o
print(m8.pivot_table(index='hour', columns='edit', values='d', aggfunc='mean').round(2).to_string())
print(m8.groupby('edit').d.agg(['mean','count']).round(3))
# readout nonlinearity on sel100 insulin bundles
Minv = np.linalg.inv(M); LN1 = lambda u: lnw(u, P+'norm1')
res = []
for k_i, k in enumerate(rows):
    st = g_start.get((sid[k], round(hr[k],4)))
    if st is None: continue
    for mult in [0.5, 1.0, 2.0, 3.0, 0.0]:
        atoms = []
        for r in st.itertuples():
            if r.event=='insulin':
                if mult>0: atoms.append(drug_atom('insulin', r.rate*mult, r.dose*mult))
            else: atoms.append(drug_atom('norepinephrine', r.rate, r.dose))
        u = enc(atoms,'encode_action') if atoms else u_const
        fh = Minv @ (A.T @ Dkv @ LN1(u)); res.append((k_i, mult, st[st.event=='insulin'].rate.sum()*mult, fh[0], fh[2]))
res = pd.DataFrame(res, columns=['i','mult','r','tI','pI']); res = res[res.groupby('i').r.transform('max')>0]
res['err'] = res.tI - res.r
print(res.groupby('mult').agg(r=('r','mean'), tI=('tI','mean'), bias=('err','mean'), rms=('err', lambda e: np.sqrt((e**2).mean())), pI=('pI','mean')).round(3))
```