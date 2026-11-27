The factual file gives each reading's observed value and predicted mean. The states give the z posterior mean per bundle with exact hours. I join them at measurement bundles and regress z on the four reading surprises, with presence indicators and an intercept. The surprise coefficients are the z-direction per mg/dL or mmHg for each code. Their cosine similarities tell me whether finger and lab share a direction, and likewise arterial and cuff.

```python
F2k = F2.copy(); F2k['s'] = F2k.observed - F2k.predicted_mean
codes4 = ['glucose_finger','glucose_lab','map_arterial','map_cuff']
pv = F2k.pivot_table(index=['stay_id','hour'], columns='event', values='s', aggfunc='first')
zidx = {(s, round(h,4)): i for i, (s, h) in enumerate(zip(np.asarray(sid3), np.asarray(hr3)))}
rows_i, Xs_ = [], []
for (s, h), r in pv.iterrows():
    i = zidx.get((s, round(h,4)))
    if i is None: continue
    v = [r.get(c, np.nan) for c in codes4]
    rows_i.append(i); Xs_.append([0 if np.isnan(x) else x for x in v] + [0 if np.isnan(x) else 1 for x in v])
Xs_ = np.array(Xs_, dtype=float); Zm = np.asarray(Z3)[rows_i]
Xd = np.c_[Xs_, np.ones(len(Xs_))]
Bz, *_ = np.linalg.lstsq(Xd, Zm, rcond=None)
res_z = Zm - Xd @ Bz; print('n bundles', len(rows_i), ' R2 per z-dim (mean) %.3f' % (1 - res_z.var(0)/Zm.var(0)).mean())
D4 = Bz[:4].T  # 32 x 4 directions
nrm = np.linalg.norm(D4, axis=0); print('direction norms per unit surprise', np.round(nrm,4))
cs = (D4.T @ D4)/np.outer(nrm, nrm); print('cosines f-l %.2f a-c %.2f f-a %.2f l-c %.2f' % (cs[0,1], cs[2,3], cs[0,2], cs[1,3]))
```