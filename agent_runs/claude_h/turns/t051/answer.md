This run turns up a pattern a simple drug effect cannot explain:
- **Glucose_lab:** shows no clear insulin effect (×2 readout +0.30 ± 0.62).
- **Insulin → MAP placebo:** cleanly zero (se 0.02).
- **Norepinephrine → glucose placebo:** noisy positive values (about +0.2 to +0.7, se 0.26–0.59), in line with the noise level I am working at.
- **By base level:** the per-unit slopes are inconsistent (0.19, 0.02, 0.03, 0.11, 0.32).
- **Jitter bins:** the decisive finding. Rows where the nurse gave exactly the sliding-scale dose (jitter = 1, a large point mass) have the lowest later glucose residual (−1.17 ± 0.05). Rows given both less (−0.67) and more (−0.62) than the scale sit about 0.5 mg/dL higher.

That U-shape (about 5–8 sigma between "exact" and "jittered" rows) cannot come from a linear drug effect, which would be monotone in dose. It suggests "followed the scale exactly" versus "deviated" marks something systematic, such as a type of nurse or measurement, or a rounding artefact. The "insulin raises glucose" slope is then probably an artefact of that structure rather than a drug effect.

To understand it, I will look at:
- the raw insulin rate values (rounding?) and the jitter distribution (share exactly 1, values outside [0.5, 2]);
- whether the jitter category relates to hour, base level, lab versus finger source, or the previous hour's jitter category;
- per-lag residuals by category (below, exact, above) within each base level.

Two outcomes would settle it:
- **Real effect:** later glucose rises steadily with jitter within each base level.
- **Artefact:** the "exact" category differs regardless of dose direction.

```python
t0=time.time()
v=INS0[INS0>0]
vc=pd.Series(v).round(4).value_counts()
print('n insulin',len(v),'unique',len(vc)); print(vc.head(25).to_dict())
si3,ti3=np.nonzero(irows_pos)
g3=Gpol[si3,ti3]; b3=base_of(g3); r3=INS0[si3,ti3]; j3=r3/np.maximum(b3,1e-9)
src=np.where(np.isnan(GF[si3,ti3]),'lab','fing')
print('rows',len(r3),'base==0 but insulin>0:',(b3==0).sum())
okb=b3>0
print('frac jitter exactly 1:',np.mean(np.isclose(j3[okb],1)).round(4))
print('jitter quantiles',np.quantile(j3[okb],[0,.01,.05,.1,.25,.5,.75,.9,.95,.99,1]).round(3))
cat=np.where(np.isclose(j3,1),1,np.where(j3<1,0,2))
for bv in (1,2,3,4,6):
    m=okb&(b3==bv); print(' base',bv,'n',m.sum(),'cat shares',np.bincount(cat[m],minlength=3)/m.sum(), 'lab share',np.mean(src[m]=='lab').round(3))
hr3=ti3-1
print('exact share by hour block:',[np.mean(cat[okb&(hr3>=a)&(hr3<a+4)]==1).round(3) for a in range(0,24,4)])
# previous-hour category persistence
prev=np.full(len(si3),-1)
key=dict(zip(zip(si3,ti3),cat))
prevc=np.array([key.get((s,t-1),-1) for s,t in zip(si3,ti3)])
for c in (0,1,2):
    m=okb&(prevc==c); print(' prev cat',c,'-> cat shares',(np.bincount(cat[m],minlength=3)/max(1,m.sum())).round(3))
# per-lag residual by category within base (use RI residuals; RI rows = irows_pos rows in same order as np.nonzero)
assert (RI['si']==si3).all() and (RI['ti']==ti3).all()
for k in (1,3,5):
    y=RI['Ly'][k-1]
    row=[]
    for bv in (1,2,3,4):
        m=okb&(b3==bv)&~np.isnan(y)
        row.append([round(float(np.mean(y[m&(cat==c)])),2) for c in (0,1,2)])
    print(' k',k,'resid by base[1,2,3,4] x cat[below,exact,above]:',row)
print('time',round(time.time()-t0,1))
```