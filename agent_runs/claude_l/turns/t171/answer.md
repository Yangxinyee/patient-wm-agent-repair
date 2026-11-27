The linear fit cuts the median residual from 4.26 to 1.07, but its coefficients are huge and cancel. The two age features are nearly collinear because almost every register age is exactly 1 hour, so the fit only says a roughly constant per-column shift is missing. One concrete candidate: the register age may be measured to the time of the previous bundle, where the prediction is made, rather than to the predicted bundle. That would usually be 300 s, giving log1p(300)/10 = 0.57 instead of 0.82. I'll test that age definition, an age-in-hours variant, and a pure per-column constant-shift fit. If one age definition brings the median error near 1, the register is solved.

```python
def reg_feats2(B_, k, code, ref='prev'):
    tref = B_[k-1][0] if ref=='prev' else B_[k][0]
    for j in range(k-1, -1, -1):
        for a in B_[j][1]:
            if a['code']==code and a['kind']==0 and a['num'][1]==1:
                return dict(has=1.0, age=(tref-B_[j][0])*3600.0, z=a['num'][0], vb=a['vb'])
    return dict(has=0.0, age=0.0, z=0.0, vb=0)
def evalH(fvfun, ref):
    errs=[]
    for (s,k,c,pm) in tests:
        rf=reg_feats2(BUN[s],k,c,ref); errs.append(pm_fv(VV[(s,k)],c,rf,fvfun(rf))-pm)
    errs=np.array(errs); return round(float(np.median(np.abs(errs))),3), round(float(errs.mean()),3)
H = {
 'A_prev': (lambda rf:[rf['has'], math.log1p(rf['age'])/10, rf['z'],0,0,0,0,0], 'prev'),
 'A_cur':  (lambda rf:[rf['has'], math.log1p(rf['age'])/10, rf['z'],0,0,0,0,0], 'cur'),
 'Ah_prev':(lambda rf:[rf['has'], math.log1p(rf['age']/3600)/10, rf['z'],0,0,0,0,0], 'prev'),
 'A0':     (lambda rf:[rf['has'], 0, rf['z'],0,0,0,0,0], 'cur'),
}
for n_,(f_,r_) in H.items(): print(n_, evalH(f_, r_))
coef0,*_=np.linalg.lstsq(Gr,Res,rcond=None); print('const shift per col', np.round(coef0,3), 'resid', round(float(np.median(np.abs(Res-Gr@coef0))),3))
```