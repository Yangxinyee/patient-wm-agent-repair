Two findings.

**The dose token is not linear in dose.** `u` moves a long way as the rate changes (norm about 20 across the range), but along a curve: a straight-line fit in rate explains only 33% (insulin) and 10% (norepinephrine) of its variation. My earlier PCA showed `u` lies almost entirely in a 3-dimensional subspace, so copying `u` into z is cheap. A single "dose direction", however, is a crude summary.

**The mechanism is safe so far.**
- Duplicating `u` as the z token leaves NLL unchanged (1.556292 vs 1.556315).
- Injecting 4 × residual × slope direction changes NLL only slightly. Finger and lab improve (1.4743→1.4736, 1.3008→1.2987), but arterial MAP gets worse (1.3895→1.3911). So this direction also leaks into MAP predictions.

Next I need the actual effect sizes. I'll regress the change in predicted means (κ = 4 minus duplicate) at each reading type on the insulin residual at lags 1–3. To test linearity, I'll repeat with κ = 8 and check whether the effect doubles or saturates.
- **Good outcome:** a lag-1 glucose effect of order −1 per unit residual at κ = 4 that doubles at κ = 8, with small MAP effects. Then the token route has enough gain and I only need a cleaner direction.
- **Bad outcome:** saturation, meaning prelude normalisation caps this route. The large shifts that doubled doses require would then need an amplifier in the head, the z_prior/z_head adapter.

```python
# residuals per (stay,hour) for BUN stays
rmap={}
for d_ in allD:
    if d_['hi']:
        ri_=d_['ins']-float(torch.tensor(d_['ctx'],dtype=T)@pi_i[0]+pi_i[1]); rmap[(d_['s'],int(math.floor(d_['t']+1e-6)))]=ri_
rows=[]
for lab_,ki in [('ki8',8)]:
    rr=[]
    for s in BUN: rr+=[(s,)+r for r in tf_stay_z(W_cur,BUN[s],make_zfn(ki,0))]
    res_tf[lab_]=rr; nl=np.array([r[4] for r in rr]); cd=np.array([r[2] for r in rr])
    print(lab_,'NLL %.6f'%nl.mean(),{c:round(nl[cd==c].mean(),4) for c in [6,10,7,5]})
base=res_tf['dup']
for lab_ in ['ki4','ki8']:
    cur=res_tf[lab_]
    for c in [6,10,7,5]:
        idx=[j for j,r in enumerate(base) if r[2]==c and abs(r[1]-round(r[1]))<1e-6]
        y=np.array([cur[j][3]-base[j][3] for j in idx])
        X_=np.array([[rmap.get((base[j][0],int(round(base[j][1]))-k),0.0) for k in (1,2,3)] for j in idx])
        A=np.column_stack([np.ones(len(y)),X_]); b,*_=np.linalg.lstsq(A,y,rcond=None)
        fit=A@b; r2=1-((y-fit)**2).sum()/((y-y.mean())**2).sum()
        print(lab_,c,'n',len(y),'coef lag1-3',np.round(b[1:],3),'R2 %.3f'%r2,'sd dmean %.3f'%y.std())
```