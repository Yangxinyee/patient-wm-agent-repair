With `make_edit(1,1)`, finger glucose predictions move by about −0.15 on average, with SD 2.0 across rows. The NLL changes are small: +0.001 for finger, +0.003 for lab and +0.012 for cuff MAP. The cuff cost needs watching once the NE coefficient grows.

The rows carry the true lag-1 deviations (`dev1`, `nedev1`). I will regress the shift on them, for each code, on this batch and on three more training batches. If a_ins is "mg/dL per u/h of deviation", the slope for finger and lab glucose should be about +1. The slope for MAP on NE deviation should be about +1 per unit of rate, or whatever unit a_ne is in. What remains is readout noise.

```python
def shift_stats(i,E,lab=''):
    Xs_,R_=datasets[i]; o=head_run(Xs_,R_,E); pmb,nllb,_,_=base_out[i]
    sh=npv(o[0])-pmb; dn=npv(o[1])-nllb; cd=R_['code'].values; out=[]
    for c,dc in [(6,'dev1'),(10,'dev1'),(7,'nedev1'),(5,'nedev1')]:
        m=(cd==c)&np.isfinite(R_[dc].values)
        x=R_[dc].values[m]; y=sh[m]
        sl=np.cov(x,y)[0,1]/np.var(x,ddof=1); res=y-sl*(x-x.mean())-y.mean()
        out.append((c,m.sum(),round(sl,3),round(y.mean(),3),round(res.std(),3),round(np.std(x),3),round(dn[m].mean(),4)))
    return out
for i in [0,2,3,4]:
    print(i, shift_stats(i,E1))
```