The run failed on a NumPy version detail: `ndarray.ptp` no longer exists. I rerun the same diagnostics with `np.ptp`. The expectations are unchanged:
- **Insulin:** the out-of-fold fraction recovers to about 0.45.
- **Range and noise:** the range/sd ratio stays small enough that the gate-noise ratio is about 0.3 or less.

```python
for nm,fa_,c_,v_,rho,lam,pos in [('INS',fac,cf,vG,0.4,30,[6,10]),('NE',faN,cN,vM,0.3,300,[7,5])]:
    wg=gate1(pos); tc_=np.array([1.0 if c in pos else -1.0 for c in cidM]); Re=np.ptp(Xm@wg-tc_); sde=np.std(Xw@wg)
    for lam2 in [0.0,3.0,30.0]:
        w,fr,cst=fitw(fa_,c_,v_,rho,lam,lam2); r=Xm@w; Rr=np.ptp(r); sr=np.std(Xw@w)
        print(nm,'lam2',lam2,'frac',round(float(fr),3),'cost g1/full',cst,'Rr/sr',round(Rr/sr,1),'gate noise ratio',round(Rr/(2-Re)*sde/sr,3))
```