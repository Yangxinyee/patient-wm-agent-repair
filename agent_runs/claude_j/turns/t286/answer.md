The gate itself now works. Its code dot products come out exactly ±1, and it adds tilt noise of only about 0.19 of the signal scale. But projecting the readout off the three bin tables destroyed it: the insulin fraction fell from 0.46 to 0.14 and drug B's to about 0.01. Those roughly 48 directions evidently carry much of the dose signal in the query.

The bin terms do not need projecting out. Within a (code, last bin) group they are constant, so the centring absorbs them, and they matter only for the gate margin. Next I project the readout only off the code-embedding rows and the last_numeric columns, about 21 directions. A raw-variance penalty keeps its range modest. If the fraction comes back to about 0.45 for insulin, with range/sd small enough for a gate-noise ratio around 0.3 or less, I build the edit.

```python
Cm2=np.vstack([EC,LNW.T]); Cm2=Cm2[np.linalg.norm(Cm2,axis=1)>1e-6]; Pc=np.eye(192)-Cm2.T@np.linalg.pinv(Cm2.T)
for nm,fa_,c_,v_,rho,lam,pos in [('INS',fac,cf,vG,0.4,30,[6,10]),('NE',faN,cN,vM,0.3,300,[7,5])]:
    wg=gate1(pos); tc_=np.array([1.0 if c in pos else -1.0 for c in cidM]); Re=(Xm@wg-tc_).ptp(); sde=np.std(Xw@wg)
    for lam2 in [0.0,3.0,30.0]:
        w,fr,cst=fitw(fa_,c_,v_,rho,lam,lam2); r=Xm@w; Rr=r.max()-r.min(); sr=np.std(Xw@w)
        print(nm,'lam2',lam2,'frac',round(float(fr),3),'cost g1/full',cst,'Rr/sr',round(Rr/sr,1),'gate noise ratio',round(Rr/(2-Re)*sde/sr,3))
```