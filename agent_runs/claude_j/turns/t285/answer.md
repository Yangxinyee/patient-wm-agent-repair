The bin tables are embed.bins.value, embed.bins.rate and embed.bins.dose (17 × 192 each). The gate solve failed on a singular constraint matrix. Some constraint rows are exactly zero (bin 0, padding, unused last_numeric inputs), and the zero padding row still carried a ±1 target.

Next run: drop near-zero constraint rows, solve with a pseudo-inverse, and rerun the same diagnostics. What I need is a gate-noise ratio well below about 0.3, with the out-of-fold fraction and cost close to the earlier fit.

```python
nz=np.linalg.norm(Cm,axis=1)>1e-6; Cz=Cm[nz]; print('constraints kept',int(nz.sum()),'of',len(nz))
def gate1(pos):
    t=np.array([1.0 if c in pos else -1.0 for c in range(13)]+[0.0]*(Cm.shape[0]-13))[nz]
    return Swi@Cz.T@np.linalg.pinv(Cz@Swi@Cz.T)@t
for nm,fa_,c_,v_,rho,lam,pos in [('INS',fac,cf,vG,0.4,30,[6,10]),('NE',faN,cN,vM,0.3,300,[7,5])]:
    wg=gate1(pos); tc_=np.array([1.0 if c in pos else -1.0 for c in cidM]); eta=Xm@wg-tc_; sde=np.std(Xw@wg); Re=eta.max()-eta.min()
    print(nm,'gate check: code dots',np.round(EC[[5,6,7,10,8,9]]@wg,2).tolist(),'eta range',round(Re,3),'sd within',round(sde,4))
    for lam2 in [0.3,3.0]:
        w,fr,cst=fitw(fa_,c_,v_,rho,lam,lam2); r=Xm@w; Rr=r.max()-r.min(); sr=np.std(Xw@w)
        print(' ',nm,'lam2',lam2,'frac',round(float(fr),3),'cost g1/full',cst,'Rr/sr',round(Rr/sr,1),'gate noise ratio',round(Rr/(2-Re)*sde/sr,3) if Re<2 else 'NA')
```