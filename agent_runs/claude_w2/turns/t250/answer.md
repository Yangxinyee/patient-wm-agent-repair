The failure is a name clash. The kernel's `r2` is an older helper with a different signature, not a simple R² of predictions. I'll define my own `R2(y, ŷ)` and rerun the same evaluation.

```python
def R2(y,yh): y=np.asarray(y,float); return 1-np.var(y-yh)/np.var(y)
for al in [0.01,0.1,1.0]:
    w,b0=ridge_fit(Dw['wc'][glu&~te],r_all[glu&~te],al); rh=Dw['wc']@w+b0
    m0=glu&te&(Dw['arm']==0); mt=glu&te
    print('alpha',al,'R2 test',round(R2(r_all[mt],rh[mt]),3),'fact R2',round(R2(r_all[m0],rh[m0]),3),'fact err sd',round(float(np.std(rh[m0]-r_all[m0])),2),'r sd',round(float(np.std(r_all[m0])),2),'cf slopes',pair_eval(rh,glu&te))
Pg=Dw['P'][glu&(Dw['arm']==0)]; cg=CENT[Dw['code'][glu&(Dw['arm']==0)]]; p_=Pg[:,1:]/Pg[:,1:].sum(1,keepdims=True)
mu=(p_*cg).sum(1); sd=np.sqrt((p_*cg**2).sum(1)-mu**2); print('pred sd quantiles', np.quantile(sd,[.05,.25,.5,.75,.95]).round(1))
```