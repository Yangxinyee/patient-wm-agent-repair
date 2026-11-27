The jointly quiet directions are now orthogonal to everything to about 1e-6, yet the linearity numbers are identical to before (insulin slope 0.29, R² 0.70; NE slope 2.16). So the choice of direction was never the problem.

The cause I now suspect is that LayerNorm divides by each bundle's own std of x2. That std depends on which drugs are in the bundle and at what rates, through the bin embeddings. The channel's amplitude in u is then κ·rate·σ̄/σ(bundle): a multiplicative, dose-dependent distortion that no choice of direction removes. This also matches the original model's own saturating dose response.

If the realized gain is a near-deterministic function of drug type and rate, I can pre-compensate. Each drug's mlp1 unit pair can produce any piecewise-linear function of rate, f(r) = r·σ(r)/σ̄. The 1-drug and 2-drug unit pairs in mlp2 can carry separate gains.

This run tabulates the realized gain (channel ÷ rate) and σ(x2) by drug, set size and rate band. Small within-band spread supports the compensation plan; large spread would refute it.

```python
sig2=_lnstd(E0['x2']); sbar=sig2.mean()
for i,(nm_,rr) in enumerate([('I',rIb),('N',rNb)]):
    dd=Dh2[:,i]/g1_; proj=du@dd/(dd@dd)
    edges_=[0.25,0.75,1.25,2,3,4,6,9,12,40] if nm_=='I' else [0.005,0.05,0.1,0.2,0.35,0.5,0.75,1.01,1.6,3.1]
    for nn in [1,2]:
        m_=(E0['cnt']==nn)&(rr>0); b_=np.digitize(rr[m_],edges_)
        rows_=[]
        for bb_ in np.unique(b_):
            s_=b_==bb_
            if s_.sum()<5: continue
            gq=proj[m_][s_]/rr[m_][s_]
            rows_.append('%.2f:%.3f±%.3f σ%.1f(n%d)'%(np.median(rr[m_][s_]),np.median(gq),gq.std(),np.median(sig2[m_][s_]),s_.sum()))
        print(nm_,'n=%d'%nn,' | '.join(rows_))
print('sigma x2: mean %.2f  empty %.2f'%(sbar,_lnstd(E0['xe'][None])[0]))
```