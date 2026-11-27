The prelude reconstruction is exact, but the joint linear fit is poor: residual sd 0.5 at insulin bundles and 0.63 at reading bundles. The likely cause is the LayerNorm scale σ_x that every write through `prelude_norm` divides by. It ranges from 30 to 89 across bundles and looks bimodal. Probably σ_x depends on bundle type: drug bundles come 5 minutes after the reading, so the state has decayed less. If so, one map forced to serve both bundle types will be noisy, while separate maps per type could each be precise.

Next I check σ_x by bundle type. Then I refit two decoders separately:
1. Head 3 only, at drug-start bundles. Target: the insulin rate (0 at NE-only bundles).
2. Heads 2 and 3, at glucose reading bundles. Target: minus the next insulin rate.

If σ_x is tight within type and the rate fit drops to a residual of about 0.07 (as my notes recorded for raw u), the rate path is fine. In that case only the reading-bundle compensation needs a different route.

```python
sx=sdx.mean(1).numpy()
for nm_,m_ in types.items():
    if m_.sum(): print(f"{nm_:9s} sigma_x pct", np.round(np.percentile(sx[m_],[1,10,50,90,99]),1))
dtsb=ds2['dts'].numpy()
for nm_,m_ in types.items():
    if m_.sum(): print(f"{nm_:9s} dt(s) pct", np.round(np.percentile(dtsb[m_],[1,50,99]),0))
drug_b=ins_s|ne_s
X3=(A_[:,:4,3,:]/sdx[:,:,None]).mean(1).numpy(); X3r=A_[:,:4,3,:].mean(1).numpy()
for lab_,XX in [('A3/sigma',X3),('A3 raw',X3r)]:
    w3,_,_,_=np.linalg.lstsq(np.c_[XX[drug_b],np.ones(drug_b.sum())], rate_ins[drug_b], rcond=None)
    pr3=np.c_[XX,np.ones(N2)]@w3
    print(lab_, 'ins resid sd', np.std(pr3[ins_s]-rate_ins[ins_s]).round(4), 'ne_only resid', np.std(pr3[ne_s&~ins_s]).round(4), 'R2', 1-np.var(pr3[drug_b]-rate_ins[drug_b])/np.var(rate_ins[drug_b]))
    for nm_,m_ in types.items():
        if m_.sum() and not (m_&drug_b).any(): print('   off-target', nm_, 'mean', pr3[m_].mean().round(3), 'sd', pr3[m_].std().round(3))
```