Both hidden-information tests come back clean:

- The insulin dose noise is unrelated to (lab − finger): slope −1e-5 ± 1.2e-4.
- The norepinephrine surprise is unrelated to (cuff − arterial): −6e-5 ± 1e-4.

So the clinicians use no hidden information and the noise-based estimates should be unbiased. However, the discontinuity estimates for norepinephrine come out positive at one hour: 3.4 ± 1.7 mmHg per mcg/kg/min at MAP 65 and 6.1 ± 3.3 at MAP 75. That is about 2σ each, and it conflicts with the noise-based −0.04 ± 0.09.

There are three possible explanations:

1. A statistical fluke or misspecified fit.
2. Something else that changes at the thresholds, an unrecorded co-intervention.
3. The recorded rate noise is not the dose the patient got, so only policy-driven dose changes act.

Next I will sharpen the discontinuity. I will use all stays, with the running variable being whichever MAP reading drove the decision and the outcome being the next reading of the same type. I will try wider bandwidths and print binned residual means around 65 and 75 to see whether there is a visible step. A robust step of about −0.2 to −0.3 mmHg that persists across bandwidths would need explaining. If it shrinks towards zero, it was noise.

```python
C = A2[(A2.ne_p1>0)].copy()
C['hr']=C.index.get_level_values(1)
C['x'] = C.map_arterial.where(C.map_arterial.notna(), C.map_cuff)
C['xtype'] = np.where(C.map_arterial.notna(), 'art', np.where(C.map_cuff.notna(),'cuff','none'))
C = C[C.xtype!='none'].copy()
mcP = P.map_cuff.groupby(level=0); maP = P.map_arterial.groupby(level=0)
for k in [1,2]:
    fa = maP.shift(-k).reindex(C.index).values; fc = mcP.shift(-k).reindex(C.index).values
    C[f'y{k}'] = np.where(C.xtype=='art', fa, fc) - C.x.values
def rd_ne2(S, c, h, ycol):
    S = S[(S.x>=c-h)&(S.x<=c+h)&(S.x!=c)&S[ycol].notna()]
    x = S.x.values-c; a=(x>0).astype(float); n1=S.ne_p1.values
    X = np.column_stack([np.ones(len(S)), a, x, a*x, n1, n1*a, n1*x, (S.xtype=='art').astype(float)])
    by,sey = ols(S[ycol].values, X, S.stay.values); bd,sed = ols(S['ne'].values, X, S.stay.values)
    return by[1],sey[1],bd[1],len(S)
for typ in ['art','cuff','all']:
    S0 = C if typ=='all' else C[C.xtype==typ]
    for c in [65,75]:
        for h in [5,8,12]:
            jy,se,jd,n = rd_ne2(S0, c, h, 'y1')
            print(f"{typ:4s} c={c} h={h:2d}: jump y1 {jy:+.3f}+-{se:.3f}  jump NE {jd:+.4f}  per-unit {jy/jd:+.2f}+-{se/abs(jd):.2f}  n={n}")
```