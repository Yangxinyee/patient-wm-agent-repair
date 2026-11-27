The regression discontinuities agree with the jitter estimates, though they are noisier:
- At 180 the mean dose doubles (+0.93 u/h), but the later glucose change shows no jump (−0.05 to +0.27 mg/dL per unit, SE about 0.2).
- At 220 the estimates are mildly negative (−0.14 to −0.84, SE about 0.3), and at 260 mildly positive (+0.15 to +0.67).
- Together they scatter around zero with no consistent sign.

This matters because the RD varies the base dose, not just the jitter. So it also rules out a "jitter is only charting noise" story, in which only the base dose acts.

Next I will check:
- the on/off discontinuity at 150, where insulin starts, using the next finger reading where available;
- a pooled RD over the 180, 220, 260 and 300 thresholds;
- the jitter-based local projection separately by base level, to detect an effect that depends on glucose level (with absorbed fixed effects, avoiding the singular matrix).

If all of these are near zero, I will treat "insulin has no effect" as established.

```python
dd = d.copy()
# on/off at 150 using next-hour finger where observed (selection caveat)
for k in [1,2,3]:
    b,se,n = rd(dd, 'gf', f'dG{k}', 150, 10)
    jd = rd(dd, 'gf', 'ins', 150, 10)
    print(f"150 on/off k={k}: jump dG {b:+.2f} ({se:.2f}) n={n}; dose jump {jd[0]:.3f}; obs-rate below/above:",
          dd[(dd.gf>=140)&(dd.gf<150)][f'gf_f{k}'].notna().mean().round(3), dd[(dd.gf>=150)&(dd.gf<160)][f'gf_f{k}'].notna().mean().round(3))
# pooled RD across thresholds 180,220,260,300 (stacked, threshold-specific trends)
rows=[]
for c in [180,220,260,300]:
    D = dd[(dd.ins>0)&(dd.gf>=c-20)&(dd.gf<c+20)].copy(); D['c']=c; D['x']=D.gf-c; D['ab']=(D.x>=0).astype(float); rows.append(D)
S = pd.concat(rows)
for k in [1,2,3,4,6,8]:
    S2 = S[[f'dG{k}','ins','x','ab','c','stay_id']].dropna()
    C = pd.get_dummies(S2.c).values.astype(float)
    X = np.c_[S2.ab, C, C*S2.x.values[:,None], C*(S2.x*S2.ab).values[:,None]]
    # instrument dose by 'ab': 2SLS
    Xd = np.c_[S2.ins, C, C*S2.x.values[:,None], C*(S2.x*S2.ab).values[:,None]]
    Z = X
    # first stage
    bfs = np.linalg.lstsq(Z, S2.ins.values, rcond=None)[0]; ins_hat = Z@bfs
    Xh = Xd.copy(); Xh[:,0] = ins_hat
    b,se,_ = ols(S2[f'dG{k}'].values, Xh, S2.stay_id.values)
    print(f"pooled RD 2SLS k={k}: per-unit effect {b[0]:+.3f} ({se[0]:.3f}) n={len(S2)}")
# jitter LP by base level
for bl in [1,2,3,4,6]:
    sub = P[(P.ins>0)&P.gf.notna()&(P.base==bl)].copy().reset_index()
    out=[]
    for k in [1,3,6]:
        sub[f'dG{k}'] = sub[f'gf_f{k}']-sub.gf
        b,se,n = fe_ols(sub, f'dG{k}', ['ins'], 'gbin5'); out.append(f"k{k}:{b[0]:+.2f}({se[0]:.2f})")
    print("base",bl, " ".join(out), "n", len(sub))
```