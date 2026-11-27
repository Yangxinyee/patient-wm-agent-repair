Finger and lab glucose agree on average: the mean difference is −0.06 mg/dL. The scatter between them grows with the level (sd ≈ 5 at 90 mg/dL, ≈ 15 at 265), so the measurement noise is multiplicative, a few percent.

In stays without insulin, finger glucose is very persistent: correlation 0.87 at 1 hour and still 0.64 at 12 hours, with a large stay-level component. So any insulin-induced change probably carries forward for hours rather than washing out.

Next I'll estimate how a random dose shock at hour s propagates. I'll regress glucose at s+k (k = 1..12) on the dose deviation at s, I_s − base(G_s)·E[noise], controlling for the current reading. I'll do the same for later doses, to see how the clinicians' feedback reacts.

Since the shock is random, these are valid total effects. Holding later doses fixed (as the readout does) needs one more step: deconvolve out the dose feedback. I'll check selection too, since readings become sparse once insulin stops.

```python
P['ins_mean_given_base'] = P.ins_base*np.exp(on.ins_noise.mean())  # approx
# better: empirical mean rate by base level
mb = on.groupby('ins_base').insulin.mean()
P['shock'] = P.insulin - P.ins_base.map(mb)
g = P.groupby('stay_id')
for k in range(1,13):
    P[f'G+{k}'] = g.g_now.shift(-k)
    P[f'I+{k}'] = g.insulin.shift(-k)
S = P[(P.ins_on==1)&P.g_now.notna()&(P.h<=23)].copy()
S['I+0']=S.insulin
res=[]
for k in range(1,13):
    sub = S[S[f'G+{k}'].notna()]
    X = np.column_stack([np.ones(len(sub)), sub.g_now, sub.shock] + [ (sub.ins_base==b).astype(float) for b in [2,3,4,6]])
    r,_ = ols(sub[f'G+{k}'], X, ['c','g','shock','b2','b3','b4','b6'])
    # dose feedback: I_{s+k} (0 if off)
    sub2 = S[S.h+k<=24]
    X2 = np.column_stack([np.ones(len(sub2)), sub2.g_now, sub2.shock] + [ (sub2.ins_base==b).astype(float) for b in [2,3,4,6]])
    r2,_ = ols(sub2[f'I+{k}'].fillna(0), X2, ['c','g','shock','b2','b3','b4','b6'])
    # observation prob
    r3,_ = ols(sub2[f'G+{k}'].notna().astype(float), X2, ['c','g','shock','b2','b3','b4','b6'])
    res.append((k, len(sub), r.loc['shock','b'], r.loc['shock','se'], r2.loc['shock','b'], r2.loc['shock','se'], r3.loc['shock','b']))
print(pd.DataFrame(res, columns=['k','n','G_eff','se','I_eff','se_I','obs_eff']).to_string())
```