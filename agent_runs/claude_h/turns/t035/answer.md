The cuff-MAP stays, which share no stays with the arterial analysis, point the same way. Their implied norepinephrine x2 readout is −0.37 (bootstrap mean −0.28, SD 0.36). Pooled with the arterial estimate (−0.31 ± 0.15) by inverse variance, that gives about −0.32 ± 0.14 (z ≈ −2.3).

So there is moderate evidence for a small negative norepinephrine effect: about −0.3 mmHg for x2, and correspondingly about +0.15 for x1/2. That is small but roughly ten times the model's −0.03.

Before acting on it, I want to calibrate the method. I will apply exactly the same pipeline (local projections, deconvolution, implied readout on the forward-pass-1 stays, bootstrap) to insulin, where all evidence says null. A near-zero insulin readout within its SE would show the method is not biased. A spurious ~2 SE result would mean the norepinephrine result should be discounted.

```python
di2 = P[(P['ins']>0) & P['gf'].notna()].copy().reset_index()
di2['cell'] = di2['gbin5']
di2['gp_f'] = di2['gf_prev'].fillna(di2['gf']); di2['has_gp']=di2['gf_prev'].notna().astype(float)
for k in range(1,13): di2[f'dM{k}'] = di2[f'gf_f{k}'] - di2['gf']
di2 = di2.drop(columns=[f'nor_f{k}' for k in range(1,13) if f'nor_f{k}' in di2.columns])
insf = P[[f'ins_f{k}' for k in range(1,13)]].fillna(0).reset_index()
insf.columns = ['stay_id','ih']+[f'nor_f{k}' for k in range(1,13)]   # reuse est() column names
di2 = di2.merge(insf, on=['stay_id','ih'], how='left')
di2['eps'] = di2['ins'] - di2.groupby('cell')['ins'].transform('mean')
cols = ['eps','gp_f','has_gp']+[f'dM{k}' for k in range(1,13)]+[f'nor_f{k}' for k in range(1,13)]
A = di2[cols] - di2.groupby('cell')[cols].transform('mean'); A['stay_id']=di2.stay_id.values
X = A[['eps','gp_f','has_gp']].values
LPi_, Ri_, hi_ = est(np.arange(len(A)))
print("ins h:", np.round(hi_[1:],3)); print("cum:", np.round(np.cumsum(hi_[1:]),3))
reach_i = de1[(de1.edit=='insulin_x2')&de1.reached][['stay_id','hour']]
def implied_i(h, factor=1.0):
    vals=[]
    for sid, hr in reach_i.itertuples(index=False):
        r = PT.loc[sid]['ins']
        vals.append(sum(h[hr-s]*factor*r.loc[s] for s in range(12, hr) if (hr-s)<=12))
    return np.mean(vals)
print("implied ins_x2:", round(implied_i(hi_),3), " ins_half:", round(implied_i(hi_,-0.5),3))
stays_u = A.stay_id.unique(); pos = A.groupby('stay_id').indices; bs=[]
for rep in range(40):
    samp = rng.choice(stays_u, len(stays_u), replace=True); idx = np.concatenate([pos[s] for s in samp])
    _,_,hb = est(idx); bs.append(implied_i(hb,1.0))
print("bootstrap ins_x2: mean", np.mean(bs).round(3), "sd", np.std(bs).round(3))
```