The RD at the start threshold is too weak to be useful. Starting rates are low and the untreated side often starts soon after, so the cumulative-dose jump is only 0.04–0.47 rate-hours. The MAP jumps are 0 ± 0.2–0.5. It neither supports nor refutes anything.

To decide between "norepinephrine absent" and "tiny slow negative effect", I want the quantity the evaluation reads: the implied norepinephrine x2 readout with a confidence interval. I will
1. deconvolve the local projections LP_k into the open-loop impulse response h_k (removing the dose persistence R_j);
2. apply h to the recorded rates of the tuning stays used in forward pass 1, from hour 12 on, at the reached stay-hours;
3. bootstrap over training stays for the SE.

If the interval comfortably includes 0, I will target 0 for norepinephrine, as for insulin. If it excludes 0, I need to build a negative effect into the model.

```python
dnb = dn2[['stay_id','ih','eps','cell','ma_prev_f','has_mp']+[f'dM{k}' for k in range(1,13)]].copy()
dnb = dnb.merge(P[[f'nor_f{k}' for k in range(1,13)]].fillna(0).reset_index(), on=['stay_id','ih'], how='left')
# demean within cell once (absorb), then regressions are simple
cols = ['eps','ma_prev_f','has_mp']+[f'dM{k}' for k in range(1,13)]+[f'nor_f{k}' for k in range(1,13)]
A = dnb[cols] - dnb.groupby('cell')[cols].transform('mean')
A['stay_id'] = dnb.stay_id.values
X = A[['eps','ma_prev_f','has_mp']].values
def est(idx_rows):
    Xs = X[idx_rows]; LPk=[]; Rk=[1.0]
    for k in range(1,13):
        y = A[f'dM{k}'].values[idx_rows]; m = ~np.isnan(y)
        b = np.linalg.lstsq(Xs[m], y[m], rcond=None)[0]; LPk.append(b[0])
        yr = A[f'nor_f{k}'].values[idx_rows]
        b2 = np.linalg.lstsq(Xs, yr, rcond=None)[0]; Rk.append(b2[0])
    LPk=np.array(LPk); Rk=np.array(Rk)
    h = np.zeros(13)
    for k in range(1,13):
        h[k] = LPk[k-1] - sum(Rk[j]*h[k-j] for j in range(1,k))
    return LPk, Rk, h
LP0, R0, h0 = est(np.arange(len(A)))
print("h:", np.round(h0[1:],3)); print("cum h:", np.round(np.cumsum(h0[1:]),3))
# tuning stays: recorded NE rates hours 12..23, reached rows from fwd_1
de1 = pd.read_csv('/work/fwd_1/dose_edit_original.csv')
reach = de1[(de1.edit=='ne_x2')&de1.reached][['stay_id','hour']]
def implied(h, factor=1.0):
    vals=[]
    for sid, hr in reach.itertuples(index=False):
        r = PT.loc[sid]['nor'] if sid in PT.index.get_level_values(0) else None
        tot = sum(h[hr-s]*factor*r.loc[s] for s in range(12, hr) if (hr-s)<=12)
        vals.append(tot)
    return np.mean(vals)
print("implied ne_x2 readout (point):", round(implied(h0,1.0),3), " ne_half:", round(implied(h0,-0.5),3))
stays_u = A.stay_id.unique(); rng = np.random.default_rng(1)
pos = A.groupby('stay_id').indices
bs=[]
for rep in range(40):
    samp = rng.choice(stays_u, len(stays_u), replace=True)
    idx = np.concatenate([pos[s] for s in samp])
    _,_,hb = est(idx); bs.append(implied(hb,1.0))
print("bootstrap ne_x2 implied: mean", np.mean(bs).round(3), "sd", np.std(bs).round(3))
```