Deconvolving gives an open-loop response that drifts negative: the cumulative effect is −0.26 by 3 hours and −1.3 by 12 hours, per mcg/kg/min sustained. Applied to the tuning stays' recorded rates, it implies a norepinephrine x2 readout of −0.31 ± 0.15 (bootstrap SD) and x1/2 of +0.16. That is about 2 SD from zero, compared with the model's −0.03.

This is borderline. It could be a real small effect, or the chance result of my many looks. The most informative check is an independent sample. About half the stays have no arterial line, and there the titration runs on cuff MAP. I will build the innovations from cuff-MAP cells in those stays, with future cuff MAP as the outcome, and compute the same implied x2 readout with a bootstrap. A similar negative estimate in these disjoint stays would make a real effect credible. A null there would favour chance. I will also compute the implied insulin x2 readout the same way, for comparison.

```python
dc = P[(P['nor']>0) & P['ma'].isna() & P['mc'].notna() & (P['nor_prev']>0)].copy()
dc = dc[~dc.index.get_level_values(0).map(stay_art).astype(bool)].reset_index()   # non-arterial-line stays only
dc['cell'] = (dc['mc'].clip(20,130)//2).astype(int)*100 + (dc['nor_prev']*20).round().astype(int)
dc['mc_prev'] = dc.merge(P[['map_any']].groupby(level=0).shift(1).reset_index().rename(columns={'map_any':'mp'}), on=['stay_id','ih'], how='left')['mp'].values
dc['mp_f'] = dc['mc_prev'].fillna(dc['mc']); dc['has_mp']=dc['mc_prev'].notna().astype(float)
dc = dc.merge(P[[f'mc_f{k}' for k in range(1,13)]].reset_index(), on=['stay_id','ih'], how='left', suffixes=('_old',''))
for k in range(1,13): dc[f'dM{k}'] = dc[f'mc_f{k}'] - dc['mc']
dc = dc.merge(P[[f'nor_f{k}' for k in range(1,13)]].fillna(0).reset_index(), on=['stay_id','ih'], how='left', suffixes=('_old',''))
dc['eps'] = dc['nor'] - dc.groupby('cell')['nor'].transform('mean')
print("n cuff-stay innovation hours:", len(dc), " eps sd:", dc.eps.std().round(3))
cols = ['eps','mp_f','has_mp']+[f'dM{k}' for k in range(1,13)]+[f'nor_f{k}' for k in range(1,13)]
A = dc[cols] - dc.groupby('cell')[cols].transform('mean'); A['stay_id']=dc.stay_id.values
X = A[['eps','mp_f','has_mp']].values
LPc, Rc, hc_ = est(np.arange(len(A)))
print("cuff LP:", np.round(LPc,3)); print("cuff R:", np.round(Rc[1:],3)); print("cuff h:", np.round(hc_[1:],3))
print("implied ne_x2 (cuff stays IRF applied to tuning arterial stays):", round(implied(hc_,1.0),3))
stays_u = A.stay_id.unique(); pos = A.groupby('stay_id').indices; bs=[]
for rep in range(40):
    samp = rng.choice(stays_u, len(stays_u), replace=True); idx = np.concatenate([pos[s] for s in samp])
    _,_,hb = est(idx); bs.append(implied(hb,1.0))
print("bootstrap: mean", np.mean(bs).round(3), "sd", np.std(bs).round(3))
```