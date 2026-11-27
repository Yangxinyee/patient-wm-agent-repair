Insulin is null at every horizon from 1 to 12 hours: estimates range from −0.09 to +0.07 mg/dL per u/h with SEs of 0.04–0.11. The jitter does not persist into later doses, and glucose_lab is also null. So insulin has no detectable effect, short or long, and the true insulin readouts should be about 0.

For norepinephrine the only hint is the slow negative drift. Next I will test it with single summary statistics instead of 12 correlated horizons: the mean of MAP(t+k) − MAP(t) over k = 1..12 and over k = 6..12, regressed on the iid titration innovation, with stay-clustered SEs. I will repeat this with cuff MAP as an independent measurement and with alternative policy cell definitions. A consistent negative estimate several SEs from zero across these would support a real slow effect. A fragile one would point to noise.

```python
dn = P[(P['nor']>0) & P['ma'].notna() & (P['nor_prev']>0)].copy().reset_index()
dn['mbin'] = (dn['ma'].clip(20,110)//2).astype(int)
dn['pbin'] = (dn['nor_prev']*20).round().astype(int)
dn['cell'] = dn['mbin']*100 + dn['pbin']
dn['eps'] = dn['nor'] - dn.groupby('cell')['nor'].transform('mean')
dn['ma_prev_f'] = dn['ma_prev'].fillna(dn['ma']); dn['has_mp'] = dn['ma_prev'].notna().astype(float)
ctrl = ['ma_prev_f','has_mp']
F = np.stack([dn[f'ma_f{k}'].values - dn['ma'].values for k in range(1,13)],1)
dn['avg1_12'] = np.nanmean(F,1); dn['avg6_12'] = np.nanmean(F[:,5:],1); dn['avg1_3']=np.nanmean(F[:,:3],1)
for oc in ['avg1_3','avg1_12','avg6_12']:
    b,se,n = fe_ols(dn, oc, ['eps']+ctrl, 'cell'); print(f"art {oc}: {b[0]:+.3f} ({se[0]:.3f}) n={n}")
# cuff outcome (for stays where cuff measured)
for k in range(1,13):
    P[f'mc_f{k}'] = g['mc'].shift(-k)
dn = dn.merge(P[[f'mc_f{k}' for k in range(1,13)]].reset_index(), on=['stay_id','ih'], how='left')
Fc = np.stack([dn[f'mc_f{k}'].values - dn['ma'].values for k in range(1,13)],1)
with np.errstate(all='ignore'):
    dn['c1_12'] = np.nanmean(Fc,1); dn['c6_12']=np.nanmean(Fc[:,5:],1)
for oc in ['c1_12','c6_12']:
    b,se,n = fe_ols(dn, oc, ['eps']+ctrl, 'cell'); print(f"cuff {oc}: {b[0]:+.3f} ({se[0]:.3f}) n={n}")
# alternative cells: finer prev-rate bins & MAP 1mmHg
dn['cell2'] = (dn['ma'].clip(20,110)).round().astype(int)*1000 + (dn['nor_prev']*100).round().astype(int)
dn['eps2'] = dn['nor'] - dn.groupby('cell2')['nor'].transform('mean')
for oc in ['avg1_12','avg6_12']:
    b,se,n = fe_ols(dn, oc, ['eps2']+ctrl, 'cell2'); print(f"art cell2 {oc}: {b[0]:+.3f} ({se[0]:.3f}) n={n}")
```