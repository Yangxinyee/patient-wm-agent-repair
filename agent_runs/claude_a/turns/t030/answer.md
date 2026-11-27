Run 26 shows the state right after an insulin bundle does encode the dose. Ridge regression recovers the rate with cross-validated R² 0.82, and the within-tier deviation with R² 0.56 (slope 0.93). So the information the glucose head ignores is present in the state.

The norepinephrine part failed only because `dr.ne` is a pandas method; I need `dr['ne']`. Next I'll redo the norepinephrine decodability. I'll also check whether *earlier* doses (I(t−1), I(t−2), N(t−1)) survive in the state at the current drug bundle. The true one-step dependence includes lagged dose terms, and a head-level correction can only use what the state still holds.

```python
dn = dr[(dr['ne'])&(dr.N>0)&(dr.N_prev>0)].copy()
dn['N_dev'] = np.log(dn.N/dn.N_prev)
for nm, y in [('N', dn.N.values), ('logN_ratio', dn.N_dev.values), ('N_prev', dn.N_prev.values)]:
    r2, p = cvr2(X[dn.idx.values], y, dn.stay_id.values)
    print('norepi', nm, 'n', len(y), 'cv R2 from state', round(r2,3))
# lagged insulin doses decodable at drug bundle t?
Hl = H.copy(); gl_ = Hl.groupby('stay_id')
for j in [1,2,3]:
    Hl[f'I_m{j}'] = gl_.I.shift(j)
dr2 = dr.join(Hl.set_index(['stay_id','h'])[['I_m1','I_m2','I_m3']], on=['stay_id','h'])
sub = dr2.dropna(subset=['I_m1','I_m2','I_m3'])
for c in ['I','I_m1','I_m2','I_m3']:
    r2,_ = cvr2(X[sub.idx.values], sub[c].values, sub.stay_id.values)
    print('decode', c, 'at drug bundles (all drug bundles) R2', round(r2,3), 'n', len(sub))
```