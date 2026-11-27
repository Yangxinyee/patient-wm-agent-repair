The NE titration noise is also unpredictable from observed covariates (R² 0.001), so it too is a valid randomisation. Stops happen only from very low rates (0.01-0.06) when MAP ≥ 75. In 5.5% of on-NE hours there is no MAP reading at all; I set the noise to zero there.

Next I'll estimate the NE effect on map_arterial and map_cuff, using the same reduced-form / first-stage / deconvolution approach. Because the titration is a random walk, the first stage should stay near 1 for many hours, so deconvolution is essential. I'll also check the arterial-cuff relationship at the same hour (their medians differ, 74 versus 81), and whether NE noise moves glucose. A positive K_N(1) consistent across lags would mean NE truly raises MAP. Estimates near zero would mean the drug does nothing here and the model's belief should be removed.

```python
both = gg[gg.ma.notna() & gg.mc.notna()]
print('ma-mc same hour: mean', round((both.ma-both.mc).mean(),2), 'sd', round((both.ma-both.mc).std(),2), 'n', len(both))
print('arterial-line stays frac', gg.groupby('stay_id').ma.count().gt(0).mean(), ' ma count per art stay', gg[gg.ma.notna()].groupby('stay_id').size().describe().round(1).to_dict())
dn = gg[(gg.nor_prev>0) & gg.map_pol.notna() & (gg.nor0>0)].copy()
Fn=[]; Rn_ma=[]; Rn_mc=[]; Rn_gf=[]; Fi=[]
L=10
for k in range(0,L+1):
    s = dn[dn[f'nor_l{k}'].notna()]
    Fn.append(LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev, s.h], s[f'nor_l{k}']).coef_[0])
    if k>=1:
        s = dn[dn[f'ma_l{k}'].notna()]
        Rn_ma.append(LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev, s.h], s[f'ma_l{k}']).coef_[0])
        s = dn[dn[f'mc_l{k}'].notna()]
        Rn_mc.append(LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev, s.h], s[f'mc_l{k}']).coef_[0])
        s = dn[dn[f'gf_l{k}'].notna()]
        Rn_gf.append(LinearRegression().fit(np.c_[s.e_nor, s.map_pol, s.nor_prev, s.h, s.gf_ff.fillna(160)], s[f'gf_l{k}']).coef_[0])
print('NE first stage Fn(j):', np.round(Fn,3))
print('reduced MA:', np.round(Rn_ma,2)); print('reduced MC:', np.round(Rn_mc,2)); print('reduced GF:', np.round(Rn_gf,2))
def deconv(R,F):
    K=[]
    for k in range(1,len(R)+1):
        K.append((R[k-1]-sum(F[j]*K[k-j-1] for j in range(1,k)))/F[0])
    return np.array(K)
print('K_N on MA:', np.round(deconv(Rn_ma,Fn),2)); print('K_N on MC:', np.round(deconv(Rn_mc,Fn),2))
```