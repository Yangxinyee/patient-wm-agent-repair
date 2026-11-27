The norepinephrine policy is clear:
- **Step Δ:** about +0.08 when MAP < 55; +0.04 for 55–65; 0 for 65–75; −0.04 to −0.05 above 75.
- **Noise:** multiplicative, sd about 0.3, applied as rate_t = clip((rate_{t−1} + Δ) × exp(η), 0.01, 1.0).
- **Which reading:** the arterial reading when one is present; with both present, cuff-based medians are smoothed versions of the arterial pattern.
- **No reading:** the median step is +0.03, so the rule may then use the last known MAP.

As with insulin, most rate variation is noise. The task is now to check that η is exogenous and then to g-estimate the norepinephrine → arterial MAP effect.

Next:
1. Build the hourly norepinephrine and MAP grids.
2. Estimate E[rate_t | rate_{t−1}, MAP category] nonparametrically. The cells are the previous rate at 0.01 resolution × the Δ category from the recorded MAP: arterial if present, else cuff, else none.
3. Form residuals and test their correlation with the previous residual, the past MAP change, cuff − arterial, glucose and the hour.

Correlations near zero would confirm the instrument.

```python
A_ = grid('map_arterial'); C_ = grid('map_cuff'); N_ = grid('norepinephrine')
Mn = np.where(np.isnan(A_), C_, A_)
def ne_cat(m):
    return np.select([np.isnan(m), m<55, m<65, m<=75], [4,0,1,2], 3)
cat = ne_cat(Mn)
Np = np.full_like(N_, np.nan); Np[:,1:] = N_[:,:-1]
given = ~np.isnan(N_) & ~np.isnan(Np)
key = pd.Series(list(zip(np.round(Np[given]*100).astype(int), cat[given])))
vals = pd.Series(N_[given])
cellmean = vals.groupby(key.values).transform('mean').values
cellcount = vals.groupby(key.values).transform('size').values
Nres = np.zeros_like(N_); Nres[given] = N_[given] - cellmean
Nuse = np.zeros_like(given); Nuse[given] = cellcount >= 30
print("ne residual var", Nres[Nuse].var(), "n", Nuse.sum())
# exogeneity checks
def corr_check(Res, use, others):
    for nm, Xo in others.items():
        m = use & ~np.isnan(Xo)
        print(nm, m.sum(), round(np.corrcoef(Res[m], Xo[m])[0,1],4))
Res_prev = np.full_like(Nres, np.nan); Res_prev[:,1:] = np.where(Nuse[:,:-1], Nres[:,:-1], np.nan)
dmap = np.full_like(Mn, np.nan); dmap[:,1:] = Mn[:,1:] - Mn[:,:-1]
corr_check(Nres, Nuse, {'res_prev': Res_prev, 'dmap_past': dmap, 'cuff-art': C_-A_, 'glucose': Gn, 'hour': np.tile(H,(nS,1)).astype(float), 'map_now': Mn})
```