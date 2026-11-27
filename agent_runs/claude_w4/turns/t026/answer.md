## Step 25: the full norepinephrine policy

**What the last result showed.**
- The titration step depends on the previous rate in a way that looks additive rather than multiplicative:
  - below 65: new ≈ (previous + 0.05) × noise;
  - 65–75: new ≈ previous × noise;
  - above 75: new ≈ (previous − 0.04) × noise, or a stop with probability about 19%.
- The multiplicative noise has log SD about 0.3 and is i.i.d.
- Inside a band the drift does not depend on the actual MAP value.

**What this means.** The policy depends only on (previous rate, MAP band), so a non-parametric expected rate in cells of (previous rate at 0.01 resolution × band) gives valid surprises.

**What I will do next.**
1. Build e_ne = rate − E[rate | previous rate, band] from those cells.
2. Run the 2SLS impulse response for MAP, separately for:
   - arterial MAP in arterial-line stays;
   - cuff MAP in non-arterial stays.

**What would count as evidence.**
- A clear positive response (norepinephrine raises MAP) with a smooth kernel would give its true effect.
- A null response would mean the drug does nothing in this ICU, however strongly MAP and norepinephrine are associated in the records.

```python
xn = xn.sort_values(['stay_id','h'])
xn['b3'] = np.where(xn.lm_art_pref.isna(), -1, np.where(xn.lm_art_pref<65,0,np.where(xn.lm_art_pref<=75,1,2)))
xn['pr'] = np.round(xn.ne_prev,2)
# expected rate cell means (exclude h==0 where prev is unknown)
cell = xn[xn.h>0].groupby(['b3','pr']).ne_rate.agg(['mean','size'])
xn = xn.join(cell['mean'].rename('ne_exp'), on=['b3','pr'])
xn.loc[xn.h==0,'ne_exp'] = np.nan
xn['e_ne'] = xn.ne_rate - xn.ne_exp
print("e_ne stats:", xn.e_ne.describe().round(4).to_dict())
print("cells with <20 obs:", (cell['size']<20).sum(), " of", len(cell))
# check surprise independent of MAP within band
chk = xn[(xn.h>0)&(xn.ne_prev>0)&(xn.b3>=0)]
for b in [0,1,2]:
    c = chk[chk.b3==b]
    print("band",b," corr(e_ne, lm):", np.corrcoef(c.e_ne, c.lm_art_pref)[0,1].round(4), " corr(e_ne, lag MAP):", c[['e_ne']].join(c.groupby('stay_id').lm_art_pref.shift(1).rename('lag')).dropna().corr().iloc[0,1].round(4))
# lags
gn = xn.groupby('stay_id')
for m in range(0,15):
    xn[f'N_l{m}'] = gn.ne_rate.shift(m)
    xn[f'n_l{m}'] = gn.e_ne.shift(m)
    xn[f'lm_l{m}'] = gn.lm_art_pref.shift(m)
xn['art_stay'] = xn.groupby('stay_id').map_arterial.transform(lambda s: s.notna().any())
```