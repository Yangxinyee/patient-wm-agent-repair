The norepinephrine policy is now clear. It reads the arterial MAP when one is recorded and ignores the cuff then. Otherwise it reads the cuff, or the last MAP if neither was taken this hour. Thresholds are sharp: up about 0.13–0.16 in log-rate at ≤64, hold at 66–74, down about 0.2–0.3 at ≥76. The values at 65 and 75 are half-way, consistent with a rule applied to unrounded readings.

So the policy depends only on recorded data plus noise. My crude shock model was biased by the cap truncation (I dropped capped rows) and by missing inputs. That explains the rp and MAP(t-1) correlations.

Next I'll write up the notebook, then re-estimate both drugs' effects by cross-fitted partialling-out (double ML). I'll residualise the rate and the future reading on a rich set of observed history features with gradient boosting, keeping capped rows, then regress residual on residual for lags 1–8, plus a first stage for later rates. If the insulin numbers barely move and the norepinephrine numbers stay near 3.3, 5.4, 6.4…, the earlier estimates were essentially right. A large change would mean the earlier norepinephrine shock was confounded by the policy inputs.

```python
open('/work/notes.md','a').write("""
## Data facts (runs 3-19)
- All stays 25.17h; readings at integer hours -1..24; infusions at h+5min for h=0..24, 1h each.
- finger vs lab glucose: same quantity (paired diff -0.02, sd 9). arterial MAP ~3 mmHg below cuff (sd 6.7).
- INSULIN POLICY: off if last glucose<140, on if >160, random (hysteresis) 140-160. Rate = base(band)*mult noise (CV .33), round 0.5:
  base 1 (<180), 2 (180-220), 3 (220-260), ~4.3 (>260). Noise ~iid (no stay persistence, uncorrelated w/ lab@t, past glucose).
- INSULIN EFFECT (within-band dose noise): next-hr finger glucose -2.0/unit/h, linear across bands & rates 0.5-7.
  lag response to dose shock at t: k=1..6: -2.01,-2.25,-1.81,-1.30,-0.73,-0.45 ; dose feedback on later doses only -0.03..-0.05.
- NOREPI POLICY: titration on log rate, uses arterial MAP if present at t else cuff else last; <=64 up ~+0.15, 66-74 hold, >=76 down ~-0.25,
  noise sd ~0.3 log; capped at 1.0 (13% of records at cap), step 0.01.
- NOREPI EFFECT (crude shock, maybe biased): MAP resp k=1..7: 3.3,5.4,6.4,6.5,6.0,5.6,4.9 per +1 mcg/kg/min shock; shock persists in rate .85,.73,.64...
  crude deconvolved impulse: 3.3,2.6,1.8,0.9,0.4,0.3 (~9 mmHg per sustained unit).
""")
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
def hist_feats(t):
    j = col(t)
    f = {'t':np.full(len(ids),t), 'glast':glast[:,j], 'gf0':gf[:,j], 'gl0':gl[:,j], 'gm1':gbest[:,j-1], 'gm2':gbest[:,j-2],
         'mlast':mlast[:,j], 'ma0':ma[:,j], 'mc0':mc[:,j], 'mm1':mlast[:,j-1], 'mm2':mlast[:,j-2],
         'ip':ins[:,j-1], 'ipp':ins[:,j-2], 'rp':ne[:,j-1], 'rpp':ne[:,j-2], 'has_art':np.isfinite(ma[:,max(1,j-3):j+1]).any(1).astype(float)}
    return pd.DataFrame(f)
def crossfit_resid(X, y, n_splits=3, seed=0):
    r = np.full(len(y), np.nan); kf = KFold(n_splits, shuffle=True, random_state=seed)
    for a,b in kf.split(X):
        m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, max_leaf_nodes=31).fit(X.iloc[a], y[a]); r[b] = y[b]-m.predict(X.iloc[b])
    return r
print("ok")
```