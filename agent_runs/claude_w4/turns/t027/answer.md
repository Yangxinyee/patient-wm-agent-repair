## Step 26: the norepinephrine surprise still correlates slightly with MAP inside a band

**What the last result showed.**
- The surprise e_ne (rate minus its cell mean given the previous rate and MAP band) has SD 0.074.
- Inside band 0 (MAP < 65) its correlation with the current MAP is −0.06; inside band 1 (65–75) it is −0.05; inside band 2 it is about 0.
- These correlations are small but not zero. Either the policy uses the MAP value inside a band (an observable I can control for), or the surprise reflects hidden state (a threat to the instrument).

**What I will do next.** Characterise the dependence: within cells of previous rate, regress the rate (or log ratio) on candidate observables:
- the MAP value;
- the cuff and arterial readings separately;
- the age of the reading and the previous MAP;
- the hour, glucose and insulin.

**What would count as evidence.**
- If the dependence is on the observed MAP value, for example a smooth rule like "Δ ∝ (65 − MAP)", I can include it in E[rate | history] and the surprise stays valid.
- If no observable explains it, the threshold-discontinuity design becomes the safer route for norepinephrine.

```python
c = xn[(xn.h>0)&(xn.ne_prev>0)&(xn.ne_rate>0)&(xn.b3>=0)&(xn.ne_prev<0.6)].copy()
c['lr'] = np.log(c.ne_rate/c.ne_prev)
c['lag_lm'] = c.groupby('stay_id').lm_art_pref.shift(1)
c['src_art'] = (c.age_map_arterial.fillna(99) <= c.age_map_cuff.fillna(99)).astype(float)
c['age_lm'] = np.where(c.src_art==1, c.age_map_arterial, c.age_map_cuff)
for b in [0,1,2]:
    cc = c[c.b3==b].dropna(subset=['lag_lm'])
    cc = cc.assign(lpr=np.log(cc.ne_prev))
    X = [cc.lpr.values, cc.lpr.values**2, cc.lm_art_pref.values, cc.lag_lm.values, cc.src_art.values, cc.age_lm.values, cc.h.values, cc.ins_rate.values, cc.lg.fillna(150).values]
    nm = ['lpr','lpr2','lm','lag_lm','src_art','age_lm','h','ins','lg']
    r = ols(cc.lr.values, X, nm)
    print("band", b, "n", len(cc)); print(r.round(4).T.to_string())
```