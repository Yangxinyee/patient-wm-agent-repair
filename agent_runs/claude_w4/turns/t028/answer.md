## Step 27: norepinephrine titration also depends smoothly on the MAP value

**What the last result showed.** Inside a band, the titration step depends on the current MAP value:
- −0.005 per mmHg in band 0;
- −0.010 per mmHg in band 1.

Lower MAP means a bigger increase. The previous MAP has no effect (+0.0002), and neither do the hour, glucose or insulin.

**What this means.** The policy reacts to the observed current reading, not to hidden MAP trends, so the instrument can be repaired by modelling the expected rate as a flexible function of the observed reading, the band and the previous rate.

**What I will do next.** Fit flexible expected-dose models with cross-fitting by stay (gradient-boosted trees), for both drugs:
- **Norepinephrine:** E[rate | previous rate, latest MAP, band, source, age, hour].
- **Insulin:** E[rate | latest glucose, band, age, previous rate, hour], in case there is a smooth within-band dependence too.

The residual surprises then serve as instruments.

**What would count as evidence.** The new surprises should be uncorrelated with the current reading inside each band (near 0, not −0.05). Their autocorrelation should stay near 0.

```python
from sklearn.ensemble import HistGradientBoostingRegressor
def crossfit(df, feats, target, seed=0):
    sids = df.stay_id.unique(); rng = np.random.RandomState(seed); fold = dict(zip(sids, rng.randint(0,2,len(sids))))
    f = df.stay_id.map(fold).values
    pred = np.zeros(len(df))
    for k in [0,1]:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, min_samples_leaf=50)
        m.fit(df.loc[f!=k, feats].values, df.loc[f!=k, target].values)
        pred[f==k] = m.predict(df.loc[f==k, feats].values)
    return pred
t0=time.time()
xn['age_lm'] = np.where(xn.age_map_arterial.fillna(99) <= xn.age_map_cuff.fillna(99), xn.age_map_arterial, xn.age_map_cuff)
xn['src_art'] = (xn.age_map_arterial.fillna(99) <= xn.age_map_cuff.fillna(99)).astype(float)
dn = xn[(xn.h>0)].copy()
dn['lm_f'] = dn.lm_art_pref.fillna(-1)
featsN = ['ne_prev','lm_f','b3','age_lm','src_art','h']
dn['ne_hat'] = crossfit(dn, featsN, 'ne_rate')
dn['e_ne2'] = dn.ne_rate - dn.ne_hat
print("NE fit time", time.time()-t0, " resid sd", dn.e_ne2.std(), " R2", 1-dn.e_ne2.var()/dn.ne_rate.var())
for b in [0,1,2]:
    cc = dn[(dn.b3==b)&(dn.ne_prev>0)]
    print("band",b," corr(e_ne2, lm):", np.corrcoef(cc.e_ne2, cc.lm_art_pref)[0,1].round(4))
# insulin
t0=time.time()
di = xs[(xs.h>=0)].copy()
di['lg_f'] = di.lg.fillna(-1)
di['age_lg'] = np.where(di.age_glucose_lab.fillna(99) < di.age_glucose_finger.fillna(99), di.age_glucose_lab, di.age_glucose_finger)
di['src_lab'] = (di.age_glucose_lab.fillna(99) < di.age_glucose_finger.fillna(99)).astype(float)
featsI = ['lg_f','band','age_lg','src_lab','prev_ins_rate','h']
di['ins_hat'] = crossfit(di, featsI, 'ins_rate')
di['e_ins2'] = di.ins_rate - di.ins_hat
print("INS fit time", time.time()-t0, " resid sd", di.e_ins2.std())
for b in [1,2,3]:
    cc = di[(di.band==b)]
    print("band",b," corr(e_ins2, lg):", np.corrcoef(cc.e_ins2, cc.lg)[0,1].round(4), " corr(e_ins (old), lg):", np.corrcoef(cc.e_ins, cc.lg)[0,1].round(4))
```