The CSV has one row per reading: stay, hour, event, observed value, predicted mean and NLL.

Now the key diagnostic. I'll merge the model's teacher-forced predicted means into the panel and regress them, like the outcomes, on the random part of the doses:
- **Insulin:** the next finger glucose after a dose, on rate_t controlling for the decision reading, previous glucose and previous rate. The true coefficient is −2.0.
- **Norepinephrine:** the next arterial MAP, on rate_t controlling for MAP, its band and previous rates. The true coefficient is +3.2.

I'll also put the observed outcomes through the same regression on these 100 stays as a sanity check of the design.

```python
fa['h'] = fa.hour.round().astype(int)
pm = fa.pivot_table(index=['stay_id','h'], columns='event', values='predicted_mean')
pm.columns = ['pm_'+c for c in pm.columns]
PS = P.loc[selT].copy()
PS = PS.join(pm, how='left')
g = PS.groupby(level='stay_id')
PS['pm_gf_next'] = g['pm_glucose_finger'].shift(-1); PS['gf_next'] = g.gf.shift(-1)
PS['pm_ma_next'] = g['pm_map_arterial'].shift(-1); PS['ma_next'] = g.ma.shift(-1)
PS['gdec'] = PS.gf.combine_first(PS.gl); PS['g_prev'] = g.g_last.shift(1)
PS['mdec'] = PS.ma.combine_first(PS.mc); PS['m_prev'] = g.m_last.shift(1); PS['ne_prev2'] = g.ne0.shift(2)
S1 = PS[(PS.index.get_level_values('h')>=1) & PS['ins'].notna() & PS.gdec.notna() & PS.g_prev.notna()].copy()
S1['band'] = np.select([S1.gdec<180, S1.gdec<220, S1.gdec<250], [1,2,3], default=4)
for outc in ['pm_gf_next','gf_next']:
    X = [S1.gdec, S1.g_prev, S1.ins_prev, (S1.band==2).astype(float), (S1.band==3).astype(float), (S1.band==4).astype(float), S1['ins']]
    r_, n, sd = ols(S1[outc], X, ['gdec','gprev','insprev','b2','b3','b4','rate'])
    print("INS", outc, n, r_.round(3).loc[['gdec','gprev','insprev','rate']].to_dict())
S2 = PS[(PS.index.get_level_values('h')>=1) & (PS.ne_prev>0) & PS['ne'].notna() & PS.mdec.notna() & PS.m_prev.notna()].copy()
S2['b1'] = (S2.mdec<65).astype(float); S2['b3'] = (S2.mdec>=75).astype(float); S2['lprev'] = np.log(S2.ne_prev)
for outc in ['pm_ma_next','ma_next']:
    X = [S2.mdec, S2.m_prev, S2.b1, S2.b3, S2.ne_prev, S2.lprev, S2.ne_prev2, S2['ne']]
    r_, n, sd = ols(S2[outc], X, ['mdec','mprev','b1','b3','neprev','lprev','neprev2','rate'])
    print("NE", outc, n, r_.round(3).loc[['mdec','mprev','neprev','neprev2','rate']].to_dict())
```