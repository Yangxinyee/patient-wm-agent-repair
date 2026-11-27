Forward pass 2 gives the original model's factual baseline on these 100 tuning stays: value NLL 1.5116 per reading, over 5,214 readings. My factual tolerance is measured against this. GF MAE is 11 mg/dL and arterial-MAP MAE is 4 mmHg.

Now the key regression. For each glucose_finger reading one hour after an insulin decision, I'll regress the model's predicted mean on the dose surprise (recorded rate minus the band's mean rate), holding the previous reading fixed. I'll run the same regression on the actual reading. The surprise is exogenous by construction (sd ~0.31 in log), so its coefficient is causal.

If the actual-reading slope is about -2/unit and the model's is also about -2, the model learned the effect within support and only extrapolates badly. If the model's slope is far weaker than the actual one, it under-uses dose information and credits the glucose fall to the readings.

```python
fa = pd.read_csv('/work/fwd_2/factual_original.csv')
print(fa.columns.tolist()); print(fa.head(5).to_string())
def base_of(g):
    return np.select([g<=150, g<=180, g<=220, g<=260, g<=300], [0,1,2,3,4], 6)
mb = {0:0.0, 1:1.048, 2:2.09, 3:3.15, 4:4.21, 6:6.29}
gf = tu[tu.event=='glucose_finger'][['stay_id','hour','value']].copy()
gf['hr'] = gf.hour.round(3)
insu = tu[tu.event=='insulin'][['stay_id','hour','rate']].copy()
insu['hr_read'] = (insu.hour-0.0833).round(0)
gfi = gf[np.isclose(gf.hour, gf.hour.round(0))].copy(); gfi['h'] = gfi.hour.round(0).astype(int)
P2 = gfi[['stay_id','h','value']].rename(columns={'value':'g0'})
P2['h1'] = P2.h+1
nxt = gfi[['stay_id','h','value']].rename(columns={'h':'h1','value':'g1'})
P2 = P2.merge(nxt, on=['stay_id','h1'])
ins_h = insu.assign(h=insu.hr_read.astype(int))[['stay_id','h','rate']]
P2 = P2.merge(ins_h, on=['stay_id','h'], how='left').fillna({'rate':0.0})
P2['base'] = base_of(P2.g0.values); P2['dev'] = P2.rate - P2.base.map(mb)
# model predicted mean of the reading at h1
fa_gf = fa[fa.event=='glucose_finger'].copy() if 'event' in fa.columns else fa
print(fa_gf.columns.tolist())
```