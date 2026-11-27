The insulin protocol's expectation given glucose is simply the base level: E[rate | base] = 0, 1.0, 2.0, 3.0, 4.05, 6.17, including about 4% omissions. Insulin is never given without a glucose reading at that hour. So the insulin surprise is just rate − base.

For NE I first need to know whether the model's MAP predictions already use the NE level beyond what the MAP history implies. I'll regress the model's factual residuals (observed − predicted) on the preceding NE rate and on the insulin surprise and level, controlling for the last readings. A residual slope of about +3 mmHg per mcg/kg/min would mean the model ignores NE, so a centred NE correction would improve factual fit rather than hurt it. A slope near zero would mean the model already uses NE in factual prediction.

```python
PT_tr['ins_exp'] = PT_tr['base'].map(tb['mean']).fillna(0.0); PT_tr.loc[PT_tr.Gdec.isna(),'ins_exp']=0.0
PT_tr['eps_ins'] = PT_tr['ins']-PT_tr['ins_exp']
key = PT_tr.set_index(['stay_id','t'])
FR['t'] = (FR['minute']//60).astype(int)
lag = key[['ins','ne','eps_ins','base','map_arterial','map_cuff','glucose_finger','glucose_lab']].copy()
lag.index = pd.MultiIndex.from_arrays([lag.index.get_level_values(0), lag.index.get_level_values(1)+1])
J1 = FR.join(lag.add_suffix('_p'), on=['stay_id','t'])
J1['res'] = J1['observed']-J1['predicted_mean']
for ev in ['map_arterial','map_cuff','glucose_finger','glucose_lab']:
    s = J1[(J1.event==ev)].dropna(subset=['ne_p','ins_p'])
    lastv = s['last_val'].fillna(s['observed'].mean())
    Xr = np.column_stack([np.ones(len(s)), s['ne_p'], s['eps_ins_p'], s['ins_p'], lastv-lastv.mean()])
    cf,res_,*_ = np.linalg.lstsq(Xr, s['res'].values, rcond=None)
    rs = s['res'].values-Xr@cf; se = np.sqrt(np.diag(np.linalg.inv(Xr.T@Xr))*rs.var())
    print(ev, len(s), 'coef[1,ne,eps_ins,ins,last]', np.round(cf,2), 'se', np.round(se,2))
```