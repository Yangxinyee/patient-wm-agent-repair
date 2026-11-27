The offline pipeline matches the model on the selS subset (2.03 / 1.07 RMS, the sampling floor), so I trust it on all 400 stays. Next I'll attach lagged dose variables to every reading: insulin surprise, insulin expectation and NE rate at lags 1–3. Then I'll regress the model's factual residuals on them, separately for glucose and MAP.

What I expect if the model under-uses the doses: a negative glucose coefficient on the lag-1 insulin surprise (about −1.5), and a positive MAP coefficient on recent NE. Coefficients near zero would mean the factual predictions already contain these effects, and any edit would then have to be purely counterfactual.

```python
def add_lags(RDf, PT, cols, lags):
    key_ = PT.set_index(['stay_id','t'])[cols]
    RDf = RDf.copy(); RDf['t']=(RDf['minute']//60).astype(int)
    for L in lags:
        lg = key_.copy(); lg.index = pd.MultiIndex.from_arrays([lg.index.get_level_values(0), lg.index.get_level_values(1)+L])
        RDf = RDf.join(lg.add_suffix(f'_{L}'), on=['stay_id','t'])
    return RDf
RD2 = add_lags(RD, PT_tr, ['ins','ne','eps_ins','ins_exp'], [1,2,3])
RD2['res'] = RD2['observed']-RD2['pred0']
def reg(df, cols, tag):
    d = df.dropna(subset=cols); Xr = np.column_stack([np.ones(len(d))]+[d[c].values for c in cols])
    cf,*_ = np.linalg.lstsq(Xr, d['res'].values, rcond=None); rs=d['res'].values-Xr@cf
    se = np.sqrt(np.diag(np.linalg.inv(Xr.T@Xr))*rs.var())
    print(tag, len(d), ' '.join(f"{c}:{a:+.2f}({s:.2f})" for c,a,s in zip(['1']+cols,cf,se)))
for ev in ['glucose_finger','glucose_lab']:
    reg(RD2[RD2.event==ev], ['eps_ins_1','ins_exp_1','eps_ins_2','ins_exp_2','eps_ins_3','ne_1'], ev[:9])
for ev in ['map_arterial','map_cuff']:
    reg(RD2[RD2.event==ev], ['ne_1','ne_2','ne_3','ins_1'], ev[:9])
```