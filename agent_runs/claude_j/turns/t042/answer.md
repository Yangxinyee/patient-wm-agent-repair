This settles a key point. On factual records the model does not respond to the exogenous part of the dose at all. Its predicted next glucose_finger has slope -0.01 ± 0.27 per unit of dose surprise.

The actual readings in this small sample give -0.66 ± 0.59, about 2.3 standard errors from my earlier -2/unit. That is noisy, but I want to be sure of the target before building on it. Everything downstream hinges on it: the model credits post-insulin glucose falls to the high reading, which explains its weak dose answers.

So I'll re-run exactly this regression on the 30,000 training stays at lags 1–6, for both glucose_finger and glucose_lab. A lag-1 slope near -2.0 with a small standard error, rising slightly at lag 2 and then decaying, would confirm the target effect. A much weaker slope would mean my true-effect kernel is wrong.

```python
def panel_lag(df, maxk=6):
    gfx = df[df.event=='glucose_finger'][['stay_id','hour','value']]
    gfx = gfx[np.isclose(gfx.hour, gfx.hour.round(0))].copy(); gfx['h']=gfx.hour.round(0).astype(int)
    glx = df[df.event=='glucose_lab'][['stay_id','hour','value']]
    glx = glx[np.isclose(glx.hour, glx.hour.round(0))].copy(); glx['h']=glx.hour.round(0).astype(int)
    ins_ = df[df.event=='insulin'][['stay_id','hour','rate']].copy()
    ins_['h'] = (ins_.hour-0.0833).round(0).astype(int)
    B = gfx[['stay_id','h','value']].rename(columns={'value':'g0'})
    B = B.merge(ins_[['stay_id','h','rate']], on=['stay_id','h'], how='left').fillna({'rate':0.0})
    B['base']=base_of(B.g0.values); B['dev']=B.rate-B.base.map(mb)
    out={}
    for k in range(1,maxk+1):
        for nm,src in [('GF',gfx),('GL',glx)]:
            t_ = src[['stay_id','h','value']].rename(columns={'h':'hk','value':'y'})
            C = B.assign(hk=B.h+k).merge(t_, on=['stay_id','hk'])
            C = C[C.g0>150]
            out[(nm,k)] = reg(C.y.values, C) + (len(C),)
    return out
res_tr = panel_lag(tr)
for kk,v in res_tr.items(): print(kk, 'slope %.3f se %.3f n %d' % v)
```