The core is not a simple linear memory:
- **c_out is a dense mixing matrix,** far from the identity.
- **Each step keeps only about 0.29–0.43 of the old h** (a_bar), before c_out and the coda transform it again.

A dose remembered only through h's direct carry-over would fade within a few bundles. Any longer memory has to come from the nonlinear recurrent dynamics. The memory M will not help, because it stores only e vectors, and the END records in e mark that an infusion ran but not its rate.

So whether a "read the dose memory from the state" edit is feasible depends on an empirical question: how much does the state after each bundle actually encode the insulin rates of the past hours? That includes their deviation from the policy's expected rate at the recorded glucose.

Next I will fit cross-validated ridge regressions on the tool's states for the 100 stays (split by stay). They predict, for lags 0–10 h:
- the insulin rate given that hour,
- its deviation from the policy mean.

How I will read it:
- **High R² for the deviation out to ~6–8 h:** a linear probe in the value head can supply the missing dose effect.
- **R² that collapses after 1–2 h:** the state forgets dose size, and I need another carrier.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
rec_ = pd.concat([train[train.stay_id.isin(set(S3.tolist()))], tune[tune.stay_id.isin(set(S3.tolist()))]])
print('rec rows', len(rec_), 'stays', rec_.stay_id.nunique(), 'cols', list(rec_.columns)[:10])
def policy_tables(df):
    ins_ = df[df.event == 'insulin'].copy(); ins_['hh'] = np.floor(ins_.hour + 1e-6).astype(int)
    gfr = df[df.event == 'glucose_finger'][['stay_id','hour','value']]; glr = df[df.event == 'glucose_lab'][['stay_id','hour','value']]
    dec = pd.concat([glr.assign(pri=1), gfr.assign(pri=0)]).sort_values('pri').drop_duplicates(['stay_id','hour'])
    dec['hh'] = np.round(dec.hour).astype(int)
    return ins_, dec
ins_tr, dec_tr = policy_tables(train)
m_ = dec_tr.merge(ins_tr[['stay_id','hh','rate']], on=['stay_id','hh'], how='left').fillna({'rate': 0.0})
m_['gb'] = np.clip((m_.value // 5) * 5, 100, 400)
pibar_tab = m_.groupby('gb').rate.mean()
pibar = lambda G: np.interp(G, pibar_tab.index.values + 2.5, pibar_tab.values)
print('pibar at 140,160,200,240,280:', np.round(pibar(np.array([140,160,200,240,280])),2))
ins_s, dec_s = policy_tables(rec_)
R_ = dec_s.merge(ins_s[['stay_id','hh','rate']], on=['stay_id','hh'], how='left').fillna({'rate': 0.0})
R_['dev'] = R_.rate - pibar(R_.value.values)
rate_map = {(s, h): r for s, h, r in zip(R_.stay_id, R_.hh, R_.rate)}; dev_map = {(s, h): d for s, h, d in zip(R_.stay_id, R_.hh, R_.dev)}
hh_b = np.floor((np.asarray(min_) - 5) / 60).astype(int)
Xst = (X3 - X3.mean(0)) / (X3.std(0) + 1e-6)
for L in [0,1,2,3,4,6,8,10]:
    y_r = np.array([rate_map.get((int(S3[i]), int(hh_b[i]-L)), np.nan) for i in range(NB)])
    y_d = np.array([dev_map.get((int(S3[i]), int(hh_b[i]-L)), np.nan) for i in range(NB)])
    out_l = []
    for y_, msk in [(y_r, np.isfinite(y_r) & (y_r > 0)), (y_d, np.isfinite(y_d) & (y_r > 0))]:
        Xm, ym, gm = Xst[msk], y_[msk], S3[msk]
        pr_ = np.zeros_like(ym)
        for tr_i, te_i in GroupKFold(5).split(Xm, ym, gm):
            pr_[te_i] = Ridge(alpha=300.0).fit(Xm[tr_i], ym[tr_i]).predict(Xm[te_i])
        out_l.append(1 - np.mean((ym-pr_)**2)/np.var(ym))
    print('lag', L, 'n', int((np.isfinite(y_r)&(y_r>0)).sum()), 'CV R2 rate|given', round(out_l[0],3), 'dev', round(out_l[1],3))
```