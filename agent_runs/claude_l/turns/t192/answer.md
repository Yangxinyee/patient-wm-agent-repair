The teacher-forced replica reproduces the tool exactly: 1.556315 against 1.55632 overall, with the same per-code values. It takes about 0.06 s per stay, so I can run thousands of training stays offline.

Next I'll run it on 4,000 training stays and save:
- for every reading: the prediction, NLL and probabilities;
- for every drug bundle: `ctx`, `u` and the recorded rates.

This feeds the next two analyses:
1. **Expected dose from context.** How well a linear map from `ctx` predicts the recorded insulin and norepinephrine rate when a record exists. The edit must compute the dose residual from what the model can see, so this sets how clean the residual is.
2. **Error regression.** The model's prediction errors on glucose (finger and lab) and MAP (arterial and cuff) regressed on dose residuals at lags 1–12. This gives the per-lag corrections a teacher-forced fit can absorb without hurting NLL.

From my earlier GBM policy work I expect a linear R² of about 0.8 or more for insulin. A clearly lower value would mean the residual must be computed nonlinearly inside `z_post`.

```python
if not (isinstance(globals().get('tr'),pd.DataFrame) and 'stay_id' in tr.columns):
    tr=pd.read_csv('/data/train.csv')
rng=np.random.default_rng(7); all_ids=np.array(sorted(tr.stay_id.unique()))
sel=rng.choice(all_ids,4000,replace=False)
t0=time.time()
grp={s:df for s,df in tr[tr.stay_id.isin(set(sel.tolist()))].groupby('stay_id')}
BTR={s:build_bundles(grp[s].sort_values('hour')) for s in sel}
print('bundles built',time.time()-t0)
RR=[];DD=[]
for i,s in enumerate(sel):
    R,D=tf_stay(W_cur,BTR[s])
    for r in R: RR.append((i,r['t'],r['code'],r['vb'],r['z'],r['mean'],r['nll'],*r['p']))
    for d_ in D: DD.append((i,d_['t'],d_['ins'],d_['ne'],d_['hi'],d_['hn'],d_['ctx'].astype(np.float32),d_['u'].astype(np.float32)))
print('tf time',time.time()-t0)
RA=np.array(RR,dtype=np.float64)
DA_meta=np.array([d_[:6] for d_ in DD],dtype=np.float64); DA_ctx=np.stack([d_[6] for d_ in DD]); DA_u=np.stack([d_[7] for d_ in DD])
np.savez('/work/tf_train.npz',sel=sel,RA=RA,DA_meta=DA_meta,DA_ctx=DA_ctx,DA_u=DA_u)
print(RA.shape,DA_meta.shape, 'mean nll',RA[:,6].mean())
print('ins bundles',int(DA_meta[:,4].sum()),'ne bundles',int(DA_meta[:,5].sum()))
```