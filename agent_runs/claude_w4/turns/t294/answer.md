The art regression failed: CV R² went from 0.96 with controls only to −0.28 once the state was added. That fit is unreliable and needs a larger ridge penalty.

The finger fit is informative. It says v4's insulin injection should lower the next finger and lab predictions (−0.34 and −0.44 per unit). Yet in the rollout, doubling insulin raised glucose already at hour 13. So the injected state is probably transformed by the bundles between the drug bundle (h + 0.083) and the next finger reading, or by the first hour still being teacher-forced.

The direct way to see this is a factual teacher-forced pass of v4 on sel100. Its per-reading prediction changes, regressed on the preceding channel values at each lag, show the true immediate and lagged effect. That pass also gives v4's factual-fit cost.

This run prepares for it:
1. Print the sel100 IDs, since I used the wrong ones last time.
2. Refit the four prediction-sensitivity regressions over a ridge grid.
3. Compute "level" directions as an alternative target: the regression of the pre-reading state on the model's own predicted level.
4. Print the bundle sequence around hour 12 for one insulin stay, to see what lies between a drug bundle and the next reading.

```python
print(json.dumps([int(s) for s in sel]))
Wbest = {}; Glev = {}
for e in ev4:
    d = fr[fr.event==e].copy(); d = d[d.row.notna()]; d['row']=d.row.astype(int); d = d[prev_ok[d.row.values]]
    r_e = recs[recs.event==e][['stay_id','hour','value']].rename(columns={'hour':'h_reg','value':'v_reg'})
    d['hq'] = d.hour - 1e-4; d = pd.merge_asof(d.sort_values('hq'), r_e.sort_values('h_reg'), left_on='hq', right_on='h_reg', by='stay_id', direction='backward')
    d['has'] = d.v_reg.notna().astype(float); nmd = NUM[(e,'value')]
    d['zr'] = np.where(d.has>0, np.clip((d.v_reg-nmd['median'])/nmd['iqr'],-10,10), 0); d['b'] = np.where(d.has>0, 1+np.searchsorted(nmd['fine_bin_edges'], d.v_reg.fillna(0)), 0)
    d['lage'] = np.where(d.has>0, np.log1p(d.hour - d.h_reg), 0); d['ldt'] = np.log((hr[d.row.values]-hr[d.row.values-1])*3600+1)
    Ctl = np.c_[np.eye(17)[d.b.values.astype(int)], d.zr, d.lage, d.ldt]; O = outs_all[d.row.values-1]; y_ = d.predicted_mean.values
    Z_ = np.c_[Ctl, O]; n_=len(y_); fold = np.random.RandomState(0).permutation(n_)%5; best=None
    for lm in [1, 10, 100, 1000]:
        lam = np.r_[np.full(Ctl.shape[1],1e-6), np.full(192, lm)]; pr = np.zeros(n_)
        for f in range(5):
            tr, te = fold!=f, fold==f; bb = np.linalg.solve(Z_[tr].T@Z_[tr] + np.diag(lam), Z_[tr].T@y_[tr]); pr[te] = Z_[te]@bb
        r2 = 1-np.var(y_-pr)/np.var(y_); print(e, 'lam', lm, 'CV R2', round(r2,4), 'max abs err', round(np.abs(y_-pr).max(),1))
        if best is None or r2>best[0]: best=(r2, lm)
    lam = np.r_[np.full(Ctl.shape[1],1e-6), np.full(192, best[1])]; bb = np.linalg.solve(Z_.T@Z_ + np.diag(lam), Z_.T@y_); Wbest[e] = bb[-192:]
    mm = m_all[d.row.values-1]; p = y_ - y_.mean(); Glev[e] = ((mm-mm.mean(0))*p[:,None]).sum(0)/(p**2).sum()
    print(e, 'best lam', best, '|w|', round(np.linalg.norm(Wbest[e]),2), 'cos(w_old,w_best)', round(uv(W[e])@uv(Wbest[e]),3), '|level dir| per unit', round(np.linalg.norm(Glev[e]),3), 'cos(level, nu)', round(uv(Glev[e])@uv(nu[ev4.index(e)].mean(0)),3))
np.savez('/work/dirs265.npz', **{('w_'+e):Wbest[e] for e in ev4}, **{('lev_'+e):Glev[e] for e in ev4})
s_ex = [s for s in sel if (tsel[(tsel.stay_id==s)&(tsel.event=='insulin')].hour.between(11,14)).any()][0]
print('example stay', s_ex); print(tsel[(tsel.stay_id==s_ex)&(tsel.hour.between(11.9,14.1))][['hour','event','value','rate']].to_string())
```